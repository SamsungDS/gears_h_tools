import json
from pathlib import Path
from typing import Union

import ase
import numpy as np
from ase.data import atomic_numbers
from ase.io import read
from ase.units import Ha

from slhtools.utils import (
    BlockedMatrix,
    get_neighbourlist_ijD,
    make_hamiltonian_blockedmatrix,
)


# TODO this function does a bit more than it needs to.
def prepare_gpaw_slh_snapshot(
    directory,
    *,
    gpwfilename: str,
    cutoff: float,
    overlap_only=False,
    overlap_calc_parameters=None,
    reference_dtype=np.float32,
    reference_arraytype="sparse",
    block_write_threshold=1e-4,
):
    atoms = read(gpwfilename)
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)

    write_only_atoms(directory, atoms)

    # This gets us ij (atom1 atom2) matrix elements as we desire
    # with a cutoff that we decide. If there are nonzero hamiltonian matrix
    # elements outside this cutoff, for the moment, we don't care.
    ij, D = get_neighbourlist_ijD(atoms, cutoff)

    if not overlap_only:
        (
            H_MM,
            S_MM,
            ls_dict,
        ) = get_hamiltonian_and_basis_information_from_gpw(gpwfilename)

    elif overlap_only:
        assert overlap_calc_parameters is not None
        H_MM = None
        fermi_levels = None
        (S_MM, ls_dict) = get_overlap_and_basis_information(
            atoms, overlap_calc_parameters
        )

    # If overlap_only, H_MM should be none.
    # If not overlap_only, H_MM should NOT be none.
    # We do this with an XNOR operator such that True XNOR True = False XNOR False  = True
    assert not (overlap_only ^ (H_MM is None))
    H_MM = H_MM.astype(reference_dtype) if H_MM is not None else None
    S_MM = S_MM.astype(reference_dtype)

    if reference_arraytype == "sparse":
        from scipy.sparse import csr_matrix

        np.savez(
            directory / "H_MM_S_MM",
            **{
                "H_MM": csr_matrix(H_MM) if H_MM is not None else None,
                "S_MM": csr_matrix(S_MM),
            },
        )
    elif reference_arraytype == "dense":
        np.savez(directory / "H_MM_S_MM", **{"H_MM": H_MM, "S_MM": S_MM})
    else:
        raise ValueError(
            "Unknown reference arraytype, must be either 'sparse' or 'dense'."
        )

    block_sizes = np.array([sum(2 * np.array(ls_dict[n]) + 1) for n in atoms.numbers])
    assert np.sum(block_sizes) == len(S_MM), f"{np.sum(block_sizes)}, {len(S_MM)}"

    
    blocked_hamiltonian = make_hamiltonian_blockedmatrix(
            S_MM if overlap_only else H_MM,
            atoms,
            block_sizes,
        )
    ii_list = [blocked_hamiltonian.get_block(i,i) for i in range(atoms.get_global_number_of_atoms())]
    ii_array = np.array(ii_list, dtype=object)
    ij, D, hlist = filter_pairs_by_hblock_magnitude(
        ij,
        D,
        blocked_hamiltonian=blocked_hamiltonian,
        threshold=block_write_threshold,
    )
    np.savez(directory / "hblocks_on-diagonal.npz", hblocks=ii_array, allow_pickle=True)
    np.savez(directory / "ijD.npz", ij=ij, D=D)
    np.savez(directory / "hblocks_off-diagonal.npz", hblocks=np.array(hlist, dtype=object), allow_pickle=True)

    with open(directory / "orbital_ells.json", mode="w") as fd:
        json.dump(ls_dict, fd)


def write_only_atoms(directory, atoms):
    from ase.io import write

    with open(directory / "atoms.extxyz", mode="w") as fd:
        write(fd, atoms)


def get_basis_indices_from_calculation(calculation):
    """Given a GPAW DFTCalculation object, calculates a dict of ell and ordered lm indices
    for each elemental block in an LCAO hamiltonian.

    Parameters
    ----------
    calculation : DFTCalculation
        _description_

    Returns
    -------
    _type_
        _description_
    """
    setups = calculation.setups
    ls_dict = {}

    # The setups contain basis information about the calculation
    for setup in setups:
        ells = [x.l for x in setup.basis.bf_j]
        ls_dict[atomic_numbers[setup.basis.symbol]] = ells

    return ls_dict


def get_overlap_and_basis_information(atoms, parameters):
    from gpaw.new.calculation import DFTCalculation

    calculation = DFTCalculation.from_parameters(atoms, parameters)
    ls_dict = get_basis_indices_from_calculation(calculation)
    # TODO: Would be nice to be able to get the k-space S_MM and then FT back
    S_MM = calculation.state.ibzwfs.wfs_qs[0][0].S_MM
    S_MM.gather()

    return S_MM.data, ls_dict


def get_hamiltonian_and_basis_information_from_gpw(gpwfilename: str):
    from gpaw.new.ase_interface import GPAW

    # We set scalapack here even one a single core since that enables
    # sparse atomic corrections.
    calculation = GPAW(gpwfilename, parallel={"sl_auto": True})._dft
    ls_dict = get_basis_indices_from_calculation(calculation)
    matcalc = calculation.scf_loop.hamiltonian.create_hamiltonian_matrix_calculator(
        calculation.state
    )

    # TODO This is spin-paired and gamma point for the moment.
    H_MM = matcalc.calculate_matrix(calculation.state.ibzwfs.wfs_qs[0][0])
    H_MM.gather()
    S_MM = calculation.state.ibzwfs.wfs_qs[0][0].S_MM
    S_MM.gather()

    return (
        H_MM.data * Ha,
        S_MM.data,
        ls_dict,
        # calculation.state.ibzwfs.fermi_levels * Ha,
    )


def filter_pairs_by_hblock_magnitude(
    ij, D, blocked_hamiltonian,threshold
):
    
    filtered_ij_list = []
    filtered_D_list = []
    matrix_blocks_list = []
    for _ij, _D in zip(ij, D, strict=True):
        i, j = _ij

        matrix_block = blocked_hamiltonian.get_block(i, j)
        # A max abs is the infinity-norm
        if np.max(np.abs(matrix_block)) >= threshold:
            filtered_ij_list.append(_ij)
            filtered_D_list.append(_D)
            matrix_blocks_list.append(matrix_block)

    return np.array(filtered_ij_list), np.array(filtered_D_list), matrix_blocks_list
    # TODO Need to generate diagonal blocks
    # for ii in np.unique(ij[:, 0]):
    #     # if ii == 1:
    #     #     np.savetxt("dump.txt", blocked_hamiltonian.get_block(ii, ii).ravel())
    #     # Self-overlap terms will always have a lattice shift of 0, 0, 0
    #     tmp = [0, 0, 0, int(ii + 1), int(ii + 1)]
    #     h5handle.create_dataset(
    #         json.dumps(tmp), data=blocked_hamiltonian.get_block(ii, ii)
    #     )
