import json
from pathlib import Path
from typing import Union

import ase
import numpy as np
from ase.io import read, write

from gears_h_tools.utils import (
    blocked_matrix_to_hmatrix,
    filter_pairs_by_hblock_magnitude,
    get_neighbourlist_ijD,
    get_neighbourlist_ijDS,
    get_permutation_dict,
    group_ijD_by_S,
    make_hamiltonian_blockedmatrix,
    write_only_atoms
)

from gears_h_tools.abacus_utils import (
    read_H_csrs_and_shifts, 
    read_S_csr,
    get_abacus_ells_dict,
    get_abacus_ellwise_permutation_dict
)

from gears_h_tools.gpaw_utils import (
    get_overlap_and_basis_information,
    get_hamiltonian_and_basis_information_from_gpw
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
    unique_pairs: bool = True
):
    atoms = read(gpwfilename)
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)

    write_only_atoms(directory, atoms)

    # This gets us ij (atom1 atom2) matrix elements as we desire
    # with a cutoff that we decide. If there are nonzero hamiltonian matrix
    # elements outside this cutoff, for the moment, we don't care.
    ij, D = get_neighbourlist_ijD(atoms, cutoff, unique_pairs)

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

def prepare_abacus_gears_h_snapshot(abacus_out_dir: Path,
                                    write_dir: Path,
                                    cutoff: float,
                                    threshold: float = 1e-4
                                   ):
    # read in atoms, H matrices and shift vectors, and the S matrix.
    try:
        atoms = read(abacus_out_dir / "STRU.cif")
        csr_hs, svecs = read_H_csrs_and_shifts(abacus_out_dir / "hrs1_nao.csr",
                                               threshold = threshold)
        csr_s = read_S_csr(abacus_out_dir / "srs1_nao.csr")
        # extract basis set angular momenta per atom
        ells_dict = get_abacus_ells_dict(logfile_path = abacus_out_dir / "running_scf.log")
    except Exception as e:
        print(f"Failed to read in {abacus_out_dir}")
        raise e
    # number of basis functions per species
    species_basis_size_dict = {k: np.sum([2*ell+1 for ell in v]) for k,v in ells_dict.items()}
    # get neighborlist and distances
    ij, D, S = get_neighbourlist_ijDS(atoms = atoms,
                                      cutoff=cutoff)
    grouped_ijD = group_ijD_by_S(ij, D, S, svecs)
    block_sizes = np.array([sum(2 * np.array(ells_dict[n]) + 1) for n in atoms.numbers])

    perm_dict = get_permutation_dict(ells_dict=ells_dict,
                                     ellwise_permutation_dict=get_abacus_ellwise_permutation_dict())

    final_ijs = []
    final_Ds = []
    off_diag_blocks = []
    svec_num_blocks = []
    from scipy.sparse import csr_matrix
    gamma_H = csr_matrix(csr_hs[0].shape, dtype=float)
    # For every CSR H and shift vector pair:
    for csrH, shift in zip(csr_hs, svecs, strict=True):
        # Get the neighbor list and displacement vectors
        tij = grouped_ijD[tuple(shift)]['ij']
        tD = grouped_ijD[tuple(shift)]['D']
        # Densify H
        raw_h = np.array(csrH.todense())
        # Block the raw H.
        bh = make_hamiltonian_blockedmatrix(H_MM = raw_h, 
                                            atoms = atoms,
                                            basis_block_sizes = block_sizes,
                                            permutation_dict = perm_dict)
        # Build a new H matrix from the raw blocked H. 
        # This handles the permutation thanks to BlockedMatrix.
        h = blocked_matrix_to_hmatrix(bh, 
                                      atoms,
                                      species_basis_size_dict,
                                      tij)
        gamma_H += csr_matrix(h)
        
        
        # Get the off-diagonal blocks, but filter first.
        tij, tD, permuted_off_diag_blocks = filter_pairs_by_hblock_magnitude(
                                                ij = tij,
                                                D = tD,
                                                blocked_hamiltonian = bh,
                                                threshold = threshold,
                                            )
        if len(tij) == 0:
            continue
        svec_num_blocks.append(np.concat([shift, [len(tij)]]))
        
        # I loathe these next couple of lines but it's the only way to
        # stop numpy from stacking arrays.......
        permuted_off_diag_blocks_array = np.empty(tij.shape[0], dtype=object)
        for i, b in enumerate(permuted_off_diag_blocks):
            permuted_off_diag_blocks_array[i] = b

        final_ijs.append(tij)
        final_Ds.append(tD)
        off_diag_blocks.append(permuted_off_diag_blocks_array)

        # on-diagonal blocks
        # only get these for S = (0, 0, 0)
        if np.all(shift == np.zeros(3, dtype=int)):
            permuted_ii_list = [bh.get_block(i,i) for i in range(atoms.get_global_number_of_atoms())]
            on_diag_blocks = np.array(permuted_ii_list, dtype=object)

    # combine off-diagonal neighbor indicies, displacements, and blocks
    final_ijs = np.vstack(final_ijs)
    final_Ds = np.vstack(final_Ds)
    final_off_diag_blocks = np.concatenate(off_diag_blocks, dtype=object)

    # make blocked S
    raw_s = np.array(csr_s.todense())
    raw_bs = make_hamiltonian_blockedmatrix(H_MM = raw_s, 
                                            atoms = atoms,
                                            basis_block_sizes = block_sizes,
                                            permutation_dict = perm_dict)
    # get permuted S using the unfiltered neighbor list
    # since we're folding in all the shifted cells, when we reassemble
    # the permuted S, we only need the neighbors in the origin cell.
    gamma_S = blocked_matrix_to_hmatrix(raw_bs,
                                        atoms,
                                        species_basis_size_dict,
                                        grouped_ijD[(0,0,0)]['ij'])
    gamma_S = csr_matrix(S)
    
    # write training data
    if not write_dir.exists():
        write_dir.mkdir(parents = True)
    write(write_dir / "atoms.extxyz", atoms)
    np.savez(write_dir / "hblocks_on-diagonal.npz",
             hblocks=on_diag_blocks,
             allow_pickle=True)
    np.savez(write_dir / "ijD.npz",
             ij=final_ijs,
             D=final_Ds)
    np.savetxt(write_dir / "svecs_num_blocks.dat",
               svec_num_blocks,
               fmt="%d")
    np.savez(write_dir / "hblocks_off-diagonal.npz",
             hblocks=final_off_diag_blocks,
             allow_pickle=True)
    np.savez(
        write_dir / "H_MM_S_MM.npz",
        **{
            "H_MM": gamma_H,
            "S_MM": gamma_S,
        },
    )

    with open(write_dir / "orbital_ells.json", mode="w") as fd:
        json.dump(ells_dict, fd)
