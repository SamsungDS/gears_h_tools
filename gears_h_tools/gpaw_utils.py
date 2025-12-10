from ase.data import atomic_numbers
from ase.units import Ha

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
