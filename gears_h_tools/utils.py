from typing import List, Union

import ase
import numpy as np
from matscipy.neighbours import neighbour_list


def get_neighbourlist_ijD(
    atoms: ase.Atoms, cutoff: float, unique_pairs: bool = True
):
    """Returns a array where each row is atom-pair indices and an array of lattice shifts for that pair

    Parameters
    ----------
    atoms : ase.Atoms
        _description_
    cutoff : float
        _description_
    unique_pairs : bool, optional
        _description_, by default True
    """
    i, j, D = neighbour_list("ijD", atoms=atoms, cutoff=cutoff)
    ij = np.column_stack([i, j]).astype(int)

    if unique_pairs:
        for ii in range(len(atoms)):
            all_neighbours = ij[ij[:, 0] == ii, 1]
            unique_neighbours = set(all_neighbours)
            assert len(unique_neighbours) == len(
                all_neighbours
            ), f"Atom {ii} has non-unique neighbours with respect to lattice shifts!"

    return ij, D

def get_neighbourlist_ijDS(atoms: ase.Atoms, 
                           cutoff: float):
    """Returns a tuple of an array of atom-pair indices, an array of vectors 
    for that pair, and an array of shift vectors.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object for which to compute the neighborlist.
    cutoff : float
        Maximum distance to search for neighbors
    """
    i, j, D, S = neighbour_list("ijDS", atoms=atoms, cutoff=cutoff)
    ij = np.column_stack([i, j]).astype(int)

    return ij, D, S

class BlockedHamiltonian:
    def __init__(
        self,
        atoms: ase.Atoms,
        nbasis_species_dict: dict,
        hamiltonian,
        orbitals_permutation_dict=None,
    ):
        """_summary_

        Parameters
        ----------
        atoms : ase.Atoms
            _description_
        nbasis_species_dict : dict
            Number of basis functions for each element in the hamiltonian
        hamiltonian : np.ndarray
            2D array, the hamiltonian matrix
        orbitals_permutation_dict : dict
            Orbital permutation vector for each element's basis functions.
        """
        self.atoms = atoms
        self.nbasis_species_dict = nbasis_species_dict
        self.basis_starts = (
            np.cumsum([nbasis_species_dict[s] for s in atoms.symbols])
            - nbasis_species_dict[atoms[0].symbol]
        )
        self.hamiltonian = hamiltonian
        self.opd = orbitals_permutation_dict
        self.permute = False if self.opd is None else True

    def get_block(self, i, j):
        atoms_i_symbol = self.atoms[i].symbol
        atoms_j_symbol = self.atoms[j].symbol
        istart = self.basis_starts[i]
        istop = istart + self.nbasis_species_dict[atoms_i_symbol]
        jstart = self.basis_starts[j]
        jstop = jstart + self.nbasis_species_dict[atoms_j_symbol]
        try:
            hblock = (
                self.permute_rowcols(
                    self.hamiltonian[istart:istop, jstart:jstop],
                    self.opd[atoms_i_symbol],
                    self.opd[atoms_j_symbol],
                )
                if self.permute
                else self.hamiltonian[istart:istop, jstart:jstop]
            )
            return hblock
        except KeyError:
            print(list(self.opd.keys()))


class BlockedMatrix:
    def __init__(
        self,
        matrix,
        block_ids,
        block_sizes,
        permutation_dict=None,
    ):
        """Represents a minimal BlockedMatrix structure so we can call
        the (i, j)-th block of a block structured matrix. Pretty much any
        matrix made out of atom-wise basis functions will look like this.

        Parameters
        ----------
        matrix : np.ndarray
            The underlying 2D matrix.
        block_ids : int array
            Unique block identifiers for each block. Corresponds to 'atoms' for LCAO Hamiltonians, but is more general.
        block_sizes : int array
            Array of block sizes. This in principle can be a block_id : block_size dict, but it isn't.
        permutation_dict : dict, optional
            A block_id : int array dictionary to reorder rows/columns of each block, by default None
        """
        self.matrix = matrix
        self.block_ids = block_ids
        self.block_sizes = block_sizes
        self.block_starts = np.concatenate([[0], np.cumsum(block_sizes[:-1])])
        self.pd = permutation_dict

    def get_block(self, i, j):
        istart = self.block_starts[i]
        istop = istart + self.block_sizes[i]
        jstart = self.block_starts[j]
        jstop = jstart + self.block_sizes[j]
        try:
            block = self.matrix[istart:istop, jstart:jstop]
            return block
        except Exception as e:
            raise e

    def permute_rowcols(self, block, prows, pcols):
        # This particular part of the code doesn't need to be performant
        # but it does need to tell us if there's something going wrong.
        assert len(np.unique(prows)) == len(prows)
        assert len(np.unique(pcols)) == len(pcols)
        assert (
            len(prows) == block.shape[0]
        ), f"{len(prows)} elements in permutation but only {block.shape[0]} rows in Hamiltonian block!"
        assert (
            len(pcols) == block.shape[1]
        ), f"{len(pcols)} elements in permutation but only {block.shape[1]} cols in Hamiltonian block!"

        return block[prows, :][:, pcols]


def make_hamiltonian_blockedmatrix(
    H_MM, atoms, basis_block_sizes
):
    H_MM_blocked = BlockedMatrix(
        H_MM, atoms.numbers, basis_block_sizes
    )
    return H_MM_blocked

def get_permutation_dict(ells_dict: dict[int, list[int]],
                         ellwise_permutation_dict: dict[int, list[int]]):
    
    pd = {}
    # for each orbital angular momentum in the basis function for the atom,
    # add the ell-wise permutation indices, with an offset for the lower
    # angular momenta.
    for z, ells in ells_dict.items():
        idx = []
        for ell in ells:
            offset = np.max(np.concatenate(idx))+1 if len(idx) > 0 else 0
            idx.append(ellwise_permutation_dict[ell] + offset)
    
        idx = np.concatenate(idx)
        pd[z] = idx

    return pd

class VectorPermuter:
    def __init__(self, from_array, to_array):
        assert len(from_array) == len(
            to_array
        ), "Can only permute between equal-length iterables"
        assert set(from_array) == set(
            to_array
        ), "All elements of each iterable must be unique"
        permutation_matrix = to_array[:, None] == from_array[None, :]
        self._f = np.argwhere(permutation_matrix)[:, 1]
        self._b = np.argwhere(permutation_matrix.T)[:, 1]

    def forward(self, from_array):
        return from_array[self._f]

    def backward(self, to_array):
        return to_array[self._b]


# GPAW to OMX permutation dict for m's
# This is taken from DeepH-E3's data processing module
# not much else to do here
gpw2omx_lp = {
    0: np.array([0]),
    1: np.nonzero(np.eye(3, dtype=int)[[1, 2, 0]].T)[1],
    2: np.nonzero(np.eye(5, dtype=int)[[2, 4, 0, 3, 1]].T)[1],
    3: np.nonzero(np.eye(7, dtype=int)[[6, 4, 2, 0, 1, 3, 5]].T)[1],
}
