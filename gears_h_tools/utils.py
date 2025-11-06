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
        self.permute = False if self.pd is None else True
    
    def get_block(self, i, j):
        istart = self.block_starts[i]
        istop = istart + self.block_sizes[i]
        jstart = self.block_starts[j]
        jstop = jstart + self.block_sizes[j]
        
        if self.permute:
            block = self.permute_rowcols(self.matrix[istart:istop, jstart:jstop],
                                         self.pd[i], self.pd[j])
        else:
            block = self.matrix[istart:istop, jstart:jstop]
        
        return block

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
    H_MM, atoms, basis_block_sizes, permutation_dict=None
):
    H_MM_blocked = BlockedMatrix(
        H_MM, atoms.numbers, basis_block_sizes, permutation_dict
    )
    return H_MM_blocked

def get_permutation_dict(ells_dict: dict[int, list[int]],
                         ellwise_permutation_dict: dict[int, np.ndarray]):
    
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

def make_hmatrix(numbers: np.ndarray, 
                 offblocks: list[np.ndarray], 
                 onblocks: list[np.ndarray], 
                 species_basis_size_dict: dict[int, int]):
    """
    Combine the sparse Hamilotonian blocks to make the Hamiltonian matrix.
    Makes the resulting Hamiltonian Hermitian before it is returned.
    Taken from gears_h.infer.infer: 
    https://github.com/SamsungDS/gears_h/blob/main/gears_h/infer/infer.py#L202

    Args:
        numbers (np.ndarray): Atomic numbers of the inference system.
        offblocks (list[np.ndarray]): List of the off-diagonal Hamiltonian blocks.
        onblocks (list[np.ndarray]): List of the on-diagonal Hamiltonian blocks.
        species_basis_size_dict (dict[int, int]): Dictionary in which the keys are atomic numbers and values 
            are the number of basis functions for each atomic species.

    Returns:
        np.ndarray: The Hamiltonian matrix.
    """
    spd = species_basis_size_dict

    idxs = np.array([0] + [spd[i] for i in numbers], dtype=np.int32)
    idxs = np.cumsum(idxs)

    hmatrix = [[None] * len(numbers) for _ in range(len(numbers))]

    for onblock_stack in onblocks:
        for onblock, idx in zip(*onblock_stack):
            hmatrix[idx][idx] = onblock

    for offblock_stack in offblocks:
        for offblock, pair_idx in zip(*offblock_stack):
            i, j = pair_idx
            # TODO: replace this loop and these conditionals with a groupby and reduce
            if hmatrix[i][j] is None:
                hmatrix[i][j] = offblock
            else:
                hmatrix[i][j] += offblock

    blocks = np.asarray(hmatrix, dtype='object')
    if blocks.ndim == 2:
        hmatrix = block_array(hmatrix)
        return (0.5 * (hmatrix + hmatrix.T.conj())).toarray()
    elif blocks.ndim == 4:
        hmatrix = np.block(hmatrix)
        return 0.5 * (hmatrix + hmatrix.T.conj())

def blocked_matrix_to_hmatrix(blocked_hamiltonian: BlockedMatrix,
                              atoms: ase.Atoms,
                              species_basis_size_dict: dict[int, int],
                              ij: np.ndarray) -> np.ndarray:
    """This is only useful when shuffling the read in matrix is required.
    Shuffling is automatically handled by BlockedMatrix, but we need to
    reassemble the shuffled H matrix to write it out.

    Args:
        blocked_hamiltonian (BlockedMatrix): Blocked Hamiltonian matrix.
        atoms (ase.Atoms): Atomic structure
        species_basis_size_dict (dict[int, int]): keys are atomic species, values are number of basis functions.
        ij (np.ndarray): Neighbor list.

    Returns:
        np.ndarray: The reassembled Hamiltonian matrix.
    """
    # Arrange off-diagonal blocks into a list of tuples.
    # Each tuple contains a list of off-diagonal blocks and the ij of those blocks.

    atomic_number_pairs = atoms.numbers[ij]
    assert atomic_number_pairs.shape[-1] == 2
    # unique species pairs
    unique_elementpairs = np.unique(atomic_number_pairs, axis=0)

    pair_hblocks_list = []
    for pair in unique_elementpairs:
        # indices of all pairs with this unique combination of atoms
        boolean_indices_of_pairs = np.all(atomic_number_pairs == pair, axis=1)
        # pre-allocate blocks in shape (n_pairs, n_bf_1, n_bf_2)
        hblocks_of_pair = np.zeros((sum(boolean_indices_of_pairs), 
                                    *[species_basis_size_dict[s] for s in pair])).astype(np.float32)
        # for each pair of neighbors
        for i, tij in enumerate(ij[boolean_indices_of_pairs]):
            # get the hblock of that pair of neighbors, shuffle, and store in our array
            hblocks_of_pair[i] = blocked_hamiltonian.get_block(*tij)

        pair_hblocks_list.append((hblocks_of_pair, ij[boolean_indices_of_pairs]))

    # Arrange on-diagonal blocks into a list of tuples.
    # Each tuple contains the on-diagonal blocks for a species and the index of
    # H where the block belongs.
    
    # get atom indices
    atom_indices = np.arange(len(atoms.numbers))

    species_hblocks_list = []
    # for each species
    for an in np.unique(atoms.numbers):
        # get indices of atoms that match the current species
        boolean_indices_of_species = (an == atoms.numbers)
        # Allocate numpy array of the correct size, (nblocks, block size, block size)
        species_hblocks = np.zeros((sum(boolean_indices_of_species), 
                                    species_basis_size_dict[an], 
                                    species_basis_size_dict[an])).astype(np.float32)
        for i, ai in enumerate(atom_indices[boolean_indices_of_species]):
            species_hblocks[i] = blocked_hamiltonian.get_block(ai,ai)
        
        species_hblocks = np.array(species_hblocks)#.astype(np.float32) # no array problems here since species-wise on-diags always have the same size
        # append species-wise blocks and the corresponding atom indices to our return list
        species_hblocks_list.append((species_hblocks, atom_indices[boolean_indices_of_species]))

    # Assemble H
    hmatrix = make_hmatrix(atoms.numbers,
                           offblocks=pair_hblocks_list,
                           onblocks=species_hblocks_list,
                           species_basis_size_dict=species_basis_size_dict)
    return hmatrix

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
