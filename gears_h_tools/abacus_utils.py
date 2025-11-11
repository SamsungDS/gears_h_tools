from collections import defaultdict
from pathlib import Path
import re

from ase.data import atomic_numbers
from ase.units import Rydberg
import numpy as np
from scipy.sparse import csr_matrix

def read_H_csrs_and_shifts(csr_path: Path,
                           threshold: float):
    with open(csr_path, "r") as f:
        lines = f.readlines()

    # Matrix dimension and number of matrices
    dim = int(lines[1].split()[-1])
    # nhmat = int(lines[2].split()[-1])
    # Make lists for storing shift vectors and CSR matrices
    shift_vectors = []
    csr_mats = []

    nlines = len(lines)
    # Loop over lines to extract CSRs for each shift vector.
    # See documentation of the file format here:
    # https://abacus.deepmodeling.com/en/latest/advanced/elec_properties/hs_matrix.html#out-mat-hs2
    i = 3
    while i < nlines:
        svec_nnz = np.fromstring(lines[i], sep = ' ', dtype=int)
        svec, nnz = svec_nnz[:-1], svec_nnz[-1]
        if nnz > 0:
            data = np.fromstring(lines[i+1], sep = ' ', dtype=float)
            col_indices = np.fromstring(lines[i+2], sep = ' ', dtype=int)
            indptr = np.fromstring(lines[i+3], sep = ' ', dtype=int)
            tcsr = csr_matrix((data, col_indices, indptr), shape=(dim,dim))*Rydberg
            if np.max(tcsr) > threshold:
                csr_mats.append(tcsr)
                shift_vectors.append(svec)
            i += 4
        elif nnz == 0:
            i += 1
    
    return csr_mats, shift_vectors

def read_S_csrs(csr_path: Path,
               shift_vectors: list[np.ndarray]):
    with open(csr_path, "r") as f:
        lines = f.readlines()

    # Matrix dimension and number of matrices
    dim = int(lines[1].split()[-1])
    csr_mats = []

    nlines = len(lines)
    # Loop over lines to extract CSRs for each shift vector.
    # See documentation of the file format here:
    # https://abacus.deepmodeling.com/en/latest/advanced/elec_properties/hs_matrix.html#out-mat-hs2
    i = 3
    while i < nlines:
        svec_nnz = np.fromstring(lines[i], sep = ' ', dtype=int)
        svec, nnz = svec_nnz[:-1], svec_nnz[-1]
        if nnz > 0:
            # Don't include S matrices that we do not have Hamiltonians for.
            if not np.any(np.all(svec == shift_vectors, axis=1)):
                i += 4
                continue
            else:
                data = np.fromstring(lines[i+1], sep = ' ', dtype=float)
                col_indices = np.fromstring(lines[i+2], sep = ' ', dtype=int)
                indptr = np.fromstring(lines[i+3], sep = ' ', dtype=int)
                tcsr = csr_matrix((data, col_indices, indptr), shape=(dim,dim))
                csr_mats.append(tcsr)
                i += 4
        elif nnz == 0:
            i += 1
    
    return csr_mats

def get_abacus_ells_dict(logfile_path: Path | str) -> dict[int, list[int]]:
    """Returns a dictionary of the angular momenta of the basis
    functions for each species in the system.

    Args:
        logfile_path (Path | str): Path to the log file, typically named running_scf.log.

    Returns:
        dict[int, list[int]]: Dictionary with keys that are atomic species numbers
                              and values that are lists of the angular momenta of
                              the basis functions.
    """
    # I am truly sorry for this regex.
    atom_ells_pattern = re.compile(r"\sAtom label = (.+)\n((?:(?!^\s+Number of atoms for this type)[\s\S])*)", re.MULTILINE)
    ell_zeta_pattern = re.compile(r"\s+L=(\d), number of zeta = (\d)")
    with open(logfile_path, "r") as f:
        log = f.read()

    ells_dict = defaultdict(list)
    
    atom_ells_matches = atom_ells_pattern.findall(log)
    for match in atom_ells_matches:
        atomic_species = match[0]
        ells_zetas = ell_zeta_pattern.findall(match[1])
        for ez_match in ells_zetas:
            ell = int(ez_match[0])
            nzeta = int(ez_match[1])
            for i in range(nzeta):
                ells_dict[atomic_numbers[atomic_species]].append(ell)
    
    return ells_dict

def get_abacus_ellwise_permutation_dict() -> dict[int, list[int]]:
    """Get sort indices for permuting the abacus m ordering (defined below) to the 
    non-Cartesian ordering required by gears_h. We use the second convention
    discussed here: https://e3x.readthedocs.io/stable/pitfalls.html#ordering-of-irreps
    Abacus's m order is discussed here:
    https://abacus.deepmodeling.com/en/latest/advanced/pp_orb.html#basis-set

    Returns:
        dict[int, list[int]]: A dictionary with ell for the keys and the sort indices
                              to permute from the Abacus ordering to the ordering used
                              in gears_h. Defined up to ell = 4, but can be trivially
                              extended if needed.
    """
    abacus_ordering = [0, 1, -1, 2, -2, 3, -3, 4, -4]
    abacus_ordering_per_ell = [abacus_ordering[:n] for n in [2*l + 1 for l in range(5)]]
    ellwise_permutation_dict = {ell: np.argsort(a) for ell, a in enumerate(abacus_ordering_per_ell)}
    
    return ellwise_permutation_dict
