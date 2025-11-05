from pathlib import Path

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
        # print(i)
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