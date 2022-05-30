from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import sparse_dot_topn.sparse_dot_topn as ct

def cossim_top(A:csr_matrix, B:csr_matrix, ntop:int, lower_bound:int = 0) -> pd.DataFrame:
    """Computes sparse cosine similarity. 
    Only highest ntop similarities are kept and only similarities bigger then lower_bound are kept.

    Args:
        A (csr_matrix): [description]
        B (csr_matrix): [description]
        ntop (int): 
        lower_bound (int, optional): Lower treshold for similarities that are kept. Defaults to 0.

    Returns:
        csr_matrix: Sparse similarity matrix.
    """

    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape

    idx_dtype = np.int32

    nnz_max = M * ntop

    indptr = np.zeros(M + 1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data, indices, indptr), shape=(M, N))