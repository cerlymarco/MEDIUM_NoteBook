import numpy as np
from scipy import sparse as sp

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops.linalg.sparse import sparse as tfsp
from tensorflow.keras import backend as K

SINGLE  = 1   # Single         (rank(a)=2, rank(b)=2)
MIXED   = 2   # Mixed          (rank(a)=2, rank(b)=3)
iMIXED  = 3   # Inverted mixed (rank(a)=3, rank(b)=2)
BATCH   = 4   # Batch          (rank(a)=3, rank(b)=3)
UNKNOWN = -1  # Unknown


def transpose(a, perm=None, name=None):
    """
    Transposes a according to perm, dealing automatically with sparsity.
    :param a: Tensor or SparseTensor with rank k.
    :param perm: permutation indices of size k.
    :param name: name for the operation.
    :return: Tensor or SparseTensor with rank k.
    """
    if K.is_sparse(a):
        transpose_op = tf.sparse.transpose
    else:
        transpose_op = tf.transpose

    if perm is None:
        perm = (1, 0)  # Make explicit so that shape will always be preserved
    return transpose_op(a, perm=perm, name=name)


def reshape(a, shape=None, name=None):
    """
    Reshapes a according to shape, dealing automatically with sparsity.
    :param a: Tensor or SparseTensor.
    :param shape: new shape.
    :param name: name for the operation.
    :return: Tensor or SparseTensor.
    """
    if K.is_sparse(a):
        reshape_op = tf.sparse.reshape
    else:
        reshape_op = tf.reshape

    return reshape_op(a, shape=shape, name=name)


def autodetect_mode(a, b):
    """
    Return a code identifying the mode of operation (single, mixed, inverted mixed and
    batch), given a and b. See `ops.modes` for meaning of codes.
    :param a: Tensor or SparseTensor.
    :param b: Tensor or SparseTensor.
    :return: mode of operation as an integer code.
    """
    a_dim = K.ndim(a)
    b_dim = K.ndim(b)
    if b_dim == 2:
        if a_dim == 2:
            return SINGLE
        elif a_dim == 3:
            return iMIXED
    elif b_dim == 3:
        if a_dim == 2:
            return MIXED
        elif a_dim == 3:
            return BATCH
    return UNKNOWN


def filter_dot(fltr, features):
    """
    Wrapper for matmul_A_B, specifically used to compute the matrix multiplication
    between a graph filter and node features.
    :param fltr:
    :param features: the node features (N x F in single mode, batch x N x F in
    mixed and batch mode).
    :return: the filtered features.
    """
    mode = autodetect_mode(fltr, features)
    if mode == SINGLE or mode == BATCH:
        return dot(fltr, features)
    else:
        # Mixed mode
        return mixed_mode_dot(fltr, features)


def dot(a, b, transpose_a=False, transpose_b=False):
    """
    Dot product between a and b along innermost dimensions, for a and b with
    same rank. Supports both dense and sparse multiplication (including
    sparse-sparse).
    :param a: Tensor or SparseTensor with rank 2 or 3.
    :param b: Tensor or SparseTensor with same rank as a.
    :param transpose_a: bool, transpose innermost two dimensions of a.
    :param transpose_b: bool, transpose innermost two dimensions of b.
    :return: Tensor or SparseTensor with rank 2 or 3.
    """
    a_is_sparse_tensor = isinstance(a, tf.SparseTensor)
    b_is_sparse_tensor = isinstance(b, tf.SparseTensor)
    if a_is_sparse_tensor:
        a = tfsp.CSRSparseMatrix(a)
    if b_is_sparse_tensor:
        b = tfsp.CSRSparseMatrix(b)
    out = tfsp.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
    if hasattr(out, 'to_sparse_tensor'):
        return out.to_sparse_tensor()

    return out


def mixed_mode_dot(a, b):
    """
    Computes the equivalent of `tf.einsum('ij,bjk->bik', a, b)`, but
    works for both dense and sparse input filters.
    :param a: rank 2 Tensor or SparseTensor.
    :param b: rank 3 Tensor or SparseTensor.
    :return: rank 3 Tensor or SparseTensor.
    """
    s_0_, s_1_, s_2_ = K.int_shape(b)
    B_T = transpose(b, (1, 2, 0))
    B_T = reshape(B_T, (s_1_, -1))
    output = dot(a, B_T)
    output = reshape(output, (s_1_, s_2_, -1))
    output = transpose(output, (2, 0, 1))

    return output


def degree_power(A, k):
    r"""
    Computes \(\D^{k}\) from the given adjacency matrix. Useful for computing
    normalised Laplacian.
    :param A: rank 2 array or sparse matrix.
    :param k: exponent to which elevate the degree matrix.
    :return: if A is a dense array, a dense array; if A is sparse, a sparse
    matrix in DIA format.
    """
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def normalized_adjacency(A, symmetric=True):
    r"""
    Normalizes the given adjacency matrix using the degree matrix as either
    \(\D^{-1}\A\) or \(\D^{-1/2}\A\D^{-1/2}\) (symmetric normalization).
    :param A: rank 2 array or sparse matrix;
    :param symmetric: boolean, compute symmetric normalization;
    :return: the normalized adjacency matrix.
    """
    if symmetric:
        normalized_D = degree_power(A, -0.5)
        output = normalized_D.dot(A).dot(normalized_D)
    else:
        normalized_D = degree_power(A, -1.)
        output = normalized_D.dot(A)
    return output


def localpooling_filter(A, symmetric=True):
    r"""
    Computes the graph filter described in
    [Kipf & Welling (2017)](https://arxiv.org/abs/1609.02907).
    :param A: array or sparse matrix with rank 2 or 3;
    :param symmetric: boolean, whether to normalize the matrix as
    \(\D^{-\frac{1}{2}}\A\D^{-\frac{1}{2}}\) or as \(\D^{-1}\A\);
    :return: array or sparse matrix with rank 2 or 3, same as A;
    """
    fltr = A.copy()
    if sp.issparse(A):
        I = sp.eye(A.shape[-1], dtype=A.dtype)
    else:
        I = np.eye(A.shape[-1], dtype=A.dtype)
    if A.ndim == 3:
        for i in range(A.shape[0]):
            A_tilde = A[i] + I
            fltr[i] = normalized_adjacency(A_tilde, symmetric=symmetric)
    else:
        A_tilde = A + I
        fltr = normalized_adjacency(A_tilde, symmetric=symmetric)

    if sp.issparse(fltr):
        fltr.sort_indices()
    return fltr