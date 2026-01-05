import jax
import jax.numpy as jnp
import numpy as np
from .tensor_sequence import TensorSequence
from .words import word_len, word_to_index, index_to_word_len, number_of_words_up_to_trunc, index_to_word


@jax.jit
def proj(ts: TensorSequence, word: int) -> TensorSequence:
    """
    Calculates the projection of TensorSequence with respect to the given word.

    :param ts: A tensor sequence which projection should be computed.
    :param word: The word (as integer) to calculate the projection.
    :return: A new TensorSequence representing the projection.
    """
    indices = jnp.arange(len(ts))
    word_length = word_len(word)
    word_index = word_to_index(word, dim=ts.dim)

    # A mask for indices to keep
    indices_mask = (((indices - word_index) % ts.dim**word_length) == 0) & (indices >= word_index)

    # Compute new indices
    length_arr = index_to_word_len(indices, dim=ts.dim)
    new_indices = (indices - ts.dim ** length_arr + 1) // ts.dim ** word_length + \
                  ts.dim ** (length_arr - word_length) - 1
    # Set out-of-bounds index for non-valid ones
    new_indices = jnp.where(indices_mask, new_indices, len(ts) + 1)

    array = jnp.zeros_like(ts.array)
    array = array.at[new_indices].set(jnp.where(jnp.einsum("i..., i -> i...", jnp.ones_like(ts.array), indices_mask),
                                                jnp.einsum("i..., i -> i...", ts.array, indices_mask), 0))

    return TensorSequence(array=array, trunc=ts.trunc, dim=ts.dim)


def get_projection_matrix(ts: TensorSequence):
    """
    Construct the projection matrix associated with a tensor sequence.

    This function builds a square matrix P whose rows correspond to the
    projections of the tensor sequence ``ts`` onto all words over the
    alphabet {1, …, ts.dim} of length up to ``ts.trunc``. The words are
    ordered according to the indexing used by ``index_to_word``.

    More precisely, the i-th row of P is given by
        P[i] = ts.proj(v).array
    where v is the word corresponding to index i.

    Parameters
    ----------
    ts : TensorSequence
        The tensor sequence defining the projection operator. Its
        attributes ``dim`` and ``trunc`` determine the set of words
        considered.

    Returns
    -------
    jax.numpy.ndarray
        A square array of shape (dim_sig, dim_sig), where
        ``dim_sig = number_of_words_up_to_trunc(ts.dim, ts.trunc)``.
        Each row contains the coordinates of the projection of ``ts``
        onto the corresponding word.
    """
    dim_sig = number_of_words_up_to_trunc(dim=ts.dim, trunc=ts.trunc)
    proj_mat = np.zeros((dim_sig, dim_sig))

    for i in range(dim_sig):
        v = index_to_word(i, ts.dim)
        proj_mat[i] = ts.proj(v).array

    return jnp.array(proj_mat)


def left_proj_on_seq(ts: TensorSequence, proj_on: TensorSequence):
    """
    Compute the left projection of a tensor sequence ``ts`` onto another sequence ``proj_on``.

    This operation applies the projection matrix associated with ``ts``
    to the left of the coefficient array of ``proj_on``.

    Parameters
    ----------
    ts : TensorSequence
        The tensor sequence being projected.
    proj_on : TensorSequence
        Projection tensor sequence.

    Returns
    -------
    TensorSequence
        A new tensor sequence whose array is given by
        ``P @ proj_on.array``, with truncation level ``ts.trunc`` and
        dimension ``ts.dim``.
    """
    proj_mat = get_projection_matrix(ts)
    return TensorSequence(array=proj_mat @ proj_on.array, trunc=ts.trunc, dim=ts.dim)


def right_proj_on_seq(ts: TensorSequence, proj_on: TensorSequence):
    """
    Compute the right projection of a tensor sequence ``ts`` onto another sequence ``proj_on``.

    This operation applies the projection matrix associated with ``ts``
    to the right of the coefficient array of ``proj_on``.

    Parameters
    ----------
    ts : TensorSequence
        The tensor sequence being projected.
    proj_on : TensorSequence
        Projection tensor sequence.

    Returns
    -------
    TensorSequence
        A new tensor sequence whose array is given by
        ``P @ proj_on.array``, with truncation level ``ts.trunc`` and
        dimension ``ts.dim``.
    """
    proj_mat = get_projection_matrix(ts)
    return TensorSequence(array=proj_mat.T @ proj_on.array, trunc=ts.trunc, dim=ts.dim)