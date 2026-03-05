import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import jax.scipy.special as jsp
import iisignature
from functools import partial

from .tensor_sequence import TensorSequence
from .operators import D
from .tensor_product import tensor_prod
from .words import number_of_words_up_to_trunc, index_to_word_vect


def __2d_path_to_array(path: jax.Array, trunc: int) -> NDArray:
    """
    Converts a two-dimensional path into a two-dimensional array to be used to create the TensorSequence.

    :param path: A JAX array of shape (len(t_grid), dim) representing the path,
        where rows correspond to time steps and columns to dimensions.
    :param trunc: The truncation level for the signature computation.
    :return: A TensorSequence representing the signature of the path up to the specified truncation level.
    """
    array = iisignature.sig(path, trunc, 2)
    array = np.vstack([np.zeros(array.shape[1]), array])
    array = np.hstack([np.ones((array.shape[0], 1)), array])
    array = array.T
    return array


def path_to_signature(path: jax.Array, trunc: int, only_terminal_sig: bool = False) -> TensorSequence:
    """
    Converts a path into a TensorSequence by computing its signature up to a given truncation level.

    :param path: A JAX array of shape (len(t_grid), dim) representing the path,
        where rows correspond to time steps and columns to dimensions.
    :param trunc: The truncation level for the signature computation.
    :return: A TensorSequence representing the signature of the path up to the specified truncation level.
    """
    if path.ndim == 1:
        array = __2d_path_to_array(path=jnp.reshape(path, (1, -1)), trunc=trunc)
    elif path.ndim == 2:
        array = __2d_path_to_array(path=path, trunc=trunc)
    elif path.ndim == 3:
        dim = path.shape[1]
        array = np.zeros((number_of_words_up_to_trunc(trunc=trunc, dim=dim), path.shape[0], path.shape[2]))
        for i in range(path.shape[2]):
            array[:, :, i] = __2d_path_to_array(path=path[:, :, i], trunc=trunc)
    else:
        raise ValueError("Dimension of path should be less than 3.")
    if only_terminal_sig:
        return TensorSequence(array=jnp.array(array[:, -1]), trunc=trunc, dim=path.shape[1])
    else:
        return TensorSequence(array=jnp.array(array), trunc=trunc, dim=path.shape[1])


def path_to_fm_signature(path: jax.Array, trunc: int, t_grid: jax.Array, lam: jax.Array) -> TensorSequence:
    """
    Computes the EFM-signature with coefficient lam corresponding to a given d-dimensional path on the
    time grid t_grid up to the order trunc.

    :param path: Path as jax.Array of shape (len(t_grid), d).
    :param trunc: Truncation order, i.e. maximal order of coefficients to be calculated.
    :param t_grid: Time grid as jax.Array. An increasing time grid from T0 <= 0 to T > 0. The signature is calculated
        only for the positive values of grid.
    :param lam: a vector or a scalar value of signature mean reversion coefficients.

    :return: TensorSequence objet corresponding to a trajectory of signature of the path on t_grid corresponding
        to the positive values t_grid[t_grid >= 0]
    """
    dim = path.shape[1]

    if lam.size == 1:
        lam = jnp.ones(dim) * lam

    dX = jnp.diff(path, axis=0, prepend=path[0:1, :])
    dt = jnp.diff(t_grid, prepend=t_grid[0:1])

    if jnp.allclose(lam, lam[0]):
        dX_sig = __compute_inc_sig_constant_lam(dX=dX, dt=dt, lam=lam, dim=dim, trunc=trunc)
    else:
        dX_sig = __compute_inc_sig_vector_lam(dX=dX, dt=dt, lam=lam, dim=dim, trunc=trunc)

    acc_array = chen_cum_prod_efm(dX_sig, dt, trunc, lam, dim)

    return TensorSequence(array=acc_array[:, t_grid >= 0], trunc=trunc, dim=dim)


def path_to_rolling_signature(path: jax.Array, trunc: int, window_size: int) -> TensorSequence:
    """
    Computes the rolling window signature with given window size corresponding to a given d-dimensional path
    up to the order trunc.

    :param path: Path as jax.Array of shape (len(t_grid), d).
    :param trunc: Truncation order, i.e. maximal order of coefficients to be calculated.
    :param window_size: At each time t, the signature is computed over the last `window_size` time steps.

    :return: TensorSequence objet corresponding to a trajectory of rolling signature of the path.
    """
    dim = path.shape[1]

    dX = jnp.diff(path, axis=0, prepend=path[0:1, :])
    dt = jnp.ones(dX.shape[0])

    dX_sig = __compute_inc_sig_constant_lam(dX=dX, dt=dt, lam=jnp.zeros(dim), dim=dim, trunc=trunc)

    level_signs = (-1) ** dX_sig.subsequence((0,)).get_lengths_array()
    dX_sig_array_shifted_inv = jnp.zeros_like(dX_sig.array.T)
    dX_sig_array_shifted_inv = dX_sig_array_shifted_inv.at[:window_size, 0:1].set(1)
    # Shifting the increments by window_size.
    # inverting the tensor exponential by multiplying each element by (-1)^level.
    dX_sig_array_shifted_inv = dX_sig_array_shifted_inv.at[window_size:].set(dX_sig.array.T[:-window_size]) * level_signs

    acc_array = chen_cum_prod_rolling(dX_sig=dX_sig, trunc=trunc, dim=dim, dX_sig_array_shifted_inv=dX_sig_array_shifted_inv)

    return TensorSequence(array=acc_array, trunc=trunc, dim=dim)


def __compute_inc_sig_constant_lam(dX: jax.Array, dt: jax.Array, lam: jax.Array, dim: int, trunc: int):
    dt_col = dt.reshape((-1, 1))
    lam_row = lam.reshape((1, -1))
    c = jnp.where((dt_col * jnp.ones((1, dim)) > 0) & (~jnp.allclose(lam_row, 0)),
                  (1 - jnp.exp(-lam_row * dt_col)) / (lam_row * dt_col), 1)
    path_inc = dX * c

    n_indices = number_of_words_up_to_trunc(trunc, dim=dim)
    dX_sig_array = jnp.zeros((n_indices, dt.size))
    dX_sig_array = dX_sig_array.at[0].set(1)

    # Calculates step arrays: each array array_steps[:, k] corresponds to the signature bb{X}_{t_k, t_{k + 1}}.
    # n-th level of this signature is the tensor product of the path increments path[k + 1] - path[k] multiplied
    # by a signature of the linear path given by the function __h.
    tp_n = path_inc
    for n in range(1, trunc + 1):
        idx_start = number_of_words_up_to_trunc(n - 1, dim=dim)
        dX_sig_array = dX_sig_array.at[idx_start:idx_start + dim ** n].set(tp_n.T / jsp.factorial(n))
        tp_n = jnp.einsum("ij,ik->ijk", tp_n, path_inc).reshape((path_inc.shape[0], -1))

    # A more efficient computation of tensor_exp(e_1 + e_2 + ... + e_d)
    dX_sig = TensorSequence(array=dX_sig_array, trunc=trunc, dim=dim)
    return dX_sig

def __compute_inc_sig_vector_lam(dX: jax.Array, dt: jax.Array, lam: jax.Array, dim: int, trunc: int):
    """
    Computes the EFM-signature with coefficient lam corresponding to a given d-dimensional path on the
    time grid t_grid up to the order trunc.

    :param path: Path as jax.Array of shape (len(t_grid), d).
    :param trunc: Truncation order, i.e. maximal order of coefficients to be calculated.
    :param t_grid: Time grid as jax.Array. An increasing time grid from T0 < 0 to T > 0. The signature is calculated
        only for the positive values of grid.
    :param lam: a vector of signature mean reversion coefficients.

    :return: TensorSequence objet corresponding to a trajectory of signature of the path on t_grid corresponding
        to the positive values t_grid[t_grid >= 0]
    """
    n_indices = number_of_words_up_to_trunc(trunc, dim=dim)
    words = index_to_word_vect(jnp.arange(n_indices), dim)
    dX_sig_array = np.zeros((n_indices, dt.size), dtype=float)
    dX_sig_array[0] = 1

    # Calculates step arrays: each array array_steps[:, k] corresponds to the signature bb{X}_{t_k, t_{k + 1}}.
    # n-th level of this signature is the tensor product of the path increments path[k + 1] - path[k] multiplied
    # by a signature of the linear path given by the function __h.
    tp_n = dX
    for n in range(1, trunc + 1):
        idx_start = number_of_words_up_to_trunc(n - 1, dim=dim)
        dX_sig_array[idx_start:idx_start + dim ** n] = tp_n.T
        tp_n = jnp.einsum("ij,ik->ijk", tp_n, dX).reshape((dX.shape[0], -1))

    # Computes the EFM-signature of linear path word by word.
    for i, word in enumerate(words):
        if i > 0:
            word_as_indices = jnp.array([int(letter) - 1 for letter in str(word)])
            lam_word = lam[word_as_indices]
            dX_sig_array[i] *= fm_sig_from_word(dt, lam_word)

    # A more efficient computation of tensor_exp(e_1 + e_2 + ... + e_d)
    dX_sig = TensorSequence(array=dX_sig_array, trunc=trunc, dim=dim)
    return dX_sig


@jax.jit
def chen_cum_prod_efm(dX_sig, dt, trunc, lam, dim) -> jax.Array:
    """
    Transforms the signature bb{X}_{t_k, t_{k + 1}} of linear paths into the signature bb{X}_{t_{k + 1}}
    using the Chen's identity.
    """
    def f(carry, x: TensorSequence):
        dt_step, dx_arr = x
        result_array = tensor_prod(D(TensorSequence(carry, trunc=trunc, dim=dim), dt=dt_step, lam=lam),
                                   TensorSequence(dx_arr, trunc=trunc, dim=dim)).array
        return result_array, result_array

    init = jnp.zeros(len(dX_sig))
    init = init.at[0].set(1)

    _, ys = jax.lax.scan(f=f, init=init, xs=(dt, dX_sig.array.T))

    return ys.T

@jax.jit
@partial(jax.vmap, in_axes=(0, None))
def fm_sig_from_word(dt, lambda_word):
    mu = jnp.concatenate([jnp.zeros(1), jnp.cumsum(lambda_word)])
    mu_diff = mu[:, None] - mu[None, :]
    c = jnp.prod(jnp.where(mu_diff == 0, 1, 1 / mu_diff), axis=0)
    return jnp.exp(-mu * dt) @ c


@jax.jit
def chen_cum_prod_rolling(dX_sig, trunc, dim, dX_sig_array_shifted_inv: jax.Array) -> jax.Array:
    """
    Transforms the signature bb{X}_{t_k, t_{k + 1}} of linear paths into the signature
    bb{X}_{t_{k + 1 - window_size}, t_{k + 1}} using the Chen's identity.
    """

    def f(carry, x: TensorSequence):
        dx_arr, dx_arr_shifted_inv = x
        result_array = tensor_prod(
            tensor_prod(
                TensorSequence(dx_arr_shifted_inv, trunc=trunc, dim=dim),
                TensorSequence(carry, trunc=trunc, dim=dim)
            ),
            TensorSequence(dx_arr, trunc=trunc, dim=dim)
        ).array
        return result_array, result_array

    init = jnp.zeros(len(dX_sig))
    init = init.at[0].set(1)

    _, ys = jax.lax.scan(f=f, init=init, xs=(dX_sig.array.T, dX_sig_array_shifted_inv))

    return ys.T
