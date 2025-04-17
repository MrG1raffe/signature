import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

from .factory import unit_like
from .tensor_sequence import TensorSequence


@jax.jit
def _shuffle_prod_arr(
    arr1: jax.Array,
    arr2: jax.Array,
    shuffle_table: jax.Array,
):
    index_left, index_right, index_result, count = shuffle_table

    source = count * arr1[index_left] * arr2[index_right]
    linear_result = jnp.zeros_like(arr1)  # keeps the same size as ts1.
    linear_result = linear_result.at[index_result].add(source)

    return linear_result


_shuffle_prod_arr_vect = jax.jit(jax.vmap(_shuffle_prod_arr, in_axes=(1, 1, None), out_axes=1))


@jax.jit
def shuffle_prod(ts1: TensorSequence, ts2: TensorSequence, shuffle_table: jax.Array) -> TensorSequence:
    return TensorSequence(array=_shuffle_prod_arr(ts1.array, ts2.array, shuffle_table),
                          trunc=ts1.trunc, dim=ts1.dim)


@jax.jit
def shuffle_prod_2d(ts1: TensorSequence, ts2: TensorSequence, shuffle_table: jax.Array) -> TensorSequence:
    return TensorSequence(array=_shuffle_prod_arr_vect(ts1.array.reshape((len(ts1), -1)),
                                                       ts2.array.reshape((len(ts2), -1)),
                                                       shuffle_table).reshape(ts1.shape),
                          trunc=ts1.trunc, dim=ts1.dim)


@jax.jit
def shuffle_pow(ts: TensorSequence, p: int, shuffle_table: jax.Array) -> TensorSequence:
    """
    Raises the TensorSequence to a shuffle power p.

    :param ts:
    :param p: The power to which the TensorSequence is raised.
    :param shuffle_table:
    :return: A new TensorSequence representing the shuffle power.
    """
    def body_fun(i, acc):
        return shuffle_prod(ts1=acc, ts2=ts, shuffle_table=shuffle_table)

    return jax.lax.fori_loop(lower=0, upper=p, body_fun=body_fun, init_val=unit_like(ts))


@jax.jit
def shuffle_exp(ts: TensorSequence, N_trunc: int, shuffle_table: jax.Array) -> TensorSequence:
    """
    Computes the shuffle exponential of the TensorSequence up to a specified truncation level.

    :param ts:
    :param N_trunc: The truncation level for the exponential.
    :param shuffle_table:
    :return: A new TensorSequence representing the shuffle exponential.
    """

    def body_fun(i, acc):
        return acc + shuffle_pow(ts=ts, p=i, shuffle_table=shuffle_table) / jsp.factorial(i)

    return jax.lax.fori_loop(lower=1, upper=N_trunc + 1, body_fun=body_fun, init_val=unit_like(ts))
