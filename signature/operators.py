import jax
import jax.numpy as jnp
from .tensor_sequence import TensorSequence
from .words import word_len, index_to_word_vect, number_of_words_up_to_trunc, word_to_index_vect
from .shuffle_product import shuffle_prod


@jax.jit
def G(ts: TensorSequence, lam: jax.Array) -> TensorSequence:
    """
    An operator multiplying the coefficients ts[v] by lam(v), where
    lam(v) = sum of lam[i] for i in v.

    :param ts: tensor sequence to transform.
    :param lam: vector of multiplication coefficients.

    :return: G(ts) as a new instance of TensorSequence.
    """
    lams_shape = (-1, ) + (1, ) * (len(ts.array.shape) - 1)
    return TensorSequence(array=ts.array * ts.get_lambdas_sum_array(lam).reshape(lams_shape), trunc=ts.trunc, dim=ts.dim)


@jax.jit
def G_inv(ts: TensorSequence, lam: jax.Array) -> TensorSequence:
    """
    An operator dividing the coefficients of tensor sequence ts[v] by lam(v), where
    lam(v) = sum of lam[i] for i in v. (pseudo-inverse of G).
    The coefficient corresponding to the empty word is set to 0.

    :param ts: tensor sequence to transform.
    :param lam: vector of multiplication coefficients.

    :return: G^{-1}(ts) as a new instance of TensorSequence.
    """
    lams = ts.get_lambdas_sum_array(lam)
    lams_shape = (-1, ) + (1, ) * (len(ts.array.shape) - 1)
    return TensorSequence(array=ts.array * jnp.where(lams != 0, 1 / lams, 0).reshape(lams_shape), trunc=ts.trunc, dim=ts.dim)


@jax.jit
def D(ts: TensorSequence, dt: float, lam: jax.Array) -> TensorSequence:
    """
    A discounting operator with discounting rate lambda and discounting period dt.
    Multiplies the coefficient l^v by exp(-lam(v) * dt), where
    lam(v) = sum of lam[i] for i in v.

    :param ts: tensor sequence to discount.
    :param dt: length of the discounting period.
    :param lam: discounting rate.

    :return: Discounted tensor sequence.
    """
    lams = ts.get_lambdas_sum_array(lam)
    lams_shape = (-1, ) + (1, ) * (len(ts.array.shape) - 1)
    return TensorSequence(array=ts.array * jnp.reshape(jnp.exp(-lams * dt), lams_shape), trunc=ts.trunc, dim=ts.dim)


@jax.jit
def G_resolvent(ts: TensorSequence, lam: jax.Array) -> TensorSequence:
    """
    Calculates the operator (Id + G)^{-1}

    :param ts: tensor sequence to transform.
    :param lam: mean-reversion rate.

    :return: A result of application of the operator to ts.
    """
    lams = ts.get_lambdas_sum_array(lam)
    lams_shape = (-1, ) + (1, ) * (len(ts.array.shape) - 1)
    return TensorSequence(ts.array * jnp.reshape(1 / (1 + lams), lams_shape), trunc=ts.trunc, dim=ts.dim)


@jax.jit
def semi_integrated_scheme(ts: TensorSequence, dt: float, lam: jax.Array) -> TensorSequence:
    """
    A numerical scheme for integration of the equation
    psi' = -G(psi) + F
    Given by an operator G^{-1}(Id - D_h^lam).

    :param ts: tensor sequence to transform.
    :param dt: time step.
    :param lam: mean-reversion rate.

    :return: A result of application of the operator to ts.
    """
    lams = ts.get_lambdas_sum_array(lam)
    lams_shape = (-1, ) + (1, ) * (len(ts.array.shape) - 1)
    coefficients = jnp.where(lams != 0, (1 - jnp.exp(-lams * dt)) / lams, dt)
    return TensorSequence(array=ts.array * coefficients.reshape(lams_shape), trunc=ts.trunc, dim=ts.dim)


# @jax.jit
def Psi(ts: TensorSequence, word_upper: int, word_lower: int) -> TensorSequence:
    """
    Implements an operator Psi^{word_upper}_{word_lower} that returns a new tensor sequence `ts_res` defined by
    ts_res = sum_{w1, w2} ts[w1 ⊗ word_upper ⊗ w2] * (w1 ⊗ word_lower ⊗ w2).

    :param ts: Tensor sequence to transform.
    :param word_upper: Upper word of the operator.
    :param word_lower: Lower word of the operator.

    :return: A result of application of the operator to `ts`.
    """
    len_upper = word_len(word_upper)
    len_lower = word_len(word_lower)

    N = ts.trunc - jnp.minimum(len_upper, len_lower)
    cartesian_size = (1 - (N + 2) * ts.dim**(N + 1) + (N + 1) * ts.dim**(N + 2)) // (1 - ts.dim)**2
    words_cartesian = jnp.zeros((cartesian_size, 2), dtype=jnp.int64)
    lengths_cartesian = jnp.zeros((cartesian_size, 2), dtype=jnp.int64)

    ptr = 0
    for i in range(N + 1):
        # array of left words w1
        words_i = index_to_word_vect(
            number_of_words_up_to_trunc(i - 1, ts.dim) + jnp.arange(ts.dim**i), ts.dim
        )
        right_words_size = number_of_words_up_to_trunc(N - i, ts.dim)
        block_size = ts.dim ** i * right_words_size
        # array of possible right words w2 such that |w1| + |w2| <= N
        words_comp = index_to_word_vect(jnp.arange(right_words_size), ts.dim)
        # cartesian product
        words_cartesian = words_cartesian.at[ptr : ptr + block_size].set(
            jnp.stack(jnp.broadcast_arrays(words_i[:, None], words_comp[None, :]), axis=-1).reshape(-1, 2)
        )
        lengths_cartesian = lengths_cartesian.at[ptr: ptr + block_size].set(
            jnp.stack(jnp.broadcast_arrays(word_len(words_i[:, None]), word_len(words_comp[None, :])), axis=-1).reshape(-1, 2)
        )
        ptr += block_size

    words_cartesian, lengths_cartesian = words_cartesian.T, lengths_cartesian.T

    indices_lower = word_to_index_vect(
        words_cartesian[0] * 10 ** (lengths_cartesian[1] + len_lower) + word_lower * 10 ** lengths_cartesian[1] + words_cartesian[1],
        ts.dim
    )
    indices_upper = word_to_index_vect(
        words_cartesian[0] * 10 ** (lengths_cartesian[1] + len_upper) + word_upper * 10 ** lengths_cartesian[1] + words_cartesian[1],
        ts.dim
    )

    new_array = jnp.zeros_like(ts.array)
    new_array = new_array.at[indices_lower].add(ts.array[indices_upper] * (indices_upper < len(ts.array)))
    return TensorSequence(new_array, ts.trunc, ts.dim)


def diamond(ts1: TensorSequence, ts2: TensorSequence, letter_upper: int, letter_lower: int, shuffle_table: jax.Array):
    """
    Computes the diamond product ◇^{letter_upper, letter_upper}_{letter_lower} given by
    0.5(Psi(ts1 ⧢ ts2) - ts1 ⧢ Psi(ts2) - Psi(ts1) ⧢ ts2,
    where Psi = Psi^{letter_upper, letter_upper}_{letter_lower}.

    :param ts1: The first tensor sequence argument.
    :param ts2: The second tensor sequence argument.
    :param letter_upper: Upper letter in the diamond operator.
    :param letter_lower: Lower letter in the diamond operator.

    :return: A result of application of the operator to `ts1`, `ts2`.
    """
    word_upper = 10 * letter_upper + letter_upper # ii
    word_lower = letter_lower
    return 0.5 * (Psi(shuffle_prod(ts1, ts2, shuffle_table), word_upper, word_lower) -
                  shuffle_prod(ts1, Psi(ts2, word_upper, word_lower), shuffle_table) -
                  shuffle_prod(Psi(ts1, word_upper, word_lower), ts2, shuffle_table))
