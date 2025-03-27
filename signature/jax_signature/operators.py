import jax
import jax.numpy as jnp
from .tensor_sequence_jax import TensorSequenceJAX


@jax.jit
def G(ts: TensorSequenceJAX) -> TensorSequenceJAX:
    """
    An operator multiplying the coefficients of tensor sequence by the lengths of the corresponding words.

    :param ts: tensor sequence to transform.

    :return: G(ts) as a new instance of TensorSequence.
    """
    lengths_shape = (-1, ) + (1, ) * (len(ts.array.shape) - 1)
    return TensorSequenceJAX(array=ts.array * ts.get_lengths_array().reshape(lengths_shape), trunc=ts.trunc, dim=ts.dim)


@jax.jit
def G_inv(ts: TensorSequenceJAX) -> TensorSequenceJAX:
    """
    An operator dividing the coefficients of tensor sequence by the lengths of the corresponding words (pseudo-inverse of G).

    :param ts: tensor sequence to transform.

    :return: G^{-1}(ts) as a new instance of TensorSequence.
    """
    lengths = ts.get_lengths_array()
    lengths_shape = (-1, ) + (1, ) * (len(ts.array.shape) - 1)
    return TensorSequenceJAX(array=ts.array * jnp.where(lengths != 0, 1 / lengths, 0).reshape(lengths_shape), trunc=ts.trunc, dim=ts.dim)


@jax.jit
def discount_ts(ts: TensorSequenceJAX, dt: float, lam: float) -> TensorSequenceJAX:
    """
    A discounting operator with discounting rate lambda and discounting period dt.
    Multiplies the coefficient l^v by exp(-lam * |v| * dt).

    :param ts: tensor sequence to discount.
    :param dt: length of the discounting period.
    :param lam: discounting rate.

    :return: Discounted tensor sequence.
    """
    lengths = ts.get_lengths_array()
    lengths_shape = (-1, ) + (1, ) * (len(ts.array.shape) - 1)
    return TensorSequenceJAX(array=ts.array * jnp.reshape(jnp.exp(-lengths * lam * dt), lengths_shape), trunc=ts.trunc, dim=ts.dim)


@jax.jit
def G_resolvent(ts: TensorSequenceJAX, lam: float) -> TensorSequenceJAX:
    """
    Calculates the operator (Id + lam * G)^{-1}

    :param ts: tensor sequence to transform.
    :param lam: mean-reversion rate.

    :return: A result of application of the operator to ts.
    """
    lengths = ts.get_lengths_array()
    lengths_shape = (-1, ) + (1, ) * (len(ts.array.shape) - 1)
    return TensorSequenceJAX(ts.array * jnp.reshape(1 / (1 + lam * lengths), lengths_shape), trunc=ts.trunc, dim=ts.dim)


@jax.jit
def semi_integrated_scheme(ts: TensorSequenceJAX, dt: float, lam: float) -> TensorSequenceJAX:
    """
    A numerical scheme for integration of the equation
    psi' = -lam *G(psi) + F
    Given by an operator (lam * G)^{-1}(Id - D_h^lam).

    :param ts: tensor sequence to transform.
    :param dt: time step.
    :param lam: mean-reversion rate.

    :return: A result of application of the operator to ts.
    """
    lengths_arr = ts.get_lengths_array()
    lengths_shape = (-1, ) + (1, ) * (len(ts.array.shape) - 1)
    coefficients = jnp.where(lengths_arr != 0, (1 - jnp.exp(-lengths_arr * lam * dt)) / (lam * lengths_arr), dt)
    return TensorSequenceJAX(array=ts.array * coefficients.reshape(lengths_shape), trunc=ts.trunc, dim=ts.dim)
