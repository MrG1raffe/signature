import jax
import jax.numpy as jnp
from .tensor_sequence import TensorSequence


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
