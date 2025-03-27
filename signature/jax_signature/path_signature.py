import jax
import jax.numpy as jnp
import iisignature

from .tensor_sequence_jax import  TensorSequenceJAX
from .operators import discount_ts
from .tensor_product import tensor_exp, tensor_prod
from .factory import from_array


def __2d_path_to_array(path: jax.Array, trunc: int) -> jax.Array:
    """
    Converts a two-dimensional path into a two-dimensional array to be used to create the TensorSequence.

    :param path: A JAX array of shape (len(t_grid), dim) representing the path,
        where rows correspond to time steps and columns to dimensions.
    :param trunc: The truncation level for the signature computation.
    :return: A TensorSequence representing the signature of the path up to the specified truncation level.
    """
    array = iisignature.sig(path, trunc, 2)
    array = jnp.vstack([jnp.zeros(array.shape[1]), array])
    array = jnp.hstack([jnp.ones((array.shape[0], 1)), array])
    array = array.T
    return array


def path_to_signature(path: jax.Array, trunc: int) -> TensorSequenceJAX:
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
        array = jnp.zeros((2 ** (trunc + 1) - 1, path.shape[0], path.shape[2]))
        for i in range(path.shape[2]):
            array = array.at[:, :, i].set(__2d_path_to_array(path=path[:, :, i], trunc=trunc))
    else:
        raise ValueError("Dimension of path should be less than 3.")
    return TensorSequenceJAX(array=array, trunc=trunc, dim=path.shape[1])


def path_to_stationary_signature(path: jax.Array, trunc: int, t_grid: jax.Array, lam: float) -> TensorSequenceJAX:
    """
    Computes the stationary signature with coefficient lam corresponding to a given d-dimensional path on the
    time grid t_grid up to the order trunc.

    :param path: Path as jax.Array of shape (len(t_grid), d).
    :param trunc: Truncation order, i.e. maximal order of coefficients to be calculated.
    :param t_grid: Time grid as jax.Array. An increasing time grid from T0 < 0 to T > 0. The signature is calculated
        only for the positive values of grid.
    :param lam: signature mean reversion coefficient.

    :return: TensorSequence objet corresponding to a trajectory of signature of the path on t_grid corresponding
        to the positive values t_grid[t_grid >= 0]
    """
    dim = path.shape[1]
    dX = jnp.diff(path, axis=0, prepend=path[0:1, :])
    dt = jnp.diff(t_grid, prepend=t_grid[0])

    dX_ts = from_array(array=jnp.vstack([jnp.zeros(dX.shape[0]), dX.T]), trunc=trunc, dim=dim)
    acc_array = chen_cum_prod_stat(dX_ts, dt, trunc, lam, dim)
    return TensorSequenceJAX(array=acc_array[:, t_grid >= 0], trunc=trunc, dim=dim)


@jax.jit
def chen_cum_prod_stat(dX_ts, dt, trunc, lam, dim) -> jax.Array:
    """
    Transforms the signature bb{X}_{t_k, t_{k + 1}} of linear paths into the signature bb{X}_{t_{k + 1}}
    using the Chen's identity.
    """
    c = jnp.where(dt > 0, (1 - jnp.exp(-lam * dt)) / (lam * dt), 1)
    dX_sig = tensor_exp(dX_ts * c, N_trunc=trunc)

    def f(carry, x: TensorSequenceJAX):
        dt_step, dx_arr = x
        result_array = tensor_prod(discount_ts(TensorSequenceJAX(carry, trunc=trunc, dim=dim), dt=dt_step, lam=lam),
                                      TensorSequenceJAX(dx_arr, trunc=trunc, dim=dim)).array
        return result_array, result_array

    init = jnp.zeros(len(dX_sig))
    init = init.at[0].set(1)

    _, ys = jax.lax.scan(f=f, init=init, xs=(dt, dX_sig.array.T))

    return ys.T
