import jax
import jax.numpy as jnp
import iisignature

from .tensor_sequence_jax import  TensorSequenceJAX


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


def path_to_sequence(path: jax.Array, trunc: int) -> TensorSequenceJAX:
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