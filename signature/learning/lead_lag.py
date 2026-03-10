import jax
import jax.numpy as jnp
from typing import Tuple

def lead_lag_transform(path: jax.Array, lead_lag_idx: jax.Array = None) -> jax.Array:
    """
    Applies the lead-lag transform to a path.

    :param path: numpy array of shape (n_steps, dim)
    :return: numpy array of shape (2 * n_steps - 1, 2 * dim)
    """
    # n_steps, dim = path.shape
    # Repeat each point twice to create the 'steps'
    # path_repeated shape: (2 * n_steps, dim)
    path_repeated = jnp.repeat(path, 2, axis=0)

    # Lead path: take everything except the first point
    # Lag path: take everything except the last point
    if lead_lag_idx is None:
        lead_lag_idx = jnp.arange(0, path.shape[1])

    lead = path_repeated[1:]
    lag = path_repeated[:-1, lead_lag_idx]

    # Concatenate them to get a path in 2*dim space
    return jnp.concatenate([lead, lag], axis=1)


def efm_lead_lag_transform(t_grid: jax.Array, path: jax.Array, lead_lag_idx: jax.Array = None) -> Tuple[jax.Array, jax.Array]:
    """
    Applies the lead-lag transform to a path.

    :param path: numpy array of shape (n_steps, dim)
    :return: numpy array of shape (3 * n_steps - 2, 2 * dim)
    """
    # n_steps, dim = path.shape
    # Repeat each point twice to create the 'steps'
    # path_repeated shape: (2 * n_steps, dim)
    path_repeated = jnp.repeat(path, 3, axis=0)
    t_grid_repeated = jnp.repeat(t_grid.reshape((-1, 1)), 3, axis=0)

    # Lead path: take everything except the first point
    # Lag path: take everything except the last point
    if lead_lag_idx is None:
        lead_lag_idx = jnp.arange(0, path.shape[1])

    lead = path_repeated[1:-1]
    lag = path_repeated[:-2, lead_lag_idx]

    # Concatenate them to get a path in 2*dim space
    return t_grid_repeated[2:], jnp.concatenate([lead, lag], axis=1)