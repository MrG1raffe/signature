import jax
import jax.numpy as jnp
from typing import Union

from .tensor_sequence import TensorSequence
from .operators import G_inv
from .factory import unit
from .words import number_of_words_up_to_trunc
from .tensor_product import tensor_prod, tensor_exp, tensor_prod_word
from .ode_integration import ode_solver, step_fun_semi_int_pece


def expected_bm_signature(t: Union[float, jax.Array], trunc: int) -> TensorSequence:
    """
    Calculates the expected signature of X_t = (t, W_t).

    :param t: time grid to calculate the expected signature.
    :param trunc: truncation order.

    :return: expected signature evaluated at t as a TensorSequence instance.
    """
    # w = (1 + 0.5 * 22) * t
    w = get_1_22(trunc) * jnp.reshape(t, (1, -1))
    return tensor_exp(ts=w, N_trunc=trunc)


def expected_bm_stationary_signature(trunc: int, lam: jax.Array, t: float = None, n_points: int = 100) -> TensorSequence:
    """
    Computes expected stationary lambda-signature of X_t = (t, W_t). If t is not specified,
    computes stationary expected signature E^lam = E[SigX^lam]. Otherwise, computes E_t^lam = E[SigX_{0, t}^lam].

    :param trunc: truncation order of the result.
    :param lam: stationary signature parameter.
    :param t: time index of the expected signature. By default, t = inf, which corresponds to stationary signature.
    :param n_points: number of points to be used to solve an SDE on E_t^lam.

    :return: expected signature as a TensorSequence instance.
    """
    dim = 2
    # w = 1 + 0.5 * 22
    w = get_1_22(trunc)

    if t is None:
        res = unit(trunc, dim)
        v = unit(trunc, dim)
        for _ in range(trunc):
            v = G_inv(tensor_prod(v, w), lam)
            res = res + v
        return res
    else:
        t_grid = jnp.linspace(0, t, n_points)
        args = {"lam": lam}
        return ode_solver(fun=__expected_sig_ode_func, step_fun=step_fun_semi_int_pece,
                          t_grid=t_grid, init=unit(trunc, dim), args=args)


def get_1_22(trunc: int) -> TensorSequence:
    array = jnp.zeros(number_of_words_up_to_trunc(trunc, 2))
    array = array.at[jnp.array([1, 6])].set(jnp.array([1, 0.5]))
    return TensorSequence(array=array, trunc=trunc, dim=2)


@jax.jit
def __expected_sig_ode_func(ell: TensorSequence, args: dict) -> TensorSequence:
    return tensor_prod_word(ell, 1) + tensor_prod_word(ell, 22) / 2
