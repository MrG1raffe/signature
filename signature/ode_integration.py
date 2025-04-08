import jax
import jax.numpy as jnp
from typing import Callable
from .tensor_sequence import TensorSequence
from .operators import D, semi_integrated_scheme

from functools import partial


@partial(jax.jit, static_argnames=['fun', 'step_fun'])
def ode_solver(
    fun: Callable,
    step_fun: Callable,
    t_grid: jax.Array,
    init: TensorSequence,
    args: dict = None
) -> TensorSequence:
    dt = jnp.diff(t_grid)

    def fori_fun(i, psi):
        psi_next = step_fun(psi=psi, dt=dt[i], fun=fun, args=args)
        return psi_next

    psi_res = jax.lax.fori_loop(lower=0, upper=len(dt), body_fun=fori_fun, init_val=init)
    return psi_res


@partial(jax.jit, static_argnames=['fun', 'step_fun'])
def ode_solver_traj(
    fun: Callable,
    step_fun: Callable,
    t_grid: jax.Array,
    init: TensorSequence,
    args: dict = None
) -> TensorSequence:
    dt_arr = jnp.diff(t_grid)

    def scan_fun(psi, dt):
        psi_next = step_fun(psi=psi, dt=dt, fun=fun, args=args)
        return psi_next, psi_next.array

    psi_res, arr_res = jax.lax.scan(scan_fun, init=init, xs=dt_arr)
    arr_res = jnp.vstack([init.array, arr_res])
    return TensorSequence(array=arr_res.T, trunc=init.trunc, dim=init.dim)


@partial(jax.jit, static_argnames=['fun'])
def step_fun_semi_int_euler(psi: TensorSequence, dt: float, fun: Callable, args: dict) -> TensorSequence:
    return D(ts=psi, dt=dt, lam=args["lam"]) + semi_integrated_scheme(ts=fun(psi, args), dt=dt, lam=args["lam"])


@partial(jax.jit, static_argnames=['fun'])
def step_fun_semi_int_pece(psi: TensorSequence, dt: float, fun: Callable, args: dict) -> TensorSequence:
    fun_psi = fun(psi, args)
    psi_pred = D(ts=psi, dt=dt, lam=args["lam"]) + semi_integrated_scheme(ts=fun_psi, dt=dt, lam=args["lam"])
    psi_next = D(ts=psi, dt=dt, lam=args["lam"]) + \
               semi_integrated_scheme(ts=(fun(psi_pred, args) + fun_psi) * 0.5, dt=dt, lam=args["lam"])
    return psi_next


@partial(jax.jit, static_argnames=['fun'])
def step_fun_semi_int_rk4(psi: TensorSequence, dt: float, fun: Callable, args: dict):
    k1 = fun(psi, args)
    k2 = fun(psi + k1 * (dt / 2), args)
    k3 = fun(psi + k2 * (dt / 2), args)
    k4 = fun(psi + k3 * dt, args)
    psi_next = D(ts=psi, dt=dt, lam=args["lam"]) + \
               semi_integrated_scheme(ts=(k1 + k2 * 2 + k3 * 2 + k4) / 6, dt=dt, lam=args["lam"])
    return psi_next


@partial(jax.jit, static_argnames=['fun'])
def step_fun_euler(psi: TensorSequence, dt: float, fun: Callable, args: dict):
    return psi + fun(psi, args) * dt


@partial(jax.jit, static_argnames=['fun'])
def step_fun_pece(psi: TensorSequence, dt: float, fun: Callable, args: dict):
    fun_psi = fun(psi, args)
    psi_pred = psi + fun_psi * dt
    psi_next = psi + (fun(psi_pred, args) + fun_psi) * (dt / 2)
    return psi_next


@partial(jax.jit, static_argnames=['fun'])
def step_fun_rk4(psi: TensorSequence, dt: float, fun: Callable, args: dict):
    k1 = fun(psi, args)
    k2 = fun(psi + k1 * (dt / 2), args)
    k3 = fun(psi + k2 * (dt / 2), args)
    k4 = fun(psi + k3 * dt, args)
    psi_next = psi + (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6)
    return psi_next
