"""Unit tests for the time-dependent source plumbing in signature.ode_integration.

The solvers inject the (reversed-time) integration step index as ``args["i"]`` so a right-hand
side can select a time slice of a ``(n_sig, n_dw)`` source. Two invariants are pinned:

* a *constant* ``(n_sig, n_dw)`` source reproduces the old single-tensor-sequence source exactly
  (strict generalization / no regression);
* a *time-dependent* source matches a hand-written Euler loop that indexes ``source[:, i]`` by hand
  (the index/reversal convention is exactly the integration step).

A backward-compatibility check confirms a right-hand side that ignores ``args["i"]`` still runs.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import signature.tensor_algebra as ta
from signature.ode_integration import ode_solver_traj, step_fun_pece, step_fun_euler

DIM, TRUNC = 2, 6


def _setup():
    sht = ta.get_shuffle_table(table_trunc=TRUNC, dim=DIM)
    n_dw = 60
    t_grid = np.linspace(0.0, 0.1, n_dw)
    n_sig = int(ta.number_of_words_up_to_trunc(TRUNC, DIM))
    e_state, unit = ta.from_word(2, TRUNC, DIM), ta.unit(TRUNC, DIM)
    Y1 = ta.shuffle_pow(e_state, 2, sht) - unit * 0.25
    q = ta.shuffle_pow(Y1, 2, sht) * (-1.0)            # gentle so the Riccati stays bounded
    r = ta.from_word(22, TRUNC, DIM) * (-0.3)
    return sht, t_grid, n_dw, n_sig, q, r


def _rhs(psi, args):
    return psi.proj(1) + psi.proj(22) * 0.5 + ta.shuffle_pow(psi.proj(2), 2, args["shuffle_table"]) * 0.5


class TestTimeDependentSource(unittest.TestCase):
    def test_constant_source_matches_single_tensor_sequence(self):
        sht, t_grid, n_dw, n_sig, q, r = _setup()

        @jax.jit
        def ode_fun_new(psi, args):
            src = ta.TensorSequence(array=args["source"].array[:, args["i"]], trunc=psi.trunc, dim=psi.dim)
            return _rhs(psi, args) + src

        @jax.jit
        def ode_fun_old(psi, args):
            return _rhs(psi, args) + args["source"]

        src_const = ta.from_array(jnp.broadcast_to(q.array[:, None], (n_sig, n_dw)), TRUNC, DIM)
        psi_new = ode_solver_traj(fun=ode_fun_new, step_fun=step_fun_pece, t_grid=t_grid, init=r,
                                  args={"shuffle_table": sht, "source": src_const})
        psi_old = ode_solver_traj(fun=ode_fun_old, step_fun=step_fun_pece, t_grid=t_grid, init=r,
                                  args={"shuffle_table": sht, "source": q})
        np.testing.assert_allclose(np.asarray(psi_new.array), np.asarray(psi_old.array), atol=0, rtol=0)

    def test_time_dependent_source_matches_manual_loop(self):
        sht, t_grid, n_dw, n_sig, q, r = _setup()
        dt = np.diff(t_grid)
        rng = np.random.default_rng(0)
        profile = rng.uniform(0.5, 1.5, size=n_dw)
        src_td = ta.from_array(jnp.asarray(q.array[:, None] * profile[None, :]), TRUNC, DIM)

        @jax.jit
        def ode_fun(psi, args):
            src = ta.TensorSequence(array=args["source"].array[:, args["i"]], trunc=psi.trunc, dim=psi.dim)
            return _rhs(psi, args) + src

        psi_solver = ode_solver_traj(fun=ode_fun, step_fun=step_fun_euler, t_grid=t_grid, init=r,
                                     args={"shuffle_table": sht, "source": src_td})
        # explicit Euler reference using source[:, i] at integration step i
        psi, cols = r, [np.asarray(r.array)]
        for i in range(n_dw - 1):
            src_i = ta.TensorSequence(array=src_td.array[:, i], trunc=TRUNC, dim=DIM)
            psi = psi + (_rhs(psi, {"shuffle_table": sht}) + src_i) * dt[i]
            cols.append(np.asarray(psi.array))
        np.testing.assert_allclose(np.asarray(psi_solver.array), np.asarray(cols).T, atol=1e-12)

    def test_rhs_ignoring_index_still_runs(self):
        sht, t_grid, n_dw, n_sig, q, r = _setup()
        psi = ode_solver_traj(fun=jax.jit(_rhs), step_fun=step_fun_pece, t_grid=t_grid, init=r,
                              args={"shuffle_table": sht})
        self.assertEqual(psi.array.shape, (n_sig, n_dw))


if __name__ == "__main__":
    unittest.main(verbosity=2)
