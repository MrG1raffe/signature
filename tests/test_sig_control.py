"""Unit tests for signature.sig_control (the unified control rollout and the policies).

Two error-prone parts are pinned down:

* :func:`simulate` -- the shared rollout loop -- must keep the signature consistent with the
  path it integrates: for the time-augmented motion the level-1 coordinates are exactly the
  elapsed time and the state increment.
* :func:`monte_carlo_policy` -- the recentering identities for the value and control -- must
  match a brute-force evaluation of the conditional expectations.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import iisignature

import signature.tensor_algebra as ta
from signature import sig_control as sc
from signature.words import index_to_word, word_to_index, number_of_words_up_to_trunc


DIM = 2


def make_brownian(size, n_dw, seed=0):
    rng = np.random.default_rng(seed)
    t_grid = np.linspace(0.0, 0.1, n_dw)
    dt = np.diff(t_grid)
    dW = rng.normal(size=(size, n_dw - 1)) * np.sqrt(dt)
    return t_grid, dW


def make_sig_brownian(p_deg, N_MC, n_fine=60, seed=1):
    rng = np.random.default_rng(seed)
    u = np.linspace(0.0, 1.0, n_fine)
    W = np.cumsum(rng.normal(size=(N_MC, n_fine - 1)) * np.sqrt(1.0 / (n_fine - 1)), axis=1)
    paths = np.zeros((N_MC, n_fine, DIM))
    paths[:, 1:, 0] = u[1:][None, :]
    paths[:, 1:, 1] = W
    arr = np.vstack([np.ones((1, N_MC)), iisignature.sig(paths, p_deg).T])
    return ta.from_array(jnp.asarray(arr), trunc=p_deg, dim=DIM)


class TestSimulateBookkeeping(unittest.TestCase):
    def test_signature_matches_path(self):
        """With any policy, the level-1 signature coordinates are time and the state increment."""
        trunc, size, n_dw = 5, 4, 40
        t_grid, dW = make_brownian(size, n_dw, seed=3)

        # an arbitrary (state-dependent) policy to exercise the loop non-trivially
        def policy(i, t, X_sig):
            x = np.asarray(X_sig.array[word_to_index(2, DIM)])   # current state from the signature
            return -2.0 * x, np.zeros(size), {}

        res = sc.simulate(dW, t_grid, policy, trunc, DIM)
        i1, i2 = int(word_to_index(1, DIM)), int(word_to_index(2, DIM))
        # Sig^1 = elapsed time, Sig^2 = X_t - x0
        np.testing.assert_allclose(res.sig[i1], np.broadcast_to(t_grid, (size, n_dw)), atol=1e-9)
        np.testing.assert_allclose(res.sig[i2], res.X, atol=1e-9)
        # state really is the integral of the applied control plus the noise
        recon = np.cumsum(np.concatenate([np.zeros((size, 1)),
                          res.control[:, :-1] * np.diff(t_grid) + dW], axis=1), axis=1)
        np.testing.assert_allclose(res.X, recon, atol=1e-9)


class TestMonteCarloPolicy(unittest.TestCase):
    def _brute(self, p, X_col, sigW_col, insert=None):
        """Reference <p, X (x) [insert] (x) sigW> by deconcatenating the words of p."""
        total = 0.0
        pa = np.asarray(p.array)
        for idx in np.nonzero(np.abs(pa) > 1e-12)[0]:
            word = int(index_to_word(idx, p.dim)); coef = pa[idx]
            sw = "" if word == 0 else str(word)
            if insert is None:
                splits = [(sw[:k], sw[k:]) for k in range(len(sw) + 1)]
            else:
                splits = [(sw[:k], sw[k + 1:]) for k in range(len(sw)) if sw[k] == str(insert)]
            for sa, sb in splits:
                a = 0 if sa == "" else int(sa)
                b = 0 if sb == "" else int(sb)
                ia, ib = int(word_to_index(a, p.dim)), int(word_to_index(b, p.dim))
                if ia < X_col.shape[0] and ib < sigW_col.shape[0]:
                    total += coef * X_col[ia] * sigW_col[ib]
        return total

    def test_value_and_control_match_brute_force(self):
        trunc, size, n_dw, T = 6, 3, 30, 0.1
        # a finite-degree p with a few state-involving words
        p = (ta.from_word(2, trunc, DIM) * 1.3
             - ta.from_word(22, trunc, DIM) * 0.7
             + ta.from_word(212, trunc, DIM) * 0.5)
        p_deg = int(p.get_lengths_array()[np.abs(np.asarray(p.array)) > 1e-12].max())
        N_MC = 4000
        sigW1 = make_sig_brownian(p_deg, N_MC)
        policy = sc.monte_carlo_policy(p, sigW1, T, trunc, DIM)

        # a non-trivial current signature: roll a short zero-control path
        t_grid, dW = make_brownian(size, n_dw, seed=7)
        run = sc.simulate(dW, t_grid, lambda i, t, X: (np.zeros(size), np.zeros(size), {}), trunc, DIM)
        X_sig = ta.from_array(jnp.asarray(run.sig[:, :, 15]), trunc=trunc, dim=DIM)

        t = 0.03
        alpha, psi, _ = policy(5, t, X_sig)

        # brute-force recentering: psi = log E[exp<p, X (x) sigW>], alpha = E[exp<.> <p,X(x)2(x)sigW>]/exp(psi)
        sigW = np.asarray(ta.dilation(sigW1, jnp.array([T - t, np.sqrt(T - t)])).array)
        for s in range(size):
            Xc = np.asarray(X_sig.array[:, s])
            pq = np.array([self._brute(p, Xc, sigW[:, m]) for m in range(N_MC)])
            pr = np.array([self._brute(p, Xc, sigW[:, m], insert=2) for m in range(N_MC)])
            m = pq.max(); w = np.exp(pq - m)
            psi_ref = m + np.log(w.mean())
            alpha_ref = np.sum(w * pr) / np.sum(w)
            self.assertAlmostEqual(float(psi[s]), psi_ref, places=10)
            self.assertAlmostEqual(float(alpha[s]), alpha_ref, places=10)


class TestValueProcess(unittest.TestCase):
    """With a running-cost source q, the policy value is the value process
    <psi_t, Sig(X)_t> + int_0^t <q_s, Sig(X)_s> ds.  For a constant q the integral has the exact
    signature form int_0^t <q, Sig(X)_s> ds = <tensor_prod(q, e_time), Sig(X)_t>, so the value
    process equals <psi_t + tensor_prod(q, e_time), Sig(X)_t> up to the trapezoidal error."""

    def _solve(self, trunc, n_dw):
        from signature.ode_integration import ode_solver_traj, step_fun_pece
        sht = ta.get_shuffle_table(table_trunc=trunc, dim=DIM)
        t_grid = np.linspace(0.0, 0.1, n_dw)
        n_sig = int(number_of_words_up_to_trunc(trunc, DIM))
        Y1 = ta.shuffle_pow(ta.from_word(2, trunc, DIM), 2, sht) - ta.unit(trunc, DIM) * 0.25
        q = ta.shuffle_pow(Y1, 2, sht) * (-1.0)
        r = ta.from_word(22, trunc, DIM) * (-0.3)
        source = ta.from_array(jnp.broadcast_to(q.array[:, None], (n_sig, n_dw)), trunc, DIM)

        @jax.jit
        def ode_fun(psi, args):
            src = ta.TensorSequence(array=args["source"].array[:, args["i"]], trunc=psi.trunc, dim=psi.dim)
            return (psi.proj(1) + psi.proj(22) * 0.5
                    + ta.shuffle_pow(psi.proj(2), 2, args["shuffle_table"]) * 0.5 + src)

        psi = ode_solver_traj(fun=ode_fun, step_fun=step_fun_pece, t_grid=t_grid, init=r,
                              args={"shuffle_table": sht, "source": source})
        return psi, q, source, t_grid, n_sig

    def test_value_process_matches_signature_exact_for_constant_q(self):
        trunc, n_dw, size = 6, 200, 3
        psi, q, source, t_grid, n_sig = self._solve(trunc, n_dw)
        dW = np.random.default_rng(0).normal(size=(size, n_dw - 1)) * np.sqrt(np.diff(t_grid))

        res = sc.simulate(dW, t_grid, sc.riccati_policy(psi, source=source), trunc, DIM)
        res0 = sc.simulate(dW, t_grid, sc.riccati_policy(psi), trunc, DIM)

        # value-to-go (no source) is exactly <psi_t, Sig(X)_t>
        vtg = np.stack([np.asarray(psi.array[:, -(j + 1)]) @ res.sig[:, :, j] for j in range(n_dw - 1)], axis=1)
        np.testing.assert_allclose(res0.value[:, :-1], vtg, atol=1e-12)
        # value process == <psi_t + tensor_prod(q, e_time), Sig(X)_t>   (trapezoidal vs exact)
        q_int = ta.tensor_prod(q, ta.from_word(1, trunc, DIM))
        ref = np.stack([(np.asarray(psi.array[:, -(j + 1)]) + np.asarray(q_int.array)) @ res.sig[:, :, j]
                        for j in range(n_dw - 1)], axis=1)
        np.testing.assert_allclose(res.value[:, :-1], ref, atol=1e-4)
        # the running-cost term is non-trivial and vanishes at t = 0
        self.assertGreater(np.max(np.abs(res.value[:, :-1] - res0.value[:, :-1])), 1e-3)
        np.testing.assert_allclose(res.value[:, 0], res0.value[:, 0], atol=1e-12)


class TestMonteCarloSourcePolicy(unittest.TestCase):
    """The path-dependent-source MC handles a TIME-DEPENDENT running cost q; its value process and
    control must match the Riccati (which solves the time-dependent source exactly) to MC noise."""

    def test_matches_riccati_for_time_dependent_q(self):
        from signature.ode_integration import ode_solver_traj, step_fun_pece
        trunc, n_dw, T, size = 6, 150, 0.1, 4
        sht = ta.get_shuffle_table(table_trunc=trunc, dim=DIM)
        t_grid = np.linspace(0.0, T, n_dw)
        n_sig = int(number_of_words_up_to_trunc(trunc, DIM))
        Y1 = ta.shuffle_pow(ta.from_word(2, trunc, DIM), 2, sht) - ta.unit(trunc, DIM) * 0.25
        q_base = ta.shuffle_pow(Y1, 2, sht) * (-1.0)
        r = ta.from_word(22, trunc, DIM) * (-0.3)
        # time-dependent q: column c is q at calendar time T - t_grid[c]
        source = ta.from_array(jnp.asarray(np.stack(
            [np.asarray(q_base.array) * (1.0 + 2.0 * (T - t_grid[c]) / T) for c in range(n_dw)], axis=1)),
            trunc, DIM)

        @jax.jit
        def ode_fun(psi, args):
            src = ta.TensorSequence(array=args["source"].array[:, args["i"]], trunc=psi.trunc, dim=psi.dim)
            return (psi.proj(1) + psi.proj(22) * 0.5
                    + ta.shuffle_pow(psi.proj(2), 2, args["shuffle_table"]) * 0.5 + src)

        psi = ode_solver_traj(fun=ode_fun, step_fun=step_fun_pece, t_grid=t_grid, init=r,
                              args={"shuffle_table": sht, "source": source})

        deg = max(sc._max_degree(r), sc._max_degree(source))
        traj = sc.brownian_signature_trajectory(t_grid, N_MC=6000, trunc_mc=deg, dim=DIM, seed=7)

        dW = np.random.default_rng(11).normal(size=(size, n_dw - 1)) * np.sqrt(np.diff(t_grid))
        ric = sc.simulate(dW, t_grid, sc.riccati_policy(psi, source=source), trunc, DIM)
        mc = sc.simulate(dW, t_grid, sc.monte_carlo_source_policy(r, source, traj, t_grid, DIM), trunc, DIM)

        self.assertLess(np.max(np.abs(ric.value[:, :-1] - mc.value[:, :-1])), 5e-3)
        self.assertLess(np.max(np.abs(ric.control[:, :-1] - mc.control[:, :-1])), 1e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
