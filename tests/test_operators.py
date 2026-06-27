"""Unit tests for tensor-sequence operators (currently the dilation operator).

``dilation(ts, c)`` multiplies the coefficient ts^{i_1...i_n} by c_{i_1} * ... * c_{i_n},
with ``c`` either a scalar or an array of size ``dim``. Correctness is checked against a
brute-force per-letter product, against the discounting operator ``D`` (a dilation with
c_i = exp(-lam_i * dt)), for batched coefficient arrays, and under ``jax.jit``.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from signature.tensor_sequence import TensorSequence
from signature.operators import dilation, D
from signature.words import number_of_words_up_to_trunc, index_to_word


DIMS = {1: 5, 2: 4, 3: 3, 4: 2}


def make_ts(trunc, dim, seed=0, extra=()):
    n = int(number_of_words_up_to_trunc(trunc, dim))
    array = jax.random.normal(jax.random.PRNGKey(seed), (n,) + extra)
    return TensorSequence(array=array, trunc=trunc, dim=dim)


def _letters(word):
    return [] if int(word) == 0 else [int(d) for d in str(int(word))]


def brute_dilation(ts, c_vec):
    """Reference: scale coefficient of word i_1...i_n by prod_k c_vec[i_k - 1]."""
    n = len(ts)
    out = np.array(ts.array, dtype=np.asarray(c_vec).dtype).copy()
    for i in range(n):
        factor = 1.0
        for letter in _letters(index_to_word(i, ts.dim)):
            factor *= c_vec[letter - 1]
        out[i] = out[i] * factor
    return out


class TestDilation(unittest.TestCase):
    def test_per_letter_vector_matches_reference(self):
        for dim, trunc in DIMS.items():
            ts = make_ts(trunc, dim, seed=dim)
            c = np.array([0.5 + 0.3 * k for k in range(dim)])
            got = np.array(dilation(ts, jnp.array(c)).array)
            np.testing.assert_allclose(got, brute_dilation(ts, c), err_msg=f"dim={dim}")

    def test_scalar_matches_reference(self):
        for dim, trunc in DIMS.items():
            ts = make_ts(trunc, dim, seed=dim + 5)
            c_scalar = 0.7
            got = np.array(dilation(ts, c_scalar).array)
            ref = brute_dilation(ts, np.full(dim, c_scalar))
            np.testing.assert_allclose(got, ref, err_msg=f"dim={dim}")

    def test_empty_word_unchanged(self):
        for dim, trunc in DIMS.items():
            ts = make_ts(trunc, dim, seed=dim + 10)
            self.assertAlmostEqual(float(dilation(ts, 0.3).array[0]), float(ts.array[0]))

    def test_identity_when_c_is_one(self):
        for dim, trunc in DIMS.items():
            ts = make_ts(trunc, dim, seed=dim + 15)
            np.testing.assert_allclose(np.array(dilation(ts, 1.0).array), np.array(ts.array))

    def test_consistency_with_D(self):
        """D is the dilation with c_i = exp(-lam_i * dt)."""
        for dim, trunc in DIMS.items():
            ts = make_ts(trunc, dim, seed=dim + 20)
            lam = jnp.array([0.2 * (k + 1) for k in range(dim)])
            dt = 0.37
            np.testing.assert_allclose(
                np.array(dilation(ts, jnp.exp(-lam * dt)).array),
                np.array(D(ts, dt, lam).array),
                err_msg=f"dim={dim}",
            )

    def test_multiplicative_composition(self):
        """dilation(., a) then dilation(., b) equals dilation(., a*b)."""
        dim, trunc = 3, 3
        ts = make_ts(trunc, dim, seed=123)
        a = jnp.array([0.5, 1.2, 0.9])
        b = jnp.array([1.1, 0.7, 1.4])
        composed = dilation(dilation(ts, a), b).array
        direct = dilation(ts, a * b).array
        np.testing.assert_allclose(np.array(composed), np.array(direct))

    def test_batched_trailing_axis(self):
        ts = make_ts(3, 2, seed=7, extra=(5,))
        c = np.array([0.6, 1.3])
        got = np.array(dilation(ts, jnp.array(c)).array)
        for s in range(ts.array.shape[1]):
            sl = TensorSequence(array=ts.array[:, s], trunc=3, dim=2)
            np.testing.assert_allclose(got[:, s], brute_dilation(sl, c), err_msg=f"slice={s}")

    def test_jit(self):
        ts = make_ts(3, 2, seed=9)
        c = jnp.array([0.6, 1.3])
        np.testing.assert_allclose(
            np.array(jax.jit(dilation)(ts, c).array),
            np.array(dilation(ts, c).array),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
