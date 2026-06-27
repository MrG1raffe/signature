"""Unit tests for the projection / shift operators on tensor sequences.

The operators under test are:

* ``TensorSequence.proj``        -- right shift,  result^u = ts^{u word}
* ``TensorSequence.left_proj``   -- left shift,   result^u = ts^{word u}
* ``signature.projection.proj``  -- free-function version of the right shift
* ``left_proj_on_seq`` / ``right_proj_on_seq`` / ``get_projection_matrix``

Correctness is checked against an independent brute-force reference that manipulates
words directly through their decimal-string encoding (concatenation of letters), so it
does not reuse the index arithmetic being tested. All operators are exercised for
several dimensions, including ``dim == 1`` and ``dim > 2``.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from signature.tensor_sequence import TensorSequence
from signature.projection import (
    proj as proj_fn,
    left_proj_on_seq,
    right_proj_on_seq,
    get_projection_matrix,
)
from signature.words import (
    number_of_words_up_to_trunc,
    word_to_index,
    index_to_word,
    word_len,
)


# --------------------------------------------------------------------------- helpers
def make_ts(trunc, dim, seed=0):
    """A tensor sequence filled with reproducible pseudo-random coefficients."""
    n = int(number_of_words_up_to_trunc(trunc, dim))
    array = jax.random.normal(jax.random.PRNGKey(seed), (n,))
    return TensorSequence(array=array, trunc=trunc, dim=dim)


def basis_ts(word, trunc, dim):
    """The sequence e_word: 1 on ``word``, 0 elsewhere."""
    n = int(number_of_words_up_to_trunc(trunc, dim))
    array = np.zeros(n)
    array[int(word_to_index(word, dim))] = 1.0
    return TensorSequence(array=jnp.array(array), trunc=trunc, dim=dim)


def _word_str(w):
    return "" if int(w) == 0 else str(int(w))


def brute_shift(ts, word, side):
    """Reference shift via decimal-string word concatenation.

    side='right': result^u = ts^{u word}   (u must end with word)
    side='left' : result^u = ts^{word u}   (u must start with word)
    """
    n = len(ts)
    out = np.zeros(n)
    sw = _word_str(word)
    for i in range(n):
        su = _word_str(index_to_word(i, ts.dim))
        cat = (su + sw) if side == "right" else (sw + su)
        cword = 0 if cat == "" else int(cat)
        if int(word_len(cword)) <= ts.trunc:
            idx = int(word_to_index(cword, ts.dim))
            if idx < n:
                out[i] = ts.array[idx]
    return out


# words to probe, per dimension (empty word 0 included everywhere)
WORDS = {
    1: [0, 1, 11, 111],
    2: [0, 1, 2, 11, 12, 21, 22, 121, 212],
    3: [0, 1, 2, 3, 11, 23, 31, 312, 123],
    4: [0, 1, 4, 12, 34, 144, 421],
}
TRUNC = {1: 4, 2: 4, 3: 3, 4: 3}


# --------------------------------------------------------------------------- tests
class TestIndexationRoundTrip(unittest.TestCase):
    """word <-> index must be a bijection consistent with the length ordering."""

    def test_round_trip(self):
        for dim in (1, 2, 3, 4):
            n = int(number_of_words_up_to_trunc(TRUNC[dim], dim))
            for i in range(n):
                w = index_to_word(i, dim)
                self.assertEqual(int(word_to_index(w, dim)), i,
                                 msg=f"dim={dim} index={i} word={int(w)}")


class TestRightShift(unittest.TestCase):
    """proj (method and free function) implements the right shift for any dim."""

    def test_method_matches_reference(self):
        for dim in (1, 2, 3, 4):
            ts = make_ts(TRUNC[dim], dim, seed=dim)
            for w in WORDS[dim]:
                got = np.array(ts.proj(w).array)
                ref = brute_shift(ts, w, "right")
                np.testing.assert_allclose(got, ref, err_msg=f"dim={dim} word={w}")

    def test_free_function_matches_method(self):
        for dim in (1, 2, 3, 4):
            ts = make_ts(TRUNC[dim], dim, seed=dim + 10)
            for w in WORDS[dim]:
                np.testing.assert_allclose(
                    np.array(proj_fn(ts, w).array),
                    np.array(ts.proj(w).array),
                    err_msg=f"dim={dim} word={w}",
                )

    def test_empty_word_is_identity(self):
        for dim in (1, 2, 3, 4):
            ts = make_ts(TRUNC[dim], dim, seed=dim + 20)
            np.testing.assert_allclose(np.array(ts.proj(0).array), np.array(ts.array))


class TestLeftShift(unittest.TestCase):
    """left_proj implements the left shift for any dim."""

    def test_method_matches_reference(self):
        for dim in (1, 2, 3, 4):
            ts = make_ts(TRUNC[dim], dim, seed=dim + 30)
            for w in WORDS[dim]:
                got = np.array(ts.left_proj(w).array)
                ref = brute_shift(ts, w, "left")
                np.testing.assert_allclose(got, ref, err_msg=f"dim={dim} word={w}")

    def test_empty_word_is_identity(self):
        for dim in (1, 2, 3, 4):
            ts = make_ts(TRUNC[dim], dim, seed=dim + 40)
            np.testing.assert_allclose(np.array(ts.left_proj(0).array), np.array(ts.array))


class TestCompositionOfRightShifts(unittest.TestCase):
    """Right shifts compose: ``proj(v).proj(w)`` gives result^u = ts^{u w v},
    i.e. it equals the single right shift by the concatenated word ``w v``."""

    def test_proj_composition(self):
        cases = {2: [(1, 2), (2, 1), (12, 1)], 3: [(1, 2), (23, 3)]}
        for dim, pairs in cases.items():
            ts = make_ts(TRUNC[dim], dim, seed=dim + 50)
            for v, w in pairs:
                composed = np.array(ts.proj(v).proj(w).array)
                concat = int(_word_str(w) + _word_str(v))
                direct = np.array(ts.proj(concat).array)
                np.testing.assert_allclose(composed, direct,
                                           err_msg=f"dim={dim} v={v} w={w}")


class TestMatrixProjections(unittest.TestCase):
    """The matrix-based operators are consistent with the direct shifts."""

    def test_projection_matrix_shape(self):
        for dim in (1, 2, 3, 4):
            ts = make_ts(TRUNC[dim], dim, seed=dim + 60)
            n = int(number_of_words_up_to_trunc(ts.trunc, dim))
            self.assertEqual(get_projection_matrix(ts).shape, (n, n))

    def test_left_proj_on_seq_basis_matches_left_proj(self):
        for dim in (1, 2, 3, 4):
            ts = make_ts(TRUNC[dim], dim, seed=dim + 70)
            for w in WORDS[dim]:
                got = np.array(left_proj_on_seq(ts, basis_ts(w, ts.trunc, dim)).array)
                np.testing.assert_allclose(got, np.array(ts.left_proj(w).array),
                                           err_msg=f"dim={dim} word={w}")

    def test_right_proj_on_seq_basis_matches_proj(self):
        for dim in (1, 2, 3, 4):
            ts = make_ts(TRUNC[dim], dim, seed=dim + 80)
            for w in WORDS[dim]:
                got = np.array(right_proj_on_seq(ts, basis_ts(w, ts.trunc, dim)).array)
                np.testing.assert_allclose(got, np.array(ts.proj(w).array),
                                           err_msg=f"dim={dim} word={w}")

    def test_on_seq_accepts_raw_array(self):
        """Passing a raw jax array as ``proj_on`` matches passing a TensorSequence."""
        dim = 2
        ts = make_ts(TRUNC[dim], dim, seed=99)
        e = basis_ts(12, ts.trunc, dim)
        np.testing.assert_allclose(
            np.array(left_proj_on_seq(ts, e).array),
            np.array(left_proj_on_seq(ts, e.array).array),
        )


class TestJitCompatibility(unittest.TestCase):
    """Both shifts must be jit-compilable (the word is a static argument)."""

    def test_jit_proj_and_left_proj(self):
        for dim in (1, 2, 3):
            ts = make_ts(TRUNC[dim], dim, seed=dim + 90)
            for w in (0, WORDS[dim][1]):
                jit_proj = jax.jit(lambda t, _w=w: t.proj(_w))
                jit_left = jax.jit(lambda t, _w=w: t.left_proj(_w))
                np.testing.assert_allclose(np.array(jit_proj(ts).array),
                                           np.array(ts.proj(w).array))
                np.testing.assert_allclose(np.array(jit_left(ts).array),
                                           np.array(ts.left_proj(w).array))


if __name__ == "__main__":
    unittest.main(verbosity=2)
