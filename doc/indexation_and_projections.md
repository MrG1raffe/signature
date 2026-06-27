# Word indexation and projection operators

This note documents how words over a finite alphabet are encoded as integers and laid
out in the flat coefficient array of a `TensorSequence`, and how the projection (shift)
operators are computed from that layout. All formulas below hold for **any dimension
`dim >= 1`**.

The relevant code lives in [`signature/words.py`](../signature/words.py),
[`signature/tensor_sequence.py`](../signature/tensor_sequence.py) and
[`signature/projection.py`](../signature/projection.py).

## 1. Alphabet and words

The alphabet is `{1, 2, ..., dim}`. A *word* is a finite sequence of letters
`u = a_1 a_2 ... a_ℓ`, `ℓ = |u|` is its length, and the empty word is denoted `∅`.

A word is encoded as a single integer by writing its letters as **decimal digits**:

```
u = a_1 a_2 ... a_ℓ   <->   word = a_1 * 10^(ℓ-1) + ... + a_{ℓ-1} * 10 + a_ℓ
```

For example, with `dim = 2` the word `121` is the integer `121`; the empty word `∅`
is the integer `0`. Because letters live in `1..dim`, every nonzero digit is in
`1..dim` and there is no `0` digit — this is what lets `word_len` recover the length
as the number of decimal digits (`floor(log10(word)) + 1`, and `0` for `∅`).

> Note: this decimal encoding limits `dim` to `9` in practice (a letter must be a
> single decimal digit). The indexing math itself is base-`dim` and is exact for any
> `dim`.

## 2. From a word to its base-`dim` number

Within the set of words of a **fixed length** `ℓ`, a word is ordered by the integer it
represents in base `dim` over the digit set `{0, ..., dim-1}` (i.e. letter `a_k`
contributes `a_k - 1`):

```
b(u) = Σ_{k=1}^{ℓ} (a_k - 1) * dim^(ℓ-k),        0 <= b(u) < dim^ℓ
```

`a_1` is the most significant digit. This is computed by `word_to_base_dim_number`.
There are exactly `dim^ℓ` words of length `ℓ`, and `b` is a bijection between them and
`{0, ..., dim^ℓ - 1}`.

Key identity used by the projections — if `u = w · u''` (concatenation), with
`|w| = L`, then splitting the digits gives

```
b(u) = b(w) * dim^(|u|-L) + b(u'')          (w is a prefix of u)
b(u) = b(u') * dim^L      + b(w)            (w is a suffix of u, u = u' · w)
```

## 3. The flat index of a word

Coefficients are stored in a 1-D array. Words are ordered **first by length, then by
their base-`dim` number**:

```
∅ , (length-1 words) , (length-2 words) , ...
```

The number of words of length `< ℓ` (equivalently, the index of the first length-`ℓ`
word) is

```
S_ℓ = number_of_words_up_to_trunc(ℓ-1, dim) = (dim^ℓ - 1) / (dim - 1)
    = 1 + dim + dim^2 + ... + dim^(ℓ-1).
```

`number_of_words_up_to_trunc(trunc, dim)` returns the total number of words of length
`<= trunc`; it is implemented as `max((dim^(trunc+1) - 1)//(dim - 1), trunc + 1)` so
that the `dim == 1` case (where the geometric-series formula is `0/0`) collapses to
`trunc + 1`, which is correct since there is exactly one word of each length when the
alphabet has a single letter.

The flat index of a word and its inverse are therefore

```
index(u)        = S_{|u|} + b(u)                 # word_to_index
|u|             = index_to_word_len(index)       # invert S_ℓ <= index < S_{ℓ+1}
b(u)            = index - S_{|u|}
```

`index_to_word` reconstructs the digits of `u` from `index` by peeling off
`b(u)` in base `dim`.

## 4. Projection (shift) operators

A `TensorSequence` `p` is identified with the linear form `u ↦ p^u` on words. Two
adjoint shift operators remove a fixed word `w` (`|w| = L`) from one side:

| operator | definition | kept words `u` | image word `u'` |
|----------|------------|----------------|-----------------|
| `proj(w)`      (right shift) | `result^u = p^{u w}` | `u` **ends** with `w` | `u'` = `u` with suffix `w` removed |
| `left_proj(w)` (left shift)  | `result^u = p^{w u}` | `u` **starts** with `w` | `u'` = `u` with prefix `w` removed |

Both are computed in a fully vectorized, `jax.jit`-compatible way by acting on the
array of indices `i = 0, ..., N-1`. Write `ℓ = |u|` (`= index_to_word_len(i)`),
`b_u = i - S_ℓ`, `b_w = index(w) - S_L`, and `r = ℓ - L`.

**Right shift `proj(w)`** — `u = u' · w`, so `b_u = b(u') * dim^L + b_w`:

```
keep   :  ℓ >= L   and   b_u mod dim^L == b_w        # trailing L digits equal w
b(u')  :  b_u // dim^L
new idx :  S_r + b(u')          with  r = ℓ - L
```

**Left shift `left_proj(w)`** — `u = w · u'`, so `b_u = b_w * dim^r + b(u')`:

```
keep   :  ℓ >= L   and   b_u // dim^r == b_w         # leading L digits equal w
b(u')  :  b_u - b_w * dim^r
new idx :  S_r + b(u')          with  r = ℓ - L
```

Implementation details that keep this `jit`-safe for every `dim >= 1`:

- The exponent `r = ℓ - L` is clamped with `max(r, 0)` before being used as a power,
  because JAX raises on negative integer powers. Words with `ℓ < L` are discarded by
  the `ℓ >= L` mask anyway.
- All block offsets `S_k` go through `number_of_words_up_to_trunc`, so the `dim == 1`
  case needs no special-casing.
- Indices of discarded words are sent to the out-of-bounds slot `N + 1`; the scatter
  `array.at[new_idx].set(...)` then drops them, leaving zeros.

Right shifts compose in reverse order of application: applying `proj(v)` then `proj(w)`
gives `result^u = p^{u w v}`, i.e.

```
proj(w) ∘ proj(v) = proj(w · v)        # ts.proj(v).proj(w) == ts.proj(concat(w, v))
```

These operators are the building blocks for derivations / Riccati schemes; see the unit
tests in [`tests/test_projection.py`](../tests/test_projection.py).

## 5. Matrix form of the projections

`get_projection_matrix(ts)` builds the matrix `P` with rows `P[i] = ts.proj(v_i).array`,
where `v_i` is the word at index `i`. Acting with `P` reproduces the shifts against a
whole sequence `q` instead of a single word:

```
left_proj_on_seq(ts, q)  = P  @ q      # (P q)_i  = Σ_j ts^{w_j v_i} q^{w_j}
right_proj_on_seq(ts, q) = Pᵀ @ q      # (Pᵀq)_i  = Σ_j ts^{w_i v_j} q^{w_j}
```

In particular, when `q = e_w` is the basis sequence of a single word `w`:

```
left_proj_on_seq(ts, e_w)  == ts.left_proj(w)
right_proj_on_seq(ts, e_w) == ts.proj(w)
```

These identities are exactly the consistency checks exercised by the unit tests.
