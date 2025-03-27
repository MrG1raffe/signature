from signature.old_versions.tensor_sequence import TensorSequence
from signature.old_versions.tensor_algebra import TensorAlgebra


def get_signature_of_linear_form(ts: TensorSequence, trunc_moments: int, ta: TensorAlgebra):
    """
    Computes the coefficients of the signature Sig(t, L)_t of the linear form on signature L_t = <ts, Sig(t, X)_t> as
    the tensor sequences l(v) satisfying <v, Sig(t, L)> = <l(v), Sig(X)_t> and returns l as a dictionary.
    """
    signal_sig_coefs_exact = dict()
    n_moments = ta.alphabet.number_of_elements(trunc_moments)

    for idx in range(n_moments):
        word = ta.alphabet.index_to_word(idx)
        if not idx:
            signal_sig_coefs_exact[""] = TensorSequence.unit(ta.alphabet, trunc=ts.trunc)
        else:
            if word.endswith("1"):
                signal_sig_coefs_exact[word] = signal_sig_coefs_exact[word[:-1]].tensor_prod_word("1")
            elif word.endswith("2"):
                signal_sig_coefs_exact[word] = ta.shuop.shuffle_prod(signal_sig_coefs_exact[word[:-1]], ts.proj("1")).tensor_prod_word("1") + \
                                               ta.shuop.shuffle_prod(signal_sig_coefs_exact[word[:-1]], ts.proj("2")).tensor_prod_word("2")
    return signal_sig_coefs_exact
