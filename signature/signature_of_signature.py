import jax

from signature.path_signature import TensorSequence
from signature.words import number_of_words_up_to_trunc, index_to_word
from signature.factory import unit
from signature.tensor_product import tensor_prod_word
from signature.shuffle_product import shuffle_prod


def get_signature_of_linear_form(ts: TensorSequence, trunc_moments: int, shuffle_table: jax.Array):
    """
    Computes the coefficients of the signature Sig(t, L)_t of the linear form on signature L_t = <ts, Sig(t, X)_t> as
    the tensor sequences l(v) satisfying <v, Sig(t, L)> = <l(v), Sig(X)_t> and returns l as a dictionary.
    """
    signal_sig_coefs_exact = dict()
    n_moments = number_of_words_up_to_trunc(trunc_moments, dim=ts.dim)

    for idx in range(n_moments):
        word = index_to_word(idx, dim=ts.dim)
        if not idx:
            signal_sig_coefs_exact[0] = unit(trunc=ts.trunc, dim=ts.dim)
        else:
            if word % 10 == 1:
                signal_sig_coefs_exact[word] = tensor_prod_word(signal_sig_coefs_exact[word[:-1]], 1)
            elif word % 10 == 2:
                signal_sig_coefs_exact[word] = shuffle_prod(signal_sig_coefs_exact[word // 10], ts.proj(1), shuffle_table=shuffle_table).tensor_prod_word(1) + \
                                               shuffle_prod(signal_sig_coefs_exact[word // 10], ts.proj(2), shuffle_table=shuffle_table).tensor_prod_word(2)
    return signal_sig_coefs_exact
