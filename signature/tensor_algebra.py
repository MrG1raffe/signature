import jax
jax.config.update("jax_enable_x64", True)

from .words import (word_len, number_of_words_up_to_trunc, index_to_word_len, index_to_word, index_to_word_vect,
                    index_to_lam_sum, word_to_base_dim_number, word_to_base_dim_number_vect, word_to_index,
                    word_to_index_vect, index_to_lam_sum_vect)
from .factory import zero, zero_like, unit, unit_like, from_word, from_dict, from_array
from .shuffle_table import get_shuffle_table
from .shuffle_product import shuffle_prod, shuffle_prod_2d, shuffle_pow, shuffle_exp
from .tensor_product import tensor_prod_word, tensor_prod, tensor_pow, tensor_exp, resolvent
from .algebra_basis import AlgebraBasis
from .path_signature import path_to_signature, path_to_fm_signature
from .expected_signature import expected_bm_signature, expected_bm_stationary_signature
from .operators import G, G_inv, G_resolvent, D, semi_integrated_scheme
from .tensor_sequence import TensorSequence
from .projection import proj
from .visualisation import plot_coefficients
