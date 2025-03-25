import jax
jax.config.update("jax_enable_x64", True)

from .words import (word_len, number_of_words_up_to_trunc, index_to_word_len, index_to_word, index_to_word_vect,
                    word_to_base_dim_number, word_to_base_dim_number_vect, word_to_index, word_to_index_vect)
