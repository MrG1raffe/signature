import jax.numpy as jnp
import numpy as np

from ..shuffle import shuffle_product
from .words import number_of_words_up_to_trunc, index_to_word_len, index_to_word, word_to_index_vect


def get_shuffle_table(table_trunc: int, dim: int = 2):
    n_words = number_of_words_up_to_trunc(table_trunc, dim)

    arrays_to_stack = []
    table_len = 0

    ij_array = np.array(np.meshgrid(np.arange(n_words), np.arange(n_words))).T.reshape(-1, 2)
    ij_array = ij_array[index_to_word_len(jnp.array(ij_array), dim=dim).sum(axis=1) <= table_trunc]

    for i, j in ij_array:
        words_res, counts = shuffle_product(int(index_to_word(i)), int(index_to_word(j)))
        indices_res = word_to_index_vect(jnp.array(words_res, dtype=jnp.int64), dim)

        arr_to_add = np.zeros((len(counts), 4), dtype=int)
        arr_to_add[:, 0] = i
        arr_to_add[:, 1] = j
        arr_to_add[:, 2] = indices_res
        arr_to_add[:, 3] = counts

        table_len += arr_to_add.shape[0]
        arrays_to_stack.append(arr_to_add)

    shuffle_table = np.vstack(arrays_to_stack)
    return shuffle_table.T
