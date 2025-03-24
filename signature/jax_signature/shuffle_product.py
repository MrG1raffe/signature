import jax


@jax.jit
def shuffle_prod_jax(
    ts1: jax.Array,
    ts2: jax.Array,
    shuffle_table: jax.Array,
):
    index_left, index_right, index_result, count = shuffle_table

    source = count * ts1[index_left] * ts2[index_right]
    linear_result = ts1 * 0  # keeps the same size as ts1.
    linear_result = linear_result.at[index_result].add(source)

    return linear_result


shuffle_prod_jax_vect = jax.jit(jax.vmap(shuffle_prod_jax, in_axes=(1, 1, None), out_axes=1))
