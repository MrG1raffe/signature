import numpy as np
from numba import jit, int64


@jit(nopython=True)
def factorial(x):
    n = 1
    for i in range(2, x+1):
        n *= i
    return n


@jit(nopython=True)
def combinations(pool, r):
    n = len(pool)
    if r > n or r <= 0:
        return np.empty((0, r), dtype=np.int64)  # Handle edge cases

    # Calculate the number of combinations
    l = 1
    for i in range(r):
        l *= (n - i)
        l //= (i + 1)

    # Preallocate result array
    res = np.empty((l, r), dtype=np.int64)
    indices = np.arange(r)
    it = 0

    while True:
        # Write current combination to result
        for i in range(r):
            res[it, i] = pool[indices[i]]
        it += 1

        # Generate the next combination
        i = r - 1
        while i >= 0 and indices[i] == i + n - r:
            i -= 1

        if i < 0:
            break

        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1

    return res