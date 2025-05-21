import matplotlib.pyplot as plt
import numpy as np

from .tensor_sequence import TensorSequence
from .words import number_of_words_up_to_trunc, index_to_word_vect, index_to_word

def plot_coefficients(ts: TensorSequence, trunc: int = None, nonzero: bool = False,
                      ax: plt.axis = None, **kwargs) -> None:
    """
    Plots the coefficients of a given tensor sequence.

    :param ts: tensor sequence to plot.
    :param trunc: truncation order, the coefficients of order <= trunc will be plotted. By default, equals to ts.trunc.
    :param nonzero: whether to plot only non-zero coefficients.
    :param ax: plt axis to plot on.
    """
    if trunc is None:
        trunc = ts.trunc

    n_coefficients = number_of_words_up_to_trunc(trunc=trunc, dim=ts.dim)

    indices = np.arange(n_coefficients)
    coefficients = np.zeros(n_coefficients)
    coefficients[:min(n_coefficients, len(ts))] = ts.array[:min(n_coefficients, len(ts))].squeeze().real

    if ax is None:
        fig, ax = plt.subplots()

    if nonzero:
        plotting_idx = np.where(coefficients != 0)[0]
    else:
        plotting_idx = np.arange(coefficients.size)

    ax.plot(coefficients[plotting_idx], "o", **kwargs)
    ax.grid("on")

    ax.set_xticks(ticks=np.arange(plotting_idx.size),
                  labels=index_to_word_vect(indices, ts.dim)[plotting_idx],
                  rotation=-90)
