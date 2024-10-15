import numpy as np
from scipy.stats import norm
from typing import Union
from numpy.typing import NDArray
from numpy import float_
import matplotlib.pyplot as plt


class MonteCarlo:
    batch: NDArray[float_]
    accuracy: Union[float, NDArray[float_]]
    confidence_level: Union[float, NDArray[float_]]
    mean: Union[float, NDArray[float_]]
    var: Union[float, NDArray[float_]]
    batch_size: int
    axis: int

    def __init__(
        self,
        batch: NDArray[float_],
        confidence_level: Union[float, NDArray[float_]] = None,
        accuracy: Union[float, NDArray[float_]] = None,
        axis: int = 0,
        control_variate: NDArray[float_] = None
    ) -> None:
        """
        Calculates the Monte Carlo statistics for the given batch and the corresponding confidence intervals.

        Args:
            batch: array to compute the mean.
            confidence_level: confidence level for the intervals.
            accuracy: an error estimated via the CLT.
            axis: axis along which the mean to be computed
            control_variate: control variate to be added to batch with the appropriate weight which is calculated
                in such a manner that the estimator is unbiased. Assumed to have zero mean.
        """
        self.axis = axis
        self.batch_size = batch.shape[axis]
        self.n_batch = 0
        self.mean = 0
        self.squares_mean = 0
        if confidence_level is not None and accuracy is not None:
            raise ValueError('Either accuracy or confidence level should be specified, not both.')
        if confidence_level is not None:
            self.__ci_flag = 'conf'
            self.confidence_level = confidence_level
        elif accuracy is not None:
            self.__ci_flag = 'acc'
            self.accuracy = accuracy
        else:
            raise ValueError('Either accuracy or confidence level should be specified.')
        self.add_batch(batch=batch, control_variate=control_variate)

    def __update_mean(
        self,
        current_mean: Union[float, NDArray[float_]],
        batch: NDArray[float_]
    ) -> Union[float, NDArray[float_]]:
        """
        Iteratively updates the mean 'current_mean' calculated over the previous batches given the new batch.

        Args:
            current_mean: mean of all the previous self.n_batches.
            batch: new batch to update the mean.

        Returns:
            updated mean.
        """
        return self.n_batch / (self.n_batch + 1) * current_mean + \
            1 / (self.n_batch + 1) * np.mean(batch, axis=self.axis)

    def add_batch(
            self,
            batch: NDArray[float_],
            control_variate: NDArray[float_] = None
    ) -> None:
        """
        Replace the existing batch, updates the mean, variance and confidence interval.

        Args:
            batch: new batch of the same shape.
            control_variate: new control variate of the same shape.
        """
        if self.batch_size != batch.shape[self.axis]:
            raise ValueError("Batches should be of the same shape.")
        if control_variate is not None:
            weights = self._unbiased_weights(batch, control_variate)
            self.batch = batch - weights * control_variate
        else:
            self.batch = batch
        # updating mean and variance
        self.mean = self.__update_mean(self.mean, batch)
        self.squares_mean = self.__update_mean(self.squares_mean, batch**2)
        self.var = self.squares_mean - self.mean**2
        self.n_batch += 1
        if self.__ci_flag == 'conf':
            quantile = norm.ppf((1 + self.confidence_level) / 2)
            self.accuracy = quantile * np.sqrt(self.var / self.batch_size / self.n_batch)
        elif self.__ci_flag == 'acc':
            quantile = self.accuracy * np.sqrt(self.batch_size * self.n_batch / self.var)
            self.confidence_level = 2 * norm.cdf(quantile) - 1

    def convergence_diagram(
        self,
        step: int = 1,
        ax: plt.axes = None,
        plot_intervals: bool = False,
        log: bool = False,
        color: str = 'b',
        label: str = 'MC estimator'
    ) -> None:
        """
        Plots a convergence diagram for the given one-dimensional batch.

        Args:
            step: a step used to iterate over the batch.
            ax: axis to plot the diagram.
            plot_intervals: whether to plot confidence intervals.
            log: whether to use log-axis for x.
            color: line color.
            label: matplotlib legend label.
        """
        if ax is None:
            _, ax = plt.subplots()

        subbatch = np.cumsum(self.batch)[step::step]
        ns = self.batch_size * (self.n_batch - 1) + np.arange(1, len(subbatch) + 1) * step
        means = subbatch / ns

        x = np.log(ns) if log else ns
        xlabel = 'log(n)' if log else 'n'

        ax.plot(x, means, color, label=label)
        ax.grid('on')

        if plot_intervals:
            ax.plot(x, means - self.accuracy * np.sqrt(self.batch_size * self.n_batch / ns), color + '--',
                    label=f'CI of level {self.confidence_level}', lw=1)
            ax.plot(x, means + self.accuracy * np.sqrt(self.batch_size * self.n_batch / ns), color + '--', lw=1)
            ax.fill_between(x, means - self.accuracy * np.sqrt(self.batch_size / ns),
                            means + self.accuracy * np.sqrt(self.batch_size / ns),
                            color=color, alpha=0.05)
        ax.legend()
        ax.set_xlabel(xlabel)

    def results(
        self,
        decimals=5,
    ) -> str:
        """
        Represents results as a string.

        Args:
            decimals: Number of decimal places to round to.

        Returns:
            String containing mean and accuracy.
        """
        return str(np.round(self.mean, decimals=decimals)) + " ± " + str(np.round(self.accuracy, decimals=decimals))

    @staticmethod
    def _adapted_weights(
        batch: NDArray[float_],
        control_variate: NDArray[float_]
    ) -> NDArray[float_]:
        """
        Calculates the control variate weights using adaptive method: for the k-th observation the weight
        lambda_k = cov(batch[0:k-1], control_variate[0:k-1]) / var(control_variate[0:k-1]).
        A new batch shall be calculated as 'batch - weights * control_variate'.

        Args:
            batch: array to compute the mean.
            control_variate: array of control variates. Assumed to have zero mean.

        Returns:
            An array of weights corresponding to the given control variates.
        """
        weights = np.zeros_like(control_variate)
        v = np.cumsum(control_variate[:-1]**2)
        np.divide(np.cumsum(batch[:-1] * control_variate[:-1]), v, out=weights[1:], where=v > 0)
        return weights

    def _unbiased_weights(
        self,
        batch: NDArray[float_],
        control_variate: NDArray[float_]
    ) -> NDArray[float_]:
        """
        Calculates the control variate weights using unbiased method: for the k-th observation the weight
        lambda_{-k} = cov(batch_{-k}, control_variate_{-k}) / var(control_variate_{-k}). Here array_{-k}
        means array with all the elements except for k-th.
        A new batch shall be calculated as 'batch - weights * control_variate'.

        Args:
            batch: array to compute the mean.
            control_variate: array of control variates. Assumed to have zero mean.

        Returns:
            An array of weights corresponding to the given control variates.
        """
        weights = np.zeros_like(control_variate)
        v = np.sum(control_variate**2, axis=self.axis, keepdims=True) - control_variate**2
        np.divide(np.sum(batch * control_variate, axis=self.axis, keepdims=True) - batch * control_variate,
                  v, out=weights, where=v > 0)
        return weights
