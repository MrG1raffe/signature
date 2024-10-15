import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from numpy import float_
from typing import Union, List
from scipy.interpolate import PchipInterpolator, CubicSpline

from utility.colors import ENGIE_BLUE


class PieceWiseConstantFunction:
    changing_points: NDArray[float_]
    function_values: NDArray[float_]
    interpolation_method: str = None

    def __init__(
        self,
        changing_points: NDArray[float_],
        function_values: NDArray[float_],
        interpolation_method: str = None,
        extrapolation_right: float = None,
        extrapolation_left: float = None
    ):
        """
        Creates an object for piece-wise constant function which takes values function_values[i] on the interval
        [changing_points[i], changing_points[i + 1]).

        :param changing_points: array of changing points.
        :param function_values: array of function values.
        :param interpolation_method: interpolation method used to smooth the function when it is called.
        :param extrapolation_right: value used to extrapolate the function for the arguments greater than
            changing_points[-1]. By default, function_values[-1].
        :param extrapolation_left:value used to extrapolate the function for the arguments smaller than
            changing_points[0]. By default, function_values[0].
        """
        self.changing_points = changing_points
        self.function_values = function_values
        self.interpolation_method = interpolation_method
        self.extrapolation_right = extrapolation_right
        self.extrapolation_left = extrapolation_left

    def __call__(self, t: Union[float, NDArray[float_]]) -> Union[float, NDArray[float_]]:
        """
        Evaluates the function in `t` using the interpolation method defined in the object attribute.

        :param t: number or array for the function to be estimated in.
        :return: a number or array corresponding to the values of the function on `t`.
        """
        t = np.array(t)

        # no function_values means that the function is equal to 1
        if len(self.function_values) == 0:
            return np.ones_like(t)

        y_right = self.extrapolation_right if self.extrapolation_right is not None else self.function_values[-1]
        y_left = self.extrapolation_left if self.extrapolation_left is not None else self.function_values[0]
        if self.interpolation_method is None:
            t = np.reshape(t, (-1, 1))
            return (((self.changing_points[:-1][None, :] <= t) &
                    (self.changing_points[1:][None, :] > t)) @ self.function_values +
                    (self.changing_points[-1] <= t.squeeze()) * y_right +
                    (self.changing_points[0] > t.squeeze()) * y_left).astype(float)
        elif self.interpolation_method == "linear":
            return np.interp(
                x=t,
                xp=np.concatenate([np.array([self.changing_points[0]]),
                                   0.5 * (self.changing_points[1:] + self.changing_points[:-1]),
                                   np.array([self.changing_points[-1]])]),
                fp=np.concatenate([np.array([y_left]),
                                   self.function_values,
                                   np.array([y_right])]),
            )
        elif self.interpolation_method == "cubic":
            f = CubicSpline(x=np.concatenate([np.array([self.changing_points[0]]),
                                              0.5 * (self.changing_points[1:] + self.changing_points[:-1]),
                                              np.array([self.changing_points[-1]])]),
                            y=np.concatenate([np.array([y_left]),
                                              self.function_values,
                                              np.array([y_right])]),
                            bc_type='clamped')
            return f(t)
        elif self.interpolation_method == "pchip":
            f = PchipInterpolator(
                x=np.concatenate([np.array([self.changing_points[0]]),
                                  0.5 * (self.changing_points[1:] + self.changing_points[:-1]),
                                  np.array([self.changing_points[-1]])]),
                y=np.concatenate([np.array([y_left]),
                                  self.function_values,
                                  np.array([y_right])])
            )
            return f(t)

    def _plot_function(
        self,
        ax: plt.axes = None,
        title: str = None,
        xticks: Union[List, NDArray[float_]] = None,
        xlabels: Union[List, NDArray[float_]] = None
    ) -> None:
        """
        Plot the (possibly interpolated) function on the range defined by self.changing_points.

        :param ax: matplotlib axis to plot on.
        :param title: plot title.
        :param xticks: x ticks.
        :param xlabels: labels corresponding to x ticks.
        """
        if ax is None:
            _, ax = plt.subplots()
        if self.interpolation_method is None:
            ax.plot(
                np.repeat(self.changing_points, 2)[1:-1],
                np.repeat(self.function_values, 2),
                color=ENGIE_BLUE,
                lw=2
            )
        else:
            N_points = 1000
            x_grid = np.linspace(0, np.max(self.changing_points), N_points)
            ax.plot(
                x_grid,
                self.__call__(x_grid),
                color=ENGIE_BLUE,
                lw=2
            )
        ax.grid("on")
        if xticks is not None and xlabels is not None:
            ax.set_xticks(xticks, labels=xlabels, rotation=-90)
        if title is not None:
            ax.set_title(title)
