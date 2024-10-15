import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from numpy import float_
from typing import List, Tuple

from products.vanilla_option import VanillaOption
from utility.piece_wise_constant import PieceWiseConstantFunction


class FunctionH(PieceWiseConstantFunction):
    option_indices: List[int]
    observation_date: np.datetime64

    def __init__(
        self,
        vanilla_options_with_indices: List[Tuple[int, VanillaOption]],
        observation_date: np.datetime64,
        function_values: NDArray[float_] = None,
        interpolation_method: str = None
    ):
        """
        Creates the piece-wise constant function h with the changing points formed by the maturities of the options
        on CAL, SUM and WIN contracts.

        :param vanilla_options_with_indices: list of vanilla options with its indices in hjm.vanilla_options
            used to construct the changing points.
        :param observation_date: observation date taken as t = 0.
        :param function_values: values of the piece-wise function of size len(changing_points) - 1.
        :param interpolation_method: interpolation method used to smooth the piece-wise constant function.
        """
        maturities = [0]
        option_indices = [np.nan]
        for i, option in vanilla_options_with_indices:
            maturities += list(option.T)
            option_indices.append(i)

        maturities, option_indices = zip(*sorted(zip(maturities, option_indices)))
        changing_points = np.array(maturities)
        if function_values is None:
            function_values = np.ones(len(changing_points) - 1)
        else:
            if len(function_values) != len(changing_points) - 1:
                raise ValueError("Inconsistent length of changing_points and function_values")
            function_values = function_values
        super().__init__(changing_points=changing_points,
                         function_values=function_values,
                         interpolation_method=interpolation_method)
        self.option_indices = option_indices
        self.observation_date = observation_date

    def plot(
        self,
        ax: plt.axes = None
    ) -> None:
        """
        Plots the function on a given axis.

        :param ax: matplotlib axis.
        """
        xticks = self.changing_points
        xlabels = (self.observation_date + (xticks * 365).astype(int)).astype(str)
        title = r"Function $h(t)$"
        self._plot_function(ax=ax, xlabels=xlabels, xticks=xticks, title=title)
