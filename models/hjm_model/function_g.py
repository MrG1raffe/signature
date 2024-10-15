import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from numpy import float_, int_
from typing import List

from models.hjm_model.contracts import ForwardContract
from utility.piece_wise_constant import PieceWiseConstantFunction

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


class FunctionG(PieceWiseConstantFunction):
    function_shape: NDArray[float_]
    contract_indices: dict
    observation_date: np.datetime64

    def __init__(
        self,
        contracts: List[ForwardContract],
        interpolation_method: str = None
    ):
        """
        Creates a piece-wise constant function g equal to 1 using the time delivery start / end dates of
        the contracts as changing points.

        :param contracts: list of contracts used to construct the grid of changing points.
        :param interpolation_method: interpolation method that will be used to smooth the peace-wise constant function.
        """
        observation_date = contracts[0].observation_date
        max_date = max([contract.delivery_end_date for contract in contracts]) + np.timedelta64(30)
        max_changing_point = (max_date - observation_date) / np.timedelta64(365, 'D')
        changing_points = [0, max_changing_point]
        changing_points_dates = [observation_date, max_date]
        for contract in contracts:
            if contract.observation_date != observation_date:
                raise ValueError("All contracts should have the same observation date.")
            if contract.delivery_start_date not in changing_points_dates:
                changing_points_dates.append(contract.delivery_start_date)
                changing_points.append(contract.time_to_delivery_start)
            if contract.delivery_end_date not in changing_points_dates:
                changing_points_dates.append(contract.delivery_end_date)
                changing_points.append(contract.time_to_delivery_end)

        changing_points, changing_points_dates = zip(*sorted(zip(changing_points, changing_points_dates)))
        changing_points = np.array(changing_points)

        # i-th value corresponds to t between i-th and (i+1)-th changing points.
        function_values = np.ones_like(changing_points[:-1])
        super().__init__(changing_points=changing_points,
                         function_values=function_values,
                         interpolation_method=interpolation_method,
                         extrapolation_left=1,
                         extrapolation_right=1)

        self.contract_indices = dict()
        for contract in contracts:
            idx_start = changing_points_dates.index(contract.delivery_start_date)
            idx_end = changing_points_dates.index(contract.delivery_end_date)
            self.contract_indices[contract.name] = np.arange(idx_start, idx_end + 1)

        self.observation_date = observation_date

    def __getitem__(
        self,
        key: str
    ) -> NDArray[float_]:
        """
        Returns the indices of changing points corresponding to the contract name `key`.

        :param key: contract name.
        :return: list of indices corresponding to the intervals forming the delivery period of the contract "key"
            which was used to construct the changing points grid.
        """
        if key in self.contract_indices.keys():
            return self.function_values[self.contract_indices[key][:-1]]
        else:
            return self.function_values[-2:-1]

    def __setitem__(
        self,
        key: str,
        value: float
    ) -> None:
        """
        Set the `value` on the interval corresponding to delivery period of the contract `key`.

        :param key: contract name.
        :param value: value of function g to be set.
        :return:
        """
        self.function_values[self.contract_indices[key][:-1]] = value

    def get_contract_changing_points(
        self,
        contract_name: str
    ) -> NDArray[float_]:
        """
        Returns the changing points corresponding to the contract "contract_name".

        :param contract_name: contract name.
        :return: the changing points T1, ..., Tn, where T1 coincides with time to delivery start and Tn coincides
            with time ti delivery end for the given contract.
        """
        return self.changing_points[self.contract_indices[contract_name]]

    def get_contract_indices(
        self,
        contract_name: str
    ) -> NDArray[int_]:
        """
        Calculates the indices of the intervals forming the repartition of the contract's delivery period.

        :param contract_name: name of the contract.
        :return: list of indices of the intervals forming the delivery period of the contract.
        """
        return self.contract_indices[contract_name][:-1]

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
        title = r"Function $g(T)$"
        self._plot_function(ax=ax, xlabels=xlabels, xticks=xticks, title=title)

    def reset(
        self
    ) -> None:
        """
        Reset the function values equal to 1.
        """
        self.function_values = np.ones_like(self.changing_points[:-1])
