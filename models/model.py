from abc import abstractmethod, ABC
from dataclasses import asdict, dataclass
import numpy as np
from numpy.typing import NDArray
from numpy import float_
from typing import Union, Tuple

@dataclass
class Model(ABC):
    """
    A generic class for the pricing models.
    """
    @abstractmethod
    def get_price_trajectory(
        self,
        t_grid: NDArray[float_],
        size: int,
        F0: Union[float, NDArray[float_]],
        rng: np.random.Generator = None,
        *args,
        **kwargs
    ) -> Union[NDArray[float_], Tuple[NDArray[float_], ...]]:
        """
        Generic method to simulate the underlying price trajectories on the given time grid.

        :param t_grid: time grid.
        :param size: number of trajectories to simulate.
        :param F0: initial value of the underlying price.
        :param rng: random number generator to simulate the trajectories with.
        :return: an array `F_traj` of shape (size, len(t_grid)) of simulated price trajectories
        """
        raise NotImplementedError()

    def __post_init__(self) -> None:
        """
        Checks the correctness of the given parameters, does necessary pre-calculations.

        :return: None
        """
        pass

    def update_params(
        self,
        new_params: dict
    ) -> None:
        """
        Updates the attributes of the class instance and runs post_init after.

        :param new_params: a dictionary with new parameters values. Its keys are attributes names, the values are
            new attribute values.
        :return: None
        """
        for param_name in new_params:
            if param_name not in self.__dict__:
                raise KeyError(f"Wrong parameter name {param_name} was given to the model.")
            self.__dict__[param_name] = new_params[param_name]
        self.__post_init__()
