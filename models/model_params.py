import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from numpy import float_

from utility.utility import to_numpy


@dataclass
class HistoricalParams:
    """
    Historical HJM calibration parameters. Factors are supposed to be of the form
    σ_i * exp{-(T - t) / 𝜏_i} dW_t^i, i = 1, ... N.

    sigmas: array of σ_i > 0.
    taus: array of 𝜏_i. May take the value of np.inf. In this case, exp{-(T - t) / 𝜏_i} = 1.
    corr_mat: a correlation matrix of the Brownian motion W_t = (W_t^1, ..., W_t^N).
    """
    sigmas: NDArray[float_]
    taus: NDArray[float_]
    corr_mat: NDArray[float_]

    def __post_init__(self):
        self.sigmas = to_numpy(self.sigmas)
        self.taus = to_numpy(self.taus)
        self.corr_mat = to_numpy(self.corr_mat)

        if self.sigmas.size != self.taus.size:
            raise ValueError("`taus` and `sigmas` should be one-dimensional arrays of the same size.")
        if self.corr_mat.shape != (self.taus.size, self.taus.size):
            raise ValueError("Inconsistent dimensions of `corr_mat`, `taus`, and `sigmas`.")
        if np.min(np.linalg.eigvals(self.corr_mat)) < 0:
            raise ValueError("Correlation matrix should be positively semi-definite. "
                             f"Minimal eigenvalue is {np.min(np.linalg.eigvals(self.corr_mat))}.")


@dataclass
class ModelParams:
    """
    A generic class for stochastic volatility model parameters.
    """


@dataclass
class LiftedHestonParams(ModelParams):
    """
    Lifted heston parameters corresponding to the stochastic volatility. See the definition of
    the class `LiftedHeston` for details about the parameters.
    """

    rhos: NDArray[float_]
    x: NDArray[float_]
    c: NDArray[float_]
    V0: float = 1
    theta: float = 1
    lam: float = 0
    nu: float = 1
    model_type: str = "log-normal"
    normalize_variance: bool = False


@dataclass
class SteinSteinParams(ModelParams):
    """
    Stein Stein parameters corresponding to the stochastic volatility. See the definition of
    the class `SteinStein` for details about the parameters.
    """

    theta: float
    kappa: float
    nu: float
    rhos: NDArray[float_]
    X0: float
    model_type: str
    normalize_variance: bool
    vol_normalization: float = 1
    

@dataclass
class PricingParams:
    """
    A generic class for product pricing parameters.
    """


@dataclass
class CosParams(PricingParams):
    """
    Parameters for the COS-method.

    N_trunc: number of terms in the Cosine series to be calculated.
    cf_timestep: a timestep to be used in numerical scheme in the characteristic function.
    scheme: numerical scheme for the Riccati equation. Either "exp" or "semi-implicit".
    """
    N_trunc: int = 50
    cf_timestep: float = 0.003
    max_grid_size: int = 10000
    scheme: str = "exp"


@dataclass
class LewisParams(PricingParams):
    """
    Parameters for the COS-method.

    N_trunc: number of terms in the Cosine series to be calculated.
    cf_timestep: a timestep to be used in numerical scheme in the characteristic function.
    scheme: numerical scheme for the Riccati equation. Either "exp" or "semi-implicit".
    """
    N_points: int = 20
    cf_timestep: float = 0.003
    max_grid_size: int = 10000
    scheme: str = "exp"
    control_variate_sigma: float = 0.4


@dataclass
class MCParams(PricingParams):
    """
    Parameters for the Monte Carlo pricing

    size: number of trajectories to simulate.
    return_accuracy: whether to return the confidence interval for the prices / vols.
    confidence_level: Monte Carlo simulation confidence level.
    timestep: time step for the Euler's discretization.
    batch_size: batch_size to be used in Monte Carlo simulation.
    rng: random number generator to simulate the trajectories with.
    """
    size: int = 10 ** 5
    return_accuracy: bool = False
    confidence_level: float = 0.95
    timestep: float = 1 / 250
    batch_size: int = 10 ** 5
    rng: np.random.Generator = None
    scheme: str = "exp"


@dataclass
class SemiAnalyticParams(PricingParams):
    """
    Parameters for semianalytic Bergomi-Guyon expansions.

    dt: integration time step.
    """
    dt: float


