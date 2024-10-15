import numpy as np
from typing import List, Union, Tuple, Callable
from dataclasses import dataclass
from numpy.typing import NDArray
from numpy import float_

from models.model_params import PricingParams


@dataclass
class CalibrationStrategyHJM:
    """
    HJM calibration strategy describes the calibration scenario.

    params_to_calibrate: names of the parameters to be calibrated at the smile calibration step.
    optimiser: solver used to minimize the shape calibration loss.
    pricing_method: a pricing method used to calculate the option prices.
    pricing_params: parameters of the pricing method used to calculate the option prices.
    smile_weights: array of smile weights in the calibration loss.
    bid_ask_penalization: coefficient in front of regularization term penalizing the smiles being out of the given
        bid-ask spread.
    vector_to_params: a callable object that transforms a finite-dimensional vector of variables being calibrated
        into a dictionary with stochastic volatility model parameters. If not specified, will be calculated
        automatically from `params_to_calibrate`.
    bounds: list of tuples, defining the optimization bounds for each of the variables being calibrated.
        If not specified, will be calculated automatically from `params_to_calibrate`.
    """
    params_to_calibrate: Tuple[str] = ("c", "rho_spot_vol")
    optimiser: str = "Powell"
    pricing_method: str = "lewis"
    pricing_params: PricingParams = PricingParams()
    smile_weights: NDArray[float_] = None
    bid_ask_penalization: float = 0
    vector_to_params: Callable = None
    bounds: List[Tuple] = None


@dataclass
class CalibrationDataHJM:
    """
    Implied volatility market data needed for the implied calibration.

    option_names: list of names corresponding to the vanilla_options attribute of HJM.
    atm_idx: array of size len(vanilla_options), where atm_idx[i] is the index of the ATM volatility for the smile i.
    bid_ask_spread: an array of IV bid-ask spreads for the market IV's corresponding to `vanilla_options`.
    prices_market: array of market option prices corresponding to `vanilla_options`.
    implied_vols_market: array of market implied volatilities corresponding to `vanilla_options`.
    prices_market_bid: option prices corresponding to (implied_vols_market - 0.5 * bid_ask_spread)
    prices_market_ask option prices corresponding to (implied_vols_market + 0.5 * bid_ask_spread)
    prices_log_contract: array prices of the log-contracts with payoff log(F_T) for each IV slice.
    """
    option_names: List[str]
    atm_idx: List[int]
    bid_ask_spread: Union[float, List[float]] = 0.04
    prices_market: Union[NDArray[float_], List[NDArray[float_]]] = None
    implied_vols_market: Union[NDArray[float_], List[NDArray[float_]]] = None
    prices_market_bid: Union[NDArray[float_], List[NDArray[float_]]] = None
    prices_market_ask: Union[NDArray[float_], List[NDArray[float_]]] = None
    prices_log_contract: List[NDArray[float_]] = None

    def __post_init__(self):
        if self.implied_vols_market is None and self.prices_market is None:
            raise ValueError('You must specify either the prices of the iv market.')

   