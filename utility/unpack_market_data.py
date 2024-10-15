import numpy as np
from typing import Tuple

from products.vanilla_option import VanillaOption
from utility.utility import from_delta_call_to_strike
from models.hjm_model.contracts import ForwardContract
from models.hjm_model.calibration_params import CalibrationDataHJM


def unpack_iv_market_data(
    iv_data: dict,
    observation_date: str,
    source: str = "ICAP",
    market: str = "PW",
    bid_ask_default: float = 0.04
) -> Tuple:
    """
    A utility function used to transform the IV dictionary into the list of contracts, list of vanilla options
    and the calibration data.

    :param iv_data: a dictionary containing the implied volatility data with keys
        SOURCE -> MARKET -> OBSERVATION_DATE -> ["1-deltas", "F0", "IV", "expiry"].
    :param observation_date: observation date in the format "YYYY-MM-DD".
    :param source: IV data source.
    :param market: IV data market.
    :param bid_ask_default: bid-ask spread to be set by default.
    :return: a triplet (contracts, vanilla_options, calibration_data) to be used in calibration procedure.
    """
    iv_dict = iv_data[source][market][str(observation_date)]
        
    option_names = list(iv_dict.keys())
    implied_vols_market = []
    vanilla_options = []
    contracts = []
    contract_names = []
    bid_ask_spread = []

    for option_name in iv_dict:
        deltas = 1.0 - np.array(iv_dict[option_name]["1-deltas"])
        F0 = iv_dict[option_name]["F0"]
        iv = np.array(iv_dict[option_name]["IV"]) / 100
        ttm = (np.datetime64(iv_dict[option_name]["expiry"]) - np.datetime64(observation_date)) /\
            np.timedelta64(365, 'D')
        K = from_delta_call_to_strike(deltas=deltas, F0=F0, sigma=iv, ttm=ttm)
        implied_vols_market.append(iv)
        vanilla_options.append(VanillaOption(T=ttm, K=K, underlying_name=option_name[:5]))
        if option_name[:5] not in contract_names:
            contract_names.append(option_name[:5])
            contracts.append(ForwardContract(name=option_name[:5],
                                             observation_date=np.datetime64(observation_date),
                                             F0=F0))
        if "bid_ask_spread" in iv_dict[option_name].keys():
            bid_ask_spread.append(iv_dict[option_name]['bid_ask_spread'])
        else:
            bid_ask_spread.append(bid_ask_default)

    calibration_data = CalibrationDataHJM(
        implied_vols_market=implied_vols_market,
        atm_idx=[4] * len(vanilla_options),
        option_names=option_names,
        bid_ask_spread=bid_ask_spread
    )

    return contracts, vanilla_options, calibration_data
