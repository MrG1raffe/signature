import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from numpy import float_
from typing import List, Union
from dataclasses import asdict
from scipy.optimize import minimize
from scipy.special import gamma
import pandas as pd
from matplotlib import cm
from math import ceil
import warnings
from tqdm import tqdm

from models.hjm_model.contracts import ForwardContract
from models.hjm_model.forwards_HJM import ForwardsHJM
from models.hjm_model.calibration_params import CalibrationStrategyHJM, CalibrationDataHJM
from models.model_params import LewisParams
from products.vanilla_option import VanillaOption
from volatility_surface.svi import SSVISmileData, SSVI
from volatility_surface.volatility_surface import get_log_contract_price
from utility.utility import from_delta_call_to_strike, get_months_grid
from utility.colors import ENGIE_BLUE


class CalibrationHJM:
    hjm: ForwardsHJM
    calibration_strategy: CalibrationStrategyHJM
    calibration_data: CalibrationDataHJM
    ssvi: SSVI

    def __init__(
        self,
        hjm: ForwardsHJM,
        calibration_strategy: CalibrationStrategyHJM,
        calibration_data: CalibrationDataHJM,
    ):
        """
        Initializes the calibration object from the HJM object, partially defined calibration strategy
        and partially defined calibration data.

        :param hjm: instance of ForwardsHJM, containing the forward contract models being the underlying of
            the options hjm.vanilla_options.
        :param calibration_strategy: defines the parameters of option pricing, as well as calibration parameters.
        :param calibration_data: options IV market data containing the market prices of options `hjm.vanilla_options`.
        """
        self.hjm = hjm
        if not isinstance(self.hjm.vanilla_options, list):
            self.hjm.vanilla_options = [self.hjm.vanilla_options]

        self.calibration_strategy = calibration_strategy
        # Set the function transforming calibration variable into model parameters and calibration variable bounds.
        self.set_vect_to_params_and_bounds()

        # Fill missing attributes of calibration data
        self.calibration_data = calibration_data
        self.__complete_calibration_data()

        # Get an initial guess for the stochastic volatility parameters.
        x0 = self.get_initial_guess()
        self.update_stoch_vol_params(x0)

    def _smile_idx_to_contract_idx(
        self,
        i: int
    ) -> int:
        """
        Transforms the vanilla option index into the index of its underlying contract.

        :param i: index if option is self.hjm.vanilla_options.
        :return: index of contract in self.hjm.contracts corresponding to the underlying of the i-th option.
        """
        return self.hjm.contract_names.index(self.hjm.vanilla_options[i].underlying_name)

    # --- Calibration setup ---

    def _get_log_contract_prices_ssvi(self) -> None:
        """
        Calibrates the multi-contract SSVI parametrization to market IVs.
        Keeps the calibration results in `self.ssvi`.
        """
        smile_data_list = []
        for option, vols in zip(self.hjm.vanilla_options, self.calibration_data.implied_vols_market):
            contract_idx = [contract.name for contract in self.hjm.contracts].index(option.underlying_name)
            contract = self.hjm.contracts[contract_idx]
            F0 = contract.F0
            k = np.reshape(np.log(option.K / F0), (-1,))
            vols = np.reshape(vols, (-1,))
            # Include the smile if the ATM volatility can be interpolated from other strikes
            if np.min(k) <= 0 <= np.max(k):
                smile_data_list.append(SSVISmileData(
                    contract=contract.name,
                    ttm=option.T[0],
                    log_mon=k,
                    vols=vols,
                    atm_vol=float(np.interp(x=0, xp=k, fp=vols))
                ))
            else:
                smile_data_list.append(None)
        # Calibrate all SSVI params
        ssvi = SSVI()
        ssvi.fit(smile_data_list=[smile_data for smile_data in smile_data_list if smile_data is not None])
        self.ssvi = ssvi
        self.calibration_data.prices_log_contract = []
        for smile_data in smile_data_list:
            if smile_data is not None:
                self.calibration_data.prices_log_contract.append(
                    get_log_contract_price(
                        ttm=smile_data.ttm,
                        smile_func=lambda x: ssvi.implied_vol_from_smile(smile_data=smile_data, log_mon=x)
                    )
                )
            else:
                self.calibration_data.prices_log_contract.append(np.nan)

    def __complete_calibration_data(self) -> None:
        """
        Preprocesses the calibration date, calculates missing attributes.
        """
        if not isinstance(self.calibration_data.prices_market, list) \
            and self.calibration_data.prices_market is not None\
                and not isinstance(self.calibration_data.prices_market, np.ndarray):
            self.calibration_data.prices_market = [self.calibration_data.prices_market]
        if not isinstance(self.calibration_data.implied_vols_market, list) \
            and self.calibration_data.implied_vols_market is not None \
                and not isinstance(self.calibration_data.implied_vols_market, np.ndarray):
            self.calibration_data.implied_vols_market = [self.calibration_data.implied_vols_market]
        if not isinstance(self.calibration_data.atm_idx, list):
            self.calibration_data.atm_idx = [self.calibration_data.atm_idx] * len(self.hjm.vanilla_options)
        if not isinstance(self.calibration_data.bid_ask_spread, list):
            self.calibration_data.bid_ask_spread = [self.calibration_data.bid_ask_spread] * len(self.hjm.vanilla_options)

        self.calibration_data.prices_market_bid = []
        self.calibration_data.prices_market_ask = []

        if self.calibration_data.prices_market is not None:
            self.calibration_data.implied_vols_market = []
            for i, (option, price_market) in enumerate(zip(self.hjm.vanilla_options,
                                                           self.calibration_data.prices_market)):
                F0 = self.hjm.contracts[self._smile_idx_to_contract_idx(i)].F0
                if not self.hjm.is_kemna_vorst:
                    model = self.hjm.models[self._smile_idx_to_contract_idx(i)]
                    F0 = (F0 @ np.diff(model.kemna_vorst_grid)) / (model.kemna_vorst_grid[-1] - model.kemna_vorst_grid[0])
                iv = option.black_iv(option_price=price_market, F=F0).squeeze()
                self.calibration_data.implied_vols_market.append(iv)
                if self.calibration_data.prices_market_bid is not None:
                    self.calibration_data.prices_market_bid.append(
                        option.black_vanilla_price(iv - self.calibration_data.bid_ask_spread[i], F=F0)
                    )
                if self.calibration_data.prices_market_ask is not None:
                    self.calibration_data.prices_market_ask.append(
                        option.black_vanilla_price(iv + self.calibration_data.bid_ask_spread[i], F=F0)
                    )
        else:
            self.calibration_data.prices_market = []
            for i, (option, iv_market) in enumerate(zip(self.hjm.vanilla_options,
                                                        self.calibration_data.implied_vols_market)):
                F0 = self.hjm.contracts[self._smile_idx_to_contract_idx(i)].F0
                price = option.black_vanilla_price(sigma=iv_market, F=F0).squeeze()
                self.calibration_data.prices_market.append(price)
                if self.calibration_data.prices_market_bid is not None:
                    self.calibration_data.prices_market_bid.append(
                        option.black_vanilla_price(iv_market - self.calibration_data.bid_ask_spread[i], F=F0)
                    )
                if self.calibration_data.prices_market_ask is not None:
                    self.calibration_data.prices_market_ask.append(
                        option.black_vanilla_price(iv_market + self.calibration_data.bid_ask_spread[i], F=F0)
                    )
        self._get_log_contract_prices_ssvi()

    def set_vect_to_params_and_bounds(self) -> None:
        """
        Constructs the function self.calibration_strategy.vect_to_params and the parameters bounds
        from self.calibration_strategy.params_to_calibrate.
        """
        contracts_to_calib = list(dict.fromkeys([self.hjm.vanilla_options[i].underlying_name for i in
                                                 self.hjm.smiles_to_calib]))

        def correl_spot_vol_to_rhos(rho_spot_vol: NDArray[float_]) -> NDArray[float_]:
            L = self.hjm.models[0].L
            H = np.zeros((len(rho_spot_vol), len(self.hjm.hist_params.taus)))
            for i, index in enumerate(self.hjm.smiles_to_calib):
                cont_idx = contracts_to_calib.index(self.hjm.vanilla_options[index].underlying_name)
                T = self.hjm.vanilla_options[index].T[0]
                t_grid = np.linspace(0, T, 100)
                sigmas = self.hjm.models[self._smile_idx_to_contract_idx(index)].sigmas(t_grid).T
                H[cont_idx] = (L.T @ sigmas).mean(axis=1) / (np.linalg.norm((L.T @ sigmas), axis=0)).mean()

            x0 = np.zeros(self.hjm.models[0].sigmas(np.zeros(1)).size)
            
            def loss(x):
                return np.sum((H @ x - rho_spot_vol)**2) + 100 * np.maximum(x @ x - 1, 0)
            
            res = minimize(loss, x0=x0)
            return res.x

        n_models = len(contracts_to_calib)
        n_hist = len(self.hjm.hist_params.taus)

        def vector_to_params(x: NDArray[float_]) -> dict:
            EPS = 0.0001
            idx = 0
            params_dict = dict()
        
            for param in self.calibration_strategy.params_to_calibrate:
                if param == "rho_spot_vol" or param == "rho_solo":
                    if param == "rho_solo":
                        rho_spot_vol = [x[idx]] * n_models
                    else:
                        rho_spot_vol = x[idx:idx + n_models]
                    rhos = correl_spot_vol_to_rhos(rho_spot_vol)
                    rhos_norm = np.linalg.norm(rhos)
                    if rhos_norm > 1:
                        rhos /= rhos_norm + EPS
                    params_dict["rhos"] = rhos
                    if param == "rho_solo":
                        idx += 1
                    else:
                        idx += n_models
                elif param == "rhos":
                    params_dict[param] = x[idx:idx+n_hist]
                    idx += n_hist
                elif param == "x" or param == "c":
                    n_factors = len(self.hjm.models[0].c)
                    params_dict[param] = x[idx:idx+n_factors]
                    idx += n_factors
                elif param == "eps_H":
                    n_factors = len(self.hjm.models[0].c)
                    eps, H = x[idx:idx + 2]
                    alpha = H + 0.5
                    ii = np.arange(1, n_factors + 1)
                    r = 1 + 10 * n_factors**(-0.9)
                    params_dict["x"] = (1 - alpha) / (2 - alpha) * (r**(2 - alpha) - 1) / (r**(1 - alpha) - 1) * r**(ii - 1 - 0.5*n_factors)
                    params_dict["c"] = (r**(1 - alpha) - 1) * r**((alpha - 1) * (1 + 0.5 * n_factors)) * r**((1-alpha)*ii) / \
                        gamma(alpha) / gamma(2 - alpha) * np.exp(-params_dict["x"] * eps)
                    idx += 2
                else:
                    params_dict[param] = x[idx]
                    idx += 1
            return params_dict

        BOUNDS_DICT = {
            "V0": [(0.01, 2)],
            "theta": [(0.001, 4)],
            "nu": [(0, 10)],
            "lam": [(0, 30)],
            "c": [(0, 5)] * len(self.hjm.models[0].c),
            "x": [(0, 100)] * len(self.hjm.models[0].x),
            "eps_H": [(0.001, 2)] + [(-1, 1)],
            "rho_spot_vol": [(-1, 1)] * n_models,
            "rhos": [(-1, 1)] * n_hist,
            "kappa": [(0, 30)],
            "X0": [(0.01, 5)],
            "rho_solo": [(-1, 1)]
        }
        bounds = []
        for p in self.calibration_strategy.params_to_calibrate:
            bounds = bounds + BOUNDS_DICT[p]

        self.calibration_strategy.vector_to_params = vector_to_params
        self.calibration_strategy.bounds = bounds

    def get_initial_guess(self) -> NDArray[float_]:
        """
        Gives and initial guess for stochastic volatility parameters.

        :return: x0, and array corresponding to self.calibration_strategy.params_to_calibrate.
        """
        contracts_to_calib = list(dict.fromkeys([self.hjm.vanilla_options[i].underlying_name for i in
                                                 self.hjm.smiles_to_calib]))
        n_hist = len(self.hjm.hist_params.taus)

        VALS_DICT = {
            "V0": [1],
            "theta": [1],
            "nu": [1],
            "lam": [1],
            "c": [1 / len(self.hjm.models[0].c)] * len(self.hjm.models[0].c),
            "x": np.logspace(-0.3, 1.6, len(self.hjm.models[0].x)),
            "eps_H": [0.1, 0.4],
            "rho_spot_vol": [self.ssvi.rho.get(name, 0) for name in contracts_to_calib],
            "rhos": [0] * n_hist,
            "rho_solo": [0.4]
        }

        x0 = []
        for param in self.calibration_strategy.params_to_calibrate:
            x0 = x0 + list(VALS_DICT[param])
        return x0

    # --- Direct calibration of the function g ---
    def calibrate_g_to_smiles(
        self,
        mode="shape",
        g_calib_idx=None
    ):
        """
        Calibrates the function g for the smiles with indices in `g_calib_idx` via the list square minilization
        either for the whole smile if mode=="shape", either to match the ATM vol if mode == "atm".

        :param mode: either "shape" or "atm". If "shape", calibrates the whole smile, otherwise, calibrates only the
            ATM volatility.
        :param g_calib_idx: indices of smiles to be calibrated. If not specified, taken equal to `self.hjm.g_indices`.
        """
        if g_calib_idx is None:
            g_calib_idx = self.hjm.g_indices
        calibrated = set()
        for i in g_calib_idx:
            name = self.hjm.contract_names[self._smile_idx_to_contract_idx(i)]
            g_idx_to_calib = list(set(self.hjm.g_function.get_contract_indices(name)) - calibrated)
            if len(g_idx_to_calib) == 0:
                warnings.warn(f"No degrees of freedom to calibrate smile level for {name}")
            else:
                def loss(x):
                    self.hjm.g_function.function_values[g_idx_to_calib] = x[0]
                    self.hjm.update_sigmas()
                    loss_val = 0
                    contract_idx = self._smile_idx_to_contract_idx(i)
                    shape_prices = self.hjm.vanilla_options[i].K.shape

                    prices_mkt = np.reshape(self.calibration_data.prices_market[i], shape_prices)
                    prices_model = np.reshape(
                        self.hjm.vanilla_options[i].get_price(
                            model=self.hjm.models[contract_idx],
                            method=self.calibration_strategy.pricing_method,
                            F0=self.hjm.contracts[contract_idx].F0,
                            **asdict(self.calibration_strategy.pricing_params)),
                        shape_prices
                    )
                    if mode == "atm":
                        atm_diff = (prices_mkt[-1, self.calibration_data.atm_idx[i]] -
                                    prices_model[-1, self.calibration_data.atm_idx[i]])
                        loss_val += atm_diff ** 2
                    elif mode == "shape":
                        vega = np.reshape(
                            self.hjm.vanilla_options[i].vega(
                                sigma=self.calibration_data.implied_vols_market[i],
                                F=self.hjm.contracts[contract_idx].F0
                            ),
                            shape_prices
                        )
                        smile_diff = (prices_mkt - prices_model) / vega
                        loss_val += np.sum(smile_diff ** 2)
                    return loss_val

                res = minimize(loss, x0=np.ones(1), method='Powell', bounds=[(0.001, 3)])
                if res.x < 0.01 or res.x > 2:
                    warnings.warn(f"Extreme value {res.x} of g(T) was obtained in the calibration procedure.")
                calibrated.update(g_idx_to_calib)

    # --- Term structure calibration ---
    def _update_g_from_list(
        self,
        g_arr: Union[List[float], NDArray[float_]],
        interpolate_sigmas: bool = False
    ) -> None:
        """
        Updates the piece-wise function g from the values corresponding to delivery periods of options in
        `self.hjm.g_indices`. If the delivery periods overlap, uses the part of the delivery period which was not
        modified before.

        :param g_arr: values of the function g corresponding to the delivery periods of underlyings
            in `self.hjm.g_indices`.
        :param interpolate_sigmas: whether to use interpolated function `sigmas` when updating the deterministic vol.
        """
        updated = set()
        # TODO: Understand whether the result depends on the order of contracts in `self.hjm.g_indices`.
        for g, option_idx in zip(g_arr, self.hjm.g_indices):
            name = self.hjm.vanilla_options[option_idx].underlying_name
            g_idx_to_calib = list(set(self.hjm.g_function.get_contract_indices(name)) - updated)
            if len(g_idx_to_calib) == 0:
                raise ValueError(f"No degrees of freedom to calibrate the smile level with function g for {name}")
            self.hjm.g_function.function_values[g_idx_to_calib] = g
            updated.update(g_idx_to_calib)
        self.hjm.update_sigmas(is_interpolate=interpolate_sigmas)

    def __sigma_reduced(
        self,
        t_grid: NDArray[float_],
        option_idx: int,
        hjm_without_g: ForwardsHJM
    ) -> NDArray[float_]:
        """
        Calculates the value of the reduced deterministic volatility `sigmas` that corresponds to a subset of the
        delivery period that does not overlap with contracts corresponding to the smiles in `self.hjm.g_indices`.

        :param t_grid: grid for the reduced volatility to be evaluated.
        :param option_idx: index of the option for the underlying of which the volatility is being computed.
        :param hjm_without_g: a copy of `self.hjm` with the function g set equal to 1.
        :return: an array of the reduced volatility evaluated on `t_grid`.
        """
        if option_idx not in self.hjm.inclusion_dict:
            return hjm_without_g.models[self._smile_idx_to_contract_idx(option_idx)].sigmas(t_grid)
        else:
            cont_idx = self._smile_idx_to_contract_idx(option_idx)
            dependent_options_indices = self.hjm.inclusion_dict[option_idx]
            sigmas_cont = hjm_without_g.models[cont_idx].sigmas
            contract = hjm_without_g.contracts[cont_idx]
            res = sigmas_cont(t_grid)
            for depend_option_idx in dependent_options_indices:
                depend_cont_idx = self._smile_idx_to_contract_idx(depend_option_idx)
                depend_contract = hjm_without_g.contracts[depend_cont_idx]
                sigmas_depend_cont = hjm_without_g.models[depend_cont_idx].sigmas
                weight = (depend_contract.time_to_delivery_end - depend_contract.time_to_delivery_start) / \
                         (contract.time_to_delivery_end - contract.time_to_delivery_start)
                res -= weight * sigmas_depend_cont(t_grid)
            return res

    def calibrate_h(
        self,
        dt: float = 5e-4,
    ) -> None:
        """
        Strips the function h to match the variance swap prices of the options in `self.hjm.h_indices`.

        :param dt: numerical integration step used to calculate the quadratic variation.
        """
        N_steps = len(self.hjm.h_function.function_values)
        self.hjm.h_function.function_values = np.ones(N_steps)
        for i in range(N_steps):
            option_idx = self.hjm.h_function.option_indices[i + 1]
            contract_idx = self._smile_idx_to_contract_idx(option_idx)
            model = self.hjm.models[contract_idx]
            T1 = self.hjm.h_function.changing_points[i]
            T2 = self.hjm.h_function.changing_points[i + 1]

            t_grid = np.linspace(T1, T2, int(np.ceil((T2 - T1) / dt)))
            integrated_determ_var = np.trapz(
                y=model.deterministic_variance(t_grid),
                x=t_grid
            )
            option_idx_prev = self.hjm.h_function.option_indices[i]
            if i == 0:
                log_contract_diff = self.calibration_data.prices_log_contract[option_idx]
                updated_forward_variance = log_contract_diff / integrated_determ_var
            elif self.hjm.vanilla_options[option_idx].underlying_name == \
                    self.hjm.vanilla_options[option_idx_prev].underlying_name:
                log_contract_diff = self.calibration_data.prices_log_contract[option_idx]
                log_contract_diff -= self.calibration_data.prices_log_contract[option_idx_prev]
                updated_forward_variance = log_contract_diff / integrated_determ_var
            else:
                log_contract_diff = self.calibration_data.prices_log_contract[option_idx]
                log_contract_diff -= self.hjm.models[contract_idx].quadratic_variation(T=T1, dt=dt)
                updated_forward_variance = log_contract_diff / integrated_determ_var

            self.hjm.h_function.function_values[i] = np.sqrt(updated_forward_variance)
            if updated_forward_variance < 0.01 or updated_forward_variance > 100:
                warnings.warn(f"Extreme value of forward variance was obtained: "
                              f"{self.hjm.h_function.function_values[i]}")

    def calibrate_h_numerical(
        self,
        dt: float = 1 / 365,
    ):
        """
        Calibrates the values of function one by one numerically to match the variance swap volatilities
        of the contract when the interpolated function h is used.

        :param dt: numerical integration step used to calculate the quadratic variation.
        """
        N_steps = len(self.hjm.h_function.function_values)
        for i in range(N_steps):
            option_idx = self.hjm.h_function.option_indices[i + 1]
            contract_idx = self._smile_idx_to_contract_idx(option_idx)
            model = self.hjm.models[contract_idx]
            T1 = self.hjm.h_function.changing_points[i]
            T2 = self.hjm.h_function.changing_points[i + 1]
            qv_1 = model.quadratic_variation(T=T1, dt=dt)

            def loss(h):
                self.hjm.h_function.function_values[i] = h
                self.hjm.update_sigmas(is_interpolate=False)
                qv_12 = model.quadratic_variation(T0=T1, T=T2, dt=dt)
                vs_vol_model = np.sqrt((qv_1 + qv_12) / T2)
                vs_vol_mkt = np.sqrt(self.calibration_data.prices_log_contract[option_idx] / T2)
                return (vs_vol_model - vs_vol_mkt)**2

            res = minimize(loss, x0=self.hjm.h_function.function_values[i:i+1], method="L-BFGS-B", bounds=[(0.01, 3)])
            self.hjm.h_function.function_values[i] = res.x
            self.hjm.update_sigmas(is_interpolate=False)
            if res.x < 0.01 or res.x > 100:
                warnings.warn(f"Extreme value of forward variance was obtained: "
                              f"{self.hjm.h_function.function_values[i]}")

    def fixed_point_iteration(
        self,
        g_arr: NDArray[float_],
        dt: float = 0.001,
    ) -> NDArray[float_]:
        """
        Performs one interation of the fixed-point calibration algorithm:
        1. Initializes the function g with `g_arr`.
        2. Calibrates the function h
        3. Calculates the values of g to match the forward variances with new function h.

        :param g_arr: values of the function g corresponding to the delivery periods of underlyings of options
            in `self.hjm.g_indices`.
        :param dt: numerical intergration step used to approximate the integrals of the determensitic volatility.
        :return: and array of the same size as `g_arr` with updated values of the function g.
        """
        self._update_g_from_list(g_arr, interpolate_sigmas=False)
        self.calibrate_h()
        hjm_without_g = self.hjm.get_hjm_without_g()
        g_res = np.zeros_like(g_arr) * np.nan
        for i, (g, option_idx) in enumerate(zip(g_arr, self.hjm.g_indices)):
            # Calibrated one value per delivery period for the contracts which delivery period do not contain
            # other delivery periods.
            if option_idx not in self.hjm.inclusion_dict:
                contract_idx = self._smile_idx_to_contract_idx(option_idx)
                T = self.hjm.vanilla_options[option_idx].T[0]
                quadratic_variation = hjm_without_g.models[contract_idx].quadratic_variation(T=T)
                g_res[i] = np.sqrt(self.calibration_data.prices_log_contract[option_idx] / quadratic_variation)
        for i, (g, option_idx) in enumerate(zip(g_arr, self.hjm.g_indices)):
            # For the contracts containing other contracts, the function g is calibrated on the part of the delivery
            # period having empty intersection with other contracts.
            if option_idx in self.hjm.inclusion_dict:
                contract = hjm_without_g.contracts[self._smile_idx_to_contract_idx(option_idx)]
                T = self.hjm.vanilla_options[option_idx].T[0]
                t_grid = np.linspace(0, T, int(np.ceil(T / dt)))
                sigmas = self.__sigma_reduced(t_grid=t_grid, option_idx=option_idx, hjm_without_g=hjm_without_g)
                R = self.hjm.hist_params.corr_mat
                a = np.trapz(y=np.sum(sigmas.T * (R @ sigmas.T), axis=0), x=t_grid)
                b = 0
                c = -self.calibration_data.prices_log_contract[option_idx]
                for depend_option_idx_1 in self.hjm.inclusion_dict[option_idx]:
                    depend_contract_1 = hjm_without_g.contracts[self._smile_idx_to_contract_idx(depend_option_idx_1)]
                    weight_1 = (depend_contract_1.time_to_delivery_end - depend_contract_1.time_to_delivery_start) / \
                               (contract.time_to_delivery_end - contract.time_to_delivery_start)

                    sigmas_dep_1 = self.__sigma_reduced(t_grid=t_grid, option_idx=depend_option_idx_1,
                                                        hjm_without_g=hjm_without_g)
                    b += 2 * g_res[self.hjm.g_indices.index(depend_option_idx_1)] * weight_1 * \
                        np.trapz(y=np.sum(sigmas.T * (R @ sigmas_dep_1.T), axis=0), x=t_grid)
                    for depend_option_idx_2 in self.hjm.inclusion_dict[option_idx]:
                        depend_contract_2 = hjm_without_g.contracts[self._smile_idx_to_contract_idx(depend_option_idx_2)]
                        weight_2 = (depend_contract_2.time_to_delivery_end - depend_contract_2.time_to_delivery_start) \
                            / (contract.time_to_delivery_end - contract.time_to_delivery_start)
                        sigmas_dep_2 = self.__sigma_reduced(t_grid=t_grid, option_idx=depend_option_idx_2,
                                                            hjm_without_g=hjm_without_g)
                        c += g_res[self.hjm.g_indices.index(depend_option_idx_1)] * weight_1 * \
                            g_res[self.hjm.g_indices.index(depend_option_idx_2)] * weight_2 * \
                            np.trapz(y=np.sum(sigmas_dep_1.T * (R @ sigmas_dep_2.T), axis=0), x=t_grid)
                g_res[i] = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        return g_res

    def fixed_point_iteration_numerical(
        self,
        g_arr: NDArray[float_],
        dt: float = 1 / 365
    ) -> NDArray[float_]:
        """
        Performs one interation of the fixed-point calibration algorithm:
        1. Initializes the function g with `g_arr`.
        2. Calibrates numerically the function h
        3. Calculates numerically the values of g to match the VS volatilities with new function h.

        :param g_arr: values of the function g corresponding to the delivery periods of underlyings of options
            in `self.hjm.g_indices`.
        :param dt: numerical integration step used to calculate the quadratic variation.
        :return: and array of the same size as `g_arr` with updated values of the function g.
        """
        self._update_g_from_list(g_arr, interpolate_sigmas=False)
        self.calibrate_h_numerical()
        g_arr_new = g_arr.astype(float)
        for i, (g, option_idx) in enumerate(zip(g_arr, self.hjm.g_indices)):
            contract_idx = self._smile_idx_to_contract_idx(option_idx)
            T = self.hjm.vanilla_options[option_idx].T[0]

            def loss(g):
                g_arr_new[i] = g
                self._update_g_from_list(g_arr=g_arr_new, interpolate_sigmas=False)
                quadratic_variation = self.hjm.models[contract_idx].quadratic_variation(T=T, dt=dt)
                return (np.sqrt(quadratic_variation / T) -
                        np.sqrt(self.calibration_data.prices_log_contract[option_idx] / T))**2

            res = minimize(fun=loss, x0=g_arr[i:i+1], method="L-BFGS-B", bounds=[(0.05, 3)])
            g_arr_new[i] = res.x
            self._update_g_from_list(g_arr=g_arr_new, interpolate_sigmas=False)
        return g_arr_new

    def calibrate_term_structure(
            self,
            tol: float = 1e-4,
            max_iter: int = 50,
            dt: float = 1 / 365
    ) -> None:
        """
        Performs the fixed-point calibration algorithm to calibrate the functions g and h.

        :param tol: tolerance, algorithm stops once ||g_{N + 1} - g_{N}|| < tol.
        :param max_iter: maximal number of iterations to be done.
        :param dt: numerical integration step used in the algorithm iterations.
        """
        print("Calibrating IV term structure...")
        self.hjm.g_function.reset()
        g_prev = np.ones_like(self.hjm.g_indices)
        numerical = self.hjm.interpolation_method is not None
        for i in range(max_iter):
            if numerical:
                g = self.fixed_point_iteration_numerical(
                    g_arr=g_prev,
                    dt=dt
                )
            else:
                g = self.fixed_point_iteration(
                    g_arr=g_prev,
                    dt=dt,
                )
            err = np.max(np.abs(np.array(g) - np.array(g_prev)))
            if err < tol:
                print(f"Calibrated g(T) successfully in {i + 1} iterations.")
                break
            g_prev = g
        else:
            print(f"Fixed point may have not been achieved after {max_iter} iterations, "
                  f"current error: {err}.")
        self._update_g_from_list(g_prev)
        if numerical:
            self.calibrate_h_numerical()
        else:
            self.calibrate_h()
        self.hjm.update_sigmas(is_interpolate=True)

        print("Calibrated VS volatilities:")
        self.print_vs_vols()

    def print_vs_vols(self) -> None:
        """
        Prints the variance swap volatilities given by the model and implied from the market data to check the
        calibration quality.
        """
        for i, option in enumerate(self.hjm.vanilla_options):
            T = option.T[0]
            contract_idx = self._smile_idx_to_contract_idx(i)
            print(str(self.calibration_data.option_names[i]),
                  ": Market = ", np.round(np.sqrt(self.calibration_data.prices_log_contract[i] / T), 4),
                  ", Model = ", np.round(np.sqrt(self.hjm.models[contract_idx].quadratic_variation(T) / T), 4))

    # --- Smile calibration ---

    def update_stoch_vol_params(
        self,
        x: NDArray[float_]
    ) -> None:
        """
        Updates the stochastic volatility parameters in `self.hjm` from the calibration variable `x`.

        :param x: calibration variable corresponding to the parameters listed in
            `self.calibration_strategy.params_to_calibrate`.
        """
        # Use it in all calibration losses.
        model_params_dict = self.calibration_strategy.vector_to_params(x)
        self.hjm.update_stoch_vol_params(model_params_dict)

    def loss_prices(
        self,
        x: NDArray[float_]
    ) -> float:
        """
        Calculates the smile shape calibration loss corresponding to the calibration variable `x`.

        :param x: calibration variable corresponding to the parameters listed in
            `self.calibration_strategy.params_to_calibrate`.
        :return: the value of the loss function calculated as a weighted mean squared error of the option prices
            divided by vega.
        """
        loss = 0
        self.update_stoch_vol_params(x)
        for i in self.hjm.smiles_to_calib:
            contract_idx = self.hjm.contract_names.index(self.hjm.vanilla_options[i].underlying_name)
            K_shape = self.hjm.vanilla_options[i].K.shape
            prices_model = np.reshape(
                self.hjm.vanilla_options[i].get_price(
                    model=self.hjm.models[contract_idx],
                    method=self.calibration_strategy.pricing_method,
                    F0=self.hjm.contracts[contract_idx].F0,
                    **asdict(self.calibration_strategy.pricing_params)
                ),
                K_shape
            )[-1]
            vega = np.reshape(
                self.hjm.vanilla_options[i].vega(
                    sigma=self.calibration_data.implied_vols_market[i],
                    F=self.hjm.contracts[contract_idx].F0
                ),
                K_shape
            )[-1]
            prices_mkt = np.reshape(self.calibration_data.prices_market[i], K_shape)[-1]
            prices_mkt_ask = np.reshape(self.calibration_data.prices_market_ask[i], K_shape)[-1]
            prices_mkt_bid = np.reshape(self.calibration_data.prices_market_bid[i], K_shape)[-1]
            w = 1 if self.calibration_strategy.smile_weights is None else self.calibration_strategy.smile_weights[i]
            loss += w * np.sum(
                ((prices_model - prices_mkt) / vega) ** 2) + self.calibration_strategy.bid_ask_penalization * \
                np.sum((np.maximum(prices_model - prices_mkt_ask, 0) / vega) ** 2 +
                       (np.maximum(prices_mkt_bid - prices_model, 0) / vega) ** 2)
        return loss

    def calibrate_smile_shape(
        self,
        x0: NDArray[float_] = None,
        callback_type: str = "standard"
    ) -> None:
        """
        Performs the smile shape calibration.

        :param x0: Initial value. By default, take the initial value given by `self.get_initial_guess`.
        :param callback_type: "standard" or "update_vol_level". If "update_vol_level", at each call of the optimizer
            callback function, updates the function g to match the smile shape.
        """
        if x0 is None:
            x0 = self.get_initial_guess()
        else:
            if len(x0) != len(self.calibration_strategy.bounds):
                raise ValueError(f"Inconsistent dimensions of x0 ({len(x0)}) and "
                                 f"given bounds ({len(self.calibration_strategy.bounds)})")
            self.update_stoch_vol_params(x0)

        print("Calibrating stoch vol params...")

        def callback(x):
            if callback_type == "update_vol_level":
                self.calibrate_g_to_smiles(mode="shape")
            print(x)

        def loss_fun(x):
            return self.loss_prices(x=x)

        self.hjm.update_sigmas(is_interpolate=True)
        callback(x0)
        res = minimize(
            fun=loss_fun,
            x0=x0,
            method=self.calibration_strategy.optimiser,
            bounds=self.calibration_strategy.bounds,
            callback=callback
        )
        self.update_stoch_vol_params(res.x)
        print(res)
        print('Done')

    # --- Plotting section ---

    def plot_gh(self) -> None:
        """
        Plots the functions h and g.
        """
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        fig.tight_layout(pad=6)

        self.hjm.h_function.plot(ax[0])
        self.hjm.g_function.plot(ax[1])

    def plot_smiles(
        self,
        ncols: int = 2,
        mode: str = "all",
    ) -> None:
        """
        Plots the IV smiles given by the model and market data.

        :param ncols: number of plt.subplots columns.
        :param mode: "all" or "calibrated". If "all", plots all the smiles. If "calibrated", plots only the smiles
            from `self.hjm.smiles_to_calib`.
        """
        if mode == "all":
            idx_to_plot = np.arange(len(self.hjm.vanilla_options))
        elif mode == "calibrated":
            idx_to_plot = self.hjm.smiles_to_calib
        else:
            raise ValueError("Wrong value of `mode` was given.")

        nrows = ceil(len(idx_to_plot) / ncols)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(4.5*ncols, 3.5*nrows))
        if nrows == 1:
            ax = [ax]
        fig.tight_layout(pad=3.5)

        for i, idx in enumerate(idx_to_plot):
            ax_i = ax[int(i // ncols)][i % ncols]
            contract_idx = self._smile_idx_to_contract_idx(idx)
            price_model = self.hjm.vanilla_options[idx].get_price(
                model=self.hjm.models[contract_idx],
                method=self.calibration_strategy.pricing_method,
                F0=self.hjm.contracts[contract_idx].F0,
                **asdict(self.calibration_strategy.pricing_params))            

            # Modification due to no KV approximation and MC method
            if not self.hjm.is_kemna_vorst:
                F0 = (self.hjm.contracts[contract_idx].F0 @ np.diff(self.hjm.models[contract_idx].kemna_vorst_grid)) / \
                    (self.hjm.models[contract_idx].kemna_vorst_grid[-1] - self.hjm.models[contract_idx].kemna_vorst_grid[0])
            else:
                F0 = self.hjm.contracts[contract_idx].F0

            if self.calibration_strategy.pricing_method == 'mc':
                if self.calibration_strategy.pricing_params.return_accuracy:
                    price_model_real, price_model_low, price_model_up = price_model
                else:
                    price_model_real, price_model_low, price_model_up = price_model, None, None
            else:
                price_model_real, price_model_low, price_model_up = price_model, None, None

            self.hjm.vanilla_options[idx].plot_smiles(
                option_name=self.calibration_data.option_names[idx],
                option_prices_model=price_model_real,
                option_prices_low_model=price_model_low,
                option_prices_up_model=price_model_up,
                option_prices_market=self.calibration_data.prices_market[idx],
                F0=F0,
                ax=ax_i,
                bid_ask_spread=[self.calibration_data.bid_ask_spread[idx]]
            )
        plt.show()

    def plot_monthly_smiles(self) -> None:
        """
        Plots the surface of monthly contract smiles with maturities five days before the delivery start date.
        The month grid is constructed between the first and the last month covered by delivery periods of
        `self.hjm.contracts`.
        """
        monthly_contracts = []
        monthly_models = []
        start_date = (self.hjm.observation_date + pd.DateOffset(months=1)).replace(day=1)
        end_date_idx = np.argmax(np.array([contract.time_to_delivery_end for contract in self.hjm.contracts]))
        end_date = [contract.delivery_end_date for contract in self.hjm.contracts][end_date_idx - 1]
        monthly_contract_names = get_months_grid(T_s=start_date, T_e=end_date)

        for i, name in enumerate(monthly_contract_names):
            mon_contract = ForwardContract(name=name, observation_date=self.hjm.observation_date, F0=100)
            model_monthly = self.hjm.get_model_for_contract(contract=mon_contract)
            monthly_models.append(model_monthly)
            monthly_contracts.append(mon_contract)

        options = []
        smiles = []
        delta_arr = np.linspace(0.1, 0.9, 50)
        for i, (model, m_contract) in enumerate(zip(monthly_models, monthly_contracts)):
            T = m_contract.time_to_delivery_start - 5 / 365
            # WARNING: MAGIC CONSTANT 0.5
            strikes = np.flip(from_delta_call_to_strike(deltas=delta_arr, F0=m_contract.F0, sigma=0.5, ttm=T))
            option = VanillaOption(T=T, K=strikes)
            smile = option.get_price(model=model, method=self.calibration_strategy.pricing_method, F0=m_contract.F0,
                                     is_vol_surface=True, **asdict(self.calibration_strategy.pricing_params))
            smiles.append(smile)
            options.append(option)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 8))

        XX = np.zeros((smiles[0].size, len(options)))
        YY = np.zeros_like(XX)
        ZZ = np.zeros_like(XX)
        for i, (option, smile, m_contract) in enumerate(zip(options, smiles, monthly_contracts)):
            XX[:, i] = option.T * np.ones_like(smile)
            YY[:, i] = np.log(option.K / m_contract.F0).squeeze()
            ZZ[:, i] = smile

        ax.plot_surface(XX, YY, ZZ, cmap=cm.cool, linewidth=1, antialiased=False)

        mon_step = 3
        ax.set_xticks([option.T[0] for option in options[::mon_step]],
                      labels=monthly_contract_names[::mon_step], rotation=45)
        ax.set_ylabel("Log-moneyness")
        ax.set_zlabel("IV")

        plt.show()

    def plot_daily_vols(
        self,
        n_days: int = 3 * 365
    ) -> None:
        """
        Plots the daily ATM volatility structure for the `n_days` days following the observation date. Namely, for
        each day a daily contract is created with maturity one day before the delivery start.

        :param n_days: number of days in grid.
        """
        F = 100
        days_range = range(2, n_days)
        daily_contracts = [ForwardContract(name="D" + str(day), observation_date=self.hjm.observation_date, F0=F)
                           for day in days_range]
        daily_options = [VanillaOption(T=contract.time_to_delivery_start - 1 / 365, K=F) for contract in
                         daily_contracts]
        daily_models = [self.hjm.get_model_for_contract(contract) for contract in daily_contracts]

        daily_vols_atm = []
        for option, model in tqdm(zip(daily_options, daily_models)):
            daily_vols_atm.append(option.get_price(model=model, method="lewis", F0=F,
                                                   is_vol_surface=True,
                                                   pricing_params=LewisParams(N_points=25, cf_timestep=0.0001)))

        fig, ax = plt.subplots()
        ax.plot(days_range, daily_vols_atm, color=ENGIE_BLUE)
        ax.grid()

        xticks = days_range[::90]
        xlabels = (self.hjm.observation_date + xticks).astype(str)
        ax.set_xticks(xticks, labels=xlabels, rotation=-90)
        plt.show()

    def plot_corr_matrix_difference(self) -> None:
        """
        Plots the matrix of differences (in ||.||_∞) between the initial instantaneous correlations and the
        instantaneous correlations after the calibration (impacted by the function g).
        """
        hjm_unnormalized = self.hjm.get_hjm_without_g()
        dist_mat = np.zeros((len(self.hjm.contract_names), len(self.hjm.contract_names)))

        for i in range(len(self.hjm.contract_names)):
            for j in range(i + 1, len(self.hjm.contract_names)):
                t_grid, rhos = self.hjm.get_instantaneous_correlation(contract_idx_1=i, contract_idx_2=j)
                t_grid, rhos_hist = hjm_unnormalized.get_instantaneous_correlation(contract_idx_1=i, contract_idx_2=j)
                dist_mat[i, j] = dist_mat[j, i] = np.max(np.abs(rhos - rhos_hist))

        fig, ax = plt.subplots(figsize=(9, 9))
        im = ax.imshow(dist_mat, cmap='cool', alpha=1)
        for i in range(len(self.hjm.contract_names)):
            for j in range(len(self.hjm.contract_names)):
                ax.text(j, i, np.round(dist_mat[i, j], 4),
                        ha="center", va="center", color="k")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(len(self.hjm.contracts)), labels=self.hjm.contract_names, rotation=0)
        ax.set_yticks(np.arange(len(self.hjm.contracts)), labels=self.hjm.contract_names)
        ax.set_title("Max absolute deviation of instantaneous correlations")
        plt.show()

    def plot_spot_vol_correlations(self) -> None:
        """
        Plots the instantaneous spot-vol correlations between for all contracts in `self.hjm.contracts`.
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.grid("on")
        ax.set_ylim([-1, 1])

        for contract, model in zip(self.hjm.contracts, self.hjm.models):
            t_grid = np.linspace(0, contract.time_to_delivery_start, 500)
            rho_spot_vol = model.spot_vol_correlation(t_grid)
            ax.plot(t_grid, rho_spot_vol, label=contract.name)
        ax.legend()
        xticks = np.array([contract.time_to_delivery_start for contract in self.hjm.contracts])
        xlabels = (self.hjm.observation_date + (xticks * 365).astype(int)).astype(str)
        ax.set_xticks(xticks, labels=xlabels, rotation=-90)
