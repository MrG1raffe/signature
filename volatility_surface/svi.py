from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from numpy.typing import NDArray
from numpy import float_
import matplotlib.pyplot as plt
from typing import List, Union

PARAM_BOUNDS_SSVI = {"rho": (-1, 1), "eta": (0, 2), "gamma": (0, 0.5)}
RHO_DEFAULT = 0.4


class SSVISmileData:
    contract: str
    ttm: float
    log_mon: NDArray[float_]
    vols: NDArray[float_]
    atm_vol: float

    def __init__(
        self,
        contract: str,
        vols: NDArray[float_],
        ttm: float = None,
        expiry_date: str = None,
        observation_date: str = None,
        strikes: NDArray[float_] = None,
        log_mon: NDArray[float_] = None,
        delta_strike: NDArray[float_] = None,
        F0: float = None,
        atm_vol: float = None
    ):
        """
        Creates an SSVISmileData object which can be passed to the SSVI model for calibration. The timt to maturity in
        years can be specified either directly `ttm`, or via the expiry data and the observation date. Strikes can
        be defined either directly (strikes), or via log-moneyness (log-mon), or as delta-strikes (delta_strike). If
        the strikes are given directly, the underlying price F0 should be specified. ATM implied volatility `atm_vol`
        should be given if it cannot be interpolated from vols.

        :param contract: contract name.
        :param vols: array of the implied volatilities corresponding to the smile.
        :param ttm: time to maturity in years.
        :param expiry_date: expiry date in the format "YYYY-MM-DD".
        :param observation_date: observation date in the format "YYYY-MM-DD".
        :param strikes: array of absolute strikes.
        :param log_mon: log-moneyness array.
        :param delta_strike: array of delta-strikes.
        :param F0: underlying price at t = 0.
        :param atm_vol: ATM volatility.
        """
        self.contract = contract
        vols = np.array(vols)
        self.vols = vols

        if ttm is not None:
            self.ttm = ttm
        elif expiry_date is not None and observation_date is not None:
            self.ttm = (np.datetime64(expiry_date) - np.datetime64(observation_date)) / np.timedelta64(365,'D')
        else:
            raise ValueError("For the smile maturity either Time To Maturity (ttm), "
                             "either observation date and expiry date should be given.")

        if log_mon is not None:
            self.log_mon = np.array(log_mon)
        elif strikes is not None and F0 is not None:
            self.log_mon = np.log(np.array(strikes) / F0)
        elif delta_strike is not None:
            self.log_mon = 0.5 * self.vols**2 * self.ttm - self.vols * np.sqrt(self.ttm) * norm.ppf(delta_strike)
        else:
            raise ValueError("For the option strikes, either log-moneyness, or absolute strikes and spot price,"
                             "or delta-strike should be provided.")
        if self.log_mon.size != self.vols.size:
            raise ValueError(f"Strikes and IV array sizes should be equal (for {self.contract}, "
                             f"strike size {self.log_mon.size} and IV size {self.vols.size} were given).")

        if atm_vol is not None:
            self.atm_vol = atm_vol
        elif np.min(self.log_mon) <= 0 and np.max(self.log_mon) >= 0:
            self.atm_vol = np.interp(x=0, xp=self.log_mon, fp=self.vols)
        else:
            raise ValueError("ATM volatility cannot be interpolated from the given data. "
                             "Please, provide atm_vol explicitly.")


@dataclass
class SSVI:
    rho: dict = None
    eta: float = 0.5
    gamma: float = 0.5

    def __post_init__(self):
        if self.rho is None:
            self.rho = dict()

    def set_params(
        self,
        x: NDArray[float_],
        params_to_set: List
    ) -> None:
        """
        Set the values given in the vector 'x' as the parameters of the model.

        :param x: vector with values.
        :param params_to_set: list of parameter names corresponding to the elements of 'x'.
        """
        x_ptr = 0
        for param in params_to_set:
            if param == "rho":
                for idx in sorted(self.rho.keys()):
                    self.rho[idx] = x[x_ptr]
                    x_ptr += 1
            # elif param == "eta":
            #     # No-arbitrage condition for the function phi
            #     self.eta = x[x_ptr]
            #     if self.eta * (1 + np.max(np.abs(list(self.rho.values())))) > 2:
            #         self.eta = 2 / (1 + np.max(np.abs(list(self.rho.values()))))
            #     x_ptr += 1
            else:
                self.__dict__[param] = x[x_ptr]
                x_ptr += 1

    def set_contract_rho(self, contract: str, rho: float) -> None:
        """
        Set the correlation parameter rho for a given contract.

        :param contract: contract name.
        :param rho: correlation value.
        """
        self.rho[contract] = rho

    def phi(
        self,
        x: Union[float, NDArray[float_]]
    ) -> Union[float, NDArray[float_]]:
        """
        The function phi used in the SSVI parametrization.

        :param x: the function argument.
        :return: phi(x)
        """
        return self.eta / x ** self.gamma / ((1 + x) ** (1 - self.gamma))

    def _get_smile(
        self,
        k: NDArray[float_],
        T: float,
        atm_vol: float,
        contract: str
    ) -> NDArray[float_]:
        """
        Calculates the IV in the SSVI parametrization for the given contract.

        :param k: log-moneyness.
        :param T: time to maturity.
        :param atm_vol: ATM volatility.
        :param contract: contract name.
        :return: Black-Scholes implied volatility parametrized with SSVI for the given smile.
        """
        theta = atm_vol**2 * T
        if contract not in self.rho:
            raise KeyError(f"No spot-vol correlation defined for the contract {contract}. "
                           f"Use the method set_contract_rho to add the correlation.")
        return np.sqrt(0.5 * theta / T * (1 + self.rho[contract] * self.phi(theta) * k + np.sqrt(
            (self.phi(theta) * k + self.rho[contract])**2 + (1 - self.rho[contract]**2)
        )))

    def implied_vol_from_smile(
        self,
        smile_data: SSVISmileData,
        strikes: NDArray[float_] = None,
        log_mon: NDArray[float_] = None,
        delta_strike: NDArray[float_] = None,
        F0: float = None,
    ):
        """
        Interpolates the IV smile defined by smile_data.

        :param smile_data: an object of type SSVISmileData containing the smile data.
        :param strikes: absolute strikes.
        :param log_mon: log-moneyness.
        :param delta_strike: delta-strikes.
        :param F0: underlying price at t = 0.
        :return: an array of the interpolated implied volatilities for the given strikes.
        """
        if log_mon is not None:
            log_mon = np.array(log_mon)
        elif strikes is not None and F0 is not None:
            log_mon = np.log(np.array(strikes) / F0)
        elif delta_strike is not None:
            log_mon = 0.5 * smile_data.vols**2 * smile_data.ttm - smile_data.vols * np.sqrt(smile_data.ttm) * norm.ppf(delta_strike)
        else:
            raise ValueError("For the option strikes, either log-moneyness, or absolute strikes and spot price,"
                             "or delta-strike should be provided.")
        return self._get_smile(k=log_mon, T=smile_data.ttm, atm_vol=smile_data.atm_vol, contract=smile_data.contract)

    def implied_vol(
        self,
        contract: str,
        atm_vol: float,
        ttm: float = None,
        expiry_date: str = None,
        observation_date: str = None,
        strikes: NDArray[float_] = None,
        log_mon: NDArray[float_] = None,
        F0: float = None,
    ):
        """
        Calculates the implied volatility of the given contract for arbitrary expiry and strike.

        :param contract: contract name.
        :param atm_vol: ATM volatility.
        :param ttm: time to maturity in years.
        :param expiry_date: expiry date in the format "YYY-MM-DD"
        :param observation_date: observation date in the format "YYYY-MM-DD".
        :param strikes: array of absolute strikes.
        :param log_mon: log-moneyness array.
        :param F0: underlying price at t = 0.
        :return: an array of implied volatilities corresponding to the given maturity and strikes.
        """
        if ttm is not None:
            ttm = ttm
        elif expiry_date is not None and observation_date is not None:
            ttm = (np.datetime64(expiry_date) - np.datetime64(observation_date)) / np.timedelta64(365,'D')
        else:
            raise ValueError("For the smile maturity either Time To Maturity (ttm), "
                             "either observation date and expiry date should be given.")

        if log_mon is not None:
            log_mon = np.array(log_mon)
        elif strikes is not None and F0 is not None:
            log_mon = np.log(np.array(strikes) / F0)
        else:
            raise ValueError("For the option strikes, either log-moneyness, or absolute strikes and spot price"
                             " should be provided.")
        return self._get_smile(k=log_mon, T=ttm, atm_vol=atm_vol, contract=contract)

    def fit(
        self,
        smile_data_list: List[SSVISmileData],
        params_to_fit=("rho", "eta", "gamma")
    ) -> None:
        """
        Fit the SSVI model using given smile data.

        :param smile_data_list: list of SSVISmileData objects used in the calibratoins.
        :param params_to_fit: which parameters should be fitted in the calibration routine.
        """
        def loss(x):
            self.set_params(x, params_to_fit)
            res = 0
            for smile_data in smile_data_list:
                res += np.mean((self._get_smile(k=smile_data.log_mon, T=smile_data.ttm,
                                               atm_vol=smile_data.atm_vol, contract=smile_data.contract) -
                               smile_data.vols) ** 2)
            return res
        x0 = []
        bounds = []
        for param in params_to_fit:
            if param == "rho":
                self.rho = dict()
                for idx in sorted(set([smile_data.contract for smile_data in smile_data_list])):
                    self.rho[idx] = RHO_DEFAULT
                    x0.append(self.rho[idx])
                    bounds.append(PARAM_BOUNDS_SSVI["rho"])
            else:
                x0.append(self.__dict__[param])
                bounds.append(PARAM_BOUNDS_SSVI[param])
        res = minimize(fun=loss, x0=x0, method="L-BFGS-B", bounds=bounds)
        self.set_params(res.x, params_to_fit)

    def plot_smiles(
        self,
        smile_data_list: List[SSVISmileData],
        bid_ask_spread=None
    ) -> None:
        """
        Plot the IV smiles and SSVI interpolation corresponding to the given SSVISmileData objects.

        :param smile_data_list: list of SSVISmileData to plot.
        :param bid_ask_spread: bid-ask spread to plot if needed.
        """
        n_plots = len(smile_data_list)
        fig, ax_arr = plt.subplots(n_plots, 1, figsize=(5, 3.5 * n_plots))
        if n_plots == 1:
            ax_arr = [ax_arr]
        fig.tight_layout(pad=3)

        for ax, smile_data in zip(ax_arr, smile_data_list):
            T = smile_data.ttm
            log_mon = smile_data.log_mon
            smile_ssvi = self._get_smile(k=log_mon, T=T, atm_vol=smile_data.atm_vol,
                                        contract=smile_data.contract)
            ax.plot(log_mon, smile_ssvi, color='b', label="SSVI")
            if bid_ask_spread is not None and bid_ask_spread > 0:
                ax.fill_between(
                    log_mon,
                    smile_ssvi - 0.5 * bid_ask_spread,
                    smile_ssvi + 0.5 * bid_ask_spread,
                    color="b",
                    alpha=0.1
                )
            ax.plot(log_mon, smile_data.vols, "vr", label="Market Data")
            ax.set_xlabel(r"$\log(K / F_0)$")
            ax.legend()
            ax.grid()
            ax.set_title(f"Underlying: {smile_data.contract}, TTM = {np.round(T, 2)}")


@dataclass
class SVI:
    delta: float = 0
    mu: float = 0
    rho: float = 0.4
    omega: float = 0.6
    zeta: float = 1

    def get_smile(
        self,
        k: Union[float, NDArray[float_]]
    ) -> Union[float, NDArray[float_]]:
        """
        Calculates the IV in the SSVI parametrization for the given contract.

        :param k: log-moneyness.
        :return: implied volatilities given by the SVI parametrization for the log-moneyness `k`.
        """
        return np.sqrt(self.delta + 0.5 * self.omega * (1 + self.zeta * self.rho * (k - self.mu) +
                       np.sqrt((self.zeta * (k - self.mu) + self.rho) ** 2 + (1 - self.rho ** 2))))

    def fit(
        self,
        k: NDArray[float_],
        smile: NDArray[float_]
    ) -> None:
        """
        Minimizes the MSE between the SVI IV smile and `smile` given as an input and set the optimized parameters
        to the model.

        :param k: array of log-moneyness.
        :param smile: implied volatilities corresponding to `k`.
        """
        if np.array(k).size == 1:
            self.delta = smile[0]**2
            self.omega = 0
        else:
            def loss(x):
                self.delta, self.mu, self.rho, self.omega, self.zeta = x
                return np.sum((self.get_smile(k) - smile) ** 2)

            x0 = np.array([self.delta, self.mu, self.rho, self.omega, self.zeta])
            bounds = [(0, 1), (-3, 3), (-1, 1), (0, 1), (0, 5)]
            res = minimize(fun=loss, x0=x0, method="L-BFGS-B", bounds=bounds)
            self.delta, self.mu, self.rho, self.omega, self.zeta = res.x