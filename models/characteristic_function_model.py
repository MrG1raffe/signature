from abc import abstractmethod
import numpy as np
from numpy.typing import NDArray
from numpy import float_, complex_
from scipy.special import roots_laguerre
from typing import Union
from dataclasses import dataclass


from models.model import Model
from utility.utility import is_put, is_call, to_numpy
from volatility_surface.volatility_surface import black_iv, black_vanilla_price


@dataclass
class CharacteristicFunctionModel(Model):
    """
    An abstract class to price options in the models with the characteristic function available.

    model_type: whether the underlying price follows normal or log-normal dynamics. Either "normal" or "log-normal".
    r: risk-free rate. By default, equal to zero.
    """
    model_type: str

    @abstractmethod
    def _char_func(
            self,
            T: float,
            x: float,
            u1: complex,
            u2: complex = 0,
            f1: complex = 0,
            f2: complex = 0,
            **kwargs
    ) -> complex_:
        """
        Computes the generalized characteristic function

        E[exp{i * u1 * X_T + i * u2 * V_T + i * f1 * ∫ X_s ds + i * f2 * ∫ V_s ds}]     (1)

        for the given model, where X_t = F_t if `model_type` == "normal" and
        X_t = log(F_t) if `model_type` == "log-normal".

        :param u1: X_T coefficient in the characteristic function, see (1).
        :param u2: V_T coefficient in the characteristic function, see (1).
        :param f1: ∫ X_s ds coefficient in the characteristic function, see (1).
        :param f2: ∫ V_s ds coefficient in the characteristic function, see (1).
        :param T: date in the characteristic function, see (1).
        :param x: X_0, equals to F_0 if `model_type` == "normal" and to log(F_0) if `model_type` == "log-normal".
        :return: a value of the characteristic function (1) for the given coefficients.
        """
        raise NotImplementedError()

    @abstractmethod
    def quadratic_variation(
            self,
            T: float,
    ) -> float:
        """
        Computes the quadratic variation of the process <X>_T at t = T.

        :param T: the date the quadratic variation to be calculated on.
        :return: the value of numerical approximation of <X>_T.
        """
        raise NotImplementedError

    def _get_truncation_bounds(
            self,
            F0: float,
            T: float
    ) -> tuple:
        """
        Calculates truncation bounds for the COS-method based on the quadratic variation of (log)price.
        For the normal model the interval is centered around F0 since the

        :param F0: initial spot value.
        :param T: maturity to be used in the COS-method.
        :return: a tuple, containing truncation bounds for the COS-method.
        """
        if self.model_type == "normal":
            c1, c2 = F0, self.quadratic_variation(T)
            a, b = c1 - 12 * np.sqrt(c2), c1 + 12 * np.sqrt(c2)
        elif self.model_type == "log-normal":
            # Truncation bounds for log(F_T / K)
            total_variance = self.quadratic_variation(T)
            c1, c2 = - 0.5 * total_variance, total_variance
            a, b = c1 - 10 * np.sqrt(c2), c1 + 10 * np.sqrt(c2)
        else:
            raise ValueError("`model_type` should be either 'normal' or 'log-normal'.")
        return a, b

    @staticmethod
    def _psi_cos(
            N_trunc: int,
            a: float,
            b: float,
            c: Union[float, NDArray[float_]],
            d: Union[float, NDArray[float_]]
    ) -> NDArray[float_]:
        """
        Calculates ∫ cos(π * k * (y - a) / (b - a)) dy, k = 0, 1, ..., `N_trunc` - 1,
        where the integration limits are given by `c` and `d`.

        :param N_trunc: number of coefficients to be calculated.
        :param a: lower Cosine transform truncation bound.
        :param b: upper Cosine transform truncation bound.
        :param c: lower integral limits, a number or an array.
        :param d: upper integral limits, a number or an array.
        :return: an array of calculated integrals indexed by k.
        """
        psi = np.zeros((N_trunc, max(np.array(c).size, np.array(d).size)))
        psi[0] = d - c
        interval_normalization = np.reshape(np.arange(1, N_trunc), (-1, 1)) * np.pi / (b - a)
        psi[1:] = (np.sin(interval_normalization * (d - a)) -
                   np.sin(interval_normalization * (c - a))) / interval_normalization
        return psi

    def _chi_cos(
            self,
            N_trunc: int,
            a: float,
            b: float,
            c: Union[float, NDArray[float_]],
            d: Union[float, NDArray[float_]]
    ) -> NDArray[float_]:
        """
        Calculates ∫ y cos(π * k * (y - a) / (b - a)) dy, k = 0, 1, ..., `N_trunc` - 1, if model_type == "normal" and
        ∫ exp(y) * cos(π * k * (y - a) / (b - a)) dy, k = 0, 1, ..., `N_trunc` - 1, if model_type == "log-normal",
        where the integration limits are given by `c` and `d`.

        :param N_trunc: number of coefficients to be calculated.
        :param a: lower Cosine transform truncation bound.
        :param b: upper Cosine transform truncation bound.
        :param c: lower integral limits, a number or an array.
        :param d: upper integral limits, a number or an array.
        :return: an array of calculated integrals indexed by k.
        """
        if self.model_type == "normal":
            chi = np.zeros((N_trunc, max(np.array(c).size, np.array(d).size)))
            chi[0] = 0.5 * (d ** 2 - c ** 2)
            interval_normalization = np.reshape(np.arange(1, N_trunc), (-1, 1)) * np.pi / (b - a)
            chi[1:] = (d * np.sin(interval_normalization * (d - a)) -
                       c * np.sin(interval_normalization * (c - a))) / interval_normalization + \
                      (np.cos(interval_normalization * (d - a)) -
                       np.cos(interval_normalization * (c - a))) / interval_normalization ** 2
        elif self.model_type == "log-normal":
            interval_normalization = np.reshape(np.arange(N_trunc), (-1, 1)) * np.pi / (b - a)
            chi = (np.exp(d) * np.cos(interval_normalization * (d-a)) -
                   np.exp(c) * np.cos(interval_normalization * (c-a)) +
                   np.exp(d) * interval_normalization * np.sin(interval_normalization * (d-a)) -
                   np.exp(c) * interval_normalization * np.sin(interval_normalization * (c-a))) / \
                  (1 + interval_normalization**2)
        else:
            raise ValueError("`model_type` should be either 'normal' or 'log-normal'.")
        return chi

    def get_vanilla_option_price_cos(
            self,
            T: Union[float, NDArray[float_]],
            K: Union[float, NDArray[float_]],
            F0: float,
            flag: str = "call",
            is_vol_surface: bool = False,
            N_trunc: int = 200,
            **kwargs
    ) -> Union[float, NDArray[float_]]:
        """
        Calculates the prices of the European vanilla options via the COS-method.

        :param T: option maturities, a number or a 1-dimensional array.
        :param K: options strikes. Either a number, or a 1D array of strikes, or a 2D array of shape
            (len(T), len(strikes)) containing in the i-th raw the strikes corresponding to maturity T[i].
        :param F0: initial value of the underlying price.
        :param flag: determines the option type: "c" or "call" for calls, "p" or "put" for puts.
        :param is_vol_surface: whether to return the Black implied volatility value instead of option prices.
        :param N_trunc: number of terms in the Cosine series to be calculated.
        :param cf_timestep: a timestep to be used in numerical scheme in the characteristic function.
        :return: an array of shape (T.size, K.shape[-1]) with the option prices or implied vols.
        """
        if self.model_type == "normal":
            if is_call(flag):
                def get_u_cos(K: Union[float, NDArray[float_]], N_trunc: int, a: float, b: float):
                    return (2 / (b - a) * self._chi_cos(N_trunc=N_trunc, a=a, b=b, c=K, d=b),
                            -2 / (b - a) * self._psi_cos(N_trunc=N_trunc, a=a, b=b, c=K, d=b))
            elif is_put(flag):
                def get_u_cos(K: Union[float, NDArray[float_]], N_trunc: int, a: float, b: float):
                    return (-2 / (b - a) * self._chi_cos(N_trunc=N_trunc, a=a, b=b, c=a, d=K),
                            2 / (b - a) * self._psi_cos(N_trunc=N_trunc, a=a, b=b, c=a, d=K))
            else:
                raise ValueError("Invalid value of flag.")
        elif self.model_type == "log-normal":
            if is_call(flag):
                def get_u_cos(K: Union[float, NDArray[float_]], N_trunc: int, a: float, b: float):
                    return (2 / (b - a) * self._chi_cos(N_trunc=N_trunc, a=a, b=b, c=0, d=b) -
                            2 / (b - a) * self._psi_cos(N_trunc=N_trunc, a=a, b=b, c=0, d=b))
            elif is_put(flag):
                def get_u_cos(K: Union[float, NDArray[float_]], N_trunc: int, a: float, b: float):
                    return (-2 / (b - a) * self._chi_cos(N_trunc=N_trunc, a=a, b=b, c=a, d=0) +
                            2 / (b - a) * self._psi_cos(N_trunc=N_trunc, a=a, b=b, c=a, d=0))
            else:
                raise ValueError("Invalid value of flag.")
        else:
            ValueError("`model_type` should be either 'normal' or 'log-normal'.")

        T = np.reshape(T, (-1))
        if T.shape != (T.size,):
            raise ValueError("`T` should be a float or a one-dimensional array.")
        K = to_numpy(K)
        prices = np.zeros((T.size, K.shape[-1]))
        for i, maturity in enumerate(T):
            a, b = self._get_truncation_bounds(F0=F0, T=maturity)
            strikes = K[i] if len(K.shape) == 2 else K

            def char_func(u: complex):
                return self._char_func(u1=u, u2=0, f1=0, f2=0, T=maturity, x=0, **kwargs)

            if self.model_type == "normal":
                u, v = get_u_cos(strikes, N_trunc, a, b)
                cf_array = np.zeros(N_trunc, dtype=complex)
                for k in range(N_trunc):
                    cf_array[k] = char_func(k * np.pi / (b - a))
                u[0] /= 2
                v[0] /= 2
                real_part = np.real(cf_array * np.exp(1j * np.arange(N_trunc) * np.pi * (F0 - a) / (b - a)))
                res = real_part @ u + (real_part @ v) * strikes
                prices[i] = res
            else:
                u = get_u_cos(strikes, N_trunc, a, b)
                cf_array = np.zeros(N_trunc, dtype=complex)
                for k in range(N_trunc):
                    cf_array[k] = char_func(k * np.pi / (b - a))
                u[0] /= 2
                real_part = np.real(cf_array * np.exp(1j * np.arange(N_trunc) * np.pi *
                                                    (np.log(F0 / strikes)[:, None] - a) / (b - a)))
                res = real_part @ np.squeeze(u)
                prices[i] = res * strikes
            if is_vol_surface:
                prices[i] = black_iv(option_price=prices[i], T=maturity, K=strikes, F=F0, r=0, flag=flag)
        return prices.squeeze()

    def get_vanilla_option_price_lewis(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        F0: float,
        flag: str = "call",
        is_vol_surface: bool = False,
        N_points: int = 30,
        control_variate_sigma: float = 0.4,
        **kwargs
    ):
        if self.model_type == "normal":
            raise NotImplementedError()

        T = np.reshape(T, (-1))
        if T.shape != (T.size,):
            raise ValueError("`T` should be a float or a one-dimensional array.")
        K = to_numpy(K)
        prices = np.zeros((T.size, K.shape[-1]))

        def black_cf(u: complex, T: float):
            return np.exp(-0.5 * control_variate_sigma**2 * (u**2 + 1j * u) * T)

        def integrand(z: complex, T: float, k: NDArray[float_]):
            return (np.exp(1j * (z - 1j / 2) * k) * (self._char_func(u1=z - 1j / 2, T=T, x=0, **kwargs) -
                    black_cf(u=z - 1j / 2, T=T)) / (z ** 2 + 0.25)).real
        
        for i, maturity in enumerate(T):
            strikes = K[i] if len(K.shape) == 2 else K
            k = np.log(F0 / strikes)
            z_arr, w_arr = roots_laguerre(n=N_points)
            try:
                z_arr = np.reshape(z_arr, (-1, 1))
                integrand_arr = (np.exp(1j * (z_arr - 1j / 2) * k.reshape((1, -1))) * (
                        self._char_func(u1=z_arr - 1j / 2, T=maturity, x=0, **kwargs).reshape((-1, 1)) -
                        black_cf(u=z_arr - 1j / 2, T=maturity)
                ) / (z_arr ** 2 + 0.25)).real
                integral = (w_arr * np.exp(z_arr.squeeze())) @ integrand_arr
            except ValueError as e:
                integral = (w_arr * np.exp(z_arr)) @ np.array([integrand(z, maturity, k) for z in z_arr])
                raise e  # to remove if the model with non-vectorized CF will be used
            prices[i] = black_vanilla_price(sigma=control_variate_sigma, T=maturity, K=strikes, F=F0, r=0, flag='c') - \
                        strikes / np.pi * integral
            if is_put(flag):
                prices[i] += strikes - F0
            if is_vol_surface:
                prices[i] = black_iv(option_price=prices[i], T=maturity, K=strikes, F=F0, r=0, flag=flag)
        return prices.squeeze()
