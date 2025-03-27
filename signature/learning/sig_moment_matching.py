import numpy as np
from numpy.typing import NDArray
from numpy import float64
from typing import Tuple, List
from scipy.optimize import minimize
from scipy.special import gamma
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from math import factorial

from simulation.diffusion import Diffusion
import matplotlib.pyplot as plt

from signature.old_versions.tensor_sequence import TensorSequence
from signature.old_versions.tensor_algebra import TensorAlgebra
from ..expected_signature import expected_stationary_signature, expected_signature
from ..stationary_signature import stationary_signature_from_path
from ..utility import get_lengths_array
from ..signature_of_signature import get_signature_of_linear_form


class SigSignal:
    trunc: int
    t_grid: NDArray[float64]
    trunc_signature_moments: int
    eSig: None # by default extended Brownian motion
    optimizer: str

    l: TensorSequence = None
    words: List
    ta: TensorAlgebra
    eSig: TensorSequence
    SigS: TensorSequence

    def __init__(
        self,
        trunc,
        T: float,
        trunc_signature_moments: int = 5,
        eSig: TensorSequence = None,
        optimizer="Powell",
    ):
        self.trunc = trunc
        self.trunc_signature_moments = trunc_signature_moments
        self.optimizer = optimizer

        self.ta = TensorAlgebra(dim=2, trunc=self.trunc * self.trunc_signature_moments)

        if eSig is None:
            self.eSig = expected_signature(t=T, trunc=self.trunc * self.trunc_signature_moments)
        else:
            self.eSig = eSig

        n_moments = self.ta.alphabet.number_of_elements(self.trunc_signature_moments)
        words = [self.ta.alphabet.index_to_word(idx) for idx in range(n_moments)]
        self.words = words

    def x_to_ts(self, x):
        array = np.zeros(self.ta.alphabet.number_of_elements(self.trunc))
        array[1:] = x
        return self.ta.from_array(trunc=self.trunc * self.trunc_signature_moments, array=array)

    def fit(self, expected_signal_sig: TensorSequence):
        x0 = np.zeros(self.ta.alphabet.number_of_elements(self.trunc) - 1)

        def callback(x, f=None, context=None, accept=None, convergence=None):
            val = self.loss(x, expected_signal_sig)
            print(f"New iteration: \n x = {x}, \n val={val}. \n")

        res = minimize(lambda x: self.loss(x, expected_signal_sig), x0, method=self.optimizer, callback=callback)
        self.l = self.x_to_ts(res.x)

    def loss(self, x, expected_signal_sig):
        l = self.x_to_ts(x)
        signal_sig_coefs = get_signature_of_linear_form(ts=l, trunc_moments=self.trunc_signature_moments, ta=self.ta)

        loss = 0
        for word in self.words:
            if "2" in word:
                loss_word = np.sqrt(
                    np.mean(np.abs((signal_sig_coefs[word] @ self.eSig).real.squeeze() - expected_signal_sig[word].real.squeeze()) ** 2))
                loss += loss_word * factorial(len(word))

        return loss




class StatSigSignal:
    trunc: int
    lam: float
    t_grid: NDArray[float64]
    window_size: int
    max_stationary_moment: int
    trunc_signature_moments: int
    loss_weights: Tuple
    optimizer: str
    rng: np.random.Generator

    l: TensorSequence = None
    ta: TensorAlgebra
    eSSig: TensorSequence
    SigS: TensorSequence

    def __init__(
        self,
        trunc,
        lam,
        t_grid,
        window_size=100,
        max_stationary_moment=7,
        trunc_signature_moments=5,
        optimizer="BFGS",
        loss_weights=(1, 5e-4),
        rng=None
    ):
        self.trunc = trunc
        self.lam = lam
        self.t_grid = t_grid
        self.window_size = window_size
        self.max_stationary_moment = max_stationary_moment
        self.trunc_signature_moments = trunc_signature_moments
        self.loss_weights = loss_weights
        self.optimizer = optimizer

        self.ta = TensorAlgebra(dim=2, trunc=self.trunc * 2)

        EPS = 1e-8
        n_t_past = 1000
        t_past = np.linspace(np.log(2 * lam * EPS**2) / 2 / lam, 0, n_t_past)
        t_grid_extended = np.concatenate([t_past[:-1] + t_grid[0], t_grid])

        if rng is None:
            self.rng = np.random.default_rng(seed=42)
        else:
            self.rng = rng

        diffusion = Diffusion(t_grid=t_grid_extended - t_grid_extended[0], size=1, rng=self.rng)

        brownian_motion = diffusion.brownian_motion()[0, 0, :]
        path = np.vstack([t_grid_extended, brownian_motion]).T
        self.SigS = stationary_signature_from_path(path=path, trunc=self.trunc, t_grid=t_grid_extended - t_grid[0], lam=self.lam)
        self.eSSig = expected_stationary_signature(trunc=self.trunc * 2, lam=lam)

    def fit(self, signal: NDArray[float64]):
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        normalized_signal = (signal - signal_mean) / signal_std

        signal_moments = (normalized_signal[:, None] ** np.arange(3, 7)).mean(axis=0)

        num_of_win = len(self.t_grid) - self.window_size
        path_signal = np.empty((self.window_size, 2, num_of_win))
        path_signal[:, 0, :] = np.tile(self.t_grid[:self.window_size][:, None], num_of_win)
        for idx_start in range(num_of_win):
            path_signal[:, 1, idx_start] = normalized_signal[idx_start:idx_start + self.window_size]

        SignalSig = self.ta.path_to_sequence(path=path_signal, trunc=self.trunc_signature_moments)
        e_signal_sig = SignalSig.array[:, -1, :].mean(axis=1).real

        rng = np.random.default_rng(seed=42)
        if self.trunc == 3:
            x_size = 8
        elif self.trunc == 2:
            x_size = 4
        else:
            raise NotImplementedError("Implement properly the number of linearly independent elements!")
        x0 = rng.normal(size=x_size) * 0.01

        def callback(x, f=None, context=None, accept=None, convergence=None):
            val = self.loss(x, e_signal_sig=e_signal_sig, signal_moments=signal_moments, verbose=True)
            print(f"New iteration: \n x = {x}, \n val={val}. \n")

        def calib_loss(x):
            return self.loss(x, e_signal_sig=e_signal_sig, signal_moments=signal_moments)

        res = minimize(calib_loss, x0, callback=callback, method=self.optimizer)

        print(res)

        self.l = self.normalize_ts(ts=self.ts_from_x(res.x), loc=signal_mean, scale=signal_std)

    def ts_from_x(self, x):
        mask = np.array([0, 2, 4, 6, 8, 10, 12, 14])
        number_of_elements = self.ta.alphabet.number_of_elements(self.trunc)
        array = np.zeros(number_of_elements)
        array[mask[mask < number_of_elements]] = x
        return self.ta.from_array(trunc=self.trunc, array=array)

    def normalize_ts(self, ts: TensorSequence, loc: float = 0, scale: float = 1):
        l_mean = (self.eSSig @ ts).squeeze().real
        l_second_moment = (self.eSSig @ self.ta.shuop.shuffle_prod(ts, ts)).squeeze().real
        l_std = np.sqrt(l_second_moment - l_mean ** 2)

        new_array = ts.array
        new_array[0] += -l_mean
        new_array = new_array / l_std * scale
        new_array[0] += loc
        return self.ta.from_array(trunc=self.trunc, array=new_array)

    def empirical_expected_signal_sig(self, signal: NDArray[float64]):
        num_of_win = len(self.t_grid) - self.window_size
        path_signal = np.empty((self.window_size, 2, num_of_win))
        path_signal[:, 0, :] = np.tile(self.t_grid[:self.window_size][:, None], num_of_win)
        for idx_start in range(num_of_win):
            path_signal[:, 1, idx_start] = signal[idx_start:idx_start + self.window_size]

        SignalSig = self.ta.path_to_sequence(path=path_signal, trunc=self.trunc_signature_moments)

        # possibly add a tensor normalization here

        return SignalSig.array[:, -1, :].mean(axis=1).squeeze().real

    def loss(self, x, e_signal_sig, signal_moments, verbose=False):
        l = self.ts_from_x(x)
        l = self.normalize_ts(l)

        signal_l = (l @ self.SigS).squeeze().real
        e_signal_sig_l = self.empirical_expected_signal_sig(signal_l)

        signal_l_moments = (signal_l[:, None] ** np.arange(3, self.max_stationary_moment)).mean(axis=0)

        weight_level = gamma(get_lengths_array(self.ta.alphabet, trunc=self.trunc_signature_moments) + 1)
        loss_esig = np.sqrt(np.mean(((e_signal_sig - e_signal_sig_l) * weight_level) ** 2)) * self.loss_weights[0]
        # print("loss:", ((e_signal_sig - e_signal_sig_l)) ** 2)
        # print("loss (weighted):", ((e_signal_sig - e_signal_sig_l) * weight_level) ** 2)
        loss_higher_moments = np.sqrt(np.mean((signal_l_moments - signal_moments) ** 2)) * self.loss_weights[1]

        if verbose:
            print(f"Esig: {loss_esig}, Stationary moments: {loss_higher_moments}")

        return loss_esig + loss_higher_moments

    def analysis(self, signal, window_size=1000):
        if self.l is not None:
            l_SigS = (self.l @ self.SigS).squeeze().real

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        ax[0][0].plot(self.t_grid[:window_size], signal[:window_size], label="Signal")
        if self.l is not None:
            ax[0][0].plot(self.t_grid[:window_size], l_SigS[:window_size], "--", label="<l*, W_t>")
        ax[0][0].set_title("Sample paths")
        ax[0][0].legend()

        plot_acf(signal, ax=ax[0][1], label="Signal")
        if self.l is not None:
            plot_acf(l_SigS, ax=ax[0][1], label="<l*, W_t>")
        ax[0][1].legend()
        ax[0][1].set_ylim([-1.05, 1.05])

        ax[1][0].hist(signal, bins=100, alpha=0.5, density=True, label="Signal")
        if self.l is not None:
            ax[1][0].hist(l_SigS, bins=100, alpha=0.5, density=True, label="<l*, W_t>")
        ax[1][0].set_title("Stationary distribution histogram")
        ax[1][0].legend()

        sm.qqplot(signal, ax=ax[1][1], label="signal")
        if self.l is not None:
            color_2 = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
            sm.qqplot(l_SigS, ax=ax[1][1], marker="v", markerfacecolor=color_2, markeredgecolor=color_2, label="fit")
        ax[1][1].set_title("Q-Q Plot")

        plt.show()
