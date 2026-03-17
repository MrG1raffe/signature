import jax
import jax.numpy as jnp
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import Union
from abc import ABC, abstractmethod

from signature.path_signature import path_to_signature, path_to_rolling_signature, path_to_fm_signature
from signature.learning.lead_lag import lead_lag_transform, efm_lead_lag_transform

import jax
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class GenericSignatureTransform(ABC, BaseEstimator, TransformerMixin):
    def __init__(self, trunc: int, burn_in: int = 0, time_flag: bool = False, lead_lag: bool = False, lead_lag_idx: jax.Array = None):
        self.trunc = trunc
        self.lead_lag = lead_lag
        self.burn_in = burn_in
        self.time_flag = time_flag # Whether the first column of X corresponds to the time_grid (not included in path).
        self.lead_lag_idx = lead_lag_idx

    def preprocess_path(self, X: jax.Array) -> jax.Array:
        if self.lead_lag and not self.time_flag:
            X = lead_lag_transform(X, lead_lag_idx=self.lead_lag_idx)
        elif self.lead_lag and self.time_flag:
            X = np.hstack(efm_lead_lag_transform(t_grid=X[:, 0], path=X[:, 1:], lead_lag_idx=self.lead_lag_idx))
        return X

    def postprocess_output(self, sig: jax.Array) -> jax.Array:
        if self.lead_lag and not self.time_flag:
            sig = sig[::2]
        elif self.lead_lag and self.time_flag:
            sig = sig[::3]
        return sig

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    @abstractmethod
    def _compute_signature(self, X_jax: jax.Array) -> jax.Array:
        """Subclasses only implement the core signature logic here."""
        pass

    def transform(self, X):
        check_is_fitted(self)

        # 1. Standardize and Preprocess Path
        X_jax = jnp.asarray(X)
        X_pre = self.preprocess_path(X_jax)

        # 2. Compute Signature (Specific to Child)
        # We expect .array.T logic to be handled here or in the child
        sigs = self._compute_signature(X_pre)
        # 3. Postprocess and Convert to NumPy
        sigs_post = self.postprocess_output(sigs)
        return np.array(sigs_post)[self.burn_in:]


class SignatureTransform(GenericSignatureTransform):
    def _compute_signature(self, X_jax: jax.Array) -> jax.Array:
        return path_to_signature(path=X_jax, trunc=self.trunc).array.T


class RollingSignatureTransform(GenericSignatureTransform):
    def __init__(self, trunc: int, window_size: int, lead_lag: bool = False, burn_in: int = 0, lead_lag_idx: jax.Array = None):
        """
        Scikit-Learn wrapper for JAX-based rolling signature computation.

        :param trunc: Truncation level of the signature.
        :param window_size: Window size of the rolling window.
        """
        super().__init__(trunc=trunc, lead_lag=lead_lag, burn_in=burn_in, lead_lag_idx=lead_lag_idx)
        self.window_size = window_size

    def _compute_signature(self, X_jax: jax.Array) -> jax.Array:
        return path_to_rolling_signature(
            path=X_jax, trunc=self.trunc, window_size=self.window_size
        ).array.T


class EFMSignatureTransform(GenericSignatureTransform):
    def __init__(self, trunc: int, lam: Union[float, np.ndarray], lead_lag: bool = False, burn_in: int = 0, lam_idx: jax.Array = None, lead_lag_idx: jax.Array = None):
        """
        Scikit-Learn wrapper for JAX-based EFM-signature computation. The first column of X must correspond to
        the time grid, while the signature transform is applied to X[:, 1:].

        :param trunc: Truncation level of the signature.
        :param lam: Mean-revrsion parameters of the EFM-signature.
        """
        super().__init__(trunc=trunc, lead_lag=lead_lag, burn_in=burn_in, time_flag=True, lead_lag_idx=lead_lag_idx)
        self.lam_idx = lam_idx
        self.lam = np.asarray(lam)

    def _compute_signature(self, X_jax: jax.Array) -> jax.Array:
        dim = X_jax.shape[1] - 1 # the first column corresponds to the time grid and is not included in the path.
        if self.lam_idx is not None:
            lam_arr = self.lam[self.lam_idx]
        elif self.lam.size == 1:
            lam_arr = self.lam * jnp.ones(dim)
        else:
            lam_arr = self.lam

        if lam_arr.size != dim:
            raise ValueError(f'lam size must be 1 or {dim}, but is {self.lam.size}')

        return path_to_fm_signature(
            path=X_jax[:, 1:],
            trunc=self.trunc,
            t_grid=X_jax[:, 0],
            lam=jnp.asarray(lam_arr)
        ).array.T
