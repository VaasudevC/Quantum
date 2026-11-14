"""PLS-8 preprocessing for the 8-qubit hybrid VQC.

Classical preprocessing pipeline (runs on CPU / classical control computer):
    1. StandardScaler: zero-mean, unit-variance on all 251 ECG features.
    2. PLSRegression(n_components=8): 251-D -> 8 maximally label-predictive components.
       PLS-8 + LR baseline: 88.91% accuracy, AUC=0.9481.
    3. Angle encoding: angle_i = pi * tanh(pls_component_i), in (-pi, pi).
    4. Data re-uploading: tile (batch, 8) -> (batch, 32), same 8 angles at each of 4 layers.

Quantum circuit (unchanged from Vqc_hybrid8 / Run 7):
    8 qubits, 4 layers, Ry+Rz+CNOT ring, 72 ansatz params.
    MLP readout: 8 -> 32 -> 1, 321 params.
    Total: 393 params.

Hardware deployment:
    Classical: 251-feature ECG -> PLS matrix multiply -> 8 angles.
    QPU: 8 angles -> circuit -> 8 Z-expectation values.
    Classical: 8 expectations -> MLP -> p(MI).
    The PLS matrix (251x8) is a fixed linear transform computed once at training.
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from Vqc_angle8 import N_QUBITS, N_LAYERS
from Vqc_hybrid8 import N_PARAMS, predict_proba, evaluate, train_adam, EvalResult  # noqa: F401

N_PLS_COMPONENTS = N_QUBITS  # 8


class PLSPreprocessor:
    """StandardScaler + PLSRegression.

    Stores a combined linear transform A, b so that T = X @ A + b.
    This avoids sklearn version dependencies at inference time and is
    directly translatable to a matrix multiply on classical control hardware.
    """

    def __init__(self, n_components: int = N_PLS_COMPONENTS):
        self.n_components = n_components
        self._scaler = StandardScaler()
        self._pls    = PLSRegression(n_components=n_components, max_iter=1000)
        self.A: np.ndarray | None = None  # (n_features, n_components)
        self.b: np.ndarray | None = None  # (n_components,)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSPreprocessor":
        Xs = self._scaler.fit_transform(X)          # (n, 251)
        self._pls.fit(Xs, y)
        # Precompute combined transform: T = X @ A + b
        # StandardScaler: Xs = (X - mu_sc) / sig_sc
        # PLS.transform : T  = (Xs - mu_pls) / std_pls @ R  (sklearn PLSRegression)
        # Combined      : T  = X @ A + b
        mu_sc  = self._scaler.mean_              # (251,)
        sig_sc = self._scaler.scale_             # (251,)
        R      = self._pls.x_rotations_          # (251, 8)
        mu_pls = self._pls._x_mean               # (251,)  (on Xs space)
        std_pls = self._pls._x_std               # (251,)

        # Xs = (X - mu_sc) / sig_sc
        # Xp = (Xs - mu_pls) / std_pls
        # T  = Xp @ R
        # T  = ((X - mu_sc)/sig_sc - mu_pls) / std_pls @ R
        # T  = X @ (R / (sig_sc * std_pls)[:, None])
        #      - (mu_sc / sig_sc + mu_pls) / std_pls @ R
        denom = sig_sc * std_pls                  # (251,)
        denom = np.where(denom == 0, 1.0, denom)
        A1 = R / denom[:, None]
        b1 = -((mu_sc / sig_sc + mu_pls) / std_pls) @ R

        # Normalize each PLS component to unit-std on training data to prevent
        # tanh saturation at +-pi, which makes Ry(+pi) and Ry(-pi) indistinguishable.
        T_train   = X @ A1 + b1
        comp_mean = T_train.mean(axis=0)
        comp_std  = T_train.std(axis=0)
        comp_std  = np.where(comp_std == 0, 1.0, comp_std)
        self.A = A1 / comp_std[None, :]
        self.b = (b1 - comp_mean) / comp_std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return PLS components: T = X @ A + b, shape (n, 8)."""
        return X @ self.A + self.b

    def to_dict(self) -> dict:
        return {
            "n_components": self.n_components,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PLSPreprocessor":
        obj = cls(n_components=d["n_components"])
        obj.A = np.array(d["A"])
        obj.b = np.array(d["b"])
        return obj


def compute_pls_angles(pls_components: np.ndarray) -> np.ndarray:
    """Map PLS-8 components to (batch, 32) angles with data re-uploading.

    angle_i = pi * tanh(pls_i) in (-pi, pi).
    The same 8 angles are re-uploaded at each of 4 circuit layers.
    Returns: (batch, 32) compatible with circuit_forward.
    """
    angles_8 = np.pi * np.tanh(pls_components)  # (batch, 8)
    return np.tile(angles_8, N_LAYERS)            # (batch, 32)
