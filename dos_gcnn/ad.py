"""Applicability-domain runtime for Transformer DOS-GCNN inference.

Faithful inference-compatible replicas of the classes pickled by
``spectral_applicability_domain_v2`` / ``spectral_applicability_domain_transformer_v3``.
A custom Unpickler remaps objects pickled while those scripts ran as ``__main__``.

The Transformer AD model was calibrated on 370-D atom-wise embeddings taken
AFTER the shared graph encoder (Linear 205->370, 3 x TransformerConv, 3 x
GC_block) and immediately BEFORE the s/p/d/f decoders. Scoring must use that
post-encoder hidden state, never raw ``data.x``.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

import numpy as np


AD_QUANTS = ("s", "p", "d", "f")
AD_SPARSE_QUANTS = ("d", "f")
AD_ESTIMATORS = ("knn", "kde", "lof")


class ADConstantCalibrator:
    def __init__(self, value=0.5):
        self.value = float(value)

    def predict(self, x):
        return np.full(len(np.asarray(x)), self.value, dtype=np.float64)


class ADEmbeddingSpace:
    def __init__(self, pca_dim=32, random_state=42):
        self.pca_dim = int(pca_dim)
        self.random_state = int(random_state)
        self.scaler = None
        self.pca = None

    def transform_rows(self, rows):
        if self.scaler is None or self.pca is None:
            raise RuntimeError("AD embedding space is incomplete")
        x = np.asarray(rows, dtype=np.float32)
        xs = self.scaler.transform(x)
        return self.pca.transform(xs).astype(np.float32, copy=False)


class ADKNNAD:
    def __init__(self, k=5, atom_quantile=0.90, n_jobs=-1):
        self.k = int(k)
        self.atom_quantile = float(atom_quantile)
        self.nn = None

    def score_rows(self, z):
        dist, _ = self.nn.kneighbors(z, return_distance=True)
        per_atom = dist.mean(axis=1)
        return float(np.quantile(per_atom, self.atom_quantile))


class ADKDEAD:
    def __init__(self, bandwidth=0.2, kde_dims=8, atom_quantile=0.90):
        self.bandwidth = float(bandwidth)
        self.kde_dims = int(kde_dims)
        self.atom_quantile = float(atom_quantile)
        self.kde = None

    def score_rows(self, z):
        d = min(self.kde_dims, z.shape[1])
        nll = -self.kde.score_samples(z[:, :d])
        return float(np.quantile(nll, self.atom_quantile))


class ADLOFAD:
    def __init__(self, n_neighbors=20, atom_quantile=0.90, n_jobs=-1):
        self.n_neighbors = int(n_neighbors)
        self.atom_quantile = float(atom_quantile)
        self.model = None

    def score_rows(self, z):
        novelty = -self.model.score_samples(z)
        return float(np.quantile(novelty, self.atom_quantile))


class ADADBundle:
    def __init__(self, space=None, knn=None, kde=None, lof=None,
                 calibrators=None, ensemble_weights=None, taus=None, settings=None):
        self.space = space
        self.knn = knn
        self.kde = kde
        self.lof = lof
        self.calibrators = calibrators or {}
        self.ensemble_weights = ensemble_weights or {}
        self.taus = taus or {}
        self.settings = settings or {}

    def predict_from_z(self, z_list, activity=None):
        raw = {
            "knn": np.array([self.knn.score_rows(z) for z in z_list], dtype=np.float64),
            "kde": np.array([self.kde.score_rows(z) for z in z_list], dtype=np.float64),
            "lof": np.array([self.lof.score_rows(z) for z in z_list], dtype=np.float64),
        }
        out = {}
        for name, score in raw.items():
            out[f"raw_{name}"] = score
            for qname in (*AD_QUANTS, "all"):
                p = np.asarray(self.calibrators[name][qname].predict(score), dtype=np.float64)
                if activity is not None and qname in AD_SPARSE_QUANTS:
                    known_active = np.asarray(activity[qname], dtype=bool)
                    p = p.copy()
                    p[~known_active] = 1.0
                out[f"p_{name}_{qname}"] = p

        for qname in (*AD_QUANTS, "all"):
            p = np.zeros(len(z_list), dtype=np.float64)
            for name in AD_ESTIMATORS:
                p += float(self.ensemble_weights[qname][name]) * out[f"p_{name}_{qname}"]
            if activity is not None and qname in AD_SPARSE_QUANTS:
                p = p.copy()
                p[~np.asarray(activity[qname], dtype=bool)] = 1.0
            out[f"p_ensemble_{qname}"] = np.clip(p, 0.0, 1.0)

        # Match the final v2 implementation: the object-level score is the mean
        # of channel-level ensemble correctness probabilities.
        ch = np.vstack([out[f"p_ensemble_{q}"] for q in AD_QUANTS]).T
        out["p_ensemble_all"] = np.nanmean(ch, axis=1)
        return out


class _ADCompatUnpickler(pickle.Unpickler):
    _main_map = {
        "ADBundle": ADADBundle,
        "EmbeddingSpace": ADEmbeddingSpace,
        "KNNAD": ADKNNAD,
        "KDEAD": ADKDEAD,
        "LOFAD": ADLOFAD,
        "ConstantCalibrator": ADConstantCalibrator,
    }

    def find_class(self, module, name):
        if module == "__main__" and name in self._main_map:
            return self._main_map[name]
        # Also accept pickles created after importing the calibration module.
        if (
            module.endswith("spectral_applicability_domain_v2")
            or module.endswith("spectral_applicability_domain_transformer_v3")
        ) and name in self._main_map:
            return self._main_map[name]
        return super().find_class(module, name)


def load_dos_ad_bundle(path):
    path = os.path.abspath(os.fspath(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"AD model not found: {path}")
    with open(path, "rb") as f:
        magic = f.read(2)
        f.seek(0)
        if magic == b"PK":
            raise TypeError(
                f"{path} is a ZIP/PyTorch checkpoint, not an ADBundle pickle. "
                "Place the ADBundle pickle from spectral_applicability_domain_transformer_v3."
            )
        bundle = _ADCompatUnpickler(f).load()
    required = ("space", "knn", "kde", "lof", "calibrators", "ensemble_weights")
    missing = [name for name in required if not hasattr(bundle, name)]
    if missing:
        raise TypeError(f"Invalid AD bundle; missing attributes: {missing}")
    return bundle


def _ad_expected_input_dim(bundle):
    scaler = bundle.space.scaler
    if hasattr(scaler, "n_features_in_"):
        return int(scaler.n_features_in_)
    if hasattr(scaler, "mean_") and scaler.mean_ is not None:
        return int(np.asarray(scaler.mean_).shape[0])
    return None


def validate_ad_embedding(embedding, bundle, *, save_path=None):
    """Validate the post-encoder Transformer embedding used to calibrate AD."""
    if hasattr(embedding, "detach"):
        emb = embedding.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        emb = np.asarray(embedding, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"AD embedding must be 2D [n_atoms,n_features], got {emb.shape}")
    expected = _ad_expected_input_dim(bundle)
    if expected is not None and emb.shape[1] != expected:
        raise ValueError(
            "AD embedding dimensionality mismatch: current post-encoder "
            f"embedding has {emb.shape[1]} features/atom, whereas ad_model.pkl "
            f"expects {expected}."
        )
    if emb.shape[1] != 370:
        raise ValueError(
            f"Expected the Transformer post-encoder embedding dimension 370, "
            f"got {emb.shape[1]}."
        )
    if not np.all(np.isfinite(emb)):
        raise ValueError("Non-finite values found in AD embedding")
    if save_path is not None:
        save_path = os.fspath(save_path)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        np.save(save_path, emb)
    return emb


def validate_transformer_ad_bundle(bundle):
    """Fail early if an AD model is incompatible with the Transformer latent space."""
    expected = _ad_expected_input_dim(bundle)
    if expected is not None and expected != 370:
        raise RuntimeError(
            f"AD model expects {expected} features/atom; the Transformer checkpoint "
            "produces 370-D post-encoder embeddings."
        )
    settings = getattr(bundle, "settings", {}) or {}
    provenance = str(settings.get("embedding_provenance", "")).lower()
    if provenance and ("transformer" not in provenance or "post-encoder" not in provenance):
        raise RuntimeError(
            "The AD bundle embedding provenance does not match the Transformer "
            "post-encoder latent space: " + str(settings.get("embedding_provenance"))
        )


def _contains_f_element(atomic_numbers):
    z = np.asarray(atomic_numbers, dtype=np.int64).ravel()
    return bool(np.any(((57 <= z) & (z <= 71)) | ((89 <= z) & (z <= 103))))


def _contains_d_block_element(atomic_numbers):
    z = np.asarray(atomic_numbers, dtype=np.int64).ravel()
    return bool(np.any(
        ((21 <= z) & (z <= 30)) |
        ((39 <= z) & (z <= 48)) |
        ((72 <= z) & (z <= 80)) |
        ((104 <= z) & (z <= 112))
    ))


def infer_sparse_channel_activity(atomic_numbers, pred_d=None, pred_f=None,
                                  policy="element_block", threshold=1e-10):
    """
    Return object-level d/f activity for target-free inference.

    element_block: deterministic structure-only production rule. This is the
                   recommended default for the final AD deployment.
    predicted:     derive activity from predicted d/f spectra using threshold.
    none:          activity is unknown; do not apply deterministic-correct shortcut.
    """
    policy = str(policy).lower()
    if policy == "none":
        return None
    if policy == "element_block":
        return {
            "d": np.array([_contains_d_block_element(atomic_numbers)], dtype=bool),
            "f": np.array([_contains_f_element(atomic_numbers)], dtype=bool),
        }
    if policy == "predicted":
        if pred_d is None or pred_f is None:
            raise ValueError("predicted activity policy requires pred_d and pred_f")
        active_d = bool(np.max(np.abs(np.asarray(pred_d))) > float(threshold))
        active_f = bool(np.max(np.abs(np.asarray(pred_f))) > float(threshold))
        return {
            "d": np.array([active_d], dtype=bool),
            "f": np.array([active_f], dtype=bool),
        }
    raise ValueError(f"Unknown AD sparse activity policy: {policy}")


def score_dos_ad(bundle, embedding, *, activity=None, threshold=0.5):
    z = bundle.space.transform_rows(embedding)
    pred = bundle.predict_from_z([z], activity=activity)
    row = {key: float(np.asarray(value).reshape(-1)[0]) for key, value in pred.items()}
    row["inside_ad"] = bool(row["p_ensemble_all"] >= float(threshold))
    row["ad_threshold"] = float(threshold)
    return row


def score_structure(
    bundle,
    embedding,
    atomic_numbers,
    pred_d,
    pred_f,
    threshold=0.5,
    policy="element_block",
):
    """Score one crystal from post-encoder embeddings and the activity policy."""
    embedding = validate_ad_embedding(embedding, bundle, save_path=None)
    if atomic_numbers is not None and hasattr(atomic_numbers, "detach"):
        atomic_numbers = atomic_numbers.detach().cpu().numpy()
    settings = getattr(bundle, "settings", None) or {}
    activity = infer_sparse_channel_activity(
        atomic_numbers,
        pred_d=pred_d,
        pred_f=pred_f,
        policy=policy,
        threshold=float(settings.get("active_threshold", 1e-10)),
    )
    row = score_dos_ad(bundle, embedding, activity=activity, threshold=threshold)
    row["n_atoms"] = int(embedding.shape[0])
    row["contains_f_element"] = (
        bool(_contains_f_element(atomic_numbers)) if atomic_numbers is not None else False
    )
    if activity is None:
        row["active_d"] = False
        row["active_f"] = False
    else:
        row["active_d"] = bool(activity["d"][0])
        row["active_f"] = bool(activity["f"][0])
    row["f_element_warning"] = bool(row["contains_f_element"])
    return row


def public_ad_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    """JSON/UI subset of an AD score row."""
    return {
        "inside_ad": bool(row["inside_ad"]),
        "ad_threshold": float(row["ad_threshold"]),
        "p_ensemble_all": float(row["p_ensemble_all"]),
        "p_ensemble_s": float(row["p_ensemble_s"]),
        "p_ensemble_p": float(row["p_ensemble_p"]),
        "p_ensemble_d": float(row["p_ensemble_d"]),
        "p_ensemble_f": float(row["p_ensemble_f"]),
        "raw_knn": float(row["raw_knn"]),
        "raw_kde": float(row["raw_kde"]),
        "raw_lof": float(row["raw_lof"]),
        "active_d": bool(row["active_d"]),
        "active_f": bool(row["active_f"]),
        "contains_f_element": bool(row["contains_f_element"]),
        "f_element_warning": bool(row["f_element_warning"]),
    }


def try_load_ad_bundle(path: Optional[os.PathLike]) -> tuple[Optional[Any], Optional[str]]:
    """Load a calibrated bundle, or return (None, reason) if it is absent/invalid."""
    if path is None or str(path).strip() == "":
        return None, "AD model path is not configured"
    if not os.path.exists(os.fspath(path)):
        return None, f"AD model not found: {os.path.abspath(os.fspath(path))}"
    try:
        bundle = load_dos_ad_bundle(path)
        validate_transformer_ad_bundle(bundle)
        return bundle, None
    except (pickle.UnpicklingError, TypeError, ValueError, RuntimeError) as exc:
        return None, str(exc)
