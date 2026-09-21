"""Unit tests for DOS-GCNN applicability-domain runtime."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dos_gcnn.ad import (
    infer_sparse_channel_activity,
    load_dos_ad_bundle,
    score_dos_ad,
    score_structure,
    validate_ad_embedding,
)
from dos_gcnn.config import Config

AD_PKL = Path(__file__).resolve().parents[1] / "bulk_new" / "ad" / "ad_model.pkl"


class _ArrayTensor:
    """Minimal torch-like tensor so tests do not need a live CUDA/torch graph."""

    def __init__(self, array):
        self._array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class DummyData:
    def __init__(self, x, z):
        self.x = _ArrayTensor(x)
        self.z = _ArrayTensor(z)


class DummyScaler:
    def __init__(self, n_features):
        self.n_features_in_ = int(n_features)


class DummySpace:
    def __init__(self, n_features=370):
        self.scaler = DummyScaler(n_features)

    def transform_rows(self, rows):
        return np.asarray(rows, dtype=np.float32)


class DummyBundle:
    def __init__(self, n_features=370, p_all=0.72, settings=None):
        self.space = DummySpace(n_features)
        self.settings = settings or {"active_threshold": 1e-10}
        self._p_all = float(p_all)

    def predict_from_z(self, z_list, activity=None):
        n = len(z_list)
        p_d = np.full(n, 0.6)
        p_f = np.full(n, 1.0)
        if activity is not None:
            if not bool(np.asarray(activity["d"]).reshape(-1)[0]):
                p_d[:] = 1.0
            if not bool(np.asarray(activity["f"]).reshape(-1)[0]):
                p_f[:] = 1.0
        return {
            "raw_knn": np.full(n, 0.1),
            "raw_kde": np.full(n, 0.2),
            "raw_lof": np.full(n, 0.3),
            "p_ensemble_s": np.full(n, 0.8),
            "p_ensemble_p": np.full(n, 0.7),
            "p_ensemble_d": p_d,
            "p_ensemble_f": p_f,
            "p_ensemble_all": np.full(n, self._p_all),
        }


def test_element_block_marks_main_group_d_and_f_inactive():
    activity = infer_sparse_channel_activity([14, 8], policy="element_block")
    assert activity is not None
    assert activity["d"].tolist() == [False]
    assert activity["f"].tolist() == [False]


def test_element_block_marks_transition_metal_d_active():
    activity = infer_sparse_channel_activity([26, 8], policy="element_block")
    assert activity["d"].tolist() == [True]
    assert activity["f"].tolist() == [False]


def test_element_block_marks_f_block_active_and_warns_via_score_structure():
    activity = infer_sparse_channel_activity([58], policy="element_block")
    assert activity["f"].tolist() == [True]

    data_emb = np.zeros((1, 370), dtype=np.float32)
    row = score_structure(
        DummyBundle(p_all=0.9),
        data_emb,
        atomic_numbers=[58],
        pred_d=np.zeros((1, 400)),
        pred_f=np.ones((1, 400)),
        threshold=0.50,
        policy="element_block",
    )
    assert row["contains_f_element"] is True
    assert row["f_element_warning"] is True
    assert row["active_f"] is True


def test_unknown_activity_policy_raises():
    with pytest.raises(ValueError, match="Unknown AD sparse activity policy"):
        infer_sparse_channel_activity([14], policy="not-a-policy")


def test_score_dos_ad_uses_threshold_for_inside_flag():
    bundle = DummyBundle(p_all=0.72)
    embedding = np.zeros((3, 370), dtype=np.float32)

    inside = score_dos_ad(bundle, embedding, threshold=0.50)
    assert inside["inside_ad"] is True
    assert inside["ad_threshold"] == 0.50
    assert inside["p_ensemble_all"] == pytest.approx(0.72)

    outside = score_dos_ad(bundle, embedding, threshold=0.80)
    assert outside["inside_ad"] is False


def test_extract_ad_embedding_rejects_dimension_mismatch():
    bundle = DummyBundle(n_features=370)
    with pytest.raises(ValueError, match="dimensionality mismatch"):
        validate_ad_embedding(np.zeros((2, 10), dtype=np.float32), bundle)


def test_score_structure_applies_element_block_shortcut_for_inactive_df():
    row = score_structure(
        DummyBundle(p_all=0.4),
        np.zeros((2, 370), dtype=np.float32),
        atomic_numbers=[14, 8],
        pred_d=np.ones((2, 400)),
        pred_f=np.ones((2, 400)),
        threshold=0.50,
        policy="element_block",
    )
    assert row["active_d"] is False
    assert row["active_f"] is False
    assert row["inside_ad"] is False
    assert row["p_ensemble_d"] == pytest.approx(1.0)
    assert row["p_ensemble_f"] == pytest.approx(1.0)


def test_config_exposes_ad_job_parameters():
    config = Config()
    params = config.get_job_parameters()
    assert params["ad_threshold"] == pytest.approx(0.50)
    assert params["ad_activity_policy"] == "element_block"
    assert Path(params["ad_model_path"]).name == "ad_model.pkl"


def test_missing_ad_pkl_raises_file_not_found():
    missing = Path(__file__).resolve().parent / "definitely_missing_ad_model.pkl"
    with pytest.raises(FileNotFoundError, match="AD model not found"):
        load_dos_ad_bundle(missing)


def _is_pickle_protocol(path: Path) -> bool:
    try:
        return path.read_bytes()[:1] == b"\x80"
    except OSError:
        return False


def test_load_rejects_pytorch_zip_checkpoint():
    import tempfile

    handle = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    fake = Path(handle.name)
    handle.close()
    try:
        fake.write_bytes(b"PK\x03\x04" + b"\x00" * 24)
        with pytest.raises(TypeError, match="ZIP/PyTorch checkpoint"):
            load_dos_ad_bundle(fake)
    finally:
        fake.unlink(missing_ok=True)


@pytest.mark.skipif(
    not AD_PKL.exists() or not _is_pickle_protocol(AD_PKL),
    reason="calibrated ADBundle pickle not available",
)
def test_load_calibrated_bundle_scaler_dimension():
    bundle = load_dos_ad_bundle(AD_PKL)
    scaler = bundle.space.scaler
    n_features = int(getattr(scaler, "n_features_in_", np.asarray(scaler.mean_).shape[0]))
    assert n_features == 370
    from dos_gcnn.ad import validate_transformer_ad_bundle
    validate_transformer_ad_bundle(bundle)
    settings = getattr(bundle, "settings", {}) or {}
    provenance = str(settings.get("embedding_provenance", "")).lower()
    assert "transformer" in provenance
    assert "post-encoder" in provenance


def test_validate_transformer_ad_bundle_rejects_non_transformer_provenance():
    from dos_gcnn.ad import validate_transformer_ad_bundle

    bundle = DummyBundle(settings={"embedding_provenance": "v2 data.x 370-D"})
    with pytest.raises(RuntimeError, match="post-encoder"):
        validate_transformer_ad_bundle(bundle)


def test_try_load_ad_bundle_reports_missing_path():
    from dos_gcnn.ad import try_load_ad_bundle

    bundle, reason = try_load_ad_bundle(Path(__file__).resolve().parent / "no_such_ad.pkl")
    assert bundle is None
    assert reason is not None
    assert "not found" in reason.lower()
