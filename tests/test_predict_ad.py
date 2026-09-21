"""Integration test: Transformer DOS + post-encoder AD on a tiny CIF."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CIF = Path(__file__).resolve().parent / "fixtures" / "Si.cif"
PTH = ROOT / "bulk_new" / "saved_models" / "model_bulk_lorentz_28_spdf_tansformer_terms_new.pth"
PKL = ROOT / "bulk_new" / "ad" / "ad_model.pkl"


def _is_pickle_protocol(path: Path) -> bool:
    try:
        return path.read_bytes()[:1] == b"\x80"
    except OSError:
        return False


@pytest.mark.skipif(
    not CIF.exists() or not PTH.exists() or not PKL.exists() or not _is_pickle_protocol(PKL),
    reason="Transformer checkpoint or calibrated AD pickle not available",
)
def test_predict_dos_scores_post_encoder_ad():
    from dos_gcnn import Config, predict_dos

    result = predict_dos(CIF, config=Config())
    assert result.dos_s.ndim == 2
    assert result.dos_s.shape[1] == 400
    assert result.ad is not None, result.ad_reason
    assert result.ad_reason is None
    assert "inside_ad" in result.ad
    assert result.ad["ad_threshold"] == pytest.approx(0.50)
    assert result.ad["active_d"] is False
    assert result.ad["active_f"] is False
    assert result.ad["p_ensemble_d"] == pytest.approx(1.0)
    assert result.ad["p_ensemble_f"] == pytest.approx(1.0)
