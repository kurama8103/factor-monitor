import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from factor_monitor import DataError, Settings, analyze, load_dir  # noqa: E402
from factor_monitor.loader import Inputs, load_holdings  # noqa: E402

SAMPLE = ROOT / "data" / "sample"


@pytest.fixture(scope="module")
def inp():
    return load_dir(SAMPLE)


@pytest.mark.parametrize("mode", ["active", "portfolio"])
@pytest.mark.parametrize("halflife", [None, 60])
def test_risk_contributions_add_up(inp, mode, halflife):
    res = analyze(inp, Settings(mode=mode, halflife=halflife))
    s, ft, stt = res.summary, res.factor_table, res.stock_table
    assert s["total_risk"] ** 2 == pytest.approx(s["factor_risk"] ** 2 + s["specific_risk"] ** 2)
    # ファクター寄与 + 残差寄与 = 総リスク
    assert ft["リスク寄与"].sum() + stt["うち残差"].sum() == pytest.approx(s["total_risk"])
    # 銘柄別寄与の合計 = 総リスク、ファクター分の合計はファクター別と一致
    assert stt["リスク寄与"].sum() == pytest.approx(s["total_risk"])
    assert stt["うちファクター"].sum() == pytest.approx(ft["リスク寄与"].sum())
    assert ft["リスク寄与率"].sum() == pytest.approx(s["factor_share"])


def test_return_attribution_adds_up(inp):
    res = analyze(inp)
    a = res.attribution
    parts = a.drop(columns="合計").sum(axis=1)
    np.testing.assert_allclose(parts.to_numpy(), a["合計"].to_numpy(), atol=1e-12)
    assert res.summary["factor_return"] + res.summary["residual_return"] == pytest.approx(
        res.summary["total_return"]
    )


def test_matches_brute_force_covariance(inp):
    """σ² = wᵀ(BFBᵀ + D)w を直接計算した値と一致する。"""
    res = analyze(inp, Settings(mode="active"))
    w = res.weights["used"].to_numpy()
    B = res.B.to_numpy()
    D = np.diag(res.residuals.var(ddof=1).to_numpy() * 252)
    sigma = B @ res.factor_cov.to_numpy() @ B.T + D
    assert np.sqrt(w @ sigma @ w) == pytest.approx(res.summary["total_risk"], rel=1e-9)


def test_regression_recovers_exposures(inp):
    true_B = inp.exposures
    est = analyze(Inputs(inp.holdings, inp.stock_returns, inp.factor_returns, None))
    assert est.exposure_source == "regression"
    err = (est.B - true_B.loc[est.B.index, est.B.columns]).abs()
    assert err["Market"].max() < 0.3          # β の推定誤差
    assert np.corrcoef(est.B["Value"], true_B.loc[est.B.index, "Value"])[0, 1] > 0.9


def test_planted_residual_events_are_flagged(inp):
    z = analyze(inp).stock_table["残差z"]
    assert z.idxmax() == inp.holdings.index[3] and z.max() > 2
    assert z.idxmin() == inp.holdings.index[17] and z.min() < -2


def test_missing_data_handling(inp):
    sr = inp.stock_returns.copy()
    sr.iloc[:50, 0] = np.nan                   # 上場が遅い銘柄
    sr = sr.drop(columns=sr.columns[1])        # リターンが無い銘柄
    res = analyze(Inputs(inp.holdings, sr, inp.factor_returns, inp.exposures))
    assert inp.holdings.index[1] not in res.stock_table.index
    assert any("除外" in w for w in res.warnings)
    assert np.isfinite(res.summary["total_risk"])
    assert res.stock_table["リスク寄与"].sum() == pytest.approx(res.summary["total_risk"])


def test_no_benchmark_falls_back_to_portfolio(inp):
    h = inp.holdings.drop(columns="benchmark_weight")
    res = analyze(Inputs(h, inp.stock_returns, inp.factor_returns, inp.exposures), Settings(mode="active"))
    assert res.summary["mode"] == "portfolio"


def test_loader_cp932_and_japanese_headers(tmp_path):
    p = tmp_path / "h.csv"
    p.write_bytes("銘柄コード,銘柄名,ウェイト\n7203,トヨタ,0.6\n0001,テスト,0.4\n".encode("cp932"))
    h = load_holdings(p)
    assert list(h.index) == ["7203", "0001"]
    assert h["weight"].sum() == pytest.approx(1.0)


def test_too_few_dates_raises(inp):
    with pytest.raises(DataError):
        analyze(inp, Settings(start=pd.Timestamp("2026-11-10")))
