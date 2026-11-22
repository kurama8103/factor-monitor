"""線形ファクターモデルによるリスク分解・リターン寄与・残差分析。

    r_t = B f_t + ε_t            （銘柄リターン = エクスポージャー × ファクターリターン + 残差）
    Σ   = B F Bᵀ + D             （F: ファクター共分散, D: 残差分散の対角行列）
    σ²  = xᵀ F x + Σ w_i² d_i     （x = Bᵀ w: ポートフォリオのファクターエクスポージャー）

ウェイト w は「アクティブ（ポート − ベンチ）」か「ポートフォリオ（絶対）」を選ぶ。
リターン寄与は期間中ウェイト・エクスポージャー固定（直近の値）と仮定した近似。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .loader import DataError, Inputs


@dataclass
class Settings:
    mode: str = "active"              # "active" | "portfolio"
    ann: int = 252                    # 年率換算係数（日次252 / 週次52 / 月次12）
    halflife: float | None = None     # EWMA 半減期（期間数）。None なら等ウェイト
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    z_window: int = 20                # 直近残差 z スコアの期間数
    roll_window: int = 60             # ローリング実現リスクの期間数


@dataclass
class Result:
    settings: Settings
    exposure_source: str              # "csv" | "regression"
    dates: pd.DatetimeIndex
    weights: pd.DataFrame             # ticker × [portfolio, benchmark, active, used]
    B: pd.DataFrame                   # ticker × factor
    factor_cov: pd.DataFrame          # 年率
    factor_corr: pd.DataFrame
    summary: dict
    factor_table: pd.DataFrame        # ファクター別: エクスポージャー・リスク寄与・リターン寄与
    stock_table: pd.DataFrame         # 銘柄別: リスク寄与・残差の特徴
    attribution: pd.DataFrame         # date × [factor..., 残差, 合計]（期間ごとの寄与）
    rolling_risk: pd.DataFrame        # date × [ファクター, 残差, 合計]（年率・実現）
    residuals: pd.DataFrame           # date × ticker
    factor_returns: pd.DataFrame      # date × factor
    resid_corr: pd.DataFrame          # ticker × ticker（保有上位）
    warnings: list[str] = field(default_factory=list)


def _time_weights(n: int, halflife: float | None) -> np.ndarray:
    if not halflife:
        return np.full(n, 1.0 / n)
    w = 0.5 ** (np.arange(n - 1, -1, -1) / halflife)
    return w / w.sum()


def _wcov(X: np.ndarray, tw: np.ndarray) -> np.ndarray:
    """時間ウェイト付き共分散（不偏補正つき）。"""
    mu = tw @ X
    Xc = X - mu
    return (Xc * tw[:, None]).T @ Xc / (1.0 - (tw**2).sum())


def _wvar_nan(E: np.ndarray, tw: np.ndarray) -> np.ndarray:
    """列ごとの時間ウェイト付き分散。NaN は除外しウェイトを再正規化。"""
    ok = ~np.isnan(E)
    W = np.where(ok, tw[:, None], 0.0)
    s = W.sum(axis=0)
    s[s == 0] = np.nan
    W = W / s
    E0 = np.where(ok, E, 0.0)
    mu = (W * E0).sum(axis=0)
    var = (W * (E0 - mu) ** 2).sum(axis=0)
    return var / (1.0 - (W**2).sum(axis=0))


def _estimate_betas(R: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    """銘柄ごとに r = a + B f の時系列 OLS（切片つき）でエクスポージャーを推定。"""
    X = np.column_stack([np.ones(len(F)), F.to_numpy()])
    out = {}
    for t in R.columns:
        y = R[t].to_numpy()
        ok = ~np.isnan(y)
        if ok.sum() < X.shape[1] + 5:
            continue
        coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        out[t] = coef[1:]
    return pd.DataFrame.from_dict(out, orient="index", columns=F.columns)


def analyze(inp: Inputs, st: Settings | None = None) -> Result:
    st = st or Settings()
    warns = list(inp.warnings)

    # --- 日付の整合 -------------------------------------------------------
    F = inp.factor_returns.dropna(how="any")
    dates = inp.stock_returns.index.intersection(F.index)
    if st.start is not None:
        dates = dates[dates >= pd.Timestamp(st.start)]
    if st.end is not None:
        dates = dates[dates <= pd.Timestamp(st.end)]
    factors = list(F.columns)
    K = len(factors)
    if len(dates) < max(K + 5, 20):
        raise DataError(
            f"銘柄リターンとファクターリターンの共通日付が {len(dates)} 件しかありません"
            f"（最低 {max(K + 5, 20)} 件必要）"
        )
    F = F.loc[dates]

    # --- 銘柄の整合 -------------------------------------------------------
    h = inp.holdings
    tickers = [t for t in h.index if t in inp.stock_returns.columns]
    no_ret = [t for t in h.index if t not in inp.stock_returns.columns]
    R = inp.stock_returns.loc[dates, tickers]

    if inp.exposures is not None:
        source = "csv"
        B = inp.exposures.reindex(tickers)[factors]
    else:
        source = "regression"
        B = _estimate_betas(R, F).reindex(tickers)
    no_expo = B.index[B.isna().any(axis=1)].tolist()
    tickers = [t for t in tickers if t not in no_expo]
    if not tickers:
        raise DataError("保有銘柄とリターン／エクスポージャーの銘柄コードが1つも一致しません")
    B, R = B.loc[tickers], R[tickers]

    wp = h["weight"].astype(float)
    wb = h["benchmark_weight"].astype(float) if "benchmark_weight" in h.columns else wp * 0.0
    has_bench = bool((wb.abs() > 0).any())
    mode = st.mode if has_bench else "portfolio"
    if st.mode == "active" and not has_bench:
        warns.append("benchmark_weight が無いため、ポートフォリオ（絶対）ベースで計算しました。")
    dropped = no_ret + no_expo
    if dropped:
        lost = wp.reindex(dropped).abs().sum()
        warns.append(
            f"リターンまたはエクスポージャーが無い {len(dropped)} 銘柄を除外しました"
            f"（ポートウェイト計 {lost:.2%}）: {', '.join(dropped[:10])}{' …' if len(dropped) > 10 else ''}"
        )
    weights = pd.DataFrame({"portfolio": wp, "benchmark": wb}).loc[tickers]
    weights["active"] = weights["portfolio"] - weights["benchmark"]
    weights["used"] = weights["active"] if mode == "active" else weights["portfolio"]
    w = weights["used"].to_numpy()

    # --- 残差・共分散 -----------------------------------------------------
    Bm, Fm, Rm = B.to_numpy(float), F.to_numpy(float), R.to_numpy(float)
    E = Rm - Fm @ Bm.T                                   # T × N
    nan_share = np.isnan(Rm).mean()
    if nan_share > 0:
        warns.append(f"銘柄リターンの欠損 {nan_share:.1%} は寄与計算で 0 として扱いました。")
    tw = _time_weights(len(dates), st.halflife)
    Fcov = _wcov(Fm, tw) * st.ann                        # K × K（年率）
    d = _wvar_nan(E, tw) * st.ann                        # N（年率の残差分散）
    d = np.where(np.isnan(d), 0.0, d)

    # --- リスク分解 -------------------------------------------------------
    x = Bm.T @ w
    Fx = Fcov @ x
    var_f = float(x @ Fx)
    var_s = float((w**2 * d).sum())
    var_t = var_f + var_s
    sig = np.sqrt(var_t) if var_t > 0 else np.nan
    fvol = np.sqrt(np.diag(Fcov))
    sd = np.outer(fvol, fvol)
    Fcorr = np.divide(Fcov, sd, out=np.zeros_like(Fcov), where=sd > 0)

    E0 = np.where(np.isnan(E), 0.0, E)
    R0 = np.where(np.isnan(Rm), 0.0, Rm)
    f_contrib = Fm * x                                    # T × K
    resid_p = E0 @ w
    attribution = pd.DataFrame(f_contrib, index=dates, columns=factors)
    attribution["残差"] = resid_p
    attribution["合計"] = R0 @ w

    x_p = Bm.T @ weights["portfolio"].to_numpy()
    x_b = Bm.T @ weights["benchmark"].to_numpy()
    factor_table = pd.DataFrame(
        {
            "ポートExp": x_p,
            "ベンチExp": x_b,
            "アクティブExp": x_p - x_b,
            "使用Exp": x,
            "ファクターVol": fvol,
            "単独リスク": np.abs(x) * fvol,
            "限界寄与": Fx / sig,
            "リスク寄与": x * Fx / sig,
            "リスク寄与率": x * Fx / var_t,
            "期間リターン": Fm.sum(axis=0),
            "リターン寄与": f_contrib.sum(axis=0),
        },
        index=pd.Index(factors, name="factor"),
    )

    # --- 銘柄別 -----------------------------------------------------------
    n = len(dates)
    zw = min(st.z_window, n)
    d_per = _wvar_nan(E, np.full(n, 1.0 / n))            # 等ウェイトの1期間あたり残差分散
    recent = E0[-zw:].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = recent / np.sqrt(d_per * zw)
        r2 = 1.0 - d_per / np.nanvar(Rm, axis=0, ddof=1)
    stock_table = pd.DataFrame(
        {
            "name": h["name"].reindex(tickers) if "name" in h.columns else "",
            "sector": h["sector"].reindex(tickers) if "sector" in h.columns else "",
            "ポート": weights["portfolio"],
            "ベンチ": weights["benchmark"],
            "アクティブ": weights["active"],
            "リスク寄与": w * (Bm @ Fx + d * w) / sig,
            "うちファクター": w * (Bm @ Fx) / sig,
            "うち残差": w**2 * d / sig,
            "残差Vol": np.sqrt(d),
            "R2": r2,
            "累積残差": E0.sum(axis=0),
            f"直近{zw}期残差": recent,
            "残差z": z,
            "残差リターン寄与": w * E0.sum(axis=0),
        },
        index=pd.Index(tickers, name="ticker"),
    )
    stock_table["リスク寄与率"] = stock_table["リスク寄与"] / sig

    # --- ローリング実現リスク ----------------------------------------------
    rw = min(st.roll_window, n)
    parts = pd.DataFrame(
        {"ファクター": f_contrib.sum(axis=1), "残差": resid_p, "合計": attribution["合計"].to_numpy()},
        index=dates,
    )
    rolling = parts.rolling(rw).std() * np.sqrt(st.ann)

    # --- 残差相関（|使用ウェイト| 上位 25 銘柄）とモデル診断 ---------------
    resid = pd.DataFrame(E, index=dates, columns=tickers)
    top = weights["used"].abs().sort_values(ascending=False).index[:25]
    resid_corr = resid[top].corr(min_periods=20)
    full_corr = resid.corr(min_periods=20).to_numpy()
    off = full_corr[~np.eye(len(tickers), dtype=bool)]
    realized = float(parts["合計"].std() * np.sqrt(st.ann))

    summary = {
        "mode": mode,
        "has_benchmark": has_bench,
        "n_stocks": len(tickers),
        "n_held": int((weights["portfolio"].abs() > 0).sum()),
        "n_dates": n,
        "total_risk": float(sig),
        "factor_risk": float(np.sqrt(var_f)),
        "specific_risk": float(np.sqrt(var_s)),
        "factor_share": var_f / var_t if var_t > 0 else np.nan,
        "realized_risk": realized,
        "total_return": float(parts["合計"].sum()),
        "factor_return": float(parts["ファクター"].sum()),
        "residual_return": float(parts["残差"].sum()),
        "mean_abs_resid_corr": float(np.nanmean(np.abs(off))) if off.size else np.nan,
        "weight_sum": float(weights["portfolio"].sum()),
    }

    return Result(
        settings=st, exposure_source=source, dates=dates, weights=weights, B=B,
        factor_cov=pd.DataFrame(Fcov, index=factors, columns=factors),
        factor_corr=pd.DataFrame(Fcorr, index=factors, columns=factors),
        summary=summary, factor_table=factor_table, stock_table=stock_table,
        attribution=attribution, rolling_risk=rolling, residuals=resid,
        factor_returns=F, resid_corr=resid_corr, warnings=warns,
    )
