"""サンプルCSV（架空データ）を data/sample/ に生成する。

    python scripts/make_sample_data.py

銘柄・ファクターリターンはすべて乱数による架空の値で、実在の銘柄とは無関係。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "sample"
FACTORS = ["Market", "Size", "Value", "Momentum", "Quality", "LowVol"]
SECTORS = ["情報通信", "電気機器", "輸送用機器", "銀行", "医薬品", "小売", "化学", "機械"]


def main(seed: int = 7, n_stocks: int = 40, n_days: int = 500) -> None:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-11-13", periods=n_days)
    tickers = [str(1001 + 7 * i) for i in range(n_stocks)]

    # ファクターリターン（日次）: 年率ボラ Market 16%、スタイル 3〜6%、弱い相関あり
    ann_vol = np.array([0.16, 0.05, 0.06, 0.06, 0.04, 0.05])
    corr = np.eye(6)
    corr[0, 5] = corr[5, 0] = -0.45   # Market × LowVol
    corr[2, 3] = corr[3, 2] = -0.35   # Value × Momentum
    corr[3, 4] = corr[4, 3] = 0.25    # Momentum × Quality
    cov = np.outer(ann_vol, ann_vol) * corr / 252
    drift = np.array([0.07, 0.00, 0.03, 0.02, 0.02, 0.00]) / 252
    f = rng.multivariate_normal(drift, cov, size=n_days)

    # エクスポージャー: Market は β≒1 近辺、スタイルは標準化スコア
    expo = rng.normal(0, 1, size=(n_stocks, 6))
    expo[:, 0] = np.clip(rng.normal(1.0, 0.25, n_stocks), 0.4, 1.8)
    expo[:, 5] = -1.2 * (expo[:, 0] - 1.0) / 0.25 * 0.5 + 0.6 * expo[:, 5]  # 高β ⇔ 低LowVol

    # 残差: 年率 15〜35%。一部の銘柄に直近の固有イベント（残差の急騰・急落）を仕込む
    spec_vol = rng.uniform(0.15, 0.35, n_stocks) / np.sqrt(252)
    eps = rng.normal(0, 1, size=(n_days, n_stocks)) * spec_vol
    eps[-12:, 3] += 0.012
    eps[-6:, 17] -= 0.02
    r = f @ expo.T + eps

    # 保有: ベンチマークは全銘柄、ポートフォリオは Value・Quality 寄りの 22 銘柄
    bench = rng.lognormal(0.0, 0.6, n_stocks)
    bench /= bench.sum()
    score = 0.8 * expo[:, 2] + 0.6 * expo[:, 4] + rng.normal(0, 0.5, n_stocks)
    held = np.argsort(-score)[:22]
    port = np.zeros(n_stocks)
    port[held] = bench[held] * rng.uniform(0.8, 2.2, len(held))
    port /= port.sum()

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ticker": tickers,
            "name": [f"サンプル{i + 1:02d}" for i in range(n_stocks)],
            "sector": [SECTORS[i % len(SECTORS)] for i in range(n_stocks)],
            "weight": port.round(6),
            "benchmark_weight": bench.round(6),
        }
    ).to_csv(OUT / "holdings.csv", index=False, encoding="utf-8-sig")

    idx = pd.Index(dates.strftime("%Y-%m-%d"), name="date")
    pd.DataFrame(r, index=idx, columns=tickers).round(6).to_csv(OUT / "stock_returns.csv")
    pd.DataFrame(f, index=idx, columns=FACTORS).round(6).to_csv(OUT / "factor_returns.csv")
    pd.DataFrame(expo, index=pd.Index(tickers, name="ticker"), columns=FACTORS).round(4).to_csv(
        OUT / "exposures.csv"
    )
    print(f"wrote sample CSVs to {OUT}")


if __name__ == "__main__":
    main()
