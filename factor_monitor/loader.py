"""CSV 読み込み。パス・アップロードファイル（file-like）のどちらも受け付ける。

- 文字コードは UTF-8(BOM可) → CP932(Excel の日本語CSV) の順に試す
- 銘柄コードは文字列として扱う（"7203" と 7203 を同一視、先頭ゼロを保持）
- 列名は英語のほか一部の日本語別名を受け付ける
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ALIASES = {
    "date": {"date", "日付", "年月日"},
    "ticker": {"ticker", "code", "symbol", "銘柄コード", "コード", "銘柄"},
    "name": {"name", "銘柄名", "名称"},
    "sector": {"sector", "セクター", "業種"},
    "weight": {"weight", "ウェイト", "保有比率", "構成比"},
    "benchmark_weight": {"benchmark_weight", "bm_weight", "benchmark", "ベンチマークウェイト", "bmウェイト"},
    "return": {"return", "ret", "リターン", "収益率"},
}


class DataError(ValueError):
    """入力CSVの形式エラー（画面にそのまま表示する）。"""


@dataclass
class Inputs:
    holdings: pd.DataFrame            # index=ticker; weight, benchmark_weight(任意), name, sector
    stock_returns: pd.DataFrame       # index=date, columns=ticker
    factor_returns: pd.DataFrame      # index=date, columns=factor
    exposures: pd.DataFrame | None    # index=ticker, columns=factor（無ければ回帰で推定）
    warnings: list[str] = field(default_factory=list)


def _read(src) -> pd.DataFrame:
    if isinstance(src, (str, Path)):
        raw = Path(src).read_bytes()
    else:
        raw = src.getvalue() if hasattr(src, "getvalue") else src.read()
    for enc in ("utf-8-sig", "cp932"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise DataError("文字コードを判別できません（UTF-8 か Shift_JIS で保存してください）")
    df = pd.read_csv(io.StringIO(text), dtype=str, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(how="all")


def _canon(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """別名の列を正規名にリネームする。"""
    ren = {}
    for c in df.columns:
        low = c.lower().replace(" ", "_")
        for k in keys:
            if low in ALIASES[k] and k not in ren.values():
                ren[c] = k
    return df.rename(columns=ren)


def _num(df: pd.DataFrame) -> pd.DataFrame:
    # pandas 3 では文字列列の dtype が object ではなく str になるため、「数値型でなければ変換」で判定する
    return df.apply(lambda s: s if pd.api.types.is_numeric_dtype(s)
                    else pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce"))


def _dated(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = _canon(df, ["date"])
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    try:
        idx = pd.to_datetime(df["date"])
    except Exception as e:  # noqa: BLE001
        raise DataError(f"{label}: 日付列を解釈できません（{e}）") from e
    out = _num(df.drop(columns="date"))
    out.index = pd.DatetimeIndex(idx, name="date")
    if out.index.has_duplicates:
        raise DataError(f"{label}: 日付が重複しています")
    return out.sort_index()


def load_holdings(src) -> pd.DataFrame:
    df = _canon(_read(src), ["ticker", "name", "sector", "weight", "benchmark_weight"])
    for col in ("ticker", "weight"):
        if col not in df.columns:
            raise DataError(f"holdings: 必須列 `{col}` がありません（列: {list(df.columns)}）")
    df["ticker"] = df["ticker"].str.strip()
    if df["ticker"].duplicated().any():
        raise DataError("holdings: ticker が重複しています")
    df = df.set_index("ticker")
    num_cols = [c for c in ("weight", "benchmark_weight") if c in df.columns]
    df[num_cols] = _num(df[num_cols]).fillna(0.0)
    return df


def load_stock_returns(src) -> pd.DataFrame:
    df = _read(src)
    long = _canon(df, ["date", "ticker", "return"])
    if {"date", "ticker", "return"} <= set(long.columns):  # ロング形式
        long["return"] = pd.to_numeric(long["return"], errors="coerce")
        long["ticker"] = long["ticker"].str.strip()
        long["date"] = pd.to_datetime(long["date"])
        if long.duplicated(["date", "ticker"]).any():
            raise DataError("stock_returns: (date, ticker) が重複しています")
        return long.pivot(index="date", columns="ticker", values="return").sort_index()
    return _dated(df, "stock_returns")


def load_factor_returns(src) -> pd.DataFrame:
    return _dated(_read(src), "factor_returns")


def load_exposures(src) -> pd.DataFrame:
    df = _canon(_read(src), ["ticker"])
    if "ticker" not in df.columns:
        df = df.rename(columns={df.columns[0]: "ticker"})
    df["ticker"] = df["ticker"].str.strip()
    if df["ticker"].duplicated().any():
        raise DataError("exposures: ticker が重複しています")
    return _num(df.set_index("ticker"))


def load_inputs(holdings, stock_returns, factor_returns, exposures=None) -> Inputs:
    inp = Inputs(
        holdings=load_holdings(holdings),
        stock_returns=load_stock_returns(stock_returns),
        factor_returns=load_factor_returns(factor_returns),
        exposures=load_exposures(exposures) if exposures is not None else None,
    )
    for label, df in (("stock_returns", inp.stock_returns), ("factor_returns", inp.factor_returns)):
        med = df.abs().stack().median()
        if med > 0.2:
            inp.warnings.append(
                f"{label}: 値が大きすぎます（絶対値の中央値 {med:.2f}）。"
                "リターンは小数（1% = 0.01）で入力してください。"
            )
    if inp.exposures is not None:
        missing = [f for f in inp.factor_returns.columns if f not in inp.exposures.columns]
        if missing:
            raise DataError(f"exposures: factor_returns にあるファクター列がありません: {missing}")
    return inp


def load_dir(path: str | Path) -> Inputs:
    """フォルダ内の holdings.csv / stock_returns.csv / factor_returns.csv / exposures.csv(任意) を読む。"""
    # Windows の「パスのコピー」は "C:\..." のように引用符が付くので取り除く
    p = Path(str(path).strip().strip('"').strip("'")).expanduser()
    for name in ("holdings.csv", "stock_returns.csv", "factor_returns.csv"):
        if not (p / name).exists():
            raise DataError(f"{p} に {name} がありません")
    expo = p / "exposures.csv"
    return load_inputs(p / "holdings.csv", p / "stock_returns.csv", p / "factor_returns.csv",
                       expo if expo.exists() else None)
