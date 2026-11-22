"""株式ファクターモニター（Streamlit）。

    streamlit run app.py
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from factor_monitor import DataError, Settings, analyze, load_dir, load_inputs

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "data" / "sample"

# 配色: カテゴリ色は固定順（ファクターの並び順に割り当て、9本目以降は「その他」に畳む）
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#898781"          # 残差・その他
POS, NEG = "#2a78d6", "#e34948"
DIVERGING = [[0.0, NEG], [0.5, "#f0efec"], [1.0, POS]]
MAX_LINES = 7

st.set_page_config(page_title="ファクターモニター", page_icon="📊", layout="wide")


# ---------------------------------------------------------------- helpers
def pct(v: float, digits: int = 2, sign: bool = False) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    return f"{v:+.{digits}%}" if sign else f"{v:.{digits}%}"


def L(a) -> list:
    """Plotly に渡す値は list にする（numpy 配列のバイナリ直列化を避ける）。"""
    if isinstance(a, pd.DatetimeIndex):
        return a.strftime("%Y-%m-%d").tolist()
    return [None if isinstance(v, float) and not np.isfinite(v) else v for v in np.asarray(a).tolist()]


def layout(fig: go.Figure, height: int = 340, ypct: bool = True, xpct: bool = False, legend: bool = True):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), hovermode="x unified" if ypct and not xpct else "closest",
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        bargap=0.35,
    )
    if ypct:
        fig.update_yaxes(tickformat=".1%")
    if xpct:
        fig.update_xaxes(tickformat=".1%")
    return fig


def _stretch(func) -> dict:
    """横幅いっぱいに表示する引数。Streamlit 1.50 以降は width="stretch" が既定で
    use_container_width は非推奨なので何も渡さず、それより古い版にだけ use_container_width を渡す。"""
    try:
        w = inspect.signature(func).parameters.get("width")
    except (TypeError, ValueError):
        w = None
    return {} if w is not None and w.default == "stretch" else {"use_container_width": True}


PLOT_KW, DF_KW = _stretch(st.plotly_chart), _stretch(st.dataframe)


def dataframe(data, height: int | None = None, **kw):
    """st.dataframe の版差吸収（新しい版は height=None を受け付けない）。"""
    if height is not None:
        kw["height"] = height
    st.dataframe(data, **DF_KW, **kw)


def show(fig: go.Figure):
    st.plotly_chart(fig, config={"displayModeBar": False}, **PLOT_KW)


def hbar(values: pd.Series, colors=None, height: int | None = None, fmt: str = ".2%") -> go.Figure:
    """符号つき横棒（上から values の順）。色は符号で塗り分け。"""
    v = values.iloc[::-1]
    col = colors[::-1] if colors is not None else [POS if x >= 0 else NEG for x in v]
    fig = go.Figure(go.Bar(
        x=L(v), y=[str(i) for i in v.index], orientation="h", marker_color=col,
        hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>",
    ))
    fig.update_layout(height=height or max(220, 26 * len(v) + 60), margin=dict(l=8, r=8, t=8, b=8),
                      showlegend=False, bargap=0.35)
    fig.update_xaxes(tickformat=fmt, zeroline=True, zerolinecolor="#c3c2b7")
    fig.update_yaxes(type="category")
    return fig


def lines(df: pd.DataFrame, colors: dict[str, str], height: int = 340, dash: dict | None = None) -> go.Figure:
    fig = go.Figure()
    x = L(df.index)
    for c in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=L(df[c]), name=str(c), mode="lines",
            line=dict(color=colors[c], width=2, dash=(dash or {}).get(c)),
            hovertemplate="%{y:.2%}",
        ))
    return layout(fig, height)


def heatmap(m: pd.DataFrame, height: int = 420, text: bool = True) -> go.Figure:
    labels = [str(c) for c in m.columns]
    z = [L(r) for r in m.to_numpy()]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, zmin=-1, zmax=1, colorscale=DIVERGING, xgap=2, ygap=2,
        text=[[("" if v is None else f"{v:.2f}") for v in r] for r in z] if text else None,
        texttemplate="%{text}" if text else None,
        hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>", colorbar=dict(thickness=10, len=0.8),
    ))
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8))
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category", autorange="reversed")
    return fig


def fold(df: pd.DataFrame, keep: int = MAX_LINES) -> pd.DataFrame:
    """列が多いときは累積値の絶対値が大きい順に keep 本残し、残りを「その他」に合算。"""
    if df.shape[1] <= keep + 1:
        return df
    top = df.iloc[-1].abs().sort_values(ascending=False).index[:keep]
    out = df[[c for c in df.columns if c in top]].copy()
    out["その他"] = df.drop(columns=top).sum(axis=1)
    return out


def pct_cols(df: pd.DataFrame, cols: list[str], digits: int = 2) -> dict:
    return {c: st.column_config.NumberColumn(c, format=f"%.{digits}f%%") for c in cols if c in df.columns}


def table(df: pd.DataFrame, pcols: list[str], height: int | None = None, digits: int = 2):
    """pcols を % 表示（値を100倍）にして表示。ソートは数値のまま効く。"""
    d = df.copy()
    for c in pcols:
        if c in d.columns:
            d[c] = d[c] * 100
    cfg = pct_cols(d, pcols, digits)
    cfg.update({c: st.column_config.NumberColumn(c, format="%.2f") for c in d.columns
                if c not in cfg and pd.api.types.is_float_dtype(d[c])})
    dataframe(d, height=height, column_config=cfg)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("データ")
    src = st.radio("入力元", ["サンプル", "フォルダ", "アップロード"], horizontal=True)
    inp = None
    try:
        if src == "サンプル":
            inp = load_dir(SAMPLE_DIR)
            st.caption("架空のサンプルデータを表示中")
        elif src == "フォルダ":
            folder = st.text_input("CSVフォルダのパス", placeholder=r"~/data/factor  または  C:\data\factor")
            st.caption("holdings.csv / stock_returns.csv / factor_returns.csv / exposures.csv（任意）")
            if folder:
                inp = load_dir(folder)
        else:
            up_h = st.file_uploader("holdings.csv（保有・ベンチ）", type="csv")
            up_r = st.file_uploader("stock_returns.csv（銘柄リターン）", type="csv")
            up_f = st.file_uploader("factor_returns.csv（ファクターリターン）", type="csv")
            up_e = st.file_uploader("exposures.csv（任意。無ければ回帰で推定）", type="csv")
            if up_h and up_r and up_f:
                inp = load_inputs(up_h, up_r, up_f, up_e)
    except DataError as e:
        st.error(str(e))

    st.header("設定")
    has_b = inp is not None and "benchmark_weight" in inp.holdings.columns
    mode_label = st.radio("分析ベース", ["アクティブ（対ベンチ）", "ポートフォリオ（絶対）"],
                          index=0 if has_b else 1, disabled=not has_b)
    freq = st.selectbox("データ頻度", ["日次 (252)", "週次 (52)", "月次 (12)"])
    ann = {"日次 (252)": 252, "週次 (52)": 52, "月次 (12)": 12}[freq]
    cov_label = st.radio("共分散の推定", ["等ウェイト", "EWMA"], horizontal=True)
    halflife = st.number_input("半減期（期間数）", 5, 500, 90) if cov_label == "EWMA" else None
    start = end = None
    if inp is not None:
        common = inp.stock_returns.index.intersection(inp.factor_returns.index)
        if len(common):
            lo, hi = common.min().date(), common.max().date()
            rng = st.date_input("分析期間", (lo, hi), min_value=lo, max_value=hi)
            if isinstance(rng, tuple) and len(rng) == 2:
                start, end = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    z_window = st.number_input("直近残差の期間数", 5, 120, 20)
    roll_window = st.number_input("ローリング期間数", 20, 250, 60)

st.title("ファクターモニター")

if inp is None:
    st.info("サイドバーでCSVを指定してください。形式は README.md の「入力CSV」を参照。")
    st.stop()

try:
    res = analyze(inp, Settings(
        mode="active" if mode_label.startswith("アクティブ") else "portfolio",
        ann=ann, halflife=halflife, start=start, end=end, z_window=int(z_window), roll_window=int(roll_window),
    ))
except DataError as e:
    st.error(str(e))
    st.stop()

s, ft, stt = res.summary, res.factor_table, res.stock_table
factors = list(ft.index)
fcolor = {f: (SERIES[i] if i < len(SERIES) else NEUTRAL) for i, f in enumerate(factors)}
fcolor.update({"残差": NEUTRAL, "その他": "#c3c2b7", "合計": "#0b0b0b", "ファクター": SERIES[0]})
base = "アクティブ" if s["mode"] == "active" else "ポートフォリオ"
risk_word = "アクティブリスク（TE）" if s["mode"] == "active" else "総リスク"
zcol = next(c for c in stt.columns if c.startswith("直近"))

st.caption(
    f"{base}ベース ／ {res.dates[0]:%Y-%m-%d} 〜 {res.dates[-1]:%Y-%m-%d}（{s['n_dates']} 期間） ／ "
    f"{s['n_stocks']} 銘柄（保有 {s['n_held']}） ／ エクスポージャー: "
    f"{'CSV' if res.exposure_source == 'csv' else '時系列回帰で推定'} ／ リスクは年率"
)
for wmsg in res.warnings:
    st.warning(wmsg)

tab_sum, tab_risk, tab_ret, tab_res, tab_fac, tab_data = st.tabs(
    ["サマリー", "ファクターリスク", "寄与度", "残差", "ファクター動向", "データ"]
)

# ---------------------------------------------------------------- サマリー
with tab_sum:
    c = st.columns(6)
    c[0].metric(risk_word, pct(s["total_risk"]))
    c[1].metric("ファクターリスク", pct(s["factor_risk"]))
    c[2].metric("残差（固有）リスク", pct(s["specific_risk"]))
    c[3].metric("ファクター比率", pct(s["factor_share"], 1), help="総分散に占めるファクター分散の割合")
    c[4].metric("実現リスク（期間）", pct(s["realized_risk"]))
    c[5].metric("期間リターン", pct(s["total_return"], 2, True))

    left, right = st.columns(2)
    with left:
        st.subheader("リスク寄与")
        st.caption(f"{risk_word} {pct(s['total_risk'])} の内訳。合計が総リスクに一致。")
        rc = ft["リスク寄与"].copy()
        rc["残差"] = stt["うち残差"].sum()
        rc = rc.sort_values(ascending=False)
        show(hbar(rc, colors=[NEUTRAL if i == "残差" else (POS if v >= 0 else NEG) for i, v in rc.items()]))
    with right:
        st.subheader("累積リターンの分解")
        st.caption(f"期間計: ファクター {pct(s['factor_return'], 2, True)} ／ 残差 {pct(s['residual_return'], 2, True)}")
        cum = pd.DataFrame({
            "合計": res.attribution["合計"].cumsum(),
            "ファクター": res.attribution[factors].sum(axis=1).cumsum(),
            "残差": res.attribution["残差"].cumsum(),
        })
        show(lines(cum, fcolor))

    left, right = st.columns(2)
    with left:
        st.subheader(f"{'アクティブ' if s['mode'] == 'active' else ''}エクスポージャー")
        show(hbar(ft["使用Exp"], fmt=".2f"))
    with right:
        st.subheader(f"残差 z スコア（直近 {int(z_window)} 期間）")
        st.caption("直近の累積残差 ÷ 期間中の残差ボラ。|z| が大きい銘柄はモデルで説明できない動き。")
        zz = stt["残差z"].dropna()
        pick = pd.concat([zz.nlargest(5), zz.nsmallest(5).iloc[::-1]])
        pick = pick[~pick.index.duplicated()]
        pick.index = [f"{t} {stt.at[t, 'name']}".strip() for t in pick.index]
        show(hbar(pick, fmt=".2f"))

# ---------------------------------------------------------------- ファクターリスク
with tab_risk:
    st.subheader("ファクター別リスク分解")
    cols = ["使用Exp", "ファクターVol", "単独リスク", "限界寄与", "リスク寄与", "リスク寄与率"]
    if s["has_benchmark"]:
        cols = ["ポートExp", "ベンチExp", "アクティブExp"] + cols[1:]
    t = ft[cols].copy()
    t.loc["残差（固有）"] = np.nan
    t.loc["残差（固有）", ["単独リスク", "リスク寄与", "リスク寄与率"]] = [
        s["specific_risk"], stt["うち残差"].sum(), 1 - s["factor_share"]]
    t.loc["合計"] = np.nan
    t.loc["合計", ["リスク寄与", "リスク寄与率"]] = [s["total_risk"], 1.0]
    table(t, ["ファクターVol", "単独リスク", "限界寄与", "リスク寄与", "リスク寄与率"])
    st.caption("単独リスク = |Exp| × ファクターVol ／ 限界寄与 = ∂σ/∂Exp ／ リスク寄与 = Exp × 限界寄与（相関込み）")

    left, right = st.columns(2)
    with left:
        st.subheader("エクスポージャー")
        if s["has_benchmark"]:
            fig = go.Figure()
            for name, col, color in (("ポート", "ポートExp", SERIES[0]), ("ベンチ", "ベンチExp", NEUTRAL),
                                     ("アクティブ", "アクティブExp", SERIES[1])):
                fig.add_trace(go.Bar(x=factors, y=L(ft[col]), name=name, marker_color=color,
                                     hovertemplate="%{y:.2f}"))
            fig.update_layout(barmode="group")
            fig.update_xaxes(type="category")
            show(layout(fig, 360, ypct=False))
        else:
            show(hbar(ft["使用Exp"], fmt=".2f"))
    with right:
        st.subheader("ファクター相関")
        show(heatmap(res.factor_corr, 360))

    st.subheader(f"ローリング実現リスク（{int(roll_window)} 期間・年率）")
    st.caption("固定ウェイトでのリターンをファクター部分と残差部分に分けた実現ボラティリティ。")
    show(lines(res.rolling_risk.dropna(), fcolor))

# ---------------------------------------------------------------- 寄与度
with tab_ret:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("ファクター別 累積リターン寄与")
        cumf = fold(res.attribution[factors].cumsum())
        cumf["残差"] = res.attribution["残差"].cumsum()
        show(lines(cumf, fcolor, 380))
    with right:
        st.subheader("期間リターン寄与")
        st.caption(f"合計 {pct(s['total_return'], 2, True)}")
        rcn = ft["リターン寄与"].copy()
        rcn["残差"] = s["residual_return"]
        rcn = rcn.sort_values(ascending=False)
        show(hbar(rcn, colors=[NEUTRAL if i == "残差" else (POS if v >= 0 else NEG) for i, v in rcn.items()],
                  height=380))

    st.subheader("銘柄別リスク寄与（上位）")
    n_top = st.slider("表示銘柄数", 5, min(40, len(stt)), min(15, len(stt)), key="n_top")
    topn = stt.reindex(stt["リスク寄与"].abs().sort_values(ascending=False).index[:n_top]).iloc[::-1]
    ylab = [f"{t} {n}".strip() for t, n in zip(topn.index, topn["name"])]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=L(topn["うちファクター"]), y=ylab, orientation="h", name="ファクター",
                         marker_color=SERIES[0], hovertemplate="%{x:.2%}"))
    fig.add_trace(go.Bar(x=L(topn["うち残差"]), y=ylab, orientation="h", name="残差",
                         marker_color=NEUTRAL, hovertemplate="%{x:.2%}"))
    fig.update_layout(barmode="relative")
    fig.update_yaxes(type="category")
    fig = layout(fig, max(300, 26 * n_top + 80), ypct=False, xpct=True)
    fig.update_layout(hovermode="y unified")
    show(fig)

    st.subheader("銘柄別明細")
    table(stt[["name", "sector", "ポート", "ベンチ", "アクティブ", "リスク寄与", "リスク寄与率",
               "うちファクター", "うち残差", "残差リターン寄与"]].sort_values("リスク寄与", ascending=False),
          ["ポート", "ベンチ", "アクティブ", "リスク寄与", "リスク寄与率", "うちファクター", "うち残差",
           "残差リターン寄与"], height=420)

# ---------------------------------------------------------------- 残差
with tab_res:
    c = st.columns(4)
    c[0].metric("残差（固有）リスク", pct(s["specific_risk"]))
    c[1].metric("残差リターン（期間）", pct(s["residual_return"], 2, True))
    c[2].metric("平均 R²", f"{stt['R2'].mean():.2f}")
    c[3].metric("残差相関の平均絶対値", f"{s['mean_abs_resid_corr']:.3f}",
                help="0 に近いほどファクターで共通変動を説明できている。大きい場合はファクター不足の可能性。")

    st.subheader("累積残差リターン（銘柄）")
    default = stt["残差z"].abs().sort_values(ascending=False).index[:4].tolist()
    sel = st.multiselect("銘柄（最大8）", list(stt.index), default=default, max_selections=8,
                         format_func=lambda t: f"{t} {stt.at[t, 'name']}".strip())
    if sel:
        cr = res.residuals[sel].fillna(0).cumsum()
        show(lines(cr, {t: SERIES[i] for i, t in enumerate(sel)}, 360))

    left, right = st.columns(2)
    with left:
        st.subheader("残差 z スコア（全銘柄）")
        zz = stt["残差z"].dropna().sort_values(ascending=False)
        zz.index = [f"{t} {stt.at[t, 'name']}".strip() for t in zz.index]
        show(hbar(zz, fmt=".2f", height=max(300, 18 * len(zz) + 60)))
    with right:
        st.subheader("残差相関（ウェイト上位）")
        st.caption("ファクターで説明した後に残る銘柄間の相関。")
        show(heatmap(res.resid_corr, 560, text=False))

    st.subheader("残差の特徴（銘柄別）")
    table(stt[["name", "sector", "アクティブ" if s["mode"] == "active" else "ポート", "残差Vol", "R2",
               "累積残差", zcol, "残差z", "残差リターン寄与", "うち残差"]]
          .sort_values("残差z", key=np.abs, ascending=False),
          ["アクティブ", "ポート", "残差Vol", "累積残差", zcol, "残差リターン寄与", "うち残差"], height=420)
    st.caption("残差Vol は年率。R² = 1 − 残差分散 ÷ リターン分散。うち残差 = 銘柄固有リスクのリスク寄与。")

# ---------------------------------------------------------------- ファクター動向
with tab_fac:
    st.subheader("ファクター累積リターン")
    fr = res.factor_returns
    show(lines(fold(fr.cumsum()), fcolor, 380))

    def last(n):  # 直近 n 期間の累積
        return fr.iloc[-n:].sum() if len(fr) >= n else pd.Series(np.nan, index=fr.columns)

    perf = pd.DataFrame({
        "直近1期": last(1), "直近5期": last(5), "直近20期": last(20), "直近60期": last(60),
        "期間計": fr.sum(), "Vol（年率）": ft["ファクターVol"],
        f"直近{int(roll_window)}期Vol": fr.iloc[-int(roll_window):].std() * np.sqrt(ann),
    })
    table(perf, list(perf.columns))

# ---------------------------------------------------------------- データ
with tab_data:
    st.subheader("銘柄別エクスポージャー")
    dataframe(res.B.join(res.weights[["portfolio", "benchmark", "active"]]), height=360)
    st.subheader("出力")
    c = st.columns(3)
    c[0].download_button("ファクター別（CSV）", ft.to_csv().encode("utf-8-sig"), "factor_table.csv", "text/csv")
    c[1].download_button("銘柄別（CSV）", stt.to_csv().encode("utf-8-sig"), "stock_table.csv", "text/csv")
    c[2].download_button("日次寄与（CSV）", res.attribution.to_csv().encode("utf-8-sig"), "attribution.csv",
                         "text/csv")
    st.subheader("ファクター共分散（年率）")
    dataframe(res.factor_cov.style.format("{:.5f}"))
