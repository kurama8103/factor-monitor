# ファクターモニター

株式ポートフォリオのファクターリスク・寄与度・残差を CSV 入力から可視化する Streamlit ダッシュボード。

## 起動

動作環境: Python 3.10 以上。macOS / Windows / Linux。

### Mac / Linux

```bash
cd ~/Developer/repos/factor-monitor
pip install -r requirements.txt   # 初回のみ
streamlit run app.py
```

### Windows（Anaconda）

Anaconda Prompt で実行する。pandas・numpy・plotly は Anaconda に同梱されているので、追加で入るのは主に streamlit。

```bat
git clone https://github.com/kurama8103/factor-monitor.git
cd factor-monitor
conda create -n factor-monitor python=3.12 -y
conda activate factor-monitor
pip install -r requirements.txt
streamlit run app.py
```

- 環境を分けずに base に入れる場合は `conda activate` までを飛ばす（`pip install` が base のパッケージを更新する点に注意）。
- git が無ければ GitHub の「Code → Download ZIP」で取得して展開する。
- ブラウザが自動で開かなければ表示された `http://localhost:8501` を開く。終了は Anaconda Prompt で Ctrl+C。
- フォルダ指定の入力欄には `C:\data\factor` のようなパスを入れる。エクスプローラーの「パスのコピー」で付く `"` はそのままでよい。
- Excel で保存した Shift_JIS の CSV もそのまま読める。ダウンロードする CSV は BOM 付き UTF-8 なので Excel で文字化けしない。

### 確認済みの組み合わせ

macOS 上で次の組み合わせについて、テストと画面のヘッドレス実行が通ることを確認している（Windows 実機では未確認）。

| Python | streamlit | pandas | numpy | plotly |
|---|---|---|---|---|
| 3.10 | 1.38 | 2.0 | 1.24 | 5.18 |
| 3.12 | 1.45 / 1.50 | 2.2 | 2.2 | 6.0 |
| 3.13 | 1.67 | 3.1 | 2.5 | 7.1 |

サイドバーの「入力元」で以下を選ぶ。

| 入力元 | 内容 |
|---|---|
| サンプル | `data/sample/` の架空データ（`python scripts/make_sample_data.py` で再生成） |
| フォルダ | 下記のファイル名で CSV を置いたフォルダのパスを指定 |
| アップロード | CSV を個別にアップロード |

## 入力CSV

文字コードは UTF-8（BOM可）または Shift_JIS。**リターンは小数**（1% = `0.01`）。銘柄コードは文字列として扱う。

| ファイル | 必須 | 形式 |
|---|---|---|
| `holdings.csv` | ○ | `ticker, weight` ＋ 任意で `benchmark_weight, name, sector`。1行1銘柄 |
| `stock_returns.csv` | ○ | ワイド形式（`date` 列 ＋ 銘柄コード列）。ロング形式（`date, ticker, return`）も可 |
| `factor_returns.csv` | ○ | `date` 列 ＋ ファクター列 |
| `exposures.csv` | 任意 | `ticker` 列 ＋ ファクター列（`factor_returns.csv` と同じ列名） |

- `exposures.csv` が無い場合は、銘柄ごとに時系列回帰（切片つき OLS）でエクスポージャーを推定する。
- アクティブ分析をする場合、**非保有のベンチマーク構成銘柄も `weight=0` で `holdings.csv` に入れる**。
  `benchmark_weight` が無ければ絶対ベースで計算する。
- 列名は `銘柄コード` `ウェイト` `日付` などの日本語別名も可（`factor_monitor/loader.py` の `ALIASES`）。
- リターン／エクスポージャーが無い保有銘柄は除外し、画面に警告を出す。

`data/sample/` の4ファイルがそのまま雛形になる。

## 画面

| タブ | 内容 |
|---|---|
| サマリー | 総リスク（アクティブ時は TE）、ファクター／残差リスク、リスク寄与、累積リターン分解、残差 z スコア上下位 |
| ファクターリスク | ファクター別のエクスポージャー・Vol・限界寄与・リスク寄与、ファクター相関、ローリング実現リスク |
| 寄与度 | ファクター別の累積リターン寄与、銘柄別リスク寄与（ファクター分／残差分） |
| 残差 | 銘柄別の累積残差・残差Vol・R²・直近残差 z スコア、残差相関（モデル診断） |
| ファクター動向 | ファクター累積リターン、直近パフォーマンスと Vol |
| データ | 銘柄別エクスポージャー、ファクター共分散、結果の CSV ダウンロード |

設定: 分析ベース（アクティブ／絶対）、データ頻度（年率換算）、共分散推定（等ウェイト／EWMA）、分析期間、z スコア・ローリングの期間数。

## 計算方法

```
r_t = B f_t + ε_t                 銘柄リターン = エクスポージャー × ファクターリターン + 残差
σ²  = xᵀ F x + Σ w_i² d_i          x = Bᵀw、F: ファクター共分散、d_i: 残差分散（いずれも年率）
リスク寄与_k = x_k (F x)_k / σ      ファクター寄与と残差寄与の合計が σ に一致
リターン寄与_k,t = x_k f_k,t        残差寄与_t = Σ w_i ε_i,t
残差z_i = 直近n期の累積残差 / (残差σ_i √n)
```

前提と限界:

- ウェイトとエクスポージャーは**直近値で固定**して全期間に適用する（保有履歴は使わない）。リターン寄与は現在のポートフォリオを過去に当てはめた近似で、実績のパフォーマンス要因分析ではない。
- 累積は単純和（複利ではない）。
- 残差同士は無相関と仮定してリスクを計算する。「残差相関の平均絶対値」が大きい場合はファクター不足の可能性がある。
- 回帰推定の場合、残差に切片（α）が含まれる。

## 構成

```
app.py                     Streamlit 画面
factor_monitor/loader.py   CSV 読み込み・検証
factor_monitor/model.py    リスク分解・寄与・残差の計算（analyze）
scripts/make_sample_data.py サンプル生成
tests/                     pytest（python -m pytest tests）
docs/DEVELOPMENT.md        開発方針・セッション記録
```

画面を使わず Python から使う場合:

```python
from factor_monitor import load_dir, analyze, Settings
res = analyze(load_dir("data/sample"), Settings(mode="active"))
res.factor_table, res.stock_table, res.summary
```

## ライセンス

[MIT](LICENSE)
