# 開発方針・セッション記録

## 方針

- 計算（`factor_monitor/`）と画面（`app.py`）を分離する。計算は pandas/numpy のみに依存し、単体でテストできる状態を保つ。
- 入力は CSV のみ。形式エラーは `DataError` で日本語メッセージを返し、画面にそのまま表示する。
- 加法性（リスク寄与の合計 = 総リスク、リターン寄与の合計 = 合計リターン）はテストで担保する。
- グラフ: Plotly には list で値を渡す（Plotly 6 の numpy バイナリ直列化が古い Streamlit 同梱の plotly.js で描画されない問題を避けるため）。
  カテゴリ色はファクターの並び順に固定で割り当て、8色を超える分は「その他」に畳む。残差はグレー。二軸グラフは使わない。

## 2026-11-15 初版

- 作成: CSV ローダー、分析エンジン、Streamlit 画面（6タブ）、サンプル生成、テスト12件。
- 決めたこと
  - スタック: Python + Streamlit + Plotly（手元の環境に導入済みだったため）。
  - ウェイト・エクスポージャーは直近値固定。保有履歴・エクスポージャー履歴は未対応。
  - `exposures.csv` が無ければ時系列回帰で推定。
- 動作確認: サンプルデータで全タブを表示確認。実データでは未確認。
- 置き場所: `~/Developer/repos/factor-monitor`。GitHub（public）: https://github.com/kurama8103/factor-monitor 、`scripts/externals.toml` 登録済み。

## 2026-11-22 Windows / Anaconda 対応の確認

- Windows 実機は無いため、macOS 上で新旧のライブラリ構成（README「確認済みの組み合わせ」）でテストと AppTest を実行して確認した。
- 修正
  - pandas 3 で CSV の数値変換が効かずエラーになっていた（文字列列の dtype が object ではなく str になったため）。`_num` を「数値型でなければ変換」に変更。
  - Streamlit 1.50 以降は `use_container_width` が非推奨、新しい版では `st.dataframe(height=None)` がエラー。シグネチャを見て版ごとに引数を切り替えるヘルパー（`_stretch` / `dataframe`）に集約した。
  - フォルダ指定でパスの前後の引用符（Windows の「パスのコピー」）を除去。
- 文字コード（UTF-8 / Shift_JIS 読み込み、BOM 付き出力）とパス処理（pathlib）は元から OS 非依存。

## 未対応・検討候補

- 保有履歴（日付つき holdings）による実績ベースの要因分析
- 業種ファクター（ダミー）とクロスセクション回帰によるファクターリターン推定
- ファクター共分散の直接入力（ベンダーモデルの共分散 CSV）
- セクター別集計、シナリオ／ストレステスト
