# LDF kinetics analysis — Streamlit アプリ

`LDF.ipynb` を Streamlit アプリ (`app.py`) に変換したものです。実験の
uptake データから速度定数分布 k_LDF を逆解析します（Tikhonov 正則化 +
NNLS / SLSQP、L-curve 法で最適 λ を選択）。

## ファイル

- `app.py` — Streamlit アプリ本体
- `requirements.txt` — 依存ライブラリ
- `00_exp.csv` / `00_sec.csv` / `00_k.csv` / `01_lambda.csv` — サンプル入力（同梱）

## ローカルで動かす

```bash
pip install -r requirements.txt
streamlit run app.py
```

ブラウザで `http://localhost:8501` が開きます。サイドバーで α 値・最適化手法・
λ グリッドを選び「計算を実行」を押すと、L-curve・Uptake fitting・k_LDF 分布が
表示され、全出力を ZIP でダウンロードできます。ファイルを何もアップロード
しなければ同梱のサンプルが使われます。

## L-curve まわりの改善（今回の修正）

- **変曲点(角)の検出を Menger 曲率に変更。** 元ノートの有限差分による曲率は
  曲率配列と点の対応が 1 つずれており、既定で正しい角を選べていなかった。
  Menger 曲率は3点が作る外接円の曲率で、x が単調でなくても角を正しく捉える。
- **λ を昇順にソートしてから描画。** 点の並びが乱れず、線が滑らかに見える。
- **λ の自動生成（log 等間隔）を追加。** サイドバーで範囲と点数を指定でき、
  密なグリッドにすると L-curve が滑らかになり角も安定して検出できる。
  `01_lambda.csv` の値は間隔が不均一なため、これが「曲線がきれいにならない」
  主因だった。
- **ユーザーが角を選べるスライダーを追加。** 自動検出した角を既定にしつつ、
  L-curve を見ながらスライダーで λ を動かすと、選択点が青で強調され、
  その λ での fitting と k_LDF 分布が即座に更新される（再計算なし）。

> Tikhonov 正則化では L-curve の角(変曲点)に対応する λ を選ぶのが定石です。
> 非負制約(NNLS)を課すと曲線が多少ギザつくことがあるため、密な log 等間隔
> グリッドを使うと角が見つけやすくなります。

## ノートからのその他の変更点

- `print()` → 画面表示、`plt.show()` → `st.pyplot()`
- α値・最適化手法・使用λ → サイドバー / スライダーのウィジェット
- `output/` フォルダへの保存 → ブラウザからの ZIP ダウンロードに変更
- 単一列CSVの読み込みを CRLF・区切り文字に強い形に修正

## Streamlit Community Cloud で公開（無料）

1. `app.py`・`requirements.txt`・サンプルCSVを **GitHub の公開リポジトリ** に置く
2. https://share.streamlit.io に GitHub でサインイン
3. 「New app」→ リポジトリ / ブランチ / `app.py` を選択 →「Deploy」
4. 数分で公開 URL が発行されます

> 実データが機密の場合は公開リポジトリに入れないでください。アプリ起動後に
> ブラウザからアップロードする運用にすれば、データをリポジトリに置かずに済みます。
