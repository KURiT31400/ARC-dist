"""
LDF kinetics analysis — Streamlit app
速度定数分布(k_LDF)を実験のuptakeデータから逆解析する。
Tikhonov正則化 + NNLS / SLSQP、L-curve法で最適λ(変曲点/角)を選択。

LDF.ipynb を Streamlit アプリに変換したもの。
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize, nnls
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="LDF kinetics analysis", layout="wide")

# アプリと同じフォルダに置いた既定データファイル（sec / k）を
# 絶対パスで参照する。オンライン(Streamlit Cloud等)ではカレントディレクトリが
# 実行環境によって変わりうるため、相対パス"01_sec.csv"だけに頼らない。
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SEC_PATH = BASE_DIR / "01_sec.csv"
DEFAULT_K_PATH = BASE_DIR / "02_k.csv"


# --------------------------------------------------------------------------
# データ読み込みユーティリティ
# --------------------------------------------------------------------------
def load_single_column(uploaded, default_path):
    """1列のCSV(CRLF可)を1次元配列で読む。"""
    src = uploaded if uploaded is not None else default_path
    df = pd.read_csv(src, header=None)
    return df.iloc[:, 0].to_numpy(dtype=float)


def load_exp(uploaded, default_path):
    """実験データ (t, F) の2列を読む。タブ/カンマ/空白いずれも許容。"""
    src = uploaded if uploaded is not None else default_path
    df = pd.read_csv(src, sep=r"[\t,]", engine="python", header=None)
    df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
    df.columns = ["t", "F"]
    return df


# --------------------------------------------------------------------------
# 最適化 (NNLS + Tikhonov 正則化)
# --------------------------------------------------------------------------
def opt_NNLS(A, b, lambda_values):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    results, weights = [], []
    for lam in lambda_values:
        A_reg = np.vstack([A, np.sqrt(lam) * np.eye(A.shape[1])])
        b_reg = np.concatenate([b, np.zeros(A.shape[1])])
        x_nnls, _ = nnls(A_reg, b_reg)
        Ax_b = np.linalg.norm(A @ x_nnls - b)
        norm_x = np.linalg.norm(x_nnls)
        L_value = Ax_b ** 2 + lam * norm_x ** 2
        results.append([lam, L_value,
                        np.log(Ax_b) if Ax_b > 0 else -np.inf,
                        np.log(norm_x) if norm_x > 0 else -np.inf])
        weights.append(x_nnls.tolist())
    return _pack(results, weights)


# --------------------------------------------------------------------------
# 最適化 (SLSQP, 非負制約)
# --------------------------------------------------------------------------
def opt_SLSQP(A, b, lambda_values):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x0 = np.full(A.shape[1], 0.5)
    results, weights = [], []
    for lam in lambda_values:
        def objective(x):
            return np.linalg.norm(A @ x - b) ** 2 + lam * np.linalg.norm(x) ** 2
        cons = [{"type": "ineq", "fun": lambda x, i=i: x[i]} for i in range(len(x0))]
        res = minimize(objective, x0, constraints=cons, method="SLSQP",
                       options={"maxiter": 100000})
        if not res.success:
            continue
        Ax_b = np.linalg.norm(A @ res.x - b)
        norm_x = np.linalg.norm(res.x)
        results.append([lam, objective(res.x),
                        np.log(Ax_b) if Ax_b > 0 else -np.inf,
                        np.log(norm_x) if norm_x > 0 else -np.inf])
        weights.append(res.x.tolist())
    return _pack(results, weights)


def _pack(results, weights):
    df_results = pd.DataFrame(
        results, columns=["Lambda", "L_value", "log||Ax-b||", "log||x||"])
    df_weight = pd.DataFrame({"weight, x": weights})
    return df_results, df_weight


# --------------------------------------------------------------------------
# L-curve の角(変曲点)の検出
#   軸の尺度・λ点の粗密・データの大きさに依存しないよう、
#   1) 両軸を 0〜1 に正規化  2) 曲線を弧長で等間隔に再サンプル
#   3) ガウス平滑  4) 符号付き曲率が最大の点を角とする
#   (λ昇順で右下がりの曲線が「下向き→右向き」に曲がる側の角のみ候補)
#   生の log 値の3点曲率(Menger)は、点が密な急峻枝や飽和した小λ側を
#   角と誤認しやすかったため置き換えた。
# --------------------------------------------------------------------------
def detect_corner(x, y, n_resample=400, smooth=0.02, edge=0.05):
    """L-curveの角のインデックス(元配列上)を返す。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.where(np.isfinite(x) & np.isfinite(y))[0]
    if len(ok) < 4:
        return int(ok[0]) if len(ok) else 0
    xs, ys = x[ok], y[ok]
    rx, ry = np.ptp(xs), np.ptp(ys)
    if rx == 0 or ry == 0:
        return int(ok[len(ok) // 2])
    xn, yn = (xs - xs.min()) / rx, (ys - ys.min()) / ry
    seg = np.hypot(np.diff(xn), np.diff(yn))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] == 0:
        return int(ok[len(ok) // 2])
    u = np.linspace(0.0, s[-1], n_resample)
    xu, yu = np.interp(u, s, xn), np.interp(u, s, yn)
    sig = max(1.0, smooth * n_resample)
    xu = gaussian_filter1d(xu, sig, mode="nearest")
    yu = gaussian_filter1d(yu, sig, mode="nearest")
    du = u[1] - u[0]
    x1, y1 = np.gradient(xu, du), np.gradient(yu, du)
    x2, y2 = np.gradient(x1, du), np.gradient(y1, du)
    kappa = (x1 * y2 - y1 * x2) / np.maximum((x1 ** 2 + y1 ** 2) ** 1.5, 1e-12)
    m = max(1, int(edge * n_resample))  # ends are affected by smoothing: exclude
    kk = kappa.copy()
    kk[:m] = -np.inf
    kk[-m:] = -np.inf
    j = int(np.argmax(kk))
    if kk[j] <= 0:  # no corner convex toward the origin: fall back to max |curvature|
        j = int(np.argmax(np.abs(np.where(np.isfinite(kk), kk, 0.0))))
    d = np.hypot(xn - xu[j], yn - yu[j])
    return int(ok[int(np.argmin(d))])


# --------------------------------------------------------------------------
# サイドバー: 入力
# --------------------------------------------------------------------------
st.title("LDF kinetics analysis")
st.caption("速度定数分布 k_LDF を実験uptakeデータから逆解析（Tikhonov正則化 + L-curve法）")

with st.sidebar:
    st.header("1. 実験データ")
    st.caption("t, F の2列CSV/TXTをアップロードしてください（タブ/カンマ/空白いずれも可）")
    f_exp = st.file_uploader("実験データ (t, F)", type=["csv", "txt"])

    st.header("2. パラメータ")
    alpha = st.number_input("α値 (=A_ini/A_end)", value=1.193975685, format="%.9f")
    qe_in = st.number_input(
        "平衡吸着量 qₑ（任意）", value=None, min_value=0.0, format="%.6g",
        help="入力すると F を q/qₑ に規格化して解析します。空欄ならファイルのFをそのまま使います。"
             "α を q₀/qₑ で定義している場合は、同じ qₑ で α も計算し直してください。",
    )
    method = st.radio("最適化手法", ["NNLS + Tikhonov", "SLSQP (非負制約)"])

    # sec / k は通常アプリに同梱した既定ファイルを使う。
    # 上級者が差し替えたい場合だけ、折りたたみを開いてアップロードする。
    with st.expander("詳細設定（通常は変更不要）"):
        st.caption("時間グリッド・k値を差し替える場合のみアップロードしてください。"
                   "未指定なら同梱の既定ファイルを使用します。")
        f_sec = st.file_uploader("01_sec.csv — 計算用の時間 sec（既定ファイルを使う場合は空欄）",
                                 type=["csv", "txt"])
        f_k = st.file_uploader("02_k.csv — k値（既定ファイルを使う場合は空欄）",
                               type=["csv", "txt"])

        st.subheader("λ グリッド")
        lam_source = st.radio(
            "λ の与え方",
            ["自動 (Aのスケール基準・推奨)", "手動 (log等間隔・絶対値)"],
            help="推奨: λ を A の最大特異値の2乗 σ₁² に対する比で振る。データの大きさ・αが変わっても"
                 "L-curve の角が範囲内に入りやすい。",
        )
        if lam_source.startswith("自動"):
            lam_min_exp = st.number_input("log10(λ/σ₁²) 最小", value=-14.0, step=0.5)
            lam_max_exp = st.number_input("log10(λ/σ₁²) 最大", value=0.0, step=0.5)
            lam_n = st.slider("λ 点数", 20, 200, 60)
        else:
            lam_min_exp = st.number_input("log10(λ最小)", value=-4.0, step=0.5)
            lam_max_exp = st.number_input("log10(λ最大)", value=3.0, step=0.5)
            lam_n = st.slider("λ 点数", 20, 200, 60)

    run = st.button("計算を実行", type="primary", use_container_width=True,
                    disabled=f_exp is None)
    if f_exp is None:
        st.caption("⬆ 実験データをアップロードすると計算を実行できます。")


# --------------------------------------------------------------------------
# 計算（ボタン押下時のみ実行し、結果を session_state に保持）
# --------------------------------------------------------------------------
def compute():
    # 既定ファイル（sec/k）が同梱されているか先にチェックし、
    # デプロイし忘れの場合はここで分かりやすく知らせる。
    missing = [p.name for p in (DEFAULT_SEC_PATH, DEFAULT_K_PATH)
               if not p.exists()]
    if f_sec is None and DEFAULT_SEC_PATH.name in missing:
        st.error(f"既定ファイル {DEFAULT_SEC_PATH.name} が見つかりません。"
                 "アプリと同じフォルダに配置するか、詳細設定からアップロードしてください。")
        return
    if f_k is None and DEFAULT_K_PATH.name in missing:
        st.error(f"既定ファイル {DEFAULT_K_PATH.name} が見つかりません。"
                 "アプリと同じフォルダに配置するか、詳細設定からアップロードしてください。")
        return

    exp_data = load_exp(f_exp, None)
    t_values = load_single_column(f_sec, DEFAULT_SEC_PATH)
    k_values = load_single_column(f_k, DEFAULT_K_PATH)

    sec_data = pd.DataFrame({"sec": t_values})
    sec_data["F"] = np.interp(sec_data["sec"], exp_data["t"], exp_data["F"])
    qe_used = float(qe_in) if (qe_in is not None and qe_in > 0) else None
    if qe_used:
        sec_data["F"] = sec_data["F"] / qe_used

    kernel = 1 - np.exp(-alpha * np.outer(t_values, k_values))
    A = kernel
    b = sec_data["F"].to_numpy()

    sigma1 = float(np.linalg.norm(A, 2))
    if lam_source.startswith("自動"):
        lambda_values = sigma1 ** 2 * np.logspace(lam_min_exp, lam_max_exp, int(lam_n))
    else:
        lambda_values = np.logspace(lam_min_exp, lam_max_exp, int(lam_n))

    if method.startswith("NNLS"):
        df_results, df_weight = opt_NNLS(A, b, lambda_values)
    else:
        df_results, df_weight = opt_SLSQP(A, b, lambda_values)

    if df_results.empty:
        st.session_state.pop("res", None)
        st.error("最適化に成功したλがありません。入力を確認してください。")
        return

    # λ昇順にソート（L-curveを滑らかに描くため）
    order = np.argsort(df_results["Lambda"].to_numpy(), kind="stable")
    df_results = df_results.iloc[order].reset_index(drop=True)
    df_weight = df_weight.iloc[order].reset_index(drop=True)

    # 残差(rms)と λ/σ₁² を L_curve.csv にも出力する
    _rms = np.exp(df_results["log||Ax-b||"].to_numpy()) / np.sqrt(len(b))
    df_results["rms residual"] = _rms
    df_results["rms residual [% of F_max]"] = 100.0 * _rms / float(np.max(np.abs(b)))
    df_results["Lambda/sigma1^2"] = df_results["Lambda"] / sigma1 ** 2

    x = df_results["log||Ax-b||"].to_numpy()
    y = df_results["log||x||"].to_numpy()
    corner_idx = detect_corner(x, y)

    st.session_state["res"] = dict(
        df_results=df_results, df_weight=df_weight,
        kernel=kernel, sec_data=sec_data, k_values=k_values,
        corner_idx=corner_idx, A_shape=A.shape,
        edge_warn=bool(corner_idx <= 2 or corner_idx >= len(x) - 3),
        qe_used=qe_used, f_max=float(sec_data["F"].max()),
        sigma1=sigma1, n_pts=len(b),
        n_lambda=len(lambda_values),
    )
    # 既定の選択を角にリセット
    st.session_state["sel_idx"] = corner_idx


if run:
    compute()


# --------------------------------------------------------------------------
# 表示（結果があればスライダーで再計算なしにλを選べる）
# --------------------------------------------------------------------------
def display():
    r = st.session_state["res"]
    df_results = r["df_results"]
    df_weight = r["df_weight"]
    kernel = r["kernel"]
    sec_data = r["sec_data"]
    k_values = r["k_values"]
    corner_idx = r["corner_idx"]

    x = df_results["log||Ax-b||"].to_numpy()
    y = df_results["log||x||"].to_numpy()
    z = df_results["Lambda"].to_numpy()

    c1, c2, c3 = st.columns(3)
    c1.metric("A (kernel) の形状", f"{r['A_shape'][0]} × {r['A_shape'][1]}")
    c2.metric("b の長さ", f"{len(sec_data)}")
    c3.metric("λ 個数", f"{r['n_lambda']}")

    st.subheader("L-curve — Select infelction point")
    st.caption("スライダーで L-curve 上の点(=λ)を動かして角を選べます。既定は自動検出した角。")
    if r.get("qe_used"):
        st.caption(f"F を q/qₑ に規格化（qₑ = {r['qe_used']:g}）。規格化後の F の最大値 = {r['f_max']:.3f}")
        if r["f_max"] > 1.0 + 1e-9:
            st.warning("実験値が qₑ を超えています。qₑ の値を確認してください。")
    if r.get("edge_warn"):
        st.warning("自動検出した角が λ 範囲の端にあります。λ の範囲を広げて再計算してください。")

    # インタラクティブなλ選択（再計算なし）
    # 既定値は session_state 経由で渡す（value= と key= の併用は警告になるため）
    # 残差(rms, F_maxに対する%)。λを大きくするほど増える。
    res_pct = 100.0 * (np.exp(x) / np.sqrt(r["n_pts"])) / r["f_max"]

    def _jump_to_residual():
        ok = np.where(res_pct <= st.session_state["target_res_pct"])[0]
        st.session_state["sel_idx"] = int(ok[-1]) if len(ok) else 0

    jc1, jc2, _jc3 = st.columns([1, 1, 2])
    jc1.number_input("目標残差 [% of F_max, rms]", min_value=0.0, value=0.1, step=0.05,
                     format="%.3f", key="target_res_pct",
                     help="残差(rms)がこの値以下になる最大のλにスライダーを移動します。")
    jc2.write("")
    jc2.button("この残差以下で最大のλを選ぶ", on_click=_jump_to_residual)

    if "sel_idx" not in st.session_state:
        st.session_state["sel_idx"] = corner_idx
    st.session_state["sel_idx"] = min(st.session_state["sel_idx"], len(z) - 1)
    sel_idx = st.slider(
        "λ index（Left = small λ / Right = large λ）",
        min_value=0, max_value=len(z) - 1,
        key="sel_idx",
    )
    row_lambda = float(z[sel_idx])
    opt_lambda = float(z[corner_idx])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Auto selected λ", f"{opt_lambda:g}")
    m2.metric("Selecting λ", f"{row_lambda:g}")
    m3.metric("残差 @Auto (rms, %F_max)", f"{res_pct[corner_idx]:.3f} %")
    m4.metric("残差 @Selecting (rms, %F_max)", f"{res_pct[sel_idx]:.3f} %")
    st.caption(f"Selecting: log10(λ/σ₁²) = {np.log10(row_lambda / r['sigma1'] ** 2):.2f}")

    with st.expander("λ と残差の一覧"):
        tbl = pd.DataFrame({
            "index": np.arange(len(z)), "λ": z,
            "log10(λ/σ₁²)": np.log10(z / r["sigma1"] ** 2),
            "rms residual": np.exp(x) / np.sqrt(r["n_pts"]),
            "rms residual [% of F_max]": res_pct,
            "log||Ax-b||": x, "log||x||": y,
        })
        tbl["mark"] = ""
        tbl.loc[corner_idx, "mark"] = "Auto"
        tbl.loc[sel_idx, "mark"] = "Auto+Selecting" if sel_idx == corner_idx else "Selecting"
        st.dataframe(tbl, use_container_width=True, height=300)

   # L-curve 描画（小さめ・左半分に収めて fitting/分布 を同時に見やすく）
    fig_l, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y, "-o", color="black", markersize=3, label="L-curve", zorder=1)
    ax.scatter(x[corner_idx], y[corner_idx], s=55, facecolors="none",
               edgecolors="red", linewidths=1.5, label="Auto selected λ", zorder=4)
    ax.scatter(x[sel_idx], y[sel_idx], s=45, color="blue",
               label="Selecting", zorder=5)
    ax.annotate(f"λ={row_lambda:g}", (x[sel_idx], y[sel_idx]),
                textcoords="offset points", xytext=(6, 4), color="blue", fontsize=6)
    ax.set_xlabel("log||Ax-b||", fontsize=9)
    ax.set_ylabel("log||x||", fontsize=9)
    ax.set_title("L-curve", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig_l.tight_layout()
    lcol, _ = st.columns(2)
    with lcol:
        st.pyplot(fig_l, use_container_width=False)

    # 選択したλで fitting & 分布
    st.subheader("Fitting & 分布")
    dist_y = np.array(df_weight.loc[sel_idx, "weight, x"], dtype=float)
    dist_x = k_values
    uptake_x = sec_data["sec"].to_numpy()
    uptake_y_exp = sec_data["F"].to_numpy()
    uptake_y_fit = kernel @ dist_y

    g1, g2 = st.columns(2)
    with g1:
        fig1, a1 = plt.subplots(figsize=(5, 4))
        a1.plot(uptake_x, uptake_y_exp, label="Experimental", color="red")
        a1.plot(uptake_x, uptake_y_fit, "o", markersize=3, markerfacecolor="white", label="Fitting", color="green")
        a1.set_xscale("log")
        a1.set_title("Uptake curve")
        a1.set_xlabel("sec")
        a1.set_ylabel("Uptake")
        a1.legend()
        st.pyplot(fig1)
    with g2:
        fig2, a2 = plt.subplots(figsize=(5, 4))
        #a2.plot(dist_x, dist_y, label="Distribution of k_LDF", color="blue")
        a2.plot(dist_x, dist_y, color="blue")
        a2.set_xscale("log")
        a2.set_title("k_LDF distribution")
        a2.set_xlabel("k_LDF [1/s]")
        a2.set_ylabel("weight [-]")
        #a2.legend()
        st.pyplot(fig2)

    # 出力
    kernel_df = pd.DataFrame(kernel, index=sec_data["sec"].to_numpy(), columns=k_values)
    k_dist = pd.DataFrame({"k_LDF[1/s]": dist_x, "Weight[-]": dist_y})
    lambda_set = pd.DataFrame({"Set": [row_lambda], "Optimal": [opt_lambda]})

    st.subheader("出力データ")
    st.dataframe(k_dist, use_container_width=True, height=250)

    outputs = {
        "set_tF.csv": sec_data.to_csv(sep="\t", index=False),
        "kernel.csv": kernel_df.to_csv(sep="\t", index=False),
        "L_curve.csv": df_results.to_csv(sep="\t", index=False),
        "opt_weight.csv": df_weight.to_csv(sep="\t", index=False),
        "k_distribution.csv": k_dist.to_csv(sep="\t", index=False),
        "lambda_set.csv": lambda_set.to_csv(sep="\t", index=False),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in outputs.items():
            zf.writestr(name, content)
    st.download_button(
        "全出力をZIPでダウンロード", data=buf.getvalue(),
        file_name="LDF_output.zip", mime="application/zip",
        use_container_width=True,
    )


if "res" in st.session_state:
    display()
else:
    st.info("左のサイドバーで入力を確認し、「計算を実行」を押してください。")
