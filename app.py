"""
LDF kinetics analysis — Streamlit app
Inversely solves for the rate-constant distribution (k_LDF) from experimental
uptake data. Uses Tikhonov regularization + NNLS / SLSQP, with the L-curve
method to pick the optimal lambda (inflection point / corner).

Converted from LDF.ipynb into a Streamlit app.
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

# Reference the default data files (sec / k) that ship alongside
# this app using an absolute path. In online environments (e.g. Streamlit
# Cloud) the working directory can vary, so we don't rely solely on a
# relative path like "01_sec.csv".
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SEC_PATH = BASE_DIR / "01_sec.csv"
DEFAULT_K_PATH = BASE_DIR / "02_k.csv"


# --------------------------------------------------------------------------
# Data loading utilities
# --------------------------------------------------------------------------
def load_single_column(uploaded, default_path):
    """Read a single-column CSV (CRLF allowed) into a 1D array."""
    src = uploaded if uploaded is not None else default_path
    df = pd.read_csv(src, header=None)
    return df.iloc[:, 0].to_numpy(dtype=float)


def load_exp(uploaded, default_path):
    """Read the two experimental columns (t, F). Tab/comma/whitespace all allowed."""
    src = uploaded if uploaded is not None else default_path
    df = pd.read_csv(src, sep=r"[\t,]", engine="python", header=None)
    df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
    df.columns = ["t", "F"]
    return df


# --------------------------------------------------------------------------
# Optimization (NNLS + Tikhonov regularization)
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
# Optimization (SLSQP, non-negativity constraint)
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
# Detect the L-curve corner (inflection point)
#   To be independent of axis scale, lambda-point density and data magnitude:
#   1) normalize both axes to 0-1  2) resample the curve at equal arc length
#   3) Gaussian smoothing  4) take the point of maximum signed curvature
#   (only corners where a right-descending curve (lambda ascending) turns from
#   "downward" to "rightward" are candidates).
#   The old 3-point Menger curvature on raw log values tended to mistake the
#   densely sampled steep branch / saturated small-lambda end for the corner.
# --------------------------------------------------------------------------
def detect_corner(x, y, n_resample=400, smooth=0.02, edge=0.05):
    """Return the index (in the original arrays) of the L-curve corner."""
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
# Sidebar: inputs
# --------------------------------------------------------------------------
st.title("LDF kinetics analysis")
st.caption("Inverse analysis of the rate-constant distribution k_LDF from experimental uptake data (Tikhonov regularization + L-curve method)")

with st.sidebar:
    st.header("1. Experimental data")
    st.caption("Upload a 2-column CSV/TXT of (t, F) — tab, comma, or whitespace separated are all fine")
    f_exp = st.file_uploader("Experimental data (t, F)", type=["csv", "txt"])

    st.header("2. Parameters")
    alpha = st.number_input("alpha value (=A_ini/A_end)", value=1.193975685, format="%.9f")
    qe_in = st.number_input(
        "Equilibrium uptake q_e (optional)", value=None, min_value=0.0, format="%.6g",
        help="If given, F is normalized to q/q_e before the analysis. Leave empty to use F in the file as it is. "
             "If you define alpha as q0/q_e, recompute alpha with the same q_e.",
    )
    method = st.radio("Optimization method", ["NNLS + Tikhonov", "SLSQP (non-negative constraint)"])

    # sec / k normally use the default files bundled with the app.
    # Only advanced users need to open this expander to override them.
    with st.expander("Advanced settings (usually no need to change)"):
        st.caption("Only upload these if you need to override the time grid or k values. "
                   "If left blank, the bundled default files are used.")
        f_sec = st.file_uploader("01_sec.csv — time grid for computation (leave blank to use default)",
                                 type=["csv", "txt"])
        f_k = st.file_uploader("02_k.csv — k values (leave blank to use default)",
                               type=["csv", "txt"])

        st.subheader("Lambda grid")
        lam_source = st.radio(
            "How to specify lambda",
            ["Auto (scaled to A, recommended)", "Manual (log-spaced, absolute)"],
            help="Recommended: lambda is scanned as a ratio to sigma1^2 (largest singular value of A, squared). "
                 "The L-curve corner then stays inside the range even when the data magnitude or alpha changes.",
        )
        if lam_source.startswith("Auto"):
            lam_min_exp = st.number_input("log10(lambda / sigma1^2) min", value=-14.0, step=0.5)
            lam_max_exp = st.number_input("log10(lambda / sigma1^2) max", value=0.0, step=0.5)
            lam_n = st.slider("Number of lambda points", 20, 200, 60)
        else:
            lam_min_exp = st.number_input("log10(lambda min)", value=-4.0, step=0.5)
            lam_max_exp = st.number_input("log10(lambda max)", value=3.0, step=0.5)
            lam_n = st.slider("Number of lambda points", 20, 200, 60)

    run = st.button("Run calculation", type="primary", use_container_width=True,
                    disabled=f_exp is None)
    if f_exp is None:
        st.caption("⬆ Upload your experimental data to enable the calculation.")


# --------------------------------------------------------------------------
# Computation (runs only on button press; result is kept in session_state)
# --------------------------------------------------------------------------
def compute():
    # Check upfront whether the default files (sec/k) are present, and give
    # a clear message here if the deployment forgot to include them.
    missing = [p.name for p in (DEFAULT_SEC_PATH, DEFAULT_K_PATH)
               if not p.exists()]
    if f_sec is None and DEFAULT_SEC_PATH.name in missing:
        st.error(f"Default file {DEFAULT_SEC_PATH.name} was not found. "
                 "Place it in the same folder as the app, or upload it via Advanced settings.")
        return
    if f_k is None and DEFAULT_K_PATH.name in missing:
        st.error(f"Default file {DEFAULT_K_PATH.name} was not found. "
                 "Place it in the same folder as the app, or upload it via Advanced settings.")
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
    if lam_source.startswith("Auto"):
        lambda_values = sigma1 ** 2 * np.logspace(lam_min_exp, lam_max_exp, int(lam_n))
    else:
        lambda_values = np.logspace(lam_min_exp, lam_max_exp, int(lam_n))

    if method.startswith("NNLS"):
        df_results, df_weight = opt_NNLS(A, b, lambda_values)
    else:
        df_results, df_weight = opt_SLSQP(A, b, lambda_values)

    if df_results.empty:
        st.session_state.pop("res", None)
        st.error("No lambda value converged successfully. Please check your inputs.")
        return

    # Sort by ascending lambda (to draw a smooth L-curve)
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
    # Reset the default selection to the detected corner
    st.session_state["sel_idx"] = corner_idx


if run:
    compute()


# --------------------------------------------------------------------------
# Display (if a result exists, lambda can be picked via slider without
# recomputing)
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
    c1.metric("Shape of A (kernel)", f"{r['A_shape'][0]} x {r['A_shape'][1]}")
    c2.metric("Length of b", f"{len(sec_data)}")
    c3.metric("Number of lambda values", f"{r['n_lambda']}")

    st.subheader("L-curve — Select inflection point")
    st.caption("Use the slider to move along the L-curve (= lambda) and select the corner. "
               "Default is the automatically detected corner.")
    if r.get("qe_used"):
        st.caption(f"F normalized as q/q_e (q_e = {r['qe_used']:g}). Max of normalized F = {r['f_max']:.3f}")
        if r["f_max"] > 1.0 + 1e-9:
            st.warning("Experimental values exceed q_e. Please check the q_e value.")
    if r.get("edge_warn"):
        st.warning("The auto-detected corner is at the edge of the lambda range. "
                   "Widen the lambda range and recompute.")

    # Interactive lambda selection (no recomputation)
    # The default value is passed via session_state (using both value= and
    # key= together would trigger a warning).
    # Residual (rms, % of F_max). It grows as lambda increases.
    res_pct = 100.0 * (np.exp(x) / np.sqrt(r["n_pts"])) / r["f_max"]

    def _jump_to_residual():
        ok = np.where(res_pct <= st.session_state["target_res_pct"])[0]
        st.session_state["sel_idx"] = int(ok[-1]) if len(ok) else 0

    jc1, jc2, _jc3 = st.columns([1, 1, 2])
    jc1.number_input("Target residual [% of F_max, rms]", min_value=0.0, value=0.1, step=0.05,
                     format="%.3f", key="target_res_pct",
                     help="Moves the slider to the largest lambda whose rms residual is at or below this value.")
    jc2.write("")
    jc2.button("Select largest lambda below this residual", on_click=_jump_to_residual)

    if "sel_idx" not in st.session_state:
        st.session_state["sel_idx"] = corner_idx
    st.session_state["sel_idx"] = min(st.session_state["sel_idx"], len(z) - 1)
    sel_idx = st.slider(
        "Lambda index (Left = small lambda / Right = large lambda)",
        min_value=0, max_value=len(z) - 1,
        key="sel_idx",
    )
    row_lambda = float(z[sel_idx])
    opt_lambda = float(z[corner_idx])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Auto selected λ", f"{opt_lambda:g}")
    m2.metric("Selecting λ", f"{row_lambda:g}")
    m3.metric("Residual @Auto (rms, %F_max)", f"{res_pct[corner_idx]:.3f} %")
    m4.metric("Residual @Selecting (rms, %F_max)", f"{res_pct[sel_idx]:.3f} %")
    st.caption(f"Selecting: log10(lambda/sigma1^2) = {np.log10(row_lambda / r['sigma1'] ** 2):.2f}")

    with st.expander("Table of lambda and residual"):
        tbl = pd.DataFrame({
            "index": np.arange(len(z)), "lambda": z,
            "log10(lambda/sigma1^2)": np.log10(z / r["sigma1"] ** 2),
            "rms residual": np.exp(x) / np.sqrt(r["n_pts"]),
            "rms residual [% of F_max]": res_pct,
            "log||Ax-b||": x, "log||x||": y,
        })
        tbl["mark"] = ""
        tbl.loc[corner_idx, "mark"] = "Auto"
        tbl.loc[sel_idx, "mark"] = "Auto+Selecting" if sel_idx == corner_idx else "Selecting"
        st.dataframe(tbl, use_container_width=True, height=300)

   # Draw the L-curve (kept small, fit in the left half so the fitting/
   # distribution plots can be viewed at the same time)
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

    # Fitting & distribution for the selected lambda
    st.subheader("Fitting & distribution")
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

    # Output
    kernel_df = pd.DataFrame(kernel, index=sec_data["sec"].to_numpy(), columns=k_values)
    k_dist = pd.DataFrame({"k_LDF[1/s]": dist_x, "Weight[-]": dist_y})
    lambda_set = pd.DataFrame({"Set": [row_lambda], "Optimal": [opt_lambda]})

    st.subheader("Output data")
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
        "Download all outputs as ZIP", data=buf.getvalue(),
        file_name="LDF_output.zip", mime="application/zip",
        use_container_width=True,
    )


if "res" in st.session_state:
    display()
else:
    st.info("Check your inputs in the sidebar on the left, then press \"Run calculation\".")
