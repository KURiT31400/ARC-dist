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

st.set_page_config(page_title="LDF kinetics analysis", layout="wide")

# Reference the default data files (sec / k / lambda) that ship alongside
# this app using an absolute path. In online environments (e.g. Streamlit
# Cloud) the working directory can vary, so we don't rely solely on a
# relative path like "01_sec.csv".
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SEC_PATH = BASE_DIR / "01_sec.csv"
DEFAULT_K_PATH = BASE_DIR / "02_k.csv"
DEFAULT_LAMBDA_PATH = BASE_DIR / "03_lambda.csv"


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
# Detect the L-curve corner (inflection point) using Menger curvature
#   Curvature of the circle through 3 points: c = 4*Area / (a*b*c).
#   Works even when x is not monotonic, and is more robust for corner
#   detection than finite differences.
# --------------------------------------------------------------------------
def menger_curvature(x, y):
    n = len(x)
    curv = np.zeros(n)  # endpoints are 0
    for i in range(1, n - 1):
        x1, y1 = x[i - 1], y[i - 1]
        x2, y2 = x[i],     y[i]
        x3, y3 = x[i + 1], y[i + 1]
        area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        a = np.hypot(x2 - x1, y2 - y1)
        b = np.hypot(x3 - x2, y3 - y2)
        c = np.hypot(x3 - x1, y3 - y1)
        denom = a * b * c
        curv[i] = 4.0 * abs(area) / denom if denom > 0 else 0.0
    return curv


def _cross2(ox, oy, ax, ay, bx, by):
    """z-component of the 2D cross product (a-o) x (b-o)."""
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def detect_corner(x, y):
    """Return the index of the L-curve corner (inflection point).

    Taking the global maximum of curvature can pick up a jagged, spurious
    kink on the flat, over-regularized branch (large log||Ax-b||). Instead,
    we only consider points that curve toward the "ideal corner" side
    (minimum residual and minimum norm), and among those pick the point
    with maximum curvature. Both endpoints are automatically excluded
    since they are unstable.
    """
    n = len(x)
    if n < 3:
        return 0
    curv = menger_curvature(x, y)
    ideal_x, ideal_y = float(np.min(x)), float(np.min(y))  # small residual, small norm
    cand = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        # For the chord P_{i-1}—P_{i+1}, point i curves toward the origin
        # side if it's on the same side as the ideal corner.
        side_pt = _cross2(x[i - 1], y[i - 1], x[i + 1], y[i + 1], x[i], y[i])
        side_id = _cross2(x[i - 1], y[i - 1], x[i + 1], y[i + 1], ideal_x, ideal_y)
        cand[i] = side_pt * side_id > 0
    cc = np.where(cand, curv, 0.0)
    return int(np.argmax(cc)) if cc.any() else int(np.argmax(curv))


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
    alpha = st.number_input("alpha value (=A_ini/A_end)", value=0.0021233689, format="%.9f")
    method = st.radio("Optimization method", ["NNLS + Tikhonov", "SLSQP (non-negative constraint)"])

    # sec / k / lambda normally use the default files bundled with the app.
    # Only advanced users need to open this expander to override them.
    with st.expander("Advanced settings (usually no need to change)"):
        st.caption("Only upload these if you need to override the time grid, k values, "
                   "or lambda list. If left blank, the bundled default files are used.")
        f_sec = st.file_uploader("01_sec.csv — time grid for computation (leave blank to use default)",
                                 type=["csv", "txt"])
        f_k = st.file_uploader("02_k.csv — k values (leave blank to use default)",
                               type=["csv", "txt"])

        st.subheader("Lambda grid")
        lam_source = st.radio(
            "How to specify lambda",
            ["Auto-generate (log-spaced, recommended)", "From file (03_lambda.csv)"],
            help="A dense, log-spaced grid makes the L-curve smoother and the corner easier to detect.",
        )
        if lam_source.startswith("Auto"):
            lam_min_exp = st.number_input("log10(lambda min)", value=-4.0, step=0.5)
            lam_max_exp = st.number_input("log10(lambda max)", value=3.0, step=0.5)
            lam_n = st.slider("Number of lambda points", 20, 200, 60)
            f_lam = None
        else:
            lam_min_exp = lam_max_exp = lam_n = None
            f_lam = st.file_uploader("03_lambda.csv — list of lambda values", type=["csv", "txt"])

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

    if lam_source.startswith("Auto"):
        lambda_values = np.logspace(lam_min_exp, lam_max_exp, int(lam_n))
    else:
        lambda_values = load_single_column(f_lam, DEFAULT_LAMBDA_PATH)

    sec_data = pd.DataFrame({"sec": t_values})
    sec_data["F"] = np.interp(sec_data["sec"], exp_data["t"], exp_data["F"])

    kernel = 1 - np.exp(-alpha * np.outer(t_values, k_values))
    A = kernel
    b = sec_data["F"].to_numpy()

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

    x = df_results["log||Ax-b||"].to_numpy()
    y = df_results["log||x||"].to_numpy()
    corner_idx = detect_corner(x, y)

    st.session_state["res"] = dict(
        df_results=df_results, df_weight=df_weight,
        kernel=kernel, sec_data=sec_data, k_values=k_values,
        corner_idx=corner_idx, A_shape=A.shape,
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

    # Interactive lambda selection (no recomputation)
    # The default value is passed via session_state (using both value= and
    # key= together would trigger a warning).
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

    m1, m2 = st.columns(2)
    m1.metric("Auto selected λ", f"{opt_lambda:g}")
    m2.metric("Selecting λ", f"{row_lambda:g}")

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
