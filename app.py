"""
=====================================================================
  Nonlinear Equation Solver
  Methods: Newton-Raphson / Modified Newton-Raphson / Secant
  Run with:  streamlit run app.py
=====================================================================
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# ---------------------------------------------------------------------
# 1) Page config and styling (CSS)
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Nonlinear Equation Solver",
    page_icon="📐",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, label, button {
        font-family: 'Inter', sans-serif !important;
    }
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        color: white; padding: 1.6rem 2rem; border-radius: 18px;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(79,70,229,.25);
    }
    .hero h1 { margin: 0; font-size: 2rem; color: white; }
    .hero p  { margin: .3rem 0 0; opacity: .92; }

    div[data-testid="stMetric"] {
        background: rgba(127,127,127,.08); border: 1px solid rgba(127,127,127,.2);
        padding: .9rem 1.1rem; border-radius: 14px;
    }
    .stButton > button { border-radius: 12px; font-weight: 700; padding: .6rem 1rem; }
    </style>
    <div class="hero">
        <h1>📐 Nonlinear Equation Solver</h1>
        <p>Numerical methods: Newton-Raphson • Modified Newton-Raphson • Secant</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# 2) Parsing text equations into numeric functions
# ---------------------------------------------------------------------
X = sp.Symbol("x", real=True)

# Functions and constants allowed inside the input text
ALLOWED = {
    "x": X,
    "e": sp.E,          # Euler's number e
    "pi": sp.pi,
    "exp": sp.exp,
    "ln": sp.log,       # natural logarithm
    "log": sp.log,      # log = natural logarithm
    "log10": lambda v: sp.log(v, 10),
    "sqrt": sp.sqrt,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "abs": sp.Abs,
}

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,  # allows 2x instead of 2*x
    convert_xor,                          # allows ^ instead of **
)

# Method names (used both in the UI and in the algorithm)
NEWTON = "Newton-Raphson"
MODIFIED = "Modified Newton-Raphson"
SECANT = "Secant"


def parse_function(text: str) -> sp.Expr:
    """Convert text to a SymPy expression and make sure x is the only variable."""
    expr = parse_expr(text.strip(), local_dict=ALLOWED, transformations=TRANSFORMS)
    extra = expr.free_symbols - {X}
    if extra:
        raise ValueError(
            f"Unknown symbols: {', '.join(map(str, extra))} (only the variable x is allowed)"
        )
    return expr


def to_numeric(expr: sp.Expr):
    """Convert a SymPy expression into a fast NumPy function."""
    return sp.lambdify(X, expr, modules=["numpy"])


def safe_eval(fn, value: float) -> float:
    """Safe evaluation (returns nan on mathematical errors)."""
    try:
        with np.errstate(all="ignore"):
            return float(np.real(fn(value)))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------
# 3) Numerical algorithms
# ---------------------------------------------------------------------
@dataclass
class SolveResult:
    table: pd.DataFrame
    root: float
    converged: bool
    message: str


COLUMNS = ["i", "x_i", "f(x_i)", "f'(x_i) / slope", "x_{i+1}", "|x_{i+1} - x_i|"]


def solve_iterative(method, f, df, x0, tol, max_iter, multiplicity=1.0):
    """
    Run the selected method and return the iteration table.

    Newton-Raphson :  x_{i+1} = x_i - f(x_i)/f'(x_i)
    Modified Newton:  x_{i+1} = x_i - m * f(x_i)/f'(x_i)      (m = root multiplicity)
    Secant         :  x_{i+1} = x_i - f(x_i)(x_i - x_{i-1}) / (f(x_i) - f(x_{i-1}))
    """
    rows = []
    x = float(x0)
    x_prev = x - 0.01 * max(1.0, abs(x))  # second starting point (Secant only)
    converged = False
    message = "Stopping criterion was not met within the maximum number of iterations."

    for i in range(max_iter):
        fx = safe_eval(f, x)

        # --- slope depends on the method ---
        if method == SECANT:
            f_prev = safe_eval(f, x_prev)
            slope = (fx - f_prev) / (x - x_prev) if x != x_prev else 0.0
        else:
            slope = safe_eval(df, x)

        # --- safety checks ---
        if not np.isfinite(fx) or not np.isfinite(slope):
            message = f"Undefined value (NaN/inf) at x = {x:.6g}. Try a different initial guess."
            break
        if slope == 0:
            message = f"Zero slope at x = {x:.6g} (division by zero). Try a different initial guess."
            break

        # --- iteration step ---
        step = fx / slope
        if method == MODIFIED:
            step *= multiplicity
        x_new = x - step
        err = abs(x_new - x)

        rows.append([i, x, fx, slope, x_new, err])

        x_prev, x = x, x_new

        if not np.isfinite(x):
            message = "The iterations diverged. Try a different initial guess."
            break
        if err < tol:
            converged = True
            message = f"Stopping criterion met after {i + 1} iteration(s)."
            break

    table = pd.DataFrame(rows, columns=COLUMNS)
    return SolveResult(table, x, converged, message)


# ---------------------------------------------------------------------
# 4) Interactive plot (Plotly)
# ---------------------------------------------------------------------
def build_plot(f, expr_text, result: SolveResult, x0, show_path):
    root = result.root
    table = result.table

    # Plot range: covers the initial guess and the root with some margin
    span = max(abs(x0 - root), 1.0) * 1.6
    xs = np.linspace(root - span, root + span, 1200)
    with np.errstate(all="ignore"):
        ys = np.array(f(xs), dtype=float) * np.ones_like(xs)
    ys[~np.isfinite(ys)] = np.nan

    # Limit the y-range to avoid huge values near discontinuities
    finite = ys[np.isfinite(ys)]
    y_lim = None
    if finite.size:
        lo, hi = np.percentile(finite, [2, 98])
        pad = (hi - lo) * 0.25 or 1.0
        y_lim = [lo - pad, hi + pad]

    fig = go.Figure()

    # Function curve
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name="y = f(x)",
        line=dict(color="#4f46e5", width=3),
        hovertemplate="x=%{x:.5f}<br>f(x)=%{y:.5f}<extra></extra>",
    ))

    # Iteration path (zigzag: axis point -> curve point)
    if show_path and not table.empty:
        px, py = [], []
        for xi, fi in zip(table["x_i"], table["f(x_i)"]):
            px += [xi, xi]
            py += [0, fi]
        px.append(table["x_{i+1}"].iloc[-1])
        py.append(0)
        fig.add_trace(go.Scatter(
            x=px, y=py, mode="lines", name="Iteration path",
            line=dict(color="#f59e0b", width=1.5, dash="dot"),
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=table["x_i"], y=table["f(x_i)"], mode="markers+text",
            name="x_i", text=[f"x{int(i)}" for i in table["i"]],
            textposition="top center",
            marker=dict(color="#f59e0b", size=9),
        ))

    # Root point
    fig.add_trace(go.Scatter(
        x=[root], y=[0], mode="markers+text", name="Root",
        text=[f"Root ≈ {root:.6f}"], textposition="bottom right",
        marker=dict(color="#ef4444", size=15, symbol="diamond",
                    line=dict(color="white", width=2)),
        hovertemplate="Root = %{x:.10f}<extra></extra>",
    ))

    # X and Y axes
    fig.add_hline(y=0, line=dict(color="gray", width=1.5))
    fig.add_vline(x=0, line=dict(color="gray", width=1.5))

    fig.update_layout(
        title=f"f(x) = {expr_text}",
        xaxis_title="x", yaxis_title="f(x)",
        height=560, hovermode="closest",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", y=-0.15),
        template="plotly_white",
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False, range=y_lim)
    return fig


# ---------------------------------------------------------------------
# 5) Presets and UI state
# ---------------------------------------------------------------------
PRESETS = {
    "— Choose an example —": None,
    "x³ − 8x + 5 (x0 = 3)": ("x**3 - 8*x + 5", 3.0),
    "e^(−x) − x (x0 = 1)": ("e**-x - x", 1.0),
    "cos(x) − x (x0 = 1)": ("cos(x) - x", 1.0),
    "x² − 2 (x0 = 1)": ("x**2 - 2", 1.0),
    "x·sin(x) − 1 (x0 = 2)": ("x*sin(x) - 1", 2.0),
}

st.session_state.setdefault("f_text", "x**3 - 8*x + 5")
st.session_state.setdefault("x0", 3.0)


def apply_preset():
    """Fill the input fields when a preset example is selected."""
    choice = PRESETS[st.session_state["preset"]]
    if choice:
        st.session_state["f_text"], st.session_state["x0"] = choice


# ---------------------------------------------------------------------
# 6) Sidebar: user inputs
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Inputs")

    st.selectbox("Example problems", list(PRESETS), key="preset", on_change=apply_preset)

    method = st.selectbox("Solution method", [NEWTON, MODIFIED, SECANT])

    f_text = st.text_input(
        "Function f(x)", key="f_text",
        help="Examples: x**3 - 8*x + 5   or   e**-x - x   or   cos(x) - x",
    )

    auto_deriv = st.checkbox("Compute derivative f'(x) automatically", value=True)
    df_text = ""
    if not auto_deriv and method != SECANT:
        df_text = st.text_input("Derivative f'(x)", value="3*x**2 - 8")

    x0 = st.number_input("Initial guess x₀", key="x0", format="%.6f")

    c1, c2 = st.columns(2)
    tol = c1.number_input("Tolerance", value=1e-8, format="%.1e", min_value=1e-15)
    max_iter = c2.number_input("Max iterations", value=50, min_value=1, max_value=1000, step=1)

    multiplicity = 1.0
    if method == MODIFIED:
        multiplicity = st.number_input("Root multiplicity m", value=2.0, min_value=1.0, step=1.0)

    show_path = st.checkbox("Show iteration path on the plot", value=True)
    run = st.button("🚀 Solve", type="primary", use_container_width=True)

    st.caption(
        "Supported functions: sin, cos, tan, exp, ln/log, log10, sqrt, abs, sinh, cosh, tanh… "
        "Constants: e, pi. Use ^ or ** for powers."
    )

# ---------------------------------------------------------------------
# 7) Run and display results
# ---------------------------------------------------------------------
if not run:
    st.info("👈 Enter the equation and initial guess in the sidebar, then click **Solve**.")
    st.stop()

try:
    f_expr = parse_function(f_text)
    df_expr = parse_function(df_text) if df_text else sp.diff(f_expr, X)
except Exception as exc:  # equation syntax errors
    st.error(f"❌ Could not parse the equation: {exc}")
    st.stop()

f_num = to_numeric(f_expr)
df_num = to_numeric(df_expr)

result = solve_iterative(method, f_num, df_num, x0, tol, int(max_iter), multiplicity)

# --- Function and derivative in LaTeX ---
st.subheader("📝 Function and Derivative")
lc1, lc2 = st.columns(2)
lc1.latex(rf"f(x) = {sp.latex(f_expr)}")
if method != SECANT:
    lc2.latex(rf"f'(x) = {sp.latex(df_expr)}")

# --- Result summary ---
st.subheader("🎯 Result")
if result.converged:
    st.success(result.message)
else:
    st.warning(result.message)

m1, m2, m3 = st.columns(3)
m1.metric("Final root", f"{result.root:.12f}")
m2.metric("f(root)", f"{safe_eval(f_num, result.root):.3e}")
m3.metric("Iterations", len(result.table))

# --- Iteration table ---
st.subheader("📊 Iteration Table")
st.dataframe(
    result.table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "i": st.column_config.NumberColumn("i", format="%d"),
        "x_i": st.column_config.NumberColumn("x_i", format="%.10f"),
        "f(x_i)": st.column_config.NumberColumn("f(x_i)", format="%.4e"),
        "f'(x_i) / slope": st.column_config.NumberColumn("f'(x_i) / slope", format="%.6f"),
        "x_{i+1}": st.column_config.NumberColumn("x_{i+1}", format="%.10f"),
        "|x_{i+1} - x_i|": st.column_config.NumberColumn("|x_{i+1} - x_i|", format="%.3e"),
    },
)
st.download_button(
    "⬇️ Download table (CSV)",
    result.table.to_csv(index=False).encode("utf-8-sig"),
    file_name="iterations.csv",
    mime="text/csv",
)

# --- Plot ---
st.subheader("📈 Plot")
if np.isfinite(result.root):
    st.plotly_chart(build_plot(f_num, f_text, result, x0, show_path),
                    use_container_width=True)
else:
    st.error("Cannot plot the function because the root is undefined.")