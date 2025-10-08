import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import re
from datetime import date

st.set_page_config(page_title="CENACE | Node Price Spread Explorer", layout="wide")
st.title("⚡ CENACE – Node Price Spread Explorer")
st.caption("Upload multiple CENACE CSVs, then configure filters in **Settings**. Heavy computations are cached; voltage/date filters won’t recompute the expensive steps unless needed.")

# =========================================================
# Helpers
# =========================================================
def extract_voltage_from_node(nodo: str):
    """Extract voltage suffix after last '-' e.g. '01AAN-85' -> '85' (string, preserves '34.5')."""
    if not isinstance(nodo, str):
        return None
    m = re.search(r"-([0-9]+(?:\.[0-9]+)?)$", nodo.strip())
    return m.group(1) if m else None

@st.cache_data(show_spinner=False)
def parse_cenace_csv_bytes(filename: str, content_bytes: bytes) -> pd.DataFrame:
    """
    Robust parser for uploaded CENACE CSV bytes:
      - finds first 'Fecha' anywhere (case-insensitive),
      - starts CSV at that header line (even if mid-line),
      - maps core columns, coerces types,
      - adds 'nivel_tension'.
    Cached by file content bytes.
    """
    text = content_bytes.decode("latin1", errors="replace")
    idx = text.lower().find("fecha")
    if idx == -1:
        raise ValueError(f"'Fecha' header not found in {filename}")

    prev_nl = text.rfind("\n", 0, idx)
    start = 0 if prev_nl == -1 else prev_nl + 1
    csv_text = text[start:]

    first_line_end = csv_text.find("\n")
    header_line = csv_text[: first_line_end if first_line_end != -1 else len(csv_text)]
    if not header_line.strip().lower().startswith("fecha"):
        sub_idx = csv_text.lower().find("fecha")
        if sub_idx != -1:
            csv_text = csv_text[sub_idx:]

    df = pd.read_csv(
        StringIO(csv_text),
        quotechar='"',
        skipinitialspace=True,
        engine="python",
    )

    # Normalize headers
    df.columns = [str(c).strip().replace('"', "") for c in df.columns]

    def pick(colnames, options):
        # exact
        for opt in options:
            if opt in colnames:
                return opt
        # tolerant startswith
        low_map = {c.lower(): c for c in colnames}
        for opt in options:
            for lc, orig in low_map.items():
                if lc.startswith(opt.lower()):
                    return orig
        return None

    cols = list(df.columns)
    fecha_col = pick(cols, ["Fecha"])
    hora_col  = pick(cols, ["Hora"])
    nodo_col  = pick(cols, ["Clave del nodo", "Clave del Nodo", "Clave del nodo "])
    pml_col   = pick(cols, ["Precio marginal local ($/MWh)", "Precio marginal local ($/MWh) ", "Precio marginal local"])

    missing = [name for name, val in {
        "Fecha": fecha_col, "Hora": hora_col, "Clave del nodo": nodo_col, "Precio marginal local": pml_col
    }.items() if val is None]
    if missing:
        raise ValueError(f"Could not map required columns in {filename}. Columns: {cols} | Missing: {missing}")

    df = df.rename(columns={fecha_col: "fecha", hora_col: "hora", nodo_col: "nodo", pml_col: "pml"})[
        ["fecha", "hora", "nodo", "pml"]
    ]

    # Types
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", format="%Y-%m-%d")
    df["hora"]  = pd.to_numeric(df["hora"].astype(str).str.replace('"', '').str.strip(), errors="coerce").astype("Int64")
    df["pml"]   = pd.to_numeric(df["pml"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df["nodo"]  = df["nodo"].astype(str).str.strip()

    # Voltage
    df["nivel_tension"] = df["nodo"].apply(extract_voltage_from_node)

    # Clean
    df = df.dropna(subset=["fecha", "hora", "nodo", "pml"])
    try:
        df["pml"] = df["pml"].astype("float32")
        df["hora"] = df["hora"].astype("Int8")
    except Exception:
        pass
    return df

@st.cache_data(show_spinner=False)
def combine_uploaded_files(files_meta, files_bytes):
    """
    Combine all uploaded files into a single DataFrame.
    files_meta: list of (filename, size)
    files_bytes: list of raw bytes (same order).
    Cached based on files_meta + md5 hex to avoid re-parsing on UI tweaks.
    """
    import hashlib
    # Build stable hash signature per file
    sigs = []
    for (name, _), b in zip(files_meta, files_bytes):
        md5 = hashlib.md5(b).hexdigest()
        sigs.append((name, md5))

    # Parse each file (benefits from per-file cache in parse_cenace_csv_bytes)
    dfs, info = [], []
    for (name, _), b in zip(files_meta, files_bytes):
        df = parse_cenace_csv_bytes(name, b)
        dfs.append(df)
        info.append({
            "file": name,
            "min_date": df["fecha"].min(),
            "max_date": df["fecha"].max(),
            "n_rows": len(df),
        })

    df_all = pd.concat(dfs, ignore_index=True).sort_values(["nodo", "fecha", "hora"])
    try:
        df_all["nodo"] = df_all["nodo"].astype("category")
        df_all["nivel_tension"] = df_all["nivel_tension"].astype("category")
    except Exception:
        pass

    files_info = pd.DataFrame(info).sort_values("min_date")
    return df_all, files_info, tuple(sigs)  # sigs included just to bind cache

@st.cache_data(show_spinner=False)
def compute_daily_spreads_all(df_all: pd.DataFrame, min_hours_per_day: int) -> pd.DataFrame:
    """
    Expensive step computed ONCE per dataset + min_hours.
    Returns daily spreads for ALL nodes and ALL dates in df_all.
    """
    counts = df_all.groupby(["nodo", "fecha"], as_index=False)["pml"].count().rename(columns={"pml": "n_hours"})
    agg = df_all.groupby(["nodo", "fecha"], as_index=False)["pml"].agg(["max", "min"]).reset_index()
    agg["spread"] = agg["max"] - agg["min"]
    out = agg.merge(counts, on=["nodo", "fecha"], how="left")
    out.loc[out["n_hours"] < min_hours_per_day, "spread"] = np.nan
    return out[["nodo", "fecha", "spread", "n_hours"]]

def compute_avg_spread_from_daily(daily_spread: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """Cheap: filter by date range and average per node."""
    mask = (daily_spread["fecha"] >= pd.to_datetime(start_date)) & (daily_spread["fecha"] <= pd.to_datetime(end_date))
    subset = daily_spread.loc[mask].copy()
    result = (
        subset.groupby("nodo", as_index=False)
              .agg(avg_spread=("spread", "mean"),
                   n_days_used=("spread", lambda s: s.notna().sum()))
    )
    result = result.sort_values("avg_spread", ascending=False, na_position="last")
    return result

def sort_key_num_str(x):
    try:
        return float(x)
    except Exception:
        return float("inf")

# =========================================================
# Sidebar — Upload first
# =========================================================
with st.sidebar:
    st.header("📤 Upload CSVs")
    uploaded = st.file_uploader(
        "Upload CENACE CSV files (multi-select)",
        type=["csv"],
        accept_multiple_files=True,
        help="Drop Ene01/Ene02, Feb01/Feb02, ..., Ago01/Ago02, etc."
    )

if not uploaded:
    st.info("Upload your CENACE CSV files in the sidebar to begin.")
    st.stop()

# Prepare metadata & bytes for caching
files_meta = [(uf.name, uf.size) for uf in uploaded]
files_bytes = [uf.getvalue() for uf in uploaded]

with st.spinner("Reading & combining uploaded CSVs... (cached)"):
    df_all, files_info, _sig = combine_uploaded_files(files_meta, files_bytes)

with st.expander("📋 Diagnostics: files loaded", expanded=False):
    if not files_info.empty:
        _disp = files_info.copy()
        _disp["min_date"] = _disp["min_date"].dt.strftime("%Y-%m-%d")
        _disp["max_date"] = _disp["max_date"].dt.strftime("%Y-%m-%d")
        st.dataframe(_disp, use_container_width=True)

# Global bounds for date picker
global_min_date = pd.to_datetime(df_all["fecha"]).min().date()
global_max_date = pd.to_datetime(df_all["fecha"]).max().date()

# Voltage list (for settings)
voltage_levels = (
    df_all["nivel_tension"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)
voltage_levels = sorted(voltage_levels, key=sort_key_num_str)

# =========================================================
# Sidebar — Settings (form to avoid triggering heavy recompute)
# =========================================================
with st.sidebar.form("settings_form"):
    st.header("⚙️ Settings")

    # Heavy step depends ONLY on min_hours -> keep here
    min_hours = st.slider(
        "Minimum hours per day to accept a spread",
        min_value=1, max_value=24, value=18,
        help="Days with fewer hourly records than this will be excluded (spread = NaN).",
    )

    # Voltage filter (inside Settings as requested)
    selected_levels = st.multiselect(
        "Voltage levels (kV)",
        options=voltage_levels,
        default=voltage_levels,  # all by default
        help="Filter nodes by voltage suffix extracted from 'Clave del nodo'.",
    )

    # Date range (independent of heavy compute; filtering happens after)
    start_date, end_date = st.date_input(
        "Date range (inclusive)",
        value=(global_min_date, global_max_date),
        min_value=global_min_date, max_value=global_max_date,
        format="YYYY-MM-DD",
    )
    if isinstance(start_date, tuple):  # Streamlit older behavior guard
        start_date, end_date = start_date

    top_n = st.number_input("Show Top N nodes", min_value=5, max_value=200, value=50, step=5)

    submitted = st.form_submit_button("Apply")

# Persist selections in session_state to avoid recompute on each keystroke
if submitted or "cfg" not in st.session_state:
    st.session_state["cfg"] = {
        "min_hours": int(min_hours),
        "selected_levels": list(selected_levels) if selected_levels else [],
        "start_date": start_date,
        "end_date": end_date,
        "top_n": int(top_n),
    }

cfg = st.session_state["cfg"]

# =========================================================
# Heavy compute (cached) — runs only if files or min_hours change
# =========================================================
with st.spinner("Computing daily spreads for all nodes/dates... (cached)"):
    daily_all = compute_daily_spreads_all(df_all, min_hours_per_day=cfg["min_hours"])

# =========================================================
# Apply cheap filters: voltage + date range
# =========================================================
# Add voltage to daily via merge (cheap)
voltage_map = df_all[["nodo", "nivel_tension"]].drop_duplicates()
daily_all = daily_all.merge(voltage_map, on="nodo", how="left")

# Voltage filter
if cfg["selected_levels"]:
    daily_filtered = daily_all[daily_all["nivel_tension"].astype(str).isin(cfg["selected_levels"])].copy()
else:
    st.warning("No voltage levels selected — results will be empty.")
    daily_filtered = daily_all.iloc[0:0].copy()

# Date range filter happens in the averaging step (cheap)
if daily_filtered.empty:
    st.stop()

# =========================================================
# KPIs (filtered)
# =========================================================
colA, colB, colC = st.columns(3)
colA.metric("Nodes (filtered)", f"{daily_filtered['nodo'].nunique():,}")
colB.metric("Dates (filtered)", f"{daily_filtered['fecha'].nunique():,}")
colC.metric("Records (daily rows)", f"{len(daily_filtered):,}")

# =========================================================
# Average Daily Spread table & chart (filtered)
# =========================================================
result = compute_avg_spread_from_daily(daily_filtered, cfg["start_date"], cfg["end_date"])
result = result.merge(voltage_map, on="nodo", how="left")

st.subheader("🏆 Average Daily Spread by Node (selected period, FILTERED)")
st.write(
    f"From **{cfg['start_date']}** to **{cfg['end_date']}** | "
    f"Min hours/day: **{cfg['min_hours']}** | "
    f"Voltage levels: {', '.join(cfg['selected_levels']) if cfg['selected_levels'] else '—'}"
)

result = result.reset_index(drop=True)
result.insert(0, "rank", np.arange(1, len(result) + 1))
result = result[["rank", "nodo", "nivel_tension", "avg_spread", "n_days_used"]]
styled = result.style.format({"avg_spread": "{:,.2f}"})
st.dataframe(styled, use_container_width=True, hide_index=True)

st.subheader(f"📈 Top {cfg['top_n']} Nodes by Average Daily Spread (FILTERED)")
chart_df = (
    result.dropna(subset=["avg_spread"])
          .head(int(cfg["top_n"]))[["nodo", "avg_spread"]]
          .set_index("nodo")
)
st.bar_chart(chart_df)

# =========================================================
# Per-node details: table + daily spread chart (filtered & range)
# =========================================================
with st.expander("🔎 Per-node daily spreads (within selected range)"):
    mask = (daily_filtered["fecha"] >= pd.to_datetime(cfg["start_date"])) & (daily_filtered["fecha"] <= pd.to_datetime(cfg["end_date"]))
    daily_in_range = daily_filtered.loc[mask].copy()
    nodes_sorted = result["nodo"].dropna().tolist()
    node_sel = st.selectbox("Select a node", options=nodes_sorted[:5000])
    if node_sel:
        node_daily = daily_in_range[daily_in_range["nodo"] == node_sel].sort_values("fecha")
        st.dataframe(node_daily, use_container_width=True)
        st.markdown("**Daily spread evolution**")
        if not node_daily.empty:
            series = node_daily.set_index("fecha")["spread"]
            st.line_chart(series)
        else:
            st.info("No daily spread data available for the selected node and date range.")
