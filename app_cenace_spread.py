import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
from io import StringIO
import re
import os

st.set_page_config(page_title="CENACE | Node Price Spread Explorer", layout="wide")
st.title("⚡ CENACE – Node Price Spread Explorer")
st.caption("Average daily spreads = (max–min by day) for LMP ('Precio marginal local'), across multiple months, with voltage-level filters.")

# -------------------------
# Helpers
# -------------------------
def extract_voltage_from_node(nodo: str):
    """Extract voltage suffix after last '-' e.g. '01AAN-85' -> '85' (string, preserves '34.5')."""
    if not isinstance(nodo, str):
        return None
    m = re.search(r"-([0-9]+(?:\.[0-9]+)?)$", nodo.strip())
    return m.group(1) if m else None

def _sort_key_num_str(x):
    try:
        return float(x)
    except Exception:
        return float("inf")

@st.cache_data(show_spinner=False)
def read_cenace_csv(file_path: Path) -> pd.DataFrame:
    """
    Robust reader for CENACE CSVs:
    - Reads entire text (latin1)
    - Finds first 'Fecha' anywhere (case-insensitive)
    - Cuts from that header line
    - Parses CSV from that slice
    """
    text = file_path.read_text(encoding="latin1", errors="replace")
    idx = text.lower().find("fecha")
    if idx == -1:
        raise ValueError(f"'Fecha' header not found anywhere in {file_path.name}")

    # Start at beginning of the line that contains 'Fecha'
    prev_nl = text.rfind("\n", 0, idx)
    start = 0 if prev_nl == -1 else prev_nl + 1
    csv_text = text[start:]

    # If the first line doesn't start with 'Fecha', shift to the token
    first_line_end = csv_text.find("\n")
    header_line = csv_text[: first_line_end if first_line_end != -1 else len(csv_text)]
    if not header_line.strip().lower().startswith("fecha"):
        sub_idx = csv_text.lower().find("fecha")
        if sub_idx != -1:
            csv_text = csv_text[sub_idx:]

    df = pd.read_csv(
        StringIO(csv_text),
        encoding="latin1",
        quotechar='"',
        skipinitialspace=True,
        engine="python",
    )

    # normalize headers
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
        raise ValueError(f"Could not map required columns in {file_path.name}. Columns: {cols} | Missing: {missing}")

    df = df.rename(columns={fecha_col: "fecha", hora_col: "hora", nodo_col: "nodo", pml_col: "pml"})[
        ["fecha", "hora", "nodo", "pml"]
    ]

    # types
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", format="%Y-%m-%d")
    df["hora"] = pd.to_numeric(df["hora"].astype(str).str.replace('"', "").str.strip(), errors="coerce").astype("Int64")
    df["pml"] = pd.to_numeric(df["pml"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df["nodo"] = df["nodo"].astype(str).str.strip()

    # voltage & light downcast to save RAM over many months
    df["nivel_tension"] = df["nodo"].apply(extract_voltage_from_node)
    df = df.dropna(subset=["fecha", "hora", "nodo", "pml"])
    try:
        df["pml"] = df["pml"].astype("float32")
        df["hora"] = df["hora"].astype("Int8")
    except Exception:
        pass
    return df

@st.cache_data(show_spinner=False)
def load_all_csvs(folder: str):
    """
    Load every .csv from 'folder', concatenate, and sort.
    Returns:
      - df_all (DataFrame)
      - errors (list[str])
      - files_info (DataFrame with file, size_kb, min_date, max_date, n_rows)
    """
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    csv_files = sorted(list(p.glob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    dfs, errors, info_rows = [], [], []
    for f in csv_files:
        try:
            df = read_cenace_csv(f)
            dfs.append(df)
            info_rows.append({
                "file": f.name,
                "size_kb": round(os.path.getsize(f) / 1024, 1),
                "min_date": df["fecha"].min(),
                "max_date": df["fecha"].max(),
                "n_rows": len(df),
            })
        except Exception as e:
            errors.append(f"{f.name}: {e}")

    if not dfs:
        raise RuntimeError(f"Failed to read any CSV. Errors: {errors}")

    df_all = pd.concat(dfs, ignore_index=True).sort_values(["nodo", "fecha", "hora"])
    # Optional memory tweaks
    try:
        df_all["nodo"] = df_all["nodo"].astype("category")
        if "nivel_tension" in df_all.columns:
            df_all["nivel_tension"] = df_all["nivel_tension"].astype("category")
    except Exception:
        pass

    files_info = pd.DataFrame(info_rows).sort_values("min_date")
    return df_all, errors, files_info

def compute_daily_spread(df: pd.DataFrame, min_hours_per_day: int = 18) -> pd.DataFrame:
    counts = df.groupby(["nodo", "fecha"], as_index=False)["pml"].count().rename(columns={"pml": "n_hours"})
    agg = df.groupby(["nodo", "fecha"], as_index=False)["pml"].agg(["max", "min"]).reset_index()
    agg["spread"] = agg["max"] - agg["min"]
    out = agg.merge(counts, on=["nodo", "fecha"], how="left")
    out.loc[out["n_hours"] < min_hours_per_day, "spread"] = np.nan
    return out[["nodo", "fecha", "spread", "n_hours"]]

def compute_avg_spread_over_range(daily_spread: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    mask = (daily_spread["fecha"] >= pd.to_datetime(start_date)) & (daily_spread["fecha"] <= pd.to_datetime(end_date))
    subset = daily_spread.loc[mask].copy()
    result = (
        subset.groupby("nodo", as_index=False)
              .agg(avg_spread=("spread", "mean"), n_days_used=("spread", lambda s: s.notna().sum()))
    )
    result = result.sort_values("avg_spread", ascending=False, na_position="last")
    return result

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    folder = st.text_input(
        "Folder with CENACE CSVs",
        value=r"C:\Users\eletard\Desktop\CENACE",
        help="Path containing *all* monthly CSV files (Ene01/Ene02, Feb01/Feb02, ..., Ago01/Ago02, etc.).",
    )
    min_hours = st.slider(
        "Minimum hours per day to accept a spread",
        min_value=1, max_value=24, value=18,
        help="Days with fewer hourly records than this will be excluded (spread = NaN).",
    )
    top_n = st.number_input("Show Top N nodes in chart", min_value=5, max_value=200, value=50, step=5)

    # Force reload to invalidate cache if schema/inputs changed
    if st.button("🔄 Force reload data (clear cache)"):
        st.cache_data.clear()
        st.rerun()

# -------------------------
# Load data (ALL CSVs in folder)
# -------------------------
with st.spinner("Loading CSVs..."):
    try:
        df_all, load_errors, files_info = load_all_csvs(folder)
    except Exception as e:
        st.error(str(e))
        st.stop()

if load_errors:
    with st.expander("⚠️ CSV read warnings/errors"):
        for err in load_errors:
            st.write(f"- {err}")

with st.expander("📋 Diagnostics: files loaded", expanded=False):
    if not files_info.empty:
        files_info_display = files_info.copy()
        files_info_display["min_date"] = files_info_display["min_date"].dt.strftime("%Y-%m-%d")
        files_info_display["max_date"] = files_info_display["max_date"].dt.strftime("%Y-%m-%d")
        st.dataframe(files_info_display, use_container_width=True)

# SAFETY: ensure voltage exists even if cache restored old schema
if "nivel_tension" not in df_all.columns:
    df_all["nivel_tension"] = df_all["nodo"].astype(str).apply(extract_voltage_from_node)

# -------------------------
# Voltage filter (checkboxes) — APPLIED EVERYWHERE
# -------------------------
st.subheader("🔌 Voltage level filter")

voltage_levels = (
    df_all["nivel_tension"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)
voltage_levels = sorted(voltage_levels, key=_sort_key_num_str)

with st.expander("Select voltage levels", expanded=True):
    select_all = st.checkbox("Select all", value=True)
    selected_levels = [
        v for v in voltage_levels
        if st.checkbox(f"{v} kV", value=select_all, key=f"v_{v}")
    ]

if selected_levels:
    df_filtered = df_all[df_all["nivel_tension"].astype(str).isin(selected_levels)].copy()
else:
    st.warning("No voltage levels selected — results will be empty.")
    df_filtered = df_all.iloc[0:0].copy()

# Metrics reflect the FILTERED dataset
colA, colB, colC, colD = st.columns(4)
colA.metric("Nodes (filtered)", f"{df_filtered['nodo'].nunique():,}")
colB.metric("Dates (filtered)", f"{df_filtered['fecha'].nunique():,}")
colC.metric("Records (filtered)", f"{len(df_filtered):,}")
colD.metric("Files loaded", f"{len(files_info):,}")

# -------------------------
# Date range across ALL MONTHS
# -------------------------
if df_filtered.empty:
    st.stop()

min_date = pd.to_datetime(df_filtered["fecha"]).min().date()
max_date = pd.to_datetime(df_filtered["fecha"]).max().date()
st.subheader("📅 Date Range (can span multiple months)")
date_range = st.date_input(
    "Select date range (inclusive)",
    value=(min_date, max_date),
    min_value=min_date, max_value=max_date,
    format="YYYY-MM-DD",
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

if start_date > end_date:
    start_date, end_date = end_date, start_date  # swap if mis-ordered

# -------------------------
# Compute spreads on FILTERED data
# -------------------------
with st.spinner("Computing daily spreads..."):
    daily = compute_daily_spread(df_filtered, min_hours_per_day=min_hours)

with st.spinner("Aggregating average spreads over selected range..."):
    result = compute_avg_spread_over_range(daily, start_date, end_date)

# Attach voltage level to the result for display
voltage_map = df_filtered[["nodo", "nivel_tension"]].drop_duplicates()
result = result.merge(voltage_map, on="nodo", how="left")

st.subheader("🏆 Average Daily Spread by Node (selected period, FILTERED)")
st.write(
    f"From **{start_date}** to **{end_date}** | Min hours/day: **{min_hours}** | "
    f"Voltage levels: {', '.join(selected_levels) if selected_levels else '—'}"
)

result = result.reset_index(drop=True)
result.insert(0, "rank", np.arange(1, len(result) + 1))
result = result[["rank", "nodo", "nivel_tension", "avg_spread", "n_days_used"]]
styled = result.style.format({"avg_spread": "{:,.2f}"})
st.dataframe(styled, use_container_width=True, hide_index=True)

st.subheader(f"📈 Top {int(st.session_state.get('top_n', 50) if 'top_n' in st.session_state else 50)} Nodes by Average Daily Spread (FILTERED)")
# Use the current top_n value from sidebar (defined above)
chart_df = (
    result.dropna(subset=["avg_spread"])
          .head(int(top_n))[["nodo", "avg_spread"]]
          .set_index("nodo")
)
st.bar_chart(chart_df)

# -------------------------
# Per-node detail: table + daily spread chart (FILTERED & date range)
# -------------------------
with st.expander("🔎 Per-node daily spreads (within selected range)"):
    mask = (daily["fecha"] >= pd.to_datetime(start_date)) & (daily["fecha"] <= pd.to_datetime(end_date))
    daily_in_range = daily.loc[mask].copy()
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
