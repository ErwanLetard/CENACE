import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import re
from datetime import date
import math

# ====== Folium / Carte ======
try:
    import folium
    from folium.plugins import MarkerCluster
    from branca.colormap import linear
    from streamlit_folium import st_folium
except ModuleNotFoundError as e:
    st.error(
        f"Module manquant: {e}. Installe d’abord dans ton venv:\n\n"
        "```bash\n"
        "python -m pip install streamlit folium streamlit-folium geopy openpyxl branca pandas numpy\n"
        "```\n"
    )
    st.stop()

# ====== Geocoding (optionnel) ======
try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    GEOPY_AVAILABLE = True
except Exception:
    GEOPY_AVAILABLE = False

st.set_page_config(page_title="CENACE | Node Price Spread Explorer", layout="wide")
st.title("⚡ CENACE – Node Price Spread Explorer")
st.caption(
    "Charge plusieurs CSV CENACE + le fichier NodosP. Règle les filtres dans **Settings**. "
    "Les calculs lourds sont mis en cache ; les filtres n’y touchent pas."
)

# =========================================================
# Helpers
# =========================================================
def extract_voltage_from_node(nodo: str):
    """Extrait le suffixe tension après le dernier '-' ex: '01AAN-85' -> '85' (string, garde '34.5')."""
    if not isinstance(nodo, str):
        return None
    m = re.search(r"-([0-9]+(?:\.[0-9]+)?)$", nodo.strip())
    return m.group(1) if m else None

def sort_key_num_str(x):
    try:
        return float(x)
    except Exception:
        return float("inf")

def normalize_city(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())

def _fmt(val):
    """Affiche '—' pour NaN/vides."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    s = str(val).strip()
    return "—" if s == "" or s.lower() == "nan" else s

# ---------- Helpers pour graphe inline dans le popup ----------
def build_price_series(df_all: pd.DataFrame, nodo: str, start_date, end_date) -> pd.DataFrame:
    """
    Retourne la série journalière du prix moyen (PML) pour 'nodo' entre start_date et end_date.
    Colonnes: ['fecha','pml_mean'] triées par fecha.
    """
    mask = (
        (df_all["nodo"] == nodo) &
        (df_all["fecha"] >= pd.to_datetime(start_date)) &
        (df_all["fecha"] <= pd.to_datetime(end_date))
    )
    sub = df_all.loc[mask, ["fecha", "pml"]].copy()
    if sub.empty:
        return pd.DataFrame(columns=["fecha", "pml_mean"])
    s = sub.groupby("fecha", as_index=False)["pml"].mean().rename(columns={"pml": "pml_mean"})
    return s.sort_values("fecha")

def series_to_svg_line(dates: pd.Series, values: pd.Series,
                       width: int = 560, height: int = 220, pad: int = 28,
                       stroke_color: str = "#1565C0") -> str:
    """
    Convertit une série en SVG autonome (axes minimes + polyline).
    - dates: série de datetime
    - values: série numérique
    """
    import html
    if len(values) < 2 or values.notna().sum() < 2:
        return "<div style='font:12px Arial,sans-serif;color:#666'>No price data</div>"

    x_idx = np.arange(len(values), dtype=float)
    v = values.astype(float).values

    # Échelles
    x_min, x_max = x_idx.min(), x_idx.max()
    y_min = np.nanmin(v)
    y_max = np.nanmax(v)
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        y_min = y_min if np.isfinite(y_min) else 0.0
        y_max = y_max if np.isfinite(y_max) else 1.0
        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5

    inner_w = width - 2 * pad
    inner_h = height - 2 * pad

    def X(i):
        if x_max == x_min:
            return pad + inner_w / 2
        return pad + inner_w * ( (i - x_min) / (x_max - x_min) )

    def Y(val):
        return pad + inner_h * (1 - ( (val - y_min) / (y_max - y_min) ))

    points = " ".join(f"{X(i):.1f},{Y(val):.1f}" for i, val in zip(x_idx, v) if np.isfinite(val))

    last_val = v[-1]
    min_txt = f"{y_min:,.0f}"
    max_txt = f"{y_max:,.0f}"
    last_txt = f"{last_val:,.0f}"
    d0 = pd.to_datetime(dates.iloc[0]).date() if len(dates) else ""
    d1 = pd.to_datetime(dates.iloc[-1]).date() if len(dates) else ""

    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="background:#ffffff">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#ddd"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#aaa" stroke-width="1"/>
  <text x="{pad-4}" y="{pad+10}" font-size="11" text-anchor="end" fill="#666">{html.escape(max_txt)}</text>
  <text x="{pad-4}" y="{height-pad}" font-size="11" text-anchor="end" fill="#666">{html.escape(min_txt)}</text>
  <text x="{pad}" y="{height-4}" font-size="11" text-anchor="start" fill="#666">{html.escape(str(d0))}</text>
  <text x="{width-pad}" y="{height-4}" font-size="11" text-anchor="end" fill="#666">{html.escape(str(d1))}</text>
  <polyline fill="none" stroke="{stroke_color}" stroke-width="2" points="{points}" />
  <circle cx="{X(x_idx[-1]):.1f}" cy="{Y(last_val):.1f}" r="3" fill="{stroke_color}" />
  <text x="{X(x_idx[-1])+6:.1f}" y="{Y(last_val)-6:.1f}" font-size="11" fill="#333">{html.escape(last_txt)}</text>
</svg>
    """
    return svg

# =========================================================
# CSV parsing & combination
# =========================================================
@st.cache_data(show_spinner=False)
def parse_cenace_csv_bytes(filename: str, content_bytes: bytes) -> pd.DataFrame:
    """Parse robuste d’un CSV CENACE (début à 'Fecha', mapping colonnes, types, ajout niveau_tension)."""
    text = content_bytes.decode("latin1", errors="replace")
    idx = text.lower().find("fecha")
    if idx == -1:
        raise ValueError(f"'Fecha' introuvable dans {filename}")
    prev_nl = text.rfind("\n", 0, idx)
    start = 0 if prev_nl == -1 else prev_nl + 1
    csv_text = text[start:]
    first_line_end = csv_text.find("\n")
    header_line = csv_text[: first_line_end if first_line_end != -1 else len(csv_text)]
    if not header_line.strip().lower().startswith("fecha"):
        sub_idx = csv_text.lower().find("fecha")
        if sub_idx != -1:
            csv_text = csv_text[sub_idx:]
    df = pd.read_csv(StringIO(csv_text), quotechar='"', skipinitialspace=True, engine="python")
    df.columns = [str(c).strip().replace('"', "") for c in df.columns]

    def pick(colnames, options):
        for opt in options:
            if opt in colnames:
                return opt
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
        raise ValueError(f"Colonnes obligatoires manquantes dans {filename}. Colonnes: {cols} | Manquantes: {missing}")

    df = df.rename(columns={fecha_col: "fecha", hora_col: "hora", nodo_col: "nodo", pml_col: "pml"})[
        ["fecha", "hora", "nodo", "pml"]
    ]
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", format="%Y-%m-%d")
    df["hora"]  = pd.to_numeric(df["hora"].astype(str).str.replace('"', '').str.strip(), errors="coerce").astype("Int64")
    df["pml"]   = pd.to_numeric(df["pml"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df["nodo"]  = df["nodo"].astype(str).str.strip()
    df["nivel_tension"] = df["nodo"].apply(extract_voltage_from_node)
    df = df.dropna(subset=["fecha", "hora", "nodo", "pml"])
    try:
        df["pml"] = df["pml"].astype("float32")
        df["hora"] = df["hora"].astype("Int8")
    except Exception:
        pass
    return df

@st.cache_data(show_spinner=False)
def combine_uploaded_files(files_meta, files_bytes):
    """Combine tous les CSV uploadés en un seul DataFrame trié (nodo, fecha, hora)."""
    import hashlib
    sigs = []
    dfs, info = [], []
    for (name, _), b in zip(files_meta, files_bytes):
        md5 = hashlib.md5(b).hexdigest()
        sigs.append((name, md5))
        df = parse_cenace_csv_bytes(name, b)
        dfs.append(df)
        info.append({"file": name, "min_date": df["fecha"].min(), "max_date": df["fecha"].max(), "n_rows": len(df)})
    df_all = pd.concat(dfs, ignore_index=True).sort_values(["nodo", "fecha", "hora"])
    try:
        df_all["nodo"] = df_all["nodo"].astype("category")
        df_all["nivel_tension"] = df_all["nivel_tension"].astype("category")
    except Exception:
        pass
    files_info = pd.DataFrame(info).sort_values("min_date")
    return df_all, files_info, tuple(sigs)

@st.cache_data(show_spinner=False)
def compute_daily_spreads_all(df_all: pd.DataFrame, min_hours_per_day: int) -> pd.DataFrame:
    """Calcule une fois tous les spreads journaliers (par nodo/date), avec filtre sur #heures/jour."""
    counts = df_all.groupby(["nodo", "fecha"], as_index=False)["pml"].count().rename(columns={"pml": "n_hours"})
    agg = df_all.groupby(["nodo", "fecha"], as_index=False)["pml"].agg(["max", "min"]).reset_index()
    agg["spread"] = agg["max"] - agg["min"]
    out = agg.merge(counts, on=["nodo", "fecha"], how="left")
    out.loc[out["n_hours"] < min_hours_per_day, "spread"] = np.nan
    return out[["nodo", "fecha", "spread", "n_hours"]]

def compute_avg_spread_from_daily(daily_spread: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """Filtre par date et calcule la moyenne de spread par nodo."""
    mask = (daily_spread["fecha"] >= pd.to_datetime(start_date)) & (daily_spread["fecha"] <= pd.to_datetime(end_date))
    subset = daily_spread.loc[mask].copy()
    result = (
        subset.groupby("nodo", as_index=False)
              .agg(avg_spread=("spread", "mean"),
                   n_days_used=("spread", lambda s: s.notna().sum()))
    )
    result = result.sort_values("avg_spread", ascending=False, na_position="last")
    return result

# =========================================================
# NodosP Excel + Gazetteer (offline-first)
# =========================================================
@st.cache_data(show_spinner=False)
def read_nodosp_excel_from_bytes(xls_bytes: bytes) -> pd.DataFrame:
    """
    Lit NodosP depuis la ligne 3 (skiprows=2).
      - Col C (index 2) = ciudad
      - Col D (index 3) = nodo (CLAVE)
      - Col E (index 4) = nombre (NOM DU NOEUD)
    Retourne: ['nodo','ciudad','nombre'].
    """
    xls = pd.read_excel(BytesIO(xls_bytes), engine="openpyxl", header=None)
    data = xls.iloc[2:].reset_index(drop=True)  # commence à la ligne 3

    if data.shape[1] < 5:
        raise ValueError(
            "Le fichier NodosP n'a pas au moins 5 colonnes. "
            "Il faut C=ville, D=CLAVE, E=nom du noeud."
        )

    ciudad = data.iloc[:, 2].astype(str).str.strip()
    clave  = data.iloc[:, 3].astype(str).str.strip()
    nombre = data.iloc[:, 4].astype(str).str.strip()
    nombre = nombre.replace({"nan": ""})

    df = pd.DataFrame({
        "nodo": clave,
        "ciudad": ciudad,
        "nombre": nombre
    })

    df = df[df["nodo"].notna() & (df["nodo"] != "")]
    return df

@st.cache_data(show_spinner=False)
def read_gazetteer_csv(bytes_csv: bytes) -> pd.DataFrame:
    """Lit un CSV gazetteer avec au moins: city, lat, lon. Retourne ['city_norm','lat','lon']."""
    df = pd.read_csv(BytesIO(bytes_csv))
    cols_lower = {c.lower(): c for c in df.columns}
    city_col = cols_lower.get("city") or cols_lower.get("ciudad") or list(df.columns)[0]
    lat_col  = cols_lower.get("lat") or cols_lower.get("latitude")
    lon_col  = cols_lower.get("lon") or cols_lower.get("lng") or cols_lower.get("longitude")
    if lat_col is None or lon_col is None:
        raise ValueError("Gazetteer doit contenir 'lat' et 'lon' (ou 'latitude'/'longitude').")
    out = pd.DataFrame({
        "city_norm": df[city_col].astype(str).map(normalize_city),
        "lat": pd.to_numeric(df[lat_col], errors="coerce"),
        "lon": pd.to_numeric(df[lon_col], errors="coerce"),
    })
    out = out.dropna(subset=["city_norm","lat","lon"]).drop_duplicates("city_norm")
    return out

@st.cache_data(show_spinner=False)
def geocode_city_cached(city_norm: str) -> tuple:
    """Géocodage cache (lat,lon) avec Nominatim. Retourne (None,None) si indisponible."""
    if not city_norm or not GEOPY_AVAILABLE:
        return (None, None)
    geocoder = Nominatim(user_agent="cenace-spread-app", timeout=10)
    geocode = RateLimiter(geocoder.geocode, min_delay_seconds=1, swallow_exceptions=True)
    loc = geocode(city_norm + ", Mexico")
    if loc is None:
        return (None, None)
    return (loc.latitude, loc.longitude)

def build_city_coordinates(city_series: pd.Series,
                           gazetteer_df: pd.DataFrame | None,
                           allow_online_geocoding: bool) -> pd.DataFrame:
    """Construit un mapping city_norm -> (lat,lon) via gazetteer puis géocodage si besoin."""
    cities = pd.Series(city_series.dropna().astype(str).unique())
    cities_norm = cities.map(normalize_city)
    mapping = pd.DataFrame({"city_norm": cities_norm}).drop_duplicates()
    mapping["lat"] = np.nan
    mapping["lon"] = np.nan

    if gazetteer_df is not None and not gazetteer_df.empty:
        mapping = mapping.merge(gazetteer_df, on="city_norm", how="left", suffixes=("","_gaz"))
        mapping["lat"] = mapping["lat"].where(mapping["lat"].notna(), mapping.get("lat_gaz"))
        mapping["lon"] = mapping["lon"].where(mapping["lon"].notna(), mapping.get("lon_gaz"))
        for col in ["lat_gaz","lon_gaz"]:
            if col in mapping.columns:
                mapping = mapping.drop(columns=[col])

    if allow_online_geocoding:
        mask_missing = mapping["lat"].isna() | mapping["lon"].isna()
        if mask_missing.any():
            total = int(mask_missing.sum())
            prog = st.progress(0, text=f"Géocodage (Nominatim) — {total} villes")
            done = 0
            for idx in mapping.index[mask_missing]:
                c = mapping.at[idx, "city_norm"]
                lat, lon = geocode_city_cached(c)
                mapping.at[idx, "lat"] = lat
                mapping.at[idx, "lon"] = lon
                done += 1
                prog.progress(done/total, text=f"Géocodage Nominatim ({done}/{total})")
            prog.empty()

    return mapping

def jitter_positions(group_df, radius_m=400):
    """Décale les marqueurs autour d’un cercle pour éviter les overlaps quand coords identiques."""
    n = len(group_df)
    if n == 1:
        return group_df[["lat", "lon"]].values
    lats = group_df["lat"].values.astype(float)
    lons = group_df["lon"].values.astype(float)
    lat0 = np.nanmean(lats); lon0 = np.nanmean(lons)
    if np.isnan(lat0) or np.isnan(lon0):
        return group_df[["lat", "lon"]].values
    angles = np.linspace(0, 2*math.pi, n, endpoint=False)
    deg_lat = 400 / 111111.0
    denom = (111111.0 * math.cos(math.radians(lat0))) or 1.0
    deg_lon = 400 / denom
    return np.array([[lat0 + deg_lat*np.sin(a), lon0 + deg_lon*np.cos(a)] for a in angles])

# =========================================================
# Sidebar — Uploads
# =========================================================
with st.sidebar:
    st.header("📤 Upload CSVs (prices)")
    uploaded = st.file_uploader("Upload CENACE CSV files (multi-select)", type=["csv"], accept_multiple_files=True)

    st.header("📑 Upload NodosP Excel (for map)")
    nodosp_upload = st.file_uploader("Catálogo NodosP (xlsx)", type=["xlsx"], accept_multiple_files=False)
    local_xls_path = st.text_input(
        "Optional local path to NodosP Excel",
        value=r"C:\Users\eletard\Desktop\CENACE\Catálogo NodosP Sistema Eléctrico Nacional (v2025-09-26).xlsx"
    )

    st.header("🗺️ Map settings")
    gazetteer_upload = st.file_uploader("Optional gazetteer CSV (city[,state],lat,lon)", type=["csv"], accept_multiple_files=False)
    allow_online_geocoding = st.checkbox(
        "Allow online geocoding with OpenStreetMap (Nominatim) for missing cities",
        value=True,
        help="Appelle OSM (1 req/s, cache). Désactive si offline/firewall."
    )

if not uploaded:
    st.info("Upload tes CSV CENACE dans la barre latérale pour commencer.")
    st.stop()

# Prépare méta & bytes pour cache
files_meta = [(uf.name, uf.size) for uf in uploaded]
files_bytes = [uf.getvalue() for uf in uploaded]

with st.spinner("Lecture & combinaison des CSV... (cache)"):
    df_all, files_info, _sig = combine_uploaded_files(files_meta, files_bytes)

with st.expander("📋 Diagnostics: fichiers chargés", expanded=False):
    if not files_info.empty:
        _disp = files_info.copy()
        _disp["min_date"] = _disp["min_date"].dt.strftime("%Y-%m-%d")
        _disp["max_date"] = _disp["max_date"].dt.strftime("%Y-%m-%d")
        st.dataframe(_disp, use_container_width=True)

# Bornes pour le date picker
global_min_date = pd.to_datetime(df_all["fecha"]).min().date()
global_max_date = pd.to_datetime(df_all["fecha"]).max().date()

# Liste des tensions
voltage_levels = df_all["nivel_tension"].dropna().astype(str).unique().tolist()
voltage_levels = sorted(voltage_levels, key=sort_key_num_str)

# =========================================================
# Sidebar — Settings (form)
# =========================================================
with st.sidebar.form("settings_form"):
    st.header("⚙️ Settings")
    min_hours = st.slider("Minimum hours per day to accept a spread", 1, 24, 18)
    selected_levels = st.multiselect("Voltage levels (kV)", options=voltage_levels, default=voltage_levels)
    start_date, end_date = st.date_input(
        "Date range (inclusive)",
        value=(global_min_date, global_max_date),
        min_value=global_min_date, max_value=global_max_date, format="YYYY-MM-DD"
    )
    if isinstance(start_date, tuple):
        start_date, end_date = start_date
    top_n = st.number_input("Show Top N nodes (table/chart)", min_value=5, max_value=200, value=50, step=5)

    map_subset_mode = st.radio(
        "Nodes to show on the map",
        options=["All filtered nodes", "Top N by avg spread"],
        index=0
    )
    map_top_n = st.number_input("N for map (if Top N)", min_value=5, max_value=1000, value=200, step=5)

    submitted = st.form_submit_button("Apply")

if submitted or "cfg" not in st.session_state:
    st.session_state["cfg"] = {
        "min_hours": int(min_hours),
        "selected_levels": list(selected_levels) if selected_levels else [],
        "start_date": start_date,
        "end_date": end_date,
        "top_n": int(top_n),
        "map_subset_mode": map_subset_mode,
        "map_top_n": int(map_top_n),
        "allow_online_geocoding": bool(allow_online_geocoding)
    }
cfg = st.session_state["cfg"]

# =========================================================
# Heavy compute (cached)
# =========================================================
with st.spinner("Calcul des spreads journaliers (cache)..."):
    daily_all = compute_daily_spreads_all(df_all, min_hours_per_day=cfg["min_hours"])

# =========================================================
# Filtres
# =========================================================
voltage_map = df_all[["nodo", "nivel_tension"]].drop_duplicates()
daily_all = daily_all.merge(voltage_map, on="nodo", how="left")

if cfg["selected_levels"]:
    daily_filtered = daily_all[daily_all["nivel_tension"].astype(str).isin(cfg["selected_levels"])].copy()
else:
    st.warning("Aucun niveau de tension sélectionné — résultats vides.")
    daily_filtered = daily_all.iloc[0:0].copy()

if daily_filtered.empty:
    st.stop()

# KPIs
colA, colB, colC = st.columns(3)
colA.metric("Nodes (filtered)", f"{daily_filtered['nodo'].nunique():,}")
colB.metric("Dates (filtered)", f"{daily_filtered['fecha'].nunique():,}")
colC.metric("Records (daily rows)", f"{len(daily_filtered):,}")

# Table & Chart des spreads moyens
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
chart_df = result.dropna(subset=["avg_spread"]).head(int(cfg["top_n"]))[["nodo", "avg_spread"]].set_index("nodo")
st.bar_chart(chart_df)

# Per-node details
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

# =========================================================
# 🗺️ Carte des nœuds (Folium) — n’affiche que les nœuds filtrés
# =========================================================
st.subheader("🗺️ Nodos map (colored by average daily spread)")

# Charger NodosP (upload > chemin local)
nodosp_df = None
nodosp_errors = []
if 'nodosp_cache' not in st.session_state:
    st.session_state['nodosp_cache'] = None

if st.session_state['nodosp_cache'] is None:
    try:
        if 'nodosp_upload' in locals() and nodosp_upload is not None:
            nodosp_df = read_nodosp_excel_from_bytes(nodosp_upload.getvalue())
        elif local_xls_path:
            with open(local_xls_path, "rb") as f:
                nodosp_df = read_nodosp_excel_from_bytes(f.read())
        else:
            nodosp_df = None
        st.session_state['nodosp_cache'] = nodosp_df
    except Exception as e:
        nodosp_errors.append(str(e))
        nodosp_df = None
else:
    nodosp_df = st.session_state['nodosp_cache']

if nodosp_errors:
    with st.expander("⚠️ NodosP read warnings/errors", expanded=False):
        for err in nodosp_errors:
            st.write(f"- {err}")

if nodosp_df is None:
    st.info("Fournis le fichier Excel NodosP (upload ou chemin local) pour activer la carte.")
else:
    # Ne garder QUE les nœuds filtrés (tension + période)
    nodes_for_map = result.copy()
    if cfg["selected_levels"]:
        nodes_for_map = nodes_for_map[nodes_for_map["nivel_tension"].astype(str).isin(cfg["selected_levels"])]

    # Préparer NodosP: s'assurer que les colonnes existent
    for col in ["nodo", "ciudad", "nombre"]:
        if col not in nodosp_df.columns:
            nodosp_df[col] = np.nan
    nodosp_df = nodosp_df[["nodo", "ciudad", "nombre"]].drop_duplicates(subset=["nodo"])

    # Jointure LEFT sur nodes_for_map ⇒ la carte n'affiche que les nœuds filtrés
    map_df = nodes_for_map.merge(nodosp_df, on="nodo", how="left")

    # Option: limiter la carte au Top N par avg_spread
    if cfg.get("map_subset_mode") == "Top N by avg spread":
        map_df = map_df.dropna(subset=["avg_spread"]).sort_values("avg_spread", ascending=False).head(int(cfg.get("map_top_n", 200)))

    # Export d’un template de villes à compléter (offline)
    unique_cities = (
        map_df["ciudad"].dropna().astype(str)
        .map(normalize_city).drop_duplicates().to_frame(name="city")
    )
    template_csv = unique_cities.copy()
    template_csv["lat"] = ""
    template_csv["lon"] = ""
    st.download_button(
        label="⬇️ Export cities template (CSV to fill with lat/lon)",
        data=template_csv.to_csv(index=False).encode("utf-8"),
        file_name="cities_template.csv",
        mime="text/csv",
        help="Renseigne 'lat' et 'lon' pour chaque ville, puis ré-uploade en tant que Gazetteer CSV."
    )

    # Gazetteer CSV (facultatif) + géocodage en ligne (si activé)
    gazetteer_df = None
    if 'gazetteer_upload' in locals() and gazetteer_upload is not None:
        try:
            gazetteer_df = read_gazetteer_csv(gazetteer_upload.getvalue())
        except Exception as e:
            st.warning(f"Gazetteer CSV error: {e}")

    city_coords = build_city_coordinates(
        city_series=map_df["ciudad"],
        gazetteer_df=gazetteer_df,
        allow_online_geocoding=cfg.get("allow_online_geocoding", True)
    )

    map_df["city_norm"] = map_df["ciudad"].astype(str).map(normalize_city)
    map_df = map_df.merge(city_coords, on="city_norm", how="left")

    have_coords = map_df[map_df["lat"].notna() & map_df["lon"].notna()].copy()
    if have_coords.empty:
        st.warning(
            "Aucune coordonnée trouvée après gazetteer/géocodage. "
            "Uploade un gazetteer (city,lat,lon) ou active le géocodage en ligne."
        )
    else:
        # Échelle de couleur sur avg_spread
        vmin = float(have_coords["avg_spread"].min()) if have_coords["avg_spread"].notna().any() else 0.0
        vmax = float(have_coords["avg_spread"].max()) if have_coords["avg_spread"].notna().any() else 1.0
        cmap = linear.YlOrRd_09.scale(vmin, vmax) if math.isfinite(vmin) and math.isfinite(vmax) and vmin != vmax else linear.YlOrRd_09.scale(0, 1)

        # Base map centrée sur le Mexique
        m = folium.Map(location=[23.5, -102.0], zoom_start=5, tiles="cartodbpositron")
        cmap.caption = "Average daily spread ($/MWh)"
        cmap.add_to(m)

        have_coords["lat"] = have_coords["lat"].astype(float)
        have_coords["lon"] = have_coords["lon"].astype(float)
        cluster = MarkerCluster(name="Nodos").add_to(m)

        # Jitter par groupe de coordonnées identiques
        grouped = have_coords.groupby(["lat", "lon"], as_index=False, sort=False)
        for (_, _), group in grouped:
            pts = jitter_positions(group, radius_m=400)
            for (idx, row), (jlat, jlon) in zip(group.iterrows(), pts):
                # --- Construire le mini graphe PML moyen journalier pour CE nœud ---
                ts = build_price_series(df_all, nodo=row["nodo"], start_date=cfg["start_date"], end_date=cfg["end_date"])
                MAX_POINTS = 200
                if len(ts) > MAX_POINTS:
                    ts = ts.iloc[:: max(1, len(ts)//MAX_POINTS) ]
                svg_chart = series_to_svg_line(ts["fecha"], ts["pml_mean"], width=560, height=220, pad=28)

                val = row["avg_spread"]
                color = "#888888" if pd.isna(val) else cmap(val)

                popup_html = f"""
<div style="font:13px Arial, sans-serif; min-width: 600px;">
  <div style="margin-bottom:6px;">
    <b>Nodo:</b> {_fmt(row.get('nodo'))} &nbsp;|&nbsp;
    <b>Nom:</b> {_fmt(row.get('nombre'))} &nbsp;|&nbsp;
    <b>Tension:</b> {_fmt(row.get('nivel_tension'))} kV &nbsp;|&nbsp;
    <b>Ciudad:</b> {_fmt(row.get('ciudad'))}
  </div>
  <div style="margin-bottom:8px;">
    <b>Avg spread:</b> {(f"{row['avg_spread']:,.2f} $/MWh" if pd.notna(row.get('avg_spread')) else "—")} &nbsp;|&nbsp;
    <b>Days used:</b> {_fmt(row.get('n_days_used'))}<br>
    <span style="color:#666">Precio promedio diario (PML) — {cfg['start_date']} → {cfg['end_date']}</span>
  </div>
  <div>{svg_chart}</div>
</div>
                """

                popup = folium.Popup(html=popup_html, max_width=1200)
                tooltip = f"{_fmt(row.get('nodo'))} | {_fmt(row.get('nombre'))}"

                folium.CircleMarker(
                    location=[jlat, jlon],
                    radius=5,
                    weight=1,
                    color="#333333",
                    fill=True,
                    fill_opacity=0.85,
                    fill_color=color,
                    tooltip=tooltip,
                    popup=popup
                ).add_to(cluster)

        st_folium(m, width="100%", height=650)
