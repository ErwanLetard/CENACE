# app.py
import os
import re
import math
import gzip
import hashlib
from io import BytesIO, TextIOWrapper
from datetime import date
from urllib.parse import quote

import requests
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import streamlit.components.v1 as components

_VOLTAGE_SUFFIX_RE = re.compile(r"-([0-9]+(?:\.[0-9]+)?)$")

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
        "python -m pip install streamlit folium streamlit-folium geopy openpyxl branca pandas numpy requests altair\n"
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

# =====================================================================================
# App meta
# =====================================================================================
st.set_page_config(page_title="CENACE | Node Price Spread Explorer", layout="wide")
st.title("⚡ CENACE – Node Price Spread Explorer")
st.caption(
    "Sélectionne une période en 2025 (jan–sep), puis clique **Load data**. "
    "L’app télécharge les **CSV.gz** (SIN/BCA/BCS) depuis tes **GitHub Releases** et le **NodosP** depuis le repo. "
    "Filtres & calculs sont mis en cache."
)

# Petite option utilitaire pour réinitialiser les caches
with st.sidebar:
    if st.button("♻️ Clear data cache"):
        st.cache_data.clear()
        st.rerun()

# =====================================================================================
# 🔧 Configuration GitHub & jeu de données
# =====================================================================================
GITHUB_REPO = "ErwanLetard/CENACE"              # <<— adapte si besoin
SYSTEMS_ALL = ["SIN", "BCA", "BCS"]             # on cherche toujours les 3 systèmes
ASSET_SUFFIX = ".csv.gz"                        # assets stockés en CSV.gz par mois/système

# NodosP dans le repo (chemin + ref)
NODOSP_PATH_IN_REPO = "NodosP.xlsx"
NODOSP_REPO_REF = "main"

# =====================================================================================
# 🧭 Utilitaires
# =====================================================================================
def extract_voltage_from_node(nodo: str):
    """Extrait le suffixe tension après le dernier '-' ex: '01AAN-85' -> '85' (string)."""
    if not isinstance(nodo, str):
        return None
    m = _VOLTAGE_SUFFIX_RE.search(nodo.strip())
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
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    s = str(val).strip()
    return "—" if s == "" or s.lower() == "nan" else s

# =====================================================================================
# 🔐 GitHub API helpers (sans planter si pas de secrets.toml)
# =====================================================================================
def _get_github_token() -> str | None:
    token = os.environ.get("GITHUB_TOKEN")
    try:
        if token is None and hasattr(st, "secrets"):
            token = st.secrets.get("GITHUB_TOKEN", None)
    except Exception:
        pass
    return token

@st.cache_data(show_spinner=False)
def list_release_assets(owner_repo: str, tag: str) -> list[dict]:
    """
    Retourne la liste des assets pour un tag donné via l'API GitHub.
    [{name, url}, ...] (url = browser_download_url)
    """
    token = _get_github_token()
    url = f"https://api.github.com/repos/{owner_repo}/releases/tags/{tag}"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    data = r.json()
    out = []
    for a in data.get("assets", []) or []:
        name = a.get("name")
        dl = a.get("browser_download_url")
        if name and dl:
            out.append({"name": name, "url": dl})
    return out

# =====================================================================================
# 💱 Devises (MXN↔USD)
# =====================================================================================
@st.cache_data(show_spinner=False, ttl=60*60)
def get_usd_mxn_rate() -> float:
    """
    Taux USD→MXN du jour (exchangerate.host). Fallback = 20.0.
    """
    url = "https://api.exchangerate.host/latest?base=USD&symbols=MXN"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        rate = float(data["rates"]["MXN"])
        if rate > 0:
            return rate
    except Exception:
        pass
    return 20.0

# =====================================================================================
# 📥 Téléchargement données depuis GitHub (releases + raw)
# =====================================================================================
def month_tags_from_period(start_date: date, end_date: date):
    """Liste 'vYYYY-MM' pour chaque mois couvert par [start_date, end_date] (borne à 2025-01..2025-09)."""
    s = date(2025, 1, 1) if start_date < date(2025,1,1) else start_date
    e = date(2025, 9, 30) if end_date > date(2025,9,30) else end_date
    tags = []
    y, m = s.year, s.month
    while (y < e.year) or (y == e.year and m <= e.month):
        tags.append(f"v{y}-{m:02d}")
        m += 1
        if m == 13:
            m = 1
            y += 1
    return tags

def _standardize_prices_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise un DF supposé déjà propre (colonnes: fecha,hora,nodo,pml),
    mais tolère des variations (casse/espaces).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["fecha","hora","nodo","pml"])
    # noms en minuscules
    low = {c.lower().strip(): c for c in df.columns}
    def pick(keys):
        for k, orig in low.items():
            if any(tok in k for tok in keys):
                return orig
        return None
    fecha_col = pick(["fecha","date"])
    hora_col  = pick(["hora","hour"])
    nodo_col  = pick(["nodo","clave"])
    pml_col   = pick(["pml","precio marginal local","precio","marginal"])
    if not all([fecha_col, hora_col, nodo_col, pml_col]):
        # tente un rename direct par défaut si c'est déjà le bon schéma
        cand = {c.lower().strip(): c for c in df.columns}
        cols = {
            "fecha": cand.get("fecha","fecha"),
            "hora": cand.get("hora","hora"),
            "nodo": cand.get("nodo","nodo"),
            "pml":  cand.get("pml","pml"),
        }
        if set(cols.values()).issubset(set(df.columns)):
            out = df.rename(columns={v:k for k,v in cols.items()})[["fecha","hora","nodo","pml"]].copy()
        else:
            return pd.DataFrame(columns=["fecha","hora","nodo","pml"])
    else:
        out = df.rename(columns={
            fecha_col: "fecha",
            hora_col:  "hora",
            nodo_col:  "nodo",
            pml_col:   "pml",
        })[["fecha","hora","nodo","pml"]].copy()

    out["fecha"] = pd.to_datetime(out["fecha"], errors="coerce").dt.normalize()
    hora_series = pd.to_numeric(out["hora"], errors="coerce", downcast="integer")
    out["hora"] = hora_series.astype("Int64")
    out["pml"] = pd.to_numeric(out["pml"], errors="coerce", downcast="float")
    out["nodo"] = out["nodo"].astype(str).str.strip()
    return out.dropna(subset=["fecha", "hora", "nodo", "pml"])

@st.cache_data(show_spinner=False)
def fetch_prices_for_month_system(owner_repo: str, tag: str, system_code: str) -> pd.DataFrame:
    """
    Récupère le CSV.gz pour (tag, system) via l'API GitHub Releases.
    On cherche un asset qui commence par '{YYYY}-{MM}_{SYS}' et finit par '.csv.gz'.
    """
    assert system_code in SYSTEMS_ALL
    assets = list_release_assets(owner_repo, tag)
    if not assets:
        return pd.DataFrame(columns=["fecha","hora","nodo","pml"])

    y, m = tag[1:].split("-")
    prefix = f"{y}-{m}_{system_code}".lower()

    match = None
    for a in assets:
        name = (a.get("name") or "").lower()
        if name.startswith(prefix) and name.endswith(ASSET_SUFFIX):
            match = a
            break

    if not match:
        return pd.DataFrame(columns=["fecha","hora","nodo","pml"])

    url = match["url"]
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with gzip.GzipFile(fileobj=BytesIO(r.content), mode="rb") as gz:
            df = pd.read_csv(TextIOWrapper(gz, encoding="utf-8"))
        return _standardize_prices_df(df)
    except Exception:
        return pd.DataFrame(columns=["fecha","hora","nodo","pml"])

@st.cache_data(show_spinner=False)
def load_all_prices(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Combine tous les CSV.gz (SIN/BCA/BCS) pour tous les mois utiles.
    Retour: DataFrame unique trié (nodo, fecha, hora) + 'nivel_tension'.
    """
    tags = month_tags_from_period(start_date, end_date)
    dfs = []
    for tag in tags:
        for sys in SYSTEMS_ALL:
            df = fetch_prices_for_month_system(GITHUB_REPO, tag, sys)
            if not df.empty:
                dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["fecha","hora","nodo","pml","nivel_tension"])
    df_all = pd.concat(dfs, ignore_index=True)
    nodo_series = df_all["nodo"].astype(str).str.strip()
    df_all["nodo"] = nodo_series
    df_all["nivel_tension"] = nodo_series.str.extract(_VOLTAGE_SUFFIX_RE, expand=False)
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    mask = (df_all["fecha"] >= start_ts) & (df_all["fecha"] <= end_ts)
    df_all = df_all.loc[mask]
    try:
        df_all["nodo"] = df_all["nodo"].astype("category")
        df_all["nivel_tension"] = df_all["nivel_tension"].astype("category")
        df_all["hora"] = df_all["hora"].astype("Int8")
        df_all["pml"] = df_all["pml"].astype("float32")
    except Exception:
        pass
    return df_all.sort_values(["nodo","fecha","hora"], ignore_index=True)

@st.cache_data(show_spinner=False)
def read_nodosp_excel_from_repo(repo=GITHUB_REPO, ref=NODOSP_REPO_REF, path=NODOSP_PATH_IN_REPO) -> pd.DataFrame:
    """
    Lit NodosP depuis le repo GitHub (URL raw).
      - Col C (index 2) = ciudad
      - Col D (index 3) = nodo (CLAVE)
      - Col E (index 4) = nombre (NOM DU NOEUD)
    """
    raw_url = f"https://raw.githubusercontent.com/{repo}/{ref}/{quote(path, safe='/')}"
    r = requests.get(raw_url, timeout=30)
    r.raise_for_status()
    xls = pd.read_excel(BytesIO(r.content), engine="openpyxl", header=None)
    data = xls.iloc[2:].reset_index(drop=True)
    if data.shape[1] < 5:
        raise ValueError("NodosP: colonnes insuffisantes (il faut au moins C,D,E).")
    ciudad = data.iloc[:, 2].astype(str).str.strip()
    clave  = data.iloc[:, 3].astype(str).str.strip()
    nombre = data.iloc[:, 4].astype(str).str.strip().replace({"nan": ""})
    df = pd.DataFrame({"nodo": clave, "ciudad": ciudad, "nombre": nombre})
    return df[df["nodo"].notna() & (df["nodo"] != "")]

# =====================================================================================
# 🔻 Downsampling LTTB (pour popup de séries horaires)
# =====================================================================================
def lttb_downsample(x, y, threshold: int):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    n = len(x)
    if n != len(y) or n == 0 or threshold >= n or threshold <= 3:
        return x, y
    every = (n - 2) / (threshold - 2)
    a = 0
    x_res = [x[0]]; y_res = [y[0]]
    for i in range(0, threshold - 2):
        avg_range_start = int(np.floor((i + 1) * every) + 1)
        avg_range_end   = int(np.floor((i + 2) * every) + 1)
        if avg_range_end >= n: avg_range_end = n
        if avg_range_end > avg_range_start:
            avg_x = np.mean(x[avg_range_start:avg_range_end])
            avg_y = np.mean(y[avg_range_start:avg_range_end])
        else:
            idx_safe = min(avg_range_start, n - 1)
            avg_x, avg_y = x[idx_safe], y[idx_safe]
        range_offs = int(np.floor(i * every) + 1)
        range_to   = int(np.floor((i + 1) * every) + 1)
        if range_to >= n: range_to = n
        x_a, y_a = x[a], y[a]
        seg_x, seg_y = x[range_offs:range_to], y[range_offs:range_to]
        if seg_x.size == 0:
            idx = range_offs
        else:
            area = np.abs((x_a - avg_x) * (seg_y - y_a) - (y_a - avg_y) * (seg_x - x_a))
            idx = int(np.argmax(area)) + range_offs
        x_res.append(x[idx]); y_res.append(y[idx]); a = idx
    x_res.append(x[-1]); y_res.append(y[-1])
    return np.asarray(x_res), np.asarray(y_res)

def build_price_series_hourly(df_all: pd.DataFrame, nodo: str, start_date, end_date) -> pd.DataFrame:
    """Extrait la série horaire PML pour un nœud sur une plage de dates."""
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    mask = (
        (df_all["nodo"] == nodo) &
        (df_all["fecha"] >= start_ts) &
        (df_all["fecha"] <= end_ts)
    )
    sub = df_all.loc[mask, ["fecha", "hora", "pml"]].dropna().copy()
    if sub.empty:
        return pd.DataFrame(columns=["ts", "pml"])
    hora0 = sub["hora"].astype(int) - 1  # 1..24 -> 0..23
    sub["ts"] = sub["fecha"] + pd.to_timedelta(hora0, unit="h")
    return sub.sort_values("ts", ignore_index=True)[["ts", "pml"]]

def ts_to_svg_line(ts: pd.Series, values: pd.Series,
                   width: int = 760, height: int = 280, pad: int = 32,
                   stroke_color: str = "#1565C0",
                   max_points: int = 1200) -> str:
    import html
    ts_dt = pd.to_datetime(ts, errors="coerce")
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    valid = ts_dt.notna().to_numpy() & np.isfinite(y)
    if not np.any(valid):
        return "<div style='font:12px Arial,sans-serif;color:#666'>No price data</div>"

    x = ts_dt[valid].astype("int64").to_numpy(dtype=np.float64, copy=False)
    y = y[valid]
    if x.size < 2:
        return "<div style='font:12px Arial,sans-serif;color:#666'>No price data</div>"

    if x.size > max_points:
        x, y = lttb_downsample(x, y, threshold=max_points)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        y_min = y_min if np.isfinite(y_min) else 0.0
        y_max = y_max if np.isfinite(y_max) else 1.0
        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5

    inner_w = width - 2 * pad
    inner_h = height - 2 * pad

    def X(xx: float) -> float:
        return pad + inner_w * ((xx - x_min) / (x_max - x_min) if x_max != x_min else 0.5)

    def Y(val: float) -> float:
        return pad + inner_h * (1 - ((val - y_min) / (y_max - y_min)))

    points = " ".join(f"{X(xx):.1f},{Y(yy):.1f}" for xx, yy in zip(x, y))

    last_val = y[-1]
    min_txt = f"{y_min:,.0f}"
    max_txt = f"{y_max:,.0f}"
    last_txt = f"{last_val:,.0f}"
    d0 = pd.Timestamp(int(x[0])).date()
    d1 = pd.Timestamp(int(x[-1])).date()

    q1_y = Y(y_min + 0.25 * (y_max - y_min))
    q3_y = Y(y_min + 0.75 * (y_max - y_min))

    return f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="background:#ffffff">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#ddd"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#aaa" stroke-width="1"/>
  <line x1="{pad}" y1="{q1_y:.1f}" x2="{width-pad}" y2="{q1_y:.1f}" stroke="#eee" stroke-width="1"/>
  <line x1="{pad}" y1="{q3_y:.1f}" x2="{width-pad}" y2="{q3_y:.1f}" stroke="#eee" stroke-width="1"/>
  <text x="{pad-4}" y="{pad+10}" font-size="11" text-anchor="end" fill="#666">{html.escape(max_txt)}</text>
  <text x="{pad-4}" y="{height-pad}" font-size="11" text-anchor="end" fill="#666">{html.escape(min_txt)}</text>
  <text x="{pad}" y="{height-4}" font-size="11" text-anchor="start" fill="#666">{html.escape(str(d0))}</text>
  <text x="{width-pad}" y="{height-4}" font-size="11" text-anchor="end" fill="#666">{html.escape(str(d1))}</text>
  <polyline fill="none" stroke="{stroke_color}" stroke-width="2" points="{points}" />
  <circle cx="{X(x[-1]):.1f}" cy="{Y(last_val):.1f}" r="3" fill="{stroke_color}" />
  <text x="{X(x[-1])+6:.1f}" y="{Y(last_val)-6:.1f}" font-size="11" fill="#333">{html.escape(last_txt)}</text>
</svg>
""".strip()

# =====================================================================================
# 🧮 Calculs spreads (daily & rolling-window)
# =====================================================================================
@st.cache_data(show_spinner=False)
def compute_daily_spreads_all(df_all: pd.DataFrame, min_hours_per_day: int) -> pd.DataFrame:
    """Calcule max–min quotidien par nœud et filtre selon le nombre d'heures disponibles."""
    if df_all.empty:
        return pd.DataFrame(columns=["nodo", "fecha", "spread", "n_hours"])
    grouped = (
        df_all.groupby(["nodo", "fecha"], observed=True)["pml"]
              .agg(max="max", min="min", n_hours="count")
              .reset_index()
    )
    grouped["spread"] = grouped["max"] - grouped["min"]
    grouped.loc[grouped["n_hours"] < int(min_hours_per_day), "spread"] = np.nan
    return grouped[["nodo", "fecha", "spread", "n_hours"]]

@st.cache_data(show_spinner=False)
def compute_window_spreads_all(df_all: pd.DataFrame, window_h: int, min_hours_per_day: int) -> pd.DataFrame:
    """
    Spread quotidien basé sur le max–min des moyennes glissantes de `window_h` heures.
    Exige au moins max(min_hours_per_day, window_h) mesures valides dans la journée.
    """
    if df_all.empty:
        return pd.DataFrame(columns=["nodo", "fecha", "spread_win", "n_hours"])
    window_h = int(window_h)
    min_hours_per_day = int(min_hours_per_day)

    base = df_all[["nodo", "fecha", "hora", "pml"]].dropna().copy()
    if base.empty:
        return pd.DataFrame(columns=["nodo", "fecha", "spread_win", "n_hours"])

    base = base.sort_values(["nodo", "fecha", "hora"])
    counts = (
        base.groupby(["nodo", "fecha"], observed=True)["pml"]
            .count()
            .rename("n_hours")
            .reset_index()
    )

    base_idxed = base.set_index(["nodo", "fecha", "hora"])
    node_dates = counts[["nodo", "fecha"]].sort_values(["nodo", "fecha"])

    hours = np.arange(1, 25, dtype=np.int16)
    full_index = pd.MultiIndex.from_arrays(
        [
            np.repeat(node_dates["nodo"].to_numpy(), len(hours)),
            np.repeat(node_dates["fecha"].to_numpy(), len(hours)),
            np.tile(hours, len(node_dates)),
        ],
        names=["nodo", "fecha", "hora"],
    )

    pml_full = base_idxed["pml"].reindex(full_index)
    rolling_mean = (
        pml_full.groupby(level=["nodo", "fecha"])
                .rolling(window=window_h, min_periods=window_h)
                .mean()
    )

    rolling_stats = (
        rolling_mean.groupby(level=["nodo", "fecha"])
                    .agg(["max", "min"])
                    .droplevel(0, axis=1)
                    .rename(columns={"max": "rolling_max", "min": "rolling_min"})
                    .reset_index()
    )
    rolling_stats["spread_win"] = rolling_stats["rolling_max"] - rolling_stats["rolling_min"]
    rolling_stats = rolling_stats[["nodo", "fecha", "spread_win"]]

    out = rolling_stats.merge(counts, on=["nodo", "fecha"], how="left")
    min_need = max(min_hours_per_day, window_h)
    out.loc[out["n_hours"] < min_need, "spread_win"] = np.nan
    return out

@st.cache_data(show_spinner=False)
def compute_avg_spread_from_daily(daily_spread: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """Retourne la moyenne des spreads journaliers max–min sur la période donnée."""
    if daily_spread.empty:
        return pd.DataFrame(columns=["nodo", "avg_spread", "n_days_used"])
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    mask = (daily_spread["fecha"] >= start_ts) & (daily_spread["fecha"] <= end_ts)
    subset = daily_spread.loc[mask]
    result = (
        subset.groupby("nodo", observed=True)["spread"]
              .agg(avg_spread="mean", n_days_used="count")
              .reset_index()
              .sort_values("avg_spread", ascending=False, na_position="last")
    )
    return result

@st.cache_data(show_spinner=False)
def compute_avg_window_spread(daily_win: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """Retourne la moyenne des spreads rolling-window sur la période donnée."""
    if daily_win.empty:
        return pd.DataFrame(columns=["nodo", "avg_spread_win", "n_days_used"])
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    mask = (daily_win["fecha"] >= start_ts) & (daily_win["fecha"] <= end_ts)
    subset = daily_win.loc[mask]
    res = (
        subset.groupby("nodo", observed=True)["spread_win"]
              .agg(avg_spread_win="mean", n_days_used="count")
              .reset_index()
              .sort_values("avg_spread_win", ascending=False, na_position="last")
    )
    return res

# =====================================================================================
# 🗺️ Gazetteer & géocodage – cache persistants
# =====================================================================================
VDM_RULES = {
    "vdm norte":  (26.0, -102.5),
    "vdm centro": (20.5, -101.0),
    "vdm sur":    (17.0, -99.0),
}

@st.cache_data(show_spinner=False, ttl=None)
def geocode_city_cached(city_norm: str) -> tuple[float | None, float | None]:
    """Renvoie (lat, lon) pour une ville normalisée, avec cache illimité."""
    if not city_norm:
        return (None, None)
    for key, (la, lo) in VDM_RULES.items():
        if key in city_norm:
            return (la, lo)
    if not GEOPY_AVAILABLE:
        return (None, None)
    geocoder = Nominatim(user_agent="cenace-spread-app", timeout=10)
    rl = RateLimiter(geocoder.geocode, min_delay_seconds=1, swallow_exceptions=True)
    loc = rl(city_norm + ", Mexico")
    if loc is None:
        return (None, None)
    return (loc.latitude, loc.longitude)

@st.cache_data(show_spinner=False, ttl=None)
def build_city_coords_cached(unique_cities: tuple[str, ...],
                             allow_online_geocoding: bool) -> pd.DataFrame:
    """
    Construit un DataFrame (city_norm, lat, lon) pour une liste triée de villes.
    Les résultats sont mémorisés durablement.
    """
    mapping = pd.DataFrame({"city_norm": list(unique_cities)})
    mapping["lat"] = np.nan
    mapping["lon"] = np.nan

    for k, (la, lo) in VDM_RULES.items():
        mask = mapping["city_norm"].str.contains(k, na=False)
        mapping.loc[mask, ["lat", "lon"]] = (la, lo)

    if allow_online_geocoding and GEOPY_AVAILABLE:
        missing = mapping["lat"].isna() | mapping["lon"].isna()
        for idx in mapping.index[missing]:
            city = mapping.at[idx, "city_norm"]
            la, lo = geocode_city_cached(city)
            mapping.at[idx, "lat"] = la
            mapping.at[idx, "lon"] = lo
    return mapping

def jitter_positions(group_df: pd.DataFrame, radius_m: float = 400.0) -> np.ndarray:
    """Décale légèrement les marqueurs pour éviter la superposition lorsque les coords sont identiques."""
    n = len(group_df)
    if n <= 1:
        return group_df[["lat", "lon"]].to_numpy(dtype=float, copy=True)
    lats = group_df["lat"].astype(float).to_numpy(copy=True)
    lons = group_df["lon"].astype(float).to_numpy(copy=True)
    lat0 = np.nanmean(lats)
    lon0 = np.nanmean(lons)
    if np.isnan(lat0) or np.isnan(lon0):
        return group_df[["lat", "lon"]].to_numpy(dtype=float, copy=True)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    deg_lat = radius_m / 111_111.0
    denom = (111_111.0 * math.cos(math.radians(lat0))) or 1.0
    deg_lon = radius_m / denom
    return np.column_stack((lat0 + deg_lat * np.sin(angles),
                            lon0 + deg_lon * np.cos(angles)))

# =====================================================================================
# 🎛️ Panneau latéral — paramètres utilisateur + bouton Load
# =====================================================================================
with st.sidebar.form("settings_form"):
    st.header("⚙️ Période & Filtres")
    min_allowed = date(2025,1,1)
    max_allowed = date(2025,9,30)
    start_date, end_date = st.date_input(
        "Date range (inclusive)",
        value=(min_allowed, max_allowed),
        min_value=min_allowed, max_value=max_allowed, format="YYYY-MM-DD"
    )
    if isinstance(start_date, tuple):
        start_date, end_date = start_date

    # Devises
    currency = st.radio("Currency", options=["MXN","USD"], index=0,
                        help="Les prix CENACE sont en MXN, malgré le label $/MWh. "
                             "USD utilise le taux du jour (exchangerate.host) avec repli.")
    rate_usd_mxn = get_usd_mxn_rate()
    if currency == "USD":
        st.caption(f"Taux USD→MXN utilisé: {rate_usd_mxn:.4f} (PML_USD = PML_MXN / taux)")

    min_hours = st.slider("Minimum hours per day (data quality)", 1, 24, 18)

    # Rolling-window ou Daily max–min
    metric_mode = st.radio(
        "Spread metric",
        options=["Daily max–min", "Rolling-window spread"],
        index=0,
        help="Daily max–min = max(pml) - min(pml) par jour.\n"
             "Rolling-window = max(moyenne glissante Hh) - min(moyenne glissante Hh)."
    )
    window_h = st.select_slider("Window size (hours)", options=[2,4,8,12], value=4,
                                help="Actif seulement en mode Rolling-window.")

    # Limites carte
    top_n = st.number_input("Show Top N nodes (table/chart)", min_value=5, max_value=200, value=50, step=5)
    map_subset_mode = st.radio("Nodes to show on the map", ["All filtered nodes","Top N by avg spread"], index=0)
    map_top_n = st.number_input("N for map (if Top N)", min_value=5, max_value=1000, value=200, step=5)

    # Carte
    allow_online_geocoding = st.checkbox("Allow online geocoding (Nominatim)", value=True)

    # Bouton pour lancer explicitement le chargement
    load_clicked = st.form_submit_button("🚀 Load data")

# Sauvegarde config en session
if load_clicked or "cfg" not in st.session_state:
    st.session_state["cfg"] = {
        "start_date": start_date,
        "end_date": end_date,
        "currency": currency,
        "rate_usd_mxn": float(rate_usd_mxn),
        "min_hours": int(min_hours),
        "metric_mode": metric_mode,
        "window_h": int(window_h),
        "top_n": int(top_n),
        "map_subset_mode": map_subset_mode,
        "map_top_n": int(map_top_n),
        "allow_online_geocoding": bool(allow_online_geocoding),
        "load_clicked": bool(load_clicked),
    }
cfg = st.session_state["cfg"]

# Bloque tant que l’utilisateur n’a pas cliqué “Load data”
if not cfg.get("load_clicked", False):
    st.info("Configure les paramètres dans la barre latérale puis clique **Load data** pour démarrer le téléchargement.")
    st.stop()

# =====================================================================================
# 📦 Chargement des données (GitHub)
# =====================================================================================
data_sig = (cfg["start_date"], cfg["end_date"])
if st.session_state.get("_data_sig") != data_sig or "_df_all" not in st.session_state:
    with st.spinner("Téléchargement & assemblage des données (GitHub Releases)…"):
        df_all = load_all_prices(cfg["start_date"], cfg["end_date"])
    st.session_state["_data_sig"] = data_sig
    st.session_state["_df_all"] = df_all
else:
    df_all = st.session_state["_df_all"]

if df_all.empty:
    st.error("Aucune donnée trouvée pour la période choisie (release vYYYY-MM manquante ? SIN/BCA/BCS absents ?)")
    st.stop()

# Conversion devise pour affichage/métriques
conv_factor = 1.0 if cfg["currency"] == "MXN" else 1.0 / max(cfg["rate_usd_mxn"], 1e-9)

# =====================================================================================
# 📊 Calculs spreads (cache)
# =====================================================================================
daily_sig = (data_sig, cfg["min_hours"])
if st.session_state.get("_daily_sig") != daily_sig or "_daily_all_mxn" not in st.session_state:
    with st.spinner("Calcul des spreads journaliers (cache)…"):
        daily_all_mxn = compute_daily_spreads_all(df_all, min_hours_per_day=cfg["min_hours"])
    st.session_state["_daily_sig"] = daily_sig
    st.session_state["_daily_all_mxn"] = daily_all_mxn
else:
    daily_all_mxn = st.session_state["_daily_all_mxn"]

# Résultat agrégé selon la métrique choisie (conversion devise ici)
if cfg["metric_mode"] == "Daily max–min":
    result = compute_avg_spread_from_daily(daily_all_mxn, cfg["start_date"], cfg["end_date"])
    metric_col = "avg_spread"
    metric_label = f"Average Daily Spread ({cfg['currency']}/MWh)"
    result[metric_col] = result[metric_col] * conv_factor
else:
    rolling_sig = (data_sig, cfg["min_hours"], cfg["window_h"])
    if st.session_state.get("_rolling_sig") != rolling_sig or "_daily_win_all_mxn" not in st.session_state:
        with st.spinner(f"Computing {cfg['window_h']}h rolling-window daily spreads… (cache)"):
            daily_win_all_mxn = compute_window_spreads_all(df_all, window_h=cfg["window_h"], min_hours_per_day=cfg["min_hours"])
        st.session_state["_rolling_sig"] = rolling_sig
        st.session_state["_daily_win_all_mxn"] = daily_win_all_mxn
    else:
        daily_win_all_mxn = st.session_state["_daily_win_all_mxn"]
    result = compute_avg_window_spread(daily_win_all_mxn, cfg["start_date"], cfg["end_date"])
    metric_col = "avg_spread_win"
    metric_label = f"Average {cfg['window_h']}h Window Spread ({cfg['currency']}/MWh)"
    result[metric_col] = result[metric_col] * conv_factor

# Ajoute tension
voltage_map = df_all[["nodo","nivel_tension"]].drop_duplicates()
result = result.merge(voltage_map, on="nodo", how="left")

# Liste & filtre des tensions
voltage_levels = df_all["nivel_tension"].dropna().astype(str).unique().tolist()
voltage_levels = sorted(voltage_levels, key=sort_key_num_str)
selected_levels = st.multiselect("Voltage levels (kV) — filter results",
                                 options=voltage_levels, default=voltage_levels, key="kv_filter")
if selected_levels:
    result = result[result["nivel_tension"].astype(str).isin(selected_levels)]
else:
    st.warning("Aucun niveau de tension sélectionné — résultats vides.")
    st.stop()

# KPIs
colA, colB, colC = st.columns(3)
colA.metric("Nodes (filtered)", f"{result['nodo'].nunique():,}")
nb_dates = pd.date_range(cfg["start_date"], cfg["end_date"], freq="D").size
colB.metric("Dates (range)", f"{nb_dates:,}")
colC.metric("Metric", metric_label)

# Tableau
st.subheader(f"🏆 {metric_label} by Node (selected period, FILTERED)")
res_show = result.reset_index(drop=True).copy()
res_show.insert(0, "rank", np.arange(1, len(res_show) + 1))
st.dataframe(res_show[["rank","nodo","nivel_tension",metric_col,"n_days_used"]].style.format({metric_col: "{:,.2f}"}),
             use_container_width=True, hide_index=True)

# =====================================================================================
# 📈 Top N bar chart (vertical, tri décroissant) + “go to node on map”
# =====================================================================================
st.subheader(f"📈 Top {cfg['top_n']} Nodes by {metric_label}")
top_df = (
    result
      .dropna(subset=[metric_col])
      .sort_values(metric_col, ascending=False)
      .head(int(cfg["top_n"]))
      .loc[:, ["nodo", metric_col]]
      .rename(columns={"nodo": "Node", metric_col: "Value"})
)

chart = (
    alt.Chart(top_df)
       .mark_bar()
       .encode(
           x=alt.X("Node:N", sort=alt.SortField(field="Value", order="descending"), title="Node"),
           y=alt.Y("Value:Q", title=metric_label),
           tooltip=[
               alt.Tooltip("Node:N", title="Node"),
               alt.Tooltip("Value:Q", title=metric_label, format=",.2f"),
           ],
       )
       .properties(height=360, width="container")
)

col_chart, col_pick = st.columns([4, 1])
with col_chart:
    st.altair_chart(chart, use_container_width=True)

with col_pick:
    st.markdown("**Go to node on map**")
    node_options = top_df["Node"].tolist()
    picked_node = st.selectbox("Top N nodes", node_options, index=0, key="top_pick")
    if st.button("📍 Show on map", type="primary", use_container_width=True):
        st.session_state["selected_node"] = picked_node

# =====================================================================================
# 🔍 Détail par nœud (daily max–min)
# =====================================================================================
with st.expander("🔎 Per-node daily spreads (max–min)"):
    nodes_sorted = res_show["nodo"].dropna().tolist()
    node_sel = st.selectbox("Select a node", options=nodes_sorted[:5000])
    if node_sel:
        start_ts = pd.to_datetime(cfg["start_date"])
        end_ts = pd.to_datetime(cfg["end_date"])
        mask = (
            (daily_all_mxn["nodo"] == node_sel) &
            (daily_all_mxn["fecha"] >= start_ts) &
            (daily_all_mxn["fecha"] <= end_ts)
        )
        node_daily = daily_all_mxn.loc[mask].sort_values("fecha").copy()
        node_daily["spread_disp"] = node_daily["spread"] * conv_factor
        st.dataframe(
            node_daily[["fecha", "spread_disp", "n_hours"]]
                     .rename(columns={"spread_disp": f"spread ({cfg['currency']})"}),
            use_container_width=True
        )
        if not node_daily.empty:
            st.line_chart(node_daily.set_index("fecha")["spread_disp"])

# =====================================================================================
# 🗺️ Carte des nœuds (Folium) — avec géocodage MISE EN CACHE
# =====================================================================================
st.subheader("🗺️ Nodos map (colored by selected spread metric)")

# NodosP (depuis repo)
try:
    nodosp_df = read_nodosp_excel_from_repo()
except Exception as e:
    st.error(f"Lecture NodosP échouée depuis le repo: {e}")
    nodosp_df = None

if nodosp_df is None or nodosp_df.empty:
    st.info("NodosP indisponible — carte désactivée.")
else:
    nodes_for_map = res_show.copy()
    for c in ["nodo","ciudad","nombre"]:
        if c not in nodosp_df.columns:
            nodosp_df[c] = np.nan
    nodosp_df = nodosp_df[["nodo","ciudad","nombre"]].drop_duplicates("nodo")
    map_df = nodes_for_map.merge(nodosp_df, on="nodo", how="left")

    # Option Top N carte
    value_col = metric_col
    if cfg.get("map_subset_mode") == "Top N by avg spread":
        map_df = (map_df.dropna(subset=[value_col])
                         .sort_values(value_col, ascending=False)
                         .head(int(cfg.get("map_top_n", 200))))

    # Coords villes (mise en cache)
    map_df["city_norm"] = map_df["ciudad"].astype(str).map(normalize_city)
    unique_cities = tuple(sorted(set(map_df["city_norm"].dropna().astype(str))))
    city_coords = build_city_coords_cached(
        unique_cities=unique_cities,
        allow_online_geocoding=cfg.get("allow_online_geocoding", True)
    )
    map_df = map_df.merge(city_coords, on="city_norm", how="left")

    have_coords = map_df[map_df["lat"].notna() & map_df["lon"].notna()].copy()
    if have_coords.empty:
        st.warning("Aucune coordonnée trouvée (fournis NodosP + activer géocodage ?)")
    else:
        # Prépare signature de la carte pour éviter les reconstructions inutiles (flicker)
        hash_cols = ["nodo", "lat", "lon", value_col, "nombre", "nivel_tension", "ciudad", "n_days_used"]
        hash_df = have_coords[hash_cols].copy()
        hash_df = hash_df.applymap(lambda v: "" if pd.isna(v) else str(v))
        hash_bytes = hash_df.to_csv(index=False).encode("utf-8")
        map_signature = (
            metric_label,
            cfg["currency"],
            str(cfg["start_date"]),
            str(cfg["end_date"]),
            cfg.get("map_subset_mode"),
            int(cfg.get("map_top_n", 0)),
            tuple(selected_levels),
            st.session_state.get("selected_node"),
            hashlib.sha1(hash_bytes).hexdigest(),
        )

        if st.session_state.get("_map_signature") != map_signature or "_map_html" not in st.session_state:
            vmin = float(have_coords[value_col].min()) if have_coords[value_col].notna().any() else 0.0
            vmax = float(have_coords[value_col].max()) if have_coords[value_col].notna().any() else 1.0
            cmap = linear.YlOrRd_09.scale(vmin, vmax) if math.isfinite(vmin) and math.isfinite(vmax) and vmin != vmax else linear.YlOrRd_09.scale(0,1)

            # Centre/zoom par défaut
            center_lat, center_lon = 23.5, -102.0
            zoom_start = 5

            # Si un nœud a été demandé depuis le bar chart, on recadre et on ouvrira sa popup
            selected_node = st.session_state.get("selected_node", None)
            selected_row = None
            if selected_node and selected_node in have_coords["nodo"].values:
                sel = have_coords.loc[have_coords["nodo"] == selected_node].head(1)
                if not sel.empty:
                    center_lat = float(sel["lat"].iloc[0])
                    center_lon = float(sel["lon"].iloc[0])
                    zoom_start = 10
                    selected_row = sel.iloc[0]

            m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="cartodbpositron")
            cmap.caption = f"{metric_label}"
            cmap.add_to(m)

            have_coords["lat"] = have_coords["lat"].astype(float)
            have_coords["lon"] = have_coords["lon"].astype(float)
            cluster = MarkerCluster(name="Nodos").add_to(m)

            grouped = have_coords.groupby(["lat","lon"], as_index=False, sort=False)
            for (_, _), group in grouped:
                pts = jitter_positions(group, radius_m=400)
                for (_, row), (jlat, jlon) in zip(group.iterrows(), pts):
                    # Série horaire pour CE nœud (convertie en devise d’affichage)
                    ts_hourly = build_price_series_hourly(
                        df_all,
                        nodo=row["nodo"],
                        start_date=cfg["start_date"],
                        end_date=cfg["end_date"]
                    )
                    if ts_hourly.empty:
                        svg_chart = "<div style='font:12px Arial;color:#666'>No price data</div>"
                    else:
                        svg_chart = ts_to_svg_line(
                            ts_hourly["ts"],
                            ts_hourly["pml"] * (1.0 if cfg["currency"] == "MXN" else 1.0 / max(cfg["rate_usd_mxn"], 1e-9)),
                            width=760, height=280, pad=32, stroke_color="#1565C0",
                            max_points=1200
                        )

                    val = row.get(value_col, np.nan)
                    color = "#888888" if pd.isna(val) else cmap(val)

                    popup_html = f"""
<div style="font:13px Arial, sans-serif; min-width: 780px;">
  <div style="margin-bottom:6px;">
    <b>Nodo:</b> {_fmt(row.get('nodo'))} &nbsp;|&nbsp;
    <b>Nom:</b> {_fmt(row.get('nombre'))} &nbsp;|&nbsp;
    <b>Tension:</b> {_fmt(row.get('nivel_tension'))} kV &nbsp;|&nbsp;
    <b>Ciudad:</b> {_fmt(row.get('ciudad'))}
  </div>
  <div style="margin-bottom:8px;">
    <b>{metric_label}:</b> {(f"{val:,.2f} {cfg['currency']}/MWh" if pd.notna(val) else "—")} &nbsp;|&nbsp;
    <b>Days used:</b> {_fmt(row.get('n_days_used'))}<br>
    <span style="color:#666">Precio horario — {cfg['start_date']} → {cfg['end_date']} ({cfg['currency']})</span>
  </div>
  <div>{svg_chart}</div>
</div>
""".strip()

                    is_selected = (selected_row is not None) and (row["nodo"] == selected_row["nodo"])
                    marker = folium.CircleMarker(
                        location=[jlat, jlon],
                        radius=8 if is_selected else 5,
                        weight=2 if is_selected else 1,
                        color="#1f2d3d" if is_selected else "#333333",
                        fill=True, fill_opacity=0.95 if is_selected else 0.85,
                        fill_color="#2b8a3e" if is_selected else color,
                        tooltip=f"{_fmt(row.get('nodo'))} | {_fmt(row.get('nombre'))}",
                    )
                    folium.Popup(html=popup_html, max_width=1400, show=is_selected).add_to(marker)
                    marker.add_to(cluster)

            st.session_state["_map_signature"] = map_signature
            st.session_state["_map_html"] = m.get_root().render()

        components.html(st.session_state["_map_html"], height=680)
