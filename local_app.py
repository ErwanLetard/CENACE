from __future__ import annotations

import argparse
import csv
import hashlib
import os
import queue
import sqlite3
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import fmean
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple
import re

try:
    import tkinter as tk  # type: ignore
    from tkinter import filedialog, messagebox, ttk  # type: ignore

    TK_AVAILABLE = True
except ModuleNotFoundError:
    tk = None  # type: ignore
    filedialog = messagebox = ttk = None  # type: ignore
    TK_AVAILABLE = False

try:
    from tkcalendar import Calendar  # type: ignore

    TKCALENDAR_AVAILABLE = True
except ModuleNotFoundError:
    Calendar = None  # type: ignore
    TKCALENDAR_AVAILABLE = False

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # type: ignore
    from matplotlib.figure import Figure  # type: ignore

    MATPLOTLIB_AVAILABLE = True
except ModuleNotFoundError:
    FigureCanvasTkAgg = Figure = NavigationToolbar2Tk = None  # type: ignore
    MATPLOTLIB_AVAILABLE = False

DB_PATH = Path(
    os.environ.get("CENACE_DB_PATH", "/Volumes/HDD/cenace_local.db")
).expanduser()
CSV_HEADER_TOKEN = "Fecha"
HOUR_SECONDS = 3600
DEFAULT_CURRENCY_RATE = 0.058
DEFAULT_GUI_CURRENCY_RATE = DEFAULT_CURRENCY_RATE
NODE_VOLTAGE_PATTERN = re.compile(r"-([0-9]+(?:\.[0-9]+)?)$")
OPERATING_DAY_OFFSET = timedelta(hours=7)
OPERATING_DAY_SPAN = timedelta(hours=24)
BESS_WINDOWS = (2, 4, 8)
SPREAD_SOURCE_CHOICES = ("auto", "cache", "recompute")

SEASON_TO_MONTHS = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}


@dataclass
class PriceRecord:
    node_code: str
    hour: int
    timestamp: datetime
    pml: float
    energy_component: float
    loss_component: float
    congestion_component: float


@dataclass
class DailySpreadRecord:
    node_id: int
    node_code: str
    day_start: datetime
    window_hours: int
    charge_start: datetime
    discharge_start: datetime
    charge_mean: float
    discharge_mean: float
    spread: float


def operational_day_start(timestamp: datetime) -> datetime:
    shifted = timestamp - OPERATING_DAY_OFFSET
    base = datetime.combine(shifted.date(), datetime.min.time())
    return base + OPERATING_DAY_OFFSET


def operational_day_bounds(
    start: Optional[datetime],
    end: Optional[datetime],
) -> Tuple[Optional[datetime], Optional[datetime]]:
    start_day = operational_day_start(start) if start else None
    end_day = operational_day_start(end) if end else None
    return start_day, end_day


def get_connection() -> sqlite3.Connection:
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Impossible de créer le dossier base {DB_PATH.parent}: {exc}") from exc
    try:
        conn = sqlite3.connect(DB_PATH)
    except sqlite3.OperationalError as exc:
        raise RuntimeError(f"Impossible d'ouvrir la base {DB_PATH}: {exc}") from exc
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def initialise_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER NOT NULL,
            month INTEGER NOT NULL,
            run_label TEXT NOT NULL,
            dataset_version TEXT,
            imported_at TEXT NOT NULL,
            source_file TEXT NOT NULL UNIQUE,
            file_hash TEXT NOT NULL UNIQUE,
            rows_ingested INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            voltage_kv REAL
        );

        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            node_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
            timestamp TEXT NOT NULL,
            hour INTEGER NOT NULL,
            pml REAL NOT NULL,
            energy_component REAL NOT NULL,
            loss_component REAL NOT NULL,
            congestion_component REAL NOT NULL,
            UNIQUE (run_id, node_id, timestamp)
        );

        CREATE INDEX IF NOT EXISTS idx_prices_node_time
            ON prices(node_id, timestamp);

        CREATE INDEX IF NOT EXISTS idx_prices_time
            ON prices(timestamp);

        CREATE TABLE IF NOT EXISTS daily_spreads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
            day_start TEXT NOT NULL,
            window_hours INTEGER NOT NULL,
            charge_start TEXT NOT NULL,
            discharge_start TEXT NOT NULL,
            charge_mean REAL NOT NULL,
            discharge_mean REAL NOT NULL,
            spread REAL NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (node_id, day_start, window_hours)
        );

        CREATE INDEX IF NOT EXISTS idx_daily_spreads_node_day
            ON daily_spreads(node_id, day_start);
        """
    )

    try:
        conn.execute("ALTER TABLE nodes ADD COLUMN voltage_kv REAL;")
    except sqlite3.OperationalError:
        pass

    rows = conn.execute(
        "SELECT id, code FROM nodes WHERE voltage_kv IS NULL"
    ).fetchall()
    for row in rows:
        voltage = parse_node_voltage(row["code"])
        if voltage is not None:
            conn.execute(
                "UPDATE nodes SET voltage_kv = ? WHERE id = ?",
                (voltage, row["id"]),
            )


def compute_file_hash(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def parse_node_voltage(node_code: str) -> Optional[float]:
    match = NODE_VOLTAGE_PATTERN.search(node_code)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_datetime_input(value: Optional[str], *, end_of_day: bool = False) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid datetime value '{value}'. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM.") from exc

    if len(value) == 10:
        if end_of_day:
            dt = dt + timedelta(days=1) - timedelta(seconds=1)
    return dt


def parse_run_metadata(path: Path, first_timestamp: datetime) -> Tuple[int, int, str, Optional[str]]:
    year = first_timestamp.year
    month = first_timestamp.month
    parts = path.stem.split()
    run_label = "unknown"
    dataset_version = None

    if "Mes" in parts:
        idx = parts.index("Mes")
        if idx + 1 < len(parts):
            run_label = parts[idx + 1]
    for part in parts:
        if part.startswith("v") and part[1:].isdigit():
            dataset_version = part
            break

    return year, month, run_label, dataset_version


def skip_to_header(reader: Iterable[List[str]]) -> Optional[List[str]]:
    for row in reader:
        cleaned = [col.strip().strip('"') for col in row if col]
        if cleaned and cleaned[0] == CSV_HEADER_TOKEN:
            return [col.strip().strip('"') for col in row]
    return None


def load_price_records(path: Path) -> List[PriceRecord]:
    records: List[PriceRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        raw_reader = csv.reader(fh)
        header_row = skip_to_header(raw_reader)
        if not header_row:
            raise ValueError(f"Header row not found in {path}")

        cleaned_headers = [col.strip().strip('"').lower() for col in header_row]
        field_indices = {name: idx for idx, name in enumerate(cleaned_headers)}

        required_fields = {
            "fecha",
            "hora",
            "clave del nodo",
            "precio marginal local ($/mwh)",
            "componente de energia ($/mwh)",
            "componente de perdidas ($/mwh)",
            "componente de congestion ($/mwh)",
        }

        if not required_fields.issubset(field_indices):
            missing = required_fields - field_indices.keys()
            raise ValueError(f"Missing expected columns {missing} in {path}")

        for row in raw_reader:
            if not row:
                continue
            cells = [cell.strip().strip('"') for cell in row]
            if len(cells) < len(required_fields):
                continue

            fecha = cells[field_indices["fecha"]]
            hora_str = cells[field_indices["hora"]]
            node_code = cells[field_indices["clave del nodo"]]

            if not fecha or not hora_str or not node_code:
                continue

            try:
                hour = int(hora_str)
                date_obj = datetime.fromisoformat(fecha)
            except ValueError:
                continue

            timestamp = date_obj + timedelta(hours=hour - 1)

            try:
                pml = float(cells[field_indices["precio marginal local ($/mwh)"]])
                energy = float(cells[field_indices["componente de energia ($/mwh)"]])
                loss = float(cells[field_indices["componente de perdidas ($/mwh)"]])
                congestion = float(cells[field_indices["componente de congestion ($/mwh)"]])
            except ValueError:
                continue

            records.append(
                PriceRecord(
                    node_code=node_code,
                    hour=hour,
                    timestamp=timestamp,
                    pml=pml,
                    energy_component=energy,
                    loss_component=loss,
                    congestion_component=congestion,
                )
            )
    return records


def ensure_node(conn: sqlite3.Connection, cache: Dict[str, int], node_code: str) -> int:
    if node_code in cache:
        return cache[node_code]
    row = conn.execute("SELECT id, voltage_kv FROM nodes WHERE code = ?", (node_code,)).fetchone()
    voltage = parse_node_voltage(node_code)
    if row:
        if voltage is not None and row["voltage_kv"] is None:
            conn.execute(
                "UPDATE nodes SET voltage_kv = ? WHERE id = ?",
                (voltage, row["id"]),
            )
        cache[node_code] = row["id"]
        return row["id"]
    now = datetime.utcnow().isoformat(timespec="seconds")
    cursor = conn.execute(
        "INSERT INTO nodes (code, created_at, voltage_kv) VALUES (?, ?, ?)",
        (node_code, now, voltage),
    )
    node_id = cursor.lastrowid
    cache[node_code] = node_id
    return node_id


def ingest_csv(conn: sqlite3.Connection, path: Path) -> Tuple[bool, str]:
    records = load_price_records(path)
    if not records:
        return False, f"No valid records parsed in {path.name}"

    file_hash = compute_file_hash(path)
    existing = conn.execute(
        "SELECT id FROM runs WHERE file_hash = ? OR source_file = ?",
        (file_hash, str(path)),
    ).fetchone()
    if existing:
        return False, f"File {path.name} already imported (run id {existing['id']})"

    year, month, run_label, dataset_version = parse_run_metadata(path, records[0].timestamp)
    imported_at = datetime.utcnow().isoformat(timespec="seconds")
    node_cache: Dict[str, int] = {}

    with conn:
        cursor = conn.execute(
            """
            INSERT INTO runs (
                year, month, run_label, dataset_version,
                imported_at, source_file, file_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (year, month, run_label, dataset_version, imported_at, str(path), file_hash),
        )
        run_id = cursor.lastrowid

        price_rows = []
        for record in records:
            node_id = ensure_node(conn, node_cache, record.node_code)
            price_rows.append(
                (
                    run_id,
                    node_id,
                    record.timestamp.isoformat(sep=" "),
                    record.hour,
                    record.pml,
                    record.energy_component,
                    record.loss_component,
                    record.congestion_component,
                )
            )

        conn.executemany(
            """
            INSERT OR IGNORE INTO prices (
                run_id, node_id, timestamp, hour,
                pml, energy_component, loss_component, congestion_component
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            price_rows,
        )
        conn.execute(
            "UPDATE runs SET rows_ingested = ? WHERE id = ?",
            (len(price_rows), run_id),
        )

    return True, f"Imported {len(records)} rows from {path.name}"


def list_runs(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, year, month, run_label, dataset_version,
               imported_at, rows_ingested, source_file
        FROM runs
        ORDER BY year, month, run_label
        """
    ).fetchall()


def to_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def filter_params_to_clause(
    year: Optional[int],
    season: Optional[str],
    node: Optional[str],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    voltage_min: Optional[float] = None,
    voltage_max: Optional[float] = None,
    *,
    timestamp_column: str = "p.timestamp",
) -> Tuple[str, List[object]]:
    conditions: List[str] = []
    params: List[object] = []

    if year:
        conditions.append(f"strftime('%Y', {timestamp_column}) = ?")
        params.append(f"{year:04d}")

    if season:
        season_key = season.lower()
        if season_key not in SEASON_TO_MONTHS:
            raise ValueError(f"Unknown season '{season}'. Expected one of {list(SEASON_TO_MONTHS)}.")
        month_values = [f"{month:02d}" for month in SEASON_TO_MONTHS[season_key]]
        placeholders = ",".join("?" for _ in month_values)
        conditions.append(f"strftime('%m', {timestamp_column}) IN ({placeholders})")
        params.extend(month_values)

    if node:
        conditions.append("n.code = ?")
        params.append(node)

    if start:
        conditions.append(f"{timestamp_column} >= ?")
        params.append(start.isoformat(sep=" "))

    if end:
        conditions.append(f"{timestamp_column} <= ?")
        params.append(end.isoformat(sep=" "))

    if voltage_min is not None:
        conditions.append("n.voltage_kv >= ?")
        params.append(voltage_min)

    if voltage_max is not None:
        conditions.append("n.voltage_kv <= ?")
        params.append(voltage_max)

    clause = ""
    if conditions:
        clause = "WHERE " + " AND ".join(conditions)
    return clause, params


def fetch_price_series(
    conn: sqlite3.Connection,
    year: Optional[int],
    season: Optional[str],
    node: Optional[str],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    voltage_min: Optional[float] = None,
    voltage_max: Optional[float] = None,
) -> List[Tuple[str, datetime, float]]:
    clause, params = filter_params_to_clause(
        year,
        season,
        node,
        start=start,
        end=end,
        voltage_min=voltage_min,
        voltage_max=voltage_max,
    )
    query = f"""
        SELECT n.code AS node_code, p.timestamp, p.pml
        FROM prices p
        JOIN nodes n ON n.id = p.node_id
        {clause}
        ORDER BY n.code, p.timestamp
    """
    rows = conn.execute(query, params).fetchall()
    return [(row["node_code"], to_datetime(row["timestamp"]), row["pml"]) for row in rows]


def fetch_price_series_with_nodes(
    conn: sqlite3.Connection,
    year: Optional[int],
    season: Optional[str],
    node: Optional[str],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    voltage_min: Optional[float] = None,
    voltage_max: Optional[float] = None,
) -> List[Tuple[int, str, datetime, float]]:
    clause, params = filter_params_to_clause(
        year,
        season,
        node,
        start=start,
        end=end,
        voltage_min=voltage_min,
        voltage_max=voltage_max,
    )
    query = f"""
        SELECT n.id AS node_id,
               n.code AS node_code,
               p.timestamp AS timestamp,
               p.pml AS price
        FROM prices p
        JOIN nodes n ON n.id = p.node_id
        {clause}
        ORDER BY n.code, p.timestamp
    """
    rows = conn.execute(query, params).fetchall()
    return [
        (row["node_id"], row["node_code"], to_datetime(row["timestamp"]), row["price"])
        for row in rows
    ]


def compute_spread_stats(
    series: List[Tuple[str, datetime, float]],
    window_hours: int,
    *,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    total_nodes: Optional[int] = None,
) -> List[Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    per_node: Dict[str, Deque[Tuple[datetime, float]]] = {}
    processed_nodes = 0
    current_node: Optional[str] = None

    for node_code, timestamp, pml in series:
        if node_code != current_node:
            current_node = node_code
            processed_nodes += 1
            if progress_callback:
                progress_callback(processed_nodes, total_nodes or processed_nodes)

        window = per_node.setdefault(node_code, deque())

        # Reset when data is not strictly hourly to enforce true rolling windows.
        if window and (timestamp - window[-1][0]).total_seconds() != HOUR_SECONDS:
            window.clear()

        window.append((timestamp, pml))

        while len(window) > window_hours:
            window.popleft()

        if len(window) != window_hours:
            continue

        expected_span = (window_hours - 1) * HOUR_SECONDS
        actual_span = (window[-1][0] - window[0][0]).total_seconds()
        if actual_span != expected_span:
            continue

        window_values = [value for _, value in window]
        spread = max(window_values) - min(window_values)
        stats = results.setdefault(
            node_code,
            {
                "node_code": node_code,
                "spreads": [],
                "max_spread": float("-inf"),
                "max_window_start": window[0][0],
                "max_window_end": window[-1][0],
            },
        )
        stats["spreads"].append(spread)
        if spread > stats["max_spread"]:
            stats["max_spread"] = spread
            stats["max_window_start"] = window[0][0]
            stats["max_window_end"] = window[-1][0]

    aggregated: List[Dict[str, object]] = []
    for stats in results.values():
        spreads = stats["spreads"]
        if not spreads:
            continue
        aggregated.append(
            {
                "node_code": stats["node_code"],
                "average_spread": fmean(spreads),
                "spread_count": len(spreads),
                "max_spread": stats["max_spread"],
                "max_window_start": stats["max_window_start"],
                "max_window_end": stats["max_window_end"],
            }
        )

    aggregated.sort(key=lambda item: item["average_spread"], reverse=True)
    return aggregated


def _is_complete_operational_day(
    entries: List[Tuple[datetime, float]],
    day_start: datetime,
) -> bool:
    if len(entries) != 24:
        return False
    cursor = day_start
    for ts, _ in entries:
        if ts != cursor:
            return False
        cursor += timedelta(hours=1)
    return True


def _best_bess_pair(
    entries: List[Tuple[datetime, float]],
    window_hours: int,
) -> Optional[Tuple[datetime, datetime, float, float, float]]:
    total_hours = len(entries)
    block = window_hours
    if block * 2 > total_hours:
        return None
    best: Optional[Tuple[datetime, datetime, float, float, float]] = None
    best_spread = float("-inf")
    for charge_idx in range(0, total_hours - (2 * block) + 1):
        charge_slice = entries[charge_idx : charge_idx + block]
        charge_avg = fmean(price for _, price in charge_slice)
        charge_start = charge_slice[0][0]
        for discharge_idx in range(charge_idx + block, total_hours - block + 1):
            discharge_slice = entries[discharge_idx : discharge_idx + block]
            discharge_avg = fmean(price for _, price in discharge_slice)
            discharge_start = discharge_slice[0][0]
            spread = discharge_avg - charge_avg
            if spread > best_spread:
                best_spread = spread
                best = (charge_start, discharge_start, charge_avg, discharge_avg, spread)
            elif spread == best_spread and best is not None:
                prev_charge, prev_discharge, *_ = best
                if (charge_start < prev_charge) or (
                    charge_start == prev_charge and discharge_start < prev_discharge
                ):
                    best = (charge_start, discharge_start, charge_avg, discharge_avg, spread)
    return best


def compute_daily_spreads_from_series(
    rows: Iterable[Tuple[int, str, datetime, float]],
    window_hours_list: Iterable[int],
    *,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    total_nodes: Optional[int] = None,
) -> List[DailySpreadRecord]:
    results: List[DailySpreadRecord] = []
    current_node_id: Optional[int] = None
    current_node_code = ""
    current_day_start: Optional[datetime] = None
    day_entries: List[Tuple[datetime, float]] = []
    processed_nodes = 0

    window_set = tuple(sorted(set(window_hours_list)))

    def flush_day() -> None:
        nonlocal day_entries, current_day_start
        if not day_entries or current_day_start is None:
            day_entries = []
            return
        day_entries.sort(key=lambda item: item[0])
        if not _is_complete_operational_day(day_entries, current_day_start):
            day_entries = []
            return
        for window in window_set:
            best = _best_bess_pair(day_entries, window)
            if not best:
                continue
            charge_start, discharge_start, charge_avg, discharge_avg, spread = best
            results.append(
                DailySpreadRecord(
                    node_id=current_node_id,  # type: ignore[arg-type]
                    node_code=current_node_code,
                    day_start=current_day_start,
                    window_hours=window,
                    charge_start=charge_start,
                    discharge_start=discharge_start,
                    charge_mean=charge_avg,
                    discharge_mean=discharge_avg,
                    spread=spread,
                )
            )
        day_entries = []

    for node_id, node_code, timestamp, price in rows:
        if current_node_id is None:
            current_node_id = node_id
            current_node_code = node_code
            current_day_start = None
            day_entries = []
        elif node_id != current_node_id:
            flush_day()
            if progress_callback:
                processed_nodes += 1
                total = total_nodes or processed_nodes
                progress_callback(processed_nodes, total)
            current_node_id = node_id
            current_node_code = node_code
            current_day_start = None
            day_entries = []
        day_start = operational_day_start(timestamp)
        if current_day_start is None:
            current_day_start = day_start
        elif day_start != current_day_start:
            flush_day()
            current_day_start = day_start
        day_entries.append((timestamp.replace(minute=0, second=0, microsecond=0), price))

    flush_day()
    if current_node_id is not None and progress_callback:
        processed_nodes += 1
        total = total_nodes or processed_nodes
        progress_callback(processed_nodes, total)
    return results


def _operational_price_range(
    start_day: Optional[datetime],
    end_day: Optional[datetime],
) -> Tuple[Optional[datetime], Optional[datetime]]:
    price_start = start_day
    price_end = None
    if end_day:
        price_end = end_day + OPERATING_DAY_SPAN - timedelta(seconds=1)
    return price_start, price_end


def compute_and_store_daily_spreads(
    conn: sqlite3.Connection,
    window_hours: int,
    *,
    year: Optional[int],
    season: Optional[str],
    node: Optional[str],
    start: Optional[datetime],
    end: Optional[datetime],
    voltage_min: Optional[float],
    voltage_max: Optional[float],
    rows: Optional[List[Tuple[int, str, datetime, float]]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> int:
    start_day, end_day = operational_day_bounds(start, end)
    price_start, price_end = _operational_price_range(start_day, end_day)
    if rows is None:
        rows = fetch_price_series_with_nodes(
            conn,
            year,
            season,
            node,
            start=price_start,
            end=price_end,
            voltage_min=voltage_min,
            voltage_max=voltage_max,
        )
    if not rows:
        return 0
    node_ids = sorted({row[0] for row in rows})
    records = compute_daily_spreads_from_series(
        rows,
        [window_hours],
        progress_callback=progress_callback,
        total_nodes=len(node_ids) if node_ids else None,
    )
    saved = save_daily_spread_records(conn, records)
    return saved


def save_daily_spread_records(
    conn: sqlite3.Connection,
    records: List[DailySpreadRecord],
) -> int:
    if not records:
        return 0
    now = datetime.utcnow().isoformat(timespec="seconds")
    rows = [
        (
            record.node_id,
            record.day_start.isoformat(sep=" "),
            record.window_hours,
            record.charge_start.isoformat(sep=" "),
            record.discharge_start.isoformat(sep=" "),
            record.charge_mean,
            record.discharge_mean,
            record.spread,
            now,
        )
        for record in records
    ]
    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO daily_spreads (
                node_id, day_start, window_hours,
                charge_start, discharge_start,
                charge_mean, discharge_mean, spread, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def fetch_daily_spread_rows(
    conn: sqlite3.Connection,
    window_hours: int,
    year: Optional[int],
    season: Optional[str],
    node: Optional[str],
    start_day: Optional[datetime],
    end_day: Optional[datetime],
    voltage_min: Optional[float],
    voltage_max: Optional[float],
) -> List[sqlite3.Row]:
    clause, params = filter_params_to_clause(
        year,
        season,
        node,
        start=start_day,
        end=end_day,
        voltage_min=voltage_min,
        voltage_max=voltage_max,
        timestamp_column="ds.day_start",
    )
    base_condition = "WHERE ds.window_hours = ?"
    params = [window_hours] + params
    if clause:
        base_condition += " AND " + clause[len("WHERE ") :]
    query = f"""
        SELECT ds.day_start,
               ds.window_hours,
               ds.charge_start,
               ds.discharge_start,
               ds.charge_mean,
               ds.discharge_mean,
               ds.spread,
               n.code AS node_code
        FROM daily_spreads ds
        JOIN nodes n ON n.id = ds.node_id
        {base_condition}
        ORDER BY n.code, ds.day_start
    """
    return conn.execute(query, params).fetchall()


def summarise_daily_spread_rows(
    rows: List[sqlite3.Row],
) -> Tuple[List[Dict[str, object]], Dict[str, List[Tuple[datetime, float]]]]:
    per_node: Dict[str, Dict[str, object]] = {}
    series: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)

    for row in rows:
        node_code = row["node_code"]
        day_start = to_datetime(row["day_start"])
        spread_value = row["spread"]
        series[node_code].append((day_start, spread_value))

        stats = per_node.setdefault(
            node_code,
            {
                "node_code": node_code,
                "total_spread": 0.0,
                "count": 0,
                "max_spread": float("-inf"),
                "charge_start": None,
                "discharge_start": None,
                "charge_mean": 0.0,
                "discharge_mean": 0.0,
                "window_hours": row["window_hours"],
            },
        )
        stats["total_spread"] += spread_value
        stats["count"] += 1
        if spread_value > stats["max_spread"]:
            stats["max_spread"] = spread_value
            stats["charge_start"] = to_datetime(row["charge_start"])
            stats["discharge_start"] = to_datetime(row["discharge_start"])
            stats["charge_mean"] = row["charge_mean"]
            stats["discharge_mean"] = row["discharge_mean"]

    aggregated: List[Dict[str, object]] = []
    for stats in per_node.values():
        count = stats["count"]
        if not count:
            continue
        aggregated.append(
            {
                "node_code": stats["node_code"],
                "average_spread": stats["total_spread"] / count,
                "spread_count": count,
                "max_spread": stats["max_spread"],
                "charge_start": stats["charge_start"],
                "discharge_start": stats["discharge_start"],
                "window_hours": stats["window_hours"],
            }
        )

    aggregated.sort(key=lambda item: item["average_spread"], reverse=True)
    return aggregated, series


def spreads_cover_range(
    rows: List[sqlite3.Row],
    start_day: Optional[datetime],
    end_day: Optional[datetime],
) -> bool:
    if not rows:
        return False
    day_values = [to_datetime(row["day_start"]) for row in rows]
    if not day_values:
        return False
    min_day = min(day_values)
    max_day = max(day_values)
    if start_day and start_day < min_day:
        return False
    if end_day and end_day > max_day:
        return False
    return True


def compute_moving_average(
    series: List[Tuple[datetime, float]],
    window: int,
) -> List[Tuple[datetime, float]]:
    if window <= 1:
        return list(series)
    result: List[Tuple[datetime, float]] = []
    buffer: Deque[Tuple[datetime, float]] = deque()
    total = 0.0
    for timestamp, value in series:
        buffer.append((timestamp, value))
        total += value
        if len(buffer) > window:
            _, popped_value = buffer.popleft()
            total -= popped_value
        if len(buffer) == window:
            result.append((buffer[-1][0], total / window))
    return result


def format_spread_line(
    entry: Dict[str, object],
    currency_rate: float,
    target_label: str,
) -> str:
    avg_converted = entry["average_spread"] * currency_rate
    max_converted = entry["max_spread"] * currency_rate
    label = target_label.upper()
    charge_start: Optional[datetime] = entry.get("charge_start")
    discharge_start: Optional[datetime] = entry.get("discharge_start")
    window_hours = entry.get("window_hours", 0) or 0
    if charge_start and discharge_start:
        charge_end = charge_start + timedelta(hours=window_hours)
        discharge_end = discharge_start + timedelta(hours=window_hours)
        detail = (
            f"charge {charge_start.strftime('%Y-%m-%d %H:%M')}→{charge_end.strftime('%H:%M')} | "
            f"décharge {discharge_start.strftime('%Y-%m-%d %H:%M')}→{discharge_end.strftime('%H:%M')}"
        )
    else:
        detail = "meilleure fenêtre indisponible"
    return (
        f"{entry['node_code']} · spread moyen={avg_converted:.2f} {label} "
        f"(journées={entry['spread_count']}) | "
        f"{detail} | meilleur spread={max_converted:.2f} {label}"
    )


def command_init_db(_: argparse.Namespace) -> None:
    coverage_ok = False
    with get_connection() as conn:
        initialise_schema(conn)
    print(f"Database initialised at {DB_PATH}")


def command_ingest(args: argparse.Namespace) -> None:
    source_dir = Path(args.source).expanduser()
    if not source_dir.exists():
        raise SystemExit(f"Source directory {source_dir} does not exist.")

    csv_files = sorted(source_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {source_dir}")

    with get_connection() as conn:
        initialise_schema(conn)
        for csv_path in csv_files:
            ok, message = ingest_csv(conn, csv_path)
            prefix = "[OK]" if ok else "[SKIP]"
            print(f"{prefix} {message}")
            if ok:
                try:
                    csv_path.unlink()
                    print(f"[DEL] Deleted {csv_path}")
                except OSError as exc:
                    print(f"[WARN] Unable to delete {csv_path}: {exc}")


def command_list_runs(_: argparse.Namespace) -> None:
    with get_connection() as conn:
        initialise_schema(conn)
        rows = list_runs(conn)
    if not rows:
        print("No runs found.")
        return
    for row in rows:
        label = f"{row['year']:04d}-{row['month']:02d} {row['run_label']}"
        print(
            f"[{row['id']}] {label} | version={row['dataset_version'] or 'n/a'} | "
            f"rows={row['rows_ingested']} | imported={row['imported_at']} | file={row['source_file']}"
        )


def command_spreads(args: argparse.Namespace) -> None:
    window_hours = args.window
    if window_hours < 2:
        raise SystemExit("Window size must be >= 2 hours.")

    try:
        start_dt = parse_datetime_input(args.start)
        end_dt = parse_datetime_input(args.end, end_of_day=True)
    except ValueError as exc:
        raise SystemExit(str(exc))

    if start_dt and end_dt and start_dt > end_dt:
        raise SystemExit("Start date must be before end date.")

    currency_rate = args.currency_rate
    if currency_rate <= 0:
        raise SystemExit("Currency conversion rate must be positive.")
    target_label = args.currency_target
    spread_source = args.spread_source

    voltage_min = args.voltage_min
    voltage_max = args.voltage_max
    if voltage_min is not None and voltage_max is not None and voltage_min > voltage_max:
        raise SystemExit("voltage-min must be <= voltage-max.")

    start_day, end_day = operational_day_bounds(start_dt, end_dt)

    with get_connection() as conn:
        initialise_schema(conn)
        spread_rows = fetch_daily_spread_rows(
            conn,
            window_hours,
            args.year,
            args.season,
            args.node,
            start_day,
            end_day,
            voltage_min,
            voltage_max,
        )

        coverage_ok = spreads_cover_range(spread_rows, start_day, end_day)
        needs_compute = spread_source == "recompute" or (spread_source == "auto" and not coverage_ok)
        if needs_compute:
            inserted = compute_and_store_daily_spreads(
                conn,
                window_hours,
                year=args.year,
                season=args.season,
                node=args.node,
                start=start_dt,
                end=end_dt,
                voltage_min=voltage_min,
                voltage_max=voltage_max,
            )
            spread_rows = fetch_daily_spread_rows(
                conn,
                window_hours,
                args.year,
                args.season,
                args.node,
                start_day,
                end_day,
                voltage_min,
                voltage_max,
            )
            if spread_source == "recompute":
                print(f"[cache] {inserted} journées recalculées.")
            coverage_ok = spreads_cover_range(spread_rows, start_day, end_day)
            if not coverage_ok and (start_day or end_day):
                print(
                    "Warning: spreads for some requested days are unavailable (données manquantes). "
                    "Résultats partiels."
                )
        elif spread_source == "cache" and not coverage_ok:
            print("Cached spreads do not fully cover the requested dates. Recompute to refresh them.")
            return
        else:
            coverage_ok = spreads_cover_range(spread_rows, start_day, end_day)

    if not spread_rows:
        print("No daily spreads available for the selected filters.")
        return

    spread_stats, _ = summarise_daily_spread_rows(spread_rows)
    if not spread_stats:
        print("No spreads found for complete operational days (07h→07h).")
        return

    top_n = args.top or 10
    for entry in spread_stats[:top_n]:
        print(
            format_spread_line(
                entry,
                currency_rate,
                target_label,
            )
        )


if TK_AVAILABLE:

    class LocalAppGUI:
        def __init__(self) -> None:
            self.root = tk.Tk()
            self.root.title("CENACE Local Analyzer")
            self.root.geometry("960x640")
            self.queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()
            self._task_running = False

            self.start_var = tk.StringVar()
            self.end_var = tk.StringVar()
            self.window_var = tk.StringVar(value="2")
            self.top_var = tk.StringVar(value="10")
            self.node_var = tk.StringVar()
            self.voltage_min_var = tk.StringVar()
            self.voltage_max_var = tk.StringVar()
            self.currency_rate_var = tk.StringVar(value=f"{DEFAULT_GUI_CURRENCY_RATE:.3f}")
            self.currency_label_var = tk.StringVar(value="USD")
            self.spread_source_var = tk.StringVar(value="auto")
            self.ma_window_var = tk.StringVar(value="7")

            self._chart_series: Dict[str, List[Tuple[datetime, float]]] = {}
            self._chart_currency_rate = DEFAULT_GUI_CURRENCY_RATE
            self._chart_currency_label = "USD"
            self._node_listbox: Optional[tk.Listbox] = None
            self._chart_axes = None
            self._chart_canvas = None
            self._chart_figure = None
            self._spread_chart_series: Dict[str, List[Tuple[datetime, float]]] = {}
            self._spread_ma_window = 7
            self._spread_axes = None
            self._spread_canvas = None
            self._spread_figure = None
            self._spread_chart_currency_rate = DEFAULT_GUI_CURRENCY_RATE
            self._spread_chart_currency_label = "USD"

            self._build_layout()
            self._poll_queue()

        def run(self) -> None:
            self.root.mainloop()

        def on_init_db(self) -> None:
            if self._task_running:
                return
            confirm = messagebox.askyesno(
                "Confirmer la réinitialisation",
                "Cette action peut reconstruire la base locale et effacer les données existantes.\n"
                "Voulez-vous continuer ?",
                icon="warning",
            )
            if not confirm:
                return
            self._start_task(indeterminate=True, status="Initialisation…")

            def work() -> None:
                try:
                    with get_connection() as conn:
                        initialise_schema(conn)
                    self.queue.put(("log", f"Base initialisée : {DB_PATH}"))
                    self.queue.put(("status", "Base initialisée."))
                except Exception as exc:  # noqa: BLE001
                    self.queue.put(("error", str(exc)))
                finally:
                    self.queue.put(("task_done", None))

            self._run_async(work)

        def on_import_csv(self) -> None:
            if self._task_running:
                return
            paths = filedialog.askopenfilenames(
                parent=self.root,
                title="Sélectionner des fichiers CSV",
                filetypes=[("Fichiers CSV", "*.csv")],
            )
            if not paths:
                return
            total = len(paths)
            self._start_task(maximum=total, status="Import en cours…")

            def work() -> None:
                try:
                    with get_connection() as conn:
                        initialise_schema(conn)
                        for idx, path_str in enumerate(paths, start=1):
                            csv_path = Path(path_str)
                            ok, message = ingest_csv(conn, csv_path)
                            prefix = "OK" if ok else "SKIP"
                            self.queue.put(("log", f"[{prefix}] {message}"))
                            if ok:
                                try:
                                    csv_path.unlink()
                                    self.queue.put(("log", f"[DEL] {csv_path} supprimé après import."))
                                except OSError as exc:
                                    self.queue.put(("log", f"[WARN] Impossible de supprimer {csv_path}: {exc}"))
                            self.queue.put(("progress", idx))
                            self.queue.put(("status", message))
                    self.queue.put(("status", "Import terminé."))
                except Exception as exc:  # noqa: BLE001
                    self.queue.put(("error", str(exc)))
                finally:
                    self.queue.put(("task_done", None))

            self._run_async(work)

        def on_compute_spread(self) -> None:
            if self._task_running:
                return

            try:
                window_hours = int(self.window_var.get())
            except ValueError:
                messagebox.showerror("Fenêtre invalide", "Choisir une fenêtre parmi 2, 4 ou 8 heures.")
                return
            if window_hours not in (2, 4, 8):
                messagebox.showerror("Fenêtre invalide", "Choisir une fenêtre parmi 2, 4 ou 8 heures.")
                return

            try:
                top_n = int(self.top_var.get())
            except ValueError:
                messagebox.showerror("Valeur invalide", "Top résultats doit être un entier positif.")
                return
            if top_n <= 0:
                messagebox.showerror("Valeur invalide", "Top résultats doit être un entier positif.")
                return

            try:
                start_dt = parse_datetime_input(self.start_var.get())
                end_dt = parse_datetime_input(self.end_var.get(), end_of_day=True)
            except ValueError as exc:
                messagebox.showerror("Date invalide", str(exc))
                return
            if start_dt and end_dt and start_dt > end_dt:
                messagebox.showerror("Dates invalides", "La date de début doit précéder la date de fin.")
                return

            try:
                currency_rate = float(self.currency_rate_var.get())
            except ValueError:
                messagebox.showerror("Taux invalide", "Le taux de conversion doit être numérique.")
                return
            if currency_rate <= 0:
                messagebox.showerror("Taux invalide", "Le taux de conversion doit être positif.")
                return
            target_label = self.currency_label_var.get().strip() or "USD"
            node_filter = self.node_var.get().strip() or None
            spread_source = self.spread_source_var.get().strip().lower()
            if spread_source not in SPREAD_SOURCE_CHOICES:
                spread_source = "auto"

            voltage_min_raw = self.voltage_min_var.get().strip()
            voltage_max_raw = self.voltage_max_var.get().strip()
            try:
                voltage_min_value = float(voltage_min_raw) if voltage_min_raw else None
            except ValueError:
                messagebox.showerror("Tension invalide", "La tension minimale doit être numérique.")
                return
            try:
                voltage_max_value = float(voltage_max_raw) if voltage_max_raw else None
            except ValueError:
                messagebox.showerror("Tension invalide", "La tension maximale doit être numérique.")
                return
            if (
                voltage_min_value is not None
                and voltage_max_value is not None
                and voltage_min_value > voltage_max_value
            ):
                messagebox.showerror(
                    "Tension invalide",
                    "La tension minimale doit être inférieure ou égale à la maximale.",
                )
                return

            try:
                ma_window = int(self.ma_window_var.get())
            except ValueError:
                messagebox.showerror("MA invalide", "La moyenne mobile doit être un entier positif.")
                return
            if ma_window <= 0:
                messagebox.showerror("MA invalide", "La moyenne mobile doit être un entier positif.")
                return
            self.ma_window_var.set(str(ma_window))

            self._start_task(indeterminate=True, status="Calcul du spread…")

            def work() -> None:
                try:
                    with get_connection() as conn:
                        initialise_schema(conn)
                        detailed_series = fetch_price_series_with_nodes(
                            conn,
                            None,
                            None,
                            node_filter,
                            start=start_dt,
                            end=end_dt,
                            voltage_min=voltage_min_value,
                            voltage_max=voltage_max_value,
                        )
                        price_series = [
                            (node_code, timestamp, price)
                            for _, node_code, timestamp, price in detailed_series
                        ]
                        start_day, end_day = operational_day_bounds(start_dt, end_dt)
                        spread_rows = fetch_daily_spread_rows(
                            conn,
                            window_hours,
                            None,
                            None,
                            node_filter,
                            start_day,
                            end_day,
                            voltage_min_value,
                            voltage_max_value,
                        )
                        coverage_ok = spreads_cover_range(spread_rows, start_day, end_day)
                        needs_compute = spread_source == "recompute" or (spread_source == "auto" and not coverage_ok)
                        if needs_compute and not detailed_series:
                            self.queue.put(
                                (
                                    "error",
                                    "Impossible de recalculer les spreads : aucune donnée horaire pour cette sélection.",
                                )
                            )
                            return
                        if needs_compute:
                            node_ids = sorted({row[0] for row in detailed_series})
                            if node_ids:
                                self.queue.put(("progress_setup", len(node_ids)))

                            def progress_cb(done: int, total: int) -> None:
                                self.queue.put(("progress_value", (done, total)))

                            compute_and_store_daily_spreads(
                                conn,
                                window_hours,
                                year=None,
                                season=None,
                                node=node_filter,
                                start=start_dt,
                                end=end_dt,
                                voltage_min=voltage_min_value,
                                voltage_max=voltage_max_value,
                                rows=detailed_series,
                                progress_callback=progress_cb,
                            )
                            spread_rows = fetch_daily_spread_rows(
                                conn,
                                window_hours,
                                None,
                                None,
                                node_filter,
                                start_day,
                                end_day,
                                voltage_min_value,
                                voltage_max_value,
                            )
                            coverage_ok = spreads_cover_range(spread_rows, start_day, end_day)
                            if not coverage_ok and (start_day or end_day):
                                self.queue.put(
                                    (
                                        "log",
                                        "Attention: certaines journées 07h→07h manquent dans les données (résultats partiels).",
                                    )
                                )
                        elif spread_source == "cache" and not coverage_ok:
                            self.queue.put(("error", "Aucun spread en cache pour ces critères. Recalculez-les."))
                            return
                        else:
                            coverage_ok = spreads_cover_range(spread_rows, start_day, end_day)
                            if not coverage_ok and (start_day or end_day):
                                self.queue.put(
                                    (
                                        "log",
                                        "Attention: certaines journées 07h→07h manquent dans les données (résultats partiels).",
                                    )
                                )

                    if not spread_rows:
                        self.queue.put(("clear_results", None))
                        self.queue.put(("log", "Aucun spread quotidien disponible pour ces critères."))
                        return

                    stats, spread_series = summarise_daily_spread_rows(spread_rows)
                    if not stats:
                        self.queue.put(("clear_results", None))
                        self.queue.put(("log", "Aucune journée complète 07h→07h trouvée."))
                        return

                    top_entries = stats[:top_n]
                    lines = [
                        format_spread_line(entry, currency_rate, target_label)
                        for entry in top_entries
                    ]

                    selected_nodes = [entry["node_code"] for entry in top_entries]
                    series_by_node: Dict[str, List[Tuple[datetime, float]]] = {
                        node: [] for node in selected_nodes
                    }
                    if series_by_node:
                        for node_code, ts, price in price_series:
                            if node_code in series_by_node:
                                series_by_node[node_code].append((ts, price))

                    self.queue.put(("show_results", lines))
                    self.queue.put(
                        (
                            "chart_data",
                            {
                                "nodes": selected_nodes,
                                "series": series_by_node,
                                "currency_rate": currency_rate,
                                "target_label": target_label,
                            },
                        )
                    )
                    self.queue.put(
                        (
                            "spread_chart_data",
                            {
                                "series": {node: values for node, values in spread_series.items()},
                                "ma_window": ma_window,
                                "currency_rate": currency_rate,
                                "target_label": target_label,
                            },
                        )
                    )
                    self.queue.put(("status", "Calcul terminé."))
                except Exception as exc:  # noqa: BLE001
                    self.queue.put(("error", str(exc)))
                finally:
                    self.queue.put(("task_done", None))

            self._run_async(work)

        def _build_layout(self) -> None:
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)

            main_frame = ttk.Frame(self.root, padding=12)
            main_frame.grid(column=0, row=0, sticky="nsew")
            main_frame.columnconfigure(0, weight=1)
            main_frame.rowconfigure(3, weight=1)

            db_frame = ttk.LabelFrame(main_frame, text="Base de données")
            db_frame.grid(column=0, row=0, sticky="ew", padx=4, pady=4)
            for col in range(2):
                db_frame.columnconfigure(col, weight=1)

            self.init_button = ttk.Button(db_frame, text="Initialiser la base", command=self.on_init_db)
            self.init_button.grid(column=0, row=0, padx=4, pady=4, sticky="ew")

            self.import_button = ttk.Button(db_frame, text="Importer CSV…", command=self.on_import_csv)
            self.import_button.grid(column=1, row=0, padx=4, pady=4, sticky="ew")

            analysis_frame = ttk.LabelFrame(main_frame, text="Analyse des spreads")
            analysis_frame.grid(column=0, row=1, sticky="ew", padx=4, pady=8)
            for col in range(6):
                analysis_frame.columnconfigure(col, weight=1)

            ttk.Label(analysis_frame, text="Début (YYYY-MM-DD)").grid(column=0, row=0, padx=4, pady=4, sticky="w")
            start_input = self._create_date_input(analysis_frame, self.start_var)
            start_input.grid(column=1, row=0, padx=4, pady=4, sticky="ew")

            ttk.Label(analysis_frame, text="Fin (YYYY-MM-DD)").grid(column=2, row=0, padx=4, pady=4, sticky="w")
            end_input = self._create_date_input(analysis_frame, self.end_var)
            end_input.grid(column=3, row=0, padx=4, pady=4, sticky="ew")

            ttk.Label(analysis_frame, text="Fenêtre (h)").grid(column=0, row=1, padx=4, pady=4, sticky="w")
            self.window_combo = ttk.Combobox(
                analysis_frame,
                textvariable=self.window_var,
                values=("2", "4", "8"),
                state="readonly",
            )
            self.window_combo.grid(column=1, row=1, padx=4, pady=4, sticky="ew")

            ttk.Label(analysis_frame, text="Top résultats").grid(column=2, row=1, padx=4, pady=4, sticky="w")
            ttk.Entry(analysis_frame, textvariable=self.top_var, width=6).grid(column=3, row=1, padx=4, pady=4, sticky="ew")

            ttk.Label(analysis_frame, text="Noeud (optionnel)").grid(column=0, row=2, padx=4, pady=4, sticky="w")
            ttk.Entry(analysis_frame, textvariable=self.node_var).grid(column=1, row=2, padx=4, pady=4, sticky="ew")

            ttk.Label(analysis_frame, text="Taux MXN → devise").grid(column=2, row=2, padx=4, pady=4, sticky="w")
            ttk.Entry(analysis_frame, textvariable=self.currency_rate_var).grid(column=3, row=2, padx=4, pady=4, sticky="ew")

            ttk.Label(analysis_frame, text="Devise cible").grid(column=4, row=2, padx=4, pady=4, sticky="w")
            ttk.Entry(analysis_frame, textvariable=self.currency_label_var, width=8).grid(column=5, row=2, padx=4, pady=4, sticky="ew")

            ttk.Label(analysis_frame, text="Tension min (kV)").grid(column=0, row=3, padx=4, pady=4, sticky="w")
            ttk.Entry(analysis_frame, textvariable=self.voltage_min_var).grid(column=1, row=3, padx=4, pady=4, sticky="ew")

            ttk.Label(analysis_frame, text="Tension max (kV)").grid(column=2, row=3, padx=4, pady=4, sticky="w")
            ttk.Entry(analysis_frame, textvariable=self.voltage_max_var).grid(column=3, row=3, padx=4, pady=4, sticky="ew")

            ttk.Label(analysis_frame, text="Source spreads").grid(column=0, row=4, padx=4, pady=4, sticky="w")
            self.spread_source_combo = ttk.Combobox(
                analysis_frame,
                textvariable=self.spread_source_var,
                values=SPREAD_SOURCE_CHOICES,
                state="readonly",
            )
            self.spread_source_combo.grid(column=1, row=4, padx=4, pady=4, sticky="ew")

            ttk.Label(analysis_frame, text="MA (jours)").grid(column=2, row=4, padx=4, pady=4, sticky="w")
            ttk.Entry(analysis_frame, textvariable=self.ma_window_var, width=8).grid(
                column=3, row=4, padx=4, pady=4, sticky="ew"
            )

            self.compute_button = ttk.Button(analysis_frame, text="Calculer le spread moyen", command=self.on_compute_spread)
            self.compute_button.grid(column=0, row=5, columnspan=6, padx=4, pady=8, sticky="ew")

            progress_frame = ttk.Frame(main_frame)
            progress_frame.grid(column=0, row=2, sticky="ew", padx=4, pady=4)
            progress_frame.columnconfigure(0, weight=1)

            self.progress = ttk.Progressbar(progress_frame, mode="determinate")
            self.progress.grid(column=0, row=0, sticky="ew", padx=4, pady=2)

            self.status_var = tk.StringVar(value="Prêt.")
            ttk.Label(progress_frame, textvariable=self.status_var).grid(column=0, row=1, sticky="w", padx=4)

            output_frame = ttk.LabelFrame(main_frame, text="Résultats")
            output_frame.grid(column=0, row=3, sticky="nsew", padx=4, pady=4)
            output_frame.columnconfigure(0, weight=1)
            output_frame.columnconfigure(1, weight=1)
            output_frame.rowconfigure(0, weight=1)

            text_container = ttk.Frame(output_frame)
            text_container.grid(column=0, row=0, sticky="nsew", padx=4, pady=4)
            text_container.columnconfigure(0, weight=1)
            text_container.rowconfigure(0, weight=1)

            self.output_text = tk.Text(text_container, wrap="word", height=20)
            self.output_text.grid(column=0, row=0, sticky="nsew")

            scrollbar = ttk.Scrollbar(text_container, orient="vertical", command=self.output_text.yview)
            scrollbar.grid(column=1, row=0, sticky="ns")
            self.output_text.configure(yscrollcommand=scrollbar.set, state="disabled")

            chart_container = ttk.LabelFrame(output_frame, text="Visualisations")
            chart_container.grid(column=1, row=0, sticky="nsew", padx=4, pady=4)
            chart_container.columnconfigure(0, weight=1)
            chart_container.rowconfigure(2, weight=1)

            ttk.Label(chart_container, text="Noeuds (sélection multiple)").grid(
                column=0, row=0, padx=4, pady=(4, 0), sticky="w"
            )
            self._node_listbox = tk.Listbox(
                chart_container, selectmode="extended", exportselection=False, height=6
            )
            self._node_listbox.grid(column=0, row=1, sticky="ew", padx=4, pady=2)
            self._node_listbox.bind("<<ListboxSelect>>", self._on_node_selection)

            notebook = ttk.Notebook(chart_container)
            notebook.grid(column=0, row=2, sticky="nsew", padx=4, pady=4)

            price_tab = ttk.Frame(notebook)
            price_tab.columnconfigure(0, weight=1)
            price_tab.rowconfigure(0, weight=1)

            spread_tab = ttk.Frame(notebook)
            spread_tab.columnconfigure(0, weight=1)
            spread_tab.rowconfigure(1, weight=1)

            notebook.add(price_tab, text="Prix horaires")
            notebook.add(spread_tab, text="Spreads (MA)")

            if MATPLOTLIB_AVAILABLE:
                self._chart_figure = Figure(figsize=(4, 3), dpi=100)
                self._chart_axes = self._chart_figure.add_subplot(111)
                self._chart_axes.set_title("Sélectionner des noeuds")
                self._chart_axes.set_ylabel(f"Prix ({self._chart_currency_label})")
                self._chart_axes.set_xlabel("Date")
                self._chart_canvas = FigureCanvasTkAgg(self._chart_figure, master=price_tab)
                widget = self._chart_canvas.get_tk_widget()
                widget.grid(column=0, row=0, sticky="nsew", padx=4, pady=4)
                widget.bind("<Button-1>", self._open_fullscreen_chart)

                ma_controls = ttk.Frame(spread_tab)
                ma_controls.grid(column=0, row=0, sticky="ew", padx=4, pady=4)
                ma_controls.columnconfigure(1, weight=1)
                ttk.Label(ma_controls, text="Fenêtre MA (jours)").grid(column=0, row=0, sticky="w", padx=2)
                ttk.Entry(ma_controls, textvariable=self.ma_window_var, width=6).grid(
                    column=1, row=0, sticky="w", padx=4
                )
                ttk.Button(ma_controls, text="Mettre à jour", command=self._on_update_ma).grid(
                    column=2, row=0, padx=4
                )

                self._spread_figure = Figure(figsize=(4, 3), dpi=100)
                self._spread_axes = self._spread_figure.add_subplot(111)
                self._spread_axes.set_title("Spreads quotidiens")
                self._spread_axes.set_ylabel("Spread (MXN)")
                self._spread_axes.set_xlabel("Jour (07h→07h)")
                self._spread_canvas = FigureCanvasTkAgg(self._spread_figure, master=spread_tab)
                spread_widget = self._spread_canvas.get_tk_widget()
                spread_widget.grid(column=0, row=1, sticky="nsew", padx=4, pady=4)
                spread_widget.bind("<Button-1>", self._open_fullscreen_spread_chart)
            else:
                tk.Label(
                    chart_container,
                    text="Installer matplotlib pour afficher les courbes.",
                    fg="red",
                    wraplength=220,
                ).grid(column=0, row=2, sticky="nsew", padx=4, pady=4)
                self._node_listbox.configure(state="disabled")

        def _run_async(self, func) -> None:
            thread = threading.Thread(target=func, daemon=True)
            thread.start()

        def _create_date_input(self, parent, variable: tk.StringVar):
            frame = ttk.Frame(parent)
            frame.columnconfigure(0, weight=1)
            entry = ttk.Entry(frame, textvariable=variable, width=12)
            entry.grid(column=0, row=0, sticky="ew")
            if TKCALENDAR_AVAILABLE and Calendar is not None:
                button = ttk.Button(
                    frame,
                    text="Calendrier",
                    command=lambda: self._open_calendar_dialog(variable),
                    width=10,
                )
            else:
                button = ttk.Button(frame, text="Calendrier", state="disabled", width=10)
            button.grid(column=1, row=0, padx=(4, 0))
            return frame

        def _open_calendar_dialog(self, variable: tk.StringVar) -> None:
            if not (TKCALENDAR_AVAILABLE and Calendar is not None):
                messagebox.showinfo(
                    "Calendrier indisponible",
                    "Installez le paquet tkcalendar pour activer la sélection visuelle.",
                )
                return

            top = tk.Toplevel(self.root)
            top.title("Choisir une date")
            top.transient(self.root)
            top.grab_set()
            top.resizable(False, False)
            top.columnconfigure(0, weight=1)

            base_date = datetime.today()
            current_value = variable.get().strip()
            if current_value:
                try:
                    base_date = datetime.fromisoformat(current_value)
                except ValueError:
                    pass

            calendar = Calendar(
                top,
                selectmode="day",
                year=base_date.year,
                month=base_date.month,
                day=base_date.day,
            )
            calendar.grid(column=0, row=0, padx=8, pady=8)

            def apply_selection() -> None:
                selection = calendar.selection_get()
                variable.set(selection.strftime("%Y-%m-%d"))
                top.destroy()

            button_frame = ttk.Frame(top)
            button_frame.grid(column=0, row=1, pady=(0, 8))
            ttk.Button(button_frame, text="Annuler", command=top.destroy).grid(column=0, row=0, padx=4)
            ttk.Button(button_frame, text="Valider", command=apply_selection).grid(column=1, row=0, padx=4)
            top.wait_window(top)

        def _start_task(self, maximum: Optional[int] = None, *, indeterminate: bool = False, status: str = "En cours…") -> None:
            self._task_running = True
            self._set_status(status)
            self._toggle_controls(False)
            self.progress.stop()
            if indeterminate or maximum is None:
                self.progress.configure(mode="indeterminate", maximum=100, value=0)
                self.progress.start(12)
            else:
                self.progress.configure(mode="determinate", maximum=maximum, value=0)

        def _configure_progress_bar(self, maximum: Optional[int]) -> None:
            self.progress.stop()
            max_value = int(maximum or 0)
            if max_value <= 0:
                self.progress.configure(mode="indeterminate", maximum=100, value=0)
                self.progress.start(12)
            else:
                self.progress.configure(mode="determinate", maximum=max_value, value=0)

        def _set_progress_value(self, current: int, total: int) -> None:
            total = max(int(total), 1)
            current = max(0, min(int(current), total))
            self.progress.configure(mode="determinate", maximum=total)
            self.progress["value"] = current

        def _finish_task(self) -> None:
            self.progress.stop()
            self.progress.configure(mode="determinate", maximum=100, value=0)
            self._toggle_controls(True)
            self._task_running = False

        def _toggle_controls(self, state: bool) -> None:
            widgets = [self.init_button, self.import_button, self.compute_button]
            for widget in widgets:
                widget.configure(state="normal" if state else "disabled")

        def _poll_queue(self) -> None:
            try:
                while True:
                    kind, payload = self.queue.get_nowait()
                    if kind == "log":
                        self._append_output(payload)
                    elif kind == "status":
                        self._set_status(payload)
                    elif kind == "progress":
                        self.progress.configure(mode="determinate")
                        self.progress["value"] = payload
                    elif kind == "progress_setup":
                        self._configure_progress_bar(payload)
                    elif kind == "progress_value":
                        if isinstance(payload, tuple) and len(payload) == 2:
                            self._set_progress_value(payload[0], payload[1])
                    elif kind == "clear_results":
                        self._clear_output()
                        self._reset_price_chart()
                        self._reset_spread_chart()
                    elif kind == "show_results":
                        self._clear_output()
                        for line in payload:
                            self._append_output(line)
                    elif kind == "chart_data":
                        self._update_chart_data(payload)
                    elif kind == "spread_chart_data":
                        self._update_spread_chart_data(payload)
                    elif kind == "error":
                        messagebox.showerror("Erreur", str(payload))
                    elif kind == "task_done":
                        self._finish_task()
            except queue.Empty:
                pass
            finally:
                self.root.after(100, self._poll_queue)

        def _append_output(self, message: str) -> None:
            self.output_text.configure(state="normal")
            self.output_text.insert("end", message + "\n")
            self.output_text.see("end")
            self.output_text.configure(state="disabled")

        def _clear_output(self) -> None:
            self.output_text.configure(state="normal")
            self.output_text.delete("1.0", "end")
            self.output_text.configure(state="disabled")

        def _set_status(self, message: str) -> None:
            self.status_var.set(message)

        def _update_chart_data(self, payload: Dict[str, object]) -> None:
            nodes = payload.get("nodes")
            series = payload.get("series")
            if not isinstance(nodes, list) or not isinstance(series, dict):
                return
            self._chart_currency_rate = float(payload.get("currency_rate", DEFAULT_GUI_CURRENCY_RATE))
            target_label = str(payload.get("target_label", self._chart_currency_label))
            self._chart_currency_label = target_label.upper()
            self._chart_series = series
            if self._node_listbox is not None:
                if MATPLOTLIB_AVAILABLE:
                    self._node_listbox.configure(state="normal")
                self._node_listbox.delete(0, "end")
                for node in nodes:
                    self._node_listbox.insert("end", node)
                if nodes and MATPLOTLIB_AVAILABLE:
                    self._node_listbox.select_set(0, "end")
                if not MATPLOTLIB_AVAILABLE:
                    self._node_listbox.configure(state="disabled")
            self._refresh_price_chart()
            self._refresh_spread_chart()

        def _update_spread_chart_data(self, payload: Dict[str, object]) -> None:
            series = payload.get("series")
            if not isinstance(series, dict):
                return
            ma_window = payload.get("ma_window")
            currency_rate = payload.get("currency_rate")
            target_label = payload.get("target_label")
            if isinstance(ma_window, int) and ma_window > 0:
                self._spread_ma_window = ma_window
                self.ma_window_var.set(str(ma_window))
            if isinstance(currency_rate, (int, float)):
                self._spread_chart_currency_rate = float(currency_rate)
            if isinstance(target_label, str) and target_label:
                self._spread_chart_currency_label = target_label.upper()
            self._spread_chart_series = series
            self._refresh_spread_chart()

        def _on_node_selection(self, _: object) -> None:  # pragma: no cover - GUI binding
            if MATPLOTLIB_AVAILABLE:
                self._refresh_price_chart()
                self._refresh_spread_chart()

        def _get_selected_nodes(self) -> List[str]:
            if not self._node_listbox:
                return []
            selection = self._node_listbox.curselection()
            if not selection:
                return []
            return [self._node_listbox.get(idx) for idx in selection]

        def _refresh_price_chart(self) -> None:
            if not (
                MATPLOTLIB_AVAILABLE
                and self._chart_axes is not None
                and self._chart_canvas is not None
            ):
                return
            nodes = self._get_selected_nodes()
            self._chart_axes.clear()
            if not nodes:
                self._chart_axes.set_title("Sélectionner des noeuds")
                self._chart_axes.set_ylabel(f"Prix ({self._chart_currency_label})")
                self._chart_axes.set_xlabel("Date")
                self._chart_canvas.draw_idle()
                return
            for node in nodes:
                points = self._chart_series.get(node, [])
                if not points:
                    continue
                timestamps: List[datetime] = []
                prices: List[float] = []
                for ts, price in points:
                    timestamps.append(ts)
                    prices.append(price * self._chart_currency_rate)
                if timestamps:
                    self._chart_axes.plot(timestamps, prices, label=node)
            self._chart_axes.set_ylabel(f"Prix ({self._chart_currency_label})")
            self._chart_axes.set_xlabel("Date/heure")
            if nodes:
                self._chart_axes.legend(loc="upper right")
            if self._chart_figure is not None:
                self._chart_figure.autofmt_xdate(rotation=30)
            self._chart_canvas.draw_idle()

        def _reset_price_chart(self) -> None:
            self._chart_series = {}
            if self._node_listbox is not None:
                self._node_listbox.delete(0, "end")
            if (
                MATPLOTLIB_AVAILABLE
                and self._chart_axes is not None
                and self._chart_canvas is not None
            ):
                self._chart_axes.clear()
                self._chart_axes.set_title("Aucune donnée")
                self._chart_axes.set_ylabel(f"Prix ({self._chart_currency_label})")
                self._chart_axes.set_xlabel("Date")
                self._chart_canvas.draw_idle()

        def _refresh_spread_chart(self) -> None:
            if not (
                MATPLOTLIB_AVAILABLE
                and self._spread_axes is not None
                and self._spread_canvas is not None
            ):
                return
            nodes = self._get_selected_nodes()
            self._spread_axes.clear()
            if not nodes:
                self._spread_axes.set_title("Spreads quotidiens")
                self._spread_axes.set_ylabel(f"Spread ({self._spread_chart_currency_label})")
                self._spread_axes.set_xlabel("Jour (07h→07h)")
                self._spread_canvas.draw_idle()
                return
            window = max(self._spread_ma_window, 1)
            for node in nodes:
                points = sorted(self._spread_chart_series.get(node, []), key=lambda item: item[0])
                if not points:
                    continue
                averaged = compute_moving_average(points, window)
                if not averaged:
                    averaged = points
                timestamps = [ts for ts, _ in averaged]
                values = [value * self._spread_chart_currency_rate for _, value in averaged]
                self._spread_axes.plot(timestamps, values, label=node)
            self._spread_axes.set_ylabel(f"Spread ({self._spread_chart_currency_label})")
            self._spread_axes.set_xlabel("Jour (07h→07h)")
            if nodes:
                self._spread_axes.legend(loc="upper right")
            if self._spread_figure is not None:
                self._spread_figure.autofmt_xdate(rotation=30)
            self._spread_canvas.draw_idle()

        def _reset_spread_chart(self) -> None:
            self._spread_chart_series = {}
            if (
                MATPLOTLIB_AVAILABLE
                and self._spread_axes is not None
                and self._spread_canvas is not None
            ):
                self._spread_axes.clear()
                self._spread_axes.set_title("Spreads quotidiens")
                self._spread_axes.set_ylabel(f"Spread ({self._spread_chart_currency_label})")
                self._spread_axes.set_xlabel("Jour (07h→07h)")
                self._spread_canvas.draw_idle()

        def _on_update_ma(self) -> None:
            try:
                window = int(self.ma_window_var.get())
            except ValueError:
                messagebox.showerror("MA invalide", "La moyenne mobile doit être un entier positif.")
                return
            if window <= 0:
                messagebox.showerror("MA invalide", "La moyenne mobile doit être un entier positif.")
                return
            self._spread_ma_window = window
            self.ma_window_var.set(str(window))
            self._refresh_spread_chart()

        def _open_fullscreen_chart(self, _: object) -> None:
            if not MATPLOTLIB_AVAILABLE:
                messagebox.showinfo(
                    "Courbe indisponible",
                    "Installez matplotlib pour consulter la courbe détaillée.",
                )
                return
            nodes = self._get_selected_nodes()
            if not nodes:
                messagebox.showinfo("Sélection requise", "Sélectionnez au moins un noeud dans la liste.")
                return
            if not self._chart_series:
                messagebox.showinfo("Données manquantes", "Aucune série à afficher pour le moment.")
                return

            top = tk.Toplevel(self.root)
            top.title("Courbe de prix détaillée")
            top.geometry("1024x720")
            top.columnconfigure(0, weight=1)
            top.rowconfigure(1, weight=1)

            figure = Figure(figsize=(9, 5), dpi=100)
            axes = figure.add_subplot(111)
            for node in nodes:
                points = self._chart_series.get(node, [])
                if not points:
                    continue
                timestamps: List[datetime] = []
                prices: List[float] = []
                for ts, price in points:
                    timestamps.append(ts)
                    prices.append(price * self._chart_currency_rate)
                if timestamps:
                    axes.plot(timestamps, prices, label=node)
            axes.set_title("Prix convertis")
            axes.set_ylabel(f"Prix ({self._chart_currency_label})")
            axes.set_xlabel("Date/heure")
            if nodes:
                axes.legend(loc="upper right")
            figure.autofmt_xdate(rotation=30)

            toolbar_frame = ttk.Frame(top)
            toolbar_frame.grid(column=0, row=0, sticky="ew")

            canvas = FigureCanvasTkAgg(figure, master=top)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.grid(column=0, row=1, sticky="nsew")

            if NavigationToolbar2Tk is not None:
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                toolbar.update()
            else:
                ttk.Label(
                    toolbar_frame,
                    text="Installez une version complète de matplotlib pour activer la barre d'outils.",
                    foreground="red",
                ).grid(column=0, row=0, sticky="w", padx=4, pady=2)

        def _open_fullscreen_spread_chart(self, _: object) -> None:
            if not MATPLOTLIB_AVAILABLE:
                messagebox.showinfo(
                    "Courbe indisponible",
                    "Installez matplotlib pour consulter la courbe de spreads.",
                )
                return
            nodes = self._get_selected_nodes()
            if not nodes:
                messagebox.showinfo("Sélection requise", "Sélectionnez au moins un noeud dans la liste.")
                return
            if not self._spread_chart_series:
                messagebox.showinfo("Données manquantes", "Aucune série de spreads à afficher.")
                return

            top = tk.Toplevel(self.root)
            top.title("Courbe des spreads (MA)")
            top.geometry("1024x720")
            top.columnconfigure(0, weight=1)
            top.rowconfigure(1, weight=1)

            figure = Figure(figsize=(9, 5), dpi=100)
            axes = figure.add_subplot(111)
            window = max(self._spread_ma_window, 1)
            for node in nodes:
                points = sorted(self._spread_chart_series.get(node, []), key=lambda item: item[0])
                if not points:
                    continue
                averaged = compute_moving_average(points, window)
                if not averaged:
                    averaged = points
                timestamps = [ts for ts, _ in averaged]
                values = [value * self._spread_chart_currency_rate for _, value in averaged]
                axes.plot(timestamps, values, label=node)
            axes.set_title(f"Moyenne mobile {window} jours")
            axes.set_ylabel(f"Spread ({self._spread_chart_currency_label})")
            axes.set_xlabel("Jour (07h→07h)")
            if nodes:
                axes.legend(loc="upper right")
            figure.autofmt_xdate(rotation=30)

            toolbar_frame = ttk.Frame(top)
            toolbar_frame.grid(column=0, row=0, sticky="ew")
            canvas = FigureCanvasTkAgg(figure, master=top)
            canvas.draw()
            canvas.get_tk_widget().grid(column=0, row=1, sticky="nsew")

            if NavigationToolbar2Tk is not None:
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                toolbar.update()
            else:
                ttk.Label(
                    toolbar_frame,
                    text="Installez une version complète de matplotlib pour activer la barre d'outils.",
                    foreground="red",
                ).grid(column=0, row=0, sticky="w", padx=4, pady=2)

else:

    class LocalAppGUI:
        def __init__(self) -> None:  # noqa: D401
            raise RuntimeError("Tkinter n'est pas disponible dans cet environnement.")

        def run(self) -> None:
            raise RuntimeError("Tkinter n'est pas disponible dans cet environnement.")


def command_gui(_: argparse.Namespace) -> None:
    if not TK_AVAILABLE:
        raise SystemExit(
            "Tkinter n'est pas disponible. "
            "Installez la prise en charge de Tk (ex: `brew install python-tk@3.11` ou utilisez Python officiel)."
        )
    app = LocalAppGUI()
    app.run()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local tools for ingesting and analysing CENACE price data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-db", help="Create the SQLite schema.")
    init_parser.set_defaults(func=command_init_db)

    ingest_parser = subparsers.add_parser("ingest", help="Import CSV files from a directory.")
    ingest_parser.add_argument(
        "--source",
        default="Data",
        help="Directory containing CSV files (default: Data).",
    )
    ingest_parser.set_defaults(func=command_ingest)

    runs_parser = subparsers.add_parser("runs", help="List imported CSV runs.")
    runs_parser.set_defaults(func=command_list_runs)

    spreads_parser = subparsers.add_parser("spreads", help="Compute price spreads over a time window.")
    spreads_parser.add_argument("--window", type=int, default=2, help="Window size in hours (>=2).")
    spreads_parser.add_argument("--top", type=int, default=10, help="Number of results to display.")
    spreads_parser.add_argument("--year", type=int, help="Filter by year (YYYY).")
    spreads_parser.add_argument(
        "--season",
        choices=sorted(SEASON_TO_MONTHS.keys()),
        help="Filter by season.",
    )
    spreads_parser.add_argument("--start", help="Start datetime (YYYY-MM-DD or YYYY-MM-DDTHH:MM).")
    spreads_parser.add_argument("--end", help="End datetime (inclusive).")
    spreads_parser.add_argument("--node", help="Filter by a specific node code.")
    spreads_parser.add_argument(
        "--voltage-min",
        type=float,
        help="Minimum node voltage (kV).",
    )
    spreads_parser.add_argument(
        "--voltage-max",
        type=float,
        help="Maximum node voltage (kV).",
    )
    spreads_parser.add_argument(
        "--currency-rate",
        type=float,
        default=DEFAULT_CURRENCY_RATE,
        help="Multiplier to convert MXN spreads into a target currency.",
    )
    spreads_parser.add_argument(
        "--currency-target",
        default="USD",
        help="Name of the target currency for converted values.",
    )
    spreads_parser.add_argument(
        "--spread-source",
        choices=SPREAD_SOURCE_CHOICES,
        default="auto",
        help="Use cached daily spreads, recompute for the filter, or recompute only if missing.",
    )
    spreads_parser.set_defaults(func=command_spreads)

    gui_parser = subparsers.add_parser("gui", help="Launch the desktop interface.")
    gui_parser.set_defaults(func=command_gui)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except RuntimeError as exc:
        print(f"Erreur: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
