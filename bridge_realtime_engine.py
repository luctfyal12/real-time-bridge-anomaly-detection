"""
bridge_realtime_engine.py
Replays the last 30% of bridge_dataset.csv as real-time data.

The first 70% is used for training (seeded by seed_historical_data.py).
This script reads rows 70%–100% from the CSV and inserts them into
PostgreSQL one row per second, simulating a live sensor feed.
"""

import psycopg2
import numpy as np
import pandas as pd
import argparse
import signal
import sys
import os
import time
from datetime import datetime
from urllib.parse import urlparse


# Database: reads DATABASE_URL env var (Supabase), falls back to localhost
_db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/timeseries")
_parsed = urlparse(_db_url)
DB_CONFIG = {
    "dbname": _parsed.path.lstrip("/") or "timeseries",
    "host": _parsed.hostname or "localhost",
    "user": _parsed.username or "postgres",
    "password": _parsed.password or "postgres",
    "port": _parsed.port or 5432,
}
if _parsed.hostname and _parsed.hostname != "localhost":
    DB_CONFIG["sslmode"] = "require"

CSV_PATH = "bridge_dataset.csv"
TRAIN_RATIO = 0.70         # Must match seed_historical_data.py
STREAM_INTERVAL = 1.0      # seconds between inserts

# Shutdown flag
_running = True


def signal_handler(sig, frame):
    global _running
    print("\n[!] Shutdown signal received. Finishing current cycle...")
    _running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def load_realtime_data(csv_path: str, train_ratio: float) -> pd.DataFrame:
    """Load the last 30% of the CSV for real-time replay."""
    print(f"[…] Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]

    total = len(df)
    split_idx = int(total * train_ratio)
    realtime_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"[✓] Total CSV rows: {total:,}")
    print(f"[✓] Training rows (already seeded): {split_idx:,}")
    print(f"[✓] Real-time rows to replay: {len(realtime_df):,}")

    return realtime_df


def insert_row(conn, row: pd.Series):
    """Insert a single CSV row into PostgreSQL with current timestamp."""
    reading = {}

    # Copy all numeric/string columns from the CSV row
    for col in row.index:
        if col == "timestamp":
            # Use current time instead of historical timestamp
            reading["timestamp"] = datetime.now()
        else:
            val = row[col]
            # Handle NaN
            if pd.isna(val):
                reading[col] = None
            elif isinstance(val, (np.floating, np.integer)):
                reading[col] = val.item()  # Cast numpy → Python native
            else:
                reading[col] = val

    # Ensure is_anomaly and anomaly_score are NULL for ML engine to fill
    # (remove them if they came from CSV, ML engine will set them)
    reading.pop("is_anomaly", None)
    reading.pop("anomaly_score", None)
    # Also remove anomaly_detection_score if it was in the CSV and re-add as NULL
    # Actually keep it — it's the CSV's original score, separate from our ML score

    columns = list(reading.keys())
    values = [reading[c] for c in columns]
    placeholders = ", ".join(["%s"] * len(columns))
    col_names = ", ".join(columns)

    sql = f"INSERT INTO bridge_dataset ({col_names}) VALUES ({placeholders})"

    cur = conn.cursor()
    cur.execute(sql, values)
    conn.commit()
    cur.close()


def main():
    global _running

    parser = argparse.ArgumentParser(description="Replay 30% of CSV as real-time data")
    parser.add_argument("--count", type=int, default=0,
                        help="Max rows to replay (0 = all remaining, default: 0)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Seconds between inserts (default: 1.0)")
    args = parser.parse_args()
    max_count = args.count
    interval = args.speed

    print("=" * 60)
    print("  Bridge Anomaly Detection — Real-Time Data Replay")
    print("=" * 60)

    # Load the 30% real-time portion
    realtime_df = load_realtime_data(CSV_PATH, TRAIN_RATIO)
    total_available = len(realtime_df)

    if max_count > 0:
        total_to_send = min(max_count, total_available)
    else:
        total_to_send = total_available

    # Connect to PostgreSQL
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print(f"[✓] Connected to PostgreSQL")
    except psycopg2.OperationalError as e:
        print(f"[✗] Connection failed: {e}")
        sys.exit(1)

    mode = f"{total_to_send:,} rows" if max_count > 0 else f"all {total_available:,} rows"
    print(f"\n[▶] Replaying {mode} at {interval}s intervals")
    print(f"    Press Ctrl+C to stop\n")
    print(f"{'#':>6}  {'Timestamp':<22}  {'Strain':>10}  {'Vib':>8}  {'SHI':>6}  {'Progress':<12}")
    print("-" * 78)

    count = 0

    for idx in range(total_to_send):
        if not _running:
            break

        row = realtime_df.iloc[idx]

        try:
            insert_row(conn, row)
            count += 1

            ts = datetime.now()
            strain = row.get("strain_microstrain", 0)
            vib = row.get("vibration_ms2", 0)
            shi = row.get("structural_health_index_shi", 0)
            strain = strain if pd.notna(strain) else 0
            vib = vib if pd.notna(vib) else 0
            shi = shi if pd.notna(shi) else 0
            pct = (count / total_to_send) * 100

            print(
                f"{count:>6}  {ts.strftime('%Y-%m-%d %H:%M:%S'):<22}  "
                f"{strain:>10.2f}  {vib:>8.4f}  {shi:>6.4f}  "
                f"{pct:>5.1f}%"
            )

        except Exception as e:
            print(f"[✗] Insert error at row {idx}: {e}")
            conn.rollback()

        if idx < total_to_send - 1:  # Don't sleep after last row
            time.sleep(interval)

    # Shutdown
    conn.close()
    print(f"\n{'=' * 60}")
    print(f"  Replay complete.")
    print(f"  Rows inserted: {count:,} / {total_to_send:,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
