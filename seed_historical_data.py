"""
seed_historical_data.py
Loads the FIRST 70% of bridge_dataset.csv into PostgreSQL as training data.
The remaining 30% is reserved for real-time replay by bridge_realtime_engine.py.
"""

import pandas as pd
from sqlalchemy import create_engine
import sys
import os
import time


# Database: reads DATABASE_URL env var (Supabase), falls back to localhost
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/timeseries")
CSV_PATH = "bridge_dataset.csv"
TABLE_NAME = "bridge_dataset"
CHUNK_SIZE = 5000
TRAIN_RATIO = 0.70  # 70% for training


def main():
    print("=" * 60)
    print("  Bridge Anomaly Detection — Seed Training Data (70%)")
    print("=" * 60)

    # ── Read CSV ──────────────────────────────────────────
    print(f"\n[…] Reading {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"[✗] File not found: {CSV_PATH}")
        sys.exit(1)

    total = len(df)
    split_idx = int(total * TRAIN_RATIO)
    train_df = df.iloc[:split_idx].copy()
    remaining = total - split_idx

    print(f"[✓] Loaded {total:,} rows × {len(df.columns)} columns")
    print(f"[✓] Split: {split_idx:,} training (70%) | {remaining:,} real-time (30%)")

    # ── Rename columns ────────────────────────────────────
    train_df.columns = [c.lower() for c in train_df.columns]
    print(f"[✓] Renamed columns to snake_case")

    # ── Parse timestamp ───────────────────────────────────
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    print(f"[✓] Timestamps: {train_df['timestamp'].min()} → {train_df['timestamp'].max()}")

    # ── Check for existing data ───────────────────────────
    engine = create_engine(DB_URL)
    try:
        existing = pd.read_sql("SELECT COUNT(*) as cnt FROM bridge_dataset", engine)
        row_count = existing["cnt"].iloc[0]
        if row_count > 0:
            print(f"\n[!] Table already has {row_count:,} rows.")
            response = input("    Clear and re-seed? (y/N): ").strip().lower()
            if response != "y":
                print("[–] Aborted.")
                sys.exit(0)
            with engine.connect() as conn:
                conn.execute(pd.io.sql.text("TRUNCATE bridge_dataset RESTART IDENTITY"))
                conn.commit()
            print("[✓] Cleared existing data")
    except Exception:
        pass

    # ── Insert training data ──────────────────────────────
    print(f"\n[…] Inserting {split_idx:,} training rows...")
    start_time = time.time()

    for i in range(0, split_idx, CHUNK_SIZE):
        chunk = train_df.iloc[i : i + CHUNK_SIZE]
        chunk.to_sql(TABLE_NAME, engine, if_exists="append", index=False, method="multi")
        inserted = min(i + CHUNK_SIZE, split_idx)
        pct = (inserted / split_idx) * 100
        print(f"    [{inserted:>6,} / {split_idx:,}] {pct:5.1f}%")

    elapsed = time.time() - start_time
    print(f"\n[✓] Seeded {split_idx:,} training rows in {elapsed:.1f}s")

    # ── Verify ────────────────────────────────────────────
    result = pd.read_sql("SELECT COUNT(*) as cnt FROM bridge_dataset", engine)
    print(f"[✓] Verified: {result['cnt'].iloc[0]:,} rows in database")

    null_count = pd.read_sql(
        "SELECT COUNT(*) as cnt FROM bridge_dataset WHERE is_anomaly IS NULL", engine
    )
    print(f"[✓] Rows awaiting ML scoring: {null_count['cnt'].iloc[0]:,}")

    print(f"\n{'=' * 60}")
    print(f"  Training data:  {split_idx:,} rows (70%)")
    print(f"  Real-time data: {remaining:,} rows (30%) — for bridge_realtime_engine.py")
    print(f"{'=' * 60}")
    print(f"\n  Next: python3 bridge_ml_engine.py   (score training data)")
    print(f"  Then: python3 bridge_realtime_engine.py   (replay remaining 30%)")


if __name__ == "__main__":
    main()
