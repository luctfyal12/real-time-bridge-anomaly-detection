"""
bridge_ml_engine.py
Real-time anomaly detection engine using IsolationForest.

Workflow:
1. Train on historical data from PostgreSQL (seeded from CSV)
2. Continuously fetch unscored rows (is_anomaly IS NULL)
3. Score them and update the database
"""

import psycopg2
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import time
import signal
import sys
from datetime import datetime


DB_CONFIG = {
    "dbname": "timeseries",
    "host": "localhost",
    "user": "postgres",
    "password": "postgres",
    "port": 5432,
}

# Features used for anomaly detection (multivariate)
FEATURE_COLUMNS = [
    "strain_microstrain",
    "deflection_mm",
    "vibration_ms2",
    "tilt_deg",
    "displacement_mm",
    "cable_member_tension_kn",
]

# Model parameters
CONTAMINATION = 0.05       # Expected fraction of anomalies
N_ESTIMATORS = 200         # Number of trees
RANDOM_STATE = 42
SCORING_INTERVAL = 2.0     # Seconds between scoring cycles
BATCH_SIZE = 100           # Rows to score per cycle

# Shutdown flag
_running = True


def signal_handler(sig, frame):
    global _running
    print("\n[!] Shutdown signal received. Finishing current cycle...")
    _running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_connection():
    """Create a new database connection."""
    return psycopg2.connect(**DB_CONFIG)


def train_model(conn) -> tuple:
    """
    Train IsolationForest on historical data.
    Returns (model, scaler, imputer).
    """
    print("[…] Loading training data from PostgreSQL...")

    # Load all historical data for training
    query = f"""
        SELECT {', '.join(FEATURE_COLUMNS)}
        FROM bridge_dataset
        ORDER BY id
    """
    df = pd.read_sql(query, conn)
    print(f"[✓] Loaded {len(df):,} rows for training")

    if len(df) == 0:
        print("[✗] No data found! Run seed_historical_data.py first.")
        sys.exit(1)

    # ── Imputation (handle NaN/NULL values) ───────────────────
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(df[FEATURE_COLUMNS])
    print(f"[✓] Imputed missing values (median strategy)")

    # ── Standardization ───────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    print(f"[✓] Fitted StandardScaler")

    # ── Train IsolationForest ─────────────────────────────────
    print(f"[…] Training IsolationForest (n_estimators={N_ESTIMATORS}, contamination={CONTAMINATION})...")
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_scaled)
    print(f"[✓] Model trained successfully")

    # Show training statistics
    train_preds = model.predict(X_scaled)
    train_scores = model.decision_function(X_scaled)
    n_anomalies = (train_preds == -1).sum()
    print(f"    Training anomalies detected: {n_anomalies:,} / {len(df):,} "
          f"({n_anomalies / len(df) * 100:.1f}%)")
    print(f"    Score range: [{train_scores.min():.4f}, {train_scores.max():.4f}]")

    return model, scaler, imputer


def score_batch(conn, model, scaler, imputer) -> int:
    """
    Fetch unscored rows, predict anomalies, and update the database.
    Returns the number of rows scored.
    """
    cur = conn.cursor()

    # Fetch unscored rows
    feature_cols_sql = ", ".join(FEATURE_COLUMNS)
    cur.execute(f"""
        SELECT id, {feature_cols_sql}
        FROM bridge_dataset
        WHERE is_anomaly IS NULL
        ORDER BY id
        LIMIT {BATCH_SIZE}
    """)
    rows = cur.fetchall()

    if not rows:
        cur.close()
        return 0

    # Parse into arrays
    ids = [row[0] for row in rows]
    X_raw = np.array([[row[i + 1] for i in range(len(FEATURE_COLUMNS))] for row in rows],
                     dtype=np.float64)

    # Handle NaN values
    X_imputed = imputer.transform(X_raw)

    # Scale
    X_scaled = scaler.transform(X_imputed)

    # Predict
    predictions = model.predict(X_scaled)        # 1 = normal, -1 = anomaly
    scores = model.decision_function(X_scaled)    # Lower = more anomalous

    # Update database
    update_sql = """
        UPDATE bridge_dataset
        SET is_anomaly = %s, anomaly_score = %s
        WHERE id = %s
    """
    updates = [
        (bool(pred == -1), float(score), row_id)
        for pred, score, row_id in zip(predictions, scores, ids)
    ]
    cur.executemany(update_sql, updates)
    conn.commit()
    cur.close()

    return len(updates)


def main():
    global _running

    print("=" * 60)
    print("  Bridge Anomaly Detection — ML Scoring Engine")
    print("=" * 60)

    # Connect to database
    try:
        conn = get_connection()
        print(f"[✓] Connected to PostgreSQL")
    except psycopg2.OperationalError as e:
        print(f"[✗] Connection failed: {e}")
        sys.exit(1)

    # Train model
    model, scaler, imputer = train_model(conn)

    # ── Scoring loop ──────────────────────────────────────────
    print(f"\n[▶] Starting scoring loop (every {SCORING_INTERVAL}s, batch={BATCH_SIZE})")
    print(f"    Press Ctrl+C to stop\n")
    print(f"{'Cycle':>6}  {'Scored':>7}  {'Anomalies':>10}  {'Total':>8}  {'Timestamp':<20}")
    print("-" * 65)

    cycle = 0
    total_scored = 0
    total_anomalies = 0

    while _running:
        cycle += 1

        try:
            scored = score_batch(conn, model, scaler, imputer)

            if scored > 0:
                # Count anomalies in this batch
                cur = conn.cursor()
                cur.execute("""
                    SELECT COUNT(*) FROM bridge_dataset
                    WHERE is_anomaly = TRUE
                """)
                total_anomalies = cur.fetchone()[0]
                cur.close()

                total_scored += scored
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"{cycle:>6}  {scored:>7}  {total_anomalies:>10}  "
                    f"{total_scored:>8}  {now}"
                )
            else:
                # No unscored rows — wait quietly
                if cycle % 10 == 0:  # Print status every 10 idle cycles
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                          f"Waiting for new data... (total scored: {total_scored:,})")

        except Exception as e:
            print(f"[✗] Scoring error: {e}")
            # Reconnect on connection errors
            try:
                conn.close()
            except Exception:
                pass
            try:
                conn = get_connection()
                print("[✓] Reconnected to database")
            except Exception:
                print("[✗] Reconnection failed, retrying...")

        time.sleep(SCORING_INTERVAL)

    # Shutdown
    conn.close()
    print(f"\n{'=' * 60}")
    print(f"  ML Engine stopped.")
    print(f"  Total scored: {total_scored:,}  |  Anomalies found: {total_anomalies:,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
