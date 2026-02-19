"""
setup_database.py
Creates the bridge_dataset table in PostgreSQL (timeseries database).
All column names use snake_case to match the implementation plan.
"""

import psycopg2
import sys


DB_CONFIG = {
    "dbname": "timeseries",
    "host": "localhost",
    "user": "postgres",
    "password": "postgres",
    "port": 5432,
}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bridge_dataset (
    id                                  SERIAL PRIMARY KEY,
    timestamp                           TIMESTAMP NOT NULL,

    -- Structural Sensors
    strain_microstrain                  DOUBLE PRECISION,
    deflection_mm                       DOUBLE PRECISION,
    vibration_ms2                       DOUBLE PRECISION,
    tilt_deg                            DOUBLE PRECISION,
    displacement_mm                     DOUBLE PRECISION,
    crack_propagation_mm                DOUBLE PRECISION,
    corrosion_level_percent             DOUBLE PRECISION,
    cable_member_tension_kn             DOUBLE PRECISION,
    bearing_joint_forces_kn             DOUBLE PRECISION,
    fatigue_accumulation_au             DOUBLE PRECISION,
    modal_frequency_hz                  DOUBLE PRECISION,

    -- Environmental Factors
    temperature_c                       DOUBLE PRECISION,
    humidity_percent                    DOUBLE PRECISION,
    wind_speed_ms                       DOUBLE PRECISION,
    wind_direction_deg                  DOUBLE PRECISION,
    precipitation_mmh                   DOUBLE PRECISION,
    water_level_m                       DOUBLE PRECISION,
    seismic_activity_ms2                DOUBLE PRECISION,
    solar_radiation_wm2                 DOUBLE PRECISION,
    air_quality_index_aqi               DOUBLE PRECISION,
    soil_settlement_mm                  DOUBLE PRECISION,

    -- Load & Traffic
    vehicle_load_tons                   DOUBLE PRECISION,
    traffic_volume_vph                  DOUBLE PRECISION,
    pedestrian_load_pph                 DOUBLE PRECISION,
    impact_events_g                     DOUBLE PRECISION,
    dynamic_load_distribution_percent   DOUBLE PRECISION,
    axle_counts_pmin                    DOUBLE PRECISION,

    -- Health & Analysis
    structural_health_index_shi         DOUBLE PRECISION,
    anomaly_detection_score             DOUBLE PRECISION,
    energy_dissipation_au               DOUBLE PRECISION,
    acoustic_emissions_levels           DOUBLE PRECISION,
    visual_analysis_defect_score        DOUBLE PRECISION,
    electrical_resistance_ohms          DOUBLE PRECISION,
    bridge_mood_meter                   VARCHAR(20),
    localized_strain_hotspot            DOUBLE PRECISION,
    vibration_anomaly_location          VARCHAR(30),

    -- Predictions
    shi_predicted_24h_ahead             DOUBLE PRECISION,
    shi_predicted_7d_ahead              DOUBLE PRECISION,
    shi_predicted_30d_ahead             DOUBLE PRECISION,
    probability_of_failure_pof          DOUBLE PRECISION,

    -- Alerts & Events
    maintenance_alert                   DOUBLE PRECISION,
    flood_event_flag                    DOUBLE PRECISION,
    simulated_water_flow_m3s            DOUBLE PRECISION,
    soil_saturation_percent             DOUBLE PRECISION,
    landslide_ground_movement           DOUBLE PRECISION,
    simulated_slope_displacement_mm     DOUBLE PRECISION,
    high_winds_storms                   DOUBLE PRECISION,
    simulated_wind_load_pressure_kpa    DOUBLE PRECISION,
    abnormal_traffic_load_surges        DOUBLE PRECISION,
    simulated_localized_stress_index    DOUBLE PRECISION,

    -- Sustainability
    energy_harvesting_potential_w       DOUBLE PRECISION,
    estimated_repair_cost_usd_incremental   DOUBLE PRECISION,
    carbon_footprint_tco2e_incremental  DOUBLE PRECISION,

    -- ML Engine Output (filled by bridge_ml_engine.py)
    is_anomaly                          BOOLEAN DEFAULT NULL,
    anomaly_score                       DOUBLE PRECISION DEFAULT NULL
);

-- Index for the ML engine to quickly find unscored rows
CREATE INDEX IF NOT EXISTS idx_bridge_is_anomaly_null
    ON bridge_dataset (id) WHERE is_anomaly IS NULL;

-- Index for Tableau time-series queries
CREATE INDEX IF NOT EXISTS idx_bridge_timestamp
    ON bridge_dataset (timestamp);
"""


def main():
    print("=" * 60)
    print("  Bridge Anomaly Detection — Database Setup")
    print("=" * 60)

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        print(f"[✓] Connected to PostgreSQL ({DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']})")
    except psycopg2.OperationalError as e:
        print(f"[✗] Failed to connect to PostgreSQL: {e}")
        print("\n  Make sure PostgreSQL is running and the 'timeseries' database exists.")
        print("  To create it:  createdb -U postgres timeseries")
        sys.exit(1)

    cur = conn.cursor()

    try:
        cur.execute(CREATE_TABLE_SQL)
        print("[✓] Table 'bridge_dataset' created (or already exists)")

        # Verify table structure
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'bridge_dataset'
            ORDER BY ordinal_position;
        """)
        columns = cur.fetchall()
        print(f"[✓] Table has {len(columns)} columns:")
        for col_name, col_type in columns:
            print(f"    • {col_name:<45s} {col_type}")

    except Exception as e:
        print(f"[✗] Error creating table: {e}")
        sys.exit(1)
    finally:
        cur.close()
        conn.close()

    print("\n[✓] Database setup complete!")
    print("    Next step: python3 seed_historical_data.py")


if __name__ == "__main__":
    main()
