"""Fix: add id column to existing bridge_dataset table."""
import psycopg2

conn = psycopg2.connect(
    dbname="timeseries", host="localhost",
    user="postgres", password="postgres", port=5432
)
conn.autocommit = True
cur = conn.cursor()

# Check if id column exists
cur.execute("""
    SELECT column_name FROM information_schema.columns
    WHERE table_name = 'bridge_dataset' AND column_name = 'id'
""")
if cur.fetchone():
    print("[OK] id column already exists")
else:
    print("[...] Adding id column...")
    cur.execute("ALTER TABLE bridge_dataset ADD COLUMN id SERIAL")
    print("[OK] id column added")

    # Create index on id
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bridge_id ON bridge_dataset (id)")
    print("[OK] Index on id created")

# Create index for ML engine
cur.execute("CREATE INDEX IF NOT EXISTS idx_bridge_is_anomaly_null ON bridge_dataset (id) WHERE is_anomaly IS NULL")
print("[OK] Partial index for unscored rows created")

# Create index on timestamp
cur.execute("CREATE INDEX IF NOT EXISTS idx_bridge_timestamp ON bridge_dataset (timestamp)")
print("[OK] Timestamp index created")

# Verify
cur.execute("SELECT MIN(id), MAX(id), COUNT(*) FROM bridge_dataset")
mn, mx, cnt = cur.fetchone()
print(f"\nid range: {mn} - {mx}, total rows: {cnt}")

cur.close()
conn.close()
print("\n[DONE] Table is ready!")
