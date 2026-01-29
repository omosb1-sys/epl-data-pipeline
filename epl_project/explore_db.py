import duckdb

def explore_db():
    conn = duckdb.connect('data/epl_analytics.duckdb', read_only=True)
    tables = conn.execute("SHOW TABLES").fetchall()
    print("Tables:", tables)
    for table_tuple in tables:
        table_name = table_tuple[0]
        print(f"\nStructure of {table_name}:")
        print(conn.execute(f"DESCRIBE {table_name}").df())
        print(f"\nFirst 5 rows of {table_name}:")
        print(conn.execute(f"SELECT * FROM {table_name} LIMIT 5").df())
    conn.close()

if __name__ == "__main__":
    explore_db()
