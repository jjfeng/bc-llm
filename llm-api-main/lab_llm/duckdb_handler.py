import duckdb

class DuckDBHandler:
    def __init__(self, db_path: str, read_only: bool = False):
        """Initialize the DuckDB handler.

        Args:
            db_path (str): Path to the DuckDB database file
            read_only (bool, optional): Whether to open the connection in read-only mode. Defaults to True.
        """
        self.db_path = db_path
        self.read_only = read_only
        self.conn = duckdb.connect(self.db_path, read_only=self.read_only)
    
    def close_connection(self):
        self.conn.close()

    def get_connection(self):
        return self.conn

