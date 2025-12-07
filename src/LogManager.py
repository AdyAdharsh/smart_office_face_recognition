# src/LogManager.py

import sqlite3
import datetime
import os

# --- CHANGE: Remove the file path definition ---
# DATABASE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'access_log.db')

class LogManager:
    # --- CHANGE: Default db_file to ':memory:' if none is provided ---
    def __init__(self, db_file=':memory:'): 
        """
        Initializes the LogManager. By default, uses an in-memory database 
        for cloud deployment stability. Logs will be lost on app restart.
        """
        self.db_file = db_file
        self._initialize_db()

    def _initialize_db(self):
        """Creates the database (in-memory or file) and the access_log table."""
        try:
            # The database connection will open an in-memory DB if self.db_file == ':memory:'
            conn = sqlite3.connect(self.db_file) 
            cursor = conn.cursor()
            
            # Table Schema: Timestamp, User, Status, Confidence
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    status TEXT NOT NULL,
                    confidence REAL
                )
            """)
            conn.commit()
            conn.close()
            # Inform the user where the database is located
            if self.db_file == ':memory:':
                print("INFO: Database initialized in-memory.")
            else:
                print(f"INFO: Database initialized at {self.db_file}")

        except sqlite3.Error as e:
            print(f"ERROR: SQLite initialization failed: {e}")

    def log_access_event(self, user_id, status, confidence=None):
        """Inserts a new access event into the database."""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Insert the new record
            cursor.execute(
                """INSERT INTO access_log (timestamp, user_id, status, confidence) 
                   VALUES (?, ?, ?, ?)""",
                (timestamp, user_id, status, confidence)
            )
            conn.commit()
            conn.close()
            
        except sqlite3.Error as e:
            print(f"ERROR: Failed to write to database: {e}")

    def get_recent_logs(self, limit=10):
        """Fetches the N most recent logs."""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT timestamp, status, user_id, confidence FROM access_log ORDER BY id DESC LIMIT ?", 
                (limit,)
            )
            logs = cursor.fetchall()
            conn.close()
            return logs
        except sqlite3.Error as e:
            print(f"ERROR: Failed to read from database: {e}")
            return []