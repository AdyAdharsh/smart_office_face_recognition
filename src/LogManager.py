# src/LogManager.py

import sqlite3
import datetime
import os

# Define the path to your database file
DATABASE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'access_log.db')

class LogManager:
    def __init__(self, db_file=DATABASE_FILE):
        self.db_file = db_file
        self._initialize_db()

    def _initialize_db(self):
        """Creates the database file and the access_log table if they don't exist."""
        try:
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
            # print(f"LOGGED: {timestamp}, {status}, {user_id}") # Optional debug print
            
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