import sqlite3
import datetime

class MetricsLogger:
    def __init__(self, db_name="training_metrics.db"):
        """
        Initialize the connection to the SQLite database and create the metrics table.
        """
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()
    
    def create_table(self):
        """
        Create a table for storing training metrics if it doesn't already exist.
        """
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                episode INTEGER,
                iteration INTEGER,
                accuracy REAL,
                training_loss REAL,
                demographic_parity REAL,
                equality_of_odds REAL,
                conditional_dp REAL,
                surrogate_loss REAL
            )
        ''')
        self.conn.commit()
    
    def log_metrics(self, episode, iteration, accuracy, training_loss, demographic_parity, equality_of_odds, conditional_dp, surrogate_loss):
        """
        Insert a new record into the metrics table.
        """
        timestamp = datetime.datetime.now().isoformat()

        print("DP:", demographic_parity)
        self.cursor.execute('''
            INSERT INTO metrics (timestamp, episode, iteration, accuracy, training_loss, demographic_parity, equality_of_odds, conditional_dp, surrogate_loss)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, episode, iteration, accuracy, training_loss, demographic_parity, equality_of_odds, conditional_dp, surrogate_loss))
        self.conn.commit()
    
    def close(self):
        """
        Close the database connection.
        """
        self.conn.close()
