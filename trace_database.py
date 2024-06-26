import hashlib
import sqlite3
import json

class TraceDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self._create_tables()

    def _create_tables(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY,
                trace_hash TEXT UNIQUE,
                trace TEXT
            )
            ''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS probs_scores (
                id INTEGER PRIMARY KEY,
                score REAL,
                token TEXT,
                trace_id INTEGER,
                FOREIGN KEY (trace_id) REFERENCES traces (id)
            )
            ''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_scores (
                id INTEGER PRIMARY KEY,
                score REAL,
                token TEXT,
                trace_id INTEGER,
                FOREIGN KEY (trace_id) REFERENCES traces (id)
            )
            ''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS brier_scores (
                id INTEGER PRIMARY KEY,
                score REAL,
                token TEXT,
                trace_id INTEGER,
                FOREIGN KEY (trace_id) REFERENCES traces (id)
            )
            ''')
            conn.commit()
        finally:
            if conn:
                conn.close()

    def hash_trace(self, trace):
        trace_str = json.dumps(trace)
        return hashlib.sha256(trace_str.encode('utf-8')).hexdigest()

    def get_trace(self, trace):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            trace_hash = self.hash_trace(trace)
            cursor.execute("SELECT id FROM traces WHERE trace_hash = ?", (trace_hash,))
            result = cursor.fetchone()
            if result:
                trace_id = result[0]
                probs_anomalies = self._get_scores(cursor, "probs_scores", trace_id)
                error_anomalies = self._get_scores(cursor, "error_scores", trace_id)
                brier_scores = self._get_scores(cursor, "brier_scores", trace_id)
                return trace_id, (probs_anomalies, error_anomalies, brier_scores)
            else:
                return None, None
        finally:
            if conn:
                conn.close()

    def _get_scores(self, cursor, table, trace_id):
        cursor.execute(f"SELECT score, token FROM {table} WHERE trace_id = ?", (trace_id,))
        rows = cursor.fetchall()
        if table == "brier_scores":
            scores_dict = {}
            for score, token in rows:
                if score not in scores_dict:
                    scores_dict[score] = []
                scores_dict[score].append(token)
            return [(score, tokens) for score, tokens in scores_dict.items()]
        else:
            return [(row[0], row[1]) for row in rows]

    def save_trace(self, trace, probs_anomalies, error_anomalies, brier_scores):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            trace_hash = self.hash_trace(trace)
            trace_str = json.dumps(trace)
            cursor.execute("INSERT OR IGNORE INTO traces (trace_hash, trace) VALUES (?, ?)", (trace_hash, trace_str))
            trace_id = cursor.execute("SELECT id FROM traces WHERE trace_hash = ?", (trace_hash,)).fetchone()[0]
            self._save_scores(cursor, "probs_scores", trace_id, probs_anomalies)
            self._save_scores(cursor, "error_scores", trace_id, error_anomalies)
            self._save_scores(cursor, "brier_scores", trace_id, brier_scores)
            conn.commit()
        finally:
            if conn:
                conn.close()

    def _save_scores(self, cursor, table, trace_id, scores):
        for score, sample in scores:
            if isinstance(sample, list):
                for token in sample:
                    cursor.execute(f"INSERT INTO {table} (score, token, trace_id) VALUES (?, ?, ?)", (score, token, trace_id))
            else:
                cursor.execute(f"INSERT INTO {table} (score, token, trace_id) VALUES (?, ?, ?)", (score, sample, trace_id))
    