import sqlite3
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "tasks_db.sqlite")

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            type TEXT,
            status INTEGER,
            input_data TEXT,
            result_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_task(t_id, t_type, input_data):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO tasks (id, type, status, input_data) VALUES (?, ?, ?, ?)",
              (t_id, t_type, 0, json.dumps(input_data)))
    conn.commit()
    conn.close()

def update_task_status(t_id, status, result=None):
    conn = sqlite3.connect(DB_NAME, timeout=20)
    c = conn.cursor()
    if result:
        c.execute("UPDATE tasks SET status = ?, result_data = ? WHERE id = ?",
                  (status, json.dumps(result), t_id))
    else:
        c.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, t_id))
    conn.commit()
    conn.close()

# --- НОВА ФУНКЦІЯ: ВИДАЛЕННЯ ---
def delete_task(t_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM tasks WHERE id = ?", (t_id,))
    conn.commit()
    conn.close()

def get_task(t_id):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM tasks WHERE id = ?", (t_id,))
    row = c.fetchone()
    conn.close()
    if row: return dict(row)
    return None

def get_all_tasks():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT id, type, status, created_at FROM tasks ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_pending_or_processing_tasks():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM tasks WHERE status IN (0, 1)")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

init_db()
