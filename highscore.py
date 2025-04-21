# highscore.py
import sqlite3

def init_db():
    conn = sqlite3.connect("highscore.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS highscores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            score INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_score(name, score):
    conn = sqlite3.connect("highscore.db")
    c = conn.cursor()
    c.execute("INSERT INTO highscores (name, score) VALUES (?, ?)", (name, score))
    conn.commit()
    conn.close()

def get_top_scores(limit=10):
    conn = sqlite3.connect("highscore.db")
    c = conn.cursor()
    c.execute("SELECT name, score FROM highscores ORDER BY score DESC LIMIT ?", (limit,))
    scores = c.fetchall()
    conn.close()
    return scores
