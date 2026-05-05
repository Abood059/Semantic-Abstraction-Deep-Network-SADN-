import sqlite3
import logging

DB_FILE = None

def get_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path):
    global DB_FILE
    DB_FILE = db_path
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            layer INTEGER NOT NULL,
            node INTEGER NOT NULL,
            sentence_index INTEGER NOT NULL,
            sentence TEXT NOT NULL,
            PRIMARY KEY (layer, node, sentence_index)
        )
    ''')
    conn.commit()
    conn.close()
    logging.info("تم تجهيز قاعدة البيانات.")

def add_results(layer, node, sentences):
    """إدراج 10 جمل لعقدة معينة."""
    conn = get_connection()
    c = conn.cursor()
    for idx, sentence in enumerate(sentences, start=1):
        c.execute('''
            INSERT OR REPLACE INTO results (layer, node, sentence_index, sentence)
            VALUES (?, ?, ?, ?)
        ''', (layer, node, idx, sentence.strip()))
    conn.commit()
    conn.close()

def get_inputs_for_node(layer, node, num_nodes):
    """
    استرجاع المدخلات لعقدة في الطبقة layer (>1):
    الجملة رقم (node) من كل عقدة في الطبقة السابقة.
    """
    previous_layer = layer - 1
    conn = get_connection()
    c = conn.cursor()
    inputs = []
    for prev_node in range(1, num_nodes + 1):
        c.execute('''
            SELECT sentence FROM results
            WHERE layer = ? AND node = ? AND sentence_index = ?
        ''', (previous_layer, prev_node, node))
        row = c.fetchone()
        if row:
            inputs.append(row['sentence'])
        else:
            inputs.append("")  # احتياط
    conn.close()
    return inputs
