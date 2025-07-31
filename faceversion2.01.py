import sqlite3
import os
from datetime import datetime

DB_PATH = "attendance.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    mobile TEXT NOT NULL,
                    image TEXT NOT NULL
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER,
                    time TEXT NOT NULL,
                    FOREIGN KEY(student_id) REFERENCES students(id)
                )''')
    conn.commit()
    conn.close()

def insert_student(name, mobile, image):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO students (name, mobile, image) VALUES (?, ?, ?)", (name, mobile, image))
    conn.commit()
    conn.close()

def get_all_students():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM students")
    rows = c.fetchall()
    conn.close()
    return rows

def get_student_by_name(name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM students WHERE name=?", (name,))
    row = c.fetchone()
    conn.close()
    return row

def insert_attendance(student_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO attendance (student_id, time) VALUES (?, ?)", (student_id, now))
    conn.commit()
    conn.close()

def get_today_attendance():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute('''
        SELECT students.name, students.mobile, attendance.time 
        FROM attendance
        JOIN students ON students.id = attendance.student_id
        WHERE attendance.time LIKE ?
        ORDER BY attendance.time DESC
    ''', (f"{today}%",))
    rows = c.fetchall()
    conn.close()
    return rows