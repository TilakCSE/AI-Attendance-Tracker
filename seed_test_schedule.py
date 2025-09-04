import sqlite3
import os

# --- Configuration ---
# Day of week: Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6
TEST_DAY_OF_WEEK = 1  # Tuesday
TEST_START_TIME = "16:44"  # Use 24-hour format (HH:MM)
TEST_END_TIME = "16:46"

TEACHERS_AND_SUBJECTS = [
    {"teacher": "Yogesh", "subject": "Computer Graphics", "course": "Computer Science", "year": 3},
    {"teacher": "Ujwala", "subject": "Software Engineering", "course": "Computer Science", "year": 4},
    {"teacher": "Mehul", "subject": "Cyber Security", "course": "Computer Science", "year": 3},
]

# --- Script ---
project_root = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(project_root, "data", "attendance.db")
os.makedirs(os.path.dirname(db_path), exist_ok=True)  # Ensure data directory exists

def seed_data():
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        for item in TEACHERS_AND_SUBJECTS:
            teacher_name = item["teacher"]
            subject_name = item["subject"]
            course = item["course"]
            year = item["year"]

            # --- 1. Add Teacher ---
            cur.execute("SELECT teacher_id FROM teachers WHERE teacher_name = ?", (teacher_name,))
            teacher_record = cur.fetchone()
            if teacher_record:
                teacher_id = teacher_record[0]
            else:
                cur.execute("INSERT INTO teachers (teacher_name) VALUES (?)", (teacher_name,))
                teacher_id = cur.lastrowid
            print(f"Teacher '{teacher_name}' processed with ID: {teacher_id}")

            # --- 2. Add Subject ---
            cur.execute("SELECT subject_id FROM subjects WHERE subject_name = ?", (subject_name,))
            subject_record = cur.fetchone()
            if subject_record:
                subject_id = subject_record[0]
            else:
                cur.execute("INSERT INTO subjects (subject_name, course, year_of_study) VALUES (?, ?, ?)", (subject_name, course, year))
                subject_id = cur.lastrowid
            print(f"Subject '{subject_name}' processed with ID: {subject_id}")

            # --- 3. Add or Update Class Schedule ---
            cur.execute("SELECT class_id FROM classes WHERE subject_id = ? AND teacher_id = ?", (subject_id, teacher_id))
            class_record = cur.fetchone()
            if class_record:
                class_id = class_record[0]
                cur.execute("UPDATE classes SET day_of_week = ?, start_time = ?, end_time = ? WHERE class_id = ?",
                            (TEST_DAY_OF_WEEK, TEST_START_TIME, TEST_END_TIME, class_id))
                print(f"  - Updated class schedule for class ID: {class_id}")
            else:
                cur.execute("INSERT INTO classes (subject_id, teacher_id, day_of_week, start_time, end_time) VALUES (?, ?, ?, ?, ?)",
                            (subject_id, teacher_id, TEST_DAY_OF_WEEK, TEST_START_TIME, TEST_END_TIME))
                class_id = cur.lastrowid
                print(f"  - Added class schedule with ID: {class_id}")
            print(f"    - Schedule: Day {TEST_DAY_OF_WEEK}, {TEST_START_TIME}-{TEST_END_TIME}")

            # --- 4. Enroll all existing students in this class ---
            cur.execute("SELECT student_id FROM students")
            student_ids = [row[0] for row in cur.fetchall()]
            enroll_count = 0
            for student_id in student_ids:
                cur.execute("INSERT OR IGNORE INTO enrollments (class_id, student_id) VALUES (?, ?)", (class_id, student_id))
                if cur.rowcount > 0:
                    enroll_count += 1
            if enroll_count > 0:
                print(f"    - Enrolled {enroll_count} new students.")

        conn.commit()
        print("\nTest data seeded successfully!")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        print("Please run the main application first to create the database.")
    else:
        seed_data()
