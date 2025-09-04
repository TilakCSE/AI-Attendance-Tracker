"""
database.py

Database management module for the AI-Powered Attendance Tracker.
Handles SQLite database setup, student management, attendance logging,
querying, and exporting to CSV/Excel.

Author: Your Name
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Optional, Dict, Any

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


@dataclass
class Student:
    student_id: str
    student_name: str
    created_at: str
    degree_level: str | None = None  # e.g., 'Bachelor' or 'Master'
    course: str | None = None
    year_of_study: int | None = None
    campus: str | None = None
    advisor: str | None = None
    is_active: int | None = 1  # 1 active, 0 inactive
    enrollment_start: str | None = None
    enrollment_end: str | None = None


@dataclass
class AttendanceRecord:
    id: int
    student_id: str
    student_name: str
    date: str
    time: str
    status: str


class DatabaseManager:
    """
    Simple wrapper around SQLite operations.
    """

    def __init__(self, db_path: str = "attendance.db") -> None:
        self.db_path = os.path.abspath(db_path)
        _ensure_dir(os.path.dirname(self.db_path) or ".")
        self._init_db()

    # --------- Internal helpers ---------

    def _table_exists(self, table_name: str) -> bool:
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
                return cur.fetchone() is not None
        except sqlite3.Error:
            return False

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        # Return rows as dictionaries
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            # Students table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    student_name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            # Attendance logs table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS attendance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    student_name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('Present','Absent')),
                    FOREIGN KEY (student_id) REFERENCES students(student_id)
                );
                """
            )
            # Face embeddings table (encrypted BLOBs)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (student_id) REFERENCES students(student_id)
                );
                """
            )
            
            # Schema migration: add optional columns to students if missing
            cur.execute("PRAGMA table_info('students');")
            cols = {row["name"] for row in cur.fetchall()}
            
            # Add any missing columns to the students table
            column_definitions = [
                ("degree_level", "TEXT"),
                ("course", "TEXT"),
                ("year_of_study", "INTEGER"),
                ("campus", "TEXT"),
                ("advisor", "TEXT"),
                ("is_active", "INTEGER DEFAULT 1"),
                ("enrollment_start", "TEXT"),
                ("enrollment_end", "TEXT")
            ]
            
            for col_name, col_type in column_definitions:
                if col_name not in cols:
                    logger.info(f"Migrating schema: adding '{col_name}' to students.")
                    cur.execute(f"ALTER TABLE students ADD COLUMN {col_name} {col_type};")
            
            conn.commit()

    # --------- Student operations ---------

    def add_student(
        self,
        student_id: str,
        student_name: str,
        course: Optional[str] = None,
        year_of_study: Optional[int] = None,
        degree_level: Optional[str] = None,
        campus: Optional[str] = None,
        advisor: Optional[str] = None,
        is_active: Optional[int] = 1,
        enrollment_start: Optional[str] = None,
        enrollment_end: Optional[str] = None,
    ) -> bool:
        """
        Insert a new student. Returns True if inserted, False if already exists.
        """
        student_id = student_id.strip()
        student_name = student_name.strip()
        course_val = (course or "").strip() or None
        year_val = int(year_of_study) if year_of_study not in (None, "",) else None
        degree_val = (degree_level or "").strip() or None
        campus_val = (campus or "").strip() or None
        advisor_val = (advisor or "").strip() or None
        active_val = 1 if (is_active is None or int(is_active) != 0) else 0
        start_val = (enrollment_start or "").strip() or None
        end_val = (enrollment_end or "").strip() or None

        if not student_id or not student_name:
            raise ValueError("student_id and student_name must be non-empty.")

        if self.get_student(student_id) is not None:
            return False

        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO students (
                    student_id, student_name, created_at,
                    course, year_of_study, degree_level,
                    campus, advisor, is_active, enrollment_start, enrollment_end
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    student_id,
                    student_name,
                    datetime.now().isoformat(timespec="seconds"),
                    course_val,
                    year_val,
                    degree_val,
                    campus_val,
                    advisor_val,
                    active_val,
                    start_val,
                    end_val,
                ),
            )
            conn.commit()
            logger.info(f"Student '{student_name}' ({student_id}) added to database.")
        return True

    def get_student(self, student_id: str) -> Optional[Student]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM students WHERE student_id = ?;", (student_id,))
            row = cur.fetchone()
            if row:
                return Student(
                    student_id=row["student_id"],
                    student_name=row["student_name"],
                    created_at=row["created_at"],
                    degree_level=row["degree_level"] if "degree_level" in row.keys() else None,
                    course=row["course"] if "course" in row.keys() else None,
                    year_of_study=row["year_of_study"] if "year_of_study" in row.keys() else None,
                    campus=row["campus"] if "campus" in row.keys() else None,
                    advisor=row["advisor"] if "advisor" in row.keys() else None,
                    is_active=row["is_active"] if "is_active" in row.keys() else None,
                    enrollment_start=row["enrollment_start"] if "enrollment_start" in row.keys() else None,
                    enrollment_end=row["enrollment_end"] if "enrollment_end" in row.keys() else None,
                )
        return None

    def list_students(self) -> List[Student]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM students ORDER BY created_at DESC;")
            rows = cur.fetchall()
            return [
                Student(
                    student_id=r["student_id"],
                    student_name=r["student_name"],
                    created_at=r["created_at"],
                    degree_level=r["degree_level"] if "degree_level" in r.keys() else None,
                    course=r["course"] if "course" in r.keys() else None,
                    year_of_study=r["year_of_study"] if "year_of_study" in r.keys() else None,
                    campus=r["campus"] if "campus" in r.keys() else None,
                    advisor=r["advisor"] if "advisor" in r.keys() else None,
                    is_active=r["is_active"] if "is_active" in r.keys() else None,
                    enrollment_start=r["enrollment_start"] if "enrollment_start" in r.keys() else None,
                    enrollment_end=r["enrollment_end"] if "enrollment_end" in r.keys() else None,
                )
                for r in rows
            ]

    # --------- Attendance operations ---------

    def has_attendance_today(self, student_id: str) -> bool:
        today = date.today().isoformat()
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT 1 FROM attendance_logs
                WHERE student_id = ? AND date = ? AND status = 'Present'
                LIMIT 1;
                """,
                (student_id, today),
            )
            return cur.fetchone() is not None

    def mark_attendance(self, student_id: str, student_name: str, status: str = "Present") -> bool:
        """
        Mark attendance for a student for today. Returns True if a new record was created,
        False if a 'Present' record already exists for today.
        """
        if status not in ("Present", "Absent"):
            raise ValueError("status must be 'Present' or 'Absent'.")

        # Prevent duplicate 'Present' for same student on same date.
        if status == "Present" and self.has_attendance_today(student_id):
            return False

        now = datetime.now()
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO attendance_logs (student_id, student_name, date, time, status)
                VALUES (?, ?, ?, ?, ?);
                """,
                (
                    student_id,
                    student_name,
                    now.date().isoformat(),
                    now.time().strftime("%H:%M:%S"),
                    status,
                ),
            )
            conn.commit()
            logger.info(f"Marked attendance for {student_id} on {now.date().isoformat()}")
        return True

    def list_attendance(self) -> List[AttendanceRecord]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, student_id, student_name, date, time, status
                FROM attendance_logs
                ORDER BY date DESC, time DESC;
                """
            )
            rows = cur.fetchall()
            return [
                AttendanceRecord(
                    id=r["id"],
                    student_id=r["student_id"],
                    student_name=r["student_name"],
                    date=r["date"],
                    time=r["time"],
                    status=r["status"],
                )
                for r in rows
            ]

    # --------- Scheduler support (classes/enrollments) ---------

    def get_classes_for_today(self) -> list[dict]:
        """
        Return today's classes as a list of dicts with keys: class_id, start_time, end_time,
        subject_name, teacher_name. If schedule tables are missing, return [].
        """
        required = ["teachers", "subjects", "classes"]
        if not all(self._table_exists(t) for t in required):
            return []

        from datetime import datetime as _dt
        dow = _dt.today().weekday()  # Monday=0
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT c.class_id,
                           c.start_time,
                           c.end_time,
                           s.subject_name,
                           t.teacher_name
                    FROM classes c
                    JOIN subjects s ON c.subject_id = s.subject_id
                    JOIN teachers t ON c.teacher_id = t.teacher_id
                    WHERE c.day_of_week = ?
                    ORDER BY c.start_time ASC;
                    """,
                    (dow,)
                )
                return [
                    {
                        "class_id": row["class_id"],
                        "start_time": row["start_time"],
                        "end_time": row["end_time"],
                        "subject_name": row["subject_name"],
                        "teacher_name": row["teacher_name"],
                    }
                    for row in cur.fetchall()
                ]
        except sqlite3.Error as e:
            logger.error(f"Error fetching classes for today: {e}")
            return []

    def mark_absentees_for_class(self, class_id: int) -> int:
        """
        For the given class_id, mark 'Absent' today for enrolled students who do not
        already have a 'Present' record today. Returns number of absentees marked.
        If required tables are missing, returns 0.
        """
        required = ["enrollments", "students"]
        if not all(self._table_exists(t) for t in required):
            return 0

        today = date.today().isoformat()
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                # Get enrolled students for class
                cur.execute(
                    "SELECT student_id FROM enrollments WHERE class_id = ?;",
                    (int(class_id),)
                )
                student_ids = [row["student_id"] for row in cur.fetchall()]
                if not student_ids:
                    return 0

                # Map to names
                cur.execute(
                    "SELECT student_id, student_name FROM students WHERE student_id IN (%s);" % (
                        ",".join(["?"] * len(student_ids))
                    ),
                    tuple(student_ids),
                )
                name_map = {row["student_id"]: row["student_name"] for row in cur.fetchall()}

                # For each student, if no Present today, insert Absent
                absent_marked = 0
                for sid in student_ids:
                    cur.execute(
                        """
                        SELECT 1 FROM attendance_logs
                        WHERE student_id = ? AND date = ? AND status='Present' LIMIT 1;
                        """,
                        (sid, today),
                    )
                    if cur.fetchone() is None:
                        cur.execute(
                            """
                            INSERT INTO attendance_logs (student_id, student_name, date, time, status)
                            VALUES (?, ?, ?, time('now'), 'Absent');
                            """,
                            (sid, name_map.get(sid, sid), today),
                        )
                        absent_marked += 1
                conn.commit()
                return absent_marked
        except sqlite3.Error as e:
            logger.error(f"Error marking absentees for class {class_id}: {e}")
            return 0

    def list_attendance_as_dicts(self) -> List[Dict[str, Any]]:
        records = self.list_attendance()
        return [
            {
                "id": rec.id,
                "student_id": rec.student_id,
                "student_name": rec.student_name,
                "date": rec.date,
                "time": rec.time,
                "status": rec.status,
            }
            for rec in records
        ]

    # --------- Export operations ---------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert attendance_logs to a pandas DataFrame.
        """
        data = self.list_attendance_as_dicts()
        if not data:
            return pd.DataFrame(columns=["id", "student_id", "student_name", "date", "time", "status"])
        return pd.DataFrame(data)

    def export_csv(self, out_path: str) -> str:
        df = self.to_dataframe()
        _ensure_dir(os.path.dirname(os.path.abspath(out_path)) or ".")
        df.to_csv(out_path, index=False)
        logger.info(f"Exported attendance to {out_path}")
        return os.path.abspath(out_path)

    def export_excel(self, out_path: str) -> str:
        df = self.to_dataframe()
        _ensure_dir(os.path.dirname(os.path.abspath(out_path)) or ".")
        # Requires openpyxl installed
        df.to_excel(out_path, index=False, engine="openpyxl")
        logger.info(f"Exported attendance to {out_path}")
        return os.path.abspath(out_path)

    def filtered_attendance_dataframe(
        self,
        student_id: Optional[str] = None,
        student_name: Optional[str] = None,
        course: Optional[str] = None,
        year_of_study: Optional[int] = None,
        status: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return a pandas DataFrame for attendance matching provided filters."""
        records = self.list_attendance_filtered(
            student_id=student_id,
            student_name=student_name,
            course=course,
            year_of_study=year_of_study,
            status=status,
            date_from=date_from,
            date_to=date_to,
        )
        if not records:
            return pd.DataFrame(columns=["id","student_id","student_name","date","time","status"])
        return pd.DataFrame([
            {"id": r.id, "student_id": r.student_id, "student_name": r.student_name, "date": r.date, "time": r.time, "status": r.status}
            for r in records
        ])

    def export_attendance_filtered(
        self,
        out_path: str,
        fmt: str = "csv",
        split_by_status: bool = False,
        **filters: Any,
    ) -> str | Dict[str, str]:
        """
        Export filtered attendance. If split_by_status is True:
        - CSV: writes two files with suffixes _present and _absent and returns their paths.
        - Excel: writes a single workbook with two sheets: Present, Absent.
        Returns absolute path (or dict of paths for CSV split).
        """
        fmt = (fmt or "csv").lower()
        _ensure_dir(os.path.dirname(os.path.abspath(out_path)) or ".")

        if not split_by_status:
            df = self.filtered_attendance_dataframe(**filters)
            if fmt == "xlsx":
                if not out_path.lower().endswith(".xlsx"):
                    out_path = os.path.splitext(out_path)[0] + ".xlsx"
                df.to_excel(out_path, index=False, engine="openpyxl")
            else:
                if not out_path.lower().endswith(".csv"):
                    out_path = os.path.splitext(out_path)[0] + ".csv"
                df.to_csv(out_path, index=False)
            logger.info(f"Exported filtered attendance to {out_path}")
            return os.path.abspath(out_path)

        # Split by status
        df_present = self.filtered_attendance_dataframe(status="Present", **{k:v for k,v in filters.items() if k!="status"})
        df_absent = self.filtered_attendance_dataframe(status="Absent", **{k:v for k,v in filters.items() if k!="status"})

        if fmt == "xlsx":
            if not out_path.lower().endswith(".xlsx"):
                out_path = os.path.splitext(out_path)[0] + ".xlsx"
            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                df_present.to_excel(writer, sheet_name="Present", index=False)
                df_absent.to_excel(writer, sheet_name="Absent", index=False)
            logger.info(f"Exported split-by-status workbook to {out_path}")
            return os.path.abspath(out_path)
        else:
            base, ext = os.path.splitext(out_path)
            if ext.lower() != ".csv":
                base = out_path
            present_path = f"{base}_present.csv"
            absent_path = f"{base}_absent.csv"
            df_present.to_csv(present_path, index=False)
            df_absent.to_csv(absent_path, index=False)
            logger.info(f"Exported split CSVs to {present_path}, {absent_path}")
            return {"present": os.path.abspath(present_path), "absent": os.path.abspath(absent_path)}

    def export_today_attendance(
        self,
        out_path: str,
        fmt: str = "csv",
        class_id: Optional[int] = None,
        only_present: bool = False,
    ) -> str:
        """
        Export today's attendance. If class_id is provided and the enrollments table exists,
        restrict to students enrolled in that class. If only_present is True, include only Present rows.
        """
        from datetime import date as _date
        today = _date.today().isoformat()

        if class_id is not None and not self._table_exists("enrollments"):
            class_id = None  # gracefully ignore if schema not present

        # Build filters
        filters: Dict[str, Any] = {"date_from": today, "date_to": today}
        if only_present:
            filters["status"] = "Present"

        df = self.filtered_attendance_dataframe(**filters)
        if class_id is not None and not df.empty:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT student_id FROM enrollments WHERE class_id = ?;", (int(class_id),))
                allowed = {row[0] for row in cur.fetchall()}
            df = df[df["student_id"].isin(list(allowed))]

        # Write output
        _ensure_dir(os.path.dirname(os.path.abspath(out_path)) or ".")
        fmt = (fmt or "csv").lower()
        if fmt == "xlsx":
            if not out_path.lower().endswith(".xlsx"):
                out_path = os.path.splitext(out_path)[0] + ".xlsx"
            df.to_excel(out_path, index=False, engine="openpyxl")
        else:
            if not out_path.lower().endswith(".csv"):
                out_path = os.path.splitext(out_path)[0] + ".csv"
            df.to_csv(out_path, index=False)
        logger.info(f"Exported today's attendance to {out_path}")
        return os.path.abspath(out_path)

    # --------- Embeddings (encrypted) operations ---------

    def add_face_embedding(self, student_id: str, embedding_blob: bytes) -> None:
        """
        Store an encrypted face embedding for a student.
        """
        student_id = student_id.strip()
        if not student_id:
            raise ValueError("student_id must be non-empty.")
        if not isinstance(embedding_blob, (bytes, bytearray)):
            raise ValueError("embedding_blob must be bytes.")

        from datetime import datetime as _dt

        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO face_embeddings (student_id, embedding, created_at)
                VALUES (?, ?, ?);
                """,
                (student_id, sqlite3.Binary(embedding_blob), _dt.now().isoformat(timespec="seconds")),
            )
            conn.commit()
            logger.info(f"Added face embedding for {student_id}")

    def get_all_face_embeddings(self) -> Dict[str, List[bytes]]:
        """
        Return a mapping of student_id -> list of encrypted embedding blobs.
        """
        mapping: Dict[str, List[bytes]] = {}
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT student_id, embedding FROM face_embeddings;")
            for row in cur.fetchall():
                sid = row["student_id"]
                emb: bytes = row["embedding"]
                mapping.setdefault(sid, []).append(emb)
        return mapping

    def get_student_name_map(self) -> Dict[str, str]:
        """
        Return a mapping of student_id -> student_name.
        """
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT student_id, student_name FROM students;")
            return {row["student_id"]: row["student_name"] for row in cur.fetchall()}

    def get_distinct_courses(self) -> List[str]:
        """Fetch a sorted list of unique courses from the students table."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT DISTINCT course FROM students WHERE course IS NOT NULL ORDER BY course;")
                return [row['course'] for row in cur.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Database error fetching distinct courses: {e}")
            return []

    def get_distinct_years(self) -> List[int]:
        """Fetch a sorted list of unique years of study from the students table."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT DISTINCT year_of_study FROM students WHERE year_of_study IS NOT NULL ORDER BY year_of_study;")
                return [row['year_of_study'] for row in cur.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Database error fetching distinct years: {e}")
            return []

    def count_embeddings(self, student_id: Optional[str] = None) -> int:
        """
        Count number of embeddings, optionally for a specific student.
        """
        with self._connect() as conn:
            cur = conn.cursor()
            if student_id:
                cur.execute("SELECT COUNT(*) AS c FROM face_embeddings WHERE student_id = ?;", (student_id,))
            else:
                cur.execute("SELECT COUNT(*) AS c FROM face_embeddings;")
            row = cur.fetchone()
            return int(row["c"] if row else 0)

    # --------- CRUD utilities ---------

    def delete_student(self, student_id: str) -> bool:
        """
        Delete a student and related data (embeddings, attendance logs).
        Returns True if a student was deleted.
        """
        student_id = student_id.strip()
        if not student_id:
            raise ValueError("student_id must be non-empty.")
        with self._connect() as conn:
            cur = conn.cursor()
            # delete dependents first due to FK constraints
            cur.execute("DELETE FROM face_embeddings WHERE student_id = ?;", (student_id,))
            cur.execute("DELETE FROM attendance_logs WHERE student_id = ?;", (student_id,))
            cur.execute("DELETE FROM students WHERE student_id = ?;", (student_id,))
            conn.commit()
            if cur.rowcount > 0:
                logger.info(f"Deleted student {student_id} and all associated data.")
            return cur.rowcount > 0

    def wipe_database(self) -> None:
        """
        Remove all rows from all tables (keeps schema).
        """
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM face_embeddings;")
            cur.execute("DELETE FROM attendance_logs;")
            cur.execute("DELETE FROM students;")
            conn.commit()
            logger.warning("Wiped all data from all tables.")

    def update_attendance_status(self, record_id: int, status: str) -> None:
        """
        Update an attendance record's status to 'Present' or 'Absent'.
        """
        if status not in ("Present", "Absent"):
            raise ValueError("status must be 'Present' or 'Absent'.")

        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE attendance_logs SET status = ? WHERE id = ?;", (status, int(record_id)))
            conn.commit()
            logger.info(f"Updated attendance record {record_id} to {status}")

    def list_attendance_filtered(
        self,
        student_id: Optional[str] = None,
        student_name: Optional[str] = None,
        course: Optional[str] = None,
        year_of_study: Optional[int] = None,
        status: Optional[str] = None,
        date_from: Optional[str] = None,  # 'YYYY-MM-DD'
        date_to: Optional[str] = None,    # 'YYYY-MM-DD'
    ) -> List[AttendanceRecord]:
        """
        Return attendance records filtered by provided criteria.
        Filters on name/course/year are applied via join with students table.
        """
        base_query = """
            SELECT a.id, a.student_id, a.student_name, a.date, a.time, a.status
            FROM attendance_logs a
            JOIN students s ON a.student_id = s.student_id
        """
        conditions = []
        params = {}

        if student_id:
            conditions.append("a.student_id LIKE :sid")
            params["sid"] = f"%{student_id}%"
        if student_name:
            conditions.append("a.student_name LIKE :sname")
            params["sname"] = f"%{student_name}%"
        if status:
            conditions.append("a.status = :status")
            params["status"] = status
        if date_from:
            conditions.append("a.date >= :date_from")
            params["date_from"] = date_from
        if date_to:
            conditions.append("a.date <= :date_to")
            params["date_to"] = date_to
        if course:
            conditions.append("s.course LIKE :course")
            params["course"] = f"%{course}%"
        if year_of_study:
            conditions.append("s.year_of_study = :year")
            params["year"] = year_of_study

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        base_query += " ORDER BY a.date DESC, a.time DESC;"

        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(base_query, params)
                return [AttendanceRecord(**row) for row in cur.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Database error while filtering attendance: {e}")
            return []

    # --------- People & Roster (institution-wide) ---------

    def list_students_filtered(
        self,
        student_id: Optional[str] = None,
        student_name: Optional[str] = None,
        degree_level: Optional[str] = None,
        course: Optional[str] = None,
        year_of_study: Optional[int] = None,
        campus: Optional[str] = None,
        is_active: Optional[int] = None,
    ) -> List[Student]:
        sql = ["SELECT * FROM students WHERE 1=1"]
        params: list[Any] = []

        if student_id:
            sql.append("AND student_id LIKE ?")
            params.append(f"%{student_id.strip()}%")
        if student_name:
            sql.append("AND student_name LIKE ?")
            params.append(f"%{student_name.strip()}%")
        if degree_level:
            sql.append("AND degree_level = ?")
            params.append(degree_level.strip())
        if course:
            sql.append("AND course LIKE ?")
            params.append(f"%{course.strip()}%")
        if year_of_study not in (None, "",):
            sql.append("AND year_of_study = ?")
            params.append(int(year_of_study))
        if campus:
            sql.append("AND campus LIKE ?")
            params.append(f"%{campus.strip()}%")
        if is_active in (0, 1):
            sql.append("AND is_active = ?")
            params.append(int(is_active))

        sql.append("ORDER BY created_at DESC")
        query = "\n".join(sql)

        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            return [
                Student(
                    student_id=r["student_id"],
                    student_name=r["student_name"],
                    created_at=r["created_at"],
                    degree_level=r["degree_level"] if "degree_level" in r.keys() else None,
                    course=r["course"] if "course" in r.keys() else None,
                    year_of_study=r["year_of_study"] if "year_of_study" in r.keys() else None,
                    campus=r["campus"] if "campus" in r.keys() else None,
                    advisor=r["advisor"] if "advisor" in r.keys() else None,
                    is_active=r["is_active"] if "is_active" in r.keys() else None,
                    enrollment_start=r["enrollment_start"] if "enrollment_start" in r.keys() else None,
                    enrollment_end=r["enrollment_end"] if "enrollment_end" in r.keys() else None,
                )
                for r in rows
            ]

    def search_students(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Omnibox search: matches by ID, name, course, degree, campus.
        """
        q = f"%{(query or '').strip()}%"
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT student_id, student_name, degree_level, course, year_of_study, campus
                FROM students
                WHERE student_id LIKE ? OR student_name LIKE ? OR course LIKE ? OR degree_level LIKE ? OR campus LIKE ?
                ORDER BY student_name ASC
                LIMIT ?;
                """,
                (q, q, q, q, q, int(limit)),
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]

    def get_last_seen_map(self) -> Dict[str, Optional[str]]:
        """
        Returns a mapping student_id -> last attendance date (YYYY-MM-DD) or None.
        """
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT student_id, MAX(date) as last_date
                FROM attendance_logs
                WHERE status='Present'
                GROUP BY student_id;
                """
            )
            return {row["student_id"]: row["last_date"] for row in cur.fetchall()}

    def roster_dataframe(self, deidentify: bool = False, filtered: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Build a complete roster DataFrame with derived metrics:
        total_sessions, present_count, attendance_rate, last_seen, consecutive_absences.
        """
        # Load students (optionally filtered)
        if filtered:
            students = self.list_students_filtered(
                student_id=filtered.get("student_id"),
                student_name=filtered.get("student_name"),
                degree_level=filtered.get("degree_level"),
                course=filtered.get("course"),
                year_of_study=filtered.get("year_of_study"),
                campus=filtered.get("campus"),
                is_active=filtered.get("is_active"),
            )
        else:
            students = self.list_students()

        if not students:
            return pd.DataFrame(columns=[
                "student_id","student_name","degree_level","course","year_of_study","campus","advisor","is_active",
                "enrollment_start","enrollment_end","last_seen","total_sessions","present_count","attendance_rate","consecutive_absences"
            ])

        # Attendance aggregates
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT student_id,
                       COUNT(*) as total_sessions,
                       SUM(CASE WHEN status='Present' THEN 1 ELSE 0 END) as present_count
                FROM attendance_logs
                GROUP BY student_id;
                """
            )
            agg = {row["student_id"]: (row["total_sessions"], row["present_count"]) for row in cur.fetchall()}

            # For consecutive absences, approximate using last 30 days
            cur.execute(
                """
                SELECT student_id, date, status
                FROM attendance_logs
                WHERE date >= date('now','-60 day')
                ORDER BY student_id, date DESC;
                """
            )
            rows = cur.fetchall()

        # Compute consecutive absences
        cons_abs: Dict[str, int] = {}
        last_seen = self.get_last_seen_map()
        from collections import defaultdict
        hist_map: Dict[str, list[tuple[str,str]]] = defaultdict(list)
        for r in rows:
            hist_map[r["student_id"]].append((r["date"], r["status"]))
        for sid, events in hist_map.items():
            count = 0
            for _, st in events:
                if st == "Absent":
                    count += 1
                else:
                    break
            cons_abs[sid] = count

        # Build rows
        records = []
        for s in students:
            total, present = agg.get(s.student_id, (0, 0))
            rate = (present / total) if total else 0.0
            rec = {
                "student_id": s.student_id if not deidentify else f"stu_{abs(hash(s.student_id))%10_000_000}",
                "student_name": s.student_name if not deidentify else "REDACTED",
                "degree_level": s.degree_level,
                "course": s.course,
                "year_of_study": s.year_of_study,
                "campus": s.campus,
                "advisor": s.advisor,
                "is_active": s.is_active,
                "enrollment_start": s.enrollment_start,
                "enrollment_end": s.enrollment_end,
                "last_seen": last_seen.get(s.student_id),
                "total_sessions": total,
                "present_count": present,
                "attendance_rate": round(rate, 3),
                "consecutive_absences": cons_abs.get(s.student_id, 0),
            }
            records.append(rec)

        df = pd.DataFrame(records)
        return df

    def export_complete_roster(self, out_path: str, fmt: str = "csv", deidentify: bool = False, filtered: Optional[Dict[str, Any]] = None) -> str:
        """
        Export the complete roster with derived metrics.
        fmt: 'csv', 'xlsx', 'json', 'parquet' (if pyarrow available).
        """
        try:
            df = self.roster_dataframe(deidentify=deidentify, filtered=filtered)
            _ensure_dir(os.path.dirname(os.path.abspath(out_path)) or ".")
            fmt = (fmt or "").lower()
            
            if fmt == "xlsx":
                df.to_excel(out_path, index=False, engine="openpyxl")
            elif fmt == "json":
                df.to_json(out_path, orient="records", indent=2, date_format="iso")
            elif fmt == "parquet":
                try:
                    df.to_parquet(out_path, index=False)  # requires pyarrow or fastparquet
                except Exception as e:
                    # Fall back to CSV if parquet not available
                    logger.warning(f"Failed to export as parquet, falling back to CSV: {str(e)}")
                    out_path = os.path.splitext(out_path)[0] + ".csv"
                    df.to_csv(out_path, index=False)
            else:
                # Default to CSV
                if not out_path.lower().endswith('.csv'):
                    out_path = os.path.splitext(out_path)[0] + ".csv"
                df.to_csv(out_path, index=False)
                
            logger.info(f"Exported roster to {out_path}")
            return os.path.abspath(out_path)
            
        except Exception as e:
            logger.error(f"Error exporting roster: {str(e)}")
            raise

    # --------- Sample data seeding ---------

    def seed_sample_students(self, count: int = 50) -> int:
        """
        Insert up to `count` sample students with Indian names and patterned IDs.
        ID pattern examples:
          5th year -> starts with "21" (e.g., 21000501)
          4th year -> "22"; ... 1st year -> "25"
        Names/courses cycle through lists to generate variety.
        Returns number of students inserted.
        """
        first_names = [
            "Aarav","Vivaan","Aditya","Vihaan","Arjun","Sai","Reyansh","Krishna","Ishaan","Rudra",
            "Ananya","Diya","Aadhya","Anika","Navya","Sara","Ira","Aarohi","Myra","Saanvi",
        ]
        last_names = [
            "Sharma","Verma","Gupta","Iyer","Patel","Reddy","Naidu","Khan","Singh","Das",
            "Roy","Mukherjee","Nair","Pillai","Chatterjee","Kulkarni","Joshi","Mehta","Kapoor","Bansal",
        ]
        courses = ["CSE","ECE","EEE","ME","CE","IT","AI","DS","MBA","BBA"]
        years_prefix = {5:"21",4:"22",3:"23",2:"24",1:"25"}

        inserted = 0
        import random
        rng = random.Random(42)
        for i in range(count):
            year = 5 - (i // max(1, (count // 5)))  # roughly distribute across years
            if year < 1:
                year = 1
            prefix = years_prefix[year]
            # Construct 8-digit ID with prefix + 6 digits varying
            mid = f"{rng.randint(0, 999999):06d}"
            sid = prefix + mid
            fname = first_names[i % len(first_names)]
            lname = last_names[(i * 7) % len(last_names)]
            name = f"{fname} {lname}"
            course = courses[(i * 3) % len(courses)]
            try:
                if self.add_student(sid, name, course=course, year_of_study=year):
                    inserted += 1
            except Exception:
                # skip duplicates or invalid
                continue
        logger.info(f"Seeded {inserted} sample students.")
        return inserted

    def seed_sample_attendance(self, days: int = 3, present_rate: float = 0.7) -> int:
        """
        Seed random attendance for existing students over the past `days` days.
        present_rate is the probability a student is marked Present on a day.
        Returns total number of attendance rows inserted.
        """
        from datetime import timedelta
        import random
        rng = random.Random(123)
        students = self.list_students()
        if not students:
            return 0
        total = 0
        with self._connect() as conn:
            cur = conn.cursor()
            today = date.today()
            for d in range(days):
                the_date = (today - timedelta(days=d)).isoformat()
                for s in students:
                    # Avoid duplicates: if any record exists for that day+student, skip
                    cur.execute("SELECT 1 FROM attendance_logs WHERE student_id=? AND date=? LIMIT 1;", (s.student_id, the_date))
                    if cur.fetchone():
                        continue
                    status = 'Present' if rng.random() < present_rate else 'Absent'
                    cur.execute(
                        """
                        INSERT INTO attendance_logs (student_id, student_name, date, time, status)
                        VALUES (?, ?, ?, ?, ?);
                        """,
                        (s.student_id, s.student_name, the_date, "09:00:00", status)
                    )
                    total += 1
            conn.commit()
        logger.info(f"Seeded {total} attendance rows over last {days} days.")
        return total

    def search_students(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Omnibox search: matches by ID, name, course, degree, campus.
        """
        q = f"%{(query or '').strip()}%"
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT student_id, student_name, degree_level, course, year_of_study, campus
                FROM students
                WHERE student_id LIKE ? OR student_name LIKE ? OR course LIKE ? OR degree_level LIKE ? OR campus LIKE ?
                ORDER BY student_name ASC
                LIMIT ?;
                """,
                (q, q, q, q, q, int(limit)),
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
