# AI-Powered Attendance Tracker (Face Recognition)

A modern, modular Python application that uses face recognition to automatically mark attendance. Built with OpenCV, `face_recognition`, SQLite, and a polished PyQt5 desktop UI. It includes encrypted face embeddings, basic liveness checks, analytics, exports, and optional class scheduling.

## Highlights

- Encrypted embeddings: face vectors are encrypted at rest using `cryptography.Fernet` and stored in SQLite.
- PyQt5 GUI: clean multi-page interface (Dashboard, Attendance, Reports, Face Registration, Settings).
- Liveness check: simple blink + smile challenge to reduce spoofing before marking presence.
- Duplicate protection: prevents multiple "Present" entries per student per day and within the same session.
- Powerful filters and export: filter attendance by ID/name/course/year/date/status and export to CSV/Excel.
- Analytics: dashboard summaries and at-risk student chart; quick student report view.
- Scheduler (optional): define classes and auto-mark absentees at end of a session/time window.
- Privacy tools: quickly clean on-disk dataset folders if you use image datasets.

## Project Structure

- `main.py` — entry point. Launches the PyQt app and wires the database path under `data/attendance.db`.
- `gui_qt.py` — PyQt5 UI with pages for Dashboard, Attendance, Reports, Face Registration, and Settings.
- `attendance_recognition.py` — webcam inference, optional liveness (blink + smile), and attendance marking.
- `dataset_creator.py` — capture a short clip and create face embeddings; can optionally persist images for debugging.
- `database.py` — SQLite schema, migrations, CRUD, filtering, analytics helpers, and CSV/XLSX exports.
- `scheduler.py` — background timing for optional class scheduling and auto-absent logic.
- `insight.py` — helpers for dashboard analytics and at-risk calculations.
- `security.py` — Fernet key management and (de)serialization of encrypted embeddings.
- `seed_test_schedule.py` — convenience script to create a small teaching schedule and enrollments for demos.
- `requirements.txt` — Python dependencies.

Data locations:

- Database file: `data/attendance.db` (created on first run).
- Encryption key: `.secrets/fernet.key` (auto-created if missing; keep it safe!).

## Quick Start (Windows/macOS/Linux)

1) Create and activate a virtual environment

- Windows (PowerShell)
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
- macOS/Linux
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Notes on installation:

- `face_recognition` depends on `dlib`. Prefer a Python version/platform with prebuilt wheels. On Windows, you may need Build Tools/CMake if a wheel isn’t available.
- PyQt5 wheels are included in `requirements.txt`.

3) Run the app
```bash
python main.py
```

The console will print the database path, for example:
```
[MAIN] Database path: <repo>/data/attendance.db
```

## Using the App

The app launches into the PyQt5 UI defined in `gui_qt.py`.

- Dashboard
  - Summary cards for today’s presence and trend vs. prior weeks.
  - Top at-risk students table (low attendance rate and consecutive absences).

- Attendance
  - Filter by Student ID, Name, Course, Year, Status, and Date range.
  - Export filtered view to CSV or Excel (also split by status or quick “Today” exports).

- Reports
  - Donut chart for today’s present vs absent (if roster size is known).
  - Horizontal bar chart for top N at-risk students.
  - Quick student report dialog by ID or name.

- Face Registration
  - Register a student by entering ID, name, and optional course/year.
  - Captures a short webcam stream and stores multiple encrypted face embeddings in the DB.
  - After success, recognition can immediately use these embeddings (no need to restart).

- Settings
  - Seed sample students and a few days of attendance for demos.
  - Clean dataset folder (privacy) if you previously saved raw images.
  - Select the camera index and test a 5-second preview.

## Recognition & Liveness

- The recognizer (`attendance_recognition.py`) loads encrypted embeddings from the DB when available.
- If no embeddings exist, it can fall back to scanning images under `dataset/` if present.
- Before marking presence, it can require a brief liveness gesture (blink + smile). You can toggle `enable_liveness` in code or adjust thresholds.
- Press `q` to end an attendance session.

## Scheduling (Optional)

If you create a teaching schedule (tables `teachers`, `subjects`, `classes`, `enrollments`), the `AttendanceScheduler` can help with time-bound sessions, and you can auto-mark absentees for a class.

- Demo seeding
  ```bash
  # Ensure the app ran once (DB exists under data/)
  python seed_test_schedule.py
  ```
  This will insert a small set of teachers/subjects/classes and enroll current students.

## Data Model (Core Tables)

- `students(student_id TEXT PRIMARY KEY, student_name TEXT NOT NULL, created_at TEXT, ...additional optional columns...)`
- `attendance_logs(id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, student_name TEXT, date TEXT, time TEXT, status TEXT CHECK(status IN ('Present','Absent')))`
- `face_embeddings(id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, embedding BLOB, created_at TEXT)`

The `students` table is automatically migrated to include useful optional columns like `course`, `year_of_study`, `degree_level`, and more when missing.

## Security

- Face embeddings are encrypted at rest using a symmetric Fernet key stored at `.secrets/fernet.key`.
- Keep this key safe and back it up securely. Without it, embeddings cannot be decrypted for recognition.

## Troubleshooting

- Camera won’t open: try a different camera index in Settings, and ensure OS-level permissions are granted.
- `face_recognition` install: use a CPython version with prebuilt `dlib` wheels; upgrade pip before installing.
- Slow or unstable recognition: ensure good lighting; capture multiple embeddings per student; keep `model='hog'` for CPU-only systems.
- Nothing recognized after registration: confirm embeddings exist in the DB (`Settings` shows counts via exports/analytics), or fall back to a `dataset/` folder with clear face images.
- Excel export errors: ensure `openpyxl` is installed (it’s in `requirements.txt`).

## Contributing / License

This codebase was built for hackathon/demo use but follows a modular structure suitable for extension. Add your preferred license if needed.

