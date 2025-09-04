# AI-Powered Attendance Tracker (Face Recognition)

A beginner-friendly, modular Python application that uses face recognition to automatically mark attendance. Built with OpenCV, `face_recognition`, SQLite, and Tkinter.

## Features

- Dataset Creation: Register students by capturing images from the webcam and saving them in `dataset/{student_id}_{student_name}/`.
- Real-time Recognition: Recognize registered students using `face_recognition` and OpenCV.
- Attendance Logging: SQLite database (`attendance.db`) with tables for Students and Attendance Logs.
- GUI: Tkinter interface to Register Students, Start Attendance, View Attendance, and Export to CSV/Excel.
- Duplicate Protection: Prevents multiple "Present" entries for the same student on the same date, and avoids duplicates within a single session.

### Project Insight (Institution-Scale Additions)

- Omnibox Search: instantly search the complete institutional roster (past and present) by ID, name, course, degree, or campus.
- People Browser: full student directory with filters (degree, course, year, campus, active), infinite-friendly list, and filtered roster export.
- Reports & Insights: “Today” summary cards, at-risk list, and quick actions.
- Complete Roster Export: CSV/Excel/JSON (Parquet if available) with derived metrics (attendance rate, last seen, consecutive absences) and optional de-identification for privacy.
- Enriched Student Model: degree level, course, year, campus, advisor, active/inactive, enrollment dates (backward compatible migrations).

## Project Structure

- `database.py` — SQLite setup and operations (students, attendance logs, exports).
- `dataset_creator.py` — Capture and store student face images from the webcam.
- `attendance_recognition.py` — Load known faces and mark attendance in real time.
- `gui.py` — Tkinter GUI integrating all modules.
- `main.py` — Entry point to launch the app.
- `requirements.txt` — Dependencies.
- `dataset/` — Generated dataset folders after registration.
- `attendance.db` — SQLite database generated at first run.

## Prerequisites

- Python 3.9+ recommended.
- A working webcam.
- Platform-specific notes:
  - The `face_recognition` package depends on `dlib`. Many platforms have prebuilt wheels; ensure you have a compatible Python version. If installation fails, refer to the `face_recognition` docs for prerequisites (e.g., CMake, Visual C++ Build Tools on Windows).
  - Tkinter and sqlite3 are part of the Python standard library. Ensure your Python installation includes Tk support.

## Setup

1. Create and activate a virtual environment (recommended):

   - Windows (PowerShell):
     ```
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - macOS/Linux:
     ```
     python3 -m venv .venv
     source .venv/bin/activate
     ```

2. Install dependencies:
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Run the app:
   ```
   python main.py
   ```

## Using the App (Hackathon Demo Flow)

1. Register Student
   - Click "Register Student".
   - Enter Student ID (e.g., S001) and Student Name (e.g., Alice).
   - Choose number of images to capture (default 20).
   - Click "Start Capture".
   - A camera window opens; position the face inside the frame. The app captures images when a face is detected.
   - Press 'q' anytime to cancel. On success, images are saved under `dataset/S001_Alice/`.
   - The student is saved in the database. The app updates its known faces automatically.

2. Start Attendance
   - Click "Start Attendance".
   - A camera window opens for live recognition.
   - When a registered student is recognized, their attendance is marked as "Present" for today (duplicates are prevented).
   - Press 'q' to end the session.

3. View Attendance
   - The "Attendance Records" table shows the latest logs.
   - Click "Refresh View" to reload.

4. Export Attendance
   - Click "Export CSV" or "Export Excel" to save the current attendance logs.
   - Choose a file path; the app will confirm when the export completes.

## Data Model

- `students(student_id TEXT PRIMARY KEY, student_name TEXT NOT NULL, created_at TEXT NOT NULL)`
- `attendance_logs(id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, student_name TEXT, date TEXT, time TEXT, status TEXT CHECK(status IN ('Present','Absent')), FOREIGN KEY(student_id) REFERENCES students(student_id))`

## Tips & Troubleshooting

- If the webcam doesn’t open, verify camera permissions and the default camera index (change `camera_index` in code if needed).
- If recognition seems inaccurate, capture more images per student under varying lighting and angles.
- On low-spec machines, use the default `hog` model (already set) instead of `cnn` for face detection.
- If no faces are recognized after registration, ensure images exist in `dataset/` and restart "Start Attendance" to reload encodings.

## License

For hackathon/demo use. Add your preferred license if needed.
