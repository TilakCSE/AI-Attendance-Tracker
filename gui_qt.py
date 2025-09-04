"""
PyQt5 GUI for the AI-Powered Attendance Tracker.
This replaces the prior Tkinter UI while keeping the same app architecture:
- launch_app(db_path) entrypoint
- Uses DatabaseManager, AttendanceRecognizer, DatasetCreator, AttendanceScheduler
- Modular pages in a QStackedWidget (Dashboard, Attendance, Face Registration, Settings)
"""
from __future__ import annotations

import sys
import threading
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QStackedWidget, QListWidget, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QMessageBox, QFileDialog, QComboBox, QLineEdit,
    QStatusBar, QDialog, QDialogButtonBox, QFormLayout, QAbstractItemView
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from database import DatabaseManager
from dataset_creator import DatasetCreator
from attendance_recognition import AttendanceRecognizer
from scheduler import AttendanceScheduler


# ---------- Helpers ----------

def info_popup(parent: QWidget, title: str, text: str) -> None:
    QMessageBox.information(parent, title, text)

def error_popup(parent: QWidget, title: str, text: str) -> None:
    QMessageBox.critical(parent, title, text)


# ---------- Main Window ----------
class MainWindow(QMainWindow):
    def __init__(self, db: DatabaseManager):
        super().__init__()
        self.db = db
        self.recognizer = AttendanceRecognizer()
        self.creator = DatasetCreator()
        self.scheduler = AttendanceScheduler(self.db, self.recognizer)
        self.scheduler.start()
        self.attendance_thread: Optional[threading.Thread] = None

        self.setWindowTitle("AI-Powered Attendance Tracker")
        self.resize(1300, 820)
        self._apply_styles()

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        self.sidebar = QListWidget()
        self.sidebar.addItems(["Dashboard", "Attendance", "Reports", "Face Registration", "Settings"])
        self.sidebar.setFixedWidth(220)
        main_layout.addWidget(self.sidebar)

        # Content area
        self.content = QStackedWidget()
        main_layout.addWidget(self.content, 1)

        # Pages
        self.dashboard_page = DashboardPage(self)
        self.attendance_page = AttendancePage(self)
        self.reports_page = ReportsPage(self)
        self.registration_page = RegistrationPage(self)
        self.settings_page = SettingsPage(self)

        self.content.addWidget(self.dashboard_page)
        self.content.addWidget(self.attendance_page)
        self.content.addWidget(self.reports_page)
        self.content.addWidget(self.registration_page)
        self.content.addWidget(self.settings_page)

        self.sidebar.currentRowChanged.connect(self.content.setCurrentIndex)
        self.sidebar.setCurrentRow(0)

        # Status bar
        self.statusBar().showMessage("Ready")

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background-color: #f4f6f9; }
            QListWidget {
                background: #2c3e50; color: #ecf0f1; border: 0; padding-top: 10px; font-size: 14px;
            }
            QListWidget::item { padding: 12px 18px; }
            QListWidget::item:selected, QListWidget::item:hover { background: #3498db; color: white; }
            QGroupBox { font-weight: bold; }
            QPushButton { background: #3498db; color: white; border: 0; padding: 8px 14px; border-radius: 4px; }
            QPushButton:hover { background: #2f89c6; }
            QTableWidget { background: white; }
            QHeaderView::section { background: #ecf0f1; padding: 6px; border: 0; }
            QStatusBar { background: #2c3e50; color: white; }
            """
        )

    def closeEvent(self, event) -> None:  # graceful shutdown
        try:
            self.scheduler.stop()
        except Exception:
            pass
        if self.attendance_thread and self.attendance_thread.is_alive():
            reply = QMessageBox.question(
                self,
                "Quit",
                "Attendance session is running. Quit anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            try:
                self.recognizer.stop()
            except Exception:
                pass
        event.accept()


# ---------- Dashboard Page ----------
class DashboardPage(QWidget):
    def __init__(self, main: MainWindow):
        super().__init__(main)
        from insight import compute_today_summary, compute_risk_list
        self.main = main
        self.db = main.db
        layout = QVBoxLayout(self)
        title = QLabel("Dashboard")
        title.setFont(QFont("Segoe UI", 22, QFont.Bold))
        layout.addWidget(title)
        # Summary labels
        self.lbl_present = QLabel("Present Today: 0")
        self.lbl_unique = QLabel("Unique Today: 0")
        self.lbl_avg = QLabel("Avg Same Weekday (last 4): 0.0")
        self.lbl_delta = QLabel("Δ vs Avg: 0.0")
        for w in (self.lbl_present, self.lbl_unique, self.lbl_avg, self.lbl_delta):
            w.setStyleSheet("color:#333; font-size:14px;")
            layout.addWidget(w)
        # Risk list table
        self.risk_table = QTableWidget(0, 6)
        self.risk_table.setHorizontalHeaderLabels(["Student ID","Student Name","Rate","Consec Abs","Last Seen","Risk"]) 
        self.risk_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.risk_table.verticalHeader().setVisible(False)
        layout.addWidget(self.risk_table)
        # Actions
        actions = QHBoxLayout()
        refresh = QPushButton("Refresh Analytics")
        def do_refresh():
            s = compute_today_summary(self.db)
            self.lbl_present.setText(f"Present Today: {s['present_today']}")
            self.lbl_unique.setText(f"Unique Today: {s['unique_students_today']}")
            self.lbl_avg.setText(f"Avg Same Weekday (last 4): {s['avg_same_weekday_past4']}")
            self.lbl_delta.setText(f"Δ vs Avg: {s['delta_vs_avg']}")
            df = compute_risk_list(self.db, top_n=10)
            self.risk_table.setRowCount(len(df))
            for i, row in df.iterrows():
                self.risk_table.setItem(i, 0, QTableWidgetItem(str(row["student_id"])))
                self.risk_table.setItem(i, 1, QTableWidgetItem(str(row["student_name"])))
                self.risk_table.setItem(i, 2, QTableWidgetItem(str(row["attendance_rate"])))
                self.risk_table.setItem(i, 3, QTableWidgetItem(str(row["consecutive_absences"])))
                self.risk_table.setItem(i, 4, QTableWidgetItem(str(row.get("last_seen", ""))))
                self.risk_table.setItem(i, 5, QTableWidgetItem(str(round(float(row["risk_score"]),3))))
        refresh.clicked.connect(do_refresh)
        actions.addWidget(refresh)
        actions.addStretch()
        layout.addLayout(actions)
        # initial
        do_refresh()


# ---------- Attendance Page ----------
class AttendancePage(QWidget):
    def __init__(self, main: MainWindow):
        super().__init__(main)
        self.main = main
        self.db = main.db
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel("Attendance Records")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        layout.addWidget(title)

        # Filters
        filters = QGroupBox("Filters")
        fl = QGridLayout(filters)
        # Row 0: ID, Name, Status
        self.filter_sid = QLineEdit()
        self.filter_sname = QLineEdit()
        self.filter_status = QComboBox(); self.filter_status.addItems(["", "Present", "Absent"])
        fl.addWidget(QLabel("Student ID"), 0, 0); fl.addWidget(self.filter_sid, 0, 1)
        fl.addWidget(QLabel("Student Name"), 0, 2); fl.addWidget(self.filter_sname, 0, 3)
        fl.addWidget(QLabel("Status"), 0, 4); fl.addWidget(self.filter_status, 0, 5)
        # Row 1: Course, Year, Date From, Date To
        self.filter_course = QComboBox(); self.filter_course.setEditable(True)
        self.filter_year = QComboBox(); self.filter_year.setEditable(True)
        self.filter_date_from = QLineEdit(); self.filter_date_from.setPlaceholderText("YYYY-MM-DD")
        self.filter_date_to = QLineEdit(); self.filter_date_to.setPlaceholderText("YYYY-MM-DD")
        # populate course/year options
        try:
            courses = [""] + self.db.get_distinct_courses()
            years = [""] + [str(y) for y in self.db.get_distinct_years()]
            self.filter_course.addItems(courses)
            self.filter_year.addItems(years)
        except Exception:
            pass
        fl.addWidget(QLabel("Course"), 1, 0); fl.addWidget(self.filter_course, 1, 1)
        fl.addWidget(QLabel("Year"), 1, 2); fl.addWidget(self.filter_year, 1, 3)
        fl.addWidget(QLabel("Date From"), 1, 4); fl.addWidget(self.filter_date_from, 1, 5)
        fl.addWidget(QLabel("Date To"), 1, 6); fl.addWidget(self.filter_date_to, 1, 7)
        # Buttons
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.refresh_table)
        fl.addWidget(apply_btn, 0, 6)
        layout.addWidget(filters)

        # Table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["ID", "Student ID", "Student Name", "Date", "Time", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.table)

        # Actions
        actions = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_table)
        actions.addWidget(refresh_btn)
        export_csv_btn = QPushButton("Export CSV (Filtered)")
        export_csv_btn.clicked.connect(self.export_csv)
        actions.addWidget(export_csv_btn)
        export_xlsx_btn = QPushButton("Export Excel (Filtered)")
        export_xlsx_btn.clicked.connect(self.export_xlsx)
        actions.addWidget(export_xlsx_btn)
        split_csv_btn = QPushButton("Export Split CSV (Present/Absent)")
        split_csv_btn.clicked.connect(self.export_split_csv)
        actions.addWidget(split_csv_btn)
        split_xlsx_btn = QPushButton("Export Split Excel (Sheets)")
        split_xlsx_btn.clicked.connect(self.export_split_xlsx)
        actions.addWidget(split_xlsx_btn)
        # Export today's attendance (quick)
        export_today_csv_btn = QPushButton("Export Today's Attendance (CSV)")
        export_today_csv_btn.clicked.connect(self.export_today_csv)
        actions.addWidget(export_today_csv_btn)
        export_today_xlsx_btn = QPushButton("Export Today's Attendance (Excel)")
        export_today_xlsx_btn.clicked.connect(self.export_today_xlsx)
        actions.addWidget(export_today_xlsx_btn)
        actions.addStretch()
        layout.addLayout(actions)

        self.refresh_table()

    def refresh_table(self) -> None:
        self.table.setRowCount(0)
        sid = self.filter_sid.text().strip() or None
        sname = self.filter_sname.text().strip() or None
        status = self.filter_status.currentText() or None
        course = (self.filter_course.currentText().strip() or None)
        year_txt = self.filter_year.currentText().strip()
        year = int(year_txt) if year_txt.isdigit() else None
        dfrom = self.filter_date_from.text().strip() or None
        dto = self.filter_date_to.text().strip() or None
        try:
            records = self.db.list_attendance_filtered(
                student_id=sid,
                student_name=sname,
                course=course,
                year_of_study=year,
                status=status,
                date_from=dfrom,
                date_to=dto,
            )
        except TypeError:
            records = self.db.list_attendance_filtered()
        self.table.setRowCount(len(records))
        for i, r in enumerate(records):
            self.table.setItem(i, 0, QTableWidgetItem(str(r.id)))
            self.table.setItem(i, 1, QTableWidgetItem(r.student_id))
            self.table.setItem(i, 2, QTableWidgetItem(r.student_name))
            self.table.setItem(i, 3, QTableWidgetItem(r.date))
            self.table.setItem(i, 4, QTableWidgetItem(r.time))
            self.table.setItem(i, 5, QTableWidgetItem(r.status))

    def export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Attendance", "attendance.csv", "CSV Files (*.csv)")
        if not path:
            return
        # Export filtered view as CSV by building DataFrame manually
        sid = self.filter_sid.text().strip() or None
        sname = self.filter_sname.text().strip() or None
        status = self.filter_status.currentText() or None
        course = (self.filter_course.currentText().strip() or None)
        year_txt = self.filter_year.currentText().strip()
        year = int(year_txt) if year_txt.isdigit() else None
        dfrom = self.filter_date_from.text().strip() or None
        dto = self.filter_date_to.text().strip() or None
        recs = self.db.list_attendance_filtered(student_id=sid, student_name=sname, course=course, year_of_study=year, status=status, date_from=dfrom, date_to=dto)
        import pandas as pd
        df = pd.DataFrame([{ 'id': r.id, 'student_id': r.student_id, 'student_name': r.student_name, 'date': r.date, 'time': r.time, 'status': r.status } for r in recs])
        if df.empty:
            open(path, 'w', encoding='utf-8').write("")
            out = path
        else:
            out = path if path.lower().endswith('.csv') else path + '.csv'
            df.to_csv(out, index=False)
        info_popup(self, "Exported", f"Saved to: {out}")

    def export_xlsx(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Attendance", "attendance.xlsx", "Excel Files (*.xlsx)")
        if not path:
            return
        # Export filtered view as Excel
        sid = self.filter_sid.text().strip() or None
        sname = self.filter_sname.text().strip() or None
        status = self.filter_status.currentText() or None
        course = (self.filter_course.currentText().strip() or None)
        year_txt = self.filter_year.currentText().strip()
        year = int(year_txt) if year_txt.isdigit() else None
        dfrom = self.filter_date_from.text().strip() or None
        dto = self.filter_date_to.text().strip() or None
        recs = self.db.list_attendance_filtered(student_id=sid, student_name=sname, course=course, year_of_study=year, status=status, date_from=dfrom, date_to=dto)
        import pandas as pd
        df = pd.DataFrame([{ 'id': r.id, 'student_id': r.student_id, 'student_name': r.student_name, 'date': r.date, 'time': r.time, 'status': r.status } for r in recs])
        if df.empty:
            # Create an empty Excel file
            pd.DataFrame().to_excel(path, index=False)
        else:
            out = path if path.lower().endswith('.xlsx') else path + '.xlsx'
            df.to_excel(out, index=False)
        info_popup(self, "Exported", f"Saved to: {out if 'out' in locals() else path}")

    def export_split_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Attendance (Split)", "attendance_split.csv", "CSV Files (*.csv)")
        if not path:
            return
        sid = self.filter_sid.text().strip() or None
        sname = self.filter_sname.text().strip() or None
        status = None  # ignored when split_by_status=True
        course = (self.filter_course.currentText().strip() or None)
        year_txt = self.filter_year.currentText().strip()
        year = int(year_txt) if year_txt.isdigit() else None
        dfrom = self.filter_date_from.text().strip() or None
        dto = self.filter_date_to.text().strip() or None
        paths = self.db.export_attendance_filtered(path, fmt="csv", split_by_status=True,
            student_id=sid, student_name=sname, course=course, year_of_study=year, status=status, date_from=dfrom, date_to=dto)
        info_popup(self, "Exported", f"Saved to:\nPresent: {paths['present']}\nAbsent: {paths['absent']}")

    def export_split_xlsx(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Attendance (Split)", "attendance_split.xlsx", "Excel Files (*.xlsx)")
        if not path:
            return
        sid = self.filter_sid.text().strip() or None
        sname = self.filter_sname.text().strip() or None
        status = None
        course = (self.filter_course.currentText().strip() or None)
        year_txt = self.filter_year.currentText().strip()
        year = int(year_txt) if year_txt.isdigit() else None
        dfrom = self.filter_date_from.text().strip() or None
        dto = self.filter_date_to.text().strip() or None
        out = self.db.export_attendance_filtered(path, fmt="xlsx", split_by_status=True,
            student_id=sid, student_name=sname, course=course, year_of_study=year, status=status, date_from=dfrom, date_to=dto)
        info_popup(self, "Exported", f"Saved to: {out}")

    def export_today_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Today's Attendance", "attendance_today.csv", "CSV Files (*.csv)")
        if not path:
            return
        out = self.db.export_today_attendance(path, fmt="csv")
        info_popup(self, "Exported", f"Saved to: {out}")

    def export_today_xlsx(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Today's Attendance", "attendance_today.xlsx", "Excel Files (*.xlsx)")
        if not path:
            return
        out = self.db.export_today_attendance(path, fmt="xlsx")
        info_popup(self, "Exported", f"Saved to: {out}")

class ReportsPage(QWidget):
    def __init__(self, main: MainWindow):
        super().__init__(main)
        self.main = main
        self.db = main.db
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel("Reports & Analytics")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        layout.addWidget(title)

        # Matplotlib Figure
        self.figure = Figure(figsize=(10, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Controls
        controls = QHBoxLayout()
        self.topn = QComboBox(); self.topn.addItems(["5","10","15","20"]) ; self.topn.setCurrentText("10")
        refresh = QPushButton("Refresh Chart")
        refresh.clicked.connect(self.refresh_chart)
        controls.addWidget(QLabel("Top N at Risk:"))
        controls.addWidget(self.topn)
        controls.addWidget(refresh)
        controls.addStretch()
        layout.addLayout(controls)

        # Single-student quick report
        search_row = QHBoxLayout()
        self.stu_query = QLineEdit(); self.stu_query.setPlaceholderText("Student ID or Name")
        btn_search = QPushButton("Student Report")
        btn_search.clicked.connect(self._open_student_report)
        search_row.addWidget(QLabel("Find Student:"))
        search_row.addWidget(self.stu_query)
        search_row.addWidget(btn_search)
        search_row.addStretch()
        layout.addLayout(search_row)

        self.refresh_chart()

    def refresh_chart(self) -> None:
        from insight import compute_risk_list
        from datetime import date as _date
        n = int(self.topn.currentText())
        df = compute_risk_list(self.db, top_n=n)
        # Figure.clear() returns None, so do not chain add_subplot on it
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)

        # Left: Donut chart for today's present vs absent
        today = _date.today().isoformat()
        df_today = self.db.filtered_attendance_dataframe(date_from=today, date_to=today)
        present_count = int(df_today[df_today["status"] == "Present"]["student_id"].nunique()) if not df_today.empty else 0
        roster = self.db.roster_dataframe()
        total_students = int(len(roster)) if not roster.empty else int(df_today["student_id"].nunique()) if not df_today.empty else 0
        absent_count = max(total_students - present_count, 0)
        if total_students > 0:
            vals = [present_count, absent_count]
            labels = ["Present", "Absent"]
            colors = ["#2ecc71", "#e74c3c"]
            wedges, _ = ax1.pie(vals, labels=labels, colors=colors, startangle=90, wedgeprops=dict(width=0.4))
            ax1.set_title("Today's Attendance")
        else:
            ax1.text(0.5, 0.5, "No data today", ha='center', va='center')

        # Right: Bar chart for at-risk students
        if df.empty:
            ax2.text(0.5, 0.5, "No data", ha='center', va='center')
        else:
            names = df['student_name'].astype(str).tolist()
            pct = (df['attendance_rate'].astype(float).rsub(1).abs() * 100.0).tolist()  # risk % proxy: 100*(1-rate)
            ax2.barh(range(len(names)), pct, color="#e74c3c")
            ax2.set_yticks(range(len(names)))
            ax2.set_yticklabels(names)
            ax2.invert_yaxis()
            ax2.set_xlabel("Risk % (lower attendance)")
            ax2.set_title("Top At-Risk Students")
        self.canvas.draw_idle()

    def _open_student_report(self) -> None:
        query = self.stu_query.text().strip()
        if not query:
            error_popup(self, "Input Error", "Enter a Student ID or Name.")
            return
        # Prefer ID match; fallback to name
        recs = self.db.list_attendance_filtered(student_id=query)
        if not recs:
            recs = self.db.list_attendance_filtered(student_name=query)
        if not recs:
            info_popup(self, "No Results", "No attendance records found for that student.")
            return
        # Build a simple dialog with summary and last 20 records
        dlg = QDialog(self)
        dlg.setWindowTitle("Student Attendance Report")
        v = QVBoxLayout(dlg)
        sid = recs[0].student_id; sname = recs[0].student_name
        # compute summary
        import collections
        cnt = collections.Counter([r.status for r in recs])
        total = len(recs)
        present = cnt.get("Present", 0)
        rate = (present / total * 100.0) if total else 0.0
        v.addWidget(QLabel(f"{sid} - {sname}"))
        v.addWidget(QLabel(f"Records: {total} | Present: {present} | Rate: {rate:.1f}%"))
        table = QTableWidget(0, 5)
        table.setHorizontalHeaderLabels(["ID","Date","Time","Status","Name"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        rows = recs[:20]
        table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(str(r.id)))
            table.setItem(i, 1, QTableWidgetItem(r.date))
            table.setItem(i, 2, QTableWidgetItem(r.time))
            table.setItem(i, 3, QTableWidgetItem(r.status))
            table.setItem(i, 4, QTableWidgetItem(r.student_name))
        v.addWidget(table)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        v.addWidget(btns)
        dlg.exec_()

    def export_xlsx(self) -> None:
        # Export roster analytics (with derived metrics) to Excel
        path, _ = QFileDialog.getSaveFileName(self, "Export Roster Analytics", "roster.xlsx", "Excel Files (*.xlsx)")
        if not path:
            return
        df = self.db.roster_dataframe()
        out = path if path.lower().endswith('.xlsx') else path + '.xlsx'
        if df.empty:
            # create empty workbook
            import pandas as pd
            pd.DataFrame().to_excel(out, index=False, engine='openpyxl')
        else:
            df.to_excel(out, index=False, engine='openpyxl')
        info_popup(self, "Exported", f"Saved to: {out}")


# ---------- Face Registration Page ----------
class RegistrationPage(QWidget):
    def __init__(self, main: MainWindow):
        super().__init__(main)
        self.main = main
        self.db = main.db
        self.creator = main.creator
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel("Face Registration")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        layout.addWidget(title)

        btn = QPushButton("Register a new student")
        btn.clicked.connect(self.open_registration_dialog)
        layout.addWidget(btn)

        start_att = QPushButton("Start Attendance Session")
        start_att.clicked.connect(self.start_attendance)
        layout.addWidget(start_att)

        layout.addStretch()

    def open_registration_dialog(self) -> None:
        dlg = RegistrationDialog(self.main)
        if dlg.exec_() == QDialog.Accepted:
            info_popup(self, "Success", "Student registered and embeddings stored.")

    def start_attendance(self) -> None:
        info_popup(self, "Starting", "Opening camera for attendance. Press 'q' in the window to stop.")
        # Run in background thread to keep UI responsive
        if self.main.attendance_thread and self.main.attendance_thread.is_alive():
            info_popup(self, "Already Running", "An attendance session is already running.")
            return
        self.main.attendance_thread = threading.Thread(target=self.main.recognizer.run, args=(self.db,), daemon=True)
        self.main.attendance_thread.start()


class RegistrationDialog(QDialog):
    def __init__(self, main: MainWindow):
        super().__init__(main)
        self.main = main
        self.db = main.db
        self.creator = main.creator

        self.setWindowTitle("Register New Student")
        self.setMinimumWidth(420)

        form = QFormLayout(self)
        self.sid = QLineEdit()
        self.name = QLineEdit()
        # Course and Year as dropdowns with suggestions
        self.course = QComboBox(); self.course.setEditable(True)
        self.year = QComboBox(); self.year.setEditable(True)
        try:
            courses = self.db.get_distinct_courses()
            years = [str(y) for y in self.db.get_distinct_years()]
            for c in courses:
                if c:
                    self.course.addItem(c)
            for y in years:
                if y:
                    self.year.addItem(y)
        except Exception:
            pass
        form.addRow("Student ID:", self.sid)
        form.addRow("Student Name:", self.name)
        form.addRow("Course:", self.course)
        form.addRow("Year of Study:", self.year)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _on_ok(self) -> None:
        student_id = self.sid.text().strip()
        student_name = self.name.text().strip()
        course = (self.course.currentText().strip() or None)
        year_val = self.year.currentText().strip()
        year = int(year_val) if year_val.isdigit() else None

        if not student_id or not student_name:
            error_popup(self, "Input Error", "Student ID and Name are required.")
            return

        # Ensure student exists in DB
        existing = self.db.get_student(student_id)
        if not existing:
            self.db.add_student(student_id, student_name, course, year)

        # Capture -> Encode -> Store (in-memory, no files on disk)
        stored = self.creator.capture_encode_and_store(self.db, student_id, student_name)
        if stored <= 0:
            error_popup(self, "Embedding Error", "Failed to create/store embeddings.")
            self.reject()
            return
        self.accept()


# ---------- Settings Page ----------
class SettingsPage(QWidget):
    def __init__(self, main: MainWindow):
        super().__init__(main)
        self.main = main
        self.db = main.db
        self.creator = main.creator
        layout = QVBoxLayout(self)
        title = QLabel("Settings")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        layout.addWidget(title)
        layout.addWidget(QLabel("Utilities"))
        # Buttons
        btn_seed_students = QPushButton("Seed 50 Sample Students")
        btn_seed_att = QPushButton("Seed Attendance (last 3 days)")
        btn_clean_dataset = QPushButton("Clean Dataset Folder (privacy)")
        btn_seed_students.clicked.connect(self._seed_students)
        btn_seed_att.clicked.connect(self._seed_attendance)
        btn_clean_dataset.clicked.connect(self._clean_dataset)
        layout.addWidget(btn_seed_students)
        layout.addWidget(btn_seed_att)
        layout.addWidget(btn_clean_dataset)

        # Camera selection
        cam_group = QGroupBox("Camera Settings")
        gl = QGridLayout(cam_group)
        self.cam_index = QComboBox()
        for i in range(0, 8):
            self.cam_index.addItem(str(i))
        self.cam_index.setCurrentText(str(self.main.recognizer.camera_index))
        btn_apply_cam = QPushButton("Apply Camera Index")
        btn_test_cam = QPushButton("Test Preview (5s)")
        btn_apply_cam.clicked.connect(self._apply_camera_index)
        btn_test_cam.clicked.connect(self._test_camera_preview)
        gl.addWidget(QLabel("Camera Index"), 0, 0)
        gl.addWidget(self.cam_index, 0, 1)
        gl.addWidget(btn_apply_cam, 0, 2)
        gl.addWidget(btn_test_cam, 1, 1)
        layout.addWidget(cam_group)
        layout.addStretch()

    def _seed_students(self) -> None:
        n = self.db.seed_sample_students(50)
        info_popup(self, "Seeding", f"Inserted {n} sample students.")

    def _seed_attendance(self) -> None:
        n = self.db.seed_sample_attendance(days=3, present_rate=0.7)
        info_popup(self, "Seeding", f"Inserted {n} attendance rows.")

    def _clean_dataset(self) -> None:
        ok = self.creator.clean_dataset_dir()
        if ok:
            info_popup(self, "Cleanup", "Dataset folder cleaned or not present.")
        else:
            error_popup(self, "Cleanup Error", "Failed to clean dataset folder.")

    def _apply_camera_index(self) -> None:
        try:
            idx = int(self.cam_index.currentText())
            self.main.recognizer.camera_index = idx
            info_popup(self, "Camera", f"Camera index set to {idx}. If using an iPhone USB webcam app, select its device in Windows and note its index here.")
        except ValueError:
            error_popup(self, "Input Error", "Invalid camera index.")

    def _test_camera_preview(self) -> None:
        import cv2
        idx_txt = self.cam_index.currentText().strip()
        if not idx_txt.isdigit():
            error_popup(self, "Input Error", "Camera index must be a number.")
            return
        idx = int(idx_txt)
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            error_popup(self, "Camera Error", f"Could not open camera at index {idx}.")
            return
        info_popup(self, "Preview", "Showing preview for ~5 seconds. Press 'q' to close earlier.")
        import time
        start = time.time()
        while time.time() - start < 5:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            cv2.imshow("Camera Preview", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


# ---------- Entrypoint ----------
def launch_app(db_path: str = "attendance.db") -> None:
    app = QApplication(sys.argv)
    db = DatabaseManager(db_path)
    window = MainWindow(db)
    window.show()
    sys.exit(app.exec_())
