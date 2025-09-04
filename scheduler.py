"""
scheduler.py

Handles the automated, schedule-driven attendance marking.
Runs in a background thread to check for upcoming classes and trigger
the recognition process automatically.
"""

import logging
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Set, Tuple

if TYPE_CHECKING:
    from attendance_recognition import AttendanceRecognizer
    from database import DatabaseManager

logger = logging.getLogger(__name__)


class AttendanceScheduler:
    """
    Manages the automated attendance process based on the class schedule.
    """

    def __init__(self, db: "DatabaseManager", recognizer: "AttendanceRecognizer") -> None:
        self.db = db
        self.recognizer = recognizer
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.run_history: Set[Tuple[str, int]] = set()  # (date, class_id)

    def start(self) -> None:
        """Start the scheduler in a background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Scheduler is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_schedule_check, daemon=True)
        self._thread.start()
        logger.info("Attendance scheduler started.")

    def stop(self) -> None:
        """Stop the scheduler thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Attendance scheduler stopped.")

    def _run_schedule_check(self) -> None:
        """The main loop that checks for classes and runs attendance."""
        while not self._stop_event.is_set():
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            today_date_str = now.strftime("%Y-%m-%d")

            try:
                classes_today = self.db.get_classes_for_today()
                for cls in classes_today:
                    class_id = cls["class_id"]
                    start_time = cls["start_time"]
                    end_time = cls.get("end_time")

                    if current_time == start_time and (today_date_str, class_id) not in self.run_history:
                        self.run_history.add((today_date_str, class_id))
                        logger.info(f"Scheduled class starting: {cls['subject_name']} with {cls['teacher_name']}.")

                        # Determine duration from start/end times (in minutes), fallback to 10
                        duration_minutes = 10
                        try:
                            if end_time:
                                # times are strings HH:MM
                                st_h, st_m = map(int, start_time.split(":"))
                                et_h, et_m = map(int, end_time.split(":"))
                                duration_minutes = max(1, (et_h * 60 + et_m) - (st_h * 60 + st_m))
                        except Exception:
                            pass

                        # Run attendance for computed duration, then mark absentees
                        self._run_attendance_for_class(class_id, cls['subject_name'], duration_minutes)

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")

            # Check every 60 seconds
            self._stop_event.wait(60)

    def _run_attendance_for_class(self, class_id: int, subject_name: str, duration_minutes: int) -> None:
        """Handle the process for a single class session."""
        # Run in a separate thread to avoid blocking the scheduler loop
        def work():
            try:
                logger.info(f"Starting {duration_minutes}-minute attendance recognition for {subject_name}...")
                # This is a blocking call that will run for the computed duration
                self.recognizer.run(self.db, duration_minutes=duration_minutes)
                logger.info(f"Attendance window for {subject_name} has closed.")

                logger.info(f"Marking absentees for {subject_name} (Class ID: {class_id})...")
                absent_count = self.db.mark_absentees_for_class(class_id)
                logger.info(f"Finished marking {absent_count} absentees for {subject_name}.")

            except Exception as e:
                logger.error(f"An error occurred during the attendance session for class {class_id}: {e}")
        
        threading.Thread(target=work, daemon=True).start()
