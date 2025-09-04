"""
main.py

Entry point to launch the AI-Powered Attendance Tracker GUI.
"""

import os
import sys
import logging

# Add project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from gui_qt import launch_app


def main() -> None:
    """Entry point of the application."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    db_path = os.path.abspath(os.path.join(project_root, "data", "attendance.db"))
    print(f"[MAIN] Database path: {db_path}")
    print(f"[MAIN] Database exists: {os.path.exists(db_path)}")
    launch_app(db_path)


if __name__ == "__main__":
    main()
