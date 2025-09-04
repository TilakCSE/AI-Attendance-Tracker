"""
Backwards-compatible wrapper for the GUI layer.

This module now forwards to the PyQt5 implementation in `gui_qt.py` to provide
an error-free, modern UI while keeping the same public entrypoint:

    launch_app(db_path: str = "attendance.db") -> None
"""

from __future__ import annotations

from gui_qt import launch_app  # re-export the PyQt5 entrypoint
"""
        ttk.Label(filter_frame, text="Status").grid(row=0, column=4, sticky="w", padx=4, pady=2)
        status_cb = ttk.Combobox(filter_frame, textvariable=self.filter_status_var, values=["", "Present", "Absent"], state="readonly")
        status_cb.grid(row=0, column=5, sticky="w", padx=4, pady=2)

        ttk.Label(filter_frame, text="Course").grid(row=1, column=4, sticky="w", padx=4, pady=2)
        self.course_cb = ttk.Combobox(filter_frame, textvariable=self.filter_course_var, state="readonly")
        self.course_cb.grid(row=1, column=5, sticky="w", padx=4, pady=2)
        self.course_cb['values'] = [""] + self.db.get_distinct_courses()

        ttk.Label(filter_frame, text="Year").grid(row=1, column=6, sticky="w", padx=4, pady=2)
        self.year_cb = ttk.Combobox(filter_frame, textvariable=self.filter_year_var, state="readonly")
        self.year_cb.grid(row=1, column=7, sticky="w", padx=4, pady=2)
        self.year_cb['values'] = [""] + self.db.get_distinct_years()

        ttk.Button(filter_frame, text="Apply Filters", command=self._refresh_table).grid(row=1, column=8, padx=4, pady=2, sticky="e")
        ttk.Button(filter_frame, text="Clear Filters", command=self._clear_filters).grid(row=1, column=9, padx=4, pady=2, sticky="w")

        # Records area
        table_frame = ttk.LabelFrame(self.root, text="Attendance Records", padding=10)
        table_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        columns = ("id", "student_id", "student_name", "date", "time", "status")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        for col in columns:
            self.tree.heading(col, text=col)
            width = 120 if col not in ("student_name", "id") else (220 if col == "student_name" else 60)
            self.tree.column(col, width=width, anchor="center")
        self.tree.pack(side=LEFT, fill=BOTH, expand=True)

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=X, padx=10, pady=(0, 10))
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=LEFT)

        # Initial load
        self._refresh_table()


    # -------- Register Student --------

    def _open_register_dialog(self) -> None:
        dialog = Toplevel(self.root)
        dialog.title("Register Student")
        dialog.geometry("420x260")
        dialog.transient(self.root)
        dialog.grab_set()

        sid_var = StringVar()
        name_var = StringVar()
        count_var = IntVar(value=20)
        degree_var = StringVar(value="Bachelor")
        course_var = StringVar()
        year_var = StringVar()

        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill=BOTH, expand=True)

        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Student ID:").grid(row=0, column=0, sticky="w", pady=6, padx=(0, 6))
        sid_entry = ttk.Entry(frame, textvariable=sid_var)
        sid_entry.grid(row=0, column=1, sticky="ew", pady=6)

        ttk.Label(frame, text="Student Name:").grid(row=1, column=0, sticky="w", pady=6, padx=(0, 6))
        name_entry = ttk.Entry(frame, textvariable=name_var)
        name_entry.grid(row=1, column=1, sticky="ew", pady=6)

        # Degree level
        ttk.Label(frame, text="Degree Level:").grid(row=2, column=0, sticky="w", pady=6, padx=(0, 6))
        degree_cb = ttk.Combobox(frame, textvariable=degree_var, values=["Bachelor", "Master"], state="readonly")
        degree_cb.grid(row=2, column=1, sticky="ew", pady=6)

        # Course selection (depends on degree)
        bachelor_courses = ["B.Tech CSE", "B.Tech ECE", "BSc CS", "BBA", "BA"]
        master_courses = ["M.Tech CSE", "MSc CS", "MBA", "MA"]

        def _update_course_values(*_):
            vals = bachelor_courses if degree_var.get() == "Bachelor" else master_courses
            course_cb["values"] = vals
            # Reset selection when degree changes
            if vals:
                course_var.set(vals[0])
            else:
                course_var.set("")

        ttk.Label(frame, text="Course:").grid(row=3, column=0, sticky="w", pady=6, padx=(0, 6))
        course_cb = ttk.Combobox(frame, textvariable=course_var, values=[], state="readonly")
        course_cb.grid(row=3, column=1, sticky="ew", pady=6)
        degree_cb.bind("<<ComboboxSelected>>", _update_course_values)
        _update_course_values()

        # Year of study (1..5)
        ttk.Label(frame, text="Year of Study:").grid(row=4, column=0, sticky="w", pady=6, padx=(0, 6))
        year_cb = ttk.Combobox(frame, textvariable=year_var, values=["1", "2", "3", "4", "5"], state="readonly")
        year_cb.grid(row=4, column=1, sticky="ew", pady=6)

        ttk.Label(frame, text="Images to Capture:").grid(row=5, column=0, sticky="w", pady=6, padx=(0, 6))
        count_entry = ttk.Entry(frame, textvariable=count_var)
        count_entry.grid(row=5, column=1, sticky="ew", pady=6)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=12)

        def start_capture() -> None:
            student_id = sid_var.get().strip()
            student_name = name_var.get().strip()
            try:
                num_images = int(count_var.get())
            except Exception:
                num_images = 20

            if not student_id or not student_name:
                messagebox.showerror("Validation Error", "Please enter both Student ID and Student Name.")
                return
            if num_images <= 0:
                messagebox.showerror("Validation Error", "Images to Capture must be a positive number.")
                return

            def work():
                try:
                    self._set_status(f"Registering {student_id} - {student_name}...")
                    # parse year (optional)
                    try:
                student_id=v_sid.get().strip() or None,
                student_name=v_name.get().strip() or None,
                degree_level=v_degree.get().strip() or None,
                course=v_course.get().strip() or None,
                year_of_study=year_val,
                is_active=active_val,
            )
            for s in students:
                values = (s.student_id, s.student_name, s.degree_level or "", s.course or "", s.year_of_study or "", "Yes" if s.is_active else "No")
                table.insert("", "end", values=values)

        # Footer actions
        foot = ttk.Frame(win)
        foot.pack(fill=X, padx=10, pady=(0,10))
        ttk.Button(foot, text="Export Filtered Roster (CSV)", command=lambda: export_filtered("csv")).pack(side=LEFT, padx=6)
        ttk.Button(foot, text="Export Filtered Roster (Excel)", command=lambda: export_filtered("xlsx")).pack(side=LEFT, padx=6)

        def export_filtered(fmt: str):
            from tkinter import filedialog
            path = filedialog.asksaveasfilename(defaultextension=f".{fmt}", filetypes=[(fmt.upper(), f"*.{fmt}")], initialfile=f"roster_filtered.{fmt}")
            if not path:
                return
            filtered = {
                "student_id": v_sid.get().strip() or None,
                "student_name": v_name.get().strip() or None,
                "degree_level": v_degree.get().strip() or None,
                "course": v_course.get().strip() or None,
                "year_of_study": (int(v_year.get()) if v_year.get().strip() else None),
                "campus": v_campus.get().strip() or None,
                "is_active": {"All": None, "Active": 1, "Inactive": 0}.get(v_active.get(), None),
            }
            out = self.db.export_complete_roster(path, fmt=fmt, deidentify=False, filtered=filtered)
            messagebox.showinfo("Export Complete", f"Saved to:\n{out}")

        refresh()

    def _open_reports_window(self) -> None:
        from insight import compute_today_summary, compute_risk_list

        win = Toplevel(self.root)
        win.title("Reports & Insights")
        win.geometry("900x620")
        win.transient(self.root)

        # Summary cards (simple labels)
        cards = ttk.Frame(win, padding=10)
        cards.pack(fill=X)

        summary = compute_today_summary(self.db)
        ttk.Label(cards, text=f"Present Today: {summary['present_today']}", font=("Segoe UI", 12, "bold")).pack(side=LEFT, padx=12)
        ttk.Label(cards, text=f"Unique Today: {summary['unique_students_today']}", font=("Segoe UI", 12)).pack(side=LEFT, padx=12)
        ttk.Label(cards, text=f"Avg Same Weekday (last 4): {summary['avg_same_weekday_past4']}", font=("Segoe UI", 12)).pack(side=LEFT, padx=12)
        delta = summary["delta_vs_avg"]
        ttk.Label(cards, text=f"Δ vs Avg: {delta:+.2f}", font=("Segoe UI", 12)).pack(side=LEFT, padx=12)

        # Risk list
        risk_frame = ttk.LabelFrame(win, text="At-Risk (Top 10)", padding=10)
        risk_frame.pack(fill=BOTH, expand=True, padx=10, pady=(6, 10))

        tree = ttk.Treeview(risk_frame, columns=("student_id","student_name","rate","consec_abs","last_seen","risk"), show="headings")
        for col, w in [("student_id",120),("student_name",200),("rate",100),("consec_abs",120),("last_seen",120),("risk",100)]:
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor="center")
        tree.pack(fill=BOTH, expand=True, side=LEFT)
        sb = ttk.Scrollbar(risk_frame, orient="vertical", command=tree.yview)
        tree.configure(yscroll=sb.set)
        sb.pack(side=RIGHT, fill=Y)

        df = compute_risk_list(self.db, top_n=10)
        for _, r in df.iterrows():
            tree.insert("", "end", values=(r["student_id"], r["student_name"], r["attendance_rate"], r["consecutive_absences"], r["last_seen"] or "", round(float(r["risk_score"]),3)))

        # Export Complete Roster
        export_frame = ttk.LabelFrame(win, text="Complete Roster Export", padding=10)
        export_frame.pack(fill=X, padx=10, pady=(0, 10))
        ttk.Label(export_frame, text="Format:").pack(side=LEFT, padx=(0,6))
        fmt_var = StringVar(value="csv")
        fmt_cb = ttk.Combobox(export_frame, textvariable=fmt_var, values=["csv","xlsx","json","parquet"], state="readonly", width=10)
        fmt_cb.pack(side=LEFT, padx=6)
        deid_var = IntVar(value=0)
        ttk.Checkbutton(export_frame, text="De-identify (hash IDs, redact names)", variable=deid_var).pack(side=LEFT, padx=12)
        ttk.Button(export_frame, text="Export Complete Roster", command=lambda: export_roster(fmt_var.get(), bool(deid_var.get()))).pack(side=LEFT, padx=6)

        def export_roster(fmt: str, deid: bool):
            from tkinter import filedialog
            path = filedialog.asksaveasfilename(defaultextension=f".{fmt}", filetypes=[(fmt.upper(), f"*.{fmt}")], initialfile=f"complete_roster.{fmt}")
            if not path:
                return
            out = self.db.export_complete_roster(path, fmt=fmt, deidentify=deid, filtered=None)
            messagebox.showinfo("Export Complete", f"Saved to:\n{out}")

    def _on_close(self) -> None:
        if self.attendance_thread is not None and self.attendance_thread.is_alive():
            if not messagebox.askyesno(
                "Quit",
                "An attendance session is running.\nAre you sure you want to quit?\nTip: Press 'q' in the camera window to end the session.",
            ):
                return
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def launch_app(db_path: str = "attendance.db") -> None:
    db = DatabaseManager(db_path)
    app = AttendanceApp(db)
    app.run()
"""
