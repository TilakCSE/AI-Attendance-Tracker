"""
attendance_recognition.py

Module to recognize registered students in real time using face_recognition and OpenCV,
and mark attendance in the SQLite database without duplicates per session.

Author: Your Name
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple, Dict, Set, Optional

import cv2
import numpy as np
import face_recognition

from database import DatabaseManager
from security import get_fernet, decrypt_embedding_array

import logging

logger = logging.getLogger(__name__)


class AttendanceRecognizer:
    def __init__(
        self,
        dataset_dir: str = "dataset",
        camera_index: int = 0,
        tolerance: float = 0.5,
        model: str = "hog",  # "hog" for CPU, "cnn" if dlib with CUDA is available
        enable_liveness: bool = True,
    ) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = os.path.join(base_dir, dataset_dir)
        self.camera_index = camera_index
        self.tolerance = tolerance
        self.model = model
        self.enable_liveness = enable_liveness

        self.known_encodings: List[np.ndarray] = []
        self.known_labels: List[str] = []  # label like "studentid_studentname"
        self.label_to_meta: Dict[str, Tuple[str, str]] = {}  # label -> (student_id, student_name)

    # ---------- Liveness helpers ----------

    @staticmethod
    def _euclidean(p1, p2) -> float:
        return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))

    def _eye_aspect_ratio(self, eye_pts: List[tuple]) -> float:
        # eye_pts: list of 6 (x, y) tuples
        if len(eye_pts) < 6:
            return 0.0
        p1, p2, p3, p4, p5, p6 = eye_pts[:6]
        A = self._euclidean(p2, p6)
        B = self._euclidean(p3, p5)
        C = self._euclidean(p1, p4)
        if C <= 1e-6:
            return 0.0
        return (A + B) / (2.0 * C)

    def _mouth_aspect_ratio(self, mouth_pts: List[tuple]) -> float:
        # Simple approximation of mouth "smiling" by vertical distance of lips
        if len(mouth_pts) < 8:
            return 0.0
        top_lip = np.mean(mouth_pts[2:5], axis=0)
        bottom_lip = np.mean(mouth_pts[8:11], axis=0)
        return float(np.linalg.norm(top_lip - bottom_lip))

    def _run_liveness_check(self, cap: cv2.VideoCapture, timeout_sec: int = 20) -> bool:
        """
        Require both a blink and a smile within timeout to verify liveness.
        This blocks briefly (<= timeout_sec) while reading frames from the same camera.
        """
        EAR_THRESH = 0.21
        EAR_CONSEC_FRAMES = 2
        MAR_THRESH = 7.0  # Threshold for mouth aspect ratio indicating smile/open mouth
        ear_below_count = 0
        blink_done = False
        smile_done = False
        start = time.time()

        while (time.time() - start) < timeout_sec:
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks_list = face_recognition.face_landmarks(rgb, model="large")
            
            if landmarks_list:
                lm = landmarks_list[0]
                left_eye = lm.get("left_eye", [])
                right_eye = lm.get("right_eye", [])
                mouth = lm.get("top_lip", []) + lm.get("bottom_lip", [])
                
                # Eye aspect ratio for blink detection
                ear_left = self._eye_aspect_ratio(left_eye) if left_eye else 1.0
                ear_right = self._eye_aspect_ratio(right_eye) if right_eye else 1.0
                ear = (ear_left + ear_right) / 2.0

                if ear < EAR_THRESH:
                    ear_below_count += 1
                else:
                    if ear_below_count >= EAR_CONSEC_FRAMES:
                        blink_done = True
                    ear_below_count = 0
                
                # Mouth aspect ratio for smile detection
                mar = self._mouth_aspect_ratio(mouth) if mouth else 0.0
                if mar > MAR_THRESH:
                    smile_done = True
                
                # Display status
                info_text = f"Blink: {'✓' if blink_done else '✗'} | Smile: {'✓' if smile_done else '✗'}"
                cv2.putText(
                    frame, 
                    info_text, 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0) if blink_done and smile_done else (0, 255, 255), 
                    2
                )
                
                cv2.imshow("Attendance - Face Recognition", frame)
                
                # If both conditions are met, return success
                if blink_done and smile_done:
                    cv2.waitKey(500)  # Show success briefly
                    return True
            
            # Display instructions
            instruction = "Please BLINK and SMILE to verify liveness"
            cv2.putText(
                frame,
                instruction,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            
            cv2.imshow("Attendance - Face Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False

        # Timeout without completing both actions
        return False


    def load_known_faces(self, db: Optional[DatabaseManager] = None) -> int:
        """
        Load all known face encodings.
        Priority: encrypted embeddings from DB; fallback to dataset images.
        """
        self.known_encodings.clear()
        self.known_labels.clear()
        self.label_to_meta.clear()

        total_count = 0

        # Try database first
        if db is not None:
            try:
                name_map = db.get_student_name_map()
                emb_map = db.get_all_face_embeddings()
                fernet = get_fernet()
                for sid, blobs in emb_map.items():
                    sname = name_map.get(sid, sid)
                    label = f"{sid}_{sname.replace(' ', '_')}"
                    self.label_to_meta[label] = (sid, sname)
                    for blob in blobs:
                        try:
                            arr = decrypt_embedding_array(blob, fernet=fernet)
                            if arr is None or not isinstance(arr, np.ndarray):
                                continue
                            self.known_encodings.append(arr)
                            self.known_labels.append(label)
                            total_count += 1
                        except Exception:
                            continue
                if total_count > 0:
                    logger.info(f"Loaded {total_count} encrypted embeddings for {len(set(self.known_labels))} students from DB.")
                    return total_count
            except Exception as e:
                logger.warning(f"Warning: failed to load embeddings from DB, falling back to dataset. Detail: {e}")

        # Fallback to dataset images
        if not os.path.isdir(self.dataset_dir):
            logger.warning(f"Dataset directory not found at {self.dataset_dir}.")
            return 0

        for entry in os.listdir(self.dataset_dir):
            folder_path = os.path.join(self.dataset_dir, entry)
            if not os.path.isdir(folder_path):
                continue

            label = entry
            # Parse student_id and student_name from folder
            if "_" in entry:
                sid, sname = entry.split("_", 1)
            else:
                sid, sname = entry, entry
            sname = sname.replace("_", " ")
            self.label_to_meta[label] = (sid, sname)

            # Iterate images and compute encodings
            for file in os.listdir(folder_path):
                if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(folder_path, file)
                image = face_recognition.load_image_file(img_path)
                # Optionally resize large images to speed up encoding
                if max(image.shape[:2]) > 1600:
                    scale = 1600.0 / max(image.shape[:2])
                    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

                boxes = face_recognition.face_locations(image, model=self.model)
                encs = face_recognition.face_encodings(image, boxes)
                if not encs:
                    continue

                self.known_encodings.append(encs[0])
                self.known_labels.append(label)
                total_count += 1

        logger.info(f"Loaded {total_count} face encodings for {len(set(self.known_labels))} students (dataset fallback).")
        return total_count

    def run(self, db: DatabaseManager, duration_minutes: Optional[int] = None) -> None:
        """
        Start webcam, recognize faces, and mark attendance.
        Press 'q' to quit the attendance session.
        If duration_minutes is provided, the session auto-stops after that many minutes.
        """
        if not self.known_encodings:
            self.load_known_faces(db)

        if not self.known_encodings:
            logger.error("No known faces loaded. Please register students first.")
            return

        # Open default numeric camera index only
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error("Could not open webcam. Please check camera access and index.")
            return

        recognized_this_session: Set[str] = set()
        process_every_n = 1  # process every frame (no skipping) to stabilize multi-face UX
        frame_idx = 0
        end_time: Optional[float] = None
        if duration_minutes and duration_minutes > 0:
            end_time = time.time() + (duration_minutes * 60.0)

        try:
            while True:
                # Stop when duration elapsed
                if end_time is not None and time.time() >= end_time:
                    logger.info("Attendance session ended (duration elapsed).")
                    break
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera.")
                    break

                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Process every Nth frame (N=1 => every frame)
                frame_idx = (frame_idx + 1) % process_every_n

                # Detect faces and compute encodings on selected frames
                face_locations = face_recognition.face_locations(rgb_small, model=self.model)
                face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

                face_labels: List[str] = []
                for face_encoding in face_encodings:
                    # Compare against known encodings (original approach)
                    matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=self.tolerance)
                    name_label = "Unknown"

                    if True in matches:
                        # Select best match based on distance
                        face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                        best_match_index = int(np.argmin(face_distances))
                        if matches[best_match_index]:
                            name_label = self.known_labels[best_match_index]

                    face_labels.append(name_label)

                # Draw results and mark attendance
                scale_back = 4  # inverse of 0.25 resize
                for (top, right, bottom, left), label in zip(face_locations, face_labels):
                    # Scale back up face locations
                    top *= scale_back
                    right *= scale_back
                    bottom *= scale_back
                    left *= scale_back

                    if label == "Unknown":
                        color = (0, 0, 255)
                        display_text = "Unknown"
                    else:
                        sid, sname = self.label_to_meta.get(label, ("", label))
                        color = (0, 255, 0)
                        display_text = f"{sid} - {sname}"

                        # Avoid duplicates within this session
                        if sid and sid not in recognized_this_session:
                            # Run liveness verification (blink/smile) to prevent spoofing
                            liveness_ok = True
                            if self.enable_liveness:
                                liveness_ok = self._run_liveness_check(cap, timeout_sec=4)
                            if not liveness_ok:
                                logger.warning(
                                    f"Liveness failed or cancelled for {sid} - {sname}. Spoof suspected; not marking attendance.")
                            else:
                                # Mark attendance if not already marked today
                                created = db.mark_attendance(student_id=sid, student_name=sname, status="Present")
                                if created:
                                    logger.info(f"Attendance marked for {sid} - {sname}")
                                else:
                                    logger.info(f"Attendance already marked today for {sid} - {sname}")
                                recognized_this_session.add(sid)
                                recognized_this_session.add(sid)

                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
                    cv2.putText(
                        frame,
                        display_text,
                        (left + 6, bottom - 8),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.7,
                        (255, 255, 255),
                        1,
                    )

                cv2.putText(
                    frame,
                    "Press 'q' to end attendance session",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Attendance - Face Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Attendance session ended by user.")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Attendance recognition session ended.")
