"""
dataset_creator.py

Module to register new students by capturing multiple face images
from the webcam and saving them to a dataset folder.

Images are stored under: dataset/{student_id}_{student_name}/img_<n>.jpg

Adds:
- Liveness challenge (blink + head turns) before capture to prevent spoofing.
- Helper to encode captured images into embeddings, encrypt, and store in DB.
"""

from __future__ import annotations

import os
import time
import shutil
from typing import Tuple, Optional, List

import cv2
import numpy as np
import face_recognition

from database import DatabaseManager
from security import get_fernet, encrypt_embedding_array


class DatasetCreator:
    def __init__(self, camera_index: int = 0, dataset_dir: str = "dataset") -> None:
        self.camera_index = camera_index
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = os.path.join(base_dir, dataset_dir)

    def _ensure_dir(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def _euclidean(self, p1, p2) -> float:
        return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))

    def _eye_aspect_ratio(self, eye_pts: List[tuple]) -> float:
        if len(eye_pts) < 6:
            return 0.0
        p1, p2, p3, p4, p5, p6 = eye_pts[:6]
        A = self._euclidean(p2, p6)
        B = self._euclidean(p3, p5)
        C = self._euclidean(p1, p4)
        if C <= 1e-6:
            return 0.0
        return (A + B) / (2.0 * C)

    def _run_liveness_challenge(self, cap: cv2.VideoCapture, window_name: str = "Liveness Check", timeout_sec: int = 20) -> bool:
        """
        Perform a simple liveness challenge:
        1) Blink once (detect by EAR drop)
        2) Turn head left
        3) Turn head right
        """
        EAR_THRESH = 0.21
        EAR_CONSEC_FRAMES = 2
        HEAD_MOVE_PIX = 12  # minimal horizontal movement in pixels (landmarks x)

        blink_done = False
        left_done = False
        right_done = False
        ear_below_count = 0
        baseline_x: Optional[float] = None

        start_time = time.time()

        while True:
            if (time.time() - start_time) > timeout_sec:
                print("Liveness: timeout.")
                return False

            ret, frame = cap.read()
            if not ret:
                print("Liveness: failed to read frame.")
                return False

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks_list = face_recognition.face_landmarks(rgb, model="large")

            instruction = ""
            if not blink_done:
                instruction = "Please BLINK"
            elif not left_done:
                instruction = "Turn HEAD LEFT"
            elif not right_done:
                instruction = "Turn HEAD RIGHT"
            else:
                # All done
                cv2.putText(frame, "Liveness PASSED", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(window_name, frame)
                cv2.waitKey(500)
                return True

            if landmarks_list:
                lm = landmarks_list[0]
                # Blink detection via EAR
                if not blink_done:
                    left_eye = lm.get("left_eye", [])
                    right_eye = lm.get("right_eye", [])
                    ear_left = self._eye_aspect_ratio(left_eye) if left_eye else 1.0
                    ear_right = self._eye_aspect_ratio(right_eye) if right_eye else 1.0
                    ear = (ear_left + ear_right) / 2.0
                    if ear < EAR_THRESH:
                        ear_below_count += 1
                    else:
                        if ear_below_count >= EAR_CONSEC_FRAMES:
                            blink_done = True
                        ear_below_count = 0

                # Head left/right using nose tip x movement relative to baseline
                nose_points = lm.get("nose_tip", [])
                if nose_points:
                    nose_x = float(np.mean([p[0] for p in nose_points]))
                    if baseline_x is None:
                        baseline_x = nose_x
                    else:
                        if blink_done and not left_done and (nose_x <= baseline_x - HEAD_MOVE_PIX):
                            left_done = True
                            baseline_x = nose_x  # update for next stage
                        elif blink_done and left_done and not right_done and (nose_x >= baseline_x + HEAD_MOVE_PIX):
                            right_done = True

            # Draw instruction
            cv2.putText(
                frame,
                f"Liveness: {instruction}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Liveness: cancelled by user.")
                return False

    def capture_and_save_images(
        self, student_id: str, student_name: str, num_images: int = 20, liveness_required: bool = True
    ) -> Optional[str]:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Error: Cannot open camera index {self.camera_index}")
            return None

        window_name = f"Registering {student_name} ({student_id})"
        cv2.namedWindow(window_name)

        if liveness_required:
            passed = self._run_liveness_challenge(cap, window_name=window_name)
            if not passed:
                cap.release()
                cv2.destroyWindow(window_name)
                return None

        s_name_sanitized = student_name.replace(" ", "_").lower()
        dir_name = f"{student_id}_{s_name_sanitized}"
        student_dir = os.path.join(self.dataset_dir, dir_name)
        self._ensure_dir(student_dir)

        count = 0
        try:
            while count < num_images:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame from camera.")
                    break

                msg = f"Capturing {count + 1}/{num_images}. Press 'c' to snap, 'q' to quit."
                cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Image capture cancelled by user.")
                    break
                elif key == ord('c'):
                    img_path = os.path.join(student_dir, f"{student_id}_{count + 1}.jpg")
                    cv2.imwrite(img_path, frame)
                    print(f"Saved image: {img_path}")
                    count += 1

            if count < num_images:
                print("Registration cancelled or incomplete. Rolling back.")
                shutil.rmtree(student_dir)
                return None

        finally:
            cap.release()
            cv2.destroyAllWindows()

        return student_dir

    def capture_frames(
        self, student_id: str, student_name: str, num_images: int = 20, liveness_required: bool = True
    ) -> Optional[list[np.ndarray]]:
        """
        Capture frames for a student WITHOUT writing raw images to disk.
        Returns a list of BGR frames (numpy arrays) or None if cancelled.
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Error: Cannot open camera index {self.camera_index}")
            return None

        window_name = f"Registering {student_name} ({student_id})"
        cv2.namedWindow(window_name)

        if liveness_required:
            passed = self._run_liveness_challenge(cap, window_name=window_name)
            if not passed:
                cap.release()
                cv2.destroyWindow(window_name)
                return None

        frames: list[np.ndarray] = []
        count = 0
        try:
            while count < num_images:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame from camera.")
                    break

                msg = f"Capturing {count + 1}/{num_images}. Press 'c' to snap, 'q' to quit."
                cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Image capture cancelled by user.")
                    break
                elif key == ord('c'):
                    frames.append(frame.copy())
                    print(f"Captured frame {count + 1}")
                    count += 1

            if count < num_images:
                print("Registration cancelled or incomplete. No frames saved.")
                return None
        finally:
            cap.release()
            cv2.destroyAllWindows()

        return frames

    def encode_and_store_embeddings(
        self, db: DatabaseManager, student_id: str, dataset_dir_path: str, privacy_delete_images: bool = True
    ) -> int:
        if not os.path.isdir(dataset_dir_path):
            print(f"Warning: Dataset directory not found for encoding: {dataset_dir_path}")
            return 0

        files = [f for f in os.listdir(dataset_dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            return 0

        fernet = get_fernet()
        stored_count = 0
        for fname in files:
            img_path = os.path.join(dataset_dir_path, fname)
            try:
                image = face_recognition.load_image_file(img_path)
                boxes = face_recognition.face_locations(image, model="hog")
                encodings = face_recognition.face_encodings(image, boxes)

                if encodings:
                    encrypted_embedding = encrypt_embedding_array(encodings[0], fernet)
                    db.add_face_embedding(student_id, encrypted_embedding)
                    stored_count += 1
            except Exception as e:
                print(f"Error: Failed to process and store embedding for {img_path}: {e}")

        if privacy_delete_images:
            try:
                shutil.rmtree(dataset_dir_path)
                print(f"Info: Privacy cleanup: removed dataset directory {dataset_dir_path}")
            except Exception as e:
                print(f"Error during privacy cleanup of {dataset_dir_path}: {e}")

        return stored_count

    def encode_and_store_from_frames(self, db: DatabaseManager, student_id: str, frames: list[np.ndarray]) -> int:
        """
        Encode a list of BGR frames, encrypt the first face embedding per frame,
        and store them in the DB. Returns number of stored embeddings.
        """
        if not frames:
            return 0
        fernet = get_fernet()
        stored = 0
        for i, frame in enumerate(frames):
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(rgb, model="hog")
                encs = face_recognition.face_encodings(rgb, boxes)
                if encs:
                    token = encrypt_embedding_array(encs[0], fernet)
                    db.add_face_embedding(student_id, token)
                    stored += 1
            except Exception as e:
                print(f"Error encoding in-memory frame {i}: {e}")
        return stored

    def capture_encode_and_store(
        self, db: DatabaseManager, student_id: str, student_name: str, num_images: int = 20, liveness_required: bool = True
    ) -> int:
        """
        High-level helper: capture frames in-memory (no disk I/O),
        then encode, encrypt, and store embeddings. Returns stored count.
        """
        frames = self.capture_frames(student_id, student_name, num_images=num_images, liveness_required=liveness_required)
        if not frames:
            return 0
        return self.encode_and_store_from_frames(db, student_id, frames)

    def clean_dataset_dir(self) -> bool:
        """Delete the entire dataset directory tree for privacy.
        Returns True if removed or did not exist."""
        try:
            if os.path.isdir(self.dataset_dir):
                import shutil
                shutil.rmtree(self.dataset_dir)
                print(f"Privacy cleanup: removed dataset folder {self.dataset_dir}")
            return True
        except Exception as e:
            print(f"Failed to clean dataset folder {self.dataset_dir}: {e}")
            return False
