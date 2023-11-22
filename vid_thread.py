import numpy as np
from PyQt5.QtCore import pyqtSignal, QThread
import mediapipe as mp
import cv2
import copy
from typing import Tuple
import time

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    angular_data_signal = pyqtSignal(np.float64, np.float64)
    def __init__(self):
        super().__init__()
        self._run_flag = True

        #medida pipe
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            # upper_body_only=upper_body_only,
            enable_segmentation = True,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )


    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                frame = copy.deepcopy(cv_img)
                frame = cv2.flip(frame,1)

            result = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if result.pose_landmarks != None:
                frame, angle = self.find_right_arms(result.pose_landmarks, frame)
                #media pipe here
                self.change_pixmap_signal.emit(frame)
                self.angular_data_signal.emit(angle, np.float64(time.time_ns()))
        # shut down capture system
        cap.release()

    def find_right_arms(self, landmarks, img, threshold = 0.65):
        angle = np.float64(0)
        img_width, img_height = img.shape[1], img.shape[0]
        landmark_point = {11:None, 13:None, 15:None}
        for i, landmark in enumerate(landmarks.landmark):
            if i in [11,13,15] and landmark.visibility > threshold:
                landmark_x = min(int(landmark.x * img_width), img_width - 1)
                landmark_y = min(int(landmark.y * img_height), img_height - 1)
                # landmark_z = landmark.z
                landmark_point[i] = (landmark_x, landmark_y)
                cv2.circle(img, (landmark_x, landmark_y), 5, (0, 255, 0), 2)

        #draw line, and calculate angle
        if not None in landmark_point.values():
            # sholder to elbow
            cv2.line(img, landmark_point[11], landmark_point[13],
                        (0, 255, 0), 2)
            cv2.line(img, landmark_point[13], landmark_point[15],
                        (0, 255, 0), 2)
            es = np.array(landmark_point[11]) - np.array(landmark_point[13])
            ew = np.array(landmark_point[15]) - np.array(landmark_point[13])
            angle = np.arccos(np.dot(es, ew) / (np.linalg.norm(es) * np.linalg.norm(ew))) * 180 / np.pi
            cv2.putText(img, f"{angle:.3f}",
                            landmark_point[13],
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                            cv2.LINE_AA)
        return img, angle

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        # self.wait()