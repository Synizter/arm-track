from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap
import sys
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QTimer
import cv2
import numpy as np
from vid_thread import VideoThread
from eeg_thread import EEGThread

import winsound
import os
import time
import datetime


SUBJ_CODE_NAME = 'CHI'
CUE_TIME = 2000 # 2s
END_TIME_DELAY = 1000 #1s

class App(QWidget):

    global TRIAL_N, CUE_TIME, END_TIME_DELAY, SUBJ_CODE_NAME

    angular = np.array([])
    signal = np.array([])
    elapse = float(0)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt live label demo")
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('Webcam')

        # create a vertical box layout and add the two labels
        self.start_btn = QPushButton('view stream')
        self.start_btn.setProperty('class', 'danger')
        self.start_btn.resize(150, 50)
        self.start_btn.clicked.connect(self.onStartBtnClicked)

        self.record_btn = QPushButton('record movement')
        self.record_btn.setEnabled(False)
        self.record_btn.setProperty('class', 'danger')
        self.record_btn.resize(150,50)
        self.record_btn.clicked.connect(self.onRecordBtnClicked)    

        # self.record_idle_btn = QPushButton('record idle')
        # self.record_idle_btn.setEnabled(False)
        # self.record_idle_btn.setProperty('class', 'danger')
        # self.record_idle_btn.resize(150,50)
        # self.record_idle_btn.clicked.connect(self.onRecordIdleBtnClicked)

        self.end_btn = QPushButton('End')
        self.end_btn.setEnabled(True)
        self.end_btn.setProperty('class', 'danger')
        self.end_btn.resize(150,50)
        self.end_btn.clicked.connect(self.onEndBtnClicked)        
        self.initialize_thread()
        self.is_recorded_eeg = False
        self.is_recorded_angle = False

        #sound path
        self.fsound = os.path.join(os.getcwd(), 'assets', 'beep-07a.wav')
        self.timer = QTimer()
        self.timer.timeout.connect(self.onTimeOut)

        self.end_timer = QTimer()
        self.end_timer.timeout.connect(self.onEndTrialTimeOut)


        self.trial_cnt = 1

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        vbox.addWidget(self.start_btn)
        vbox.addWidget(self.record_btn)
        vbox.addWidget(self.end_btn)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

    def onStartBtnClicked(self):
        self.record_btn.setEnabled(True)
        self.vidThread.start()

    def onRecordBtnClicked(self):
        #start countdown 2.5s before play beep sounD
        self.record_btn.setEnabled(False)
        self.is_recorded_eeg = True
        self.timer.start(CUE_TIME)
    
    def onEndBtnClicked(self):
        self.vidThread.stop()
        self.eegThread.stop()
        self.close()

    def onEndTrialTimeOut(self):
        self.end_timer.stop()
        self.is_recorded_eeg = False
        np.save(f'outputs/{SUBJ_CODE_NAME}_EEG_{ self.trial_cnt}', self.signal)

        self.signal = np.array([])
        print(f"end trial {self.trial_cnt}")
        self.trial_cnt += 1

    def onTimeOut(self):
        winsound.PlaySound(self.fsound, winsound.SND_FILENAME | winsound.SND_ASYNC)
        self.timer.stop()
        self.elapse = time.time()
        self.is_recorded_angle = True

    def initialize_thread(self):
        # create the video capture thread
        self.vidThread = VideoThread()
        self.eegThread = EEGThread()

        # connect its signal to the update_image slot
        self.vidThread.change_pixmap_signal.connect(self.update_image)
        self.vidThread.angular_data_signal.connect(self.update_angular)
        self.eegThread.filtered_chunk.connect(self.update_eeg)
        self.eegThread.start()

    def closeEvent(self, event):
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_eeg(self, eeg):
        if self.is_recorded_eeg:
            self.signal = np.hstack([self.signal, eeg]) if self.signal.size else eeg

    @pyqtSlot(np.float64, np.float64)
    def update_angular(self, *args):
        if self.is_recorded_angle:
            #push to array
            self.angular = np.vstack([self.angular, np.array(args)]) if self.angular.size else np.array(args) 
            if args[0] != 0 and args[0] < 55:
                self.end_timer.start(END_TIME_DELAY)
                
                self.is_recorded_angle = False
                self.record_btn.setEnabled(True)
                np.save(f'outputs/{SUBJ_CODE_NAME}_ANGLE_{ self.trial_cnt}', self.angular)
                self.angular = np.array([])
                print(f"trajectory time = {time.time() - self.elapse}")
                self.elapse = time.time()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())