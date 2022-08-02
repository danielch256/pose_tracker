import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import time
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
source = [cv2.VideoCapture('vids/kid.mp4')]
test = [0]

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Pose Inductive Stance Sensor")
        self.main_layout = QHBoxLayout()

        #source selector radio button
        self.source_select = QHBoxLayout()
        self.file_button = QRadioButton("file")
        self.file_button.setChecked(True)
        self.file_button.toggled.connect(self.file_selected)
        self.source_select.addWidget(self.file_button)

        self.camera_button = QRadioButton("camera")
        self.camera_button.toggled.connect(self.camera_selected)
        self.source_select.addWidget(self.camera_button)

        #self.main_layout.addLayout(self.top_layout)

        #test_selector dropdown menu
        self.test_selector = QComboBox()
        self.test_selector.addItems(["test1:ssh", "test2", "test3"])
        self.test_selector.activated.connect(self.activated)
        self.test_selector.currentTextChanged.connect(self.text_changed)
        self.test_selector.currentIndexChanged.connect(self.index_changed)
        #self.main_layout.addWidget(self.test_selector)

        #video control buttons
        self.vc_buttons = QHBoxLayout()
        self.vc_buttons.addStretch()

            #start button
        self.startBTN = QPushButton("start")
        self.startBTN.clicked.connect(self.startFeed)
            #stop button
        self.CancelBTN = QPushButton("stop")
        self.CancelBTN.clicked.connect(self.CancelFeed)

        self.vc_buttons.addWidget(self.startBTN)
        self.vc_buttons.addWidget(self.CancelBTN)
        #self.main_layout.addLayout(self.vc_buttons)
        self.vc_buttons.addStretch()

        #hline
        self.line = QFrame()
        self.line.setGeometry(QRect(60, 110, 751, 20))
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        # self.main_layout.addWidget(self.line)

        #groupbox
        self.groupbox = QGroupBox("Option Checkboxes")
        self.grid = QGridLayout()

        self.groupbox.setLayout(self.grid)

        for i in range(1, 4):
            for j in range(1, 4):
                self.grid.addWidget(QCheckBox("Option " + str(i) + str(j)), i, j)




        self.createForm()

        # feed widget
        self.FeedLabel = QLabel()
        self.FeedLabel.setPixmap(QPixmap('gui_bk.png'))
        self.main_layout.addWidget(self.FeedLabel)

        self.setLayout(self.main_layout)


        self.Worker1 = Worker1()
        #self.Worker1.start()
        #self.Worker1.running = True
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)

    def camera_selected(self, selected):
        if selected:
            print("camera selected")
            source[0] = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def file_selected(self, selected):
        if selected:
            print("file selected")
            source[0] = cv2.VideoCapture('vids/kid.mp4')

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def startFeed(self):
        self.Worker1.start()

    def CancelFeed(self):
        self.Worker1.stop()

    def activated(Self, index):
        pass
        #print("Activated index:", index)

    def text_changed(self, s):
        pass
        #print("Text changed:", s)

    def index_changed(self, index):
        #print("Index changed", index)
        print('test[0]=', index)
        test[0] = index

    def createForm(self):
        # creating a form layout
        self.left_form = QFormLayout()
        # adding rows
        # for name and adding input text
        self.left_form.addRow(QLabel("SOURCE:"), self.source_select)
        self.left_form.addRow(QLabel("TEST:"), self.test_selector)
        self.left_form.addRow(self.groupbox)

        self.left_form.addRow(self.line)
        self.left_form.addRow(self.vc_buttons)

        # for degree and adding combo box
        #layout.addRow(QLabel("Degree"), self.degreeComboBox)
        # for age and adding spin box
        #layout.addRow(QLabel("Age"), self.ageSpinBar)
        # setting layout
        #self.formGroupBox.setLayout(layout)
        # make a form layout

        #self.left_form = QFormLayout()
        # self.left_form.addRow("TEST:", )
        self.main_layout.addLayout(self.left_form)


print(source)
class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        #capture = cv2.VideoCapture('vids/kid.mp4')

        capture = source[0]

        while self.ThreadActive:
            ret, frame = capture.read()
            #time.sleep(0.015)
            Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if test[0] == 0:
                results = pose.process(Image)
                lmList = []
                if results.pose_landmarks:
                    mpDraw.draw_landmarks(Image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                    for id, lm in enumerate(results.pose_landmarks.landmark):
                        h, w, c = Image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)  # get pixel values instead of ratio of picture width
                        # print(id, cx,cy)
                        lmList.append([id, cx, cy])  # list of id, x and y coords of all 33 landmarks
                        cv2.circle(Image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)  # draw blue dots
                    cv2.circle(Image, (lmList[14][1], lmList[14][2]), 5, (0, 255, 0),
                               cv2.FILLED)  # draw green dot on landmark 14
            else:
                pass

            FlippedImage = cv2.flip(Image, 1)
            ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
            Pic = ConvertToQtFormat.scaled(960, 720, Qt.KeepAspectRatio).copy() #/copy() to fix crash when scaling to 1.5x of 640x480. why idk
            self.ImageUpdate.emit(Pic)

    def stop(self):
        self.ThreadActive = False
        self.quit()


if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())
