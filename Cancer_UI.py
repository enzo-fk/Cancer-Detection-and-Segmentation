# -*- coding: utf-8 -*-
# Form implementation generated from combining original functions
# with a new PyQt5 layout template, now displaying images

import os
import traceback
from PyQt5 import QtCore, QtGui, QtWidgets
from best_det import DetModel
from best_seg import SegmModel
from ground import GtGenerator
from metrics import MetricsRaiza

class Ui_MainWindow(object):
    def __init__(self):
        self.detect = DetModel()
        self.segmentation = SegmModel()
        self.Gt_utils = GtGenerator()
        self.metrics = MetricsRaiza()

        self.images = []
        self.actual_index = 0

        self.img_dir = os.path.join('.', 'dataset', 'Sorted Data', 'Original_imgs')
        self.json_dir = os.path.join('.', 'dataset', 'Sorted Data', 'Json_files')
        self.GT_dir = os.path.join('.', 'dataset', 'Sorted Data', 'GT')

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # --- Left panel: navigation and loading ---
        self.loadFolderBtn = QtWidgets.QPushButton("Load Folder", self.centralwidget)
        self.loadFolderBtn.setGeometry(QtCore.QRect(20, 50, 150, 30))
        self.prevBtn = QtWidgets.QPushButton("Previous", self.centralwidget)
        self.prevBtn.setGeometry(QtCore.QRect(20, 100, 150, 30))
        self.nextBtn = QtWidgets.QPushButton("Next", self.centralwidget)
        self.nextBtn.setGeometry(QtCore.QRect(20, 150, 150, 30))
        self.statusText = QtWidgets.QTextEdit(self.centralwidget)
        self.statusText.setGeometry(QtCore.QRect(20, 200, 150, 100))
        self.statusText.setReadOnly(True)

        # --- Image display widget ---
        self.imageLabel = QtWidgets.QLabel(self.centralwidget)
        self.imageLabel.setGeometry(QtCore.QRect(200, 240, 580, 350))
        self.imageLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setScaledContents(True)

        # --- Center: Model operations group ---
        self.groupBoxImage = QtWidgets.QGroupBox("Image Operations", self.centralwidget)
        self.groupBoxImage.setGeometry(QtCore.QRect(200, 20, 580, 200))
        self.detectBtn = QtWidgets.QPushButton("Detection", self.groupBoxImage)
        self.detectBtn.setGeometry(QtCore.QRect(20, 30, 200, 30))
        self.segBtn = QtWidgets.QPushButton("Segmentation", self.groupBoxImage)
        self.segBtn.setGeometry(QtCore.QRect(20, 70, 200, 30))

        self.iouValue = QtWidgets.QLabel("IoU:  ", self.groupBoxImage)
        self.iouValue.setGeometry(QtCore.QRect(250, 30, 300, 30))
        self.accValue = QtWidgets.QLabel("Accuracy:  ", self.groupBoxImage)
        self.accValue.setGeometry(QtCore.QRect(250, 70, 300, 30))
        self.precValue = QtWidgets.QLabel("Precision:  ", self.groupBoxImage)
        self.precValue.setGeometry(QtCore.QRect(250, 110, 300, 30))
        self.diceValue = QtWidgets.QLabel("Dice Coef:  ", self.groupBoxImage)
        self.diceValue.setGeometry(QtCore.QRect(250, 150, 300, 30))

        self.loadFolderBtn.clicked.connect(self.load_img_folder)
        self.prevBtn.clicked.connect(self.prev_image)
        self.nextBtn.clicked.connect(self.next_image)
        self.detectBtn.clicked.connect(self.det_pred_act)
        self.segBtn.clicked.connect(self.seg_pred_act)


        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle("Image Processing App")

    def load_img_folder(self):
        try:
            files = os.listdir(self.img_dir)
            self.images = sorted(
                os.path.join(self.img_dir, f)
                for f in files
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            )
            self.actual_index = 0
            self.display_current_image()
        except Exception as e:
            traceback.print_exc()
            self.statusText.setText(str(e))

    def display_current_image(self):
        if not self.images:
            self.statusText.setText("No images loaded")
            self.imageLabel.clear()
            return
        path = self.images[self.actual_index]
        fname = os.path.basename(path)
        self.statusText.setText(f"{self.actual_index+1}/{len(self.images)}: {fname}")
        pixmap = QtGui.QPixmap(path)
        if pixmap.isNull():
            self.imageLabel.setText("Cannot load image")
        else:
            self.imageLabel.setPixmap(pixmap)

    def prev_image(self):
        if not self.images:
            return
        self.actual_index = max(0, self.actual_index - 1)
        self.display_current_image()

    def next_image(self):
        if not self.images:
            return
        self.actual_index = min(len(self.images) - 1, self.actual_index + 1)
        self.display_current_image()

    def det_pred_act(self):
        if not self.images:
            return
        try:
            img_path = self.images[self.actual_index]
            det_results = self.detect.predict(img_path)
            json_name = os.path.basename(img_path).rsplit('.', 1)[0] + '.json'
            gt_path = os.path.join(self.json_dir, json_name)
            gt_data = self.Gt_utils.load(gt_path)
            iou = self.metrics.calculate_iou(gt_data, det_results)
            acc = self.metrics.calculate_accuracy(gt_data, det_results)
            prec = self.metrics.calculate_precision(gt_data, det_results)
            self.iouValue.setText(f"IoU: {iou:.3f}")
            self.accValue.setText(f"Accuracy: {acc:.3f}")
            self.precValue.setText(f"Precision: {prec:.3f}")
        except Exception as e:
            traceback.print_exc()

    def seg_pred_act(self):
        if not self.images:
            return
        try:
            img_path = self.images[self.actual_index]
            seg_mask = self.segmentation.predict(img_path)
            mask_name = os.path.basename(img_path).rsplit('.', 1)[0] + '_mask.png'
            gt_mask_path = os.path.join(self.GT_dir, mask_name)
            gt_mask = self.Gt_utils.load_mask(gt_mask_path)
            dice = self.metrics.calculate_dice(gt_mask, seg_mask)
            self.diceValue.setText(f"Dice Coef: {dice:.3f}")
        except Exception as e:
            traceback.print_exc()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())