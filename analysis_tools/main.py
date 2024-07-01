import sys
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDateTimeEdit,
    QDial,
    QDoubleSpinBox,
    QFontComboBox,
    QLabel,
    QLCDNumber,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QListWidget,
)
from PySide6.QtCore import Qt, QThread, QThreadPool, Signal, Slot

from analysis_tools.tool_multi_lv_masks import ImageProcessing

cnt = 0


class MaskLayerSettings(QtWidgets.QWidget):
  def __init__(self, *args, **kwargs):
    super(MaskLayerSettings, self).__init__(*args, **kwargs)

    layout = QHBoxLayout()
    combobox = QComboBox()
    combobox.addItems(["1", "2", "4"])
    combobox.currentIndexChanged.connect(self.index_changed)
    combobox.currentTextChanged.connect(self.text_changed)

    layout.addWidget(combobox)
    self.setLayout(layout)

  def index_changed(self, index):  # index is an int stating from 0
    global cnt
    cnt += 1
    print(index)

  def text_changed(self, text):  # text is a str
    print(text)


class FeatureSelection(QtWidgets.QWidget):
  def __init__(self, *args, **kwargs):
    super(FeatureSelection, self).__init__(*args, **kwargs)

    layout = QHBoxLayout()
    combobox = QComboBox()
    combobox.addItems(["One", "Two", "Three"])
    combobox.currentIndexChanged.connect(self.index_changed)
    combobox.currentTextChanged.connect(self.text_changed)

    layout.addWidget(combobox)
    self.setLayout(layout)

  def index_changed(self, index):  # index is an int stating from 0
    print(index)

  def text_changed(self, text):  # text is a str
    print(text)


class FeaturesList(QtWidgets.QWidget):
  def __init__(self, *args, **kwargs):
    super(FeaturesList, self).__init__(*args, **kwargs)

    layout = QVBoxLayout()

    self.listwidget = QListWidget()
    self.listwidget.addItems(["dad", "www", "vff", "vff", "vff", "vff"])

    # In QListWidget there are two separate signals for the item, and the str
    self.listwidget.currentItemChanged.connect(self.index_changed)
    self.listwidget.currentTextChanged.connect(self.text_changed)

    layout.addWidget(self.listwidget)
    self.setLayout(layout)

  def index_changed(self, index):  # Not an index, index is a QListWidgetItem
    print(index.text())

  def text_changed(self, text):  # text is a str
    self.listwidget.item(cnt).setText("dasd")
    print(text)


class FeatureSettings(QtWidgets.QWidget):
  def __init__(self, *args, **kwargs):
    super(FeatureSettings, self).__init__(*args, **kwargs)


class MainWindow(QMainWindow):
  def __init__(self):
    super(MainWindow, self).__init__()

    self.back_end = ImageProcessing()
    self.threadpool = QThreadPool()
    self.threadpool.start(self.back_end)

    self.setWindowTitle("My App")
    layer = QVBoxLayout()

    layer_settings_layer = QHBoxLayout()
    layer_settings_layer.addWidget(MaskLayerSettings())

    feature_selection_layer = QHBoxLayout()
    feature_selection_layer.addWidget(FeatureSelection())

    feature_selected_list_layer = QHBoxLayout()
    feature_selected_list_layer.addWidget(FeaturesList())

    feature_settings_layer = QHBoxLayout()
    feature_settings_layer.addWidget(FeatureSettings())

    layer.addLayout(layer_settings_layer)
    layer.addLayout(feature_selection_layer)
    layer.addLayout(feature_selected_list_layer)
    layer.addLayout(feature_settings_layer)

    widget = QWidget()
    widget.setLayout(layer)

    self.setCentralWidget(widget)


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
