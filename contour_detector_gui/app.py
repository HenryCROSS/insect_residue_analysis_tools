import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel,
    QVBoxLayout, QWidget, QPushButton,
    QHBoxLayout, QStackedLayout, QSpinBox
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen
from PySide6.QtCore import Qt, QRect
import cv2
import os

from contour_detector_gui.image_utils import Img_State, Lv_Mask, calculate_lv_masks, create_img_unit, create_mask, draw_mask_contour, draw_overlap_transparently, load_imgs_path

IMG_DIR_IN: str = "./Pictures"
IMG_DIR_OUT: str = f"{IMG_DIR_IN}_out"

def cvt_opencv2QImage(cv_img):
  height, width, channel = cv_img.shape
  bytes_per_line = 3 * width
  q_image = QImage(cv_img.data, width, height,
                    bytes_per_line, QImage.Format.Format_BGR888)
  return q_image

class ImageLabel(QLabel):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.setMouseTracking(True)
    self.start_point = None
    self.end_point = None
    self.is_editing = False

  def set_image(self, image: Img_State):
    # read OpenCV Image
    self.img_state = image
    (img_name, cv_image) = self.img_state.get_img()

    if self.is_editing:
      if self.img_state.mask_list_size < 1:
        # convert openCV image to QImage
        q_image = cvt_opencv2QImage(cv_image)

        # display image
        self.setPixmap(QPixmap.fromImage(q_image))
      else:
        show_img_unit = self.img_state.get_img_with_mask()

        (_, cv_image) = show_img_unit
        q_image = cvt_opencv2QImage(cv_image)
        self.setPixmap(QPixmap.fromImage(q_image))
    else:
      if self.img_state.after_process_img is None:
        # convert openCV image to QImage
        q_image = cvt_opencv2QImage(cv_image)

        # display image
        self.setPixmap(QPixmap.fromImage(q_image))
      else:
        # convert openCV image to QImage
        q_image = cvt_opencv2QImage(self.img_state.after_process_img)

        # display image
        self.setPixmap(QPixmap.fromImage(q_image))
  
  def update_img(self):
    self.set_image(self.img_state)

  def mousePressEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton and self.is_editing:
      self.start_point = event.position().toPoint()
      self.end_point = event.position().toPoint()
      self.update()

  def mouseMoveEvent(self, event):
    if self.start_point:
      self.end_point = event.position().toPoint()
      self.update()

  def mouseReleaseEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton and self.start_point:
      self.end_point = event.position().toPoint()

      # TODO: create mask
      # BUG:
      (_, cv_image) = self.img_state.get_img()
      img_shape = cv_image.shape[:2]

      #  QPoint to tuple
      start_point = (self.start_point.x(), self.start_point.y())
      end_point = (self.end_point.x(), self.end_point.y())
      self.img_state.append_mask(
          create_mask(img_shape, (start_point, end_point)))

      self.start_point = None
      self.end_point = None
      self.set_image(self.img_state)
      self.update()

  def paintEvent(self, event):
    super().paintEvent(event)
    if not self.pixmap():
      return

    painter = QPainter(self)
    painter.drawPixmap(self.rect(), self.pixmap())

    pen = QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.SolidLine)
    painter.setPen(pen)

    if self.start_point and self.end_point:
      rect = QRect(self.start_point, self.end_point)
      painter.drawRect(rect)

  def switchToEditMode(self):
    self.is_editing = True

  def switchToViewMode(self):
    self.is_editing = False

  def processImg(self):
    self.img_state.process_img()

  def setPriority(self, val: int):
    self.img_state.current_priority = val
    self.update_img()


class MainWindow(QMainWindow):
  def __init__(self):
    super().__init__()
    # 设置图像路径
    self.image_folder = "./Pictures"
    # self.image_list = sorted(os.listdir(self.image_folder))
    self.image_index: int = 0
    self.image_list: list[Img_State] = load_imgs_path(IMG_DIR_IN)

    self.setWindowTitle("Draw Rectangles on Image")
    self.setGeometry(0, 0, 1000, 600)

    # main layout
    main_layout = QHBoxLayout()

    # Image label
    self.label = ImageLabel()
    main_layout.addWidget(self.label)

    # features layout
    self.side_panel = QWidget()
    self.stacked_layout = QStackedLayout()

    # 模式1
    self.mode1_widget = QWidget()
    self.mode1_layout = QVBoxLayout()
    self.mode1_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    self.prev_button = QPushButton("Previous Image")
    self.prev_button.clicked.connect(self.show_prev_image)
    self.mode1_layout.addWidget(self.prev_button)

    self.next_button = QPushButton("Next Image")
    self.next_button.clicked.connect(self.show_next_image)
    self.mode1_layout.addWidget(self.next_button)

    self.switch_to_mode2_button = QPushButton("Edit")
    self.switch_to_mode2_button.clicked.connect(self.switch_to_edit_mode)
    self.mode1_layout.addWidget(self.switch_to_mode2_button)

    self.mode1_widget.setLayout(self.mode1_layout)
    self.stacked_layout.addWidget(self.mode1_widget)

    # 模式2
    self.mode2_widget = QWidget()
    self.mode2_layout = QVBoxLayout()
    self.mode2_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    label = QLabel("Priority:")
    self.mode2_layout.addWidget(label)

    self.priority_slider = QSpinBox()
    self.priority_slider.setRange(0, 10000)
    self.priority_slider.setSingleStep(1)
    self.priority_slider.valueChanged.connect(self.label.setPriority)
    self.mode2_layout.addWidget(self.priority_slider)

    self.undo_button = QPushButton("undo")
    self.undo_button.clicked.connect(self.undo_mask)
    self.mode2_layout.addWidget(self.undo_button)

    self.redo_button = QPushButton("redo")
    self.redo_button.clicked.connect(self.redo_mask)
    self.mode2_layout.addWidget(self.redo_button)

    self.process_button = QPushButton("Process")
    self.process_button.clicked.connect(self.process_images)
    self.mode2_layout.addWidget(self.process_button)

    self.switch_to_mode1_button = QPushButton("Cancel")
    self.switch_to_mode1_button.clicked.connect(self.switch_to_view_mode)
    self.mode2_layout.addWidget(self.switch_to_mode1_button)

    self.mode2_widget.setLayout(self.mode2_layout)
    self.stacked_layout.addWidget(self.mode2_widget)

    self.side_panel.setLayout(self.stacked_layout)
    main_layout.addWidget(self.side_panel)

    # 主窗口容器
    container = QWidget()
    container.setLayout(main_layout)
    self.setCentralWidget(container)

    self.update_image()

  def update_image(self):
    if 0 <= self.image_index < len(self.image_list):
      self.label.set_image(self.image_list[self.image_index])

  def show_prev_image(self):
    if self.image_index > 0:
      self.image_index -= 1
      self.update_image()

  def show_next_image(self):
    if self.image_index < len(self.image_list) - 1:
      self.image_index += 1
      self.update_image()

  def switch_to_view_mode(self):
    self.stacked_layout.setCurrentIndex(0)
    self.label.switchToViewMode()
    self.update_image()

  def switch_to_edit_mode(self):
    self.stacked_layout.setCurrentIndex(1)
    self.label.switchToEditMode()
    self.priority_slider.setValue(
        self.image_list[self.image_index].current_priority)
    self.update_image()

  def process_images(self):
    print("Processing images...")
    self.label.processImg()
    self.switch_to_view_mode()

  def undo_mask(self):
    self.image_list[self.image_index].undo_mask()
    self.update_image()

  def redo_mask(self):
    self.image_list[self.image_index].redo_mask()
    self.update_image()


if __name__ == "__main__":
  app = QApplication(sys.argv)
  window = MainWindow()
  window.show()
  sys.exit(app.exec())
