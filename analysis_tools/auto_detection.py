import cv2
import numpy as np
import os
from cv2.typing import *
from typing import *
from functools import reduce

IMG_DIR_IN: str = "./Pictures"
IMG_DIR_OUT: str = "./Pictures_out"


class Image:
  def __init__(self, path):
    self.path = path
    self.img = None
    self.preprocessed_shape_mask = None
    self.preprocessed_shape = None
    self.shape_mask = None

  def get_img(self):
    if self.img is None:
      self.img = cv2.imread(self.path)

    return self.img


def remove_blur_fft(image):
  if len(image.shape) == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = np.float32(image)

  dft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
  dft_shift = np.fft.fftshift(dft)

  rows, cols = image.shape
  crow, ccol = rows // 2, cols // 2
  mask = np.ones((rows, cols, 2), np.uint8)
  r = 30
  center = (ccol, crow)
  x, y = np.ogrid[:rows, :cols]
  mask_area = (x - crow) ** 2 + (y - ccol) ** 2 <= r * r
  mask[mask_area] = 0

  fshift = dft_shift * mask

  f_ishift = np.fft.ifftshift(fshift)
  img_back = cv2.idft(f_ishift)
  img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

  cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
  return np.uint8(img_back)


def process_image_larger_shape(img: MatLike) -> Tuple[MatLike, MatLike]:
    # Your existing image processing function
  img_color = img.copy()
  img = cv2.bitwise_not(img)

  # Apply Gaussian Blur
  blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

  # Remove blur using FFT
  fft_img = remove_blur_fft(blurred_img)

  # Create a convolution kernel for erosion
  kernel = np.ones((3, 3), np.uint8)
  eroded_img = cv2.erode(fft_img, kernel, iterations=1)

  # Create a CLAHE object
  clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
  cl1 = clahe.apply(eroded_img)

  # Apply Otsu's thresholding
  otsu_thresh_value, otsu_img = cv2.threshold(
      cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  # Remove small circles (morphological open)
  kernel = np.ones((3, 3), np.uint8)
  cleaned_otsu_img = cv2.morphologyEx(otsu_img, cv2.MORPH_OPEN, kernel)

  # Dilate and close to connect nearby areas
  kernel = np.ones((10, 10), np.uint8)  # Adjust size as needed
  filter_mask = cv2.dilate(cleaned_otsu_img, kernel, iterations=1)
  connected_img = cv2.morphologyEx(filter_mask, cv2.MORPH_CLOSE, kernel)

  # Generate solid polygon
  contours, _ = cv2.findContours(
      connected_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  solid_polygon_img = np.zeros_like(connected_img)
  cv2.fillPoly(solid_polygon_img, contours, 255)

  # Remove small circles
  min_area = 2000  # Adjust min area as needed
  contours, _ = cv2.findContours(
      solid_polygon_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  filtered_img = np.zeros_like(solid_polygon_img)
  for contour in contours:
    if cv2.contourArea(contour) >= min_area:
      cv2.drawContours(filtered_img, [contour], -1, 255, thickness=cv2.FILLED)

  # Create a transparent blue layer
  blue_layer = np.zeros_like(img_color)
  blue_layer[:, :] = (255, 0, 0)  # Blue
  alpha = 0.2  # Transparency

  # Apply blue layer to the dilated area
  mask_bool = filtered_img.astype(bool)
  img_color[mask_bool] = cv2.addWeighted(
      img_color, 1 - alpha, blue_layer, alpha, 0)[mask_bool]

  # Draw red contours on the original image
  contours, hierarchy = cv2.findContours(
      filtered_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  for i in range(len(contours)):
    cv2.drawContours(img_color, contours, i, (0, 0, 255), 2)

  return img_color, filtered_img


def do_erosion(img: MatLike):
  kernel = np.ones((5, 5), np.uint8)
  return cv2.erode(img, kernel, iterations=1)

def do_img_segmentation(img: MatLike):
  """Find segmentation based on K means clustering"""
  Z = img.reshape((-1, 3))
  # convert to np.float32
  Z_f32 = np.float32(Z)
  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 8
  ret, label, center = cv2.kmeans(
      Z_f32, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  # Now convert back into uint8, and make original image
  center = np.uint8(center)
  res = center[label.flatten()]
  res2 = res.reshape((img.shape))
  return res2

def do_erosion(img: MatLike):
  kernel = np.ones((9, 9), np.uint8)
  return cv2.dilate(img, kernel, iterations=1)


def detail_process_img(img: MatLike, mask_img: MatLike):
  new_process = [
      do_erosion,
      do_smoothing,
      contrast_enhancement,
      remove_blur_fft,
      mask_with_shape(mask_img),
      cvt_black2white,
      cvt_BGR2HSV,
      do_img_segmentation,
      do_smoothing,
      cvt_HSV2BGR,
      cvt_BGR2GRAY,
      revert_white_black,
      do_otsu_thresholding,
  ]


def process_img(img: MatLike):
  preprocessed_img_with_selection, preprocessed_mask = process_image_larger_shape(
      img)
