import sys
from time import sleep
import cv2
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from cv2.typing import *
from typing import *
from functools import reduce
from uuid import UUID, uuid1


IMG_UNIT = NewType("IMG_UNIT", tuple[str, MatLike])

IMG_DIR_IN: str = "./Pictures"
IMG_DIR_OUT: str = f"{IMG_DIR_IN}_out"

class Lv_Mask:
  def __init__(self, mask: MatLike, priority: int = 1000):
    self.mask = mask
    self.priority = priority


class Img_State:
  def __init__(self, path: str = "", img_unit: IMG_UNIT | None = None):
    self.preload_path = path
    self.img_unit: IMG_UNIT | None = img_unit
    self.current_priority = 1000
    self.mask_list: list[Lv_Mask] = []
    self.layered_masks: list[Lv_Mask] = None
    self.process: dict[int, int]
    self.mask_list_size: int = 0
    self.is_masks_modified = False

    self.after_process_img: MatLike | None = None

    # for caching
    self.img_with_masks: IMG_UNIT | None = None

    if img_unit is not None:
      (self.h, self.w) = img_unit[1].shape[:2]
    else:
      self.h = 0
      self.w = 0

  def undo_mask(self):
    if self.mask_list_size > 0:
      self.mask_list_size -= 1
      self.after_process_img = None
      self.is_masks_modified = True

  def redo_mask(self):
    if self.mask_list_size < len(self.mask_list):
      self.mask_list_size += 1
      self.after_process_img = None
      self.is_masks_modified = True

  def reset_img(self):
    self.mask_list = []
    self.mask_list_size = 0
    self.after_process_img = None
    self.is_masks_modified = False
    self.current_priority = 1000

  def append_mask(self, mask: MatLike):
    # remove undo masks
    self.mask_list = self.mask_list[:self.mask_list_size]

    self.mask_list.append(Lv_Mask(mask, self.current_priority))
    self.is_masks_modified = True
    self.mask_list_size += 1

  def get_current_masks(self):
    return self.mask_list[:self.mask_list_size]

  def get_img(self):
    if self.img_unit is None:
      print(f"loading {self.preload_path}")
      self.img_unit = create_img_unit(
          os.path.basename(self.preload_path), cv2.imread(self.preload_path))

    return self.img_unit

  def get_img_with_mask(self):
    if self.is_masks_modified:
      img_with_masks = create_img_unit("", self.get_img()[1])
      self.layered_masks = calculate_lv_masks(self.get_current_masks())
      for mask in self.layered_masks:
        # draw selected mask region
        if mask.priority == self.current_priority:
          img_with_masks = draw_overlap_transparently(
              img_with_masks, create_img_unit("", mask.mask), 1)

        # draw contour
        img_with_masks = draw_mask_contour(
            img_with_masks, create_img_unit("", mask.mask))
      self.img_with_masks = img_with_masks
      self.is_masks_modified = False

    return self.img_with_masks

  def process_img(self):
    if self.layered_masks is not None and self.img_unit is not None:
      self.after_process_img = proc_for_multi_masks(
          self.img_unit, self.layered_masks)[1]
      
      print("work")


def create_img_unit(name: str, img: MatLike) -> IMG_UNIT:
  return IMG_UNIT((name, img))


def create_mask(mask_size: tuple[int, int], mask_coord: tuple[tuple[int, int], tuple[int, int]]):
  start_point, end_point = mask_coord
  mask = np.zeros(mask_size, dtype=np.uint8)
  cv2.rectangle(mask, start_point, end_point, (255, 0, 0), -1)
  return mask


def load_imgs_path(path: str) -> list[Img_State]:
  files = os.listdir(path)
  img_names = map(lambda file: file, files)
  return list(map(lambda name: Img_State(f"{path}/{name}"), img_names))


def output_img(dst_dir: str, img_unit: IMG_UNIT) -> None:
  (name, img) = img_unit
  cv2.imwrite(f"{dst_dir}/{name}", img)
  print(f"{name} is done")


def wrap_imgs(img_units: list[IMG_UNIT]) -> list[Img_State]:
  img_state_list: list[Img_State] = []
  for img_unit in img_units:
    img_state_list.append(Img_State(img_unit=img_unit))
  return img_state_list


def remove_noise_by_FFT(img_unit: IMG_UNIT) -> IMG_UNIT:
  """remove noise by FFT"""
  def do_FFT(img: MatLike):
    f = np.fft.fft2(img)
    img_fft = np.fft.fftshift(f)
    return img_fft

  def do_IFFT(img_fft):
    f_ishift = np.fft.ifftshift(img_fft)
    # img_back = cv2.idft(f_ishift)
    # img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

  (name, img) = img_unit
  rows, cols, c = img.shape
  crow, ccol = rows // 2, cols // 2

  # convert to frequency
  img_fft = do_FFT(img)
  original = np.copy(img_fft)

  mask = np.ones((rows, cols, c), np.uint8)
  # mask[crow-10:crow+10, ccol-10:ccol+10] = 0

  r = 50  # radius
  center = [crow, ccol]
  x, y = np.ogrid[:rows, :cols]
  mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
  mask[mask_area] = 0

  # apply mask and inverse back to image
  img_fft = img_fft * mask
  img = do_IFFT(img_fft)

  # Convert image to 8-bit unsigned integer
  img = cv2.convertScaleAbs(img)

  return create_img_unit(name, img)


def cvt_BGR2HSV(img_unit: IMG_UNIT) -> IMG_UNIT:
  """convert BGR to HSV for better result from image enhancement"""
  (name, img) = img_unit
  return create_img_unit(name, cv2.cvtColor(img, cv2.COLOR_BGR2HSV))


def cvt_HSV2BGR(img_unit: IMG_UNIT) -> IMG_UNIT:
  """convert BGR to HSV for better result from image enhancement"""
  (name, img) = img_unit
  return create_img_unit(name, cv2.cvtColor(img, cv2.COLOR_HSV2BGR))


def cvt_BGR2GRAY(img_unit: IMG_UNIT) -> IMG_UNIT:
  """convert HSV to GRAY for better result from image enhancement"""
  (name, img) = img_unit
  return create_img_unit(name, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


def do_erosion(img_unit: IMG_UNIT) -> IMG_UNIT:
  """apply erosion to remove small noise"""
  (name, img) = img_unit
  kernel = np.ones((5, 5), np.uint8)
  return create_img_unit(name, cv2.erode(img, kernel, iterations=1))


def do_dilation(img_unit: IMG_UNIT) -> IMG_UNIT:
  (name, img) = img_unit
  kernel = np.ones((9, 9), np.uint8)
  return create_img_unit(name, cv2.dilate(img, kernel, iterations=1))


def do_img_segmentation(img_unit: IMG_UNIT) -> IMG_UNIT:
  """Find segmentation based on K means clustering"""
  (name, img) = img_unit

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

  return create_img_unit(name, res2)


def do_smoothing(img_unit: IMG_UNIT) -> IMG_UNIT:
  """smooth image by GaussianBlur"""
  (name, img) = img_unit
  kernel_size = (11, 11)
  sigma = 1
  img = cv2.GaussianBlur(img, kernel_size, sigma)
  return create_img_unit(name, img)


def do_bgr_global_thresholding_inv(img_unit: IMG_UNIT) -> IMG_UNIT:
  """smooth image by Otsu's Binarization"""
  (name, img) = img_unit

  ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

  return create_img_unit(name, thresh)


def do_bgr_global_thresholding(img_unit: IMG_UNIT) -> IMG_UNIT:
  """smooth image by Otsu's Binarization"""
  (name, img) = img_unit

  ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)

  return create_img_unit(name, thresh)


def do_hsv_global_thresholding(img_unit: IMG_UNIT) -> IMG_UNIT:
  """smooth image by Otsu's Binarization"""
  (name, img) = img_unit
  # ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO_INV)
  # return create_img_unit(name, img)
  # Define lower/upper color
  # H: 0 - 180
  # S: 0 - 255
  # V: 0 - 255
  lower = np.array([0, 0, 3])
  upper = np.array([180, 255, 240])

  # Check the region of the image actually with a color in the range defined below
  # inRange returns a matrix in black and white
  bw = cv2.inRange(img, lower, upper)
  return create_img_unit(name, bw)


def do_adaptive_threshold(img_unit: IMG_UNIT) -> IMG_UNIT:
  (name, img) = img_unit
  img_threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 21, 2)
  return create_img_unit(name, img_threshold)


def do_adaptive_threshold_inv(img_unit: IMG_UNIT) -> IMG_UNIT:
  (name, img) = img_unit
  img_threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 21, 2)
  return create_img_unit(name, img_threshold)


def do_otsu_thresholding(img_unit: IMG_UNIT) -> IMG_UNIT:
  (name, img) = img_unit

  ret3, th3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  return create_img_unit(name, th3)


def draw_overlap_transparently(img_unit: IMG_UNIT,
                               img_unit_overlapped: IMG_UNIT,
                               alpha: float = 0.5,
                               color: Tuple[int, int, int] = (255, 0, 0)) -> IMG_UNIT:
  (name, img) = img_unit
  (_, img_overlapped) = img_unit_overlapped

  # Ensure the mask is in the same size as the image
  mask_resized = cv2.resize(img_overlapped, (img.shape[1], img.shape[0]))

  # Normalize mask to range [0, 1]
  mask_normalized = mask_resized / 255.0

  # Create a blue version of the mask
  mask = np.zeros_like(img, dtype=np.uint8)
  # mask[:, :, 0] = mask_resized  # Assign the mask to the blue channel
  for i in range(3):
    mask[:, :, i] = (mask_normalized * color[i]).astype(np.uint8)

  # Blend the image and the mask
  overlay = cv2.addWeighted(img, 1, mask, alpha, 0)

  return create_img_unit(name, overlay)


def draw_mask_contour(img_unit: IMG_UNIT, img_unit_overlapped: IMG_UNIT, bgr: tuple[int, int, int] = (0, 255, 0), thickness: int = 1) -> IMG_UNIT:
  # BUG: it will affect the original image, need to be fix
  (name, img) = img_unit
  (_, img_overlapped) = img_unit_overlapped

  if len(img_overlapped.shape) == 3:
    img_overlapped = cv2.cvtColor(img_overlapped, cv2.COLOR_BGR2GRAY)

  ret, thresh = cv2.threshold(img_overlapped, 127, 255, 0)
  contours, _ = cv2.findContours(
      thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(img, contours, -1, bgr, thickness)

  return create_img_unit(name, img)


def mask_with_rectangle(img_unit: IMG_UNIT) -> IMG_UNIT:
  (name, img) = img_unit
  (rows, cols) = img.shape[:2]
  mask = np.zeros(img.shape[:2], dtype="uint8")
  cv2.rectangle(mask, (90, 90), (cols-200, rows-150), 255, -1)
  masked = cv2.bitwise_and(img, img, mask=mask)

  return create_img_unit(name, masked)


def mask_with_shape(mask_img_unit: IMG_UNIT) -> Callable[[IMG_UNIT], IMG_UNIT]:
  def mask_it(img_unit: IMG_UNIT) -> IMG_UNIT:
    if mask_img_unit is None:
      return img_unit

    (name, img) = img_unit
    (_, mask) = mask_img_unit

    if len(mask.shape) == 3:
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Ensure the mask is of type CV_8U
    if mask.dtype != np.uint8:
      mask = mask.astype(np.uint8)

    # Ensure the mask is the same size as the image
    if mask.shape[:2] != img.shape[:2]:
      mask = cv2.resize(
          mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    masked = cv2.bitwise_and(img, img, mask=mask)

    return create_img_unit(name, masked)

  return mask_it


def mask_with_shape_not(mask_img_unit: IMG_UNIT) -> Callable[[IMG_UNIT], IMG_UNIT]:
  def mask_it(img_unit: IMG_UNIT) -> IMG_UNIT:
    if mask_img_unit is None:
      return img_unit

    (name, img) = img_unit
    (_, mask) = mask_img_unit

    if len(mask.shape) == 3:
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Ensure the mask is of type CV_8U
    if mask.dtype != np.uint8:
      mask = mask.astype(np.uint8)

    # Ensure the mask is the same size as the image
    if mask.shape[:2] != img.shape[:2]:
      mask = cv2.resize(
          mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    masked = cv2.bitwise_not(img, img, mask=mask)

    return create_img_unit(name, masked)

  return mask_it


def contrast_enhancement(img_unit: IMG_UNIT) -> IMG_UNIT:
  (name, img) = img_unit
  # control Contrast by 1.5
  alpha = 1
  # control brightness by 50
  beta = 50
  img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
  return create_img_unit(name, img)


def cvt_black2white(img_unit: IMG_UNIT) -> IMG_UNIT:
  (name, img) = img_unit
  # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
  black_pixels = np.where(
      (img[:, :, 0] == 0) &
      (img[:, :, 1] == 0) &
      (img[:, :, 2] == 0)
  )

  # set those pixels to white
  img[black_pixels] = [255, 255, 255]

  return create_img_unit(name, img)


def revert_white_black(img_unit: IMG_UNIT) -> IMG_UNIT:
  (name, img) = img_unit

  img = cv2.bitwise_not(img, img)

  return create_img_unit(name, img)


def proc_for_white_bg(img_unit: IMG_UNIT, mask_img_unit: IMG_UNIT) -> IMG_UNIT:
  new_process = [
      do_erosion,
      do_smoothing,
      contrast_enhancement,
      remove_noise_by_FFT,
      mask_with_shape(mask_img_unit),
      cvt_black2white,
      cvt_BGR2HSV,
      do_img_segmentation,
      do_smoothing,
      cvt_HSV2BGR,
      cvt_BGR2GRAY,
      revert_white_black,
      do_otsu_thresholding,
  ]

  img_unit_mask = reduce(lambda img, func: func(img), new_process, img_unit)

  return img_unit_mask


def proc_for_multi_masks(img_unit: IMG_UNIT, mask_img_unit_list: list[Lv_Mask]) -> IMG_UNIT:
  # TODO
  (name, img) = img_unit
  result_mask: MatLike = None
  for mask_img_unit in mask_img_unit_list:
    (_, img_mask) = proc_for_white_bg(
        img_unit, create_img_unit("", mask_img_unit.mask))
    if result_mask is None:
      result_mask = img_mask
    else:
      result_mask = cv2.bitwise_or(result_mask, img_mask)

  # count white pixel
  n_white_px = np.sum(result_mask != 0)

  # 135 pixel per millimeter
  ratio = 140
  len_per_px_side = 1.0 / ratio
  area_per_px = len_per_px_side ** 2

  print(f"{name} has {n_white_px} pixels, area is {area_per_px * n_white_px}")

  img_unit_mask = create_img_unit(name, result_mask)
  output_img(IMG_DIR_OUT, img_unit_mask)

  img_unit_labelled = draw_overlap_transparently(img_unit, img_unit_mask, 1)
  img_unit_labelled = draw_mask_contour(
      img_unit_labelled, img_unit_mask, (0, 0, 255))
  output_img(IMG_DIR_OUT, create_img_unit(
      f"{name}_labelled.jpg", img_unit_labelled[1]))

  return img_unit_labelled


def calculate_lv_masks(masks: list[Lv_Mask]) -> list[Lv_Mask]:
  layered_masks: list[list[Lv_Mask]] = []
  new_masks: list[Lv_Mask] = []
  result_masks: list[Lv_Mask] = []
  priority_set: set[int] = set()

  # identify how many layers
  for mask in masks:
    priority_set.add(mask.priority)

  # seperate mask into its own layer
  for priority in priority_set:
    layered_masks.append(
        list(
            filter(
                lambda mask: mask.priority == priority,
                masks)))

  # combine the masks for each layer
  for masks in layered_masks:
    # TODO: fix type issue
    new_mask: MatLike = None
    for mask in masks:
      if new_mask is None:
        new_mask = mask.mask.copy()
      new_mask = cv2.bitwise_or(new_mask, mask.mask)
    new_masks.append(Lv_Mask(new_mask.copy(), mask.priority))

    # reset the mask
    new_mask = np.zeros_like(new_mask)

  if len(priority_set) > 1:
    # TODO: calculate the overlapped area and remove it
    for mask in new_masks:
      tmp_mask = mask.mask
      for other_mask in new_masks:
        if mask is other_mask:
          continue

        # priority: higher priority take the part
        if mask.priority < other_mask.priority:
          tmp_mask = cv2.bitwise_xor(tmp_mask, other_mask.mask, mask=tmp_mask)

      result_masks.append(Lv_Mask(tmp_mask, mask.priority))
  else:
    result_masks = new_masks

  return result_masks
