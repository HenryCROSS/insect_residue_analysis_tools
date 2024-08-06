import cv2
import numpy as np
import os
from cv2.typing import *
from typing import *
from functools import reduce


IMG_DIR_IN: str = "./Pictures"
IMG_DIR_OUT: str = "./Pictures_out"
IMG_UNIT = NewType("IMG_UNIT", tuple[str, MatLike])


class Lv_Mask:
  def __init__(self, mask: MatLike, priority: int = 1000):
    self.mask = mask
    self.priority = priority


class Img_State:
  def __init__(self, img_unit: IMG_UNIT):
    self.img_unit: IMG_UNIT = img_unit
    self.after_process_img: MatLike | None = None
    self.current_priority = 1000
    self.mask_list: list[Lv_Mask] = []
    self.preprocessed_img = None
    self.preprocessed_img_mask = None
    self.mask_list_size: int = 0
    (self.h, self.w) = img_unit[1].shape[:2]

  def undo_mask(self):
    if self.mask_list_size > 0:
      self.mask_list_size -= 1
      self.after_process_img = None

  def redo_mask(self):
    if self.mask_list_size < len(self.mask_list):
      self.mask_list_size += 1
      self.after_process_img = None

  def reset_img(self):
    self.mask_list = []
    self.mask_list_size = 0
    self.after_process_img = None
    self.current_priority = 1000

  def append_mask(self, mask: MatLike):
    # remove undo masks
    self.mask_list = self.mask_list[:self.mask_list_size]

    self.mask_list.append(Lv_Mask(mask, self.current_priority))
    self.mask_list_size += 1

  def get_current_masks(self):
    return self.mask_list[:self.mask_list_size]

  def preprocessing(self):
    if self.preprocessed_img_mask is None:
      self.preprocessed_img, self.preprocessed_img_mask = process_image_larger_shape(
          self.img_unit[1])


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


def create_img_unit(name: str, img: MatLike) -> IMG_UNIT:
  return IMG_UNIT((name, img))


def load_imgs(path: str) -> list[IMG_UNIT]:
  files = os.listdir(path)
  img_names = map(lambda file: file, files)
  return list(map(lambda name: create_img_unit(name, cv2.imread(f"{path}/{name}")), img_names))


def output_img(dst_dir: str, img_unit: IMG_UNIT) -> None:
  (name, img) = img_unit
  cv2.imwrite(f"{dst_dir}/{name}", img)
  print(f"{name} is done")


def wrap_imgs(img_units: list[IMG_UNIT]) -> list[Img_State]:
  img_state_list: list[Img_State] = []
  for img_unit in img_units:
    img_state_list.append(Img_State(img_unit))
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

  r = 25  # radius
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


def change_HSV_value(value: int):
  def change_to(img_unit: IMG_UNIT) -> IMG_UNIT:
    (name, img) = img_unit

    img[:, :, 2] = value

    return img_unit
  return change_to


def cvt_BGR2GRAY(img_unit: IMG_UNIT) -> IMG_UNIT:
  """convert HSV to GRAY for better result from image enhancement"""
  (name, img) = img_unit
  return create_img_unit(name, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


def do_erosion(img_unit: IMG_UNIT) -> IMG_UNIT:
  """apply erosion to remove small noise"""
  (name, img) = img_unit
  kernel = np.ones((5, 5), np.uint8)
  image = create_img_unit(name, cv2.erode(img, kernel, iterations=1))
  cv2.imshow("test", image[1].copy())
  return image


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

  # control Contrast by 1
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


def mask_masks(masks: list[Lv_Mask], restricted_mask: MatLike) -> list[Lv_Mask]:
    new_masks = []

    for mask in masks:
        # Ensure the mask is single channel
        if len(mask.mask.shape) == 3:
            mask_mask = cv2.cvtColor(mask.mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_mask = mask.mask

        # Ensure the mask is of type CV_8U
        if mask_mask.dtype != np.uint8:
            mask_mask = mask_mask.astype(np.uint8)

        # Ensure the mask is the same size as the restricted_mask
        if mask_mask.shape[:2] != restricted_mask.shape[:2]:
            mask_mask = cv2.resize(mask_mask, (restricted_mask.shape[1], restricted_mask.shape[0]))

        new_masks.append(Lv_Mask(cv2.bitwise_and(mask_mask, mask_mask, mask=restricted_mask), mask.priority))
    
    return new_masks



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


def image_resize(image: MatLike, width=None, height=None, inter=cv2.INTER_AREA) -> MatLike:
  # TODO: function need to be changed
  # initialize the dimensions of the image to be resized and
  # grab the image size
  dim = None
  (h, w) = image.shape[:2]

  # if both the width and height are None, then return the
  # original image
  if width is None and height is None:
    return image

  # check to see if the width is None
  if width is None:
    # calculate the ratio of the height and construct the
    # dimensions
    r = height / float(h)
    dim = (int(w * r), int(height))

  # otherwise, the height is None
  else:
    # calculate the ratio of the width and construct the
    # dimensions
    r = width / float(w)
    dim = (int(width), int(h * r))

  # resize the image
  resized: MatLike = cv2.resize(image, dim, interpolation=inter)

  # return the resized image
  return resized


def manage_img(img: Img_State) -> Img_State:
  img.preprocessing()
  (name, image) = img.img_unit
  original_image = image.copy()
  (scaled_height, scaled_width) = image.shape[:2]

  def resize_image(image, scale_percent):
    nonlocal scaled_width, scaled_height
    (my, mx) = image.shape[:2]
    scaled_width = int(mx * scale_percent / 100)
    scaled_height = int(my * scale_percent / 100)
    resized_image = cv2.resize(
        image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
    return resized_image

  image = resize_image(image, 50)
  preprocessed_img = resize_image(img.preprocessed_img, 50)
  preprocessed_img_mask = resize_image(img.preprocessed_img_mask, 50)
  scale_back_percent = 100 / 50 * 100
  temp_image = preprocessed_img
  drawing = False
  start_point = None  # starting point of rectangle

  def draw_square(event, x, y, flags, param):
    nonlocal start_point, drawing, image, temp_image, scaled_height, scaled_width

    if event == cv2.EVENT_LBUTTONDOWN:
      drawing = True
      start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
      if drawing:
        if x > scaled_width:
          x = scaled_width - 1
        elif x < 0:
          x = 0

        if y > scaled_height:
          y = scaled_height - 1
        elif y < 0:
          y = 0

        temp_image = preprocessed_img.copy()
        cv2.rectangle(temp_image, start_point, (x, y), (0, 0, 255), 1)

    elif event == cv2.EVENT_LBUTTONUP:
      drawing = False
      temp_image = preprocessed_img.copy()
      mask = np.zeros(image.shape[:2], dtype=np.uint8)
      cv2.rectangle(mask, start_point, (x, y), 255, -1)
      img.append_mask(mask)

  # set window name and mouse callback
  cv2.namedWindow('Image')
  cv2.setMouseCallback('Image', draw_square)

  print(f"enter edit mode for {name}")

  while True:
    layered_masks = None
    if drawing:
      cv2.imshow('Image', temp_image)
    else:
      if img.mask_list_size < 1:
        cv2.imshow('Image', temp_image)
      else:
        show_img_unit = create_img_unit("", preprocessed_img.copy())
        layered_masks = calculate_lv_masks(img.get_current_masks())
        for mask in layered_masks:
          # draw selected mask region
          if mask.priority == img.current_priority:
            show_img_unit = draw_overlap_transparently(
                show_img_unit, create_img_unit("", cv2.bitwise_and(mask.mask, mask.mask, mask=preprocessed_img_mask)), 1)

          # draw contour
          show_img_unit = draw_mask_contour(
              show_img_unit, create_img_unit("", mask.mask))

        (_, show_img) = show_img_unit
        cv2.imshow('Image', show_img)

    # handle the event
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC # exit
      break

    elif key == 117:  # u # undo mask
      img.undo_mask()
      print(f"undo mask, mask number {img.mask_list_size}")

    elif key == 114:  # r # redo mask
      img.redo_mask()
      print(f"redo mask, mask number {img.mask_list_size}")

    elif key == 108:  # l # lower priority
      img.current_priority -= 1
      print(f"priority: {img.current_priority}")

    elif key == 104:  # h # higher priority
      img.current_priority += 1
      print(f"priority: {img.current_priority}")

    elif key == 32:  # Space # process image by different mask and return
      # TODO process image, save the result, return to preview mode
      if layered_masks is not None:
        print("processing!")
        img.after_process_img = proc_for_multi_masks(
            img.img_unit, mask_masks(layered_masks, img.preprocessed_img_mask))[1]
        print("finish!")
      break

  print(f"quit edit mode for {name}")

  return img


def manage_imgs(imgs: list[Img_State]) -> None:
  idx: int = 0
  current_img = imgs[idx]
  resize_ratio: float = 0.5
  print(f"preview {current_img.img_unit[0]}")

  while True:
    cv2.namedWindow('Image')
    # remove mouse callback
    cv2.setMouseCallback('Image', lambda *args: None)

    if current_img.after_process_img is None:
      cv2.imshow('Image', image_resize(
          current_img.img_unit[1], current_img.w * resize_ratio))
    else:
      cv2.imshow('Image', image_resize(
          current_img.after_process_img, current_img.w * resize_ratio))

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC # exit
      break

    elif key == 97:  # a # prev img
      if idx == 0:
        idx = len(imgs) - 1
      else:
        idx -= 1
      current_img = imgs[idx]
      print(f"preview {current_img.img_unit[0]}")

    elif key == 100:  # d # next img
      if idx == len(imgs) - 1:
        idx = 0
      else:
        idx += 1
      current_img = imgs[idx]
      print(f"preview {current_img.img_unit[0]}")

    elif key == 32:  # Space # process image
      manage_img(current_img)


# def main():
#   # load images from IMG_DIR_IN
#   imgs = load_imgs(IMG_DIR_IN)
#   imgs = wrap_imgs(imgs)
#   print(f"loaded {len(imgs)} images!")
#   manage_imgs(imgs)

  # for img in imgs:
  #   (img_unit, mask_img_unit) = drawing(img)

  #   proc_for_white_bg(img_unit, mask_img_unit)

  # speed up by applying multi processing
  # with ProcessPoolExecutor() as executor:
  #   futures = [executor.submit(proc, img, None) for img in imgs]
  #   for future in futures:
  #     # waiting for each to complete, handle exceptions if needed
  #     future.result()

if __name__ == "__main__":
  imgs = load_imgs(IMG_DIR_IN)
  imgs_state = wrap_imgs(imgs)
  print(f"loaded {len(imgs_state)} images!")
  manage_imgs(imgs_state)
