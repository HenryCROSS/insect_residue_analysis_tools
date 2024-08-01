import cv2
import numpy as np
import os
from typing import List, Tuple
import multiprocessing
import psutil
from functools import reduce

IMG_DIR_IN: str = "./Pictures"
IMG_DIR_OUT: str = "./Pictures_1_out"


class Image:
  def __init__(self, path, filename):
    self.path = path
    self.filename = filename
    self.img = None
    self.preprocessed_shape_mask = None
    self.preprocessed_shape = None
    self.shape_mask = None

  def get_img(self):
    if self.img is None:
      self.img = cv2.imread(os.path.join(self.path, self.filename))
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


def remove_noise_by_FFT(img):
  def do_FFT(img):
    f = np.fft.fft2(img)
    img_fft = np.fft.fftshift(f)
    return img_fft

  def do_IFFT(img_fft):
    f_ishift = np.fft.ifftshift(img_fft)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

  rows, cols, c = img.shape
  crow, ccol = rows // 2, cols // 2

  img_fft = do_FFT(img)
  mask = np.ones((rows, cols, c), np.uint8)

  r = 25
  center = [crow, ccol]
  x, y = np.ogrid[:rows, :cols]
  mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
  mask[mask_area] = 0

  img_fft = img_fft * mask
  img = do_IFFT(img_fft)

  img = cv2.convertScaleAbs(img)

  return img


def do_smoothing(img):
  kernel_size = (11, 11)
  sigma = 1
  return cv2.GaussianBlur(img, kernel_size, sigma)


def contrast_enhancement(img):
  alpha = 1
  beta = 50
  return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def mask_with_shape(mask_img):
  def mask_it(img):
    if len(mask_img.shape) == 3:
      mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
      mask = mask_img

    if mask.dtype != np.uint8:
      mask = mask.astype(np.uint8)

    if mask.shape[:2] != img.shape[:2]:
      mask = cv2.resize(
          mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return cv2.bitwise_and(img, img, mask=mask)
  return mask_it


def do_erosion(img):
  kernel = np.ones((5, 5), np.uint8)
  return cv2.erode(img, kernel, iterations=1)


def cvt_black2white(img):
  if len(img.shape) == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  black_pixels = np.where((img[:, :, 0] == 0) & (
      img[:, :, 1] == 0) & (img[:, :, 2] == 0))
  img[black_pixels] = [255, 255, 255]
  return img


def cvt_BGR2HSV(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def do_img_segmentation(img):
  Z = img.reshape((-1, 3))
  Z_f32 = np.float32(Z)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 8
  _, label, center = cv2.kmeans(
      Z_f32, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  res = center[label.flatten()]
  return res.reshape((img.shape))


def cvt_HSV2BGR(img):
  return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def cvt_BGR2GRAY(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def revert_white_black(img):
  return cv2.bitwise_not(img)


def do_otsu_thresholding(img):
  _, th3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  return th3

def apply_mosaic_min(image, block_size: int):
  if block_size < 1:
    block_size = 1

  (h, w) = image.shape[:2]
  for y in range(0, h, block_size):
    for x in range(0, w, block_size):
      block = image[y:y + block_size, x:x + block_size]
      (B, G, R) = np.min(block.reshape(-1, 3), axis=0)
      image[y:y + block_size, x:x + block_size] = (B, G, R)
  return image

def enhance_contrast_histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    Enhance the contrast of an image using Histogram Equalization.

    Args:
    - img: Input image (BGR or grayscale).

    Returns:
    - Enhanced contrast image.
    """
    # Convert the image to grayscale if it is in BGR format
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # Apply Histogram Equalization
    enhanced_img = cv2.equalizeHist(img_gray)

    # If the input image was in BGR, convert the enhanced grayscale image back to BGR
    if len(img.shape) == 3 and img.shape[2] == 3:
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)

    return enhanced_img

def filter_contours_by_area(image: np.ndarray, min_area: int, fill: bool = True) -> np.ndarray:
    """
    Filter contours by area.

    Parameters:
    - image: Input binary image (single channel).
    - min_area: Minimum area of contours to retain.
    - fill: Whether to fill the interior of the contours. Default is True.

    Returns:
    - Filtered image containing only contours with area greater than or equal to min_area.
    """
    # Find all external contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_img = np.zeros_like(image)

    # Filter contours and draw them on the new image
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            if fill:
                cv2.drawContours(filtered_img, [contour], -1, 255, thickness=cv2.FILLED)
            else:
                cv2.drawContours(filtered_img, [contour], -1, 255, thickness=1)

    return filtered_img

def process_image_larger_shape(img):
  img_color = img.copy()
  img = cv2.bitwise_not(img)


  # mosaic_min_img = apply_mosaic_min(img, 3)
  enhanced_img = enhance_contrast_histogram_equalization(img)
  blurred_img = cv2.GaussianBlur(enhanced_img, (7, 7), 0)
  fft_img = remove_blur_fft(blurred_img)

  kernel = np.ones((3, 3), np.uint8)
  eroded_img = cv2.erode(fft_img, kernel, iterations=1)

  clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
  cl1 = clahe.apply(eroded_img)

  otsu_thresh_value, otsu_img = cv2.threshold(
      cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  kernel = np.ones((3, 3), np.uint8)
  cleaned_otsu_img = cv2.morphologyEx(otsu_img, cv2.MORPH_OPEN, kernel)

  kernel = np.ones((10, 10), np.uint8)
  filter_mask = cv2.dilate(cleaned_otsu_img, kernel, iterations=1)
  connected_img = cv2.morphologyEx(filter_mask, cv2.MORPH_CLOSE, kernel)

  contours, _ = cv2.findContours(
      connected_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  solid_polygon_img = np.zeros_like(connected_img)
  cv2.fillPoly(solid_polygon_img, contours, 255)

  min_area = 2000
  contours, _ = cv2.findContours(
      solid_polygon_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  filtered_img = np.zeros_like(solid_polygon_img)
  for contour in contours:
    if cv2.contourArea(contour) >= min_area:
      cv2.drawContours(filtered_img, [contour], -1, 255, thickness=cv2.FILLED)

  blue_layer = np.zeros_like(img_color)
  blue_layer[:, :] = (255, 0, 0)
  alpha = 0.2

  mask_bool = filtered_img.astype(bool)
  img_color[mask_bool] = cv2.addWeighted(
      img_color, 1 - alpha, blue_layer, alpha, 0)[mask_bool]

  contours, hierarchy = cv2.findContours(
      filtered_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  for i in range(len(contours)):
    cv2.drawContours(img_color, contours, i, (0, 0, 255), 2)

  return img_color, filtered_img


def generate_center_mask(img, top_pct=10.0, bottom_pct=10.0, left_pct=10.0, right_pct=10.0):
  height, width = img.shape[:2]

  top = int(height * top_pct / 100)
  bottom = int(height * bottom_pct / 100)
  left = int(width * left_pct / 100)
  right = int(width * right_pct / 100)

  mask = np.zeros((height, width), dtype=np.uint8)
  mask[top:height - bottom, left:width - right] = 255

  return mask


def detail_process_img(img, mask_img):
  new_process = [
      do_erosion,
      # enhance_contrast_histogram_equalization,
      do_smoothing,
      contrast_enhancement,
      # enhance_contrast_histogram_equalization,
      remove_noise_by_FFT,
      # lambda x: apply_mosaic_min(x, 3),
      mask_with_shape(mask_img),
      cvt_black2white,
      cvt_BGR2HSV,
      do_img_segmentation,
      do_smoothing,
      cvt_HSV2BGR,
      cvt_BGR2GRAY,
      revert_white_black,
      do_otsu_thresholding,
      # lambda x: filter_contours_by_area(x, 500, True),
  ]

  img_mask = reduce(lambda img, func: func(img), new_process, img)
  return img_mask


def draw_overlap_transparently(img, img_overlapped, alpha=0.5, color=(255, 0, 0)):
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

  return overlay


def draw_mask_contour(img, img_overlapped, bgr=(0, 255, 0), thickness=1):
  if len(img_overlapped.shape) == 3:
    img_overlapped = cv2.cvtColor(img_overlapped, cv2.COLOR_BGR2GRAY)

  ret, thresh = cv2.threshold(img_overlapped, 127, 255, 0)
  contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(img, contours, -1, bgr, thickness)

  return img


def process_and_save_image(image: Image, out_dir: str):
  img = image.get_img()
  if img is None:
    print(f"Failed to load image at {os.path.join(image.path, image.filename)}")
    return

  preprocessed_img_with_selection, preprocessed_mask = process_image_larger_shape(
      img)

  rect_mask = generate_center_mask(
      img, 3, 10, 3, 3)
  img_mask = cv2.bitwise_and(rect_mask, rect_mask, mask=preprocessed_mask)

  result_mask = detail_process_img(img, img_mask)

  img_labelled = draw_overlap_transparently(img, result_mask, 1)
  img_labelled = draw_mask_contour(img_labelled, preprocessed_mask, (235, 32, 37), 1)
  img_labelled = draw_mask_contour(img_labelled, result_mask, (0, 0, 255), 1)
  img_labelled = draw_mask_contour(img_labelled, rect_mask, (0, 182, 235), 1)

  output_path = os.path.join(out_dir, image.filename)
  cv2.imwrite(output_path, result_mask)
  cv2.imwrite(f"{output_path}_label.jpg", img_labelled)
  print(f"Processed image saved at {output_path}")


def main(in_dir: str, out_dir: str) -> None:
  if not os.path.exists(in_dir):
    print("Input directory does not exist")
    return

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  images: List[Image] = [Image(in_dir, filename) for filename in os.listdir(
      in_dir) if filename.endswith((".png", ".jpg", ".jpeg"))]

  mem_gb = psutil.virtual_memory().total / (1024 ** 3)
  num_processes = int(mem_gb / 2) + 1

  with multiprocessing.Pool(num_processes) as pool:
    pool.starmap(process_and_save_image, [
                 (img, out_dir) for img in images])


if __name__ == "__main__":
  main(IMG_DIR_IN, IMG_DIR_OUT)
