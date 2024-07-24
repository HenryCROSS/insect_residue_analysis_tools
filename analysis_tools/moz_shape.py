import cv2
import os
import numpy as np
from cv2.typing import MatLike
from typing import Tuple, List, Optional
import multiprocessing
import psutil


input_directory: str = './Pictures'
output_directory: str = './try_out'


class Image:
  def __init__(self, path: str, name: str):
    self._path: str = path
    self._name: str = name
    self._image: Optional[MatLike] = None

  def get_img(self) -> Optional[MatLike]:
    if self._image is None:
      input_path: str = os.path.join(self._path, self._name)
      self._image = cv2.imread(input_path)
    return self._image

  def get_path(self) -> str:
    return self._path

  def get_name(self) -> str:
    return self._name

  def get_full_path(self) -> str:
    return os.path.join(self._path, self._name)


def apply_mosaic_min(image: MatLike, block_size: int) -> MatLike:
  if block_size < 1:
    block_size = 1

  (h, w) = image.shape[:2]
  for y in range(0, h, block_size):
    for x in range(0, w, block_size):
      block = image[y:y + block_size, x:x + block_size]
      (B, G, R) = np.min(block.reshape(-1, 3), axis=0)
      image[y:y + block_size, x:x + block_size] = (B, G, R)
  return image


def apply_mosaic_mean(image: MatLike, block_size: int) -> MatLike:
  if block_size < 1:
    block_size = 1

  (h, w) = image.shape[:2]
  for y in range(0, h, block_size):
    for x in range(0, w, block_size):
      block = image[y:y + block_size, x:x + block_size]
      (B, G, R) = np.mean(block.reshape(-1, 3), axis=0).astype(int)
      image[y:y + block_size, x:x + block_size] = (B, G, R)
  return image


def apply_mosaic_white_if_white(image: MatLike, block_size: int) -> MatLike:
  if block_size < 1:
    block_size = 1

  (h, w) = image.shape[:2]
  for y in range(0, h, block_size):
    for x in range(0, w, block_size):
      block = image[y:y + block_size, x:x + block_size]
      if np.any(block == 255):
        image[y:y + block_size, x:x + block_size] = 255
  return image


def remove_small_circles(image: MatLike, min_radius: int) -> MatLike:
  contours, _ = cv2.findContours(
      image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  mask = np.ones_like(image) * 255
  for contour in contours:
    area = cv2.contourArea(contour)
    radius = int(np.sqrt(area / np.pi))
    if radius < min_radius:
      cv2.drawContours(mask, [contour], -1, 0, thickness=cv2.FILLED)
  result = cv2.bitwise_and(image, mask)
  return result


def find_and_remove_black_rectangles(image: MatLike, binary_image: MatLike, min_length: int) -> MatLike:
  if len(image.shape) == 3:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    gray_image = image

  _, thresh_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
  contours, _ = cv2.findContours(
      thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if max(w, h) > min_length:
      cv2.rectangle(binary_image, (x, y), (x + w, y + h), 0, -1)
  return binary_image


def process_image_larger_shape(img: MatLike) -> Tuple[MatLike, MatLike]:
  # deepen the colour
  block_size: int = 5
  mosaci_min: MatLike = apply_mosaic_min(img.copy(), block_size)

  # smooth the background
  mosaci_mean: MatLike = apply_mosaic_mean(mosaci_min.copy(), block_size * 2)

  # detect edge by Canny
  gray_mosaci_mean: MatLike = cv2.cvtColor(
      mosaci_mean.copy(), cv2.COLOR_BGR2GRAY)
  canny_edges: MatLike = cv2.Canny(gray_mosaci_mean, 100, 200)

  # remove small dots
  kernel: MatLike = np.ones((5, 5), np.uint8)
  dilated_edges: MatLike = cv2.dilate(canny_edges, kernel, iterations=1)
  closed_edges: MatLike = cv2.morphologyEx(
      dilated_edges, cv2.MORPH_CLOSE, kernel)
  eroded_edges: MatLike = cv2.erode(closed_edges, kernel, iterations=1)

  # remove the black long rectangle if exist
  min_length: int = 1000
  mosaic_white: MatLike = apply_mosaic_white_if_white(eroded_edges, block_size)
  mosaic_white = find_and_remove_black_rectangles(
      gray_mosaci_mean, mosaic_white, min_length)

  # remove bigger dots
  min_radius: int = 25
  cleaned_edges: MatLike = remove_small_circles(mosaic_white, min_radius)

  # Set the specified rectangle areas to black
  rectangles: List[Tuple[int, int, int, int]] = [
      (0, 1458, 319, 1535),
      (1722, 1359, 2022, 1519)
  ]
  for (x1, y1, x2, y2) in rectangles:
    cleaned_edges[y1:y2, x1:x2] = 0

  # Apply the mosaic effect: if there is any white pixel, make the block white
  final_result: MatLike = apply_mosaic_white_if_white(
      cleaned_edges, block_size * 5)

  # Dilate and close to connect nearby white regions
  kernel = np.ones((15, 15), np.uint8)
  img_dilated: MatLike = cv2.dilate(final_result, kernel, iterations=1)
  img_closed: MatLike = cv2.morphologyEx(img_dilated, cv2.MORPH_CLOSE, kernel)

  # Find contours on the final result
  contours, _ = cv2.findContours(
      img_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Draw filled contours to remove holes on the mask
  mask_with_contours: MatLike = np.zeros_like(img_closed)
  cv2.drawContours(mask_with_contours, contours, -1, 255, thickness=cv2.FILLED)

  # Convert to BGR for color drawing
  mask_bgr: MatLike = cv2.cvtColor(mask_with_contours, cv2.COLOR_GRAY2BGR)

  # Overlay the mask with contours on the original image
  overlayed_image: MatLike = cv2.addWeighted(img, 0.8, mask_bgr, 0.2, 0)

  # Draw contours with deeper red color on the overlayed image
  deep_red_color: Tuple[int, int, int] = (0, 0, 255)  # Dark red
  cv2.drawContours(overlayed_image, contours, -1, deep_red_color, 2)

  return overlayed_image, mask_bgr

def process_image_detail_shape(img: MatLike, mask: MatLike) -> Tuple[MatLike, MatLike]:
  # 将mask转换为灰度图
  gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
  # 提取mask区域内的像素
  mask_pixels = img[gray_mask != 0]

  # 找到mask区域内出现最多的颜色
  colors, counts = np.unique(mask_pixels.reshape(-1, 3), axis=0, return_counts=True)
  most_frequent_color = colors[counts.argmax()]

  # 创建一个全白图像用于后续操作
  new_img = img.copy()

  # 将mask区域外的像素替换为出现最多的颜色
  new_img[gray_mask == 0] = most_frequent_color

  new_img = apply_mosaic_mean(new_img, 10)

  # 返回处理后的图像和mask
  return new_img, mask

def process_image(src: Image) -> Tuple[Optional[MatLike], Optional[MatLike]]:
  origianl_img = src.get_img()
  if origianl_img is None:
    print(f"Failed to load image {src.get_full_path()}")
    return None, None
  
  large_overlayed_image, large_mask_bgr = process_image_larger_shape(origianl_img)
  detailed_overlayed_image, detailed_mask_bgr = process_image_detail_shape(origianl_img, large_mask_bgr)

  return detailed_overlayed_image, detailed_mask_bgr

def calculate_area(mask: MatLike) -> float:
  return 0.0

def process_and_save_image(img: Image, out_dir: str) -> None:
  overlayed_image, mask_bgr = process_image(img)


  if overlayed_image is not None and mask_bgr is not None:
    area = calculate_area(mask_bgr)
    print(f"Area of {img.get_name()} = {area}")
    cv2.imwrite(f"{out_dir}/{img.get_name()}", overlayed_image)
    cv2.imwrite(f"{out_dir}/{img.get_name()}_mask.jpg", mask_bgr)
    print(f"Finished {out_dir}/{img.get_name()}")


def main(in_dir: str, out_dir: str) -> None:
  if not os.path.exists(in_dir):
    print("input directory not exist")
    return

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  images: List[Image] = []

  for filename in os.listdir(in_dir):
    if filename.endswith((".png", ".jpg", ".jpeg")):
      images.append(Image(in_dir, filename))

  # calculate processors number
  # number = (RAM(GB) / 2) + 1
  mem_gb = psutil.virtual_memory().total / (1024 ** 3)
  num_processes = int(mem_gb / 2) + 1

  # process images at the same time
  with multiprocessing.Pool(num_processes) as pool:
    pool.starmap(process_and_save_image, [(img, out_dir) for img in images])


if __name__ == "__main__":
  main(input_directory, output_directory)
