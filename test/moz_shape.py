import os
import cv2
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


def apply_fft_to_region(image, mask):
    # 对于每个独立的区域，进行FFT处理
  result_image = image.copy()
  contours, _ = cv2.findContours(
      mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  for contour in contours:
      # 计算轮廓的边界框
    x, y, w, h = cv2.boundingRect(contour)
    region = image[y:y+h, x:x+w]
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)  # 将区域转换为灰度图像
    fft_region = remove_blur_fft(gray_region)
    result_image[y:y+h, x:x +
                 w] = cv2.cvtColor(fft_region, cv2.COLOR_GRAY2BGR)  # 将结果转换回BGR

  # 计算掩膜区域的平均颜色
  fft_masked_area = cv2.bitwise_and(result_image, result_image, mask=mask)
  mean_color = cv2.mean(fft_masked_area, mask)[:3]

  # 将检测到的区域外部变成掩膜区域FFT后的平均颜色
  inverse_mask = cv2.bitwise_not(mask)
  colored_mean = np.full_like(result_image, mean_color, dtype=np.uint8)
  result_image = cv2.bitwise_and(result_image, result_image, mask=mask)
  result_image += cv2.bitwise_and(colored_mean, colored_mean, mask=inverse_mask)

  return result_image


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
  block_size: int = 5
  mosaic_min: MatLike = apply_mosaic_min(img.copy(), block_size)
  mosaic_mean: MatLike = apply_mosaic_mean(mosaic_min.copy(), block_size * 2)
  gray_mosaic_mean: MatLike = cv2.cvtColor(
      mosaic_mean.copy(), cv2.COLOR_BGR2GRAY)
  canny_edges: MatLike = cv2.Canny(gray_mosaic_mean, 100, 200)

  kernel: MatLike = np.ones((5, 5), np.uint8)
  dilated_edges: MatLike = cv2.dilate(canny_edges, kernel, iterations=1)
  closed_edges: MatLike = cv2.morphologyEx(
      dilated_edges, cv2.MORPH_CLOSE, kernel)
  eroded_edges: MatLike = cv2.erode(closed_edges, kernel, iterations=1)

  min_length: int = 1000
  mosaic_white: MatLike = apply_mosaic_white_if_white(
      eroded_edges, block_size)
  mosaic_white = find_and_remove_black_rectangles(
      gray_mosaic_mean, mosaic_white, min_length)

  min_radius: int = 25
  cleaned_edges: MatLike = remove_small_circles(mosaic_white, min_radius)

  rectangles: List[Tuple[int, int, int, int]] = [
      (0, 1458, 319, 1535),
      (1722, 1359, 2022, 1519)
  ]
  for (x1, y1, x2, y2) in rectangles:
    cleaned_edges[y1:y2, x1:x2] = 0

  final_result: MatLike = apply_mosaic_white_if_white(
      cleaned_edges, block_size * 5)

  kernel = np.ones((15, 15), np.uint8)
  img_dilated: MatLike = cv2.dilate(final_result, kernel, iterations=1)
  img_closed: MatLike = cv2.morphologyEx(img_dilated, cv2.MORPH_CLOSE, kernel)

  contours, _ = cv2.findContours(
      img_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  mask_with_contours: MatLike = np.zeros_like(img_closed)
  cv2.drawContours(mask_with_contours, contours, -
                   1, 255, thickness=cv2.FILLED)

  mask_bgr: MatLike = cv2.cvtColor(mask_with_contours, cv2.COLOR_GRAY2BGR)
  overlayed_image: MatLike = cv2.addWeighted(img, 0.8, mask_bgr, 0.2, 0)

  deep_red_color: Tuple[int, int, int] = (0, 0, 255)
  cv2.drawContours(overlayed_image, contours, -1, deep_red_color, 2)

  return overlayed_image, mask_bgr


def remove_blur_fft(image):
  # 确保输入图像是浮点类型且是单通道的
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


def remove_bright_regions(fshift_channels, threshold):
  cleaned_fshift_channels = []
  for fshift in fshift_channels:
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    mask = magnitude_spectrum < threshold
    cleaned_fshift = fshift * mask
    cleaned_fshift_channels.append(cleaned_fshift)
  return cleaned_fshift_channels


def process_image_detail_shape(img: MatLike, mask: MatLike) -> Tuple[MatLike, MatLike]:
  img_color = img.copy()
  img = cv2.bitwise_not(img)

  # 应用高斯模糊
  blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

  # 使用FFT去除模糊区域
  fft_img = remove_blur_fft(blurred_img)

  # 创建一个卷积核进行腐蚀操作
  kernel = np.ones((3, 3), np.uint8)
  eroded_img = cv2.erode(fft_img, kernel, iterations=1)

  # 创建一个 CLAHE 对象
  clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
  cl1 = clahe.apply(eroded_img)

  # 初次应用 Otsu's 阈值处理
  otsu_thresh_value, otsu_img = cv2.threshold(
      cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  # 去除小圆点噪声（形态学开操作）
  kernel = np.ones((3, 3), np.uint8)
  cleaned_otsu_img = cv2.morphologyEx(otsu_img, cv2.MORPH_OPEN, kernel)

  # 膨胀和闭操作以连接距离较近的色块
  kernel = np.ones((10, 10), np.uint8)  # 根据需要调整大小
  filter_mask = cv2.dilate(cleaned_otsu_img, kernel, iterations=1)
  connected_img = cv2.morphologyEx(filter_mask, cv2.MORPH_CLOSE, kernel)

  # 生成实心多边形
  contours, _ = cv2.findContours(
      connected_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  solid_polygon_img = np.zeros_like(connected_img)
  cv2.fillPoly(solid_polygon_img, contours, 255)

  # 去除小的圆点
  min_area = 2000  # 根据需要调整最小面积
  contours, _ = cv2.findContours(
      solid_polygon_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  filtered_img = np.zeros_like(solid_polygon_img)
  for contour in contours:
    if cv2.contourArea(contour) >= min_area:
      cv2.drawContours(filtered_img, [contour], -1, 255, thickness=cv2.FILLED)

  # 确保 filtered_img 和 mask 的尺寸匹配，并将 filtered_img 转换为三通道
  if filtered_img.shape != mask.shape[:2]:
    filtered_img = cv2.resize(filtered_img, (mask.shape[1], mask.shape[0]))

  filtered_img_3ch = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)

  # bitwise and
  final_mask = cv2.bitwise_and(filtered_img_3ch, mask)

  # 创建透明的蓝色图层
  blue_layer = np.zeros_like(img_color)
  blue_layer[:, :] = (255, 0, 0)  # 蓝色
  alpha = 0.2  # 透明度

  # 将蓝色图层应用到膨胀后的区域
  mask_bool = final_mask.astype(bool)
  img_color[mask_bool] = cv2.addWeighted(
      img_color, 1 - alpha, blue_layer, alpha, 0)[mask_bool]

  # 在原图上绘制红色边界，包括内部的圈
  contours, hierarchy = cv2.findContours(
      cv2.cvtColor(final_mask, cv2.COLOR_BGR2GRAY), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  for i in range(len(contours)):
    cv2.drawContours(img_color, contours, i, (0, 0, 255), 2)

  return img_color, final_mask


def process_image(src: Image) -> Tuple[Optional[MatLike], Optional[MatLike]]:
  origianl_img = src.get_img()
  if origianl_img is None:
    print(f"Failed to load image {src.get_full_path()}")
    return None, None

  large_overlayed_image, large_mask_bgr = process_image_larger_shape(
      origianl_img)
  detailed_overlayed_image, detailed_mask_bgr = process_image_detail_shape(
      origianl_img, large_mask_bgr)

  return detailed_overlayed_image, detailed_mask_bgr


def calculate_area(mask: MatLike) -> float:
  # 计算白色像素的数量
  white_pixel_count = cv2.countNonZero(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

  # 每像素边长
  ratio = 140
  len_per_px_side = 1.0 / ratio

  # 每像素面积
  area_per_px = len_per_px_side ** 2

  # 总面积
  total_area = white_pixel_count * area_per_px

  return total_area


def process_and_save_image(img: Image, out_dir: str) -> None:
  overlayed_image, mask_bgr = process_image(img)

  if overlayed_image is not None and mask_bgr is not None:
    area = calculate_area(mask_bgr)
    print(f"Area of {img.get_name()} = {area:.2f}")
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

  mem_gb = psutil.virtual_memory().total / (1024 ** 3)
  num_processes = int(mem_gb / 2) + 1

  with multiprocessing.Pool(num_processes) as pool:
    pool.starmap(process_and_save_image, [(img, out_dir) for img in images])


if __name__ == "__main__":
  main(input_directory, output_directory)
