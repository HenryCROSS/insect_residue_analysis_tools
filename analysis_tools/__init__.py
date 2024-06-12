from time import sleep
import cv2
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from cv2.typing import MatLike
from typing import NewType, Callable
from functools import reduce

IMG_UNIT = NewType("IMG_UNIT", tuple[str, MatLike])

IMG_DIR_IN: str = "./Pictures"
IMG_DIR_OUT: str = "./Pictures_out"

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

  # pixel_values = img.reshape((-1, 3))
  # pixel_values = np.float32(pixel_values)

  # kmeans = KMeans(n_clusters=8, random_state=0)
  # labels = kmeans.fit_predict(pixel_values)
  # centers = np.uint8(kmeans.cluster_centers_)
  # segmented_image = centers[labels.flatten()]
  # segmented_image = segmented_image.reshape(img.shape)
  
  Z = img.reshape((-1,3))
  # convert to np.float32
  Z_f32 = np.float32(Z)
  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 8
  ret,label,center=cv2.kmeans(Z_f32,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
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
  
  ret3,th3 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  return create_img_unit(name, th3)

def do_overlap_transparently(img_unit: IMG_UNIT, img_unit_overlapped: IMG_UNIT, alpha: float) -> IMG_UNIT:
  (name, img) = img_unit
  (_, img_overlapped) = img_unit_overlapped
  
  # Ensure the mask is in the same size as the image
  mask_resized = cv2.resize(img_overlapped, (img.shape[1], img.shape[0]))

  # Create a blue version of the mask
  blue_mask = np.zeros_like(img)
  blue_mask[:, :, 0] = mask_resized  # Assign the mask to the blue channel

  # Blend the image and the mask
  overlay = cv2.addWeighted(img, 1, blue_mask, alpha, 0)

  return create_img_unit(name, overlay)


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
    masked = cv2.bitwise_and(img, img, mask=mask)

    return create_img_unit(name, masked)

  return mask_it

def mask_with_shape_not(mask_img_unit: IMG_UNIT) -> Callable[[IMG_UNIT], IMG_UNIT]:
  def mask_it(img_unit: IMG_UNIT) -> IMG_UNIT:
    if mask_img_unit is None:
      return img_unit

    (name, img) = img_unit
    (_, mask) = mask_img_unit
    masked = cv2.bitwise_not(img, img, mask=mask)

    return create_img_unit(name, masked)

  return mask_it


def contrast_enhancement(img_unit: IMG_UNIT) -> IMG_UNIT:
  (name, img) = img_unit
  # control Contrast by 1.5
  alpha = 1.5
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

def proc_for_white_bg(img_unit: IMG_UNIT, mask_img_unit: IMG_UNIT) -> None:
  new_process = [
      do_erosion,
      do_smoothing,
      mask_with_shape(mask_img_unit),
      cvt_black2white,
      cvt_BGR2HSV,
      contrast_enhancement,
      do_img_segmentation,
      do_smoothing,
      cvt_HSV2BGR,
      cvt_BGR2GRAY,
      do_otsu_thresholding,
      revert_white_black
  ]

  (name, img) = img_unit

  img_unit_mask = reduce(lambda img, func: func(img), new_process, img_unit)
  (_, img_mask) = img_unit_mask

  # img_unit_contour = do_adaptive_threshold(img_unit_mask)

  # count white pixel
  n_white_px = np.sum(img_mask != 0)

  # 135 pixel per millimeter
  ratio = 140
  len_per_px_side = 1.0 / ratio
  area_per_px = len_per_px_side ** 2

  print(f"{name} has {n_white_px} pixels, area is {area_per_px * n_white_px}")

  (_, img_labelled) = do_overlap_transparently(img_unit, img_unit_mask, 1)

  output_img(IMG_DIR_OUT, img_unit_mask)
  output_img(IMG_DIR_OUT, create_img_unit(f"{name}_labelled.jpg", img_labelled))
  



def drawing(img_unit: IMG_UNIT) -> tuple[IMG_UNIT, IMG_UNIT | None]:
  (name, image) = img_unit
  original_image = image.copy()
  (scaled_height, scaled_width) = image.shape[:2]
  history = []

  def resize_image(image, scale_percent):
    nonlocal scaled_width, scaled_height
    (my, mx) = image.shape[:2]
    scaled_width = int(mx * scale_percent / 100)
    scaled_height = int(my * scale_percent / 100)
    resized_image = cv2.resize(
        image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
    return resized_image

  # 创建一个空白的黑色图像
  # height, width = 500, 500
  # image = np.zeros((height, width, 3), dtype=np.uint8)
  image = resize_image(image, 50)
  scale_back_percent = 100 / 50 * 100
  temp_image = image
  drawing = False  # 是否正在绘制
  start_point = None  # 正方形的起点
  all_masks = None

  # 鼠标回调函数

  def draw_square(event, x, y, flags, param):
    nonlocal start_point, drawing, image, temp_image, scaled_height, scaled_width, history

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

        temp_image = image.copy()
        cv2.rectangle(temp_image, start_point, (x, y), (0, 0, 255), 1)

    elif event == cv2.EVENT_LBUTTONUP:
      drawing = False
      temp_image = image.copy()
      mask = np.zeros(image.shape[:2], dtype=np.uint8)
      cv2.rectangle(mask, start_point, (x, y), 255, -1)
      history.append(mask)

  # 创建窗口并设置鼠标回调函数
  cv2.namedWindow('Image')
  cv2.setMouseCallback('Image', draw_square)

  while True:
    if drawing:
      cv2.imshow('Image', temp_image)
    else:

      if len(history) == 0:
        cv2.imshow('Image', temp_image)
      else:
        show_img = create_img_unit("", image.copy())
        all_masks = None
        for mask in history:
          if all_masks is None:
            all_masks = mask.copy()
          all_masks = cv2.bitwise_or(all_masks, mask)
          show_img = do_overlap_transparently(
              show_img, create_img_unit("", all_masks), 1)
        (_, img) = show_img
        cv2.imshow('Image', img)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
      break
    elif key == 32:  # Space
      if len(history) > 0:
        history.pop()

  cv2.destroyAllWindows()

  if len(history) > 0:
    resized_all_masks = create_img_unit(
        f"{name}_mask.jpg", resize_image(all_masks, scale_back_percent))
    resized_img = do_overlap_transparently(
        img_unit, resized_all_masks, 1)
    output_img(IMG_DIR_OUT, resized_all_masks)
    output_img(IMG_DIR_OUT, resized_img)
    return (img_unit, resized_all_masks)
  else:
    output_img(IMG_DIR_OUT, img_unit)
    return (img_unit, None)


def main():
  # load images from IMG_DIR_IN
  imgs = load_imgs(IMG_DIR_IN)

  for img in imgs:
    (img_unit, mask_img_unit) = drawing(img)

    proc_for_white_bg(img_unit, mask_img_unit)

  # speed up by applying multi processing
  # with ProcessPoolExecutor() as executor:
  #   futures = [executor.submit(proc, img, None) for img in imgs]
  #   for future in futures:
  #     # waiting for each to complete, handle exceptions if needed
  #     future.result()


main()
