import os
import cv2
import numpy as np

"""_summary_
The background image should have the same name as the folder with jpg format

input folder format:
human_output/
|- target1/
|  |- target1.jpg
|  \- images....
|
|- target2/
|  |- target2.jpg
|  \- images....
|
|- Test/
|  |- Test.jpg
|  \- images....
|
...
"""

def read_images_from_folder(folder_path, target_image_name):
  """
  Read all image files from the specified folder, excluding the target image.

  :param folder_path: Path to the folder
  :param target_image_name: Name of the target image to exclude
  :return: List of images and corresponding filenames
  """
  images = []
  filenames = []
  for filename in os.listdir(folder_path):
    if filename == target_image_name:
      continue
    img_path = os.path.join(folder_path, filename)
    img = cv2.imread(img_path)
    if img is not None:
      images.append(img)
      filenames.append(filename)
  return images, filenames


def filter_and_highlight_color(image, target_colors, color_ranges, highlight_color):
  """
  Filter specific RGB colors in the image and convert them to a bright color, making other areas black.

  :param image: Input image
  :param target_colors: List of target colors (B, G, R)
  :param color_ranges: List of color ranges (B, G, R)
  :param highlight_color: Highlight color (B, G, R)
  :return: Processed image
  """
  result = np.zeros_like(image)

  for i, target_color in enumerate(target_colors):
    color_range = color_ranges[i % len(color_ranges)]

    lower_bound = np.array([max(0, target_color[0] - color_range[0]),
                            max(0, target_color[1] - color_range[1]),
                            max(0, target_color[2] - color_range[2])])
    upper_bound = np.array([min(255, target_color[0] + color_range[0]),
                            min(255, target_color[1] + color_range[1]),
                            min(255, target_color[2] + color_range[2])])

    mask = cv2.inRange(image, lower_bound, upper_bound)
    result[mask > 0] = highlight_color

  return result


def adjust_image_opacity(image, alpha=0.5):
  """
  Adjust the opacity of the image by scaling pixel values.

  :param image: Input image
  :param alpha: Opacity factor (0.0 to 1.0)
  :return: Image with adjusted opacity
  """
  return cv2.addWeighted(image, alpha, np.zeros_like(image), 0, 0)


def overlay_images(base_image, overlay_image):
  """
  Overlay the processed image onto the base image.

  :param base_image: Base image (background image)
  :param overlay_image: Image to overlay
  :return: Combined image
  """
  overlay_mask = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2GRAY)
  _, overlay_mask = cv2.threshold(overlay_mask, 1, 255, cv2.THRESH_BINARY)

  base_image[overlay_mask > 0] = overlay_image[overlay_mask > 0]

  return base_image


def process_directory(directory_path, output_folder, alpha, target_colors, color_ranges, highlight_colors):
  """
  Process all images in a directory, overlay them onto the target image with transparency, and save the result.

  :param directory_path: Path to the directory
  :param output_folder: Path to the output folder
  :param alpha: Opacity factor for the target image
  :param target_colors: List of target colors (B, G, R)
  :param color_ranges: List of color ranges (B, G, R)
  :param highlight_colors: List of highlight colors (B, G, R)
  """
  # Use the directory name plus an image suffix as the target image name
  directory_name = os.path.basename(directory_path)
  # Assuming the target image is in jpg format, adjust the suffix as needed
  target_image_name = f"{directory_name}.jpg"

  target_image_path = os.path.join(directory_path, target_image_name)
  base_image = cv2.imread(target_image_path)

  if base_image is None:
    raise FileNotFoundError(
        f"Target image {target_image_name} not found in {directory_path}")

  # Adjust the opacity of the base image
  base_image = adjust_image_opacity(base_image, alpha)

  # Read the other images in the directory
  images, _ = read_images_from_folder(directory_path, target_image_name)

  # Process and overlay each image onto the target image
  for idx, img in enumerate(images):
    # Cycle through the highlight colors
    highlight_color = highlight_colors[idx % len(highlight_colors)]
    processed_image = filter_and_highlight_color(
        img, target_colors, color_ranges, highlight_color)
    base_image = overlay_images(base_image, processed_image)

  # Save the final combined image in PNG format
  output_path = os.path.join(output_folder, f"{directory_name}_combined.png")
  cv2.imwrite(output_path, base_image)
  print(f"Final combined image saved to {output_path}")


def main(input_folder, output_folder, alpha=0.5, target_colors=None, color_ranges=None, highlight_colors=None):
  """
  Main function: Process all subdirectories in the input folder, adjust transparency, and save the combined images.

  :param input_folder: Path to the input folder containing multiple subdirectories
  :param output_folder: Path to the output folder
  :param alpha: Opacity factor for the target images
  :param target_colors: List of target colors (B, G, R) for filtering
  :param color_ranges: List of color ranges (B, G, R) for filtering
  :param highlight_colors: List of highlight colors (B, G, R)
  """
  if target_colors is None:
    target_colors = [(117, 240, 238), (255, 0, 0)]  # Example target colors

  if color_ranges is None:
    color_ranges = [(20, 20, 20), (30, 30, 30)]  # Example color ranges

  if highlight_colors is None:
    highlight_colors = [
        (255, 255, 0),      # Cyan
        (199, 199, 255),   # Light Blue
        (0, 255, 166),     # Light Green
        (0, 255, 0),       # Green
        (146, 228, 255),   # Light Cyan
        (255, 0, 0),       # Red
        (255, 0, 255),     # Magenta
    ]

  # Ensure the output folder exists
  os.makedirs(output_folder, exist_ok=True)

  # Process each subdirectory in the input folder
  for subdir in os.listdir(input_folder):
    subdir_path = os.path.join(input_folder, subdir)
    if os.path.isdir(subdir_path):
      process_directory(subdir_path, output_folder, alpha,
                        target_colors, color_ranges, highlight_colors)


if __name__ == "__main__":
  # Main input folder containing multiple subdirectories
  input_folder = "./human_output"
  output_folder = "./human_output_combine"  # Folder to save the combined images
  alpha = 0.4  # Opacity factor for the target image

  # Parameters
  # List of target colors (B, G, R)
  target_colors = [(117, 240, 238), (119, 241, 203),
                   (130, 233, 236), (0, 255, 255),
                   (95, 214, 216)]
  color_ranges = [(15, 15, 15), (15, 15, 15), (15, 15, 15), (15, 15, 15), (15, 15, 15)
                  ]     # List of color ranges (B, G, R)
  highlight_colors = [
      (199, 199, 255),   # 浅蓝 (Light Blue)
      (146, 228, 255),   # 浅青 (Light Cyan)
      (0, 255, 166),     # 浅绿 (Light Green)
      (0, 255, 0),       # 绿 (Green)
      (255, 0, 255),     # 洋红 (Magenta)
      (255, 255, 0),     # 黄 (Yellow)
      (255, 128, 0),     # 橙 (Orange)
      (255, 105, 180),   # 亮粉红 (Hot Pink)
      (128, 255, 0),     # 酸橙绿 (Lime Green)
      (128, 0, 255),     # 紫 (Purple)
      (255, 64, 64),     # 浅红 (Light Red)
      (0, 191, 255)      # 深天蓝 (Deep Sky Blue)
  ]

  main(input_folder, output_folder, alpha,
       target_colors, color_ranges, highlight_colors)
