import cv2
import numpy as np
import os
import random
import math

input_folder = './Masks'
output_folder = './Masks_out'


def apply_mosaic_ratio(image, block_size: int, ratio: float) -> list[list[int]]:
  if block_size < 1:
    block_size = 1

  grid: list[list[int]] = []

  (h, w) = image.shape[:2]
  for y in range(0, h, block_size):
    row = []
    for x in range(0, w, block_size):
      block = image[y:y + block_size, x:x + block_size]
      white_pixels = np.sum(block >= 255)  # get white pixel
      total_pixels = block.size
      white_ratio = white_pixels / total_pixels

      if white_ratio >= ratio:
        row.append(1)
      else:
        row.append(0)

    grid.append(row)

  return grid


def find_cluster(grid: list[list[int]], pos: tuple[int, int], value: int) -> tuple[list[list[int]], list[list[int]]]:
  stack = [pos]
  cluster_grid = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]

  while stack:
    x, y = stack.pop()
    if grid[x][y] == value:
      grid[x][y] = 0
      cluster_grid[x][y] = value

      for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == value:
          stack.append((nx, ny))

  return grid, cluster_grid


def grid_clustering(grid: list[list[int]], value: int = 1) -> list[list[list[int]]]:
  clusters = []

  for i in range(len(grid)):
    for j in range(len(grid[0])):
      if grid[i][j] == value:
        grid, cluster = find_cluster(grid, (i, j), value)
        clusters.append(cluster)

  return clusters


def image_clustering(image, clustered_grid: list[list[list[int]]], block_size: int) -> list[np.ndarray]:
  (h, w) = image.shape[:2]
  cluster_images = []

  for cluster in clustered_grid:
    cluster_image = np.zeros((h, w), dtype=np.uint8)

    for i in range(len(cluster)):
      for j in range(len(cluster[0])):
        if cluster[i][j] == 1:
          y_start, y_end = i * block_size, min((i + 1) * block_size, h)
          x_start, x_end = j * block_size, min((j + 1) * block_size, w)
          cluster_image[y_start:y_end, x_start:x_end] = 255

    cluster_images.append(cluster_image)

  return cluster_images


def edge_expand(image, clusters: list[list[list[int]]], block_size: int, ratio: float, distance: int) -> list[list[list[int]]]:
  (h, w) = image.shape[:2]
  expanded_clusters = []

  for cluster in clusters:
    expanded_cluster = [row[:] for row in cluster]  # Deep copy

    for i in range(len(cluster)):
      for j in range(len(cluster[0])):
        if cluster[i][j] == 1:
          for dx in range(-distance, distance + 1):
            for dy in range(-distance, distance + 1):
              ni, nj = i + dx, j + dy
              if 0 <= ni < len(cluster) and 0 <= nj < len(cluster[0]) and expanded_cluster[ni][nj] == 0:
                y_start, y_end = ni * block_size, min((ni + 1) * block_size, h)
                x_start, x_end = nj * block_size, min((nj + 1) * block_size, w)
                block = image[y_start:y_end, x_start:x_end]
                white_pixels = np.sum(block >= 255)
                total_pixels = block.size
                white_ratio = white_pixels / total_pixels

                if white_ratio >= ratio:
                  expanded_cluster[ni][nj] = 1

    expanded_clusters.append(expanded_cluster)

  return expanded_clusters


def combine_clusters(clusters: list[list[list[int]]], dist: int) -> list[list[list[int]]]:
  def clusters_are_close(cluster1, cluster2, dist):
    for i in range(len(cluster1)):
      for j in range(len(cluster1[0])):
        if cluster1[i][j] == 1:
          for dx in range(-dist, dist + 1):
            for dy in range(-dist, dist + 1):
              ni, nj = i + dx, j + dy
              if 0 <= ni < len(cluster2) and 0 <= nj < len(cluster2[0]) and cluster2[ni][nj] == 1:
                return True
    return False

  combined = []
  while clusters:
    base_cluster = clusters.pop(0)
    merge_indices = []
    for idx, other_cluster in enumerate(clusters):
      if clusters_are_close(base_cluster, other_cluster, dist):
        merge_indices.append(idx)

    for idx in sorted(merge_indices, reverse=True):
      other_cluster = clusters.pop(idx)
      for i in range(len(base_cluster)):
        for j in range(len(base_cluster[0])):
          if other_cluster[i][j] == 1:
            base_cluster[i][j] = 1

    combined.append(base_cluster)

  return combined


def eliminate_clusters(clusters: list[list[list[int]]], smallest_cluster: int) -> list[list[list[int]]]:
  filtered_clusters = []
  for cluster in clusters:
    cluster_size = sum(sum(row) for row in cluster)
    if cluster_size >= smallest_cluster:
      filtered_clusters.append(cluster)
  return filtered_clusters


def rotating_calipers(points: np.ndarray) -> tuple:
  hull_points = cv2.convexHull(points, returnPoints=True)
  hull_points = hull_points.reshape(-1, 2)
  num_points = len(hull_points)

  if num_points < 2:
    return ((None, None, 0, 0), (None, None, 0), (0, 0))

  # Find diameter
  max_length = 0
  length_start_point = None
  length_end_point = None
  for i in range(num_points):
    for j in range(i + 1, num_points):
      pt1 = hull_points[i]
      pt2 = hull_points[j]
      distance = np.linalg.norm(pt1 - pt2)
      if distance > max_length:
        max_length = distance
        length_start_point = tuple(pt1)
        length_end_point = tuple(pt2)

  dx = length_end_point[0] - length_start_point[0]
  dy = length_end_point[1] - length_start_point[1]
  angle = math.degrees(math.atan2(dy, dx))

  # Minimum width
  rect = cv2.minAreaRect(points)
  width = min(rect[1])
  box = cv2.boxPoints(rect)
  box = np.intp(box)
  width_start_point = tuple(box[0])
  width_end_point = tuple(box[1])

  # Calculate center point of the convex hull
  center_x = np.mean(hull_points[:, 0])
  center_y = np.mean(hull_points[:, 1])
  center_point = (center_x, center_y)

  return (
      (length_start_point, length_end_point, max_length, angle),
      (width_start_point, width_end_point, width),
      center_point
  )


def generate_rectangle(image_shape: tuple, length_info: tuple, width_info: tuple, center_point: tuple) -> np.ndarray:
  (height, width) = image_shape
  mask = np.zeros((height, width), dtype=np.uint8)

  if None in length_info or None in width_info:
    return mask

  # Center of the rectangle based on the calculated center point
  center_x, center_y = center_point

  rect_length = length_info[2]
  rect_width = width_info[2]
  angle = length_info[3]

  rectangle = ((center_x, center_y), (rect_length, rect_width), angle)
  box = cv2.boxPoints(rectangle)
  box = np.intp(box)

  cv2.drawContours(mask, [box], 0, 255, -1)

  return mask


def generate_ellipse(image_shape: tuple, length_info: tuple, width_info: tuple, center_point: tuple) -> np.ndarray:
  (height, width) = image_shape
  mask = np.zeros((height, width), dtype=np.uint8)

  if None in length_info or None in width_info:
    return mask

  # Center of the ellipse based on the calculated center point
  center_x, center_y = center_point

  axis_length = length_info[2] / 2
  axis_width = width_info[2] / 2
  angle = length_info[3]

  center = (int(center_x), int(center_y))
  axes = (int(axis_length), int(axis_width))

  cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

  return mask


def generate_triangle(image_shape: tuple, length_info: tuple, width_info: tuple, center_point: tuple) -> np.ndarray:
  (height, width) = image_shape
  mask = np.zeros((height, width), dtype=np.uint8)

  if None in length_info or None in width_info:
    return mask

  # Use the length and width information to estimate the enclosing triangle
  points = np.array([length_info[0], length_info[1],
                    width_info[0], width_info[1]], dtype=np.float32)
  ret, triangle = cv2.minEnclosingTriangle(points)

  if ret:
    triangle = np.intp(triangle).reshape(3, 2)

    # Calculate translation vector based on center point
    tri_center = np.mean(triangle, axis=0)
    translation = np.array(center_point) - tri_center

    # Translate triangle to the correct position
    triangle += translation.astype(np.intp)
    cv2.drawContours(mask, [triangle], 0, 255, -1)

  return mask


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
  intersection = np.logical_and(mask1, mask2).sum()
  union = np.logical_or(mask1, mask2).sum()
  if union == 0:
    return 0.0
  else:
    iou = intersection / union
    return iou


def find_simple_shape(cluster_dimensions: list, cluster_images: list[np.ndarray], shape_generators: list) -> list[np.ndarray]:
  best_shapes = []

  for idx, (dimensions, cluster_image) in enumerate(zip(cluster_dimensions, cluster_images)):
    image_shape = cluster_image.shape
    length_info, width_info, center_point = dimensions

    best_iou = -1
    best_shape_mask = None

    for generator in shape_generators:
      shape_mask = generator(image_shape, length_info, width_info, center_point)
      iou = calculate_iou(cluster_image > 0, shape_mask > 0)

      if iou > best_iou:
        best_iou = iou
        best_shape_mask = shape_mask

    best_shapes.append(best_shape_mask)

  return best_shapes


def save_raw_in_colour(image, cluster_images: list[np.ndarray], output_path: str):
  color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

  for cluster_image in cluster_images:
    color = [random.randint(0, 255) for _ in range(3)]
    mask = cluster_image > 0
    color_image[mask] = color

  cv2.imwrite(output_path, color_image)
  print(f"Clustered image with colors saved to {output_path}")


def save_shapes_overlay(image: np.ndarray, shape_masks: list[np.ndarray], output_path: str):
  """Save the image with best matching shapes overlayed with transparency on top of the background and red boundaries.

  Args:
      image (np.ndarray): The original image.
      shape_masks (list[np.ndarray]): List of shape masks.
      output_path (str): Path to save the output image.
  """
  # Convert the original grayscale image to BGR
  color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

  # Create an empty image to hold the overlay (transparent background)
  overlay = np.zeros_like(color_image, dtype=np.uint8)

  # Set transparency factor (alpha) for shapes
  alpha = 0.5

  for shape_mask in shape_masks:
      # Generate a random color for the shape
    color = [random.randint(0, 255) for _ in range(3)]

    # Create a colored shape mask
    colored_shape = np.zeros_like(color_image)
    for i in range(3):  # Apply the color to each channel
      colored_shape[:, :, i] = shape_mask * color[i]

    # Add the colored shape mask to the overlay
    overlay = cv2.add(colored_shape, overlay)

    # Find contours of the shape mask for drawing the red boundary
    contours, _ = cv2.findContours(
        shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw red boundaries on the overlay
    cv2.drawContours(overlay, contours, -1, (0, 0, 255),
                     2)  # Red color with thickness 2

  # Combine the overlay with the original image using alpha transparency
  result_image = cv2.addWeighted(overlay, alpha, color_image, 1 - alpha, 0)

  # Save the result
  cv2.imwrite(output_path, result_image)
  print(f"Shapes overlay image saved to {output_path}")


def do_clustering(image, block_size: int, ratio: float, new_ratio: float, n: int, dist: int, smallest_cluster: int):
  grid = apply_mosaic_ratio(image, block_size, ratio)
  clusters = grid_clustering(grid)
  expanded_clusters = edge_expand(image, clusters, block_size, new_ratio, n)
  combined_clusters_list = combine_clusters(expanded_clusters, dist)
  final_clusters = eliminate_clusters(combined_clusters_list, smallest_cluster)
  cluster_images = image_clustering(image, final_clusters, block_size)

  cluster_dimensions = []
  for cluster_image in cluster_images:
    points = cv2.findNonZero(cluster_image)
    if points is not None and len(points) >= 3:
      length_info, width_info, center_point = rotating_calipers(points)
      cluster_dimensions.append((length_info, width_info, center_point))
    else:
      cluster_dimensions.append(((None, None, 0, 0), (None, None, 0), (0, 0)))

  # Include triangle generator in the list of shape generators
  shape_generators = [generate_rectangle, generate_ellipse, generate_triangle]
  best_shape_masks = find_simple_shape(
      cluster_dimensions, cluster_images, shape_generators)

  return cluster_images, cluster_dimensions, best_shape_masks


def main():
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  block_size = 10  # 设置块大小
  ratio = 0.4  # 设置白色像素比例
  new_ratio = 0.1  # 边缘扩展时的白色像素比例
  n = 1  # 扩展的格子数
  dist = 2  # 聚类合并的最大距离
  smallest_cluster = 10  # 最小的聚类大小

  for filename in os.listdir(input_folder):
    input_image_path = os.path.join(input_folder, filename)

    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
      print(f"Cannot read {input_image_path}, skipped")
      continue

    cluster_images, cluster_dimensions, best_shape_masks = do_clustering(
        image, block_size, ratio, new_ratio, n, dist, smallest_cluster)

    output_image_colored_path = os.path.join(
        output_folder, f"{os.path.splitext(filename)[0]}_clusters_colored.png")
    save_raw_in_colour(image, cluster_images, output_image_colored_path)

    output_shapes_path = os.path.join(
        output_folder, f"{os.path.splitext(filename)[0]}_shapes.png")
    save_shapes_overlay(image, best_shape_masks, output_shapes_path)


if __name__ == "__main__":
  main()
