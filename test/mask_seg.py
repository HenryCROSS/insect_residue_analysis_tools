import cv2
import numpy as np
import os
import random

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
  """Find and extract a connected cluster from the grid starting at the given position.

  Args:
      grid (list[list[int]]): The 2D array grid.
      pos (tuple[int, int]): The starting position (x, y).
      value (int): The value that defines the cluster (e.g., 1 for white blocks).

  Returns:
      tuple: Updated grid with the cluster removed, and the extracted cluster as a 2D array.
  """
  stack = [pos]

  # Create a new 2D array for the cluster with the same dimensions as the grid
  cluster_grid = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]

  while stack:
    x, y = stack.pop()
    if grid[x][y] == value:
      # Mark this cell as visited by setting it to 0 (or another value)
      grid[x][y] = 0
      # Set the corresponding cell in the cluster grid
      cluster_grid[x][y] = value

      # Check neighbors (up, down, left, right, and diagonals)
      for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (-1, 1), (1, 1), (-1, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == value:
          stack.append((nx, ny))

  return grid, cluster_grid


def grid_clustering(grid: list[list[int]], value: int = 1) -> list[list[list[int]]]:
  """Generate clusters based on the grid.

  Args:
      grid (list[list[int]]): 2D array with values representing clusters.
      value (int): The value that defines the cluster (default is 1).

  Returns:
      list[list[list[int]]]: A list of clusters, where each cluster is a 2D array.
  """
  clusters = []

  for i in range(len(grid)):
    for j in range(len(grid[0])):
      if grid[i][j] == value:
        grid, cluster = find_cluster(grid, (i, j), value)
        clusters.append(cluster)

  return clusters


def image_clustering(image, clustered_grid: list[list[list[int]]], block_size: int) -> list[np.ndarray]:
  """Generate a list of MatLike objects based on the clustered grid.

  Args:
      image: Original image.
      clustered_grid (list[list[list[int]]]): List of clusters from grid_clustering.
      block_size (int): Size of the block used for the grid.

  Returns:
      list[np.ndarray]: A list of images where each image corresponds to a cluster.
  """
  (h, w) = image.shape[:2]
  cluster_images = []

  for cluster in clustered_grid:
    # Create a blank image
    cluster_image = np.zeros((h, w), dtype=image.dtype)

    for i in range(len(cluster)):
      for j in range(len(cluster[0])):
        if cluster[i][j] == 1:
          # Determine the coordinates in the original image
          y_start, y_end = i * block_size, min((i + 1) * block_size, h)
          x_start, x_end = j * block_size, min((j + 1) * block_size, w)
          # Copy the pixels from the original image that correspond to the current cluster
          cluster_image[y_start:y_end,
                        x_start:x_end] = image[y_start:y_end, x_start:x_end]

    cluster_images.append(cluster_image)

  return cluster_images


def edge_expand(image, clusters: list[list[list[int]]], block_size: int, ratio: float, distance: int) -> list[list[list[int]]]:
  """Expand the edge of each cluster by a given distance based on a new ratio.

  Args:
      image: Original image.
      clusters (list[list[list[int]]]): List of clusters from grid_clustering.
      block_size (int): Size of the block used for the grid.
      ratio (float): White pixel ratio threshold for edge expansion.
      distance (int): Number of blocks to expand.

  Returns:
      list[list[list[int]]]: List of expanded clusters.
  """
  (h, w) = image.shape[:2]
  expanded_clusters = []

  for cluster in clusters:
    expanded_cluster = [row[:] for row in cluster]  # Deep copy the cluster

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
  """Combine clusters that are within a specified distance from each other.

  Args:
      clusters (list[list[list[int]]]): List of clusters.
      dist (int): Maximum distance within which clusters should be combined.

  Returns:
      list[list[list[int]]]: List of combined clusters.
  """
  def clusters_are_close(cluster1, cluster2, dist):
    for i in range(len(cluster1)):
      for j in range(len(cluster1[0])):
        if cluster1[i][j] == 1:
          for di in range(-dist, dist + 1):
            for dj in range(-dist, dist + 1):
              ni, nj = i + di, j + dj
              if 0 <= ni < len(cluster2) and 0 <= nj < len(cluster2[0]) and cluster2[ni][nj] == 1:
                return True
    return False

  combined = []
  while clusters:
    cluster = clusters.pop(0)
    merged = True
    while merged:
      merged = False
      for i in range(len(clusters) - 1, -1, -1):
        if clusters_are_close(cluster, clusters[i], dist):
          # Merge clusters
          for x in range(len(cluster)):
            for y in range(len(cluster[0])):
              if clusters[i][x][y] == 1:
                cluster[x][y] = 1
          clusters.pop(i)
          merged = True
    combined.append(cluster)

  return combined


def eliminate_clusters(clusters: list[list[list[int]]], smallest_cluster: int) -> list[list[list[int]]]:
  """Eliminate clusters smaller than the specified size.

  Args:
      clusters (list[list[list[int]]]): List of clusters.
      smallest_cluster (int): Minimum size of clusters to keep.

  Returns:
      list[list[list[int]]]: List of clusters that meet the size requirement.
  """
  return [cluster for cluster in clusters if np.sum(cluster) >= smallest_cluster]


def do_clustering(image, block_size: int, ratio: float, new_ratio: float, n: int, dist: int, smallest_cluster: int) -> list[np.ndarray]:
  """Clustering algorithm with edge expansion, cluster combination, and elimination.

  Args:
      image: Input image.
      block_size (int): Block size for mosaic.
      ratio (float): White pixel ratio threshold.
      new_ratio (float): New white pixel ratio threshold for edge expansion.
      n (int): Number of blocks to expand.
      dist (int): Maximum distance within which clusters should be combined.
      smallest_cluster (int): Minimum size of clusters to keep.

  Returns:
      list[np.ndarray]: List of images representing each cluster.
  """
  grid = apply_mosaic_ratio(image, block_size, ratio)
  clusters = grid_clustering(grid)
  expanded_clusters = edge_expand(image, clusters, block_size, new_ratio, n)
  combined_clusters_list = combine_clusters(expanded_clusters, dist)
  final_clusters = eliminate_clusters(combined_clusters_list, smallest_cluster)
  cluster_images = image_clustering(image, final_clusters, block_size)
  return cluster_images


def save_raw_in_colour(image, cluster_images: list[np.ndarray], output_path: str):
  """Save the image with clusters drawn in different colors.

  Args:
      image: The original image.
      cluster_images (list[np.ndarray]): List of cluster images.
      output_path (str): Path to save the output image.
  """
  color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

  for cluster_image in cluster_images:
    # Generate a random color for the cluster
    color = [random.randint(0, 255) for _ in range(3)]

    # Find the non-zero regions in the cluster image and color them
    mask = cluster_image > 0
    color_image[mask] = color

  cv2.imwrite(output_path, color_image)
  print(f"Clustered image with colors saved to {output_path}")


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

    # Perform clustering and get cluster images with edge expansion, combination, and elimination
    cluster_images = do_clustering(
        image, block_size, ratio, new_ratio, n, dist, smallest_cluster)

    # Save the original image with clusters highlighted in color
    output_image_colored_path = os.path.join(
        output_folder, f"{os.path.splitext(filename)[0]}_clusters_colored.png")
    save_raw_in_colour(image, cluster_images, output_image_colored_path)


main()
