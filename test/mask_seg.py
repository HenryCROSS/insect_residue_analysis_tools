import cv2
import numpy as np
import os

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

def find_cluster(grid: list[list[int]], pos: tuple[int, int]) -> tuple[list[list[int]], list[list[int]]]:
  ...

def grid_clustering(image, grid: list[list[int]]):
  """generate clusters

  Args:
      image (_type_): _description_
      grid (list[list[int]]): _description_

  Returns:
      _type_: _description_
  """
  clusters: list[list[list[int]]] = []
  for i in range(0, len(grid)):
    for j in range(0, len(grid[0])):
      if grid[i][j] == 1:
        new_grid, cluster = find_cluster(grid, (i, j))
        grid = new_grid
        clusters.append(cluster)
  return clusters


def do_clustering(image, block_size: int, ratio: float):
  """clustering algorithm

  Args:
      image (_type_): _description_
      block_size (int): _description_
      ratio (float): _description_
  """
  grid = apply_mosaic_ratio(image, block_size, ratio)
  grid_clustering(image, grid)


def main():
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  for filename in os.listdir(input_folder):
    input_image_path = os.path.join(input_folder, filename)

    image = cv2.imread(input_image_path)

    if image is None:
      print(f"cannot read {input_image_path}, skipped")
      continue

    output_image_path = os.path.join(output_folder, filename)

    cv2.imwrite(output_image_path, image)
    print(f"Image {filename} saved to {output_image_path}")


main()
