import numpy as np


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

      # Check neighbors (up, down, left, right)
      for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
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


# 示例代码使用
grid = [
    [1, 1, 0, 0, 1, 0],
    [1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 1],
    [1, 1, 0, 0, 1, 0]
]

clusters = grid_clustering(grid)

print(f"Found {len(clusters)} clusters:")
for idx, cluster in enumerate(clusters):
  print(f"Cluster {idx + 1}:")
  for row in cluster:
    print(row)
