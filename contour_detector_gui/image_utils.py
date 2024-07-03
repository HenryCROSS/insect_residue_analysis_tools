import sys
from time import sleep
import cv2
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from cv2.typing import *
from typing import *
from functools import reduce
from uuid import UUID, uuid1


IMG_UNIT = NewType("IMG_UNIT", tuple[str, MatLike])
# (id, priority)
HISTORY_UNIT = NewType("HISTORY_UNIT", tuple[UUID, int])


class Lv_Mask:
  def __init__(self, mask: MatLike | None = None, img_size: tuple[int, int] | None = None, mask_coord: tuple[tuple[int, int], tuple[int, int]] | None = None):
    self.id = uuid1()
    if mask is not None:
      self.mask: MatLike = mask
    elif mask_coord is not None and img_size is not None:
      start_point, end_point = mask_coord
      mask = np.zeros(img_size, dtype=np.uint8)
      cv2.rectangle(mask, start_point, end_point, (0, 0, 255), -1)
      self.mask = mask


class Lv_Mask_Set:
  def __init__(self):
    self.masks: dict[int, list[Lv_Mask | None]] = {}

  def get_priority_mask_list(self, priority: int) -> list[Lv_Mask | None]:
    if priority in self.masks:
      return self.masks[priority]

    self.masks[priority] = []
    return self.masks[priority]


class Lv_Mask_History:
  def __init__(self):
    self.history_size: int = 0
    self.history: list[HISTORY_UNIT] = []

  def append(self, record: HISTORY_UNIT):
    self.history.append(record)
    self.history_size += 1

  def undo(self):
    if self.history_size > 0:
      self.history_size -= 1

  def redo(self):
    if self.history_size < len(self.history):
      self.history_size += 1

  def commit(self):
    self.history = self.history[:self.history_size]

  def query_all(self):
    return self.history[:self.history_size]


class Img_State:
  def __init__(self, path: str, img_unit: IMG_UNIT | None = None):
    self.path: str = path
    self.img_unit: IMG_UNIT | None = img_unit
    self.after_process_img: MatLike | None = None
    self.current_priority = 1000
    self.mask_list: Lv_Mask_Set = Lv_Mask_Set()
    self.mask_history: Lv_Mask_History = Lv_Mask_History()
    self.mask_list_size: int = 0
    (self.h, self.w) = img_unit[1].shape[:2] if img_unit is not None else (
        None, None)

  def undo_mask(self):
    self.mask_history.undo()
    self.mask_list_size -= 1

  def redo_mask(self):
    self.mask_history.redo()
    self.mask_list_size += 1

  def reset_img(self):
    self.mask_list = Lv_Mask_Set()
    self.mask_list_size = 0
    self.after_process_img = None
    self.current_priority = 1000

  def append_mask(self, mask: Lv_Mask):
    self.mask_history.commit()

    if self.mask_history.history_size != self.mask_list_size:
      #TODO: remove the one
      ...

    masks = self.mask_list.get_priority_mask_list(self.current_priority)
    masks.append(mask)

    self.mask_history.append(HISTORY_UNIT((mask.id, self.current_priority)))
    self.mask_list_size += 1

  def get_all_priority_masks(self):
    # return self.mask_list[:self.mask_list_size]
    ...

  def get_current_priority_masks(self):
    ...

  def get_img(self):
    if self.img_unit is None:
      print(f"loading {self.path}")
      self.img_unit = create_img_unit(
          os.path.basename(self.path), cv2.imread(self.path))

    return self.img_unit


def create_img_unit(name: str, img: MatLike) -> IMG_UNIT:
  return IMG_UNIT((name, img))


def load_imgs_path(path: str) -> list[Img_State]:
  files = os.listdir(path)
  img_names = map(lambda file: file, files)
  return list(map(lambda name: Img_State(f"{path}/{name}"), img_names))
