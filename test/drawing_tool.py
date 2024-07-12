import pygame
import cv2
import numpy as np
import os
from typing import List, Optional, Tuple

# 初始化Pygame
pygame.init()

# 设置屏幕尺寸
screen_width: int = 1000
screen_height: int = 600
sidebar_width: int = 200
toolbar_height: int = 50
screen: pygame.Surface = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
pygame.display.set_caption("Pygame Drawing Tool with OpenCV")

# 颜色
WHITE: Tuple[int, int, int] = (255, 255, 255)
BLACK: Tuple[int, int, int] = (0, 0, 0)
GRAY: Tuple[int, int, int] = (200, 200, 200)
RED: Tuple[int, int, int] = (255, 0, 0)
BLUE: Tuple[int, int, int] = (0, 0, 255)
GREEN: Tuple[int, int, int] = (0, 255, 0)  # 工具栏颜色

# 变量初始化
drawing: bool = False
moving: bool = False
thumbnail_dragging: bool = False
last_pos: Optional[Tuple[float, float]] = None
move_start: Optional[Tuple[int, int]] = None
image_pos: List[int] = [0, 0]
zoom_scale: float = 1.0
image: Optional[np.ndarray] = None
image_path: Optional[str] = None
images: List[str] = []
current_image_index: int = 0
original_resolution: Tuple[int, int] = (0, 0)
thumbnail_rect: Optional[pygame.Rect] = None

# 工具栏设置
toolbar_width: int = 200
toolbar_rect: pygame.Rect = pygame.Rect((screen_width - toolbar_width) // 2, 0, toolbar_width, toolbar_height)
toolbar_dragging: bool = False
toolbar_offset: Tuple[int, int] = (0, 0)

# 工具变量
tool: str = 'brush'  # 当前工具: 'brush' 或 'rectangle'
rectangle_start_pos: Optional[Tuple[float, float]] = None

def load_images(folder_path: str) -> None:
    global images, image_path, image, zoom_scale, image_pos, original_resolution
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if images:
        image_path = images[0]
        image = cv2.imread(image_path)
        if image is not None:
            original_resolution = image.shape[1], image.shape[0]
            # 设置初始缩放比例，使图像在窗口中显示
            img_height, img_width = image.shape[:2]
            scale_w = (screen_width - sidebar_width) / img_width
            scale_h = screen_height / img_height
            zoom_scale = min(scale_w, scale_h) * 0.8  # 缩小一点
            # 居中显示图像
            image_pos = [(screen_width - sidebar_width - int(img_width * zoom_scale)) // 2, 
                         (screen_height - int(img_height * zoom_scale)) // 2]
            update_screen()

def is_within_image_bounds(pos: Tuple[int, int]) -> bool:
    x, y = pos
    return (sidebar_width <= x <= screen_width) and (toolbar_height <= y <= screen_height)

def update_screen() -> None:
    global image, zoom_scale, screen, image_pos, rectangle_start_pos, last_pos
    if screen:
        screen.fill(WHITE)
        if image is not None:
            # 计算可见区域
            visible_width = int((screen_width - sidebar_width) / zoom_scale)
            visible_height = int(screen_height / zoom_scale)
            image_center_x = -image_pos[0] / zoom_scale + visible_width / 2
            image_center_y = -image_pos[1] / zoom_scale + visible_height / 2

            # 计算裁剪框
            x1 = max(0, int(image_center_x - visible_width / 2))
            y1 = max(0, int(image_center_y - visible_height / 2))
            x2 = min(image.shape[1], int(image_center_x + visible_width / 2))
            y2 = min(image.shape[0], int(image_center_y + visible_height / 2))

            # 确保裁剪区域不为空
            if x1 < x2 and y1 < y2:
                cropped_image = image[y1:y2, x1:x2]
                resized_image = cv2.resize(cropped_image, (int((x2 - x1) * zoom_scale), int((y2 - y1) * zoom_scale)))
                image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                pygame_image = pygame.image.frombuffer(image_rgb.tobytes(), image_rgb.shape[1::-1], 'RGB')

                # 计算裁剪图像的新位置
                blit_pos = (
                    sidebar_width + image_pos[0] + x1 * zoom_scale,
                    image_pos[1] + y1 * zoom_scale
                )

                screen.blit(pygame_image, blit_pos)

            # 绘制实时矩形
            if tool == 'rectangle' and rectangle_start_pos is not None and drawing:
                current_pos = pygame.mouse.get_pos()
                if rectangle_start_pos is not None:
                    current_pos = (int((current_pos[0] - sidebar_width - image_pos[0]) / zoom_scale), 
                                   int((current_pos[1] - image_pos[1]) / zoom_scale))
                    pygame.draw.rect(screen, BLACK, (
                        min(rectangle_start_pos[0], current_pos[0]) * zoom_scale + sidebar_width + image_pos[0],
                        min(rectangle_start_pos[1], current_pos[1]) * zoom_scale + image_pos[1],
                        abs(rectangle_start_pos[0] - current_pos[0]) * zoom_scale,
                        abs(rectangle_start_pos[1] - current_pos[1]) * zoom_scale
                    ), 2)

        draw_sidebar()
        draw_image_info()
        draw_thumbnail(image)
        draw_toolbar()
        pygame.display.flip()

def draw_sidebar() -> None:
    pygame.draw.rect(screen, GRAY, (0, 0, sidebar_width, screen_height))
    next_button = pygame.Rect(50, 150, 100, 50)
    prev_button = pygame.Rect(50, 250, 100, 50)
    pygame.draw.rect(screen, WHITE, next_button)
    pygame.draw.rect(screen, WHITE, prev_button)

    font = pygame.font.SysFont(None, 24)
    next_text = font.render("Next", True, BLACK)
    prev_text = font.render("Prev", True, BLACK)
    screen.blit(next_text, (next_button.x + 10, next_button.y + 10))
    screen.blit(prev_text, (prev_button.x + 10, prev_button.y + 10))

def draw_image_info() -> None:
    font = pygame.font.SysFont(None, 24)
    index_text = font.render(f"Image {current_image_index + 1} of {len(images)}", True, BLACK)
    screen.blit(index_text, (10, 400))
    if images:
        image_name = os.path.basename(images[current_image_index])
        name_text = font.render(f"Name: {image_name}", True, BLACK)
        screen.blit(name_text, (10, 430))
        resolution_text = font.render(f"Resolution: {original_resolution[0]}x{original_resolution[1]}", True, BLACK)
        screen.blit(resolution_text, (10, 460))
        zoom_text = font.render(f"Zoom: {zoom_scale:.2f}x", True, BLACK)
        screen.blit(zoom_text, (10, 490))

def draw_toolbar() -> None:
    pygame.draw.rect(screen, GREEN, toolbar_rect)
    brush_button = pygame.Rect(toolbar_rect.x + 10, toolbar_rect.y + 10, 30, 30)
    rect_button = pygame.Rect(toolbar_rect.x + 50, toolbar_rect.y + 10, 30, 30)

    pygame.draw.rect(screen, BLUE if tool == 'brush' else WHITE, brush_button)
    pygame.draw.rect(screen, BLUE if tool == 'rectangle' else WHITE, rect_button)

    font = pygame.font.SysFont(None, 24)
    brush_text = font.render("B", True, BLACK)
    rect_text = font.render("R", True, BLACK)
    screen.blit(brush_text, (brush_button.x + 5, brush_button.y))
    screen.blit(rect_text, (rect_button.x + 5, rect_button.y))

def draw_thumbnail(image: Optional[np.ndarray]) -> None:
    global thumbnail_rect, original_resolution
    if image is None:
        return
    
    thumbnail_size = (150, 150)
    h, w = image.shape[:2]
    aspect_ratio = w / h
    if w > h:
        thumbnail_size = (150, int(150 / aspect_ratio))
    else:
        thumbnail_size = (int(150 * aspect_ratio), 150)
    thumbnail_image = cv2.resize(image, thumbnail_size)
    thumbnail_rgb = cv2.cvtColor(thumbnail_image, cv2.COLOR_BGR2RGB)
    thumbnail_surface = pygame.image.frombuffer(thumbnail_rgb.tobytes(), thumbnail_rgb.shape[1::-1], 'RGB')

    # 添加白边和黑边
    border_thickness = 2
    white_border_thickness = 4

    # 创建白边表面
    white_border_surface = pygame.Surface(
        (thumbnail_size[0] + 2 * white_border_thickness, thumbnail_size[1] + 2 * white_border_thickness))
    white_border_surface.fill(WHITE)

    # 将缩略图绘制到白边表面上
    white_border_surface.blit(thumbnail_surface, (white_border_thickness, white_border_thickness))

    # 创建黑边表面
    black_border_surface = pygame.Surface(
        (thumbnail_size[0] + 2 * (white_border_thickness + border_thickness), thumbnail_size[1] + 2 * (white_border_thickness + border_thickness)))
    black_border_surface.fill(BLACK)

    # 将白边表面绘制到黑边表面上
    black_border_surface.blit(white_border_surface, (border_thickness, border_thickness))

    # 计算缩略图位置
    thumbnail_position = (screen_width - black_border_surface.get_width() - 10, screen_height - black_border_surface.get_height() - 10)
    screen.blit(black_border_surface, thumbnail_position)

    # 记录缩略图位置和大小
    thumbnail_rect = pygame.Rect(
        thumbnail_position[0], thumbnail_position[1], black_border_surface.get_width(), black_border_surface.get_height())

    # 计算并绘制红色框
    if image is not None:
        thumbnail_ratio = thumbnail_size[0] / image.shape[1]
        view_x = -image_pos[0] / zoom_scale
        view_y = -image_pos[1] / zoom_scale
        view_w = (screen_width - sidebar_width) / zoom_scale
        view_h = screen_height / zoom_scale

        red_rect = pygame.Rect(
            thumbnail_position[0] + border_thickness + white_border_thickness + int(view_x * thumbnail_ratio),
            thumbnail_position[1] + border_thickness + white_border_thickness + int(view_y * thumbnail_ratio),
            int(view_w * thumbnail_ratio),
            int(view_h * thumbnail_ratio)
        )

        # 确保红框超出缩略图边界时保持边界不动
        if red_rect.left < thumbnail_position[0] + border_thickness + white_border_thickness:
            red_rect.width += red_rect.left - (thumbnail_position[0] + border_thickness + white_border_thickness)
            red_rect.left = thumbnail_position[0] + border_thickness + white_border_thickness
        if red_rect.top < thumbnail_position[1] + border_thickness + white_border_thickness:
            red_rect.height += red_rect.top - (thumbnail_position[1] + border_thickness + white_border_thickness)
            red_rect.top = thumbnail_position[1] + border_thickness + white_border_thickness
        if red_rect.right > thumbnail_position[0] + black_border_surface.get_width() - border_thickness - white_border_thickness:
            red_rect.width = thumbnail_position[0] + black_border_surface.get_width() - border_thickness - white_border_thickness - red_rect.left
        if red_rect.bottom > thumbnail_position[1] + black_border_surface.get_height() - border_thickness - white_border_thickness:
            red_rect.height = thumbnail_position[1] + black_border_surface.get_height() - border_thickness - white_border_thickness - red_rect.top

        pygame.draw.rect(screen, RED, red_rect, 1)

def handle_thumbnail_click(pos: Tuple[int, int]) -> None:
    global image_pos, zoom_scale, thumbnail_rect, original_resolution
    if thumbnail_rect is not None and thumbnail_rect.collidepoint(pos):
        relative_x = (pos[0] - thumbnail_rect.x - 6) / (thumbnail_rect.width - 12)  # Adjust for borders
        relative_y = (pos[1] - thumbnail_rect.y - 6) / (thumbnail_rect.height - 12)  # Adjust for borders
        image_pos = [int(-relative_x * original_resolution[0] * zoom_scale + (screen_width - sidebar_width) / 2), 
                     int(-relative_y * original_resolution[1] * zoom_scale + screen_height / 2)]
        update_screen()

def main() -> None:
    global drawing, moving, thumbnail_dragging, toolbar_dragging, last_pos, move_start, zoom_scale, current_image_index, image, screen_width, screen_height, screen, image_pos, original_resolution, tool, rectangle_start_pos, toolbar_offset
    running = True
    clock = pygame.time.Clock()

    load_images("./Pictures")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen_width, screen_height = event.size
                screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
                toolbar_rect.x = (screen_width - toolbar_rect.width) // 2  # 确保工具栏在屏幕顶部居中
                update_screen()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if toolbar_rect.collidepoint(event.pos):
                        if toolbar_rect.x + 10 <= event.pos[0] <= toolbar_rect.x + 40:
                            tool = 'brush'
                        elif toolbar_rect.x + 50 <= event.pos[0] <= toolbar_rect.x + 80:
                            tool = 'rectangle'
                        else:
                            toolbar_dragging = True
                            toolbar_offset = (event.pos[0] - toolbar_rect.x, event.pos[1] - toolbar_rect.y)
                    elif event.pos[0] < sidebar_width:
                        if 50 < event.pos[0] < 150 and 150 < event.pos[1] < 200:
                            # Next button
                            current_image_index = (current_image_index + 1) % len(images)
                            image = cv2.imread(images[current_image_index])
                            if image is not None:
                                original_resolution = image.shape[1], image.shape[0]
                                img_height, img_width = image.shape[:2]
                                scale_w = (screen_width - sidebar_width) / img_width
                                scale_h = screen_height / img_height
                                zoom_scale = min(scale_w, scale_h) * 0.8  # 缩小一点
                                image_pos = [(screen_width - sidebar_width - int(img_width * zoom_scale)) // 2, 
                                             (screen_height - int(img_height * zoom_scale)) // 2]
                                update_screen()
                        elif 50 < event.pos[0] < 150 and 250 < event.pos[1] < 300:
                            # Prev button
                            current_image_index = (current_image_index - 1) % len(images)
                            image = cv2.imread(images[current_image_index])
                            if image is not None:
                                original_resolution = image.shape[1], image.shape[0]
                                img_height, img_width = image.shape[:2]
                                scale_w = (screen_width - sidebar_width) / img_width
                                scale_h = screen_height / img_height
                                zoom_scale = min(scale_w, scale_h) * 0.8  # 缩小一点
                                image_pos = [(screen_width - sidebar_width - int(img_width * zoom_scale)) // 2, 
                                             (screen_height - int(img_height * zoom_scale)) // 2]
                                update_screen()
                    else:
                        if thumbnail_rect is not None and thumbnail_rect.collidepoint(event.pos):
                            thumbnail_dragging = True
                            handle_thumbnail_click(event.pos)
                        elif pygame.key.get_mods() & pygame.KMOD_CTRL and is_within_image_bounds(event.pos):
                            moving = True
                            move_start = event.pos
                        else:
                            drawing = True
                            if tool == 'brush':
                                last_pos = ((event.pos[0] - sidebar_width - image_pos[0]) / zoom_scale, 
                                            (event.pos[1] - image_pos[1]) / zoom_scale)
                            elif tool == 'rectangle':
                                rectangle_start_pos = (
                                    (event.pos[0] - sidebar_width - image_pos[0]) / zoom_scale, 
                                    (event.pos[1] - image_pos[1]) / zoom_scale)
                elif event.button == 4:  # Scroll up
                    if pygame.key.get_mods() & pygame.KMOD_CTRL:
                        zoom_scale *= 1.1
                        # Adjust image position to prevent it from moving while zooming
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        mouse_x -= sidebar_width
                        image_pos[0] = int(mouse_x - (mouse_x - image_pos[0]) * 1.1)
                        image_pos[1] = int(mouse_y - (mouse_y - image_pos[1]) * 1.1)
                        update_screen()
                elif event.button == 5:  # Scroll down
                    if pygame.key.get_mods() & pygame.KMOD_CTRL:
                        zoom_scale /= 1.1
                        # Adjust image position to prevent it from moving while zooming
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        mouse_x -= sidebar_width
                        image_pos[0] = int(mouse_x - (mouse_x - image_pos[0]) / 1.1)
                        image_pos[1] = int(mouse_y - (mouse_y - image_pos[1]) / 1.1)
                        update_screen()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left click
                    drawing = False
                    moving = False
                    thumbnail_dragging = False
                    toolbar_dragging = False
                    if tool == 'rectangle' and rectangle_start_pos is not None:
                        end_pos = ((event.pos[0] - sidebar_width - image_pos[0]) / zoom_scale, 
                                   (event.pos[1] - image_pos[1]) / zoom_scale)
                        if image is not None:
                            cv2.rectangle(image, (int(rectangle_start_pos[0]), int(rectangle_start_pos[1])), 
                                          (int(end_pos[0]), int(end_pos[1])), (0, 0, 0), 2)
                        rectangle_start_pos = None
                        update_screen()
            elif event.type == pygame.MOUSEMOTION:
                if toolbar_dragging:
                    toolbar_rect.x = event.pos[0] - toolbar_offset[0]
                    toolbar_rect.y = event.pos[1] - toolbar_offset[1]
                    toolbar_rect.y = max(0, toolbar_rect.y)  # 防止工具栏移动到屏幕顶部以外
                    update_screen()
                elif thumbnail_dragging:
                    handle_thumbnail_click(event.pos)
                elif moving and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    if move_start is not None:
                        image_pos[0] += event.pos[0] - move_start[0]
                        image_pos[1] += event.pos[1] - move_start[1]
                        move_start = event.pos
                        update_screen()
                elif drawing:
                    if tool == 'brush' and last_pos is not None:
                        current_pos = ((event.pos[0] - sidebar_width - image_pos[0]) / zoom_scale, 
                                       (event.pos[1] - image_pos[1]) / zoom_scale)
                        pygame.draw.line(screen, BLACK, 
                                         (last_pos[0] * zoom_scale + sidebar_width + image_pos[0], 
                                          last_pos[1] * zoom_scale + image_pos[1]), 
                                         (current_pos[0] * zoom_scale + sidebar_width + image_pos[0], 
                                          current_pos[1] * zoom_scale + image_pos[1]), 5)
                        if image is not None:
                            cv2.line(image, (int(last_pos[0]), int(last_pos[1])), 
                                     (int(current_pos[0]), int(current_pos[1])), (0, 0, 0), 5)
                        last_pos = current_pos
                        update_screen()
                    elif tool == 'rectangle' and rectangle_start_pos is not None:
                        update_screen()

        # Update the screen and ensure the toolbar is always on top
        update_screen()
        draw_toolbar()
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except pygame.error as e:
        print("Pygame error:", e)
    except Exception as e:
        print("Unexpected error:", e)
