import pygame
import cv2
import numpy as np
import os

# 初始化Pygame
pygame.init()

# 设置屏幕尺寸
screen_width = 1000
screen_height = 600
sidebar_width = 200
screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
pygame.display.set_caption("Pygame Drawing Tool with OpenCV")

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)

# 变量初始化
drawing = False
moving = False
thumbnail_dragging = False
last_pos = None
move_start = None
image_pos = [0, 0]
zoom_scale = 1.0
image = None
image_path = None
images = []
current_image_index = 0
original_resolution = (0, 0)
thumbnail_rect = None

def load_images(folder_path):
    global images, image_path, image, zoom_scale, image_pos, original_resolution
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if images:
        image_path = images[0]
        image = cv2.imread(image_path)
        original_resolution = image.shape[1], image.shape[0]
        # 设置初始缩放比例，使图像在窗口中显示
        img_height, img_width = image.shape[:2]
        scale_w = (screen_width - sidebar_width) / img_width
        scale_h = screen_height / img_height
        zoom_scale = min(scale_w, scale_h) * 0.8  # 缩小一点
        # 居中显示图像
        image_pos = [(screen_width - sidebar_width - int(img_width * zoom_scale)) // 2, (screen_height - int(img_height * zoom_scale)) // 2]
        update_screen()

def update_screen():
    global image, zoom_scale, current_image_index, screen, image_pos
    screen.fill(WHITE)
    if image is not None:
        resized_image = cv2.resize(image, (int(image.shape[1] * zoom_scale), int(image.shape[0] * zoom_scale)))
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        pygame_image = pygame.image.frombuffer(image_rgb.tobytes(), image_rgb.shape[1::-1], 'RGB')
        screen.blit(pygame_image, (sidebar_width + image_pos[0], image_pos[1]))
        draw_thumbnail(image)
    draw_sidebar()
    draw_image_info()
    pygame.display.flip()

def draw_sidebar():
    global screen
    pygame.draw.rect(screen, GRAY, (0, 0, sidebar_width, screen_height))
    next_button = pygame.Rect(50, 100, 100, 50)
    prev_button = pygame.Rect(50, 200, 100, 50)
    pygame.draw.rect(screen, WHITE, next_button)
    pygame.draw.rect(screen, WHITE, prev_button)
    
    font = pygame.font.SysFont(None, 24)
    next_text = font.render("Next", True, BLACK)
    prev_text = font.render("Prev", True, BLACK)
    screen.blit(next_text, (next_button.x + 10, next_button.y + 10))
    screen.blit(prev_text, (prev_button.x + 10, prev_button.y + 10))

def draw_image_info():
    global screen, zoom_scale, original_resolution
    font = pygame.font.SysFont(None, 24)
    index_text = font.render(f"Image {current_image_index + 1} of {len(images)}", True, BLACK)
    screen.blit(index_text, (10, 300))
    if images:
        image_name = os.path.basename(images[current_image_index])
        name_text = font.render(f"Name: {image_name}", True, BLACK)
        screen.blit(name_text, (10, 330))
        resolution_text = font.render(f"Resolution: {original_resolution[0]}x{original_resolution[1]}", True, BLACK)
        screen.blit(resolution_text, (10, 360))
        zoom_text = font.render(f"Zoom: {zoom_scale:.2f}x", True, BLACK)
        screen.blit(zoom_text, (10, 390))

def draw_thumbnail(image):
    global screen, screen_width, screen_height, thumbnail_rect, image_pos, zoom_scale
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
    white_border_surface = pygame.Surface((thumbnail_size[0] + 2 * white_border_thickness, thumbnail_size[1] + 2 * white_border_thickness))
    white_border_surface.fill(WHITE)

    # 将缩略图绘制到白边表面上
    white_border_surface.blit(thumbnail_surface, (white_border_thickness, white_border_thickness))

    # 创建黑边表面
    black_border_surface = pygame.Surface((thumbnail_size[0] + 2 * (white_border_thickness + border_thickness), thumbnail_size[1] + 2 * (white_border_thickness + border_thickness)))
    black_border_surface.fill(BLACK)

    # 将白边表面绘制到黑边表面上
    black_border_surface.blit(white_border_surface, (border_thickness, border_thickness))

    # 计算缩略图位置
    thumbnail_position = (screen_width - black_border_surface.get_width() - 10, screen_height - black_border_surface.get_height() - 10)
    screen.blit(black_border_surface, thumbnail_position)

    # 记录缩略图位置和大小
    thumbnail_rect = pygame.Rect(thumbnail_position[0], thumbnail_position[1], black_border_surface.get_width(), black_border_surface.get_height())

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

def draw_circle(screen, color, pos, radius=5):
    pygame.draw.circle(screen, color, pos, radius)

def handle_thumbnail_click(pos):
    global image_pos, zoom_scale, thumbnail_rect, original_resolution
    if thumbnail_rect is not None and thumbnail_rect.collidepoint(pos):
        relative_x = (pos[0] - thumbnail_rect.x - 6) / (thumbnail_rect.width - 12)  # Adjust for borders
        relative_y = (pos[1] - thumbnail_rect.y - 6) / (thumbnail_rect.height - 12)  # Adjust for borders
        image_pos = [-relative_x * original_resolution[0] * zoom_scale + (screen_width - sidebar_width) / 2, -relative_y * original_resolution[1] * zoom_scale + screen_height / 2]
        update_screen()

def main():
    global drawing, moving, thumbnail_dragging, last_pos, move_start, zoom_scale, current_image_index, image, screen_width, screen_height, screen, image_pos, original_resolution
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
                update_screen()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if event.pos[0] < sidebar_width:
                        if 50 < event.pos[0] < 150 and 100 < event.pos[1] < 150:
                            # Next button
                            current_image_index = (current_image_index + 1) % len(images)
                            image = cv2.imread(images[current_image_index])
                            original_resolution = image.shape[1], image.shape[0]
                            img_height, img_width = image.shape[:2]
                            scale_w = (screen_width - sidebar_width) / img_width
                            scale_h = screen_height / img_height
                            zoom_scale = min(scale_w, scale_h) * 0.8  # 缩小一点
                            image_pos = [(screen_width - sidebar_width - int(img_width * zoom_scale)) // 2, (screen_height - int(img_height * zoom_scale)) // 2]
                            update_screen()
                        elif 50 < event.pos[0] < 150 and 200 < event.pos[1] < 250:
                            # Prev button
                            current_image_index = (current_image_index - 1) % len(images)
                            image = cv2.imread(images[current_image_index])
                            original_resolution = image.shape[1], image.shape[0]
                            img_height, img_width = image.shape[:2]
                            scale_w = (screen_width - sidebar_width) / img_width
                            scale_h = screen_height / img_height
                            zoom_scale = min(scale_w, scale_h) * 0.8  # 缩小一点
                            image_pos = [(screen_width - sidebar_width - int(img_width * zoom_scale)) // 2, (screen_height - int(img_height * zoom_scale)) // 2]
                            update_screen()
                    else:
                        if thumbnail_rect is not None and thumbnail_rect.collidepoint(event.pos):
                            thumbnail_dragging = True
                            handle_thumbnail_click(event.pos)
                        elif pygame.key.get_mods() & pygame.KMOD_CTRL:
                            moving = True
                            move_start = event.pos
                        else:
                            drawing = True
                            last_pos = ((event.pos[0] - sidebar_width - image_pos[0]) / zoom_scale, (event.pos[1] - image_pos[1]) / zoom_scale)
                elif event.button == 4:  # Scroll up
                    if pygame.key.get_mods() & pygame.KMOD_CTRL:
                        zoom_scale *= 1.1
                        update_screen()
                elif event.button == 5:  # Scroll down
                    if pygame.key.get_mods() & pygame.KMOD_CTRL:
                        zoom_scale /= 1.1
                        update_screen()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left click
                    drawing = False
                    moving = False
                    thumbnail_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if thumbnail_dragging:
                    handle_thumbnail_click(event.pos)
                elif drawing:
                    if last_pos is not None:
                        current_pos = ((event.pos[0] - sidebar_width - image_pos[0]) / zoom_scale, (event.pos[1] - image_pos[1]) / zoom_scale)
                        pygame.draw.line(screen, BLACK, (last_pos[0] * zoom_scale + sidebar_width + image_pos[0], last_pos[1] * zoom_scale + image_pos[1]), (current_pos[0] * zoom_scale + sidebar_width + image_pos[0], current_pos[1] * zoom_scale + image_pos[1]), 5)
                        cv2.line(image, (int(last_pos[0]), int(last_pos[1])), (int(current_pos[0]), int(current_pos[1])), (0, 0, 0), 5)
                    last_pos = ((event.pos[0] - sidebar_width - image_pos[0]) / zoom_scale, (event.pos[1] - image_pos[1]) / zoom_scale)
                    update_screen()
                elif moving:
                    image_pos[0] += event.pos[0] - move_start[0]
                    image_pos[1] += event.pos[1] - move_start[1]
                    move_start = event.pos
                    update_screen()

        clock.tick(60)
    pygame.quit()

if __name__ == "__main__":
    main()
