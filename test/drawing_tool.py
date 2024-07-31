import pygame
import cv2
import numpy as np
import os
from cv2.typing import MatLike
from typing import List, Tuple, Optional

class Image:
    def __init__(self, path: str, name: str):
        self._path: str = path
        self._name: str = name
        self._image: Optional[MatLike] = None
        self._preprocess_image: Optional[MatLike] = None

    def get_img(self) -> Optional[MatLike]:
        if self._image is None:
            input_path: str = os.path.join(self._path, self._name)
            self._image = cv2.imread(input_path)
        return self._image
    
    def get_preprocessed_img(self) -> Optional[MatLike]:
        if self._preprocess_image is None:
            self.preprocess()
        return self._preprocess_image
    
    def preprocess(self):
        self._preprocess_image, _ = process_image(self)

    def get_path(self) -> str:
        return self._path

    def get_name(self) -> str:
        return self._name

    def get_full_path(self) -> str:
        return os.path.join(self._path, self._name)

def process_image(src: Image) -> Tuple[Optional[MatLike], Optional[MatLike]]:
    original_img = src.get_img()
    if original_img is None:
        print(f"Failed to load image {src.get_full_path()}")
        return None, None

    large_overlayed_image, large_mask_bgr = process_image_larger_shape(original_img)
    return large_overlayed_image, large_mask_bgr

def process_image_larger_shape(img: MatLike) -> Tuple[MatLike, MatLike]:
    # Your existing image processing function
    img_color = img.copy()
    img = cv2.bitwise_not(img)

    # Apply Gaussian Blur
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

    # Remove blur using FFT
    fft_img = remove_blur_fft(blurred_img)

    # Create a convolution kernel for erosion
    kernel = np.ones((3, 3), np.uint8)
    eroded_img = cv2.erode(fft_img, kernel, iterations=1)

    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(eroded_img)

    # Apply Otsu's thresholding
    otsu_thresh_value, otsu_img = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove small circles (morphological open)
    kernel = np.ones((3, 3), np.uint8)
    cleaned_otsu_img = cv2.morphologyEx(otsu_img, cv2.MORPH_OPEN, kernel)

    # Dilate and close to connect nearby areas
    kernel = np.ones((10, 10), np.uint8)  # Adjust size as needed
    filter_mask = cv2.dilate(cleaned_otsu_img, kernel, iterations=1)
    connected_img = cv2.morphologyEx(filter_mask, cv2.MORPH_CLOSE, kernel)

    # Generate solid polygon
    contours, _ = cv2.findContours(connected_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_polygon_img = np.zeros_like(connected_img)
    cv2.fillPoly(solid_polygon_img, contours, 255)

    # Remove small circles
    min_area = 2000  # Adjust min area as needed
    contours, _ = cv2.findContours(solid_polygon_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_img = np.zeros_like(solid_polygon_img)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(filtered_img, [contour], -1, 255, thickness=cv2.FILLED)

    # Create a transparent blue layer
    blue_layer = np.zeros_like(img_color)
    blue_layer[:, :] = (255, 0, 0)  # Blue
    alpha = 0.2  # Transparency

    # Apply blue layer to the dilated area
    mask_bool = filtered_img.astype(bool)
    img_color[mask_bool] = cv2.addWeighted(img_color, 1 - alpha, blue_layer, alpha, 0)[mask_bool]

    # Draw red contours on the original image
    contours, hierarchy = cv2.findContours(filtered_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cv2.drawContours(img_color, contours, i, (0, 0, 255), 2)

    return img_color, filtered_img

def remove_blur_fft(image):
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

class Widget:
    def __init__(self, rect: pygame.Rect, color: Tuple[int, int, int], draggable: bool = False):
        self.rect = rect
        self.color = color
        self.draggable = draggable
        self.dragging = False
        self.children = []
        self.drag_offset = (0, 0)

    def add_child(self, child: 'Widget') -> None:
        self.children.append(child)

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(screen, self.color, self.rect)
        for child in self.children:
            child.draw(screen)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and self.draggable and self.rect.collidepoint(event.pos):
            self.dragging = True
            self.drag_offset = (event.pos[0] - self.rect.x, event.pos[1] - self.rect.y)
            return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            dx = event.pos[0] - self.drag_offset[0] - self.rect.x
            dy = event.pos[1] - self.drag_offset[1] - self.rect.y
            self.rect.x = event.pos[0] - self.drag_offset[0]
            self.rect.y = event.pos[1] - self.drag_offset[1]
            for child in self.children:
                child.rect.x += dx
                child.rect.y += dy
        return False

    def is_clicked(self, pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)


class Button(Widget):
    def __init__(self, rect: pygame.Rect, color: Tuple[int, int, int], text: str, text_color: Tuple[int, int, int]):
        super().__init__(rect, color)
        self.text = text
        self.text_color = text_color
        self.font = pygame.font.SysFont(None, 24)
        self.rendered_text = self.font.render(self.text, True, self.text_color)
        self.on_click = lambda: None  # Default no-op click handler

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(screen, self.color, self.rect)
        text_rect = self.rendered_text.get_rect(center=self.rect.center)
        screen.blit(self.rendered_text, text_rect)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if super().handle_event(event):
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and self.is_clicked(event.pos):
            self.on_click()
            return True
        return False


class ImageWidget(Widget):
    def __init__(self, rect: pygame.Rect, color: Tuple[int, int, int]):
        super().__init__(rect, color)
        self.image = None
        self.image_pos = [0, 0]
        self.zoom_scale = 1.0
        self.drawing = False
        self.moving = False
        self.last_pos = None
        self.move_start = None
        self.rectangle_start_pos = None
        self.brush_color = (0, 0, 0)
        self.thumbnail_rect = None
        self.original_resolution = (0, 0)
        self.tool = 'brush'
        self.thumbnail_dragging = False

    def set_image(self, image: MatLike) -> None:
        self.image = image
        if self.image is not None:
            self.original_resolution = self.image.shape[1], self.image.shape[0]
            img_height, img_width = self.image.shape[:2]
            scale_w = self.rect.width / img_width
            scale_h = self.rect.height / img_height
            self.zoom_scale = min(scale_w, scale_h) * 0.8
            self.image_pos = [(self.rect.width - int(img_width * self.zoom_scale)) // 2, 
                              (self.rect.height - int(img_height * self.zoom_scale)) // 2]

    def draw(self, screen: pygame.Surface) -> None:
        super().draw(screen)
        if self.image is not None:
            self.draw_image(screen)
            self.draw_thumbnail(screen)

    def draw_image(self, screen: pygame.Surface) -> None:
        visible_width = int(self.rect.width / self.zoom_scale)
        visible_height = int(self.rect.height / self.zoom_scale)
        image_center_x = -self.image_pos[0] / self.zoom_scale + visible_width / 2
        image_center_y = -self.image_pos[1] / self.zoom_scale + visible_height / 2

        x1 = max(0, int(image_center_x - visible_width / 2))
        y1 = max(0, int(image_center_y - visible_height / 2))
        x2 = min(self.image.shape[1], int(image_center_x + visible_width / 2))
        y2 = min(self.image.shape[0], int(image_center_y + visible_height / 2))

        if x1 < x2 and y1 < y2:
            cropped_image = self.image[y1:y2, x1:x2]
            resized_image = cv2.resize(cropped_image, (int((x2 - x1) * self.zoom_scale), int((y2 - y1) * self.zoom_scale)))
            image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            pygame_image = pygame.image.frombuffer(image_rgb.tobytes(), image_rgb.shape[1::-1], 'RGB')

            blit_pos = (self.rect.x + self.image_pos[0] + x1 * self.zoom_scale,
                        self.rect.y + self.image_pos[1] + y1 * self.zoom_scale)
            screen.blit(pygame_image, blit_pos)

        if self.tool == 'rectangle' and self.rectangle_start_pos is not None and self.drawing:
            current_pos = pygame.mouse.get_pos()
            current_pos = (int((current_pos[0] - self.rect.x - self.image_pos[0]) / self.zoom_scale), 
                           int((current_pos[1] - self.rect.y - self.image_pos[1]) / self.zoom_scale))
            pygame.draw.rect(screen, (0, 0, 0), (
                min(self.rectangle_start_pos[0], current_pos[0]) * self.zoom_scale + self.rect.x + self.image_pos[0],
                min(self.rectangle_start_pos[1], current_pos[1]) * self.zoom_scale + self.rect.y + self.image_pos[1],
                abs(self.rectangle_start_pos[0] - current_pos[0]) * self.zoom_scale,
                abs(self.rectangle_start_pos[1] - current_pos[1]) * self.zoom_scale
            ), 2)

    def draw_thumbnail(self, screen: pygame.Surface) -> None:
        if self.image is None:
            return
        
        thumbnail_size = (150, 150)
        h, w = self.image.shape[:2]
        aspect_ratio = w / h
        if w > h:
            thumbnail_size = (150, int(150 / aspect_ratio))
        else:
            thumbnail_size = (int(150 * aspect_ratio), 150)
        thumbnail_image = cv2.resize(self.image, thumbnail_size)
        thumbnail_rgb = cv2.cvtColor(thumbnail_image, cv2.COLOR_BGR2RGB)
        thumbnail_surface = pygame.image.frombuffer(thumbnail_rgb.tobytes(), thumbnail_rgb.shape[1::-1], 'RGB')

        border_thickness = 2
        white_border_thickness = 4

        white_border_surface = pygame.Surface(
            (thumbnail_size[0] + 2 * white_border_thickness, thumbnail_size[1] + 2 * white_border_thickness))
        white_border_surface.fill((255, 255, 255))

        white_border_surface.blit(thumbnail_surface, (white_border_thickness, white_border_thickness))

        black_border_surface = pygame.Surface(
            (thumbnail_size[0] + 2 * (white_border_thickness + border_thickness), thumbnail_size[1] + 2 * (white_border_thickness + border_thickness)))
        black_border_surface.fill((0, 0, 0))

        black_border_surface.blit(white_border_surface, (border_thickness, border_thickness))

        thumbnail_position = (self.rect.x + self.rect.width - black_border_surface.get_width() - 10, 
                              self.rect.y + self.rect.height - black_border_surface.get_height() - 10)
        screen.blit(black_border_surface, thumbnail_position)

        self.thumbnail_rect = pygame.Rect(
            thumbnail_position[0], thumbnail_position[1], black_border_surface.get_width(), black_border_surface.get_height())

        if self.image is not None:
            thumbnail_ratio = thumbnail_size[0] / self.image.shape[1]
            view_x = -self.image_pos[0] / self.zoom_scale
            view_y = -self.image_pos[1] / self.zoom_scale
            view_w = self.rect.width / self.zoom_scale
            view_h = self.rect.height / self.zoom_scale

            red_rect = pygame.Rect(
                thumbnail_position[0] + border_thickness + white_border_thickness + int(view_x * thumbnail_ratio),
                thumbnail_position[1] + border_thickness + white_border_thickness + int(view_y * thumbnail_ratio),
                int(view_w * thumbnail_ratio),
                int(view_h * thumbnail_ratio)
            )

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

            pygame.draw.rect(screen, (255, 0, 0), red_rect, 1)

    def handle_thumbnail_click(self, pos: Tuple[int, int]) -> None:
        if self.thumbnail_rect is not None and self.thumbnail_rect.collidepoint(pos):
            relative_x = (pos[0] - self.thumbnail_rect.x - 6) / (self.thumbnail_rect.width - 12)  # Adjust for borders
            relative_y = (pos[1] - self.thumbnail_rect.y - 6) / (self.thumbnail_rect.height - 12)  # Adjust for borders
            self.image_pos = [int(-relative_x * self.original_resolution[0] * self.zoom_scale + self.rect.width / 2), 
                              int(-relative_y * self.original_resolution[1] * self.zoom_scale + self.rect.height / 2)]

    def handle_event(self, event: pygame.event.Event) -> bool:
        if super().handle_event(event):
            return True

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                if self.thumbnail_rect is not None and self.thumbnail_rect.collidepoint(event.pos):
                    self.thumbnail_dragging = True
                    self.handle_thumbnail_click(event.pos)
                elif self.is_within_image_bounds(event.pos):
                    if pygame.key.get_mods() & pygame.KMOD_CTRL:
                        self.moving = True
                        self.move_start = event.pos
                    else:
                        self.drawing = True
                        if self.tool == 'brush':
                            self.last_pos = ((event.pos[0] - self.rect.x - self.image_pos[0]) / self.zoom_scale, 
                                             (event.pos[1] - self.rect.y - self.image_pos[1]) / self.zoom_scale)
                        elif self.tool == 'rectangle':
                            self.rectangle_start_pos = (
                                (event.pos[0] - self.rect.x - self.image_pos[0]) / self.zoom_scale, 
                                (event.pos[1] - self.rect.y - self.image_pos[1]) / self.zoom_scale)
            elif event.button == 4:  # Scroll up
                if pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.zoom_scale *= 1.1
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    mouse_x -= self.rect.x
                    self.image_pos[0] = int(mouse_x - (mouse_x - self.image_pos[0]) * 1.1)
                    self.image_pos[1] = int(mouse_y - (mouse_y - self.image_pos[1]) * 1.1)
            elif event.button == 5:  # Scroll down
                if pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.zoom_scale /= 1.1
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    mouse_x -= self.rect.x
                    self.image_pos[0] = int(mouse_x - (mouse_x - self.image_pos[0]) / 1.1)
                    self.image_pos[1] = int(mouse_y - (mouse_y - self.image_pos[1]) / 1.1)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left click
                self.drawing = False
                self.moving = False
                self.thumbnail_dragging = False
                if self.tool == 'rectangle' and self.rectangle_start_pos is not None:
                    end_pos = ((event.pos[0] - self.rect.x - self.image_pos[0]) / self.zoom_scale, 
                               (event.pos[1] - self.rect.y - self.image_pos[1]) / self.zoom_scale)
                    if self.image is not None:
                        cv2.rectangle(self.image, (int(self.rectangle_start_pos[0]), int(self.rectangle_start_pos[1])), 
                                      (int(end_pos[0]), int(end_pos[1])), (0, 0, 0), 2)
                    self.rectangle_start_pos = None

        elif event.type == pygame.MOUSEMOTION:
            if self.thumbnail_dragging:
                self.handle_thumbnail_click(event.pos)
            elif self.moving and pygame.key.get_mods() & pygame.KMOD_CTRL:
                if self.move_start is not None:
                    self.image_pos[0] += event.pos[0] - self.move_start[0]
                    self.image_pos[1] += event.pos[1] - self.move_start[1]
                    self.move_start = event.pos
            elif self.drawing:
                if self.tool == 'brush' and self.last_pos is not None:
                    current_pos = ((event.pos[0] - self.rect.x - self.image_pos[0]) / self.zoom_scale, 
                                   (event.pos[1] - self.rect.y - self.image_pos[1]) / self.zoom_scale)
                    pygame.draw.line(pygame.display.get_surface(), self.brush_color,  # Use brush color
                                     (self.last_pos[0] * self.zoom_scale + self.rect.x + self.image_pos[0], 
                                      self.last_pos[1] * self.zoom_scale + self.rect.y + self.image_pos[1]), 
                                     (current_pos[0] * self.zoom_scale + self.rect.x + self.image_pos[0], 
                                      current_pos[1] * self.zoom_scale + self.rect.y + self.image_pos[1]), 5)
                    if self.image is not None:
                        cv2.line(self.image, (int(self.last_pos[0]), int(self.last_pos[1])), 
                                 (int(current_pos[0]), int(current_pos[1])), (0, 0, 0), 5)
                    self.last_pos = current_pos
        return True

    def is_within_image_bounds(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return (self.rect.x <= x <= self.rect.x + self.rect.width) and (self.rect.y <= y <= self.rect.y + self.rect.height)


class PygameDrawingTool:
    def __init__(self, screen_width: int = 1000, screen_height: int = 600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.sidebar_width = 200

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (200, 200, 200)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)

        self.images = []
        self.current_image_index = 0
        self.show_preprocessed = False

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("Pygame Drawing Tool with OpenCV")

        self.sidebar = Widget(pygame.Rect(0, 0, self.sidebar_width, self.screen_height), self.GRAY)
        self.image_widget = ImageWidget(pygame.Rect(self.sidebar_width, 0, self.screen_width - self.sidebar_width, self.screen_height), self.WHITE)
        self.create_buttons()

    def create_buttons(self) -> None:
        next_button = Button(pygame.Rect(50, 150, 100, 50), self.WHITE, "Next", self.BLACK)
        prev_button = Button(pygame.Rect(50, 250, 100, 50), self.WHITE, "Prev", self.BLACK)
        img_switch_button = Button(pygame.Rect(50, 350, 100, 50), self.WHITE, "Preprocess", self.BLACK)
        brush_button = Button(pygame.Rect(10, 10, 30, 30), self.BLUE, "B", self.BLACK)
        rect_button = Button(pygame.Rect(50, 10, 30, 30), self.WHITE, "R", self.BLACK)

        next_button.on_click = self.next_image
        prev_button.on_click = self.prev_image
        img_switch_button.on_click = self.preprocess_image
        
        def brush_click():
            self.image_widget.tool = 'brush'
            brush_button.color = self.BLUE
            rect_button.color = self.WHITE

        def rect_click():
            self.image_widget.tool = 'rectangle'
            brush_button.color = self.WHITE
            rect_button.color = self.BLUE

        brush_button.on_click = brush_click
        rect_button.on_click = rect_click

        self.sidebar.add_child(next_button)
        self.sidebar.add_child(prev_button)
        self.sidebar.add_child(img_switch_button)
        self.sidebar.add_child(brush_button)
        self.sidebar.add_child(rect_button)

    def load_images(self, folder_path: str) -> None:
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.images = [Image(folder_path, f) for f in image_files]
        if self.images:
            self.current_image_index = 0
            self.update_image_display()

    def update_image_display(self):
        if self.show_preprocessed:
            self.image_widget.set_image(self.images[self.current_image_index].get_preprocessed_img())
        else:
            self.image_widget.set_image(self.images[self.current_image_index].get_img())

    def draw_sidebar_info(self, screen: pygame.Surface) -> None:
        if self.images:
            font = pygame.font.SysFont(None, 24)
            index_text = font.render(f"Image {self.current_image_index + 1} of {len(self.images)}", True, self.BLACK)
            screen.blit(index_text, (10, 400))
            image_name = self.images[self.current_image_index].get_name()
            name_text = font.render(f"Name: {image_name}", True, self.BLACK)
            screen.blit(name_text, (10, 430))
            resolution_text = font.render(f"Resolution: {self.image_widget.original_resolution[0]}x{self.image_widget.original_resolution[1]}", True, self.BLACK)
            screen.blit(resolution_text, (10, 460))
            zoom_text = font.render(f"Zoom: {self.image_widget.zoom_scale:.2f}x", True, self.BLACK)
            screen.blit(zoom_text, (10, 490))

    def update_screen(self) -> None:
        if self.screen:
            self.screen.fill(self.WHITE)
            self.sidebar.draw(self.screen)
            self.draw_sidebar_info(self.screen)
            self.image_widget.draw(self.screen)
            pygame.display.flip()

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.screen_width, self.screen_height = event.size
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
                self.sidebar.rect.height = self.screen_height
                self.image_widget.rect.width = self.screen_width - self.sidebar_width
                self.image_widget.rect.height = self.screen_height
                self.update_screen()
            elif event.type in [pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION]:
                sidebar_event_handled = any(child.handle_event(event) for child in self.sidebar.children)
                if not sidebar_event_handled:
                    self.image_widget.handle_event(event)

    def next_image(self) -> None:
        self.current_image_index = (self.current_image_index + 1) % len(self.images)
        self.update_image_display()

    def prev_image(self) -> None:
        self.current_image_index = (self.current_image_index - 1) % len(self.images)
        self.update_image_display()
    
    def preprocess_image(self) -> None:
        self.show_preprocessed = not self.show_preprocessed
        self.update_image_display()

    def run(self) -> None:
        self.running = True
        clock = pygame.time.Clock()
        self.load_images("./Pictures")

        while self.running:
            self.handle_events()
            self.update_screen()
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

if __name__ == "__main__":
    try:
        tool = PygameDrawingTool()
        tool.run()
    except pygame.error as e:
        print("Pygame error:", e)
    except Exception as e:
        print("Unexpected error:", e)
