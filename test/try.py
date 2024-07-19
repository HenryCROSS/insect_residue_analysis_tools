import cv2
import os
import numpy as np

def apply_mosaic_min(image, block_size):
    (h, w) = image.shape[:2]
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y:y+block_size, x:x+block_size]
            (B, G, R) = np.min(block.reshape(-1, 3), axis=0)
            image[y:y+block_size, x:x+block_size] = (B, G, R)
    return image

def apply_mosaic_mean(image, block_size):
    (h, w) = image.shape[:2]
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y:y+block_size, x:x+block_size]
            (B, G, R) = np.mean(block.reshape(-1, 3), axis=0).astype(int)
            image[y:y+block_size, x:x+block_size] = (B, G, R)
    return image

def apply_mosaic_white_if_white(image, block_size):
    (h, w) = image.shape[:2]
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y:y+block_size, x:x+block_size]
            if np.any(block == 255):
                image[y:y+block_size, x:x+block_size] = 255
    return image

def remove_small_circles(image, min_radius):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones_like(image) * 255
    for contour in contours:
        area = cv2.contourArea(contour)
        radius = int(np.sqrt(area / np.pi))
        if radius < min_radius:
            cv2.drawContours(mask, [contour], -1, 0, thickness=cv2.FILLED)
    result = cv2.bitwise_and(image, mask)
    return result

def find_and_remove_black_rectangles(image, binary_image, min_length):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    _, thresh_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if max(w, h) > min_length:
            cv2.rectangle(binary_image, (x, y), (x + w, y + h), 0, -1)
    return binary_image

def process_image(input_path, output_path, block_size, min_radius, min_length):
    image = cv2.imread(input_path)
    if image is None:
        print(f"Failed to load image {input_path}")
        return

    mosaic_image = apply_mosaic_min(image.copy(), block_size)
    final_image = apply_mosaic_mean(mosaic_image, block_size * 2)
    gray_final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    canny_edges = cv2.Canny(gray_final_image, 100, 200)
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(canny_edges, kernel, iterations=1)
    closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)
    eroded_edges = cv2.erode(closed_edges, kernel, iterations=1)

    final_mosaic_white = apply_mosaic_white_if_white(eroded_edges, block_size)
    final_mosaic_white = find_and_remove_black_rectangles(gray_final_image, final_mosaic_white, min_length)
    cleaned_edges = remove_small_circles(final_mosaic_white, min_radius)

    # Apply the mosaic effect: if there is any white pixel, make the block white
    final_result = apply_mosaic_white_if_white(cleaned_edges, block_size * 5)

    # Dilate and close to connect nearby white regions
    final_kernel = np.ones((20, 20), np.uint8)
    final_result_dilated = cv2.dilate(final_result, final_kernel, iterations=1)
    final_result_closed = cv2.morphologyEx(final_result_dilated, cv2.MORPH_CLOSE, final_kernel)

    # Find contours on the final result
    contours, _ = cv2.findContours(final_result_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw filled contours on the final result
    cv2.drawContours(final_result_closed, contours, -1, 255, thickness=cv2.FILLED)
    
    # Convert to BGR for color drawing
    final_result_closed_bgr = cv2.cvtColor(final_result_closed, cv2.COLOR_GRAY2BGR)
    image_with_red_contours = image.copy()
    cv2.drawContours(image_with_red_contours, contours, -1, (0, 0, 255), 2)

    # Overlay the final result on the original image
    overlayed_image = cv2.addWeighted(image_with_red_contours, 0.8, final_result_closed_bgr, 0.2, 0)

    cv2.imwrite(output_path, overlayed_image)

def main(input_dir, output_dir, block_size, min_radius, min_length):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_image(input_path, output_path, block_size, min_radius, min_length)
            print(f"Processed {input_path} and saved to {output_path}")

input_directory = './Pictures'
output_directory = './try_out'
block_size = 5
min_radius = 25
min_length = 1000

main(input_directory, output_directory, block_size, min_radius, min_length)
