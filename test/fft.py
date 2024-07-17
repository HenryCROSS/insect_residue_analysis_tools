import os
import cv2 as cv
import numpy as np

def read_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img = cv.imread(os.path.join(folder, filename), cv.IMREAD_COLOR)
            if img is not None:
                images.append((filename, img))
    return images

def save_images_to_folder(folder, images):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for (filename, image) in images:
        cv.imwrite(os.path.join(folder, filename), image)

def perform_fft(image):
    fshift_channels = []
    magnitude_spectrum_channels = []
    for channel in cv.split(image):
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # Avoid log(0)
        fshift_channels.append(fshift)
        magnitude_spectrum_channels.append(magnitude_spectrum)
    
    # Merge the magnitude spectrums for visualization
    merged_magnitude_spectrum = cv.merge(magnitude_spectrum_channels)
    return fshift_channels, np.uint8(np.clip(merged_magnitude_spectrum, 0, 255))

def remove_bright_regions(fshift_channels, threshold):
    cleaned_fshift_channels = []
    for fshift in fshift_channels:
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        mask = magnitude_spectrum < threshold
        cleaned_fshift = fshift * mask
        cleaned_fshift_channels.append(cleaned_fshift)
    return cleaned_fshift_channels

def inverse_fft(fshift_channels):
    restored_channels = []
    for fshift in fshift_channels:
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        restored_channels.append(np.uint8(img_back))
    return cv.merge(restored_channels)

def apply_gaussian_blur(img, ksize=(5, 5)):
    return cv.GaussianBlur(img, ksize, 0)

def remove_small_objects_morphology(img, kernel_size=(3, 3)):
    # Define the structuring element for morphological operations
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    # Perform morphological opening (erosion followed by dilation)
    opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    return opening

def apply_clahe(img):
    clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def otsu_thresholding(img):
    _, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return thresh

def morphological_opening_and_closing(img, kernel_size=(3, 3)):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    # Perform opening
    opened = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    # Perform closing
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel)
    return closed

def filter_by_area(img, min_area):
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filtered_img = np.zeros_like(img)
    for contour in contours:
        if cv.contourArea(contour) >= min_area:
            cv.drawContours(filtered_img, [contour], -1, 255, thickness=cv.FILLED)
    return filtered_img

def dilate_image(img, kernel_size=(20, 20), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    dilated_img = cv.dilate(img, kernel, iterations=iterations)
    return dilated_img

def overlay_and_draw_contours(img_color, mask, alpha=0.2):
    # Create a transparent blue layer
    blue_layer = np.zeros_like(img_color)
    blue_layer[:, :] = (255, 0, 0)  # Blue
    mask_bool = mask.astype(bool)
    img_color[mask_bool] = cv.addWeighted(img_color, 1 - alpha, blue_layer, alpha, 0)[mask_bool]

    # Draw red contours
    contours, hierarchy = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cv.drawContours(img_color, contours, i, (0, 0, 255), 2)

    return img_color

def process_images(input_folder, output_folder, threshold, min_area):
    images = read_images_from_folder(input_folder)
    original_spectrum_images = []
    cleaned_spectrum_images = []
    inverse_images = []
    clahe_images = []
    otsu_images = []
    final_images = []
    polygon_images = []
    filtered_images = []
    dilated_images = []

    for (filename, image) in images:
        # Perform FFT and save the magnitude spectrum
        fshift_channels, magnitude_spectrum = perform_fft(image)
        original_spectrum_images.append((f"{filename}_original_spectrum.png", magnitude_spectrum))
        
        # Remove bright regions in the frequency domain
        cleaned_fshift_channels = remove_bright_regions(fshift_channels, threshold)
        cleaned_magnitude_spectrum_channels = [20 * np.log(np.abs(fshift) + 1) for fshift in cleaned_fshift_channels]
        cleaned_merged_magnitude_spectrum = cv.merge(cleaned_magnitude_spectrum_channels)
        cleaned_spectrum = np.uint8(np.clip(cleaned_merged_magnitude_spectrum, 0, 255))
        cleaned_spectrum_images.append((f"{filename}_cleaned_spectrum.png", cleaned_spectrum))
        
        # Inverse FFT to get the spatial domain image
        cleaned_image = inverse_fft(cleaned_fshift_channels)
        inverse_images.append((f"{filename}_inverse.png", cleaned_image))

        # Convert to grayscale if necessary and apply Gaussian blur
        gray_image = cv.cvtColor(cleaned_image, cv.COLOR_BGR2GRAY) if len(cleaned_image.shape) == 3 else cleaned_image
        blurred_image = apply_gaussian_blur(gray_image)

        # Remove small objects using morphological operations
        cleaned_image_without_small_objects = remove_small_objects_morphology(blurred_image)
        
        # Apply CLAHE
        clahe_image = apply_clahe(cleaned_image_without_small_objects)
        clahe_images.append((f"{filename}_clahe.png", clahe_image))

        # Apply Otsu's thresholding
        otsu_image = otsu_thresholding(clahe_image)
        otsu_images.append((f"{filename}_otsu.png", otsu_image))

        # Perform opening and closing morphological operations
        final_image = morphological_opening_and_closing(otsu_image)
        final_images.append((f"{filename}_final.png", final_image))

        # Filter by area
        filtered_img = filter_by_area(final_image, min_area)
        filtered_images.append((f"{filename}_filtered.png", filtered_img))

        # Dilate the filtered image
        dilated_img = dilate_image(filtered_img)
        dilated_images.append((f"{filename}_dilated.png", dilated_img))

        # Apply overlay and draw contours
        polygon_image = overlay_and_draw_contours(image.copy(), dilated_img)
        polygon_images.append((f"{filename}_polygon.png", polygon_image))

    # Save all intermediate and final images
    save_images_to_folder(os.path.join(output_folder, 'original_spectrum'), original_spectrum_images)
    save_images_to_folder(os.path.join(output_folder, 'cleaned_spectrum'), cleaned_spectrum_images)
    save_images_to_folder(os.path.join(output_folder, 'inverse_images'), inverse_images)
    save_images_to_folder(os.path.join(output_folder, 'clahe_images'), clahe_images)
    save_images_to_folder(os.path.join(output_folder, 'otsu_images'), otsu_images)
    save_images_to_folder(os.path.join(output_folder, 'final_images'), final_images)
    save_images_to_folder(os.path.join(output_folder, 'filtered_images'), filtered_images)
    save_images_to_folder(os.path.join(output_folder, 'dilated_images'), dilated_images)
    save_images_to_folder(os.path.join(output_folder, 'final_polygons'), polygon_images)

input_folder = './Pictures'
output_folder = './fft_out'
threshold = 240  # You can adjust this threshold value as needed
min_area = 2000  # Adjust the minimum area threshold as needed
process_images(input_folder, output_folder, threshold, min_area)
