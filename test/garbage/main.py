import cv2 as cv
import os
import numpy as np

def converge(image, k_size, gaussian_kernel=True):
    '''
    high pass filter via kernel convolution, low frequency signals will be merged into the average mean.
    if gaussian kernel is true then a gaussian kernel will be used, otherwise a mean kernel.
    '''
    image = image.astype(np.float32)/255
    average_color = np.asarray(cv.mean(image))*255
    average_color = average_color if len(image.shape) == 3 else average_color[0]
    if gaussian_kernel:
        blurred = cv.GaussianBlur(image,(k_size,k_size),0)
    else:
        blurred = cv.blur(image, (k_size, k_size))
    sub = blurred
    result = image - sub
    norm = cv.normalize(result, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    return norm, average_color.astype(int)

# Path to the folder containing images
folder_path = 'img'

# List all files in the folder
file_list = os.listdir(folder_path)
valid_list = []

window_names = ['image',
                'thresh',
                'blur',
                'opened',
                'roi',
                'thresh_2',
                'adaptive_thresh_mean',
                'adaptive_thresh_gaussian'
                ]
for n in window_names:
    cv.namedWindow(n, cv.WINDOW_NORMAL)


for file_name in file_list:
    # Create the full path to the file
    file_path = os.path.join(folder_path, file_name)
    # Check if the file is an image by its extension
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Read the image in BGR format (default for OpenCV)
        valid_list.append(file_path)
    else:
        print(f"Skipped non-image file {file_name}")
        
for f in valid_list:
    image = cv.imread(f)

    if image is None:
        print(f"Failed to load image {file_name}")
        continue
    # Convert the image from BGR to HSV color space

    blur = cv.medianBlur(image, 35)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 50, 255, cv.THRESH_BINARY)
    kernel = np.ones((35,35), dtype=np.uint8 )
    opened = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    
    roi_mask = np.zeros_like(thresh)
    
    contours, _ = cv.findContours(opened, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    all_points = np.vstack(contours)
    hull = cv.convexHull(all_points)
    cv.drawContours(roi_mask, [hull], 0, (255), -1)
    cv.rectangle(roi_mask, (0, 1483), (310, 2500), (0), -1)
    cv.rectangle(roi_mask, (1823, 1389), (2500, 2500), (0), -1)
    
    contours, _ = cv.findContours(roi_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image, contours, 0, (0, 0, 255), 3)
    
    cv.drawContours(roi_mask, contours, 0, 255, -1)
    
    kernel = np.ones((9, 9), np.uint8)
    roi_mask = cv.erode(roi_mask, kernel, iterations=1)
    
    
    # converged, average_color = converge(gray, 155)
    blur = cv.medianBlur(gray, 15)
    
    # abs_dist = 0
    # lb = int(average_color-abs_dist)
    # ub = int(average_color+abs_dist)
    # range_mask = cv.inRange(blur, 0, ub)

    adaptive_thresh_mean = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2)
    adaptive_thresh_mean = cv.bitwise_and(adaptive_thresh_mean, roi_mask)

    adaptive_thresh_gaussian = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    adaptive_thresh_gaussian = cv.bitwise_and(adaptive_thresh_gaussian, roi_mask)
    # hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # h, s, v = cv.split(hsv_image)
    # blur = cv.GaussianBlur(v, (15,15), 5)
    
    

    # Display the original image and the HSV channels
    cv.imshow('image', image)
    # cv.imshow('Hue Channel', h)
    # cv.imshow('Saturation Channel', s)
    # cv.imshow('Value Channel', v)
    cv.imshow('opened', opened)
    cv.imshow('roi', roi_mask)
    cv.imshow('thresh', thresh)
    cv.imshow('blur', blur)
    # cv.imshow('converged', converged)
    cv.imshow('adaptive_thresh_mean', adaptive_thresh_mean)
    cv.imshow('adaptive_thresh_gaussian', adaptive_thresh_gaussian)

    # Wait until a key is pressed
    k = cv.waitKey(0)
    if k == 27:
        break
    # Destroy all windows before moving to the next image

        
        
        
        
        
cv.destroyAllWindows()