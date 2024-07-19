import os
import cv2
import numpy as np

# 从文件夹读取图片
def read_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames

# 使用FFT去掉低频信号
def apply_fft(images):
    processed_images = []
    for img in images:
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 进行FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        # 去掉低频信号（中心部分）
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        r = 30  # 调整这个半径以适应你的需求
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - crow)**2 + (y - ccol)**2 <= r*r
        mask[mask_area] = 0
        
        fshift = fshift * mask
        
        # 进行逆FFT
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # 转换回BGR格式
        processed_img = cv2.cvtColor(np.uint8(img_back), cv2.COLOR_GRAY2BGR)
        processed_images.append(processed_img)
    return processed_images

# 应用CLAHE和形态学闭操作
def apply_clahe_and_morphology(images):
    processed_images = []
    for img in images:
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 应用CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        
        # 形态学闭操作
        kernel = np.ones((20, 20), np.uint8)
        closed_img = cv2.morphologyEx(clahe_img, cv2.MORPH_CLOSE, kernel)
        
        # 形态学开操作
        opened_img = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, kernel)
        
        # 转换回BGR格式
        processed_img = cv2.cvtColor(opened_img, cv2.COLOR_GRAY2BGR)
        processed_images.append(processed_img)
    return processed_images

# 对图像进行马赛克化处理
def apply_mosaic(images, mosaic_size=10):
    mosaic_images = []
    for img in images:
        h, w = img.shape[:2]
        temp_img = img.copy()

        # 处理每个马赛克块
        for y in range(0, h, mosaic_size):
            for x in range(0, w, mosaic_size):
                roi = temp_img[y:y + mosaic_size, x:x + mosaic_size]
                avg_color = roi.mean(axis=(0, 1)).astype(int)
                temp_img[y:y + mosaic_size, x:x + mosaic_size] = avg_color

        mosaic_images.append(temp_img)
    return mosaic_images

# 对图像进行K-means聚类处理
def apply_kmeans(images, k=6):
    kmeans_images = []
    for img in images:
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)
        
        # 定义K-means的条件
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        kmeans_img = res.reshape((img.shape))
        kmeans_images.append(kmeans_img)
    return kmeans_images

# 将处理后的图片与原图按50%比例重合
def blend_images(original_images, processed_images, alpha=0.5):
    blended_images = []
    for original, processed in zip(original_images, processed_images):
        blended = cv2.addWeighted(original, alpha, processed, 1 - alpha, 0)
        blended_images.append(blended)
    return blended_images

# 将处理后的图片保存到另一个文件夹
def save_images_to_folder(images, filenames, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for img, filename in zip(images, filenames):
        save_path = os.path.join(folder_path, filename)
        cv2.imwrite(save_path, img)

# 主函数
def main(input_folder, output_folder, mosaic_size):
    images, filenames = read_images_from_folder(input_folder)
    fft_images = apply_fft(images)
    clahe_morphology_images = apply_clahe_and_morphology(fft_images)
    mosaic_images = apply_mosaic(clahe_morphology_images, mosaic_size)
    kmeans_images = apply_kmeans(mosaic_images)
    blended_images = blend_images(images, kmeans_images)
    save_images_to_folder(kmeans_images, filenames, output_folder)

if __name__ == "__main__":
    input_folder = "./Pictures"
    output_folder = "./laplacian_out"
    mosaic_size = 5  # 设置马赛克的大小
    main(input_folder, output_folder, mosaic_size)
