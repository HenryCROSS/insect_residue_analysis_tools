import cv2
import os
import numpy as np

# 定义输入输出文件夹路径
input_folder = './Pictures'
output_folder_mosaic = './mosaic_out'
output_folder_kmeans = './kmeans_out'

# 创建输出文件夹如果它们不存在
if not os.path.exists(output_folder_mosaic):
    os.makedirs(output_folder_mosaic)

if not os.path.exists(output_folder_kmeans):
    os.makedirs(output_folder_kmeans)

def apply_mosaic(image, block_size):
    """
    对图像应用马赛克效果，用块内的最小值替换。
    
    参数:
    image (ndarray): 输入灰度图像。
    block_size (int): 马赛克块的大小。
    
    返回:
    mosaic_img (ndarray): 应用马赛克效果后的图像。
    """
    height, width = image.shape
    mosaic_img = np.zeros_like(image)
    
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)
            block = image[y:y_end, x:x_end]
            min_val = np.min(block)
            mosaic_img[y:y_end, x:x_end] = min_val
    
    return mosaic_img

def apply_kmeans(image, k):
    """
    对图像应用K-means聚类算法。
    
    参数:
    image (ndarray): 输入图像。
    k (int): 聚类数。
    
    返回:
    kmeans_img (ndarray): 应用K-means聚类后的图像。
    """
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    
    # 定义K-means的标准
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 转换回8位图像
    centers = np.uint8(centers)
    kmeans_img = centers[labels.flatten()]
    kmeans_img = kmeans_img.reshape((image.shape))
    
    return kmeans_img

# 定义马赛克块的大小和K-means聚类数
mosaic_block_size = 10  # 可以根据需要调整大小
k_clusters = 5  # 可以根据需要调整聚类数

# 处理每个图像
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 构建文件路径
        img_path = os.path.join(input_folder, filename)
        
        # 读取图像
        img = cv2.imread(img_path)
        
        # 检查是否成功读取图像
        if img is None:
            print(f"Failed to read {img_path}")
            continue
        
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 应用Laplacian算子
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        
        # 转换Laplacian结果到8位图像
        laplacian_8u = cv2.convertScaleAbs(laplacian)
        
        # 应用CLAHE来增强对比度
        clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(laplacian_8u)
        
        # 应用马赛克效果
        mosaic_img = apply_mosaic(enhanced, mosaic_block_size)

        # 对原始图像应用K-means聚类
        kmeans_img = apply_kmeans(img, k_clusters)

        # 保存处理后的图像
        output_path_mosaic = os.path.join(output_folder_mosaic, filename)
        cv2.imwrite(output_path_mosaic, mosaic_img)

        output_path_kmeans = os.path.join(output_folder_kmeans, filename)
        cv2.imwrite(output_path_kmeans, kmeans_img)

        print(f"Processed and saved: {output_path_mosaic} and {output_path_kmeans}")
