import cv2
import numpy as np
import os

def find_and_replace_black_blocks(image, lower_black_threshold, upper_black_threshold, min_black_rectangle_area):
    # 复制图像，以免修改原图
    result_image = image.copy()
    
    # 计算整张图的平均颜色
    mean_color = cv2.mean(image)[:3]
    
    # 将图片从BGR转换为HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 创建掩膜，选择在阈值范围内的黑色区域
    mask_black = cv2.inRange(hsv_image, lower_black_threshold, upper_black_threshold)
    
    # 寻找掩膜中的轮廓
    contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 遍历所有找到的黑色轮廓
    for contour in contours_black:
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        # 计算轮廓的面积
        area = w * h
        
        # 如果面积大于指定的最小面积，则将该区域替换为平均颜色
        if area > min_black_rectangle_area:
            cv2.rectangle(result_image, (x, y), (x + w, y + h), mean_color, -1)
    
    return result_image

def apply_mosaic(image, mosaic_block_size):
    # 复制图像，以免修改原图
    result_image = image.copy()
    
    height, width = result_image.shape[:2]
    for i in range(0, height, mosaic_block_size):
        for j in range(0, width, mosaic_block_size):
            # 定义马赛克区域
            x_end = min(i + mosaic_block_size, height)
            y_end = min(j + mosaic_block_size, width)
            block = result_image[i:x_end, j:y_end]
            
            # 计算区域的平均颜色
            color = block.mean(axis=(0, 1)).astype(int)
            
            # 将区域填充为平均颜色
            result_image[i:x_end, j:y_end] = color
    
    return result_image

def apply_clahe_and_manual_threshold(image, lower_threshold=128, upper_threshold=255):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)
    
    # 使用手动上下阈值处理进行二值化
    binary = cv2.inRange(cl1, lower_threshold, upper_threshold)
    
    return binary

def draw_contours(image, binary_image):
    # 转换为4通道图像（BGRA）
    result_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # 寻找二值图像中的轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 在原图上绘制红色轮廓并填充为透明的浅蓝色
    for contour in contours:
        cv2.drawContours(result_image, [contour], -1, (0, 0, 255, 255), 2)  # 红色轮廓
        cv2.drawContours(result_image, [contour], -1, (255, 200, 200, 128), -1)  # 透明的浅蓝色填充
    
    return result_image

def process_images(input_folder, output_folder, mosaic_block_size=10, lower_threshold=128, upper_threshold=255):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 定义黑色区域的HSV阈值和最小黑色长方形面积
    lower_black_threshold = np.array([0, 0, 0])
    upper_black_threshold = np.array([180, 255, 30])
    min_black_rectangle_area = 20000

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 构建完整的文件路径
            file_path = os.path.join(input_folder, filename)
            # 读取图片
            image = cv2.imread(file_path)
            if image is None:
                print(f"Failed to read {file_path}")
                continue
            
            # 找到并替换黑色块
            image_with_replaced_blocks = find_and_replace_black_blocks(
                image, lower_black_threshold, upper_black_threshold, min_black_rectangle_area)
            
            # 应用马赛克效果
            image_with_mosaic = apply_mosaic(image_with_replaced_blocks, mosaic_block_size)
            
            # 应用CLAHE和手动阈值处理
            binary_image = apply_clahe_and_manual_threshold(image_with_mosaic, lower_threshold, upper_threshold)
            
            # 找到边界并绘制红线和填充透明的浅蓝色
            final_image = draw_contours(image_with_mosaic, binary_image)
            
            # 构建输出文件路径
            output_path = os.path.join(output_folder, filename)
            
            # 保存处理后的图片
            cv2.imwrite(output_path, final_image)
            print(f"Processed and saved {output_path}")

# 示例用法
input_folder = './Pictures'
output_folder = './new_out'
mosaic_block_size = 15  # 设置马赛克块的大小
lower_threshold = 0  # 设置手动阈值的下限值
upper_threshold = 120  # 设置手动阈值的上限值

process_images(input_folder, output_folder, mosaic_block_size, lower_threshold, upper_threshold)
