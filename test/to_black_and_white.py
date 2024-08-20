import cv2
import os
import numpy as np

def convert_images_to_grayscale_and_threshold(src_folder, dst_folder, threshold_value):
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(src_folder):
        # 构建文件的完整路径
        img_path = os.path.join(src_folder, filename)
        
        # 确保文件是图像文件
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 读取图像
            img = cv2.imread(img_path)
            
            # 检查图像是否成功读取
            if img is not None:
                # 转换为灰度图像
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # 应用阈值，提取超过阈值的区域
                _, thresholded_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
                
                # 构建目标文件路径
                dst_path = os.path.join(dst_folder, filename)
                
                # 保存阈值处理后的图像
                cv2.imwrite(dst_path, thresholded_img)
                print(f"已保存处理后的图像：{dst_path}")
            else:
                print(f"无法读取图像：{img_path}")
        else:
            print(f"跳过非图像文件：{filename}")

# 示例用法
src_folder = './human_output/AI4'  # 替换为源文件夹的路径
dst_folder = './AI14'  # 替换为目标文件夹的路径
threshold_value = 195  # 阈值，您可以根据需要调整

convert_images_to_grayscale_and_threshold(src_folder, dst_folder, threshold_value)
