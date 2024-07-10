import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 输入和输出文件夹路径
input_folder = './Pictures'
output_folder = './ProcessedPictures'

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

def remove_blur_fft(image):
    # 将图像转换为浮点型
    dft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 创建一个掩膜，中间部分设置为0，其余部分设置为1
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 30  # 半径
    center = (ccol, crow)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow) ** 2 + (y - ccol) ** 2 <= r * r
    mask[mask_area] = 0

    # 应用掩膜
    fshift = dft_shift * mask

    # 逆傅里叶变换
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # 归一化到0-255范围
    cv.normalize(img_back, img_back, 0, 255, cv.NORM_MINMAX)
    return np.uint8(img_back)

def apply_mosaic_block(image, block_size):
    (h, w) = image.shape[:2]
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            roi = image[y:y + block_size, x:x + block_size]
            avg_color = np.mean(roi)
            image[y:y + block_size, x:x + block_size] = avg_color
    return image

def process_images(input_folder, output_folder):
    # 获取输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # 检查文件是否为图像文件
        if os.path.isfile(input_path) and input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                img_color = cv.imread(input_path)  # 读取彩色图像

                # 使用FFT去除平滑变色的部分
                smooth_removed_img = remove_blur_fft(img)

                # 直接对去除平滑变色部分的图像应用马赛克效果
                block_size = 7  # 根据需要调整块的大小
                mosaic_img = apply_mosaic_block(smooth_removed_img, block_size)

                # 应用CLAHE进行对比度增强
                clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4))
                enhanced_img = clahe.apply(mosaic_img)

                # 应用Otsu阈值处理使得图像变成黑白色块
                _, black_white_img = cv.threshold(enhanced_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

                # 去除小的圆点
                min_area = 1000  # 根据需要调整最小面积
                contours, _ = cv.findContours(black_white_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                filtered_img = np.zeros_like(black_white_img)
                for contour in contours:
                    if cv.contourArea(contour) >= min_area:
                        cv.drawContours(filtered_img, [contour], -1, 255, thickness=cv.FILLED)

                # 膨胀处理
                kernel = np.ones((5, 5), np.uint8)
                dilated_img = cv.dilate(filtered_img, kernel, iterations=1)

                # 创建透明的蓝色图层
                blue_layer = np.zeros_like(img_color)
                blue_layer[:, :] = (255, 0, 0)  # 蓝色
                alpha = 0.2  # 透明度

                # 将蓝色图层应用到膨胀后的区域
                mask = dilated_img.astype(bool)
                img_color[mask] = cv.addWeighted(img_color, 1 - alpha, blue_layer, alpha, 0)[mask]

                # 在原图上绘制红色边界，包括内部的圈
                contours, hierarchy = cv.findContours(dilated_img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
                for i in range(len(contours)):
                    cv.drawContours(img_color, contours, i, (0, 0, 255), 2)

                # 生成输出文件路径
                output_path_smooth_removed = os.path.join(output_folder, f'smooth_removed_{filename}')
                output_path_mosaic = os.path.join(output_folder, f'mosaic_{filename}')
                output_path_enhanced = os.path.join(output_folder, f'enhanced_{filename}')
                output_path_bw = os.path.join(output_folder, f'bw_{filename}')
                output_path_filtered = os.path.join(output_folder, f'filtered_{filename}')
                output_path_dilated = os.path.join(output_folder, f'dilated_{filename}')
                output_path_contour = os.path.join(output_folder, f'contour_{filename}')

                # 保存处理后的图像
                cv.imwrite(output_path_smooth_removed, smooth_removed_img)
                cv.imwrite(output_path_mosaic, mosaic_img)
                cv.imwrite(output_path_enhanced, enhanced_img)
                cv.imwrite(output_path_bw, black_white_img)
                cv.imwrite(output_path_filtered, filtered_img)
                cv.imwrite(output_path_dilated, dilated_img)
                cv.imwrite(output_path_contour, img_color)
            else:
                print(f"Failed to read image: {input_path}")

if __name__ == "__main__":
    process_images(input_folder, output_folder)
    print("Processing completed!")
