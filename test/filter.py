import os
import cv2 as cv
import numpy as np

# 输入和输出文件夹路径
input_folder = './Pictures'
output_folder = './filter_out'

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

def remove_bright_regions(fshift_channels, threshold):
    cleaned_fshift_channels = []
    for fshift in fshift_channels:
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        mask = magnitude_spectrum < threshold
        cleaned_fshift = fshift * mask
        cleaned_fshift_channels.append(cleaned_fshift)
    return cleaned_fshift_channels

def process_images(input_folder, output_folder):
    # 获取输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # 检查文件是否为图像文件
        if os.path.isfile(input_path) and input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                img_color = cv.imread(input_path)  # 读取彩色图像

                img = cv.bitwise_not(img)

                # 应用高斯模糊
                blurred_img = cv.GaussianBlur(img, (5, 5), 0)

                # 使用FFT去除模糊区域
                fft_img = remove_blur_fft(blurred_img)

                # 对fft_img进行FFT处理，并去除亮区
                dft = cv.dft(np.float32(fft_img), flags=cv.DFT_COMPLEX_OUTPUT)
                dft_shift = np.fft.fftshift(dft)
                channels = cv.split(dft_shift)
                cleaned_channels = remove_bright_regions(channels, threshold=25)
                cleaned_dft_shift = cv.merge(cleaned_channels)
                
                # 逆傅里叶变换恢复图像
                f_ishift = np.fft.ifftshift(cleaned_dft_shift)
                img_back = cv.idft(f_ishift)
                fft_img = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
                cv.normalize(fft_img, fft_img, 0, 255, cv.NORM_MINMAX)
                fft_img = np.uint8(fft_img)

                # 创建一个卷积核进行腐蚀操作
                kernel = np.ones((3, 3), np.uint8)
                eroded_img = cv.erode(fft_img, kernel, iterations=1)

                # 创建一个 CLAHE 对象
                clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl1 = clahe.apply(eroded_img)

                # 初次应用 Otsu's 阈值处理
                otsu_thresh_value, otsu_img = cv.threshold(cl1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

                # 去除小圆点噪声（形态学开操作）
                kernel = np.ones((3, 3), np.uint8)
                cleaned_otsu_img = cv.morphologyEx(otsu_img, cv.MORPH_OPEN, kernel)

                # 膨胀和闭操作以连接距离较近的色块
                kernel = np.ones((10, 10), np.uint8)  # 根据需要调整大小
                dilated_img = cv.dilate(cleaned_otsu_img, kernel, iterations=1)
                connected_img = cv.morphologyEx(dilated_img, cv.MORPH_CLOSE, kernel)

                # 生成实心多边形
                contours, _ = cv.findContours(connected_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                solid_polygon_img = np.zeros_like(connected_img)
                cv.fillPoly(solid_polygon_img, contours, 255)

                # 去除小的圆点
                min_area = 1000  # 根据需要调整最小面积
                contours, _ = cv.findContours(solid_polygon_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                filtered_img = np.zeros_like(solid_polygon_img)
                for contour in contours:
                    if cv.contourArea(contour) >= min_area:
                        cv.drawContours(filtered_img, [contour], -1, 255, thickness=cv.FILLED)

                # 膨胀处理
                kernel = np.ones((10, 10), np.uint8)
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
                output_path_fft = os.path.join(output_folder, f'fft_{filename}')
                output_path_clahe = os.path.join(output_folder, f'clahe_{filename}')
                output_path_gray = os.path.join(output_folder, f'gray_{filename}')
                output_path_otsu = os.path.join(output_folder, f'otsu_{filename}')
                output_path_cleaned_otsu = os.path.join(output_folder, f'cleaned_otsu_{filename}')
                output_path_connected = os.path.join(output_folder, f'connected_{filename}')
                output_path_solid_polygon = os.path.join(output_folder, f'solid_polygon_{filename}')
                output_path_filtered = os.path.join(output_folder, f'filtered_{filename}')
                output_path_dilated = os.path.join(output_folder, f'dilated_{filename}')
                output_path_contour = os.path.join(output_folder, f'contour_{filename}')

                # 保存处理后的图像
                cv.imwrite(output_path_fft, fft_img)
                cv.imwrite(output_path_clahe, cl1)
                cv.imwrite(output_path_gray, img)
                cv.imwrite(output_path_otsu, otsu_img)
                cv.imwrite(output_path_cleaned_otsu, cleaned_otsu_img)
                cv.imwrite(output_path_connected, connected_img)
                cv.imwrite(output_path_solid_polygon, solid_polygon_img)
                cv.imwrite(output_path_filtered, filtered_img)
                cv.imwrite(output_path_dilated, dilated_img)
                cv.imwrite(output_path_contour, img_color)
            else:
                print(f"Failed to read image: {input_path}")

if __name__ == "__main__":
    process_images(input_folder, output_folder)
    print("Processing completed!")
