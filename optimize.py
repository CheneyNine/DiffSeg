import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
from skimage.color import gray2rgb
from skimage.transform import resize
import random

def load_images(folder_path):
    images = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path).convert('L')  # 确保图像是灰度格式
            images.append(np.array(image))
    return np.stack(images),images

def calculate_pixelwise_mean_variance(images):
    # 计算逐像素的均值和方差
    mean = np.mean(images, axis=0)
    variance = np.var(images, axis=0)
    return mean, variance

def plot_image(data, title, cmap):
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()
def euclidean_distance(v1, v2):
    """计算两个向量之间的欧氏距离"""
    return np.sqrt(np.sum((v1 - v2) ** 2))

def generalized_energy_distance(images):
    """计算一组图像的广义能量距离（GED）"""
    N = len(images)
    distance_sum = 0

    # 计算两两图像之间的欧氏距离
    for i in range(N):
        for j in range(i + 1, N):  # 避免重复计算和自身计算
            distance_sum += euclidean_distance(images[i].flatten(), images[j].flatten())

    # 根据GED公式计算
    GED_squared = 2 * distance_sum / (N ** 2)
    return np.sqrt(GED_squared)


def densecrf_optimize(segmentations, original_image, K=3, H=4):
    """
    使用DenseCRF优化分割结果

    :param segmentations: 分割结果集合，假设是一个形状为[N, H, W]的numpy数组
    :param original_image: 原始图像，形状为[H, W, C]
    :param K: 迭代次数
    :param H: 每次迭代随机选取的分割结果数量
    :return: 优化后的分割结果
    """
    if len(original_image.shape) == 2:
        original_image = gray2rgb(original_image)
    original_image = resize(original_image, (segmentations.shape[1], segmentations.shape[2]),
                            preserve_range=True, anti_aliasing=True).astype(np.uint8)

    final_results = []

    for k in range(K):
        # 随机选择H个分割结果
        selected_indices = random.sample(range(segmentations.shape[0]), H)
        selected_segmentations = segmentations[selected_indices]

        # 计算平均分割结果作为数据项
        avg_segmentation = np.mean(selected_segmentations, axis=0)
        avg_segmentation_softmax = np.stack((avg_segmentation, 1 - avg_segmentation), axis=-1)  # 转换为Softmax概率

        avg_segmentation_softmax = np.transpose(avg_segmentation_softmax, (2, 0, 1))

        # 然后，使用reshape将其形状改为 (2, 128*119)
        avg_segmentation_softmax = avg_segmentation_softmax.reshape(2, -1)  # -1 表示自动计算该维度的大小


        print(avg_segmentation_softmax.shape)
        # 创建DenseCRF模型
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], 2)
        # 确保U是C风格的连续数组
        U = unary_from_softmax(avg_segmentation_softmax).copy()

        # 现在可以安全地传递U给setUnaryEnergy，而不会遇到内存连续性的问题
        d.setUnaryEnergy(U)

        # 添加颜色相关的平滑项
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image, compat=10)

        # 添加空间相关的平滑项
        d.addPairwiseGaussian(sxy=(3, 3), compat=3)

        # 进行能量最小化
        Q = d.inference(5)
        map_soln = np.argmax(Q, axis=0).reshape((original_image.shape[0], original_image.shape[1]))

        final_results.append(map_soln)

    # 计算所有迭代结果的平均值
    final_avg_result = np.mean(final_results, axis=0)
    final_avg_result_inverted = 255 - final_avg_result
    # 显示最终平均结果
    plt.imshow(final_avg_result_inverted, cmap='gray')
    plt.title('Final Average Result')
    plt.axis('off')
    plt.show()

    return final_avg_result


import numpy as np
from PIL import Image


def load_image_as_numpy_array(file_path):
    # 使用Pillow读取图像
    img = Image.open(file_path)
    # 确保图像是RGB格式
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # 将图像转换为Numpy数组并返回
    return np.array(img)




# 替换为你的图片文件夹路径
folder_path = '3.4'
npimages,images = load_images(folder_path)
mean, variance = calculate_pixelwise_mean_variance(npimages)
# 分别绘制映射到红蓝色彩的均值和方差图像
plot_image(mean, ' Coherence areas', 'gray')
plot_image(variance, 'Ambiguation areas', 'gray')
ged = generalized_energy_distance(images)
print(f"Generalized Energy Distance: {ged}")
originalimg=load_image_as_numpy_array('yuantu.png')
print(originalimg.shape)
print(npimages.shape)
densecrf_optimize(npimages,originalimg,3,4)
