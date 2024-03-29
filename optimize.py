import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pydensecrf.densecrf as dcrf
from PIL import Image
from pydensecrf.utils import unary_from_softmax
from skimage.color import gray2rgb
from skimage.transform import resize


def load_images(folder_path):
    images = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path).convert('L')
            images.append(np.array(image))
    return np.stack(images),images

def calculate_pixelwise_mean_variance(images):
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
    return np.sqrt(np.sum((v1 - v2) ** 2))

def generalized_energy_distance(images):
    N = len(images)
    distance_sum = 0
    for i in range(N):
        for j in range(i + 1, N):
            distance_sum += euclidean_distance(images[i].flatten(), images[j].flatten())
    GED_squared = 2 * distance_sum / (N ** 2)
    return np.sqrt(GED_squared)


def densecrf_optimize(segmentations, original_image, K=3, H=4):
    if len(original_image.shape) == 2:
        original_image = gray2rgb(original_image)
    original_image = resize(original_image, (segmentations.shape[1], segmentations.shape[2]),
                            preserve_range=True, anti_aliasing=True).astype(np.uint8)
    final_results = []
    for k in range(K):
        selected_indices = random.sample(range(segmentations.shape[0]), H)
        selected_segmentations = segmentations[selected_indices]
        avg_segmentation = np.mean(selected_segmentations, axis=0)
        avg_segmentation_softmax = np.stack((avg_segmentation, 1 - avg_segmentation), axis=-1)  # Softmax
        avg_segmentation_softmax = np.transpose(avg_segmentation_softmax, (2, 0, 1))
        avg_segmentation_softmax = avg_segmentation_softmax.reshape(2, -1)
        # print(avg_segmentation_softmax.shape)
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], 2)
        U = unary_from_softmax(avg_segmentation_softmax).copy()
        d.setUnaryEnergy(U)
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image, compat=10)
        d.addPairwiseGaussian(sxy=(3, 3), compat=3)
        Q = d.inference(5)
        map_soln = np.argmax(Q, axis=0).reshape((original_image.shape[0], original_image.shape[1]))
        final_results.append(map_soln)
    final_avg_result = np.mean(final_results, axis=0)
    final_avg_result_inverted = 255 - final_avg_result
    plt.imshow(final_avg_result_inverted, cmap='gray')
    plt.title('Final Average Result')
    plt.axis('off')
    plt.show()
    return final_avg_result

def load_image_as_numpy_array(file_path):
    img = Image.open(file_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)

folder_path = '3.4'
npimages,images = load_images(folder_path)
mean, variance = calculate_pixelwise_mean_variance(npimages)
plot_image(mean, ' Coherence areas', 'gray')
plot_image(variance, 'Ambiguation areas', 'gray')
ged = generalized_energy_distance(images)
print(f"Generalized Energy Distance: {ged}")
originalimg=load_image_as_numpy_array('yuantu.png')
print(originalimg.shape)
print(npimages.shape)
densecrf_optimize(npimages,originalimg,3,4)
