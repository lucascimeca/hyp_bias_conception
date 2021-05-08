import numpy as np
import matplotlib
from scipy import signal, stats
from utils.data_loader import *


def fft(img):
    return np.fft.fft2(img)


def fftshift(img):
    return np.fft.fftshift(fft(img))


def ifft(img):
    return np.fft.ifft2(img)


def ifftshift(img):
    return ifft(np.fft.ifftshift(img))


def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def mask_radial(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask


def generateSmoothKernel(data, r):
    result = np.zeros_like(data)
    [k1, k2, m, n] = data.shape
    mask = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                mask[i,j] = 1
            else:
                mask[i,j] = r
    mask = mask
    for i in range(m):
        for j in range(n):
            result[:,:, i,j] = signal.convolve2d(data[:,:, i,j], mask, boundary='symm', mode='same')
    return result


def generate_frequency_data_gray(imgs):
    img_freqs = []
    for i in range(imgs.shape[0]):
        fd = fftshift(imgs[i, :].reshape([28, 28]))
        img = ifftshift(fd)
        img_freqs.append(np.real(img).reshape([28 * 28]))

    return np.array(img_freqs)


def generate_frequency_data_color(dataset, n):
    img_freqs = []
    for i in range(n):
        tmp = np.zeros([dataset[i][0].shape[1], dataset[i][0].shape[2], 3])
        for j in range(3):
            tmp[:, :, j] = np.real(fftshift(dataset[i][0][j, :, :] / 255))
        img_freqs.append(tmp)

    return np.array(img_freqs)


if __name__ == '__main__':
    import sys
    version = sys.version_info

    feature_variants = ('object_number', 'color', 'shape')
    dsc = MultiColorDSpritesCreator(data_path='./../data/', filename='multi_color_dsprites.h5')
    round_one_dataset, round_two_datasets = dsc.get_dataset_fvar(
        number_of_samples=10000,
        features_variants=feature_variants,
        object_number=(0, 4),
        color=(0, 4),
        shape=(0, 3),
        scale=(0.5, 1),
        orientation=(0, 2 * np.pi),
        x_position=(0, 1),
        y_position=(0, 1),
    )

    for fig_num, feature in enumerate(feature_variants):

        dataset = round_two_datasets[feature]['train']

        img_freqs = generate_frequency_data_color(dataset, len(dataset))
        plt.figure(2*fig_num)
        plt.title("{} average spectra".format(feature))
        plt.imshow(np.mean(img_freqs, axis=0))
        plt.show()
        plt.figure(2*fig_num+1)
