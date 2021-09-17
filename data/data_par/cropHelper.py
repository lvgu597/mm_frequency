import numpy as np
from scipy import signal
import cv2 as cv
import json
import os

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


def generateDataWithDifferentFrequencies_GrayScale(Images, r):
    Images_freq_low = []
    mask = mask_radial(np.zeros([28, 28]), r)
    for i in range(Images.shape[0]):
        fd = fftshift(Images[i, :].reshape([28, 28]))
        fd = fd * mask
        img_low = ifftshift(fd)
        Images_freq_low.append(np.real(img_low).reshape([28 * 28]))

    return np.array(Images_freq_low)

def generateDataWithDifferentFrequencies_3Channel(Images, r):
    Images_freq_low = []
    Images_freq_high = []
    mask = mask_radial(np.zeros([Images.shape[1], Images.shape[2]]), r)
    for i in range(Images.shape[0]):
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        # for j in range(3):
        #     fd = fftshift(Images[i, :, :, j])
        #     fd = fd * mask
        #     img_low = ifftshift(fd)
        #     tmp[:,:,j] = np.real(img_low)
        for j in range(3):
            fd = Images[i, :, :, j]
            fd = fd * mask
            img_low = fd
            tmp[:,:,j] = np.real(img_low)
        Images_freq_low.append(tmp)
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * (1 - mask)
            img_high = ifftshift(fd)
            tmp[:,:,j] = np.real(img_high)
        Images_freq_high.append(tmp)

    return np.array(Images_freq_low), np.array(Images_freq_high)

if __name__ == '__main__':
    file_json = '../../../tianchidata_coco_base/annotations/instances_val0331.json'
    with open(file_json, 'r') as f:
        file_names_all = json.load(f).get("images")
    file_names_all = [i.get('file_name') for i in file_names_all]  #单批次900+张图片太大了
    for times in range(0, len(file_names_all), 10):
        file_names = file_names_all[times:times+10]    
        val_images = np.zeros((len(file_names), 1920, 2560, 3))
        file_root = '../../../tianchidata_coco_base/defect_images/'
        for i in range(len(file_names)):
            val_images[i, :] = cv.imread(f'{file_root}{file_names[i]}')
        for r in [1440]: #240, 480, 720, 960, 1200, 1440, 1680, 1920
            save_dir = '../../../tianchidata_coco_base/'
            if not os.path.exists(f'{save_dir}val_image_low_{r}_crop'):
                os.mkdir(f'{save_dir}val_image_low_{r}_crop')
            # if not os.path.exists(f'{save_dir}val_image_high_{r}'):
            #     os.mkdir(f'{save_dir}val_image_high_{r}')
            val_image_low, val_image_high = generateDataWithDifferentFrequencies_3Channel(val_images, r)
            for i in range(len(file_names)):
                cv.imwrite(f'{save_dir}val_image_low_{r}_crop/{file_names[i]}',val_image_low[i,:])
                # cv.imwrite(f'{save_dir}val_image_high_{r}/{file_names[i]}',val_image_high[i,:])
                print(f'save {file_names[i]}')
            print("save success")
