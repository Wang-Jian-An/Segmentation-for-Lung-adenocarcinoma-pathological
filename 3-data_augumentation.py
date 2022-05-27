import os
import gc
import cv2
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

"""
具體目標：將圖片數量增加
使用 Data Augumentation 的方法：
    1. Rotation 180度
    2. CenterCrop
    3. RandomCrop
    4. ColorJitter Contrast
    5. Mixout
整個步驟：
    1. 輸入原始圖片、Mask Image
    2. 將圖片進行 Data Augumentation
    3. 將圖片儲存起來
"""

zipfile_path = "./SEG_Train_Mask_Images_Dataset_20220509_Width_Height.zip"
crop_shape = (928, 1696)

def Rotation_image(original_img: np.array, mask_img: np.array):
    rotation_transform = transforms.Compose([
        transforms.RandomRotation((180, 180))
    ])
    
    original_img = torch.from_numpy(original_img.transpose((2, 0, 1)) / 255)
    mask_img = torch.from_numpy(mask_img.transpose((2, 0, 1)) / 255)

    rotation_original_img = rotation_transform(original_img)
    rotation_mask_img = rotation_transform(mask_img)

    rotation_original_img = (rotation_original_img.detach().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    rotation_mask_img = np.where(rotation_mask_img.detach().numpy().transpose((1, 2, 0)) * 255 > 150, 255, 0).astype(np.uint8)
    return rotation_original_img, rotation_mask_img

def CenterCrop_image(original_img, mask_img):
    centercrop_transform = transforms.Compose([
        transforms.CenterCrop(size = crop_shape)
    ])

    original_img = torch.from_numpy(original_img.transpose((2, 0, 1)) / 255)
    mask_img = torch.from_numpy(mask_img.transpose((2, 0, 1)) / 255)

    centercrop_original_img = centercrop_transform(original_img)
    centercrop_mask_img = centercrop_transform(mask_img)

    centercrop_original_img = (centercrop_original_img.detach().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    centercrop_mask_img = np.where(centercrop_mask_img.detach().numpy().transpose((1, 2, 0)) * 255 > 150, 255, 0).astype(np.uint8)
    return centercrop_original_img, centercrop_mask_img

def RandomCrop_image():
    return 


def HorizontalFlip_image(original_img: np.array, mask_img: np.array):
    HorizontalFlip_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p = 1)
    ])

    original_img = torch.from_numpy(original_img.transpose((2, 0, 1)) / 255)
    mask_img = torch.from_numpy(mask_img.transpose((2, 0, 1)) / 255)

    horizontal_original_img = HorizontalFlip_transform(original_img)
    horizontal_mask_img = HorizontalFlip_transform(mask_img)

    horizontal_original_img = (horizontal_original_img.detach().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    horizontal_mask_img = np.where(horizontal_mask_img.detach().numpy().transpose((1, 2, 0)) * 255 > 150, 255, 0).astype(np.uint8)
    return horizontal_original_img, horizontal_mask_img

def ColorJitter_brightness(original_img: np.array, mask_img: np.array, brightness_num: int = 2):
    colorJitter_brightness_transform = transforms.Compose([
        transforms.ColorJitter(brightness = brightness_num)
    ])
    original_img = torch.from_numpy(original_img.transpose((2, 0, 1)) / 255)

    colorJitter_brightness_img = colorJitter_brightness_transform(original_img)

    colorJitter_brightness_img = (colorJitter_brightness_img.detach().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    return colorJitter_brightness_img, mask_img

def ColorJitter_contrast(original_img: np.array, mask_img: np.array, contrast_num: int = 4):
    colorJitter_contrast_transform = transforms.Compose([
        transforms.ColorJitter(contrast = contrast_num)
    ])
    original_img = torch.from_numpy(original_img.transpose((2, 0, 1)) / 255)

    colorJitter_contrast_img = colorJitter_contrast_transform(original_img)

    colorJitter_contrast_img = (colorJitter_contrast_img.detach().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    return colorJitter_contrast_img, mask_img

def mixout_two_images(original_image1, original_image2, mask_image1, mask_image2, lmbda: float):
    
    mixout_two_images = lmbda * original_image1 / 255 + (1-lmbda) * original_image2 / 255
    mixout_two_mask = lmbda * mask_image1 / 255 + (1-lmbda) * mask_image2 / 255

    return mixout_two_images, mixout_two_mask


### Step1. 輸入原始圖片、Mask Image（圖片最終的 shape = (W, H, C)） ###
zipData = zipfile.ZipFile(zipfile_path)
original_image_name = [i for i in zipData.namelist() if "mask" not in i and ".jpg" in i]
mask_image_name = [i for i in zipData.namelist() if "mask" in i and ".jpg" in i]

original_image = [plt.imread(zipData.open(i)) for i in original_image_name]
mask_image = [plt.imread(zipData.open(i))[:, :, np.newaxis] for i in mask_image_name]
### Step1. 輸入原始圖片、Mask Image（圖片最終的 shape = (W, H, C)） ###

### Step2. Data Augumentation ###

# Rotation
rotation_original_mask = [Rotation_image(original_img = original_img, mask_img = mask_img) for original_img, mask_img in zip(original_image, mask_image)]
original_image += [i[0] for i in rotation_original_mask]
mask_image += [i[1] for i in rotation_original_mask]
del rotation_original_mask
gc.collect()

# # CenterCrop 
# CenterCrop_original_mask = [CenterCrop_image(original_img = original_img, mask_img = mask_img) for original_img, mask_img in zip(original_image, mask_image)]
# original_image += [i[0] for i in CenterCrop_original_mask]
# mask_image += [i[1] for i in CenterCrop_original_mask]
# del CenterCrop

# ColorJitter Brightness and Contrast
# ColorJitter_Brightness_original_mask = [ColorJitter_brightness(original_img = original_img, mask_img = mask_img) for original_img, mask_img in zip(original_image, mask_image)]
ColorJitter_Contrast_original_mask = [ColorJitter_contrast(original_img = original_img, mask_img = mask_img) for original_img, mask_img in zip(original_image, mask_image)]
original_image += [i[0] for i in ColorJitter_Contrast_original_mask] # + [i[0] for i in ColorJitter_Brightness_original_mask]
mask_image +=  [i[1] for i in ColorJitter_Contrast_original_mask] # + [i[1] for i in ColorJitter_Brightness_original_mask]
del ColorJitter_Contrast_original_mask
gc.collect()

# Mixout Augumentation
lmbda = 0.5
half_number = (original_image.__len__() // 4) + 1
mixout_original_mask = [mixout_two_images(original_image1 = original_image[i_index], 
                                          original_image2 = original_image[j_index],
                                          mask_image1 = mask_image[i_index],
                                          mask_image2 = mask_image[j_index],
                                          lmbda = lmbda) for i_index, j_index in zip(range(half_number), range(half_number, int(half_number*2-2)))]
original_image += [i[0] for i in mixout_original_mask] # + [i[0] for i in ColorJitter_Brightness_original_mask]
mask_image +=  [i[1] for i in mixout_original_mask] # + [i[1] for i in ColorJitter_Brightness_original_mask]
del mixout_original_mask
gc.collect()
### Step2. Data Augumentation ###

### Step3. ###

if "Train_images" not in os.listdir(".//"):
    os.mkdir("Train_images")

if "Train_Mask" not in os.listdir(".//"):
    os.mkdir("Train_Mask")

# Step4. 逐一輸入圖片與 .json
for file_name, (one_image, one_mask) in enumerate(zip(original_image, mask_image)):

    # Step7. 將圖片、Mask 輸出
    one_image = cv2.cvtColor(one_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"Train_images/{file_name}.jpg", one_image)
    cv2.imwrite(f"Train_Mask/{file_name}.jpg", one_mask)
### Step3. ###
