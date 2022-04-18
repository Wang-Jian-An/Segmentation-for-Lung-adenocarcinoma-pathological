import cv2
import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader

"""

程式撰寫大方向：
    1. 配對地匯入原始圖片與Mask
    2. 將所有圖片讀取
    3. 切割訓練、測試資料
    4. 建立 TensorDataset, DataLoader

"""

# 大方向 1：配對地匯入原始圖片與Mask
zipData = zipfile.ZipFile("SEG_Train_Mask_Images_Datasets.zip")
original_images = sorted([i for i in zipData.namelist() if ("mask" not in i) and (".jpg" in i)])
mask_images = sorted([i for i in zipData.namelist() if ("mask" in i) and (".jpg" in i)])

# 大方向 2：將所有圖片讀取
original_images = [plt.imread(zipData.open(i)) for i in tqdm(original_images, desc = "Load original Images")]
mask_images = [plt.imread(zipData.open(i)) for i in tqdm(mask_images, desc = "Load Mask images")]
mask_images = [np.where(i > 0, 1, 0) for i in mask_images]

# 大方向 3：切割訓練、測試資料



