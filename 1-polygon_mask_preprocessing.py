import zipfile
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.draw import polygon2mask

"""
程式目標：
步驟：
    1. 讀取 zipfile
    2. 分辨出圖片與 .json 檔
    3. 將檔案做配對
"""

# Step1. 讀取 zipfile
zipData = zipfile.ZipFile("SEG_Train_Datasets.zip")

# Step2. 分辨出圖片的 .json 檔
images_path = [i for i in zipData.namelist() if ".jpg" in i or ".png" in i]
mask_json_path = [i for i in zipData.namelist() if ".json" in i]

# Step3. 將檔案做配對
images_path = sorted(images_path)
mask_json_path = sorted(mask_json_path)

# Step4. 逐一輸入圖片與 .json
for one_image_path, one_json_path in zip(images_path, mask_json_path):
    # Step5. 分別讀入圖片與 .json
    


