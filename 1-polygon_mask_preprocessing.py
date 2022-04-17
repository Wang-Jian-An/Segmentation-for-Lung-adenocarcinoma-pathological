import os
import zipfile
import cv2
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
output_mask_name = ["mask_" + one_image.split("/")[-1] for one_image in images_path]


if "Train_images" not in os.listdir(".//"):
    os.mkdir("Train_images")

if "Train_Mask" not in os.listdir(".//"):
    os.mkdir("Train_Mask")
# Step4. 逐一輸入圖片與 .json
for one_image_path, one_json_path, one_mask_output_name in zip(images_path, mask_json_path, output_mask_name):
    # Step5. 分別讀入圖片（且要進行 Transpose）與 .json
    one_image = plt.imread(zipData.open(one_image_path)).transpose((1, 0, 2))
    one_json = json.load(zipData.open(one_json_path))
    
    # Step6. 從 .json 中挑選出各別的 mask 並合併後，得出該圖片完美的 mask
    all_mask = np.add.reduce([polygon2mask(image_shape = (one_image.shape[0], one_image.shape[1]), polygon = one_polygon["points"]) for one_polygon in one_json["shapes"]])
    
    # Step7. 將圖片、Mask 輸出
    one_image = cv2.cvtColor(one_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("Train_image/" + one_image_path.split("/")[-1], one_image)
    cv2.imwrite(f"Train_Mask/{one_mask_output_name}", all_mask)
    break
    


