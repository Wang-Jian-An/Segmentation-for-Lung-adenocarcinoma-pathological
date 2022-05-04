import gc
import cv2
import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


"""

程式撰寫大方向：
    1. 配對地匯入原始圖片與Mask
    2. 將所有圖片讀取
    3. 將所有圖片轉換成 Tensor
    3. 切割訓練、測試資料
    4. 建立 TensorDataset, DataLoader
    5. 定義模型、損失函數與優化器

程式撰寫大方向（每次訓練前再輸入圖片的版本）：
    1. 列出所有原始圖片、切片圖片的 ID
    2. 將 ID 做訓練、測試資料的切割
    3. 將訓練資料、測試資料進行 Batch
    4. 定義模型、損失函數與優化器
    5. 每個 Batch 的資料輪流取出來，進行後續的圖片讀取、圖片轉換成 Tensor、模型訓練或測試

"""

# 設定預設路徑
os.chdir("E://OneDrive - tmu.edu.tw//AI_Competition//肺腺癌病理切片影像之腫瘤氣道擴散偵測競賽 II：運用影像分割作法於切割STAS輪廓")

# 大方向 1：列出所有原始圖片、切割圖片的 ID
zipData = zipfile.ZipFile(".//SEG_Train_Mask_Images_Datasets.zip")
original_images = sorted([i for i in zipData.namelist() if ("mask" not in i) and (".jpg" in i)])
mask_images = sorted([i for i in zipData.namelist() if ("mask" in i) and (".jpg" in i)])

# 大方向 2：將 ID 做訓練、測試資料的切割
xtrain_id, xtest_id, ytrain_id, ytest_id = train_test_split(original_images, mask_images, shuffle = True, test_size = 0.2)

# 大方向 3：將訓練資料、測試資料進行 Batch
batch_size = 2
train_set_group = (xtrain_id.__len__() // batch_size) + 1
test_set_group = (xtest_id.__len__() // batch_size) + 1
trainset_batch = [(xtrain_id[one_batch * batch_size: one_batch * batch_size + batch_size], ytrain_id[one_batch * batch_size: one_batch * batch_size + batch_size])
                  if one_batch + 1 != train_set_group else (xtrain_id[one_batch * batch_size: ], ytrain_id[one_batch * batch_size: ]) for one_batch in range(train_set_group)]
testset_batch = [(xtrain_id[one_batch * batch_size: one_batch * batch_size + batch_size], ytrain_id[one_batch * batch_size: one_batch * batch_size + batch_size])
                  if one_batch + 1 != train_set_group else (xtrain_id[one_batch * batch_size: ], ytrain_id[one_batch * batch_size: ]) for one_batch in range(test_set_group)]
# print(testset_batch)


# 大方向 4：定義模型、損失函數與優化器
class segmentation_transfer_learning(nn.Module):
    def __init__(self):
        super(segmentation_transfer_learning, self).__init__()
        self.model = fcn_resnet50(pretrained = True)
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels = 21, out_channels = 1, kernel_size = 1),
            nn.Sigmoid()
        )
        return

    def forward(self, X):
        X = self.model(X)
        X = self.decoder(X["out"])
        return X

model = segmentation_transfer_learning()
loss_func = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)

# 大方向 5：每個 Batch 的資料輪流取出來，進行後續的圖片讀取、圖片轉換成 Tensor、模型訓練或測試
epochs = 10
train_loss_list, test_loss_list = list(), list()
train_loss, vali_loss = 0.0, 0.0

image_transform = transforms.Compose([
    transforms.ToTensor()
])


for epoch in range(epochs):
    train_loss, vali_loss = 0.0, 0.0
    model.train()
    for original_image, mask_image in trainset_batch:
        # 讀取圖片
        original_image = [plt.imread(zipData.open(i)) for i in tqdm(original_image, desc = "Load original Images")]
        mask_image = [plt.imread(zipData.open(i)) for i in tqdm(mask_image, desc = "Load Mask images")]

        # 將圖片轉換成 Tensor
        original_image = torch.concat([image_transform(i).unsqueeze(0) for i in original_image], axis = 0)
        mask_image = torch.concat([image_transform(i).unsqueeze(0) for i in mask_image], axis = 0)

        optimizer.zero_grad()
        yhat = model(original_image)

        loss = loss_func(torch.flatten(yhat), torch.flatten(mask_image).float())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        break
    scheduler.step()

    model.eval()
    for original_image, mask_image in testset_batch:
        # 讀取圖片
        original_image = [plt.imread(zipData.open(i)) for i in tqdm(original_image, desc = "Load original Images")]
        mask_image = [plt.imread(zipData.open(i)) for i in tqdm(mask_image, desc = "Load Mask images")]

        # 將圖片轉換成 Tensor
        original_image = torch.concat([image_transform(i).unsqueeze(0) for i in original_image], axis = 0)
        mask_image = torch.concat([image_transform(i).unsqueeze(0) for i in mask_image], axis = 0)

        with torch.no_grad():
            yhat = model(original_image)

        loss = loss_func(torch.flatten(yhat), torch.flatten(mask_image).float())
        vali_loss += loss.item()
        break
    train_loss_list.append(train_loss)
    test_loss_list.append(vali_loss)
    break
    # torch.save(model, "model//segmentation_fcn_resnet50.pth")
print(train_loss_list)
print(test_loss_list)