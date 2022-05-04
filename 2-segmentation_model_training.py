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

"""

# 大方向 1：配對地匯入原始圖片與Mask
zipData = zipfile.ZipFile("..//SEG_Train_Mask_Images_Datasets.zip")
original_images = sorted([i for i in zipData.namelist() if ("mask" not in i) and (".jpg" in i)])
mask_images = sorted([i for i in zipData.namelist() if ("mask" in i) and (".jpg" in i)])

# 大方向 2：將所有圖片讀取
original_images = [plt.imread(zipData.open(i)) for i in tqdm(original_images, desc = "Load original Images")]
mask_images = [plt.imread(zipData.open(i)) for i in tqdm(mask_images, desc = "Load Mask images")]
mask_images = [np.where(i > 0, 1, 0) for i in mask_images]

# 大方向 3：將所有圖片轉換成 Tensor
image_transform = transforms.Compose([
    transforms.ToTensor()
])
original_images_tensor = [image_transform(i).unsqueeze(0) for i in original_images]
mask_images_tensor = [image_transform(i).unsqueeze(0) for i in mask_images]
del original_images, mask_images
gc.collect()

# 大方向 3：切割訓練、測試資料
xtrain, xtest, ytrain, ytest = train_test_split(original_images_tensor, mask_images_tensor, test_size = 0.2, shuffle = True)
del original_images_tensor, mask_images_tensor # 待自動化後可使用
gc.collect() # 待自動化後可使用

train_tensordataset = TensorDataset(torch.concat(xtrain, axis = 0),
                                    torch.concat(ytrain, axis = 0))
test_tensordataset = TensorDataset(torch.concat(xtest, axis = 0),
                                   torch.concat(ytest, axis = 0))
del xtrain, ytrain, xtest, ytest # 待自動化後可使用
gc.collect() # 待自動化後可使用

# 大方向 4：建立 TensorDataset, DataLoader
train_dataloader = DataLoader(train_tensordataset, batch_size = 1)
test_dataloader = DataLoader(test_tensordataset, batch_size = 1)
del train_tensordataset, test_tensordataset
gc.collect()

# 大方向 5：定義模型、損失函數與優化器
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

epochs = 10
model = segmentation_transfer_learning()
loss_func = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
scheduler = torch.optim.lr_scheduler(optimizer)
train_loss_list, test_loss_list = list(), list()
train_loss, vali_loss = 0.0, 0.0

for epoch in range(epochs):
    train_loss, vali_loss = 0.0, 0.0
    model.train()
    for original_image, mask_image in train_dataloader:
        optimizer.zero_grad()
        yhat = model(original_image)

        loss = loss_func(torch.flatten(yhat), torch.flatten(mask_image).float())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        break
    scheduler.step()

    model.eval()
    for original_image, mask_image in test_dataloader:
        with torch.no_grad():
            yhat = model(original_image)
        loss = loss_func(torch.flatten(yhat), torch.flatten(mask_image).float())
        vali_loss += loss.item()
    train_loss_list.append(train_loss)
    test_loss_list.append(vali_loss)

    torch.save(model, "model//segmentation_fcn_resnet50.pth")

