#!/usr/bin/env python
# coding: utf-8

# Лабораторная работа №6. Применение сверточных нейронных сетей (многоклассовая классификация)
# ======
# Данные: Набор данных для распознавания языка жестов, который состоит из изображений размерности 28x28 в оттенках серого (значение пикселя от 0 до 255). Каждое из изображений обозначает букву латинского алфавита, обозначенную с помощью жеста, как показано на рисунке ниже (рисунок цветной, а изображения в наборе данных в оттенках серого). Обучающая выборка включает в себя 27,455 изображений, а контрольная выборка содержит 7172 изображения. Данные в виде csv-файлов можно скачать на сайте Kaggle -> https://www.kaggle.com/datamunge/sign-language-mnist
# 
# 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import copy
import tqdm
from PIL import Image

get_ipython().run_line_magic('matplotlib', 'inline')


# #### Загрузите данные. Разделите исходный набор данных на обучающую и валидационную выборки.

# In[2]:


train_images = pd.read_csv('../input/sign-language-mnist/sign_mnist_train.csv')
test_images = pd.read_csv('../input/sign-language-mnist/sign_mnist_test.csv')


# In[3]:


imgs = np.array(train_images.drop('label', axis=1).values)
plt.figure(figsize=(12,12))
for i in range(1,21):
    plt.subplot(4,5,i)
    plt.imshow(imgs[i].reshape(28,28))


# In[4]:


class Dataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.iloc[idx, 0]
        image = self.data.iloc[idx, 1:].values.reshape((28,28))

        if self.transforms:
            image = self.transforms(np.uint8(image))
        else:
            image = image.astype(np.float32)

        return image, label


# In[14]:


from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(train_images, test_size=0.2)

batch_size = 32
num_workers = 4

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = Dataset(train_data, transforms=transform)
train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)

val_dataset = Dataset(val_data, transforms=transform)
val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

test_dataset = Dataset(test_images, transforms=transform)
test_loader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)


# #### Реализуйте глубокую нейронную сеть со сверточными слоями.

# In[15]:


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(32 * 7 * 7, 25) 

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


# In[16]:


def validate(model, val_loader, criterion):         
    correct = 0
    total = 0
    total_loss = 0
    for images, labels in val_loader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    accuracy = correct.cpu().numpy() / total
    loss = total_loss / len(val_loader)
    return accuracy, loss

def train(model, train_loader, val_loader, num_epochs, optimizer, criterion, scheduler):
    train_acc = []
    val_acc = []
    for epoch in tqdm.tqdm(range(num_epochs)):
        total_loss = 0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum()

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct.cpu().numpy() / total
        accuracy, loss = validate(model, val_loader, criterion)
        # Print Loss
        print(f'[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.3f}; Val Loss: {loss:.4f}, Accuracy: {accuracy:.3f}')
        val_acc.append(accuracy)
        train_acc.append(train_accuracy)
    
    return train_acc, val_acc


# In[17]:


from torch.optim import lr_scheduler

model = CNNModel().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, amsgrad=True)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 30)
train_acc, val_acc = train(model, train_loader, val_loader, 5, optimizer, criterion, scheduler)


# In[18]:


accuracy, loss = validate(model, test_loader, criterion)
print(f'Test Loss: {loss:.4f}, Accuracy: {accuracy:.3f}')


# #### Примените дополнение данных (data augmentation).

# In[19]:


data_transform = transforms.Compose([
    transforms.ToPILImage(mode=None),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
    transforms.ToTensor(),
])

aug_dataset = Dataset(train_data, transforms = data_transform)
aug_loader = DataLoader(aug_dataset, batch_size = 32, shuffle=True, num_workers=4)


# In[20]:


train_acc, val_acc = train(model, aug_loader, val_loader, 5, optimizer, criterion, scheduler)


# In[21]:


accuracy, loss = validate(model, test_loader, criterion)
print(f'After augmentation Test Loss: {loss:.4f}, Accuracy: {accuracy:.3f}')


# #### Поэкспериментируйте с готовыми нейронными сетями (например, AlexNet, VGG16, Inception и т.п.), применив передаточное обучение.

# In[27]:


from torchvision import models

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 25)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, amsgrad=True)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 30)


# In[28]:


ds = ConcatDataset([train_dataset, aug_dataset])
resnet_loader = DataLoader(ds, batch_size = 32, shuffle=True, num_workers=4)


# In[ ]:


train_acc, val_acc = train(model, resnet_loader, val_loader, 5, optimizer, criterion, scheduler)


# In[ ]:


accuracy, loss = validate(model, test_loader, criterion)
print(f'ResNet18 Test Loss: {loss:.4f}, Accuracy: {accuracy:.3f}')


# ### Вывод:
# 
# В данной лабораторной мы использовали сверточную нейронную сеть для реализации многоклассового классификатора на наборе данных языка жестов. Использовали аугментацию данных и предаточное обучение.
