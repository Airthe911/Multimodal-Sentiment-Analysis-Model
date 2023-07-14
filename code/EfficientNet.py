import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# 加载图像数据和标签
img_array = np.load("../data/train_img_data.npy", allow_pickle=True)
label_array = np.load("../data/label_data.npy", allow_pickle=True)

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

# 数据预处理和转换

transform = Resize([224, 224])  # 统一图像尺寸为 224*224

# 调整图像大小
print(np.array(img_array[0]).shape)
img_array_resized = [transform(torch.from_numpy(image).permute(2, 0, 1)).float() for image in img_array]
print(np.array(img_array_resized[0]).shape)
# 创建数据集
train_imgs, valid_imgs, train_labels, valid_labels = train_test_split(img_array_resized, label_array, test_size=0.2,random_state = 911)
train_dataset = CustomDataset(train_imgs, train_labels)
valid_dataset = CustomDataset(valid_imgs, valid_labels)

# 创建数据加载器
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

# 创建模型
model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=3)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# 训练模型
num_epochs = 10
def valid():
    val_true_labels = []
    val_predicted_labels = []
    with torch.no_grad():
         for val_images, val_labels in valid_dataloader:
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            val_outputs = model(val_images)
            val_predicted_labels.extend(torch.argmax(val_outputs, dim=1).cpu().tolist())
            val_true_labels.extend(val_labels.cpu().tolist())

    acc = accuracy_score(val_true_labels, val_predicted_labels)
    f1 = f1_score(val_true_labels, val_predicted_labels, average='macro')
    print("Accuracy = {}, ".format(acc), end="")
    print("F1 score = {}".format(f1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    print("Epoch {} training finished: Loss: {}, ".format(epoch + 1, epoch_loss), end = "")
    model.eval()
    valid()
print("Traning Completed")
torch.save(model.state_dict(), '../model/efficientnet_model.pt')