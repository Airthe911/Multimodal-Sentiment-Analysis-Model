import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms import ToTensor, Compose, Resize
from efficientnet_pytorch import EfficientNet
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 自定义图片数据集类
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

transform = Resize([224, 224])  # 统一图像尺寸为 224*224
# 加载已经训练好的Roberta模型
config = RobertaConfig.from_pretrained('roberta-large')
config.num_labels = 3
roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-large', config=config).to(device)
roberta_model.load_state_dict(torch.load('../model/roberta-large.pt'))

# 加载已经训练好的EfficientNet模型
efficientnet_model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=3)
efficientnet_model.load_state_dict(torch.load('../model/efficientnet_model.pt'))

# 读取文本训练数据
data = pd.read_csv("../data/train_text_data.csv", encoding = "ISO-8859-1")
content = data["content"].tolist()
labels = data["label"].tolist()

# 读取图像训练数据
img_array = np.load("../data/train_img_data.npy", allow_pickle=True)
label_array = np.load("../data/label_data.npy", allow_pickle=True)

# 调整图像大小
img_array_resized = [transform(torch.from_numpy(image).permute(2, 0, 1)).float() for image in img_array]

# 训练与验证数据划分
train_content, valid_content, train_labels, valid_labels = train_test_split(content, labels, test_size=0.2, random_state = 911)
train_imgs, valid_imgs, train_labels, valid_labels = train_test_split(img_array_resized, labels, test_size=0.2, random_state = 911)

roberta_model.eval()
efficientnet_model.eval()
batch_size = 16

# 创建图像数据集
imgs_train_dataset = CustomDataset(train_imgs, train_labels)
imgs_valid_dataset = CustomDataset(valid_imgs, valid_labels)

# 创建图像数据加载器
imgs_train_dataloader = DataLoader(imgs_train_dataset, batch_size=batch_size, shuffle=True)
imgs_valid_dataloader = DataLoader(imgs_valid_dataset, batch_size=batch_size, shuffle=True)

# 创建文本分词器并把文本向量化
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
content_train_vectors = tokenizer(train_content, max_length=128, padding = "max_length", truncation=True, return_tensors='pt')
content_valid_vectors = tokenizer(valid_content, max_length=128, padding = "max_length", truncation=True, return_tensors='pt')

# 将文本训练数据，验证数据载入DataLoader
content_train_input_ids = torch.tensor(content_train_vectors["input_ids"]).to(device)
content_train_attention_mask = torch.tensor(content_train_vectors["attention_mask"]).to(device)
content_train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
content_train_dataset = TensorDataset(content_train_input_ids, content_train_attention_mask, content_train_labels)
content_valid_input_ids = torch.tensor(content_valid_vectors["input_ids"]).to(device)
content_valid_attention_mask = torch.tensor(content_valid_vectors["attention_mask"]).to(device)
content_valid_labels = torch.tensor(valid_labels, dtype=torch.long).to(device)
content_valid_dataset = TensorDataset(content_valid_input_ids, content_valid_attention_mask, content_valid_labels)
content_train_dataloader = DataLoader(content_train_dataset, batch_size=batch_size, shuffle=True)
content_valid_dataloader = DataLoader(content_valid_dataset, batch_size=batch_size, shuffle=True)

# 把模型放入GPU
roberta_model.to(device)
efficientnet_model.to(device)


# 定义MLP模型
# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, int(hidden_size / 2))
#         self.fc4 = nn.Linear(int(hidden_size / 2), num_classes)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = torch.relu(x)
#         x = self.fc2(x)
#         x = torch.relu(x)
#         x = self.fc3(x)
#         x = torch.relu(x)
#         x = self.fc4(x)
#         return x
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, num_heads):
        super(TransformerModel, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=input_size, nhead=num_heads)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)  # 对序列维度求平均
        x = self.fc(x)
        return x


def vote(val_content_labels, val_img_labels, val_fu_labels):
  val_predicted_labels = []
  for i in range(len(val_fu_labels)):
    voting_list = [0, 0, 0]
    voting_list[int(val_content_labels[i])] = voting_list[int(val_content_labels[i])] + 1
    voting_list[int(val_img_labels[i])] = voting_list[int(val_img_labels[i])] + 1
    voting_list[int(val_fu_labels[i])] = voting_list[int(val_fu_labels[i])] + 1
    flag = 0
    for j in range(3):
      if voting_list[j] >= 2:
        val_predicted_labels.append(j)
        flag = 1
    if flag == 0:
      val_predicted_labels.append(int(val_content_labels[i]))  # 各一票，则使用语言模型的输出
  return val_predicted_labels


def valid():
    val_true_labels = []
    val_content_labels = []
    val_img_labels = []
    val_fu_labels = []
    with torch.no_grad():
        for batch in zip(content_valid_dataloader, imgs_valid_dataloader):
          (input_ids, attention_mask, labels_text), (valid_images, valid_labels) = batch
          input_ids = input_ids.to(device)
          attention_mask = attention_mask.to(device)
          labels_text = labels_text.to(device)
          valid_images = valid_images.to(device)
          valid_labels = valid_labels.to(device)

          # 获取roberta和EfficientNet的输出
          roberta_outputs = roberta_model(input_ids, attention_mask=attention_mask).logits
          efficientnet_outputs = efficientnet_model(valid_images)
          val_content_labels.extend(torch.argmax(roberta_outputs, dim=1).cpu().tolist())
          val_img_labels.extend(torch.argmax(efficientnet_outputs, dim=1).cpu().tolist())

          # 分别展平roberta和EfficientNet的输出
          roberta_outputs = roberta_outputs.view(roberta_outputs.size(0), -1)
          efficientnet_outputs = efficientnet_outputs.view(efficientnet_outputs.size(0), -1)



          # 将roberta和EfficientNet的输出拼接为Transformer的输入
          TF_input = torch.cat((roberta_outputs, efficientnet_outputs), dim=1)
          TF_input = TF_input.unsqueeze(0)   # 在维度0上插入一个维度
          TF_input = TF_input.repeat(sequence_length, 1, 1)   # 重复输入数据sequence_length次
          TF_input = TF_input.permute(1, 0, 2)
          # 正向传播和反向传播
          TF_outputs = TF_model(TF_input)
          val_fu_labels.extend(torch.argmax(TF_outputs, dim=1).cpu().tolist())
          val_true_labels.extend(labels_text.cpu().tolist())

    # 投票结果
    val_predicted_labels = vote(val_content_labels, val_img_labels, val_fu_labels)
    acc = accuracy_score(val_true_labels, val_predicted_labels)
    f1 = f1_score(val_true_labels, val_predicted_labels, average='macro')
    print("\nVoting result:", end=" ")
    print("Accuracy = {}, ".format(acc), end="")
    print("F1 score = {}".format(f1))

    # 语言模型结果
    val_predicted_labels = val_content_labels
    acc = accuracy_score(val_true_labels, val_predicted_labels)
    f1 = f1_score(val_true_labels, val_predicted_labels, average='macro')
    print("\nText result:", end=" ")
    print("Accuracy = {}, ".format(acc), end="")
    print("F1 score = {}".format(f1))

    #图像模型结果
    val_predicted_labels = val_img_labels
    acc = accuracy_score(val_true_labels, val_predicted_labels)
    f1 = f1_score(val_true_labels, val_predicted_labels, average='macro')
    print("\nImg result:", end=" ")
    print("Accuracy = {}, ".format(acc), end="")
    print("F1 score = {}".format(f1))

    #融合模型结果
    val_predicted_labels = val_fu_labels
    acc = accuracy_score(val_true_labels, val_predicted_labels)
    f1 = f1_score(val_true_labels, val_predicted_labels, average='macro')
    print("\nFuse result:", end=" ")
    print("Accuracy = {}, ".format(acc), end="")
    print("F1 score = {}".format(f1))


# # 定义MLP模型的输入层大小、隐藏层大小和输出层大小
# input_size = 3 + 3  # roberta输出和EfficientNet输出的展平连接
# hidden_size = 128
# num_classes = 3
# num_epochs = 25
# # 创建MLP模型实例
# mlp_model = MLP(input_size, hidden_size, num_classes)
# mlp_model.to(device)

input_size = 6  # 输入特征维度，即roberta_outputs和efficientnet_outputs的维度之和
hidden_size = 64  # 隐藏层维度
num_classes = 3  # 类别数量
sequence_length = 1
num_epochs = 10
num_layers = 3  # Transformer层数
num_heads = 3  # Transformer中多头注意力的头数
TF_model = TransformerModel(input_size, hidden_size, num_classes, num_layers, num_heads)
TF_model.to(device)


# 定义MLP模型的损失函数和优化器
TF_criterion = nn.CrossEntropyLoss()
TF_optimizer = torch.optim.Adam(TF_model.parameters(), lr=0.00001)

# 模型训练
max_acc = 0.71
for epoch in range(num_epochs):
    TF_model.train()
    running_loss = 0.0
    for batch in zip(content_train_dataloader, imgs_train_dataloader):
        (input_ids, attention_mask, labels_text), (train_images, train_labels) = batch

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels_text = labels_text.to(device)
        train_images = train_images.to(device)
        train_labels = train_labels.to(device)

        # 获取roberta和EfficientNet的输出
        roberta_outputs = roberta_model(input_ids, attention_mask=attention_mask).logits
        efficientnet_outputs = efficientnet_model(train_images)

        # 分别展平roberta和EfficientNet的输出
        roberta_outputs = roberta_outputs.view(roberta_outputs.size(0), -1)
        efficientnet_outputs = efficientnet_outputs.view(efficientnet_outputs.size(0), -1)

        # 将roberta和EfficientNet的输出拼接为Transformer的输入
        TF_input = torch.cat((roberta_outputs, efficientnet_outputs), dim=1)
        TF_input = TF_input.unsqueeze(0)   # 在维度0上插入一个维度
        TF_input = TF_input.repeat(sequence_length, 1, 1)   # 重复输入数据sequence_length次
        TF_input = TF_input.permute(1, 0, 2)

        # 正向传播和反向传播
        TF_outputs = TF_model(TF_input)
        loss = TF_criterion(TF_outputs, labels_text)
        TF_optimizer.zero_grad()
        loss.backward()
        TF_optimizer.step()

        running_loss += loss.item()
    epoch_loss = running_loss / len(content_train_dataloader)
    print("Epoch {} training finished: Loss: {}, ".format(epoch + 1, epoch_loss), end="")
    TF_model.eval()
    valid()
torch.save(TF_model.state_dict(), '../model/TF_model.pt')