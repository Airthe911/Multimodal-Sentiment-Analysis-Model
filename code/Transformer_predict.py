import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms import ToTensor, Compose, Resize
from efficientnet_pytorch import EfficientNet
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = torch.relu(x)
#         x = self.fc2(x)
#         x = torch.relu(x)
#         x = self.fc3(x)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = Resize([224, 224])  # 统一图像尺寸为 224x224

input_size = 6  # 输入特征维度，即roberta_outputs和efficientnet_outputs的维度之和
hidden_size = 64  # 隐藏层维度
num_classes = 3  # 类别数量
sequence_length = 1
num_epochs = 25
num_layers = 3  # Transformer层数
num_heads = 3  # Transformer中多头注意力的头数
batch_size = 16
TF_model = TransformerModel(input_size, hidden_size, num_classes, num_layers, num_heads)
TF_model.load_state_dict(torch.load('../model/TF_model.pt'))
TF_model.to(device)

# 加载已经训练好的Roberta模型
config = RobertaConfig.from_pretrained('roberta-large')
config.num_labels = 3
roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-large', config=config).to(device)
roberta_model.load_state_dict(torch.load('../model/roberta-large.pt'))
roberta_model.eval()

# 加载已经训练好的EfficientNet模型
efficientnet_model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=3)
efficientnet_model.load_state_dict(torch.load('../model/efficientnet_model.pt'))
efficientnet_model.eval()


# 读取文本训练数据
data = pd.read_csv("../data/test_text_data.csv", encoding = "ISO-8859-1")
content = data["content"].tolist()
labels = data["label"].tolist()

# 读取图像训练数据
img_array = np.load("../data/test_img_data.npy", allow_pickle=True)

# 调整图像大小
img_array_resized = [transform(torch.from_numpy(image).permute(2, 0, 1)).float() for image in img_array]

# 创建图像数据集
imgs_dataset = CustomDataset(img_array_resized, labels)

# 创建图像数据加载器
imgs_dataloader = DataLoader(imgs_dataset, batch_size=batch_size, shuffle=True)

# 创建文本分词器并把文本向量化
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
content_vectors = tokenizer(content, max_length=128, padding = "max_length", truncation=True, return_tensors='pt')

# 将文本数据载入DataLoader
content_input_ids = torch.tensor(content_vectors["input_ids"]).to(device)
content_attention_mask = torch.tensor(content_vectors["attention_mask"]).to(device)
content_labels = torch.tensor(labels, dtype=torch.long).to(device)
content_dataset = TensorDataset(content_input_ids, content_attention_mask, content_labels)
content_dataloader = DataLoader(content_dataset, batch_size=batch_size, shuffle=True)

# 把模型放入GPU
roberta_model.to(device)
efficientnet_model.to(device)
TF_model.to(device)

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


def predict():
    val_content_labels = []
    val_img_labels = []
    val_fu_labels = []
    with torch.no_grad():
        for batch in zip(content_dataloader, imgs_dataloader):
          (input_ids, attention_mask, labels_text), (predict_images, labels) = batch
          input_ids = input_ids.to(device)
          attention_mask = attention_mask.to(device)
          labels_text = labels_text.to(device)
          predict_images = predict_images.to(device)

          # 获取roberta和EfficientNet的输出
          roberta_outputs = roberta_model(input_ids, attention_mask=attention_mask).logits
          efficientnet_outputs = efficientnet_model(predict_images)
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
    prediction = vote(val_content_labels, val_img_labels, val_fu_labels)
    print(len(prediction))
    return prediction
predictions = predict()
# 写入预测结果
guids = []
with open("../test_without_label.txt", "r") as f:
    line = f.readline()
    while 1:
        line = f.readline()
        result = line.strip().split(",")
        if len(result) == 2:
            guids.append(result[0])
        else:
            break
print(predictions)
with open("../test_with_label.txt", "w") as f:
  f.write("guid,tag\n")
  for i in range(len(guids)):
    if predictions[i] == 0:
      mood = "positive"
    elif predictions[i] == 1:
      mood = "neutral"
    elif predictions[i] == 2:
      mood = "negative"
    else:
      print("出现错误")
    f.write("{},{}\n".format(guids[i], mood))