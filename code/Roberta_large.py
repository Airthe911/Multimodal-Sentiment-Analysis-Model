from transformers import BertForSequenceClassification, BertTokenizer,BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = RobertaConfig.from_pretrained('roberta-large')
config.num_labels = 3
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
base_model = RobertaForSequenceClassification.from_pretrained('roberta-large', config=config).to(device)
#读取训练数据
data = pd.read_csv("../data/train_text_data.csv", encoding = "ISO-8859-1")
content = data["content"].tolist()
labels = data["label"].tolist()

#划分训练集和验证集
train_content, valid_content, train_labels,valid_labels = train_test_split(content,labels, test_size=0.2,random_state = 911)
train_vectors = tokenizer(train_content, max_length=128, padding="max_length", truncation=True, return_tensors='pt')
valid_vectors = tokenizer(valid_content, max_length=128, padding="max_length", truncation=True, return_tensors='pt')

#将训练数据，验证数据载入DataLoader

train_input_ids = train_vectors['input_ids'].to(device)
train_attention_mask = train_vectors['attention_mask'].to(device)
train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)

valid_input_ids = valid_vectors['input_ids'].to(device)
valid_attention_mask = valid_vectors['attention_mask'].to(device)
valid_labels = torch.tensor(valid_labels, dtype=torch.long).to(device)
valid_dataset = TensorDataset(valid_input_ids, valid_attention_mask, valid_labels)

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

#定义一个函数，用于测试当前模型在验证集上的效果
def valid():
    times = 10
    i = 0
    acc = 0
    f1 = 0
    with torch.no_grad():
        for batch in valid_dataloader:
            if i < times:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                outputs = base_model(input_ids, attention_mask=attention_mask)
                outputs = outputs.logits
                preds = np.array(outputs.argmax(axis=1).detach().cpu())
                labels_true = np.array(labels.cpu())
                acc += accuracy_score(labels_true, preds)
                f1 += f1_score(labels_true, preds, average='macro')
                i += 1
            else:
                print("Accuracy = {}, ".format(acc/times), end="")
                print("F1 score = {}".format(f1/times))
                break

#BERT模型训练
optimizer = AdamW(base_model.parameters(), lr=8e-7, eps=1e-8, weight_decay=0.001)
criterion = torch.nn.CrossEntropyLoss()
base_model.train()
epochs = 50
for epoch in range(epochs):
    for batch in train_dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch {} training finished: Loss: {}, ".format(epoch + 1, loss.item()), end="")
    base_model.eval()
    valid()
    base_model.train()
print("Traning Completed")
torch.save(base_model.state_dict(), '../model/roberta_model.pt')