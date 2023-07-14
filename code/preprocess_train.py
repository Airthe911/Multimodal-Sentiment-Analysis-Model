import pandas as pd
import numpy as np
from PIL import Image

guids = []
labels = []
text_data = pd.DataFrame()
train_guids = []
train_texts = []
train_imgs = []

with open("../train.txt", "r") as f:
    line = f.readline()
    while 1:
        line = f.readline()
        result = line.strip().split(",")
        if len(result) == 2:
            guids.append(result[0])
            if result[1] == "positive":
                labels.append(0)
            elif result[1] == "neutral":
                labels.append(1)
            elif result[1] == "negative":
                labels.append(2)
            else:
                print("出现错误")
        else:
            break

for guid in guids:
    with open("../data/{}.txt".format(guid), encoding="ISO-8859-1") as f:
        train_guids.append(guid)
        train_texts.append(f.readline().strip())
    img = Image.open("../data/{}.jpg".format(guid))
    img_cv = np.array(img)  # 转换为NumPy数组
    train_imgs.append(img_cv)

text_data["guid"] = train_guids
text_data["content"] = train_texts
text_data["label"] = labels
text_data.to_csv("../data/train_text_data.csv", index=False, encoding="ISO-8859-1")

# 将图像数据转换为NumPy数组
img_array = np.array(train_imgs)

# 保存图像数据为NumPy数组
np.save("../data/train_img_data.npy", img_array)
label_array = np.array(labels)
np.save("../data/label_data", label_array)