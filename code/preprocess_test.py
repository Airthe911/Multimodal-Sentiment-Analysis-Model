import pandas as pd
import numpy as np
from PIL import Image

guids = []
labels = []
text_data = pd.DataFrame()
test_guids = []
test_texts = []
test_imgs = []

with open("../test_without_label.txt", "r") as f:
    line = f.readline()
    while 1:
        line = f.readline()
        result = line.strip().split(",")
        if len(result) == 2:
            guids.append(result[0])
            labels.append(-1)
        else:
            break

for guid in guids:
    with open("../data/{}.txt".format(guid), encoding="ISO-8859-1") as f:
        test_guids.append(guid)
        test_texts.append(f.readline().strip())
    img = Image.open("../data/{}.jpg".format(guid))
    img_cv = np.array(img)  # 转换为NumPy数组
    test_imgs.append(img_cv)

text_data["guid"] = test_guids
text_data["content"] = test_texts
text_data["label"] = labels
text_data.to_csv("../data/test_text_data.csv", index=False, encoding="ISO-8859-1")

# 将图像数据转换为NumPy数组
img_array = np.array(test_imgs)

# 保存图像数据为NumPy数组
np.save("../data/test_img_data.npy", img_array)