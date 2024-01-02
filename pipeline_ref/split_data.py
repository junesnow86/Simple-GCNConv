import json

from sklearn.model_selection import train_test_split

data = []
# 读取json文件
with open('data/yelp_academic_dataset_review.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))
# 获取除了前1000条数据外的其他数据
data = data[1000:]
# 提取'text'和'stars'字段
data = [{'text': d['text'], 'stars': d['stars']} for d in data]
# 划分训练集和验证集
train_data, val_data = train_test_split(data, test_size=0.1)
# 将训练集写入到新的JSON文件
with open('data/train_data.json', 'w') as f:
    json.dump(train_data, f)
# 将验证集写入到新的JSON文件
with open('data/val_data.json', 'w') as f:
    json.dump(val_data, f)

data = []
with open('data/test.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))
data = [{'text': d['text'], 'stars': d['stars']} for d in data]
with open('data/test_data.json', 'w') as f:
    json.dump(data, f)
