# -*- coding: UTF-8 -*-
"""
@author: 周世聪
@contact: abner1zhou@gmail.com
@desc: 英文单词词性标注
@date:2020/6/6
"""
import numpy as np

DATA_PATH = "../data/traindata.txt"
# 初始化字典
tag2id, id2tag = {},{}
word2id, id2word = {},{}
with open(DATA_PATH, 'r') as f:
    for i in f.readlines():
        item = i.split('/')
        word = item[0]
        tag = item[1].rstrip()
        if word not in word2id:
            word2id[word] = len(word2id)
            id2word[len(id2word)] = word
        if tag not in tag2id:
            tag2id[tag] = len(tag2id)
            id2tag[len(id2tag)] = tag

# 初始化参数
num_words = len(word2id)
num_tags = len(tag2id)

pi = np.zeros(num_tags)
trans_ = np.zeros((num_tags, num_tags))  # 转移概率矩阵
launch_ = np.zeros((num_tags, num_words))  # 发射概率矩阵

# 计算概率
prev_tagID = None   # 前一个时间步的tagID
with open(DATA_PATH, 'r') as f:
    for line in f.readlines():
        item = line.split('/')
        wordID, tagID = word2id[item[0]], tag2id[item[1].rstrip()]
        # 句首
        if prev_tagID is None:
            pi[tagID] += 1
            launch_[tagID][wordID] += 1

        else:
            launch_[tagID][wordID] += 1
            trans_[prev_tagID][tagID] += 1
        if item[0] == '.':
            prev_tagID = None
        else:
            prev_tagID = tagID

# print(trans_[0])


# 归一化处理
pi = pi/sum(pi)
for i in range(num_tags):
    trans_[i] = trans_[i]/sum(trans_[i])
    launch_[i] /= sum(launch_[i])  # 简写


def log(v):
    if v == 0:
        return np.log(v+0.000001)
    return np.log(v)


# 维特比算法，动态规划
def vitervi(x, pi=pi, trans=trans_, launch=launch_):
    # 动态规划分数矩阵, 初始化让它足够的小
    words = x.split(" ")
    wordID = [word2id[word] for word in words]
    dy = np.ones([len(words), num_tags]) * -99999
    for j in range(num_tags):
        dy[0][j] = log(pi[j]) + log(launch_[j][wordID[0]])
    ptr = np.zeros([len(words), num_tags], dtype=int)
    for i in range(1, len(words)):
        for j in range(num_tags):
            for k in range(num_tags):
                score = dy[i-1][k] + log(trans[k][j]) + log(launch_[j][wordID[i]])
                if score > dy[i][j]:
                    dy[i][j] = score
                    ptr[i][j] = k  # 保存最大得分路径的下标(上一个时间步的tagID)
    # 输出结果
    best_path = np.zeros(len(words), dtype=int)
    best_path[-1] = np.argmax(dy[-1])
    # range 取值范围 [x, y) 前闭后开 不包含y点
    for i in range(len(words)-2, -1, -1):
        best_path[i] = ptr[i+1][best_path[i+1]]
    print(x)
    for path in best_path:
        print(id2tag[path])


if __name__ == '__main__':
    sentence = "I like to play computer games"
    vitervi(sentence)
