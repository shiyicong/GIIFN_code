import numpy as np
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

'''基础模型'''


class MyModel(nn.Module):
    def __init__(self, img_dim=768, text_dim=768, num_mlp_layers=3, hidden_dim=4096, num_classes=2, dropout_prob=0.2):
        super(MyModel, self).__init__()

        self.img_dim = img_dim
        self.text_dim = text_dim

        self.num_mlp_layers = num_mlp_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        # Input layer
        self.fc_in = nn.Linear(768, hidden_dim)
        # 单模态输入

        # self.fc_in = nn.Linear(text_dim, hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_mlp_layers)
        ])

        # Output layer
        self.bottle_neck = nn.Linear(hidden_dim, 64)
        self.fc_out = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

        # 添加平均池化层
        # self.avgpool = nn.AvgPool1d(kernel_size=3, stride=3)

        # 原子模态一致性分数

    def forward(self, img_feat, text_feat, kn_feat):
        # avg_pool
        # img_feat = img_feat.view(img_feat.size(0), 1, img_feat.size(1))
        # img_feat = self.avgpool(img_feat)
        # img_feat = img_feat.view(img_feat.size(0), -1)
        # # print(img_feat.shape)
        # text_feat = text_feat.view(text_feat.size(0), 1, text_feat.size(1))
        # text_feat = self.avgpool(text_feat)
        # text_feat = text_feat.view(text_feat.size(0), -1)
        # print(text_feat.shape)
        # Concatenate features

        x = torch.cat((img_feat, text_feat), dim=1)
        # print(x.shape)
        # x = (text_feat+kn_feat) / 2
        # cos_sim = torch.cosine_similarity(text_feat, kn_feat, dim=1)
        # cos_sim = cos_sim.unsqueeze(1)
        # x = cos_sim*kn_feat + text_feat
        # x = torch.cat((text_feat, cos_sim*kn_feat), dim=1)
        # x = torch.cat((img_feat, x), dim=1)
        # x = kn_feat

        # exit()

        # Input layer
        x = self.fc_in(x)
        x = self.relu(x)
        x = self.dropout(x)
        # 降维

        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)

        # Output layer
        x = self.bottle_neck(x)
        x = self.fc_out(x)

        return x


'''多头交叉注意力'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_heads=4, d_model=64):
        super(MultiHeadCrossAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model * 2, d_model)
        # self.dim_ensional = nn.Linear(768, 1024)

    def forward(self, image, text):
        # Compute queries, keys, and values for text and image

        q_t = self.w_q(text)
        k_t = self.w_k(text)
        v_t = self.w_v(text)
        q_v = self.w_q(image)
        k_v = self.w_k(image)
        v_v = self.w_v(image)

        # Split queries, keys, and values into multiple heads
        q_t = q_t.view(q_t.size(0), -1, self.n_heads, self.d_k)
        k_t = k_t.view(k_t.size(0), -1, self.n_heads, self.d_k)
        v_t = v_t.view(v_t.size(0), -1, self.n_heads, self.d_k)
        q_v = q_v.view(q_v.size(0), -1, self.n_heads, self.d_k)
        k_v = k_v.view(k_v.size(0), -1, self.n_heads, self.d_k)
        v_v = v_v.view(v_v.size(0), -1, self.n_heads, self.d_k)

        # Compute attention scores for text and image
        scores_t = torch.matmul(q_t, k_v.transpose(-2, -1))
        scores_v = torch.matmul(q_v, k_t.transpose(-2, -1))

        # 归一化
        scores_t = scores_t / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        scores_v = scores_v / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # softmax to compute attention weights for text and image
        weights_t = F.softmax(scores_t, dim=-1)
        weights_v = F.softmax(scores_v, dim=-1)

        # Apply attention weights to values to compute context vectors for text and image
        contexts_t = torch.matmul(weights_t, v_v)
        contexts_v = torch.matmul(weights_v, v_t)

        # Concatenate context vectors from text and image
        contexts = torch.cat((contexts_t, contexts_v), dim=1)

        # Apply a fully connected layer to compute the final output
        # print((contexts.view(contexts.size(0), -1)).shape)
        # 加入残差
        # output = self.output(contexts.view(contexts.size(0), -1)) + text + image
        # output.shape : torch.Size([32, 768])
        output = self.output(contexts.view(contexts.size(0), -1)) + text
        # print('1',output.shape)
        # exit()
        return output


# 使用文本作为Q，图片作为KV
class MultiModalAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super(MultiModalAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 初始化query、key和value张量的线性变换
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 初始化形状变换函数
        self.dim_ensional = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

    def forward(self, text, image):
        # 通过线性变换得到文本的query张量
        q_t = self.w_q(text)

        # 通过形状变换得到图片的key和value张量
        k_v = self.dim_ensional(image)
        k_t = self.w_k(k_v)
        v_t = self.w_v(k_v)

        # Split queries into multiple heads
        q_t = q_t.view(q_t.size(0), -1, self.n_heads, self.d_k)

        # Split keys and values into multiple heads
        k_t = k_t.view(k_t.size(0), -1, self.n_heads, self.d_k)
        v_t = v_t.view(v_t.size(0), -1, self.n_heads, self.d_k)

        # Compute attention scores for text and image
        scores = torch.matmul(q_t, k_t.transpose(-2, -1))

        # 归一化
        scores = scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # softmax to compute attention weights for text and image
        weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values to compute context vectors
        contexts = torch.matmul(weights, v_t)

        # Concatenate context vectors from text and image
        contexts = contexts.view(contexts.size(0), -1)

        return contexts


class TrainableWeightedConcat(nn.Module):
    def __init__(self, input_dim=2, n_weights=2):
        super(TrainableWeightedConcat, self).__init__()
        self.n_weights = n_weights
        self.weights = nn.Parameter(torch.ones(n_weights), requires_grad=True)
        self.fc = nn.Linear(input_dim * n_weights, n_weights, bias=False)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        weights = F.softmax(self.fc(x), dim=1)

        # weights = F.sigmoid(self.fc(x))
        # weights = self.fc(x)

        weighted_tensors = [w.unsqueeze(1) * tensor for w, tensor in zip(weights.unbind(dim=1), (x1, x2))][::-1]

        return torch.cat(weighted_tensors, dim=1)


# W1X1+W2X2 的加权求和
class TrainableWeightedSum(nn.Module):
    def __init__(self, input_dim=4, n_weights=2):
        super(TrainableWeightedSum, self).__init__()
        self.n_weights = n_weights
        self.weights = nn.Parameter(torch.ones(n_weights), requires_grad=True)
        self.fc = nn.Linear(input_dim, n_weights, bias=False)

    def forward(self, x1, x2):
        weights = F.softmax(self.fc(torch.cat([x1, x2], dim=1)), dim=1)
        print(self.weights.shape)
        exit()
        x1_weighted = weights[:, 0].unsqueeze(-1) * x1
        x2_weighted = weights[:, 1].unsqueeze(-1) * x2

        weighted_sum = x1_weighted + x2_weighted

        return weighted_sum


# 使用多个分类器，对分类器的结进行融合
class classifier(nn.Module):
    def __init__(self, num_mlp_layers=3, input_dim=1024, num_classes=2, hidden_dim=4096, dropout_prob=0.2):
        super(classifier, self).__init__()
        self.dropout_prob = dropout_prob
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        # Input layer
        self.fc_in = nn.Linear(input_dim, hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_mlp_layers)
        ])

        self.fc_out = nn.Linear(hidden_dim, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        # Input layer
        x = self.fc_in(x)
        x = self.relu(x)

        x = self.dropout(x)

        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)

        # Output layer线性分类

        x = self.fc_out(x)
        return x


# 双线性池化融合
class BilinearPooling(torch.nn.Module):
    def __init__(self, input_dim=1024, output_dim=2048):
        super(BilinearPooling, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x, y):
        batch_size = x.size(0)
        x = self.linear(x)  # x shape: batch_size x output_dim
        y = self.linear(y)  # y shape: batch_size x output_dim
        z = torch.bmm(x.unsqueeze(2), y.unsqueeze(1))  # z shape: batch_size x output_dim x output_dim
        print(z.shape)
        z = z.view(batch_size, -1)  # flatten the tensor
        print(z.shape)
        z = F.normalize(z, p=2, dim=1)  # apply L2 normalization
        return z


class SmilarityFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SmilarityFusion, self).__init__()
        self.fc1_img = nn.Linear(input_dim, hidden_dim)
        self.fc1_txt = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        def forward(self, x1, x2):
            x1 = self.fc1_img(x1)
            x2 = self.fc1_txt(x2)

            alpha = torch.sigmoid(torch.dot(x1, x2) / (torch.norm(x1) * torch.norm(x2)))
            fused_feat = alpha * x1 + (1 - alpha) * x2

            output = self.fc2(fused_feat)
            return output


# 加权投票器
class WeightedVotingClassifier:
    def __init__(self, weight1, weight2):
        self.weight1 = weight1
        self.weight2 = weight2

    def fit(self, predictions1, predictions2):
        # 确保输入的长度相等
        assert len(predictions1) == len(predictions2)
        self.predictions1 = predictions1
        self.predictions2 = predictions2

    def predict(self):
        # 初始化加权结果列表
        weighted_predictions = []

        # 对每个样本进行加权投票
        for pred1, pred2 in zip(self.predictions1, self.predictions2):
            # 计算加权得分
            weighted_score = self.weight1 * pred1 + self.weight2 * pred2

            # 根据得分进行二分类判断
            if weighted_score >= 0.5:
                weighted_predictions.append(1)  # 正类
            else:
                weighted_predictions.append(0)  # 负类

        return weighted_predictions

