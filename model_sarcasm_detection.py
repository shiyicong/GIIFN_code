import random

import numpy as np
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
# 我自己的模型库
import models

#归一化
def normalize_vector(x):
    norm = torch.norm(x)  # 计算L2范数
    x_norm = x / norm  # 归一化
    return x_norm

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 打印模型参数数量


# bilinear = nn.Bilinear(input_dim1, input_dim2, output_dim)


# hidden_dim*4
class MyModel(nn.Module):
    # hiddendim减小一半
    def __init__(self, img_dim=1024, text_dim=1024, num_mlp_layers=3, hidden_dim=4096, num_classes=2, dropout_prob=0.2):
        super(MyModel, self).__init__()

        self.img_dim = img_dim
        self.text_dim = text_dim

        # self.num_mlp_layers = num_mlp_layers
        # self.hidden_dim = hidden_dim
        # self.num_classes = num_classes
        # self.dropout_prob = dropout_prob
        # # Input layer
        # self.fc_in = nn.Linear(1024 * 2, hidden_dim)
        # # 单模态输入
        #
        # # self.fc_in = nn.Linear(text_dim, hidden_dim)
        #
        # # Hidden layers
        # self.hidden_layers = nn.ModuleList([
        #     nn.Linear(hidden_dim, hidden_dim) for _ in range(num_mlp_layers)
        # ])
        #
        # # Output layer
        # # self.bottle_neck = nn.Linear(hidden_dim, 64)
        # self.fc_out = nn.Linear(hidden_dim, num_classes)
        #
        # self.relu = nn.ReLU()
        # # self.tanh = nn.Tanh()
        # self.dropout = nn.Dropout(p=dropout_prob)

        # 升维
        self.dim_ensional = nn.Linear(768, 1024)
        # 交叉多头注意力
        self.multiheadcrossattention = models.MultiHeadCrossAttention()
        # 使用文本作为Q，图片作为KV
        self.multimodelattention = models.MultiModalAttention(d_model=64, n_heads=4)
        self.multimodelattention_f = models.MultiModalAttention(d_model=64, n_heads=4)
        self.multimodelattention_kn = models.MultiModalAttention(d_model=64, n_heads=4)
        # 可训练权重层
        self.trainable_cat = models.TrainableWeightedConcat()
        # 加权求和层
        self.trainable_sum = models.TrainableWeightedSum()
        # late fusion
        self.classifier_v = models.classifier(num_mlp_layers=3, input_dim=1024, num_classes=2, hidden_dim=4096,
                                              dropout_prob=0.2)
        self.classifier_l = models.classifier(num_mlp_layers=3, input_dim=1024, num_classes=2, hidden_dim=4096,
                                              dropout_prob=0.2)
        # BilinearPooling
        self.bilinearpooling = models.BilinearPooling()
        # 分类器
        self.classifier = models.classifier(num_mlp_layers=3, input_dim=1024, num_classes=5, hidden_dim=4096,
                                            dropout_prob=0.2)

    def forward(self, img_feat, surf_feat):
        # print(img_feat.shape)
        img_feat = self.dim_ensional(img_feat)

        # exit()
        img_featsplit = torch.split(img_feat, 64, dim=1)
        scores_v = []

        for i in range(16):
            # score = self.multiheadcrossattention(img_featsplit[i], surf_feat) + img_featsplit[i]
            score = normalize_vector(self.multimodelattention(img_featsplit[i], surf_feat) + img_featsplit[i])
            # score = self.multimodelattention(surf_feat, img_featsplit[i]) + img_featsplit[i]
            # score = torch.cat((img_featsplit[i], surf_feat),dim = 1)
            # 对知识的融合
            # score = self.multiheadcrossattention(img_featsplit[i], kn_featsplit[i]) + img_featsplit[i]
            # score = self.multimodelattention(img_featsplit[i], kn_featsplit[i]) + img_featsplit[i]
            # score = self.multimodelattention(kn_featsplit[i], img_featsplit[i]) + img_featsplit[i]
            # score = torch.cat((img_featsplit[i], kn_featsplit[i]), dim=1)1*64
            scores_v.append(score)

        x_v = torch.cat(scores_v, dim=1)
        # print(x_v.shape)
        x = img_feat
        # x = x_v

        x = self.classifier(x)

        # exit()
        return x


# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    for img_tensor, surf_tensor,  labels in train_loader:
        img_tensor = img_tensor.to(device)

        surf_tensor = surf_tensor.to(device)

        labels = labels.to(device)
        optimizer.zero_grad()
        # 这里的传参顺序决定传入的是啥
        outputs = model(img_tensor,surf_tensor, )

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * img_tensor.size(0)
        _, preds = torch.max(outputs, 1)
        train_acc += torch.sum(preds == labels.data)
    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)
    return train_loss, train_acc


'''加入其他指标'''



def Test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    with torch.no_grad():
        for img_tensor, surf_tensor,  labels in test_loader:
            img_tensor = img_tensor.to(device)
            surf_tensor = surf_tensor.to(device)

            labels = labels.to(device)

            outputs = model(img_tensor, surf_tensor)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * img_tensor.size(0)
            _, preds = torch.max(outputs, 1)
            test_acc += torch.sum(preds == labels.data)
            true_positives += ((preds == 1) & (labels.data == 1)).sum().item()
            false_positives += ((preds == 1) & (labels.data == 0)).sum().item()
            false_negatives += ((preds == 0) & (labels.data == 1)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return test_loss, test_acc, precision, recall, f1_score


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, json_path):
        self.df = pd.read_json(json_path)
        self.labels = self.df['label'].values
        self.img_features = self.df['vit_feat'].values
        self.surf_features = self.df['surf_feat'].values

    def __getitem__(self, index):
        label = torch.tensor(self.labels[index], dtype=torch.long)
        # 数据集读的有问题，暂时这样吧，img_feature=arr[arr[lsit0],arr[list1],...,arr[list=len(train_data)]]
        # 读取img特征
        # for element in self.img_features[index]:
        #     element = element
        img_features = torch.tensor(self.img_features[index], dtype=torch.float)
        # print(img_features.shape)
        # exit()
        # 读取SURF特征
        surf_features = torch.tensor(self.surf_features[index], dtype=torch.float)

        surf_features = surf_features.view(64)
        # return img_features, text_features, kn_features, surf_features, hcf_features, label
        return img_features, surf_features,label

    def __len__(self):
        return len(self.df)


seed = 42


# 设置随机种子
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    # def experiment():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 定义超参数

    lr = 0.001
    num_epochs = 50
    # 调大一点试试 100-200
    batch_size = 256
    # set_random_seed(seed)
    # 定义模型、优化器和损失函数
    model = MyModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # 创建学习率调度器，每 10 个 epoch 将学习率缩小 0.1
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    trainpath = r'D:\Cong\code1\GIIFN\train.json'
    testpath = r'D:\Cong\code1\GIIFN\test.json'

    train_dataset = MyDataset(trainpath)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MyDataset(testpath)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # best_model_path = r'D:\cong\m_paper\code\model_pth\baseline\carcasm_dete01.pth'
    # 训练模型
    best_train_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer, device)
        # print(
        #     'Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f},'.format(epoch + 1, num_epochs, train_loss,
        #                                                                    train_acc))
        print(
            'Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Learning Rate: {:.10f}'.format(epoch + 1, num_epochs,
                                                                                                  train_loss, train_acc,
                                                                                                  optimizer.param_groups[
                                                                                                      0]['lr']))
        scheduler.step()
        # 跑一个epoch直接测试{
        # torch.save(model.state_dict(), best_model_path)
        # test_model = MyModel().to(device)
        # test_model.load_state_dict(torch.load(best_model_path))
        # test_model = MyModel().to(device)
        #
        # test_loss, test_acc, recall, f1_score = Test(test_model, test_dataloader, criterion, device)
        # print('Test Loss: {:.4f}, Test Acc: {:.4f},Recall: {:.4f},F1_score: {:.4f}'.format(test_loss, test_acc, recall,
        #                                                                                    f1_score))
        #                                                                                    }
        # 保存最好模型的参数
    #     if train_acc > best_train_acc:
    #         best_train_acc = train_acc
    #         # print('best_trainacc= ', best_train_acc)
    #         torch.save(model.state_dict(), best_model_path)
    #
    # # 测试模型
    # best_model = MyModel().to(device)
    # best_model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, precision, recall, f1_score = Test(model, test_dataloader, criterion, device)
    print('Test Loss: {:.4f}, Test Acc: {:.4f},Precision: {:.4f},Recall: {:.4f},F1_score: {:.4f}'.format(test_loss,
                                                                                                         test_acc,
                                                                                                         precision,
                                                                                                         recall,
                                                                                                         f1_score))
