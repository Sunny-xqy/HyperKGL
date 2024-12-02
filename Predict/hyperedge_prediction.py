
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np

from collections import defaultdict
import pandas as pd

import random
import json
import copy

from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score)

path = '/home/xuqingying/my_work/Hypergraph/Predict/'

class NodeDataset(Dataset):
    def __init__(self, data, labels):
        """
        Args:
            data (list of lists): 输入数据，每个样本是一个包含多个特征向量的二维矩阵
            labels (list): 每个样本的标签
        """
        # 将每个样本转换为 PyTorch tensor
        self.data = [torch.tensor(sample, dtype=torch.float32) for sample in data]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class AttentionAGG(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads):
        super(AttentionAGG, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # 自注意力层的输入 x: (batch_size, sequence_length, embed_dim)
        x, _ = self.self_attn(x, x, x)  # 注意力机制
        x = torch.mean(x, dim=1)  # 对序列维度求均值
        x = torch.relu(self.fc1(x))  # 激活函数
        x = self.fc2(x)  # 分类层
        return x


class Classification(nn.Module):
    def __init__(self, input_dim, reduced_dim, num_classes):
        super(Classification, self).__init__()
        self.fc1 = nn.Linear(input_dim, reduced_dim)  # 降维层
        self.fc2 = nn.Linear(reduced_dim, num_classes)  # 分类层
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)  # 激活函数
        x = self.fc2(x)
        return x


def train(embedding,labels,reduced_dim,num_head,modelname,dataname):
    """
    embedding: 从GRL模型钟输出的嵌入
    label: 标签值
    reduced_dim: 中间层神经元个数
    num_classes： 类别数
    """
    print('Start to load data......')
    embeddings_tensor = torch.tensor(embedding, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # 创建数据集和数据加载器
    # dataset = TensorDataset(embeddings_tensor, labels_tensor)
    dataset = NodeDataset(embeddings_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    """
    # 定义模型——使用节点分类来完成下游任务
    input_dim = sum(len(x) for x in embedding[0])  # 计算拼接后的特征维度
    hidden_dim = 50
    num_classes = 2
    model = Classification(input_dim, hidden_dim, num_classes)
    """

    # 模型参数
    embed_dim = len(embedding[0][0])  # 特征维度

    # 创建模型实例
    model = AttentionAGG(embed_dim, reduced_dim, num_head)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    #定义损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    num_epochs = 200

    #训练分类器
    print('Training processing......')
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), path+modelname+'/'+dataname+'_hyperedgepre.pth')
    print(f"Saving model at {path+modelname+'/'+dataname+'_hyperedgepre.pth'}")
    return model

def load_model(input_dim, reduced_dim, num_head,name,dataname):
    model = AttentionAGG(input_dim, reduced_dim, num_head)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    model.load_state_dict(torch.load(path+name+'/'+dataname+'_hyperedgepre.pth'))
    model.eval()  # 切换模型到评估模式
    print("loading model from hyperedgepre.pth")
    return model
    
def test(embedding,y_true,reduced_dim,num_classes,name,dataname):
    """
    embedding:输入测试集所对应的嵌入
    model:训练好的分类器
    y_ture:测试集真实的标签值
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)

    # 预测类别
    embed_dim = len(embedding[0][0])

    model = load_model(embed_dim,reduced_dim,num_classes,name,dataname).to(device)
    with torch.no_grad():
        pre = model(embeddings_tensor)
        y_pred = torch.argmax(pre, dim=1).cpu()
        predictions = pre[torch.arange(pre.size(0)), y_pred].cpu()

    ps, rs, _ = precision_recall_curve(y_true, predictions)
    ROC = roc_auc_score(y_true, predictions)
    PR = auc(rs, ps)
    F1 = f1_score(y_true, y_pred)
    print(f'The ROC score of model is: {ROC}')
    print(f'The PR score of model is: {PR}')
    print(f'The F1 score of model is: {F1}')
    return ROC, PR, F1
    # return  F1,PR, F1



def obtain_embedding_fromGATNE(name):
    embedding = np.load(name,allow_pickle=True).item()
    # embedding = np.concatenate([embeddings[key] for key in embeddings], axis=0)

    concatenated_embedding = defaultdict(list)
    
    # 遍历每个主键和子键
    for main_key, sub_dict in embedding.items():
        for sub_key, vector in sub_dict.items():
            concatenated_embedding[sub_key].append(np.array(vector))
    
    # 对每个子键的数组列表进行拼接
    for sub_key in concatenated_embedding:
        try:
            concatenated_embedding[sub_key] = np.concatenate(concatenated_embedding[sub_key])
            len_embedding = len(concatenated_embedding[sub_key])
        except:
            print(concatenated_embedding[sub_key])
    
    # 转换为普通字典（如果需要）
    concatenated_embedding = dict(concatenated_embedding)
    print('Finish embedding loading......')
    return concatenated_embedding,len_embedding



def obtain_hyperedge(trainname,testname,number_hyperedge,number_node):
    """
    从train与test数据集中提取三元组并获得超图数据，并进行负采样
    trainname:train三元组
    testname:test三元组
    number_hyperedge:超边个数编码
    number_node：最大节点个数编码
    """
    with open(trainname, 'r') as f:
        traindata = json.load(f)
    with open(testname, 'r') as f:
        testdata = json.load(f)
    hyperedge_dict = {}
    for i in traindata + testdata:
        try:
            hyperedge_dict[i[0]].append(i[2])
        except:
            hyperedge_dict[i[0]] = []
            hyperedge_dict[i[0]].append(i[2])
    
    print(len(hyperedge_dict.keys()))
    hyperedges = list(hyperedge_dict.values())

    hyperedgesmall = []
    for i in hyperedges:
        if len(i) <= 3:
            hyperedgesmall.append(i)
    print(f'可用超边数量为：{len(hyperedges)-len(hyperedgesmall)}')


    # 打乱列表顺序
    indices = list(range(len(hyperedges)))
    random.shuffle(indices)

    # 使用打乱后的索引重排多个列表
    hyperedge = [hyperedges[i] for i in indices]

    # 计算划分点
    split_index = int(len(hyperedge) * 0.7)
    train_values = hyperedge[:split_index]
    test_values = hyperedge[split_index:]
    # 按行写入训练集和测试集，保留key值
    
    #负采样，并保留存在与否的label
    input_demension1 = max(len(sublist) for sublist in train_values)
    input_demension2 = max(len(sublist) for sublist in train_values)
    input_demension = max([input_demension1,input_demension2])

    train_negative = []
    for data in train_values:
        negative = copy.deepcopy(data)

        sample_paper = random.randint(number_hyperedge, number_node)
        negative[0] = sample_paper
        if negative not in train_values:
            train_negative.append(negative)
        else: 
            print(1)

    
    test_negative = []
    for data in test_values:
        negative = copy.deepcopy(data)
        sample_paper = random.randint(number_hyperedge, number_node)
        negative[0] = sample_paper
        sample_paper = random.randint(number_hyperedge, number_node)
        negative[-1] = sample_paper
        if negative not in test_values:
            test_negative.append(negative)
        
    print(f'Toaling have {len(train_negative)} negative datas......')
    return train_values,test_values,train_negative,test_negative,input_demension


def obtain_embedding_for_train_test_forGATNE(data,negativedata,embedding,input_demension,embedding_demension):
    """
    根据划分的label提取embedding,并训练<由于是超图需要提取最大超边内点数，填充其余点用zero
    data: 正样本
    negatuvedata:负样本
    embedding:GRL模型生成嵌入
    input_demension:超边内最大节点个数
    embedding_demension:GRL模型生成的嵌入维度
    """
    corresponding_embedding = []
    corresponding_label = []
    for nodes in data:
        try:
            node_embedding = [] 
            for key in nodes:
                node_embedding.append(embedding[str(key)])
                np.array(node_embedding)
            if len(node_embedding) < input_demension and node_embedding != []:
                node_zero = np.zeros((input_demension-len(node_embedding),embedding_demension))
                print(len(node_zero.shape))
                node_embedding = np.concatenate((node_embedding, node_zero), axis=0)
            corresponding_embedding.append(node_embedding)
            corresponding_label.append(1)
            print('Done 1')
        except:
            pass
        
    print(f'There are {len(corresponding_label)} hyperedges in the dataset....')

    for nodes in negativedata:
        try:
            node_embedding = [] 
            for key in nodes:
                node_embedding.append(embedding[str(key)])
                np.array(node_embedding)
            if len(node_embedding) < input_demension and node_embedding != []:
                node_zero = np.zeros((input_demension-len(node_embedding),embedding_demension))
                print(len(node_zero.shape))
                node_embedding = np.concatenate((node_embedding, node_zero), axis=0)
            corresponding_embedding.append(node_embedding)
            corresponding_label.append(0)
            print('Done negative 1')
        except:
            pass

    indices = np.arange(len(corresponding_label))
    # 随机打乱索引列表
    np.random.shuffle(indices)
    # 使用打乱后的索引重排两个列表
    corresponding_embedding = [corresponding_embedding[i] for i in indices]
    corresponding_label = [corresponding_label[i] for i in indices]
    numpy_embedding = np.array(corresponding_embedding)
    numpy_label = np.array(corresponding_label)
    print('Obtaining the labeles with embeddings......')
    print(f'There are totaly {len(corresponding_label)} hyperedges in the dataset....')

    return numpy_embedding,numpy_label

def obtain_embedding_for_train_test_forKGAT(data,negativedata,embedding,input_demension,embedding_demension):
    """
    根据划分的label提取embedding,并训练<由于是超图需要提取最大超边内点数，填充其余点用zero
    data: 正样本
    negatuvedata:负样本
    embedding:GRL模型生成嵌入
    input_demension:超边内最大节点个数
    embedding_demension:GRL模型生成的嵌入维度
    """
    corresponding_embedding = []
    corresponding_label = []
    for nodes in data:
        try:
            node_embedding = [] 
            for key in nodes:
                node_embedding.append(embedding[key])
                np.array(node_embedding)
            if len(node_embedding) < input_demension and node_embedding != []:
                node_zero = np.zeros((input_demension-len(node_embedding),embedding_demension))
                node_embedding = np.concatenate((node_embedding, node_zero), axis=0)
            corresponding_embedding.append(node_embedding)
            corresponding_label.append(1)
            # print('Done 1')
        except:
            pass
        
    print(f'There are {len(corresponding_label)} positive hyperedges in the dataset....')

    for nodes in negativedata:
        node_embedding = [] 
        for key in nodes:
            node_embedding.append(embedding[key])
            np.array(node_embedding)
        if len(node_embedding) < input_demension and node_embedding != []:
            node_zero = np.zeros((input_demension-len(node_embedding),embedding_demension))
            node_embedding = np.concatenate((node_embedding, node_zero), axis=0)
        corresponding_embedding.append(node_embedding)
        corresponding_label.append(0)
        # print('Done negative 1')
       

    indices = np.arange(len(corresponding_label))
    # 随机打乱索引列表
    np.random.shuffle(indices)
    # 使用打乱后的索引重排两个列表
    corresponding_embedding = [corresponding_embedding[i] for i in indices]
    corresponding_label = [corresponding_label[i] for i in indices]
    numpy_embedding = np.array(corresponding_embedding)
    numpy_label = np.array(corresponding_label)
    print('Obtaining the labeles with embeddings......')
    print(f'There are totaly {len(corresponding_label)} hyperedges in the dataset....')

    return numpy_embedding,numpy_label

def obtain_embedding_fromKGAT(name):
    data = np.load(name)
    
    # 打印数据类型和形状
    print(f"Data type: {data.dtype}")
    print(f"Data shape: {data.shape}")
    
    # 打印前几个元素
    if data.ndim > 1:
        print("First few rows:\n", data[:5])
    else:
        print("First few elements:\n", data[:5])
    return data, len(data[0])

def obtain_embedding_fromHyperSAGNN(filepath):
    data1 = np.load(filepath+'mymodel_0.npy')  # 论文嵌入
    data2 = np.load(filepath+'mymodel_2.npy')  # 作者嵌入

    data1_dict = {i: row.tolist() for i, row in enumerate(data1)}
    data2_dict = {i+len(data1_dict): row.tolist() for i, row in enumerate(data2)}

    data = {**data1_dict, **data2_dict}
    return data,len(data[0])


def obtain_embedding_fromKH(name):
    data1 = np.load(name[0]) 
    data2 = np.load(name[1]) 
    # 打印数据类型和形状
    print(f"Data type: {data1[0].dtype}")
    print(f"Data shape: {data1[0].shape}")
    print(f"Data type: {data2.dtype}")
    print(f"Data shape: {data2.shape}")
    return [data1[0],data2],len(data1[0][0])+len(data2[0])


def obtain_embedding_for_train_test_forKH(data,negativedata,embedding,input_demension,embedding_demension):
    """
    根据划分的label提取embedding,并训练<由于是超图需要提取最大超边内点数，填充其余点用zero
    data: 正样本
    negatuvedata:负样本
    embedding:GRL模型生成嵌入
    input_demension:超边内最大节点个数
    embedding_demension:GRL模型生成的嵌入维度
    """
    corresponding_embedding = []
    corresponding_label = []
    for nodes in data:
        try:
            node_embedding = [] 
            for key in nodes:
                node_embedding.append(np.hstack((embedding[0][key], embedding[1][key])))
                np.array(node_embedding)
            if len(node_embedding) < input_demension and node_embedding != []:
                node_zero = np.zeros((input_demension-len(node_embedding),embedding_demension))
                node_embedding = np.concatenate((node_embedding, node_zero), axis=0)
            corresponding_embedding.append(node_embedding)
            corresponding_label.append(1)
            # print('Done 1')
        except:
            pass
       
    print(f'There are {len(corresponding_label)} positive hyperedges in the dataset....')

    for nodes in negativedata:
        node_embedding = [] 
        for key in nodes:
            node_embedding.append(np.hstack((embedding[0][key], embedding[1][key])))
            np.array(node_embedding)
        if len(node_embedding) < input_demension and node_embedding != []:
            node_zero = np.zeros((input_demension-len(node_embedding),embedding_demension))
            node_embedding = np.concatenate((node_embedding, node_zero), axis=0)
        corresponding_embedding.append(node_embedding)
        corresponding_label.append(0)
        # print('Done negative 1')
       

    indices = np.arange(len(corresponding_label))
    # 随机打乱索引列表
    np.random.shuffle(indices)
    # 使用打乱后的索引重排两个列表
    corresponding_embedding = [corresponding_embedding[i] for i in indices]
    corresponding_label = [corresponding_label[i] for i in indices]
    numpy_embedding = np.array(corresponding_embedding)
    numpy_label = np.array(corresponding_label)
    print('Obtaining the labeles with embeddings......')
    print(f'There are totaly {len(corresponding_label)} hyperedges in the dataset....')

    return numpy_embedding,numpy_label

def obtain_embedding_for_train_test_forHyperSAGNN(data,negativedata,embedding,input_demension,embedding_demension):
    """
    根据划分的label提取embedding,并训练<由于是超图需要提取最大超边内点数，填充其余点用zero
    data: 正样本
    negatuvedata:负样本
    embedding:GRL模型生成嵌入
    input_demension:超边内最大节点个数
    embedding_demension:GRL模型生成的嵌入维度
    """
    corresponding_embedding = []
    corresponding_label = []
    for nodes in data:
        try:
            node_embedding = [] 
            for key in nodes:
                node_embedding.append(embedding[key])
                np.array(node_embedding)
            if len(node_embedding) < input_demension and node_embedding != []:
                node_zero = np.zeros((input_demension-len(node_embedding),embedding_demension))
                node_embedding = np.concatenate((node_embedding, node_zero), axis=0)
            corresponding_embedding.append(node_embedding)
            corresponding_label.append(1)
            # print('Done 1')
        except:
            pass
        
    print(f'There are {len(corresponding_label)} positive hyperedges in the dataset....')

    for nodes in negativedata:
        node_embedding = [] 
        for key in nodes:
            node_embedding.append(embedding[key])
            np.array(node_embedding)
        if len(node_embedding) < input_demension and node_embedding != []:
            node_zero = np.zeros((input_demension-len(node_embedding),embedding_demension))
            node_embedding = np.concatenate((node_embedding, node_zero), axis=0)
        corresponding_embedding.append(node_embedding)
        corresponding_label.append(0)
        # print('Done negative 1')
       

    indices = np.arange(len(corresponding_label))
    # 随机打乱索引列表
    np.random.shuffle(indices)
    # 使用打乱后的索引重排两个列表
    corresponding_embedding = [corresponding_embedding[i] for i in indices]
    corresponding_label = [corresponding_label[i] for i in indices]
    numpy_embedding = np.array(corresponding_embedding)
    numpy_label = np.array(corresponding_label)
    print('Obtaining the labeles with embeddings......')
    print(f'There are totaly {len(corresponding_label)} hyperedges in the dataset....')

    return numpy_embedding,numpy_label

def obtain_embedding_fromIHGNN(filepath):
    data1 = np.load(filepath+'item.npy')  # 论文嵌入
    data2 = np.load(filepath+'user.npy')  # 作者嵌入

    data1_dict = {i: row.tolist() for i, row in enumerate(data1)}
    data2_dict = {i+len(data1_dict): row.tolist() for i, row in enumerate(data2)}

    data = {**data1_dict, **data2_dict}
    return data,len(data[0])

def obtain_embedding_for_train_test_forIHGNN(data,negativedata,embedding,input_demension,embedding_demension):
    """
    根据划分的label提取embedding,并训练<由于是超图需要提取最大超边内点数，填充其余点用zero
    data: 正样本
    negatuvedata:负样本
    embedding:GRL模型生成嵌入
    input_demension:超边内最大节点个数
    embedding_demension:GRL模型生成的嵌入维度
    """
    corresponding_embedding = []
    corresponding_label = []
    for nodes in data:
        try:
            node_embedding = [] 
            for key in nodes:
                node_embedding.append(embedding[key])
                np.array(node_embedding)
            if len(node_embedding) < input_demension and node_embedding != []:
                node_zero = np.zeros((input_demension-len(node_embedding),embedding_demension))
                node_embedding = np.concatenate((node_embedding, node_zero), axis=0)
            corresponding_embedding.append(node_embedding)
            corresponding_label.append(1)
            # print('Done 1')
        except:
            pass
        
    print(f'There are {len(corresponding_label)} positive hyperedges in the dataset....')

    for nodes in negativedata:
        node_embedding = [] 
        for key in nodes:
            node_embedding.append(embedding[key])
            np.array(node_embedding)
        if len(node_embedding) < input_demension and node_embedding != []:
            node_zero = np.zeros((input_demension-len(node_embedding),embedding_demension))
            node_embedding = np.concatenate((node_embedding, node_zero), axis=0)
        corresponding_embedding.append(node_embedding)
        corresponding_label.append(0)
        # print('Done negative 1')
       

    indices = np.arange(len(corresponding_label))
    # 随机打乱索引列表
    np.random.shuffle(indices)
    # 使用打乱后的索引重排两个列表
    corresponding_embedding = [corresponding_embedding[i] for i in indices]
    corresponding_label = [corresponding_label[i] for i in indices]
    numpy_embedding = np.array(corresponding_embedding)
    numpy_label = np.array(corresponding_label)
    print('Obtaining the labeles with embeddings......')
    print(f'There are totaly {len(corresponding_label)} hyperedges in the dataset....')

    return numpy_embedding,numpy_label





def predict(embeddingpath,sourcetrain,sourcetest,numberofhyperedge,numberofnode,model,dataset):
    train_values,test_values,train_negative,test_negative,input_demension = obtain_hyperedge(sourcetrain,sourcetest,numberofhyperedge,numberofnode)
    if model == 'GATNE':
        embedding,len_embedding = obtain_embedding_fromGATNE(embeddingpath)
        train_embedding,train_label = obtain_embedding_for_train_test_forGATNE(train_values,train_negative,embedding,input_demension,len_embedding)
        print(f'The length of trainning data for hyperedge prediction is :{len(train_embedding)}')
        test_embedding,test_label = obtain_embedding_for_train_test_forGATNE(test_values,test_negative,embedding,input_demension,len_embedding)
       
    elif model == 'KGAT' or 'HyperKGL' or 'TransR':
        embedding,len_embedding = obtain_embedding_fromKGAT(embeddingpath)
        train_embedding,train_label = obtain_embedding_for_train_test_forKGAT(train_values,train_negative,embedding,input_demension,len_embedding)
        print(f'The length of trainning data for hyperedge prediction is :{len(train_embedding)}')
        test_embedding,test_label = obtain_embedding_for_train_test_forKGAT(test_values,test_negative,embedding,input_demension,len_embedding)
    
    elif model == 'Hyper-SAGNN':
        embedding,len_embedding = obtain_embedding_fromHyperSAGNN(embeddingpath)
        train_embedding,train_label = obtain_embedding_for_train_test_forHyperSAGNN(train_values,train_negative,embedding,input_demension,len_embedding)
        print(f'The length of trainning data for hyperedge prediction is :{len(train_embedding)}')
        test_embedding,test_label = obtain_embedding_for_train_test_forHyperSAGNN(test_values,test_negative,embedding,input_demension,len_embedding)
        
    elif model == 'ConvRot':
        embedding,len_embedding = obtain_embedding_fromKGAT(embeddingpath)
        train_embedding,train_label = obtain_embedding_for_train_test_forKGAT(train_values,train_negative,embedding,input_demension,len_embedding)
        print(f'The length of trainning data for hyperedge prediction is :{len(train_embedding)}')
        test_embedding,test_label = obtain_embedding_for_train_test_forKGAT(test_values,test_negative,embedding,input_demension,len_embedding)
    
    elif model == 'IHGNN':
        embedding,len_embedding = obtain_embedding_fromIHGNN(embeddingpath)
        train_embedding,train_label = obtain_embedding_for_train_test_forIHGNN(train_values,train_negative,embedding,input_demension,len_embedding)
        print(f'The length of trainning data for hyperedge prediction is :{len(train_embedding)}')
        test_embedding,test_label = obtain_embedding_for_train_test_forIHGNN(test_values,test_negative,embedding,input_demension,len_embedding)

    
    elif model == 'K+H':
        embedding,len_embedding = obtain_embedding_fromKH(embeddingpath)
        train_embedding,train_label = obtain_embedding_for_train_test_forKH(train_values,train_negative,embedding,input_demension,len_embedding)
        print(f'The length of trainning data for hyperedge prediction is :{len(train_embedding)}')
        test_embedding,test_label = obtain_embedding_for_train_test_forKH(test_values,test_negative,embedding,input_demension,len_embedding)
        
    train(train_embedding,train_label,50,1,model,dataset)
    ROC,PR,F1 = test(test_embedding,test_label,50,1,model,dataset)
    return ROC,PR,F1


if __name__ == '__main__':
    
    # sourcetrain = '/home/xuqingying/my_work/Hypergraph/DBLP_data/triple_data/train_part.json'
    # sourcetest = '/home/xuqingying/my_work/Hypergraph/DBLP_data/triple_data/test_part.json'
    # numberofhyperedge = 9500
    # numberofnode = 25676
    # dataname='dblpsmall'

    sourcetrain = '/home/xuqingying/my_work/Hypergraph/MovieLens_data/triple_data/train.json'
    sourcetest = '/home/xuqingying/my_work/Hypergraph/MovieLens_data/triple_data/test.json'
    numberofhyperedge = 4500
    numberofnode = 18454
    dataname='movielens'

    # model='GATNE'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/GATNE/data/dblpsmall/dblpsmall.npy'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/GATNE/data/movielens/movielens.npy'

    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/Predict/GATNE/numpy/moviehhyper.npy'
    # dataset = 'moviehyper'

    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/Predict/GATNE/numpy/dblp_hyper.npy'
    # dataset ='dblphyper'


    # model = 'KGAT'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/KGAT-pytorch/trained_model/KGAT/dblpsmall/dblpsmall.npy'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/KGAT-pytorch/trained_model/KGAT/movielens/movielens.npy'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/Predict/KGAT/movielenshyper.npy'
    # dataset = 'moviehyper'

    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/Predict/KGAT/dblphyper.npy'
    # dataset = 'dblphyper'

    # model = 'Hyper-SAGNN'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/Hyper-SAGNN/embed_res/DBLP/'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/Hyper-SAGNN/embed_res/MovieLens/'

    # model = 'ConvRot'
    # embeddingpath ='/home/xuqingying/my_work/Hypergraph/ConvRot/ConvRot-main/convrot_movielens_entity_embeddings.npy'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/ConvRot/ConvRot-main/convrot_movielens_hyper_entity_embeddings.npy'
    # dataset = 'moviehyper'

    # embeddingpath ='/home/xuqingying/my_work/Hypergraph/ConvRot/ConvRot-main/convrot_dblp_entity_embeddings.npy'
    # embeddingpath ='/home/xuqingying/my_work/Hypergraph/ConvRot/ConvRot-main/convrot_dblp_hyper_entity_embeddings.npy'

    # dataset = 'dblphyper'

    # model = 'IHGNN'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/IHGNN/results/DBLP/'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/IHGNN/results/MovieLens/'


    
    # model = 'K+H'
    # embeddingpath1 ='/home/xuqingying/my_work/Hypergraph/HyperGAT/HyperGAT-DBLPPaper.npy'
    # embeddingpath2 = '/home/xuqingying/my_work/Hypergraph/ConvRot/ConvRot-main/convrot_dblp_hyper_entity_embeddings.npy'
   

    # embeddingpath1 ='/home/xuqingying/my_work/Hypergraph/HyperGAT/HyperGAT-MovieLensMovie.npy'
    # embeddingpath2 = '/home/xuqingying/my_work/Hypergraph/ConvRot/ConvRot-main/convrot_movielens_hyper_entity_embeddings.npy'
    

    # embeddingpath = [embeddingpath1,embeddingpath2]

    # model = 'TransR'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/HyperKGL/pretrain/result/DBLP/transr_all_entity.npy'
    # dataname = 'dblpsmall'

    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/HyperKGL/pretrain/result/DBLP_Hyper/transr_all_entity.npy'
    # dataname = 'dblphyper'

    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/HyperKGL/pretrain/result/MovieLens/transr_all_entity.npy'
    # dataname = 'movielens'

    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/HyperKGL/pretrain/result/MovieLens_Hyper/transr_all_entity.npy'
    # dataname = 'moviehyper'

    model = 'HyperKGL'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/HyperKGL/embedding/dblp_node_embeddings.npy'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/HyperKGL/0806/dblp2-2node_embeddings.npy'
    # dataname = 'dblpsmall'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/HyperKGL/0806/dblp2_wohnode_embeddings.npy'
    # dataname = 'dblpwoh' 
    embeddingpath = '/home/xuqingying/my_work/Hypergraph/HyperKGL/0806/movielens2_woenode_embeddings.npy'
    dataname = 'movielens_woe'


    predict(embeddingpath,sourcetrain,sourcetest,numberofhyperedge,numberofnode,model,dataname)
   