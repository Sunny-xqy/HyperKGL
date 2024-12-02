from sklearn.metrics import precision_score, recall_score, f1_score


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from collections import defaultdict
import pandas as pd

import random

path = '/home/xuqingying/my_work/Hypergraph/Predict/'

class Classification(nn.Module):
    def __init__(self, input_dim, reduced_dim, num_classes):
        super(Classification, self).__init__()
        self.fc1 = nn.Linear(input_dim, reduced_dim)  # 降维层
        self.fc2 = nn.Linear(reduced_dim, num_classes)  # 分类层
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)  # 激活函数
        x = self.fc2(x)
        return x

def train(embedding,labels,reduced_dim,num_classes,modelname,dataname):
    """
    embedding: 从GRL模型钟输出的嵌入
    label: 标签值
    reduced_dim: 中间层神经元个数
    num_classes： 类别数
    """
    print('Start to load data......')
    embeddings_tensor = torch.tensor(embedding, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    print(embeddings_tensor)
    print(labels_tensor)

    # 创建数据集和数据加载器
    dataset = TensorDataset(embeddings_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # 定义模型参数
    input_dim = embedding.shape[1]  # 输入维度（原始嵌入向量的维度）

    # 创建模型实例
    model = Classification(input_dim, reduced_dim, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    #定义损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
    
    torch.save(model.state_dict(), path+modelname+'/'+dataname+'classify.pth')
    print("模型已保存到 classify.pth")
    return model

def load_model(input_dim, reduced_dim, num_classes,modelname,dataname):
    model = Classification(input_dim, reduced_dim, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    model.load_state_dict(torch.load(path+modelname+'/'+dataname+'classify.pth'))
    model.eval()  # 切换模型到评估模式
    print("模型已从 model.pth 加载")
    return model
    
def test(embedding,y_true,reduced_dim,num_classes,modelname,dataname):
    """
    embedding:输入测试集所对应的嵌入
    model:训练好的分类器
    y_ture:测试集真实的标签值
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)

    # 预测类别
    model = load_model(embedding.shape[1],reduced_dim,num_classes,modelname,dataname).to(device)
    with torch.no_grad():
        predictions = model(embeddings_tensor)
        y_pred = torch.argmax(predictions, dim=1).cpu()

    # 计算Precision, Recall和F1值
    # 计算不同 average 选项的 Precision, Recall, F1 分数
    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print(f"Precision (micro): {precision_micro:.2f}")
    print(f"Recall (micro): {recall_micro:.2f}")
    print(f"F1 Score (micro): {f1_micro:.2f}")

    print(f"Precision (macro): {precision_macro:.2f}")
    print(f"Recall (macro): {recall_macro:.2f}")
    print(f"F1 Score (macro): {f1_macro:.2f}")

    print(f"Precision (weighted): {precision_weighted:.2f}")
    print(f"Recall (weighted): {recall_weighted:.2f}")
    print(f"F1 Score (weighted): {f1_weighted:.2f}")
    return precision_micro,recall_micro,f1_micro,precision_macro,recall_macro,f1_macro,precision_weighted,recall_weighted,f1_weighted

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
        except:
            print(concatenated_embedding[sub_key])
    
    # 转换为普通字典（如果需要）
    concatenated_embedding = dict(concatenated_embedding)
    print('Finish embedding loading......')
    return concatenated_embedding

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
    return data

def obtain_embedding_fromKH(name):
    data1 = np.load(name[0]) 
    data2 = np.load(name[1]) 
    # 打印数据类型和形状
    print(f"Data type: {data1[0].dtype}")
    print(f"Data shape: {data1[0].shape}")
    print(f"Data type: {data2.dtype}")
    print(f"Data shape: {data2.shape}")
    return data1[0],data2

def obtain_embedding_fromHyperSAGNN(filepath):
    data1 = np.load(filepath+'mymodel_0.npy')  # 论文嵌入
    data2 = np.load(filepath+'mymodel_2.npy')  # 作者嵌入

    data1_dict = {i: row.tolist() for i, row in enumerate(data1)}
    data2_dict = {i+len(data1_dict): row.tolist() for i, row in enumerate(data2)}

    data = {**data1_dict, **data2_dict}
    return data

def obtain_embedding_fromIHGNN(filepath):
    data1 = np.load(filepath+'item.npy')  # 论文嵌入
    data2 = np.load(filepath+'user.npy')  # 作者嵌入

    data1_dict = {i: row.tolist() for i, row in enumerate(data1)}
    data2_dict = {i+len(data1_dict): row.tolist() for i, row in enumerate(data2)}

    data = {**data1_dict, **data2_dict}
    return data

def obtain_label(papername,authorname):
    
    df = pd.read_csv(papername)
    
    # 将 Series 转换为字典
    paper = pd.Series(df.iloc[:,-1].values,  index=df.index).to_dict()

    df = pd.read_csv(authorname)
    author = pd.Series(df.iloc[:,-1].values, index = df['index']).to_dict()

    merged_dict = {**paper , **author}

    return merged_dict


def split_train_test(label):
    """
    根据label划分训练集以及测试机
    """
    nodes = list(label.keys())
    random.shuffle(nodes)
    # 划分数据，2:8 比例
    split_index = int(0.3 * len(nodes))
    test_node = nodes[:split_index]
    train_node = nodes[split_index:]
    test_label = {key: label[key] for key in test_node}
    train_label = {key: label[key] for key in train_node}
    return train_label,test_label

def obtain_embedding_for_train_test_forGATNE(label,embedding):
    """
    根据划分的label提取embedding,并训练
    """
    corresponding_embedding = []
    corresponding_label = []
    for key in label.keys():
        if str(key) in embedding.keys():
            corresponding_embedding.append(embedding[str(key)])
            corresponding_label.append(label[key])
    numpy_embedding = np.array(corresponding_embedding)
    numpy_label = np.array(corresponding_label)
    print('Obtaining the labeles with embeddings......')
    return numpy_embedding,numpy_label


def obtain_embedding_for_train_test_forKGAT(label,embedding):
    """
    根据划分的label提取embedding,并训练
    """
    corresponding_embedding = []
    corresponding_label = []
    for key in label.keys():
        corresponding_embedding.append(embedding[key])
        corresponding_label.append(label[key])
    numpy_embedding = np.array(corresponding_embedding)
    numpy_label = np.array(corresponding_label)
    print('Obtaining the labeles with embeddings......')
    return numpy_embedding,numpy_label

def obtain_embedding_for_train_test_forKH(label,embedding):
    """
    根据划分的label提取embedding,并训练
    """
    corresponding_embedding = []
    corresponding_label = []
    print(len(embedding[0]))
    print(len(embedding[1]))
    for key in label.keys():
        # print(embedding[0][key])
        # print(embedding[1][key])
        corresponding_embedding.append(np.hstack((embedding[0][key], embedding[1][key])))
        corresponding_label.append(label[key])
    numpy_embedding = np.array(corresponding_embedding)
    numpy_label = np.array(corresponding_label)
    print('Obtaining the labeles with embeddings......')
    return numpy_embedding,numpy_label

def obtain_embedding_for_train_test_forHyperSAGNN(label,embedding):
    """
    根据划分的label提取embedding,并训练
    """
    corresponding_embedding = []
    corresponding_label = []
    for key in label.keys():
        corresponding_embedding.append(embedding[key])
        corresponding_label.append(label[key])
    numpy_embedding = np.array(corresponding_embedding)
    numpy_label = np.array(corresponding_label)
    print('Obtaining the labeles with embeddings......')
    return numpy_embedding,numpy_label

def obtain_embedding_for_train_test_forIHGNN(label,embedding):
    """
    根据划分的label提取embedding,并训练
    """
    corresponding_embedding = []
    corresponding_label = []
    for key in label.keys():
        corresponding_embedding.append(embedding[key])
        corresponding_label.append(label[key])
    numpy_embedding = np.array(corresponding_embedding)
    numpy_label = np.array(corresponding_label)
    print('Obtaining the labeles with embeddings......')
    return numpy_embedding,numpy_label

def load_npy_file(filename):
    data = np.load(filename, allow_pickle=True)
    data = data.item()
    if isinstance(data, list):
        print("List length:", len(data))
        print("First few items:", data[:5])
    elif isinstance(data, dict):
        print("Dictionary keys:", data.keys())
        # for key, value in data.items():
        #     print(f"Key: {key}, Value type: {type(value)}")
        #     # 如果值是数组或列表，检查其内容
        #     if isinstance(value, (np.ndarray, list)):
        #         print(f"Value shape (if ndarray): {getattr(value, 'shape', 'N/A')}")
        #         print(f"Value (first few items): {value[:5]}")
    else:
        print("Unrecognized data type.")

def predict(hyperedgesource,nodesource,embeddingpath,model,dataname):
    label = obtain_label(hyperedgesource,nodesource)
    train_label,test_label = split_train_test(label)
    
    items_with_value_minus_one = {key: value for key, value in train_label.items() if value == -1}

    print("Items with value -1:", items_with_value_minus_one)



    if model == 'GATNE':
        embedding = obtain_embedding_fromGATNE(embeddingpath)
        train_embedding,train_label = obtain_embedding_for_train_test_forGATNE(train_label,embedding)
        test_embedding,test_label = obtain_embedding_for_train_test_forGATNE(test_label,embedding)
        print(f'The length of trainning data for node classification is :{len(train_embedding)}')
       
    elif model == 'KGAT' or 'HyperKGL' or 'TransR':
        embedding = obtain_embedding_fromKGAT(embeddingpath)
        train_embedding,train_label = obtain_embedding_for_train_test_forKGAT(train_label,embedding)
        print(f'The length of trainning data for node classification is :{len(train_embedding)}')
        test_embedding,test_label = obtain_embedding_for_train_test_forKGAT(test_label,embedding)
    
    elif model == 'Hyper-SAGNN':
        embedding = obtain_embedding_fromHyperSAGNN(embeddingpath)
        train_embedding,train_label = obtain_embedding_for_train_test_forHyperSAGNN(train_label,embedding)
        test_embedding,test_label = obtain_embedding_for_train_test_forHyperSAGNN(test_label,embedding)
        print(f'The length of trainning data for node classification is :{len(train_embedding)}')

    elif model == 'ConvRot':
        embedding = obtain_embedding_fromKGAT(embeddingpath)
        train_embedding,train_label = obtain_embedding_for_train_test_forKGAT(train_label,embedding)
        test_embedding,test_label = obtain_embedding_for_train_test_forKGAT(test_label,embedding)
        print(f'The length of trainning data for node classification is :{len(train_embedding)}')

    elif model == 'IHGNN':
        embedding = obtain_embedding_fromIHGNN(embeddingpath)
        train_embedding,train_label = obtain_embedding_for_train_test_forIHGNN(train_label,embedding)
        test_embedding,test_label = obtain_embedding_for_train_test_forIHGNN(test_label,embedding)
        print(f'The length of trainning data for node classification is :{len(train_embedding)}')

    elif model == 'HINGE':
        embedding = obtain_embedding_fromKGAT(embeddingpath)
        train_embedding,train_label = obtain_embedding_for_train_test_forKGAT(train_label,embedding)
        test_embedding,test_label = obtain_embedding_for_train_test_forKGAT(test_label,embedding)
        print(f'The length of trainning data for node classification is :{len(train_embedding)}')
    
    
    elif model == 'K+H':
        embedding = obtain_embedding_fromKH(embeddingpath)
        train_embedding,train_label = obtain_embedding_for_train_test_forKH(train_label,embedding)
        test_embedding,test_label = obtain_embedding_for_train_test_forKH(test_label,embedding)
        print(f'The length of trainning data for node classification is :{len(train_embedding)}')

    train(train_embedding,train_label,100,max(train_label)+1,model,dataname)
    precision_micro,recall_micro,f1_micro,precision_macro,recall_macro,f1_macro,precision_weighted,recall_weighted,f1_weighted = test(test_embedding,test_label,100,max(train_label)+1,model,dataname)
    return precision_micro,recall_micro,f1_micro,precision_macro,recall_macro,f1_macro,precision_weighted,recall_weighted,f1_weighted



if __name__ == '__main__':
    hyperedgesource = '/home/xuqingying/my_work/Hypergraph/MovieLens_data/movie_list.csv'
    nodesource = '/home/xuqingying/my_work/Hypergraph/MovieLens_data/star_data.csv'
    dataname = 'movielens'

    # hyperedgesource = '/home/xuqingying/my_work/Hypergraph/DBLP_data/data_after_clean_part.csv'
    # nodesource = '/home/xuqingying/my_work/Hypergraph/DBLP_data/author_data_part.csv'
    # dataname = 'dblpsmall'

    
    # model='KGAT'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/KGAT-pytorch/trained_model/KGAT/movielens/movielens.npy'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/KGAT-pytorch/trained_model/KGAT/dblpsmall/dblpsmall.npy'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/Predict/KGAT/movielenshyper.npy'
    # dataname = 'moviehyper'

    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/Predict/KGAT/dblphyper.npy'
    # dataname = 'dblphyper'



    # model = 'GATNE'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/GATNE/data/movielens/movielens.npy'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/GATNE/data/dblpsmall/dblpsmall.npy'

    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/Predict/GATNE/numpy/moviehhyper.npy'
    # dataname = 'moviehyper'

    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/Predict/GATNE/numpy/dblp_hyper.npy'
    # dataname ='dblphyper'



    # model = 'Hyper-SAGNN'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/Hyper-SAGNN/embed_res/DBLP/'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/Hyper-SAGNN/embed_res/MovieLens/'


    # model = 'ConvRot'
    # embeddingpath ='/home/xuqingying/my_work/Hypergraph/ConvRot/ConvRot-main/convrot_movielens_entity_embeddings.npy'
    # embeddingpath ='/home/xuqingying/my_work/Hypergraph/ConvRot/ConvRot-main/convrot_movielens_hyper_entity_embeddings.npy'
    # dataname = 'moviehyper'

    # embeddingpath ='/home/xuqingying/my_work/Hypergraph/ConvRot/ConvRot-main/convrot_dblp_entity_embeddings.npy'
    # embeddingpath ='/home/xuqingying/my_work/Hypergraph/ConvRot/ConvRot-main/convrot_dblp_hyper_entity_embeddings.npy'
    # dataname = 'dblphyper'



    # model = 'HyperGAT'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/HyperGAT/HyperGAT-MovieLensMovie.npy'
    # embeddingpath ='/home/xuqingying/my_work/Hypergraph/HyperGAT/HyperGAT-DBLP.npy'

    # embeddingpath ='/home/xuqingying/my_work/Hypergraph/HyperGAT/HyperGAT-DBLPPaper.npy'
    # dataname = 'dblphyper'

    # embeddingpath ='/home/xuqingying/my_work/Hypergraph/HyperGAT/HyperGAT-MovieLensMovie.npy'
    # dataname ='moviehyper'

    # model = 'IHGNN'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/IHGNN/results/DBLP/'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/IHGNN/results/MovieLens/'

    # model = 'HINGE'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/HINGE_code/HINGE/data/DBLP/embedding.npy'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/HINGE_code/HINGE/data/MovieLens/embedding.npy'


    # model = 'K+H'
    # embeddingpath1 ='/home/xuqingying/my_work/Hypergraph/HyperGAT/HyperGAT-DBLPPaper.npy'
    # embeddingpath2 = '/home/xuqingying/my_work/Hypergraph/ConvRot/ConvRot-main/convrot_dblp_hyper_entity_embeddings.npy'
    # dataname = 'dblpsmall'

    # embeddingpath1 ='/home/xuqingying/my_work/Hypergraph/HyperGAT/HyperGAT-MovieLensMovie.npy'
    # embeddingpath2 = '/home/xuqingying/my_work/Hypergraph/ConvRot/ConvRot-main/convrot_movielens_hyper_entity_embeddings.npy'
    # dataname = 'movielens'

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
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/HyperKGL/0806/dblp_snode_embeddings.npy'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/HyperKGL/0806/dblp2-2node_embeddings.npy'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/HyperKGL/0806/dblp2_wosnode_embeddings.npy'
    # dataname = 'dblpsmall4' 
    # dataname = 'dblp2wos'

    embeddingpath = '/home/xuqingying/my_work/Hypergraph/HyperKGL/0806/movielens2_wosnode_embeddings.npy'
    dataname = 'movielen2_wos'

  
    predict(hyperedgesource,nodesource,embeddingpath,model,dataname)