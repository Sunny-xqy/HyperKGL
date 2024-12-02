
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

import random
import json

from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

path = '/home/xuqingying/my_work/Hypergraph/Predict'

def similarity(embeddings,node_embedding):
    """
    判断节点嵌入之间的相似度
    embeddings = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) 所有可能的向量几何
    node_embedding = np.array([4, 5, 6]).reshape(1, -1)  # 被测试节点
    """
    embeddings = np.array(embeddings)
    node_embedding = np.array([np.array(node_embedding)]).reshape(1, -1)
    # cos = cosine_similarity(embeddings,node_embedding)
    cos = euclidean_distances(embeddings,node_embedding)
    return cos.reshape(-1)


def mean_reciprocal_rank(y_true_list, y_pred_scores):
    """
    计算 Mean Reciprocal Rank (MRR)。
    
    - y_true_list: 每semalink的真实标签列表，1 表示真实的，0 表示不真实存在。
    - y_pred_scores: 每semalink的相应位置的预测得分。
    
    - MRR 值
    """
    mrr_sum = 0.0
    num_queries = len(y_true_list)
    
    for y_true, scores in zip(y_true_list, y_pred_scores):
        # 按得分对结果进行排序，得到排序后的索引
        sorted_indices = np.argsort(-scores)
        for rank, idx in enumerate(sorted_indices, start=1):
            if y_true[idx] == 1:
                mrr_sum += 1.0 / rank
                break
        else:
            mrr_sum += 0.0
    
    return mrr_sum / num_queries

def hit_at_k(y_true_list, y_pred_scores, k):
    """
    计算 Hit@k。
    - y_true_list: 每semalink的真实标签列表，1 表示真实的，0 表示不真实存在。
    - y_pred_scores: 每semalink的相应位置的预测得分。
    - k: Hit@k 中的 k 值
    - Hit@k 值
    """
    hit_count = 0
    num_queries = len(y_true_list)
    
    for y_true, scores in zip(y_true_list, y_pred_scores):
        # 按得分对结果进行排序，得到排序后的索引
        sorted_indices = np.argsort(-scores)
        if any(y_true[idx] == 1 for idx in sorted_indices[:k]):
            hit_count += 1
    
    return hit_count / num_queries

class property():
    def __init__(self,name,trainname,testname,model):
        self.model = model
        if self.model == 'GATNE':
            self.embedding = self.obtain_embedding_fromGATNE(name)
        elif self.model =='KGAT' or self.model =='ConvRot' or self.model =='HyperKGL'  or self.model =='TransR':
            self.embedding = self.obtain_embedding_fromKGAT(name)
        elif model == 'Hyper-SAGNN':
            self.embedding = self.obtain_embedding_fromHyperSAGNN(name)
        elif model == 'IHGNN':
            self.embedding = self.obtain_embedding_fromIHGNN(name)

        self.all_property = self.obtain_property_list(trainname,testname)

    def obtain_embedding_fromGATNE(self,name):
        """
        name:相应的数据集对应的嵌入文件路径
        """
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
                # print(concatenated_embedding[sub_key])
                pass
        
        # 转换为普通字典（如果需要）
        concatenated_embedding = dict(concatenated_embedding)
        print('Finish embedding loading......')
        return concatenated_embedding


    def obtain_embedding_fromKGAT(self,name):
        """
        name:相应的数据集对应的嵌入文件路径
        """
        data = np.load(name)
    
        # # 打印数据类型和形状
        # print(f"Data type: {data.dtype}")
        # print(f"Data shape: {data.shape}")
        
        # # 打印前几个元素
        # if data.ndim > 1:
        #     print("First few rows:\n", data[:5])
        # else:
        #     print("First few elements:\n", data[:5])
        return data
    
    def obtain_embedding_fromHyperSAGNN(self,filepath):
        data1 = np.load(filepath+'mymodel_0.npy')  # 论文嵌入
        data2 = np.load(filepath+'mymodel_2.npy')  # 作者嵌入

        data1_dict = {i: row.tolist() for i, row in enumerate(data1)}
        data2_dict = {i+len(data1_dict): row.tolist() for i, row in enumerate(data2)}

        data = {**data1_dict, **data2_dict}
        return data

    def obtain_embedding_fromIHGNN(self,filepath):
        data1 = np.load(filepath+'item.npy')  # 论文嵌入
        data2 = np.load(filepath+'user.npy')  # 作者嵌入

        data1_dict = {i: row.tolist() for i, row in enumerate(data1)}
        data2_dict = {i+len(data1_dict): row.tolist() for i, row in enumerate(data2)}

        data = {**data1_dict, **data2_dict}
        return data

    def obtain_property_list(self,trainname,testname):
        with open(trainname, 'r') as f:
            traindata = json.load(f)
        with open(testname, 'r') as f:
            testdata = json.load(f)
        hyperedge_dict = {}
        for i in traindata + testdata:
            if i[1] == 1:
                try:
                    hyperedge_dict[i[0]].append(i[2])
                except:
                    hyperedge_dict[i[0]] = []
                    hyperedge_dict[i[0]].append(i[2])
        return hyperedge_dict

    def test_property(self,test_number):
        """
        property预测测试，因为是所有的节点的共同属性，所以需要每个节点都与之相关
        test_number:被测试的property个数
        """
        #找到保存了嵌入的所有property
        if self.model == 'GATNE':
            property_embedding_list = self.obtain_embedding_for_test_GATNE(list(self.all_property.keys()))
            property_list = list(property_embedding_list.keys())
        elif self.model == 'KGAT' or self.model == 'ConvRot' or self.model =='HyperKGL'  or self.model =='TransR':
            property_embedding_list = self.obtain_embedding_for_test_KGAT(list(self.all_property.keys()))
            property_list = list(self.all_property.keys())
        elif self.model == 'Hyper-SAGNN':
            property_embedding_list = self.obtain_embedding_for_test_HyperSAGNN(list(self.all_property.keys()))
            property_list = list(self.all_property.keys())
        elif self.model == 'IHGNN':
            property_embedding_list = self.obtain_embedding_for_test_IHGNN(list(self.all_property.keys()))
            property_list = list(self.all_property.keys())

        y_true_list = []
        y_pre_list = []
        for property,node_list in tqdm(list(self.all_property.items())[:test_number]):
            #读取property对应的超边内的节点的嵌入
            if self.model == 'GATNE':
                node_embeddings = list(self.obtain_embedding_for_test_GATNE(node_list).values())
            elif self.model == 'KGAT' or self.model == 'ConvRot' or self.model =='HyperKGL'  or self.model =='TransR':
                node_embeddings = list(self.obtain_embedding_for_test_KGAT(node_list).values())
            elif self.model == 'Hyper-SAGNN':
                node_embeddings = list(self.obtain_embedding_for_test_HyperSAGNN(node_list).values())
            elif self.model == 'IHGNN':
                node_embeddings = list(self.obtain_embedding_for_test_IHGNN(node_list).values())
            
            try:

                #生成对应的y_true值
                # 找到目标键的索引
                index = property_list.index(property)
                y_true = np.zeros(len(property_list), dtype=int)
                # 将指定索引位置设置为 1
                y_true[index] = 1

                for embedding in node_embeddings:
                    y_pre = similarity(list(property_embedding_list.values()),embedding)
                    y_true_list.append(y_true)
                    y_pre_list.append(y_pre)
                # print('Done 1')
            except:
                pass
            
        y_true_list = np.array(y_true_list)
        y_pre_list = np.array(y_pre_list)
        # print(f"Predict list:{y_pre_list}")
        # print(f"True list:{y_true_list}")
        MRR = mean_reciprocal_rank(y_true_list=y_true_list,y_pred_scores=y_pre_list)
        print(f"The MRR score is: {MRR}")
        hit={}
        for i in range(1,21):
            hit[i] = hit_at_k(y_true_list,y_pre_list,i)
            print(f"The Hit@{i} score is: {hit[i]}")
        return MRR,hit
            
    
    def obtain_embedding_for_test_GATNE(self,node_list):
        """
        根据node_list提取相应的嵌入
        """
        corresponding_embedding = {}
        for key in node_list:
            if str(key) in self.embedding.keys():
                corresponding_embedding[key]=self.embedding[str(key)]
        # print('Obtaining the embeddings......')
        return corresponding_embedding
    
    def obtain_embedding_for_test_KGAT(self,node_list):
        """
        根据node_list提取相应的嵌入
        """
        corresponding_embedding = {}
        for key in node_list:
            corresponding_embedding[key] = self.embedding[key]
        # print('Obtaining the embeddings......')
        return corresponding_embedding

    def obtain_embedding_for_test_HyperSAGNN(self,node_list):
        """
        根据node_list提取相应的嵌入
        """
        corresponding_embedding = {}
        for key in node_list:
            corresponding_embedding[key] = self.embedding[key]
        # print('Obtaining the embeddings......')
        return corresponding_embedding
    
    def obtain_embedding_for_test_IHGNN(self,node_list):
        """
        根据node_list提取相应的嵌入
        """
        corresponding_embedding = {}
        for key in node_list:
            corresponding_embedding[key] = self.embedding[key]
        # print('Obtaining the embeddings......')
        return corresponding_embedding

def cosine_similarity_for_2(v1, v2):
    dot_product = np.dot(v1, v2.T)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2)

class relation():
    """
    KGAT也没有生成多个嵌入，没办法判断节点之间的关系，舍弃：仅判断GATNE
    Change the task: 可以针对各个不同类型的关系，判断其存在别的关系类型的可能性？
    针对每一种关系类型：判断两两节点之间是否存在关系
    """
    def __init__(self,name,trainname,testname,model):
        self.model = model
        if model == 'GATNE':
            self.embedding = self.obtain_embedding_fromGATNE(name)
        elif model == 'KGAT' or model == 'ConvRot' or model == 'HyperKGL' or model == 'TransR':
            self.embedding = self.obtain_embedding_fromKGAT(name)
        elif model == 'Hyper-SAGNN':
            self.embedding = self.obtain_embedding_fromHyperSAGNN(name)
        elif model == 'IHGNN':
            self.embedding = self.obtain_embedding_fromIHGNN(name)
        self.hyperedge_dict,self.first_dict = self.obtain_relation_list(trainname,testname)
    

    def obtain_relation_list(self,trainname,testname):
        """
        获取超边以及对应的第一个位置的节点
        """
        with open(trainname, 'r') as f:
            traindata = json.load(f)
        with open(testname, 'r') as f:
            testdata = json.load(f)
        hyperedge_dict = {}
        first_dict = {}
        for i in traindata + testdata:
            try:
                hyperedge_dict[i[0]].append(i[2])
            except:
                hyperedge_dict[i[0]] = []
                hyperedge_dict[i[0]].append(i[2])
            if i[1] == 0:
                first_dict[i[0]] = i[2]
        return hyperedge_dict,first_dict
    
    def test_relation(self,test_number):
        """
        relationship预测测试，超边内所有节点作为候选节点
        test_number:被测试的property个数
        """
        y_true_list = []
        y_pre_list = []
        for hyperedge,node_list in tqdm(list(self.hyperedge_dict.items())[:test_number]):
            #读取property对应的超边内的节点的嵌入
            if self.model == 'GATNE':
                node_embeddings = self.obtain_embedding_for_test_GATNE(node_list+[hyperedge])
            elif self.model == 'KGAT' or self.model == 'ConvRot' or self.model == 'HyperKGL' or self.model == 'TransR':
                node_embeddings = self.obtain_embedding_for_test_KGAT(node_list+[hyperedge])
            elif self.model == 'Hyper-SAGNN':
                node_embeddings = self.obtain_embedding_for_test_HyperSAGNN(node_list+[hyperedge])
            elif self.model == 'IHGNN':
                node_embeddings = self.obtain_embedding_for_test_IHGNN(node_list+[hyperedge])
            
            try:
                #生成对应的y_true值
                # 找到目标键的索引
                index = list(node_embeddings.keys()).index(self.first_dict[hyperedge])
                y_true = np.zeros(len(node_list), dtype=int)
                # 将指定索引位置设置为 1
                y_true[index] = 1
            
                y_pre = similarity(list(node_embeddings.values())[:-1],node_embeddings[hyperedge])
                y_true_list.append(y_true)
                y_pre_list.append(y_pre)
            except:
                pass
           
        # print(f"Predict list:{y_pre_list}")
        # print(f"True list:{y_true_list}") 
       
        MRR = mean_reciprocal_rank(y_true_list=y_true_list,y_pred_scores=y_pre_list)
        print(f"The MRR score is: {MRR}")
        hit={}
        for i in range(1,11):
            hit[i] = hit_at_k(y_true_list,y_pre_list,i)
            print(f"The Hit@{i} score is: {hit[i]}")
        return MRR,hit

    def obtain_embedding_fromKGAT(self,name):
        data = np.load(name)
        
        # # 打印数据类型和形状
        # print(f"Data type: {data.dtype}")
        # print(f"Data shape: {data.shape}")
        
        # # 打印前几个元素
        # if data.ndim > 1:
        #     print("First few rows:\n", data[:5])
        # else:
        #     print("First few elements:\n", data[:5])
        return data

    def obtain_embedding_fromGATNE(self,name):
        """
        name:相应的数据集对应的嵌入文件路径
        """
        embedding = np.load(name,allow_pickle=True).item()['0']
        # # embedding = np.concatenate([embeddings[key] for key in embeddings], axis=0)

        # concatenated_embedding = defaultdict(list)
        
        # # 遍历每个主键和子键
        # for main_key, sub_dict in embedding.items():
        #     for sub_key, vector in sub_dict.items():
        #         concatenated_embedding[sub_key].append(np.array(vector))
        
        # # 对每个子键的数组列表进行拼接
        # for sub_key in concatenated_embedding:
        #     try:
        #         concatenated_embedding[sub_key] = np.concatenate(concatenated_embedding[sub_key])
        #     except:
        #         print(concatenated_embedding[sub_key])
        
        # # 转换为普通字典（如果需要）
        # concatenated_embedding = dict(concatenated_embedding)
        print('Finish embedding loading......')
        return embedding
    
    def obtain_embedding_fromHyperSAGNN(self,filepath):
        data1 = np.load(filepath+'mymodel_0.npy')  # 论文嵌入
        data2 = np.load(filepath+'mymodel_2.npy')  # 作者嵌入

        data1_dict = {i: row.tolist() for i, row in enumerate(data1)}
        data2_dict = {i+len(data1_dict): row.tolist() for i, row in enumerate(data2)}

        data = {**data1_dict, **data2_dict}
        return data

    def obtain_embedding_fromIHGNN(self,filepath):
        data1 = np.load(filepath+'item.npy')  # 论文嵌入
        data2 = np.load(filepath+'user.npy')  # 作者嵌入

        data1_dict = {i: row.tolist() for i, row in enumerate(data1)}
        data2_dict = {i+len(data1_dict): row.tolist() for i, row in enumerate(data2)}

        data = {**data1_dict, **data2_dict}
        return data

    def obtain_embedding_for_test_GATNE(self,node_list):
        """
        根据node_list提取相应的嵌入
        """
        corresponding_embedding = {}
        for key in node_list:
            if str(key) in self.embedding.keys():
                corresponding_embedding[key]=self.embedding[str(key)]
        # print('Obtaining the embeddings......')
        return corresponding_embedding
    
    def obtain_embedding_for_test_KGAT(self,node_list):
        """
        根据node_list提取相应的嵌入
        """
        corresponding_embedding = {}
        for key in node_list:
            corresponding_embedding[key] = self.embedding[key]
        # print('Obtaining the embeddings......')
        return corresponding_embedding

    def obtain_embedding_for_test_HyperSAGNN(self,node_list):
        """
        根据node_list提取相应的嵌入
        """
        corresponding_embedding = {}
        for key in node_list:
            corresponding_embedding[key] = self.embedding[key]
        # print('Obtaining the embeddings......')
        return corresponding_embedding
    
    def obtain_embedding_for_test_IHGNN(self,node_list):
        """
        根据node_list提取相应的嵌入
        """
        corresponding_embedding = {}
        for key in node_list:
            corresponding_embedding[key] = self.embedding[key]
        # print('Obtaining the embeddings......')
        return corresponding_embedding
    
    # def obtain_relation_list(self,testname):
    #     with open(testname, 'r') as f:
    #         testdata = json.load(f)
    #     return testdata
    

    # def test_relation_forGATNE(self):
    #     """
    #     relation预测测试，因为是所有的节点的共同属性，所以需要每个节点都与之相关
    #     test_number:被测试的property个数
    #     """
    #     relation_list = list(self.embedding.keys())
    #     y_true_list = []
    #     y_pre_list = []
    #     print(self.embedding['0'].keys())
    #     for data in self.testdata:
    #         try:
    #             relation = str(data[1])
    #             index = relation_list.index(relation)  
    #             y_true = np.zeros(len(relation_list), dtype=int)
    #             # 将指定索引位置设置为 1
    #             y_true[index] = 1
    #             y_pre=[]
    #             for re in relation_list:
    #                 # print(data[-1])
                    
    #                 y_pre.append(cosine_similarity_for_2(self.embedding[re][str(data[0])],self.embedding[re][str(data[-1])]))
    #             y_true_list.append(y_true)
    #             y_pre_list.append(np.array(y_pre))
    #             # print('Done 1')
    #         except:
    #             # print('Nodes not in embeddings')
    #             pass
    #     y_true_list = np.array(y_true_list)
    #     y_pre_list = np.array(y_pre_list)
    #     MRR = mean_reciprocal_rank(y_true_list=y_true_list,y_pred_scores=y_pre_list)
    #     print(f"The MRR score is: {MRR}")
    #     hit={}
    #     for i in range(1,11):
    #         hit[i] = hit_at_k(y_true_list,y_pre_list,i)
    #         print(f"The Hit@{i} score is: {hit[i]}")
    #     return MRR,hit
    

if __name__ == '__main__':
    sourcetrain = '/home/xuqingying/my_work/Hypergraph/DBLP_data/triple_data/train_part.json'
    sourcetest = '/home/xuqingying/my_work/Hypergraph/DBLP_data/triple_data/test_part.json'
    dataset='dblpsmall'

    # sourcetrain = '/home/xuqingying/my_work/Hypergraph/MovieLens_data/triple_data/train.json'
    # sourcetest = '/home/xuqingying/my_work/Hypergraph/MovieLens_data/triple_data/test.json'
    # dataset='movielens'

    
    # model='GATNE'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/GATNE/data/dblpsmall/dblpsmall.npy'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/GATNE/data/movielens/movielens.npy'

    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/Predict/GATNE/numpy/moviehhyper.npy'
    # dataname = 'moviehyper'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/Predict/GATNE/numpy/dblp_hyper.npy'
    # dataname ='dblphyper'

    # model = 'KGAT'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/KGAT-pytorch/trained_model/KGAT/dblpsmall/dblpsmall.npy'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/KGAT-pytorch/trained_model/KGAT/movielens/movielens.npy'

    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/Predict/KGAT/movielenshyper.npy'
    # dataname = 'moviehyper'

    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/Predict/KGAT/dblphyper.npy'
    # dataname = 'dblphyper'


    
    # model = 'Hyper-SAGNN'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/Hyper-SAGNN/embed_res/DBLP/'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/Hyper-SAGNN/embed_res/MovieLens/'

    # model = 'ConvRot'
    # embeddingpath ='/home/xuqingying/my_work/Hypergraph/ConvRot/ConvRot-main/convrot_movielens_entity_embeddings.npy'
    # embeddingpath ='/home/xuqingying/my_work/Hypergraph/ConvRot/ConvRot-main/convrot_movielens_hyper_entity_embeddings.npy'

    # embeddingpath ='/home/xuqingying/my_work/Hypergraph/ConvRot/ConvRot-main/convrot_dblp_entity_embeddings.npy'
    # embeddingpath ='/home/xuqingying/my_work/Hypergraph/ConvRot/ConvRot-main/convrot_dblp_hyper_entity_embeddings.npy'
    # dataset = 'dblphyper'

    # model = 'IHGNN'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/IHGNN/results/DBLP/'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/IHGNN/results/MovieLens/'

    # model = 'HINGE'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/HINGE_code/HINGE/data/DBLP/embedding.npy'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/HINGE_code/HINGE/data/MovieLens/embedding.npy'


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
    embeddingpath = '/home/xuqingying/my_work/Hypergraph/HyperKGL/0806/dblp4node_embeddings.npy'
    # dataname = 'dblpsmall-1' 

    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/HyperKGL/0806/movielens2_woenode_embeddings.npy'
    # dataname = 'movielen1'




    property_pre = property(embeddingpath,sourcetrain,sourcetest,model)
    print('预处理完成')
    property_pre.test_property(1000)

    # relation_pre = relation(embeddingpath,sourcetrain,sourcetest,model)
    # relation_pre.test_relation(700)