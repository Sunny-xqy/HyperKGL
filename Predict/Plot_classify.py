import torch
import numpy as np
from sklearn.manifold import TSNE
import proplot as pplt
from node_classification import *


def embedding2(node_features_np):
    # 将数据转换为 numpy 数组
    # node_features_np = embeddings.numpy()
   

    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, random_state=42)
    node_features_2d = tsne.fit_transform(node_features_np)
    return node_features_2d

def plot_node(embeddings,labels_np,filename):
    # labels_np = labels.numpy()
    node_features_2d = embedding2(embeddings)

    # 设置颜色调色板
    num_classes = len(np.unique(labels_np))
    # cmap = pplt.Colormap('Set1', num_classes)

    # 创建一个散点图
    fig, ax = pplt.subplots(figsize=(10, 8))

    for class_label in np.unique(labels_np):
        mask = labels_np == class_label
        ax.scatter(node_features_2d[mask, 0], node_features_2d[mask, 1], label=f'Class {class_label}', s=10)

    # # 添加标题和标签
    # ax.format(title='Visualization of node features',
    #         xlabel='t-SNE component 1', ylabel='t-SNE component 2',
    #         legend='r', grid=False)

    # 显示图形
    ax.grid(visible=False)
    ax.format(xlocator=None, ylocator=None, borders=False)
    fig.show()
    fig.save(filename)

def get_activation(name,activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def obtain_embedding_from_trained_model(embedding, reduced_dim, num_classes,modelname,dataname):
    """
    embedding:输入测试集所对应的嵌入
    model:训练好的分类器
    y_ture:测试集真实的标签值
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)

    # 预测类别
    model = load_model(embedding.shape[1],reduced_dim,num_classes,modelname,dataname).to(device)

    # 创建一个字典来存储激活值
    activation = {}

    # 注册钩子函数到指定层
    hook_handle = model.fc2.register_forward_hook(get_activation('fc2',activation))

    with torch.no_grad():
        predictions = model(embeddings_tensor)

    # 获取中间层的嵌入
    layer2_activation = activation['fc2']

    # 输出中间层的嵌入
    print("Layer 2 activation:", layer2_activation)

    # 解除钩子函数
    hook_handle.remove()
    return layer2_activation

def vasualization(filename,hyperedgesource,nodesource,embeddingpath,model,dataname):
    label = obtain_label(hyperedgesource,nodesource)
    train_label,test_label = split_train_test(label)
    
    items_with_value_minus_one = {key: value for key, value in train_label.items() if value == -1}

    print("Items with value -1:", items_with_value_minus_one)



    if model == 'GATNE':
        embedding = obtain_embedding_fromGATNE(embeddingpath)
        train_embedding,train_label = obtain_embedding_for_train_test_forGATNE(train_label,embedding)
        test_embedding,test_label = obtain_embedding_for_train_test_forGATNE(test_label,embedding)
        print(f'The length of trainning data for node classification is :{len(train_embedding)}')
       
    elif model == 'KGAT' or model == 'ConvRot' or model == 'HINGE' or model == 'TransR' or model == 'HyperGAT' or model == 'HyperKGL':
        embedding = obtain_embedding_fromKGAT(embeddingpath)
        train_embedding,train_label = obtain_embedding_for_train_test_forKGAT(train_label,embedding)
        print(f'The length of trainning data for node classification is :{len(train_embedding)}')
        test_embedding,test_label = obtain_embedding_for_train_test_forKGAT(test_label,embedding)
    
    elif model == 'Hyper-SAGNN':
        embedding = obtain_embedding_fromHyperSAGNN(embeddingpath)
        train_embedding,train_label = obtain_embedding_for_train_test_forHyperSAGNN(train_label,embedding)
        test_embedding,test_label = obtain_embedding_for_train_test_forHyperSAGNN(test_label,embedding)
        print(f'The length of trainning data for node classification is :{len(train_embedding)}')

    elif model == 'IHGNN':
        embedding = obtain_embedding_fromIHGNN(embeddingpath)
        train_embedding,train_label = obtain_embedding_for_train_test_forIHGNN(train_label,embedding)
        test_embedding,test_label = obtain_embedding_for_train_test_forIHGNN(test_label,embedding)
        print(f'The length of trainning data for node classification is :{len(train_embedding)}')
     
    elif model == 'K+H':
        embedding = obtain_embedding_fromKH(embeddingpath)
        train_embedding,train_label = obtain_embedding_for_train_test_forKH(train_label,embedding)
        test_embedding,test_label = obtain_embedding_for_train_test_forKH(test_label,embedding)
        print(f'The length of trainning data for node classification is :{len(train_embedding)}')


    hidden_embeddings = obtain_embedding_from_trained_model(test_embedding,100,max(train_label)+1,model,dataname)
    hidden_embeddings = hidden_embeddings.cpu()
    plot_node(hidden_embeddings,test_label,filename)


if __name__ == '__main__':
    # hyperedgesource = '/home/xuqingying/my_work/Hypergraph/MovieLens_data/movie_list.csv'
    # nodesource = '/home/xuqingying/my_work/Hypergraph/MovieLens_data/star_data.csv'
    # dataname = 'movielens'

    hyperedgesource = '/home/xuqingying/my_work/Hypergraph/DBLP_data/data_after_clean_part.csv'
    nodesource = '/home/xuqingying/my_work/Hypergraph/DBLP_data/author_data_part.csv'
    dataname = 'dblpsmall'

    
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

    # model = 'HINGE'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/HINGE_code/HINGE/data/DBLP/embedding.npy'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/HINGE_code/HINGE/data/MovieLens/embedding.npy'


    # model = 'IHGNN'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/IHGNN/results/DBLP/'
    # embeddingpath = '/home/xuqingying/my_work/Hypergraph/IHGNN/results/MovieLens/'

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
    embeddingpath = '/home/xuqingying/my_work/Hypergraph/HyperKGL/0806/dblp2-2node_embeddings.npy'
    dataname = 'dblphyper'

    filename = path+model+'/'+dataname+'node.pdf'
    vasualization(filename,hyperedgesource,nodesource,embeddingpath,model,dataname)

