import logging
from config import get_args
from data_loader import DataLoader
from HyperSMP.models.hypersmp import HyperSMP
from trainer import ModelTrainer

def main():
    # 设置日志
    logging.basicConfig(
        filename='hyperSMP.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    # 获取参数
    args = get_args()
    
    # 加载数据
    data_loader = DataLoader('../Dataset')
    (entity_embeddings, relation_embeddings, hyperedge_initial_embeddings,
     hyperedges_dict, semalinks_list, num_hyperedges) = data_loader.load_data()
    
    # 初始化模型
    n_nodes = len(entity_embeddings)
    n_semantics = len(relation_embeddings)
    model = HyperSMP(
        args=args,
        n_nodes=n_nodes,
        n_hyperedges=num_hyperedges,
        n_semantics=n_semantics,
        hypergraph=hyperedges_dict
    ).cuda()
    
    # 训练模型
    trainer = ModelTrainer(model, args)
    trainer.train(n_nodes, num_hyperedges, semalinks_list)

if __name__ == '__main__':
    main() 