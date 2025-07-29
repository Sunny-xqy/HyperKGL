import argparse

def get_args():
    parser = argparse.ArgumentParser(description='HyperSMP模型参数')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--window_size', type=int, default=4, help='随机游走窗口大小')
    parser.add_argument('--num_neg_samples', type=int, default=5, help='负采样数量')
    
    # 模型相关参数
    parser.add_argument('--embed_dim', type=int, default=200, help='嵌入维度')
    parser.add_argument('--n_layers', type=int, default=2, help='图神经网络层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout率')
    
    # 随机游走参数
    parser.add_argument('--walk_length', type=int, default=10, help='随机游走长度')
    parser.add_argument('--num_walks', type=int, default=10, help='每个节点的游走次数')
    
    return parser.parse_args() 