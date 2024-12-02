import torch
import torch.nn as nn
import logging
from .layers import (
    SemalinkAttentiveAggregator,
    InnerPropagation,
    OuterPropagation
)
from .random_walk import HypergraphRandomWalk

class HyperKGL(nn.Module):
    def __init__(self, args, n_nodes, n_hyperedges, n_semantics, hypergraph, device):
        super(HyperKGL, self).__init__()
        self.device = device

        self.embed_dim = args.embed_dim
        self.n_layers = args.n_layers
        self.n_nodes = n_nodes
        self.n_hyperedges = n_hyperedges
        self.n_semantics = n_semantics

        self.pretrain = args.pretrain
        
       
        self.node_embeddings = nn.Embedding(len(n_nodes), self.embed_dim)
        self.hyperedge_embeddings = nn.Embedding(n_hyperedges, self.embed_dim)
        self.semalink_embeddings = nn.Embedding(len(n_semantics), self.embed_dim)
        if self.pretrain == '1':
            self.node_embeddings.weight.data.copy_(torch.tensor(n_nodes, dtype=torch.float32))
            self.semalink_embeddings.weight.data.copy_(torch.tensor(n_semantics, dtype=torch.float32))

        self.smalink_aggregators = nn.ModuleList()
        self.hyperedge_propagators = nn.ModuleList()
        self.edge2node_propagators = nn.ModuleList()

        for _ in range(self.n_layers):
            self.smalink_aggregators.append(SemalinkAttentiveAggregator(self.embed_dim, self.embed_dim))
            self.hyperedge_propagators.append(InnerPropagation(self.embed_dim, self.embed_dim))
            self.edge2node_propagators.append(OuterPropagation(self.embed_dim, self.embed_dim, self.embed_dim))

        self.walker = HypergraphRandomWalk(hypergraph, args.walk_length, args.num_walks)
        self.all_positive_hyperedge,self.all_negative_hyperedge = self.walker.generate_pairs(args.window_size,args.num_negative_samples)
        print('Finished random walk generated......')
        print(f'there are {len(self.all_positive_hyperedge)} positive samples and {len(self.all_negative_hyperedge)} negative samples.' )

    def forward(self, node_indices, hyperedge_indices, semalink_indices, semalinks):
        start_time = time.time()
        print("Forward pass started...")

        # filtered_semalinks = [(node_idx, hyperedge_idx, relation_idx) for (node_idx, hyperedge_idx, relation_idx) in
        #                       semalinks if node_idx in node_indices.tolist()]
        
        mask = torch.isin(semalinks[:, 0], node_indices)
        filtered_semalinks = semalinks[mask]
        

        for i in range(self.n_layers):
            layer_start_time = time.time()
            print(f"Layer {i + 1}/{self.n_layers} processing started...")

            step_start_time = time.time()
            co_hyperedge_embeddings= self.smalink_aggregators[i](self.node_embeddings.weight, self.semalink_embeddings.weight,
                                                               self.hyperedge_embeddings.weight, filtered_semalinks,filtered_semalinks[:,1].unique())
            self.hyperedge_embeddings.weight[hyperedge_indx] = co_hyperedge_embeddings
            print(
                f"Layer {i + 1}/{self.n_layers} SemalinkAttentiveAggregator processed in {time.time() - step_start_time:.2f} seconds.")
            step_start_time = time.time()

            
            node_embeddings = self.hyperedge_propagators[i](node_indices,self.node_embeddings.weight, self.hyperedge_embeddings.weight, reverse_dict)
            print(
                f"Layer {i + 1}/{self.n_layers} InnerPropagation processed in {time.time() - step_start_time:.2f} seconds.")

            step_start_time = time.time()
            node_embeddings = self.edge2node_propagators[i](node_embeddings, self.semalink_embeddings.weight, self.hyperedge_embeddings.weight,
                                                            filtered_semalinks)
            print(
                f"Layer {i + 1}/{self.n_layers} OuterPropagation processed in {time.time() - step_start_time:.2f} seconds.")


            print(
                f"Layer {i + 1}/{self.n_layers} processing completed in {time.time() - layer_start_time:.2f} seconds.")
        # print(node_embeddings[-1])
        with torch.no_grad():
            # print(self.node_embeddings.weight[node_indices].shape)
            # print(node_embeddings.shape)
            self.node_embeddings.weight[node_indices] = node_embeddings
        # self.hyperedge_embeddings.weight.data[hyperedge_indices] = hyperedge_embeddings
        return self.node_embeddings.weight, self.hyperedge_embeddings.weight
    

    def calc_semantic_loss(self, v_i, e_k, neg_samples):
        start_time = time.time()
        u_i = self.node_embeddings(v_i)
        nu_k = self.hyperedge_embeddings(e_k)
        u_j_neg = self.node_embeddings(neg_samples)
        pos_score = torch.sum(u_i * nu_k, dim=1)
        neg_score = torch.sum(u_j_neg * nu_k, dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-7)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-7)
        loss = torch.mean(pos_loss + neg_loss)
        print(f"calc_semantic_loss completed in {time.time() - start_time:.2f} seconds.")
        return loss


    def calc_hyperedge_loss(self, v_i):
        
        start_time = time.time()
        v_i = v_i.to(self.device)

        # 预处理正负样本
        # pos_indices = torch.tensor([pair for pair in self.all_positive_hyperedge if all(val in v_i.tolist() for val in pair)]).to(self.device)
        # neg_indices = torch.tensor([pair for pair in self.all_negative_hyperedge if all(val in v_i.tolist() for val in pair)]).to(self.device)

        v_i_set = set(v_i.tolist())
        pos_indices = torch.tensor([pair for pair in self.all_positive_hyperedge if set(pair).issubset(v_i_set)]).to(self.device)

        neg_indices = torch.tensor([pair for pair in self.all_negative_hyperedge if set(pair).issubset(v_i_set)]).to(self.device)
        # 批量化处理

        end_time = time.time()
       
        pos_i, pos_j = pos_indices[:, 0], pos_indices[:, 1]

        u_pos_i = self.node_embeddings(pos_i)
        u_pos_j = self.node_embeddings(pos_j)
        # pos_score = torch.sum(u_pos_i * u_pos_j, dim=1)
        # pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-7)

        pos_score = torch.sum(u_pos_i * u_pos_j, dim=1)

       
        pos_score = F.sigmoid(pos_score)
        pos_loss = -torch.log(pos_score + 1e-7)
        
       
        try:
            neg_i, neg_j = neg_indices[:, 0], neg_indices[:, 1]
            
            u_neg_i = self.node_embeddings(neg_i)
            u_neg_j = self.node_embeddings(neg_j)
            neg_score = torch.sum(u_neg_i * u_neg_j, dim=1)
            neg_loss = -torch.log(1 - F.sigmoid(neg_score) + 1e-7)
        except:
            neg_loss = torch.tensor(0.0, device=self.device)

        # 计算总损失
        if pos_loss.numel() > 0 and neg_loss.numel() > 0:
            loss = torch.mean(pos_loss) + torch.mean(neg_loss)
        elif pos_loss.numel() > 0:
            loss = torch.mean(pos_loss)
        elif neg_loss.numel() > 0:
            loss = torch.mean(neg_loss)
        else:
            loss = torch.tensor(0.0, device=self.device)

        # elapsed_time = end_time - start_time

        # print(f'Finished one batch of training, the loss is {loss.item()}')
        # print(f'Time taken for this batch: {elapsed_time:.4f} seconds')

        return loss


    def calc_total_loss(self, v_i, e_k, sem_neg_samples):
        start_time = time.time()
        sem_loss = self.calc_semantic_loss(v_i, e_k, sem_neg_samples)
        hyper_loss = self.calc_hyperedge_loss(v_i)
        l2_reg = _L2_loss_mean(self.node_embeddings.weight) + _L2_loss_mean(self.hyperedge_embeddings.weight)
        total_loss = sem_loss + hyper_loss + 0.001 * l2_reg  # 0.001为正则化系数
        logging.info(f"calc_total_loss completed in {time.time() - start_time:.2f} seconds.")
        return total_loss

    def get_embeddings(self):
        return self.node_embeddings.weight, self.hyperedge_embeddings.weight