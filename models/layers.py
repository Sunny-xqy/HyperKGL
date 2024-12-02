import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time

class SemalinkAttentiveAggregator(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, semalink_in_dim):
        super(SemalinkAttentiveAggregator, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.W_r = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.W_r)
        self.semalink_linear = nn.Linear(semalink_in_dim, out_dim)
        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, node_embeddings, semalink_embeddings, hyperedge_embeddings, semalinks):
        start_time = time.time()
        num_hyperedges = hyperedge_embeddings.size(0)
        semalink_embeddings_transformed = self.semalink_linear(semalink_embeddings)
        logging.info(
            f"SemalinkAttentiveAggregator: semalink_embeddings_transformed computed in {time.time() - start_time:.2f} seconds.")

        aggregated_features = []
        for i in range(num_hyperedges):
            start_time = time.time()
            hyperedge_emb = hyperedge_embeddings[i].unsqueeze(0)
            r_mul_t = torch.matmul(hyperedge_emb, self.W_r)
            semalink_attention_scores = []
            node_embs = []
            for node_idx, hyperedge_idx, relation_idx in semalinks:
                if hyperedge_idx == i:
                    node_emb = node_embeddings[node_idx].unsqueeze(0)
                    r_mul_h = torch.matmul(node_emb, self.W_r)
                    semalink_emb = semalink_embeddings_transformed[relation_idx].unsqueeze(0)
                    attention_score = torch.matmul(r_mul_h, torch.tanh(r_mul_t + semalink_emb).t()).squeeze(0)
                    semalink_attention_scores.append(attention_score)
                    node_embs.append(node_emb)
            if semalink_attention_scores:
                semalink_attention_scores = torch.stack(semalink_attention_scores)
                alpha_ik = F.softmax(semalink_attention_scores, dim=0)
                node_embs = torch.stack(node_embs).squeeze(1)
                hyperedge_emb = (alpha_ik.unsqueeze(1) * node_embs).sum(dim=0)
                aggregated_features.append(hyperedge_emb)
            else:
                aggregated_features.append(torch.zeros_like(hyperedge_emb.squeeze(0)))  # 确保所有张量大小一致
            logging.info(
                f"SemalinkAttentiveAggregator: hyperedge {i} processed in {time.time() - start_time:.2f} seconds.")

        aggregated_features = torch.stack(aggregated_features)
        aggregated_features = self.linear(aggregated_features)
        aggregated_features = self.activation(aggregated_features)

        if aggregated_features.dim() == 2:
            aggregated_features = self.layer_norm(aggregated_features)
        else:
            aggregated_features = self.bn(aggregated_features)

        aggregated_features = self.message_dropout(aggregated_features)

        logging.info(f"SemalinkAttentiveAggregator: Aggregated features shape: {aggregated_features.shape}")

        return aggregated_features


class InnerPropagation(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(HyperedgeBasedPropagation, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        self.W_e = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.W_e.weight)
        self.c_e = nn.Parameter(torch.randn(out_dim))

    def forward(self, node_embeddings, hyperedge_embeddings, hyperedge_to_nodes):
        start_time = time.time()
        num_nodes = node_embeddings.size(0)
        num_hyperedges = len(hyperedge_to_nodes)

        hyperedge_transformed = self.fc(hyperedge_embeddings)
        logging.info(
            f"InnerPropagation: hyperedge_transformed computed in {time.time() - start_time:.2f} seconds.")

        attention_scores = torch.zeros(num_nodes, num_hyperedges, device=node_embeddings.device)

        for i, nodes in hyperedge_to_nodes.items():
            start_time = time.time()
            hyperedge_embed = hyperedge_transformed[i]
            valid_nodes = [node for node in nodes if node < num_nodes]
            if not valid_nodes:
                continue
            node_embeddings_for_hyperedge = node_embeddings[valid_nodes]

            transformed_node_embeddings = self.W_e(node_embeddings_for_hyperedge)

            attention_scores_for_hyperedge = torch.matmul(
                F.leaky_relu(transformed_node_embeddings),
                self.c_e.unsqueeze(1)
            ).squeeze()

            attention_scores[valid_nodes, i] = attention_scores_for_hyperedge
            logging.info(
                f"InnerPropagation: hyperedge {i} attention scores computed in {time.time() - start_time:.2f} seconds.")

        attention_scores = F.softmax(attention_scores, dim=-1)

        aggregated_features = torch.zeros(num_nodes, self.fc.out_features, device=node_embeddings.device)

        for i, nodes in hyperedge_to_nodes.items():
            start_time = time.time()
            hyperedge_embed = hyperedge_transformed[i]
            valid_nodes = [node for node in nodes if node < num_nodes]
            if not valid_nodes:
                continue
            aggregated_features[valid_nodes] += attention_scores[valid_nodes, i].unsqueeze(-1) * hyperedge_embed
            logging.info(
                f"InnerPropagation: hyperedge {i} features aggregated in {time.time() - start_time:.2f} seconds.")

        updated_node_embeddings = torch.relu(aggregated_features)

        return updated_node_embeddings


class OuterPropagation(nn.Module):
    def __init__(self, in_dim, signal_dim, out_dim):
        super(Edge2NodePropagation, self).__init__()
        self.W_a = nn.Parameter(torch.randn(in_dim, signal_dim))
        self.W_f = nn.Parameter(torch.randn(in_dim * 2, out_dim))

    def forward(self, node_embeddings, semalink_embeddings, hyperedge_embeddings, semalinks):
        start_time = time.time()
        num_nodes = node_embeddings.size(0)
        final_embeddings = torch.zeros(num_nodes, self.W_f.size(1), device=node_embeddings.device)

        for node_idx, semalink_idx, hyperedge_idx in semalinks:
            if node_idx >= node_embeddings.size(0) or hyperedge_idx >= hyperedge_embeddings.size(
                    0) or semalink_idx >= semalink_embeddings.size(0):
                continue

            node_emb = node_embeddings[node_idx]
            semalink_emb = semalink_embeddings[semalink_idx]
            hyperedge_emb = hyperedge_embeddings[hyperedge_idx]

            r_mul_h = torch.matmul(node_emb.unsqueeze(0), self.W_a)
            r_mul_t = torch.matmul(hyperedge_emb.unsqueeze(0), self.W_a)

            semalink_emb_expanded = semalink_emb.unsqueeze(0).expand_as(r_mul_h)
            gamma_ki = torch.matmul(r_mul_h, torch.tanh(r_mul_t + semalink_emb_expanded).t())
            gamma_ki = F.softmax(gamma_ki.squeeze(0), dim=0)

            f_i = torch.sum(gamma_ki.unsqueeze(0) * hyperedge_emb, dim=0)

            h_i = node_embeddings[node_idx]

            u_i = torch.matmul(torch.cat([f_i, h_i], dim=0).unsqueeze(0), self.W_f)
            final_embeddings[node_idx] = u_i.squeeze(0)

        logging.info(f"OuterPropagation: processed in {time.time() - start_time:.2f} seconds.")
        return final_embeddings

