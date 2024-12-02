import torch
import logging
import time
from tqdm import tqdm
from torch.optim import Adam

class ModelTrainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.optimizer = Adam(model.parameters(), lr=args.learning_rate)
        
    def train(self, n_nodes, n_hyperedges, semalinks_list):
        logging.info("Starting training...")
        
        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss = 0
            logging.info(f"Epoch {epoch + 1}/{self.args.epochs} started...")
            
            with tqdm(total=n_nodes // self.args.batch_size + 1) as pbar:
                for i in range(0, n_nodes, self.args.batch_size):
                    loss = self._train_batch(i, n_nodes, n_hyperedges, semalinks_list)
                    total_loss += loss
                    pbar.update(1)
                    
            logging.info(f"Epoch {epoch + 1}/{self.args.epochs} completed. "
                        f"Total Loss: {total_loss}")
                        
        self._save_embeddings(n_nodes, n_hyperedges, semalinks_list)
    
    def _train_batch(self, i, n_nodes, n_hyperedges, semalinks_list):
        batch_node_indices = list(range(i, min(i + self.args.batch_size, n_nodes)))
        batch_node_indices = [idx for idx in batch_node_indices if idx < n_nodes]
        
        if not batch_node_indices:
            return 0
            
        node_indices = torch.LongTensor(batch_node_indices).cuda()
        hyperedge_indices = torch.LongTensor(range(n_hyperedges)).cuda()
        semalink_indices = torch.LongTensor(range(self.model.n_semantics)).cuda()
        semalinks = torch.tensor(semalinks_list, dtype=torch.long).cuda()
        
        self.optimizer.zero_grad()
        
        # Forward pass
        node_embeddings, hyperedge_embeddings = self.model(
            node_indices, hyperedge_indices, semalink_indices, semalinks
        )
        
        # Calculate loss
        sampled_semalinks = random.sample(semalinks_list, self.args.batch_size)
        v_i = torch.tensor([link[0] for link in sampled_semalinks]).cuda()
        e_k = torch.tensor([link[1] for link in sampled_semalinks]).cuda()
        sem_neg_samples = torch.tensor(
            random.choices(range(n_nodes), k=self.args.batch_size)
        ).cuda()
        
        loss = self.model.calc_total_loss(v_i, e_k, sem_neg_samples)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def _save_embeddings(self, n_nodes, n_hyperedges, semalinks_list):
        self.model.eval()
        with torch.no_grad():
            node_indices = torch.LongTensor(range(n_nodes)).cuda()
            hyperedge_indices = torch.LongTensor(range(n_hyperedges)).cuda()
            semalink_indices = torch.LongTensor(range(self.model.n_semantics)).cuda()
            semalinks = torch.tensor(semalinks_list, dtype=torch.long).cuda()
            
            node_embeddings, hyperedge_embeddings = self.model(
                node_indices, hyperedge_indices, semalink_indices, semalinks
            )
            
            torch.save(node_embeddings.cpu(), 'node_embeddings.pt')
            torch.save(hyperedge_embeddings.cpu(), 'hyperedge_embeddings.pt')
            logging.info("Embeddings saved successfully") 