import numpy as np
import torch
import json
import logging
import time

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def load_data(self):
        logging.info("Loading data...")
        start_time = time.time()
        
        entity_embeddings = np.load(f'{self.data_dir}/transr_train_entity.npy')
        relation_embeddings = np.load(f'{self.data_dir}/transr_train_relation.npy')
        
        with open(f'{self.data_dir}/hyperedges.json', 'r') as f:
            hyperedges = json.load(f)
            
        with open(f'{self.data_dir}/semalinks.json', 'r') as f:
            semalinks = json.load(f)
            
        logging.info(f"Data loaded in {time.time() - start_time:.2f} seconds.")
        
        return self.process_data(entity_embeddings, relation_embeddings, hyperedges, semalinks)
        
    def process_data(self, entity_embeddings, relation_embeddings, hyperedges, semalinks):
        logging.info("Processing data...")
        start_time = time.time()
        
        entity_embeddings = torch.tensor(entity_embeddings, dtype=torch.float).cuda()
        relation_embeddings = torch.tensor(relation_embeddings, dtype=torch.float).cuda()
        
        num_hyperedges = len(hyperedges)
        hyperedge_initial_embeddings = torch.zeros(
            (num_hyperedges, entity_embeddings.shape[1]), 
            dtype=torch.float
        ).cuda()
        
        hyperedges_dict = {edge['id']: edge['nodes'] for edge in hyperedges}
        semalinks_list = [(link['node_id'], link['hyperedge_id'], link['type']) 
                         for link in semalinks]
        
        logging.info(f"Data processed in {time.time() - start_time:.2f} seconds.")
        
        return (entity_embeddings, relation_embeddings, hyperedge_initial_embeddings,
                hyperedges_dict, semalinks_list, num_hyperedges) 