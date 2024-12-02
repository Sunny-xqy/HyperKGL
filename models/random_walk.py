import random
import logging
import time

class HypergraphRandomWalk:
    def __init__(self, hypergraph, walk_length, num_walks):
        self.num_walks = num_walks
        self.hypergraph = hypergraph
        self.walk_length = walk_length
        self.nodes = list(set(node for nodes in hypergraph.values() for node in nodes))

    def random_walk(self, start_node):
        walk = [start_node]
        while len(walk) < self.walk_length:
            current_node = walk[-1]
            candidate_edges = [edge for edge in self.hypergraph if current_node in self.hypergraph[edge]]
            if not candidate_edges:
                break
            chosen_edge = random.choice(candidate_edges)
            next_node = random.choice(self.hypergraph[chosen_edge])
            walk.append(next_node)
        return walk

    def generate_walks(self):
        walks = []
        for start_node in self.nodes:
            for _ in range(self.num_walks):
                walk_start_time = time.time()
                walks.append(self.random_walk(start_node))
                logging.info(
                    f"Random walk for node {start_node} completed in {time.time() - walk_start_time:.2f} seconds.")
        return walks

    def generate_pairs(self, window_size):
        walks_start_time = time.time()
        walks = self.generate_walks()
        logging.info(f"generate_walks completed in {time.time() - walks_start_time:.2f} seconds.")

        pairs_start_time = time.time()
        pairs = []
        for walk in walks:
            for i in range(len(walk)):
                for j in range(max(0, i - window_size), min(len(walk), i + window_size + 1)):
                    if i != j:
                        pairs.append((walk[i], walk[j]))
        logging.info(f"generate_pairs inner loop completed in {time.time() - pairs_start_time:.2f} seconds.")
        return pairs

    def generate_negative_samples(self, num_samples):
        nodes = list(self.nodes)
        return [random.choice(nodes) for _ in range(num_samples)]