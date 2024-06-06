import numpy as np
import random
from collections import defaultdict

class PyFLOptimizer:
    def __init__(self, clients, global_model, server_update, alpha=0.9, beta=0.1):
        self.clients = clients
        self.global_model = global_model
        self.server_update = server_update
        self.alpha = alpha
        self.beta = beta
        self.phi = {client.id: 0 for client in clients}
    
    def compute_shapley_values(self, gradients):
        # Compute Shapley values for each client based on their gradients
        shapley_values = {}
        for client_id, grad in gradients.items():
            # Placeholder for Shapley value computation
            shapley_values[client_id] = np.linalg.norm(grad)  # Example metric
        return shapley_values
    
    def update_client_weights(self, shapley_values):
        for client_id, sv in shapley_values.items():
            self.phi[client_id] = self.alpha * self.phi[client_id] + self.beta * sv
    
    def select_clients(self, num_clients):
        total_phi = sum(np.exp(self.phi[client_id]) for client_id in self.phi)
        probabilities = {client_id: np.exp(self.phi[client_id]) / total_phi for client_id in self.phi}
        selected_clients = random.choices(list(self.phi.keys()), weights=probabilities.values(), k=num_clients)
        return selected_clients
    
    def train(self, rounds, local_epochs, batch_size):
        for t in range(rounds):
            selected_clients = self.select_clients(len(self.clients) // 2)
            gradients = defaultdict(list)
            
            for client_id in selected_clients:
                client = self.clients[client_id]
                local_model = client.get_model(self.global_model)
                for _ in range(local_epochs):
                    local_data = client.get_data(batch_size)
                    local_model.train(local_data)
                gradients[client_id].append(local_model.get_gradients())
                
            shapley_values = self.compute_shapley_values(gradients)
            self.update_client_weights(shapley_values)
            self.global_model.update(gradients, self.server_update)