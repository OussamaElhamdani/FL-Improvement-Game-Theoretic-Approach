import numpy as np
import random

class Server:
    def __init__(self, global_model, clients, alpha=0.9, beta=0.1):
        self.global_model = global_model
        self.clients = clients
        self.alpha = alpha
        self.beta = beta
        self.phi = {client.id: 0 for client in clients}
        self.gradients = {}

    def compute_shapley_values(self):
        shapley_values = {}
        for client_id, grads in self.gradients.items():
            # Calculate Shapley value (example: norm of gradients)
            shapley_values[client_id] = np.linalg.norm(grads)
        return shapley_values

    def update_client_weights(self, shapley_values):
        for client_id, sv in shapley_values.items():
            self.phi[client_id] = self.alpha * self.phi[client_id] + self.beta * sv

    def select_clients(self, num_clients):
        total_phi = sum(np.exp(self.phi[client_id]) for client_id in self.phi)
        probabilities = [np.exp(self.phi[client_id]) / total_phi for client_id in self.phi]
        selected_clients = random.choices(list(self.phi.keys()), weights=probabilities, k=num_clients)
        return [client for client in self.clients if client.id in selected_clients]

    def aggregate_gradients(self):
        # Aggregate gradients from selected clients
        all_gradients = np.array([grads for grads in self.gradients.values()])
        return np.mean(all_gradients, axis=0)

    def train(self, rounds, local_epochs, batch_size):
        for t in range(rounds):
            selected_clients = self.select_clients(len(self.clients) // 2)
            self.gradients = {}

            for client in selected_clients:
                client.train(local_epochs, batch_size)
                self.gradients[client.id] = client.get_gradients()

            shapley_values = self.compute_shapley_values()
            self.update_client_weights(shapley_values)

            aggregated_gradients = self.aggregate_gradients()
            self.global_model.update_weights(aggregated_gradients)