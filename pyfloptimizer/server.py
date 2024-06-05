import numpy
import flwr as fl
from collections import defaultdict

class GameOfGradientsServer(fl.server.strategy.FedAvg):
    def __init__(self, alpha, beta, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta 
        self.ϕ = defaultdict(lambda: 1.0) # Initialize importance scores
        self.utilities = defaultdict(float)  # To track client utilities

    def initialize_parameters(self, client_manager):
        """Initilize client-specific parameters if necessary."""
        for cid in client_manager.all().keys():
            self.φ[cid] = 1.0 # Initialize importance scores

    def configure_fit(self, rnd, parameters, client_manager):
        """Initialize client-specific parameters if necessary."""
        for cid in client_manager.all().keys():
            self.ϕ[cid] = 1.0  # Initialize importance scores

    def configure_fit(self, rnd, parameters, client_manager):
        """Configure the next round of training."""
        client_ids = self.sample_clients()
        config = {"round": rnd, "epochs": 1, "batch_size": 32}
        return [(client_id, parameters, config) for client_id in client_ids]

    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights, fit_metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            self.update_ϕ_and_utilities(results)
        return aggregated_weights, fit_metrics

    def update_ϕ_and_utilities(self, results):
        for client_id, fit_res in results:
            sv = fit_res.metrics["sv"]
            self.utilities[client_id] += sv  # Update utility
            self.ϕ[client_id] = self.alpha * self.ϕ[client_id] + self.beta * sv

    def sample_clients(self):
        client_ids = list(self.ϕ.keys())
        exp_ϕ = np.exp([self.ϕ[cid] for cid in client_ids])
        P_ϕ = exp_ϕ / np.sum(exp_ϕ)
        return np.random.choice(client_ids, size=min(len(client_ids), 10), p=P_ϕ)  # Sample 10 clients or all clients if fewer

def get_server(alpha, beta):
    return GameOfGradientsServer(alpha, beta)