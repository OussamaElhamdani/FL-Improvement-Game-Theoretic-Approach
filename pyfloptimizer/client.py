import numpy as np

class Client:
    def __init__(self, id, data, model, learning_rate):
        self.id = id
        self.data = data
        self.model = model
        self.learning_rate = learning_rate

    def train(self, epochs, batch_size):
        # Perform local training
        for epoch in range(epochs):
            batches = self._get_batches(batch_size)
            for batch in batches:
                gradients = self._compute_gradients(batch)
                self.model.update_weights(gradients, self.learning_rate)

    def _get_batches(self, batch_size):
        # Split data into batches
        np.random.shuffle(self.data)
        return [self.data[i:i + batch_size] for i in range(0, len(self.data), batch_size)]

    def _compute_gradients(self, batch):
        # Compute gradients for a batch
        X, y = batch[:, :-1], batch[:, -1]
        return self.model.compute_gradients(X, y)

    def get_model(self):
        return self.model

    def get_gradients(self):
        return self.model.get_gradients()