import flwr as fl

class GameOfGradientsClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, val_data):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.train_data[0], self.train_data[1], epochs=config["epochs"], batch_size=config["batch_size"])
        sv = self.model.evaluate(self.val_data[0], self.val_data[1])[1]
        return self.model.get_weights(), len(self.train_data[0]), {"sv": sv}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.val_data[0], self.val_data[1])
        return loss, len(self.val_data[0]), {"accuracy": accuracy}

def get_client(model, train_data, val_data):
    return GameOfGradientsClient(model, train_data, val_data)
