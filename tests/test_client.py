import unittest
from pyfloptimizer.client import get_client

class TestGameOfGradientsClient(unittest.TestCase):
    def test_client_initialization(self):
        model = None  # Replace with actual model
        train_data = (None, None)  # Replace with actual data
        val_data = (None, None)  # Replace with actual data
        client = get_client(model, train_data, val_data)
        self.assertIsNotNone(client)

    def test_client_fit(self):
        model = None  # Replace with actual model
        train_data = (None, None)  # Replace with actual data
        val_data = (None, None)  # Replace with actual data
        client = get_client(model, train_data, val_data)
        parameters = None  # Replace with actual parameters
        config = {"epochs": 1, "batch_size": 32}
        new_params, num_examples, metrics = client.fit(parameters, config)
        self.assertIsNotNone(new_params)
        self.assertIsInstance(num_examples, int)
        self.assertIn("sv", metrics)

if __name__ == '__main__':
    unittest.main()
