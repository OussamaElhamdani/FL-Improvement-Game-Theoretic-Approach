import unittest
from pyfloptimizer.server import get_server

class TestGameOfGradientsServer(unittest.TestCase):
    def test_server_initialization(self):
        server = get_server(alpha=0.5, beta=0.5)
        self.assertIsNotNone(server)

    def test_server_update_ϕ_and_utilities(self):
        server = get_server(alpha=0.5, beta=0.5)
        server.ϕ = {'client_1': 1.0}
        server.utilities = {'client_1': 0.0}
        results = [('client_1', {"sv": 0.5})]
        server.update_ϕ_and_utilities(results)
        self.assertEqual(server.ϕ['client_1'], 0.75)
        self.assertEqual(server.utilities['client_1'], 0.5)

if __name__ == '__main__':
    unittest.main()
