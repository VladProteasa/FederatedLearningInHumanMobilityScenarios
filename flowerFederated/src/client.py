import os
import sys
import timeit
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
from matplotlib.cbook import ls_mapper
import numpy as np
import torch

from GRUmodel import GRUModel
from LSTMmodel import LSTMModel
from model import RNNModel
from loader import Loader
from dotenv import dotenv_values
import humanMobility
import ast

USE_FEDBN: bool = False

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Flower Client
class FedClient(fl.client.NumPyClient):

    def __init__(
        self,
        model: torch.nn.Module,
        train,
        validate,
        test,
        client_id,
        location_dict
    ) -> None:
        self.model = model
        self.train = train
        self.validate = validate
        self.test = test
        self.client_id = client_id
        self.location_dict = location_dict

    def get_parameters(self) -> List[np.ndarray]:
        self.model.train()
        if USE_FEDBN:
            # Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            # Return model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        if USE_FEDBN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        humanMobility.train(self.model, self.train, epochs=4, device=DEVICE, client_id=self.client_id)
        return self.get_parameters(), len(self.train), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        print(config)
        self.set_parameters(parameters)
        loss, acc1, acc5, acc10 = humanMobility.test(self.model, self.validate, device=DEVICE, client_id=self.client_id, location_dict=self.location_dict, step=bool(config["val_steps"]))
        return float(loss), len(self.validate), {"acc1": acc1, "acc5": acc5, "acc10": acc10}


def start_client(id: int):

    # Load data
    print(id)
    f = open(f'user_data/user_{id}.txt', 'r')
    a = f.read()
    train_batches = ast.literal_eval(a)
    f.close()

    f = open(f'validate_data/user_{id}.txt', 'r')
    a = f.read()
    validate_batches = ast.literal_eval(a)
    f.close()

    f = open(f'test_data/user_{id}.txt', 'r')
    a = f.read()
    test_batches = ast.literal_eval(a)
    f.close()

    f = open(f'location_dict.txt', 'r')
    location_dict = ast.literal_eval(f.read())
    f.close()

    # # Load model
    input_dim = 6
    hidden_dim = 32
    layer_dim = 3
    output_dim = 2
    dropout_prob = 0.0
    model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob).to(DEVICE).train()

    client = FedClient(model, train_batches, validate_batches, test_batches, id, location_dict)
    fl.client.start_numpy_client("localhost:8080", client)

if __name__ == "__main__":
    start_client(int(sys.argv[1]))