import ast
from tracemalloc import start
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
import torch

from GRUmodel import GRUModel
from client import FedClient

def evaluate_config(rnd: int):
    print(rnd)
    val_steps = True if rnd == -1 else False
    return {"val_steps": val_steps}
NUM_CLIENTS = 10
strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_eval=0.2,
        min_fit_clients=5,
        min_eval_clients=1,
        on_evaluate_config_fn=evaluate_config,
        min_available_clients=int(NUM_CLIENTS)
)

def start_server():
    fl.server.start_server(
        server_address="localhost:8080",
        config={"num_rounds": 100},
        strategy=strategy,
    )

if __name__ == "__main__":
    start_server()