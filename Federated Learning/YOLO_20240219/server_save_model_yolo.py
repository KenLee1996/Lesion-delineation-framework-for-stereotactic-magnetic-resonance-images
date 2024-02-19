# +
import os
import flwr as fl
from ultralytics import YOLO
from functools import reduce
import numpy as np
import torch
from torch import optim
from collections import OrderedDict
from flwr.common import NDArray, NDArrays
from typing import Dict, Optional, Tuple, List, Union

os.environ['CUDA_VISIBLE_DEVICES'] = ' '

model = YOLO('yolov8n.yaml')
#model.reset_weights()

# +
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Union[bool, bytes, float, int, str]]]:
        
        # Call aggregate_fit from base class (FedAvg) to aggregate metrics
        #_, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Convert results
        weights_results = [
            fl.common.parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]
        
        # Create a list of weights
        weighted_weights = [
            [layer for layer in weights] for weights in weights_results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / len(results)
            for layer_updates in zip(*weighted_weights)
        ]
        #print(weights_prime)
        parameters_aggregated = fl.common.ndarrays_to_parameters(weights_prime)
        
        if parameters_aggregated is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(parameters_aggregated)
            print(f"Saving round {server_round} aggregated_weights...")
            params_dict = zip(model.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            os.makedirs('./global_weights', exist_ok=True)
            fname = './global_weights/global_round_'+ str(server_round) +'.pt'
            torch.save(state_dict, fname)
            #model.model.load_state_dict(state_dict, strict=True)
            #torch.save(model, fname)
        return parameters_aggregated, {}

strategy = SaveModelStrategy(
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    #initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
)
fl.server.start_server(server_address='120.126.105.200:8080', config=fl.server.ServerConfig(num_rounds=10), strategy=strategy)
