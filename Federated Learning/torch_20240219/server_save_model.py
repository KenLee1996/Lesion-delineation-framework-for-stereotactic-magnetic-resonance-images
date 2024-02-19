# +
import os
import flwr as fl
from functools import reduce
import numpy as np
import torch
from torch import optim
from models import DP_resUNet
from collections import OrderedDict
from data_preprocess_and_loader import Dataset, Dataset_val
from losses_unet3d import DiceLoss, GeneralizedDiceLoss, compute_per_channel_dice
#from train_and_val import train, validation
from flwr.common import NDArray, NDArrays
from typing import Dict, Optional, Tuple, List, Union

os.environ['CUDA_VISIBLE_DEVICES'] = ' '

lr = 1e-4
local_epoch_num = 1
batch_size = 1
vbatch_size = 1
biparametric = True
base_path = 'directory to save model'
model_folder = 'model_folder_name'
spath = os.path.join(base_path, model_folder)
os.makedirs(spath, exist_ok=True)

if biparametric:
    input_channel_num = 2
else:
    input_channel_num = 1

# model
model = DP_resUNet(img_channels = input_channel_num, n_classes = 1)
optimizer = optim.Adam(model.parameters(), lr=lr)
model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

# +
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Union[bool, bytes, float, int, str]]]:
        
        # Call aggregate_fit from base class (FedAvg) to aggregate metrics
        _, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
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
        
        parameters_aggregated = fl.common.ndarrays_to_parameters(weights_prime)    
        
        if parameters_aggregated is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(parameters_aggregated)
            print(f"Saving round {server_round} aggregated_weights...")            
            params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            fname = spath + '/global_round_' + str(server_round) + '.tar'
            torch.save(model.state_dict(), fname)
            
        return parameters_aggregated, aggregated_metrics

strategy = SaveModelStrategy(
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
)
fl.server.start_server(server_address='120.126.105.200:8080',
                       config=fl.server.ServerConfig(num_rounds=10),
                       strategy=strategy)
