# +
import os
import numpy as np
from functools import reduce
from tensorflow.keras.optimizers import Adam
from M4modelGnRes import build_model
from losses import dice_coefficient, generalized_dice_loss

import flwr as fl
from typing import Dict, Optional, Tuple, List, Union
from flwr.common import NDArray, NDArrays

os.environ['CUDA_VISIBLE_DEVICES'] = ' '

base_path = 'directory to save model'
model_folder = 'model_folder_name'
os.makedirs(os.path.join(base_path, model_folder), exist_ok=True)

biparametric = True
if biparametric:
    input_channel_num = 2
else:
    input_channel_num = 1
model = build_model(input_shape=(None, None, None, input_channel_num), output_channels=1)
model.compile(optimizer = Adam(), loss = generalized_dice_loss, metrics = [dice_coefficient])

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
            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_weights...")
            model.set_weights(aggregated_ndarrays)
            model.save(os.path.join(base_path, model_folder, 'round_' + str(server_round) + '_global.h5'))
            
        return parameters_aggregated, aggregated_metrics

strategy = SaveModelStrategy(
    initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
)
fl.server.start_server(server_address="120.126.105.200:8080", 
                       config=fl.server.ServerConfig(num_rounds=3), 
                       strategy=strategy)