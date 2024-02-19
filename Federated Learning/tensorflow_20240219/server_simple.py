# +
import os
import flwr as fl

os.environ['CUDA_VISIBLE_DEVICES'] = ' '

base_path = 'directory to save model'
model_folder = 'model_folder_name'
os.makedirs(os.path.join(base_path, model_folder), exist_ok=True)

strategy = fl.server.strategy.FedAvg()
fl.server.start_server(server_address="120.126.105.200:8080", 
                       config=fl.server.ServerConfig(num_rounds=3), 
                       strategy=strategy)