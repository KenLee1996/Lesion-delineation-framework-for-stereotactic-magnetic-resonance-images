# +
import flwr as fl

strategy = fl.server.strategy.FedAvg()
fl.server.start_server(server_address='120.126.105.200:8080', 
                       config=fl.server.ServerConfig(num_rounds=10), 
                       strategy=strategy)
