import os
import flwr as fl
import numpy as np
import multiprocessing as mp
import torch
from models import DP_resUNet
from collections import OrderedDict
from data_preprocess_and_loader import Dataset, Dataset_val
from train_and_val import train, validation
#import torch.multiprocessing as mp
#mp.set_start_method('spawn', force=True)

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(DEVICE)

def main():
    mp.set_start_method('spawn')
    train_image_path = 'path that save the training data'
    val_image_path = 'path that save the validation data'

    k = os.listdir(train_image_path)
    path = []
    for j in k[:200]:
        path.append(train_image_path + '/' + j)
    vpath = []
    k = os.listdir(val_image_path)
    for j in k[:50]:
        vpath.append(val_image_path + '/' + j)

    local_epoch_num = 1
    batch_size = 1
    vbatch_size = 1
    biparametric = True
    base_path = 'directory to save model'
    model_folder = 'model_folder_name'
    spath = os.path.join(base_path, model_folder)
    os.makedirs(spath, exist_ok=True)

    train_data = Dataset(path, biparametric=biparametric)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=4,
        shuffle = True)
    val_data = Dataset_val(vpath, biparametric=biparametric)
    val_loader = torch.utils.data.DataLoader(
        val_data,           
        batch_size=vbatch_size,
        num_workers=4,
        shuffle = False)

    if biparametric:
        input_channel_num = 2
    else:
        input_channel_num = 1
    # model
    model = DP_resUNet(img_channels = input_channel_num, n_classes = 1)

    class Client(fl.client.NumPyClient):        
        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in model.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            #print(state_dict)
            model.load_state_dict(state_dict, strict=True)
            return model

        def fit(self, parameters, config):
            #parameters = get_model_params(model)        
            model = self.set_parameters(parameters)
            train(train_loader,model,local_epoch_num,spath)
            fname = spath + '/tvgh_last' + '.tar'
            torch.save(model.state_dict(), fname)
            #para = get_model_params(model)
            return self.get_parameters(config={}), len(path), {}

        def evaluate(self, parameters, config):
            model = self.set_parameters(parameters)        
            vallossnew = validation(val_loader,model,local_epoch_num,spath)
            return float(vallossnew), len(vpath), {}

    fl.client.start_numpy_client(server_address='120.126.105.200:8080', client=Client())

if __name__ == '__main__':
    main()