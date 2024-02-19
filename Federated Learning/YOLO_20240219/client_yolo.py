import os
import yaml
import copy
import torch
import glob
import flwr as fl
import numpy as np
from ultralytics import YOLO
import multiprocessing as mp
from ultralytics.nn.tasks import attempt_load_weights
from collections import OrderedDict
from flwr.common import NDArray, NDArrays
from typing import Dict, Optional, Tuple, List, Union

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    #please make sure delete or rename the folder 'runs' that contains the last training progress before every FL experiment
    mp.set_start_method('spawn')

    local_epoch_num = 5
    
    data_dir = 'C:/Users/user/FLcode/FL_yolo_pytorch/client'    
    
    model = YOLO('yolov8n.yaml').load('yolov8m.pt')
    
    class Client(fl.client.NumPyClient):        
        def get_parameters(self, config):
            if os.path.isfile(data_dir + '/runs/detect/train/weights/last.pt'):
                run_list = glob.glob(os.path.join(data_dir, 'runs/detect/train*'))
                run_num = len(run_list)
                print(run_num)
                
                if run_num == 1:
                    weight_dir = data_dir + '/runs/detect/train/weights/last.pt'
                elif run_num > 1:
                    weight_dir = data_dir + '/runs/detect/train'+ str(run_num) +'/weights/last.pt'
                    
                print('get_parameters', weight_dir)
                tmp_model = attempt_load_weights(weight_dir)
                tmp_weight = copy.deepcopy(tmp_model.state_dict())
            else:
                tmp_model = attempt_load_weights('yolov8m.pt')
                tmp_weight = copy.deepcopy(tmp_model.state_dict())            
            return [val.cpu().numpy() for _, val in tmp_weight.items()]

        def set_parameters(self, parameters):
            run_list = glob.glob(os.path.join(data_dir, 'runs/detect/train*'))
            run_num = len(run_list)
            print(run_num)
            
            if run_num == 1:
                weight_dir = data_dir + '/runs/detect/train/weights/last.pt'
            elif run_num > 1:
                weight_dir = data_dir + '/runs/detect/train'+ str(run_num) +'/weights/last.pt'
                
            params_dict = zip(model.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            
            print('set_parameters', weight_dir)
            model.load(weight_dir)
            model.model.load_state_dict(copy.deepcopy(state_dict), strict=True)
            model.ckpt['model'] = copy.deepcopy(model.model)
            #return model

        def fit(self, parameters, config):
            if os.path.isfile(data_dir + '/runs/detect/train/weights/last.pt'):
                run_list = glob.glob(os.path.join(data_dir, 'runs/detect/train*'))
                run_num = len(run_list)
                print(run_num)

                if run_num == 1:
                    weight_dir = data_dir + '/runs/detect/train/weights/last.pt'
                elif run_num > 1:
                    weight_dir = data_dir + '/runs/detect/train'+ str(run_num) +'/weights/last.pt'
                    
                print(weight_dir)
                self.set_parameters(parameters)
                model.train(data=os.path.join(data_dir, 'stas-tcga', 'data.yaml'),
                            epochs=local_epoch_num,
                            imgsz=640,
                            lr0=1e-2,
                            workers=4,
                            #fliplr=0,
                            #translate=0,
                            #scale=0,
                            #augment=True,
                            resume=True,
                            optimizer='Adam')                
            else:
                model.train(data=os.path.join(data_dir, 'stas-tcga', 'data.yaml'),
                            epochs=local_epoch_num,
                            imgsz=640,
                            lr0=1e-2,
                            workers=4,
                            #fliplr=0,
                            #translate=0,
                            #scale=0,
                            #augment=True,
                            optimizer='Adam')
            return self.get_parameters(config={}), len(os.listdir(os.path.join(data_dir, 'stas-tcga', 'images', 'train'))), {}

        def evaluate(self, parameters, config):
            #self.set_parameters(parameters)                        
            #metrics = model.val()
            return float(0), len(os.listdir(os.path.join(data_dir, 'stas-tcga', 'images', 'val'))), {}

    fl.client.start_numpy_client(server_address='120.126.105.200:8080', client=Client())

if __name__ == '__main__':
    main()