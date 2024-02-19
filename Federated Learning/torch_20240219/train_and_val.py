# -*- coding: utf-8 -*-
import time
import torch
import numpy as np
import scipy.io as sio
import os
import time
from torch import optim
from losses_unet3d import DiceLoss, GeneralizedDiceLoss, compute_per_channel_dice


#訓練模式
def train(train_loader,model,epochs,path):

    f = open(path + '/loss_curve.txt', 'a')    
    model.cuda()
    model.train() # Turn on the train mode

    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # loss function
    criterion_DICE = DiceLoss()
    criterion_GDL = GeneralizedDiceLoss()
    loss_fun = (criterion_DICE,criterion_GDL,compute_per_channel_dice)

    total_loss = 0.
    total_loss2 = 0.
    total_loss3 = 0.

    for epoch in range(1,epochs+1):
        epoch_start_time = time.time()
        start_time = time.time()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            #print('pass')
            # reset gradient of the optimizer
            optimizer.zero_grad()
            #print('pass')
            # apply GPU setup
            batch_x = batch_x.cuda() # data
            batch_y = batch_y.cuda() # ground truth
            #print('pass')
            # model prediction
            output = model(batch_x)
            #print('pass')
            # loss calculation
            loss_dice = loss_fun[0](output,batch_y)
            loss_gdl = loss_fun[1](output,batch_y)
            dice_score = loss_fun[2](output,batch_y)
            #print('pass')
            # backward and update model weights
            #loss_dice.backward()    
            loss_gdl.backward()    
            optimizer.step()
            #print('pass')
            # print log
            total_loss += loss_dice.item()
            total_loss2 += loss_gdl.item()
            total_loss3 += dice_score.mean().item()
            log_interval = 100

            if (step+1) % log_interval == 0 or step+1 == len(train_loader):
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      ' ms/batch {:5.2f} | '
                      ' loss_dice {:5.2f}  | '
                      ' loss_gdl {:5.2f}  | '
                      ' dice_score {:5.2f}  | '.format(
                          epoch, step+1, len(train_loader),
                          elapsed * 1000,
                          total_loss/(step+1),
                          total_loss2/(step+1),
                          total_loss3/(step+1)))
                f.write('| epoch {:3d} | {:5d}/{:5d} batches | '
                        ' ms/batch {:5.2f} | '
                        ' loss_dice {:5.2f}  | '
                        ' loss_gdl {:5.2f}  | '.format(
                            epoch, step+1, len(train_loader), 
                            elapsed * 1000,  
                            total_loss/(step+1),
                            total_loss2/(step+1),
                            total_loss3/(step+1))+'\r\n')
                start_time = time.time()                   
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time)))
        print('-' * 89)
    del batch_x, batch_y
    torch.cuda.empty_cache()        
    f.close()

# 每個epoch完的測試模式
def validation(val_loader,model,epoch,path):
    f = open(path + '/loss_curve.txt', 'a')
    # loss function
    criterion_DICE = DiceLoss()
    criterion_GDL = GeneralizedDiceLoss()
    loss_fun = (criterion_DICE,criterion_GDL,compute_per_channel_dice)
    total_loss = 0.
    total_loss2 = 0.
    total_loss3 = 0.    
    with torch.no_grad():
        epoch_start_time = time.time()
        start_time = time.time()
        for step, (batch_x, batch_y) in enumerate(val_loader):
            #model.to(torch.device("cuda"))
            model.cuda()
            model.train(False)
            # apply GPU setup
            batch_x = batch_x.cuda() # data
            batch_y = batch_y.cuda() # ground truth
            # model prediction
            output = model(batch_x)
            # loss calculation
            loss_dice = loss_fun[0](output,batch_y)
            loss_gdl = loss_fun[1](output,batch_y)
            dice_score = loss_fun[2](output,batch_y)
            # print log
            total_loss += loss_dice.item()
            total_loss2 += loss_gdl.item()
            total_loss3 += dice_score.mean().item()
            log_interval = 100

            if (step+1) % log_interval == 0 or step+1 == len(val_loader):
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      ' ms/batch {:5.2f} | '
                      ' loss_dice {:5.2f}  | '
                      ' loss_gdl {:5.2f}  | '
                      ' dice_score {:5.2f}  | '.format(
                          epoch, step+1, len(val_loader),
                          elapsed * 1000,
                          total_loss/(step+1),
                          total_loss2/(step+1),
                          total_loss3/(step+1)))
                f.write('| epoch {:3d} | {:5d}/{:5d} batches | '
                        ' ms/batch {:5.2f} | '
                        ' loss_dice {:5.2f}  | '
                        ' loss_gdl {:5.2f}  | '.format(
                            epoch, step+1, len(val_loader), 
                            elapsed * 1000,  
                            total_loss/(step+1),
                            total_loss2/(step+1),
                            total_loss3/(step+1))+'\r\n')
                start_time = time.time()
        print('-' * 89)
        print('| validation time: {:5.2f}s'.format((time.time() - epoch_start_time)))
        print('-' * 89)
        del batch_x, batch_y
        torch.cuda.empty_cache()        
        f.close()            
    return total_loss2