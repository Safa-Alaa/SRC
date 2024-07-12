
"""
The code incorporates elements from the following GitHub repositories. We express our gratitude to the authors, acknowledge their contributions, and give them the appropriate credit for their open-source code.

https://github.com/xudejing/video-clip-order-prediction
https://github.com/xudejing/video-clip-order-prediction/blob/master/train_classify.py
"""

"""Train 3D ConvNets to action classification."""
import os
import argparse
import time
import random
import copy
import numpy as np
import pandas as pd
import torch
import torchvision
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim

import tqdm
from tqdm import tqdm
from PIL import ImageOps, Image, ImageFilter

from model_saver import Model_Saver
from model_loader import Model_Loader
from logger import Logger


import augmentations as A
import transforms as T

from r3d import r3d_18
from rs_network import RSNetwork
from r_s_ucf101 import RSUCF101Dataset




    
def train_amp_multi(args, model, criterion, optimizer, device, train_dataloader, writer, epoch, scheduler, scaler):
    torch.set_grad_enabled(True)
    model.train()

    epoch_running_loss=0.0
    epoch_running_corrects1=0
    epoch_running_corrects2=0
    
    running_loss = 0.0
    correct1 = 0
    correct2 = 0
    

    i, train_bar = 1, tqdm(train_dataloader)
    for tuple_clips, target1, targets  in train_bar:

        

    
        inputs = tuple_clips.to(device)
        if args.pretext_mode == 'rs_rotation_multi':
            targets1 = target1.to(device)
            
            targets2 = targets[0].to(device)
            targets3 = targets[1].to(device)
            targets4 = targets[2].to(device)
            targets5 = targets[3].to(device)
        elif args.pretext_mode == 'rotation_sorting':
            targets1 = target1.to(device)
        elif args.pretext_mode == 'rotation':
            targets1 = target1.to(device)
       
            

        

        
        # zero the parameter gradients
        if args.grad_accum==False:
            optimizer.zero_grad(set_to_none=True)
        
        # forward and backward
        with torch.cuda.amp.autocast(enabled=True):
            if args.pretext_mode == 'rs_rotation_multi':   
                outputs1, outputs2, outputs3, outputs4, outputs5  = model(inputs) # return logits here              
                ##assert outputs1.dtype is torch.float16 and outputs2.dtype is torch.float16
            elif args.pretext_mode == 'rotation_sorting':
                outputs1 = model(inputs) # return logits here

            elif args.pretext_mode == 'rotation':
                outputs1 = model(inputs) # return logits here
                
                
            if args.pretext_mode == 'rs_rotation_multi':                         # Two Heads Multi Task
                loss1 = criterion(outputs1, targets1)
                loss2 = criterion(outputs2, targets2)
                loss3 = criterion(outputs3, targets3)
                loss4 = criterion(outputs4, targets4)
                loss5 = criterion(outputs5, targets5)
                
                loss  = loss1 + (loss2 + loss3 + loss4 + loss5)/4
                if args.grad_accum==True:
                    loss = loss / args.accum_iter
                    
            elif args.pretext_mode == 'rotation_sorting':                # 
                loss  = criterion(outputs1, targets1)
                if args.grad_accum==True:
                    loss = loss / args.accum_iter
                    
            elif args.pretext_mode == 'rotation':                        # 
                loss  = criterion(outputs1, targets1)
                if args.grad_accum==True:
                    loss = loss / args.accum_iter
                    
            
                
        
        scaler.scale(loss).backward()
        if args.grad_accum==False:
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_accum==True:
                if ((i + 1) % args.accum_iter == 0) or (i + 1 == len(train_dataloader)):
                    scaler.step(optimizer)
                    scaler.update()
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    if args.sch_mode=='one_cycle' or args.sch_mode=='Cosine' :
                        scheduler.step()

        

        # compute loss and acc
        # running_loss += loss.item()  # I think its not accurate
        running_loss += loss.item()* inputs.size(0)
        
        if args.pretext_mode == 'rs_rotation_multi':
            pts1 = torch.argmax(outputs1, dim=1)
            correct1 += torch.sum(targets1 == pts1).item()

      
            pts2 = torch.argmax(outputs2, dim=1)
            correct2 += torch.sum(targets2 == pts2).item()
            
            epoch_running_corrects1+=torch.sum(targets1 == pts1).item()
            epoch_running_corrects2+=torch.sum(targets2 == pts2).item()
            
            pts3 = torch.argmax(outputs3, dim=1)
            correct2 += torch.sum(targets3 == pts3).item()
            epoch_running_corrects2+=torch.sum(targets3 == pts3).item()
            
            pts4 = torch.argmax(outputs4, dim=1)
            correct2 += torch.sum(targets4 == pts4).item()
            epoch_running_corrects2+=torch.sum(targets4 == pts4).item()
            
            pts5 = torch.argmax(outputs5, dim=1)
            correct2 += torch.sum(targets5 == pts5).item()
            epoch_running_corrects2+=torch.sum(targets5 == pts5).item()
            
            
        elif args.pretext_mode == 'rotation_sorting':
            pts1 = torch.argmax(outputs1, dim=1)
            correct1 += torch.sum(targets1 == pts1).item()
            epoch_running_corrects1+=torch.sum(targets1 == pts1).item()
            
        elif args.pretext_mode == 'rotation':
            pts1 = torch.argmax(outputs1, dim=1)
            correct1 += torch.sum(targets1 == pts1).item()
            epoch_running_corrects1+=torch.sum(targets1 == pts1).item()            
            
            
        
        # stats for the complete epoch
        epoch_running_loss += loss.item() * inputs.size(0)
        
        
        if args.sch_mode=='one_cycle':
            scheduler.step()
        
        # print statistics and write summary every N batch
        if i % args.pf == 0:
            # avg_loss = running_loss / pf # I think its not accurate
            avg_loss = running_loss / (args.pf * args.bs)
            avg_acc1 = correct1 / (args.pf * args.bs)
            avg_acc2 = correct2 / (args.pf * args.bs)
            #print('[TRAIN] epoch-{}, batch-{}, loss: {:.3f}, acc: {:.3f}'.format(epoch, i, avg_loss, avg_acc))
            print('\n')
            train_bar.set_description('[TRAIN]: [{}/{}], lr: {:.10f}, loss: {:.4f}, acc1: {:.4f}, acc2: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'], avg_loss, avg_acc1, avg_acc2))
          

            running_loss = 0.0
            correct1 = 0
            correct2 = 0
        i += 1
        torch.cuda.empty_cache()
            
    epoch_loss = epoch_running_loss / len(train_dataloader.dataset)
    epoch_acc1  = epoch_running_corrects1 / len(train_dataloader.dataset)
    epoch_acc2  = epoch_running_corrects2 / len(train_dataloader.dataset)
   

    return epoch_acc1, epoch_acc2, epoch_loss, 0.0 #epoch_targets



def test_backup_multi(model, criterion, device, test_dataloader):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct1 = 0
    correct2 = 0
    
    #targets_counts=[0 for x in (classes)]
    i, test_bar = 1, tqdm(test_dataloader)
    for tuple_clips, target1, targets  in test_bar:
    #for i, data in (enumerate(test_dataloader, 1)):
        # get inputs
        #clips, idxs = data
        inputs = tuple_clips.to(device)
        
        if args.pretext_mode == 'rs_rotation_multi':
            targets1 = target1.to(device)
            targets2 = targets[0].to(device)
            targets3 = targets[1].to(device)
            targets4 = targets[2].to(device)
            targets5 = targets[3].to(device)
        elif args.pretext_mode == 'rotation_sorting':
            targets1 = target1.to(device)
            
        elif args.pretext_mode == 'rotation':
            targets1 = target1.to(device)            
        
        
        #targets_counts=targets_info(targets, targets_counts, None, None, classes)
        # forward
        
        if args.pretext_mode == 'rs_rotation_multi': 
            with torch.cuda.amp.autocast(enabled=True):                                       # Two Heads
                outputs1, outputs2, outputs3, outputs4, outputs5  = model(inputs)
                loss1 = criterion(outputs1, targets1)
                loss2 = criterion(outputs2, targets2)
                loss3 = criterion(outputs3, targets3)
                loss4 = criterion(outputs4, targets4)
                loss5 = criterion(outputs5, targets5)
            
                loss  = loss1 + (loss2 + loss3 + loss4 + loss5)/4
                # compute loss and acc
                # total_loss += loss.item() # I think this is not accurate
                total_loss += loss.item()* inputs.size(0)
                pts1 = torch.argmax(outputs1, dim=1)
                correct1 += torch.sum(targets1 == pts1).item()
        
                pts2 = torch.argmax(outputs2, dim=1)
                correct2 += torch.sum(targets2 == pts2).item()
            
                pts3 = torch.argmax(outputs3, dim=1)
                correct2 += torch.sum(targets3 == pts3).item()
            
            
                pts4 = torch.argmax(outputs4, dim=1)
                correct2 += torch.sum(targets4 == pts4).item()
           
            
                pts5 = torch.argmax(outputs5, dim=1)
                correct2 += torch.sum(targets5 == pts5).item()
            
            
        elif args.pretext_mode == 'rotation_sorting':
            with torch.cuda.amp.autocast(enabled=True):                                    # Single Head
                outputs1 = model(inputs)
                loss = criterion(outputs1, targets1)
            
                # compute loss and acc
                # total_loss += loss.item() # I think this is not accurate
                total_loss += loss.item()* inputs.size(0)
                pts1 = torch.argmax(outputs1, dim=1)
                correct1 += torch.sum(targets1 == pts1).item()
            
        elif args.pretext_mode == 'rotation':                                              # Single Head
            with torch.cuda.amp.autocast(enabled=True):
                outputs1 = model(inputs)
                loss = criterion(outputs1, targets1)
            
                # compute loss and acc
                # total_loss += loss.item() # I think this is not accurate
                total_loss += loss.item()* inputs.size(0)
                pts1 = torch.argmax(outputs1, dim=1)
                correct1 += torch.sum(targets1 == pts1).item()
        
        
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
        test_bar.set_description('[TEST]: loss: {:.4f}, corrects1 {}, acc1: {:.4f}, corrects2 {}, acc2: {:.4f}'.format( total_loss/(i*args.bs), correct1,  correct1/(i*args.bs), correct2,  correct2/(i*args.bs)))
        i += 1
        torch.cuda.empty_cache()
        
    #avg_loss = total_loss / len(test_dataloader)   # I think this is not accurate
    avg_loss = total_loss / len(test_dataloader.dataset)
    avg_acc1 = correct1 / len(test_dataloader.dataset)
    avg_acc2 = correct2 / len(test_dataloader.dataset)
    
    print('[TEST] loss: {:.3f}, acc1: {:.3f}, acc2: {:.3f}'.format(avg_loss, avg_acc1, avg_acc2))
    return avg_acc1, avg_acc2, avg_loss, 0.0 #, avg_targets 


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    wd = args.wd

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['initial_lr'] = lr
        param_group['weight_decay'] = wd 
        #param_group['momentum'] = args.momentum
       

def parse_args():
    


    

    experiment='rs_rotation_multi_resizing_128_171_[0-90-180-270]_16' 
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--model', type=str, default='r3d', help='c3d/r3d/r21d/s3dg/s3d/r2plus1d_18/resnet_18/sspl_network/multi_network')                              #s3dg to be implemented
    parser.add_argument('--dataset', type=str, default='rs_ucf101', help='ucf101/hmdb51/sspl_ucf/multi_ucf')
    parser.add_argument('--split', type=str, default='1', help='dataset split 1,2,3')
    parser.add_argument('--cl', type=int, default=16, help='clip length')                                         
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')  #1e-3
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')  #5e-4 #1e-5 #1e-4 #1e-3 #1e-2  #0 
    

    parser.add_argument('--epochs', type=int, default=430, help='number of total epochs to run')
    parser.add_argument('--start_epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=16, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=9, help='number of data loading workers')
    parser.add_argument('--pf', type=int, default=100, help='print frequency every batch')
    #parser.add_argument('--seed', type=int, default=632, help='seed for initializing training.')
    parser.add_argument('--sch_mode', type=str, default='reduce_lr', help='one_cycle/reduce_lr/None.')                 #one_cycle schedual to be implemented
    parser.add_argument('--log', type=str, help='log directory')
    parser.add_argument('--use_amp', type=bool, default=True, help='True/False.')
    parser.add_argument('--run_testing', type=bool, default=True, help='True/False.')
    
    parser.add_argument('--exp_name', type=str, default=experiment, help='experiment name.')   
    parser.add_argument('--init_mode', type=str, default='kaiming', help='kaiming/None') 
    #https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/02/19/gradient-accumulation.html
    parser.add_argument('--grad_accum', type=bool, default=False, help='')
    parser.add_argument('--accum_iter', type=int, default = 20, help='1...')   #The total batchs has to be divisible by this number
    
      
    parser.add_argument('--sampling_mode', type=str, default='fixed_skip', help='random_skip/fixed_skip/sequential.')
    parser.add_argument('--skip_rate', type=int, default=4, help='1..4.') 
    parser.add_argument('--tuple_len', type=int, default=4, help='the number of clips sampled in each tuple.')
    parser.add_argument('--pretext_mode', type=str, default='rs_rotation_multi', help='rotation_sorting/rotation/rs_rotation_multi')
    parser.add_argument('--pretext_videos', type=str, default='single', help='single/multi')
    parser.add_argument('--pretext_classes', type=int, default=16, help='')
    
    
       
    
    
    args = parser.parse_args()
    
    

   
    
 
    return args


def main(args):
    print_sep = '============================================================='
    
    
    ################################################################################
    
    current_dir=os.getcwd().replace('D:','')
    args.log=os.path.join(current_dir, 'experiments', args.exp_name)
    
    #ucf_dir=r'\Users\......'
    ucf_dir=r'C:\Users\.....'
    
   

    ###torch.backends.cudnn.benchmark = True
    # Force the pytorch to create context on the specific device 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")



    ########### model ##############
    if args.dataset == 'ucf101':
        class_num = 101
    elif args.dataset == 'hmdb51':
        class_num = 51
    elif args.dataset == 'rs_ucf101':
        class_num = args.pretext_classes
        print('Number Of Targets:',class_num)

    if  args.model == 'c3d':
        print(class_num)
        #model = C3D(with_classifier=True, num_classes=class_num).to(device)
    elif  args.model == 'r3d':
        print(class_num)
        base_model =  r3d_18(pretrained = False, progress = False)
        model = RSNetwork(base_network=base_model, feature_size=512*2, tuple_len=args.tuple_len, class_num=args.pretext_classes, args=args).to(device)
        
        print(model)
        

        
        
        
        
    def kaiming_init(m):
        
        
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    if (args.init_mode=='kaiming')and((args.model=='s3d') ): #or (args.model == 'sspl_network')
        print('applying kaiming normal init...')
        model.apply(kaiming_init)
      
        
      
    print (print_sep)    
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print ('Total Params', total_params, ':: ::', 'Trainable',total_trainable_params)
    print (print_sep) 
    

    if  args.mode == 'train':  ########### Train #############
        
    
    
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
        ### loss funciton, optimizer and scheduler ###
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        #torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        #optimizer = optim.AdamW(model.parameters())
        
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
      
        # patience was 50 I changed it to 20
        # patience was 20 I changed it to 10
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5, patience=10, factor=0.1)

        
        # resume training # check if there was a previously saved checkpoint
        # resume training # check if there was a previously saved checkpoint
        loader = Model_Loader(args, model, optimizer)
        model, optimizer, epoch_resume, starting_epoch, best_video_acc, best_video_loss = loader.load()
        # resume training # check if there was a previously saved checkpoint
        
       
            
        #(0.4218, 0.4025, 0.3738) - (0.2337, 0.2267, 0.2240)
        
        # The RSL needs  NON Consistent transforms WHILE SSL needs Consistent transforms
           
        train_transforms =torchvision.transforms.Compose([T.ToTensorVideo(),

                                               T.Resize((128, 171)),
                                               A.RandomSizedCrop(112, interpolation=Image.BICUBIC, consistent=False, p=1.0, seq_len=args.cl, bottom_area=0.2),
                                               A.RandomHorizontalFlip(consistent=False, command=None, seq_len=args.cl),
                                               transforms.RandomApply([A.RandomGray(consistent=False, p=0.5, dynamic=False, seq_len=args.cl)], p=0.7),
                                               transforms.RandomApply([A.ColorJitter(brightness=0.3, contrast=1, saturation=0.3, hue=0.3, consistent=False, p=1.0, seq_len=args.cl)], p=0.7),
                                               transforms.RandomApply([A.GaussianBlur(sigma=[.1, 2.], seq_len=args.cl)],p=0.7),
                                               
                                               
                                                          
        ])
        
        val_transforms = torchvision.transforms.Compose([T.ToTensorVideo(),

                                               T.Resize((128, 171)),
                                               A.RandomSizedCrop(112, interpolation=Image.BICUBIC, consistent=False, p=1.0, seq_len=args.cl, bottom_area=0.2),
                                               A.RandomHorizontalFlip(consistent=False, command=None, seq_len=args.cl),
                                               transforms.RandomApply([A.RandomGray(consistent=False, p=0.5, dynamic=False, seq_len=args.cl)], p=0.7),
                                               transforms.RandomApply([A.ColorJitter(brightness=0.3, contrast=1, saturation=0.3, hue=0.3, consistent=False, p=1.0, seq_len=args.cl)], p=0.7),
                                               transforms.RandomApply([A.GaussianBlur(sigma=[.1, 2.], seq_len=args.cl)],p=0.7),
                                               
        ])

        
        
        if args.dataset == 'dsucf101':
            print (print_sep) 

        if args.dataset == 'sspl_ucf':
            print (print_sep) 

        if args.dataset == 'rs_ucf101':
            train_dataset = RSUCF101Dataset(args, ucf_dir, True, train_transforms)
            val_dataset = RSUCF101Dataset(args, ucf_dir, False, val_transforms)
        
            


        print('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)


        total_batches=0
        if (epoch_resume>1):
            total_batches=(epoch_resume-1)*len(train_dataloader)
        print (print_sep)     
        print('Completed Epochs     :: ', epoch_resume-1)
        print('len(train_dataloader)::', len(train_dataloader))    
        print('Scheduler Batch      ::', total_batches)
        print (print_sep) 
        
        if (args.sch_mode=='one_cycle'):
            if (total_batches!=0):
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_dataloader), epochs=args.epochs, last_epoch= total_batches, verbose=False)
            else:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_dataloader), epochs=args.epochs, last_epoch=-1, verbose=False)
        else:
            if (args.sch_mode=='reduce_lr'):
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-6, patience=10, factor=0.1)
            else:
                if (args.sch_mode=='None'):
                    scheduler=None
                    
        
        
        ###adjust_learning_rate(optimizer, None, args)
        print(print_sep)
        print ('Using Scheduler ::', scheduler)
        print ('Using Optim     ::', optimizer)
        print(print_sep)
        
        writer=None
        for epoch in range(starting_epoch, args.start_epoch + args.epochs):
            print('Epoch::',epoch)
            
                            
            
            time_start = time.time()
            if (args.use_amp == True):
                if (args.pretext_mode == 'rs_rotation_multi'):                              # Multi Loss
                    print()
                    train_acc1, train_acc2, train_loss, epoch_targets=train_amp_multi(args, model, criterion, optimizer, device, train_dataloader, writer, epoch, scheduler, scaler)
                    train_acc = (train_acc1 + train_acc2)/2
                    
                    #train_acc1, train_acc2, train_loss, epoch_targets=train_amp(args, model, criterion, optimizer, device, train_dataloader, writer, epoch, classes, scheduler, scaler)
                    #train_acc = (train_acc1 + train_acc2)/2
                elif (args.pretext_mode == 'rotation_sorting'):                       # Single Loss
                    print()
                    train_acc1, train_acc2, train_loss, epoch_targets=train_amp_multi(args, model, criterion, optimizer, device, train_dataloader, writer, epoch, scheduler, scaler)
                    train_acc = train_acc1
                elif (args.pretext_mode == 'rotation'):
                    train_acc1, train_acc2, train_loss, epoch_targets=train_amp_multi(args, model, criterion, optimizer, device, train_dataloader, writer, epoch, scheduler, scaler)
                    train_acc = train_acc1
               
                
                
            else:
               print('Not Supported') 
               #train_acc, train_loss, epoch_targets=train(args, model, criterion, optimizer, device, train_dataloader, writer, epoch, classes, scheduler) 
           
            print('Epoch time: {:.2f} s.'.format(time.time() - time_start))
            #val_acc, val_loss = validate(model, criterion, device, val_dataloader, writer, epoch)
            if args.run_testing == True and epoch % 1 == 0 :
                if (args.pretext_mode == 'rs_rotation_multi'):
                    print()
                    val_acc1, val_acc2, val_loss, test_targets = test_backup_multi(model, criterion, device, val_dataloader)
                    val_acc = (val_acc1 + val_acc2)/2
                    #val_acc1, val_acc2, val_loss, test_targets = test_backup(model, criterion, device, val_dataloader, classes)
                    #val_acc = (val_acc1 + val_acc2)/2
                elif (args.pretext_mode == 'rotation_sorting') or (args.pretext_mode =='rotation') or (args.pretext_mode ==''):
                    print()
                    val_acc1, val_acc2, val_loss, test_targets = test_backup_multi(model, criterion, device, val_dataloader)
                    val_acc = val_acc1 
                    
            else:
                val_acc=0.0
                val_acc1=0.0
                val_acc2=0.0
                val_loss=0.0
                test_targets=0.0
                
                
            
            if (args.sch_mode=='reduce_lr'):
                scheduler.step(train_loss)
           
            
            # save model every 20 epoches
            if epoch % 1 == 0:

                saver = Model_Saver(args, model, optimizer, epoch, train_acc, train_loss, batch_count=None, best_mode=1)
                saver.save()
                
                logger = Logger(args, epoch, train_acc1, train_acc2, val_acc1, val_acc2, None, train_loss, val_loss, None, None, 'SSPL')
                logger.acc_log()
                logger.loss_log()

                
            # save model for the best val
            if val_acc > best_video_acc:

                
                saver = Model_Saver(args, model, optimizer, epoch, val_acc, val_loss, batch_count=None, best_mode=3)
                saver.save()
                best_video_acc = val_acc 
                


    elif args.mode == 'test':  ########### Test #############
        #
        #best_test_ckpt=os.path.join(args.log, ('Best-Video-'+args.exp_name))+'.tar'
        best_test_ckpt=os.path.join(args.log,'rotation_4.tar')
        checkpoint=torch.load(best_test_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Loading The Best Video Model From Epoch::', checkpoint['epoch'], ' With Val Acc:',checkpoint['acc'])
        
        
        test_transforms = torchvision.transforms.Compose([T.ToTensorVideo(),

                                               A.RandomSizedCrop(112, interpolation=Image.BICUBIC, consistent=False, p=1.0, seq_len=args.cl, bottom_area=0.2),
                                               A.RandomHorizontalFlip(consistent=False, command=None, seq_len=args.cl),
                                               transforms.RandomApply([A.RandomGray(consistent=False, p=0.5, dynamic=False, seq_len=args.cl)], p=0.7),
                                               transforms.RandomApply([A.ColorJitter(brightness=0.3, contrast=1, saturation=0.3, hue=0.3, consistent=False, p=1.0, seq_len=args.cl)], p=0.7),
                                               transforms.RandomApply([A.GaussianBlur(sigma=[.1, 2.], seq_len=args.cl)],p=0.7),
        ])

        

            
        if args.dataset == 'rs_ucf101':
            print()
            test_dataset = RSUCF101Dataset(args, ucf_dir, False, test_transforms)
        


        if (args.cl==64):
            effective_bs = 1
        else:
            effective_bs = args.bs
            
        test_dataloader = DataLoader(test_dataset, batch_size=effective_bs, shuffle=False, num_workers=args.workers, pin_memory=True)
        print('TEST video number: {}.'.format(len(test_dataset)))
        criterion = nn.CrossEntropyLoss()
        print(' model.fully_connected.weight.dtype', model.fully_connected[0].weight.dtype)
        test_backup_multi(model, criterion, device, test_dataloader)
        
   

        
if __name__ == '__main__':
    args = parse_args()
    print(vars(args))
    
    main(args)