
"""


The code incorporates elements from the following GitHub repositories. We express our gratitude to the authors, acknowledge their contributions, and give them the appropriate credit for their open-source code.
https://github.com/xudejing/video-clip-order-prediction
https://github.com/xudejing/video-clip-order-prediction/blob/master/datasets/ucf101.py
"""

"""Dataset utils for NN."""
import os
import random
from glob import glob
from pprint import pprint
import uuid
import tempfile
import cv2
import numpy as np
import ffmpeg
import skvideo.io
import pandas as pd
from skvideo.io import ffprobe
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import tqdm
from tqdm import tqdm
import  torchvision
import random
import augmentations as A
import transforms as T
import numpy as np
from itertools import permutations

  

class RSUCF101Dataset(Dataset):
    
    """Rotate and Sort UCF101 dataset for recognition. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        
    """
    def __init__(self, args, root_dir, train=True, transform=None):
        
                           #classes=[0,16,32,64]
        self.args = args        
        self.sampling_mode=args.sampling_mode
        self.skip_rate=args.skip_rate

        self.root_dir = root_dir
        self.clip_len = args.cl
        self.split = args.split
        self.train = train
        self.transform= transform
        self.toPIL = transforms.ToPILImage()
        

        if args.pretext_mode=='rotation_sorting' and args.pretext_videos=='single' and args.tuple_len==4 and args.pretext_classes==4:
            self.sort2degree = {'[0 1 2 3]': (0, 90, 180, 270), '[2 0 1 3]': (90, 180, 0, 270), '[3 1 0 2]': (180, 90, 270, 0), '[3 2 1 0]': (270, 180, 90, 0)}
            self.class_list  = ['[0 1 2 3]', '[2 0 1 3]','[3 1 0 2]','[3 2 1 0]']
            
        elif  args.pretext_mode=='rotation_sorting' and args.pretext_videos=='single' and args.tuple_len==4 and args.pretext_classes==5:
            #1 #9 #16 #24 #17
            self.sort2degree = {'[0 1 2 3]': (0, 90, 180, 270), '[2 0 1 3]': (90, 180, 0, 270), '[3 1 0 2]': (180, 90, 270, 0), '[3 2 1 0]': (270, 180, 90, 0), '[2 3 0 1]': (180, 270, 0, 90)}
            self.class_list  = ['[0 1 2 3]', '[2 0 1 3]','[3 1 0 2]','[3 2 1 0]','[2 3 0 1]']
            
        elif  args.pretext_mode=='rotation_sorting' and args.pretext_videos=='single' and args.tuple_len==4 and args.pretext_classes==6:
            #1 #9 #16 #24 #17 #4
            self.sort2degree = {'[0 1 2 3]': (0, 90, 180, 270), '[2 0 1 3]': (90, 180, 0, 270), '[3 1 0 2]': (180, 90, 270, 0), '[3 2 1 0]': (270, 180, 90, 0), '[2 3 0 1]': (180, 270, 0, 90), '[0 3 1 2]': (0, 180, 270, 90) }
            self.class_list  = ['[0 1 2 3]', '[2 0 1 3]','[3 1 0 2]','[3 2 1 0]','[2 3 0 1]', '[0 3 1 2]']
            
        elif  (args.pretext_mode=='rotation_sorting' or args.pretext_mode=='rs_rotation_multi') and args.pretext_videos=='single' and args.tuple_len==4 and args.pretext_classes==8:
            #1 #9 #16 #24 #17 #4 #20 #11
            self.sort2degree = {'[0 1 2 3]': (0, 90, 180, 270), '[2 0 1 3]': (90, 180, 0, 270), '[3 1 0 2]': (180, 90, 270, 0), '[3 2 1 0]': (270, 180, 90, 0), '[2 3 0 1]': (180, 270, 0, 90), '[0 3 1 2]': (0, 180, 270, 90), '[1 3 2 0]': (270, 0, 180, 90), '[2 0 3 1]': (90, 270, 0, 180)}
            self.class_list  = ['[0 1 2 3]', '[2 0 1 3]','[3 1 0 2]','[3 2 1 0]','[2 3 0 1]', '[0 3 1 2]', '[1 3 2 0]', '[2 0 3 1]']
        
        elif  args.pretext_mode=='rotation_sorting' and args.pretext_videos=='single' and args.tuple_len==4 and args.pretext_classes==12:
            self.sort2degree = {'[0 1 2 3]': (0, 90, 180, 270), '[2 0 1 3]': (90, 180, 0, 270), '[3 1 0 2]': (180, 90, 270, 0), '[3 2 1 0]': (270, 180, 90, 0), '[2 3 0 1]': (180, 270, 0, 90),
                                '[0 3 1 2]': (0, 180, 270, 90), '[1 3 2 0]': (270, 0, 180, 90), '[2 0 3 1]': (90, 270, 0, 180), '[0 2 1 3]': (0, 180, 90, 270), '[1 0 3 2]': (90, 0, 270, 180),
                                '[1 3 0 2]': (180, 0, 270, 90), '[2 1 3 0]': (270, 90, 0, 180)}
            self.class_list  = ['[0 1 2 3]', '[2 0 1 3]','[3 1 0 2]','[3 2 1 0]','[2 3 0 1]', '[0 3 1 2]', '[1 3 2 0]', '[2 0 3 1]', '[0 2 1 3]', '[1 0 3 2]', '[1 3 0 2]', '[2 1 3 0]']
            
        elif  args.pretext_mode=='rotation_sorting' and args.pretext_videos=='single' and args.tuple_len==4 and args.pretext_classes==16:
            self.sort2degree = {'[0 1 2 3]': (0, 90, 180, 270), '[2 0 1 3]': (90, 180, 0, 270), '[3 1 0 2]': (180, 90, 270, 0), '[3 2 1 0]': (270, 180, 90, 0), '[2 3 0 1]': (180, 270, 0, 90),
                                '[0 3 1 2]': (0, 180, 270, 90), '[1 3 2 0]': (270, 0, 180, 90), '[2 0 3 1]': (90, 270, 0, 180), '[0 2 1 3]': (0, 180, 90, 270), '[1 0 3 2]': (90, 0, 270, 180),
                                '[1 3 0 2]': (180, 0, 270, 90), '[2 1 3 0]': (270, 90, 0, 180), '[0 3 2 1]': (0, 270, 180, 90), '[3 0 1 2]':  (90, 180, 270, 0),'[1 2 0 3]': (180, 0, 90, 270),
                                '[2 3 1 0]': (270, 180, 0, 90)}
            self.class_list  = ['[0 1 2 3]', '[2 0 1 3]','[3 1 0 2]','[3 2 1 0]','[2 3 0 1]', '[0 3 1 2]', '[1 3 2 0]', '[2 0 3 1]', '[0 2 1 3]', '[1 0 3 2]', '[1 3 0 2]', '[2 1 3 0]', '[0 3 2 1]', '[3 0 1 2]', '[1 2 0 3]', '[2 3 1 0]']
            
        elif  args.pretext_mode=='rs_rotation_multi' and args.pretext_videos=='single' and args.tuple_len==4 and args.pretext_classes==16:
            self.sort2degree = {'[0 1 2 3]': (0, 90, 180, 270), '[2 0 1 3]': (90, 180, 0, 270), '[3 1 0 2]': (180, 90, 270, 0), '[3 2 1 0]': (270, 180, 90, 0), '[2 3 0 1]': (180, 270, 0, 90),
                                    '[0 3 1 2]': (0, 180, 270, 90), '[1 3 2 0]': (270, 0, 180, 90), '[2 0 3 1]': (90, 270, 0, 180), '[0 2 1 3]': (0, 180, 90, 270), '[1 0 3 2]': (90, 0, 270, 180),
                                    '[1 3 0 2]': (180, 0, 270, 90), '[2 1 3 0]': (270, 90, 0, 180), '[0 3 2 1]': (0, 270, 180, 90), '[3 0 1 2]':  (90, 180, 270, 0),'[1 2 0 3]': (180, 0, 90, 270),
                                    '[2 3 1 0]': (270, 180, 0, 90)}
            self.class_list  = ['[0 1 2 3]', '[2 0 1 3]','[3 1 0 2]','[3 2 1 0]','[2 3 0 1]', '[0 3 1 2]', '[1 3 2 0]', '[2 0 3 1]', '[0 2 1 3]', '[1 0 3 2]', '[1 3 0 2]', '[2 1 3 0]', '[0 3 2 1]', '[3 0 1 2]', '[1 2 0 3]', '[2 3 1 0]']


        self.classes_dict = {0:0, 1:90, 2:180, 3:270}
        self.degree_to_label = {0:0, 90:1, 180:2, 270:3}
        


        if self.train:
            train_split_path = os.path.join(root_dir, 'Splits', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'Splits', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
        print('Use split'+ self.split)

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index
        """
        
        
        
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        
        filename = os.path.join(self.root_dir, 'Videos', videoname)
        
        #ndarray of dimension (T, M, N, C), where T is the number of frames, M is the height, N is width, and C is depth.
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape
        
         
        # random select a clip for train
        if self.train:        
            #clip = self.clip_sampler(videodata) # clip shape : [16, 240, 320, 3] [T, H, W, C]

            if self.args.pretext_mode=='rotation_sorting' and self.args.pretext_videos=='single' and self.args.tuple_len==3  and self.args.pretext_classes==6:
                clip1 = self.clip_sampler(videodata)
                clip2 = self.clip_sampler(videodata)
                clip3 = self.clip_sampler(videodata)
                
                class_idx =  random.randint(0, 5)
                degrees_key = self.class_list[class_idx]
                degrees =  self.sort2degree [degrees_key]
                #print (degrees_key, class_idx, degrees)
                
                clip1_r = self.rotate (clip1, degrees[0])
                clip2_r = self.rotate (clip2, degrees[1])
                clip3_r = self.rotate (clip3, degrees[2])
            
         
                if self.transform:
                    clip1_r = self.transform(clip1_r.byte())
                    clip2_r = self.transform(clip2_r.byte())
                    clip3_r = self.transform(clip3_r.byte())
                
                else:
                    clip1_r = torch.tensor(clip1_r)
                    clip2_r = torch.tensor(clip2_r)
                    clip3_r = torch.tensor(clip3_r)
                    
                    
                tuple_clip = []
                tuple_clip.append(clip1_r)
                tuple_clip.append(clip2_r)
                tuple_clip.append(clip3_r)
                


                
                return torch.stack(tuple_clip), torch.tensor(int(class_idx)),  torch.tensor(int(0))
            
            elif (self.args.pretext_mode=='rotation_sorting' or self.args.pretext_mode=='rs_rotation_multi') and self.args.pretext_videos=='single' and self.args.tuple_len==4:
                clip1 = self.clip_sampler(videodata)
                clip2 = self.clip_sampler(videodata)
                clip3 = self.clip_sampler(videodata)
                clip4 = self.clip_sampler(videodata)
                if self.args.pretext_classes==4:
                    class_idx =  random.randint(0, 3)
                elif self.args.pretext_classes==5:
                    class_idx =  random.randint(0, 4)
                elif self.args.pretext_classes==6:
                    class_idx =  random.randint(0, 5)
                elif self.args.pretext_classes==8:
                    class_idx =  random.randint(0, 7)
                elif self.args.pretext_classes==12:
                     class_idx =  random.randint(0, 11)
                elif self.args.pretext_classes==16:
                     class_idx =  random.randint(0, 15)
                    
                degrees_key = self.class_list[class_idx]
                degrees =  self.sort2degree [degrees_key]
                #print (degrees_key, class_idx, degrees)
                
                clip1_r = self.rotate (clip1, degrees[0])
                clip2_r = self.rotate (clip2, degrees[1])
                clip3_r = self.rotate (clip3, degrees[2])
                clip4_r = self.rotate (clip4, degrees[3])
            
         
                if self.transform:
                    clip1_r = self.transform(clip1_r.byte())
                    clip2_r = self.transform(clip2_r.byte())
                    clip3_r = self.transform(clip3_r.byte())
                    clip4_r = self.transform(clip4_r.byte())
                
                else:
                    clip1_r = torch.tensor(clip1_r)
                    clip2_r = torch.tensor(clip2_r)
                    clip3_r = torch.tensor(clip3_r)
                    clip4_r = torch.tensor(clip4_r)
                    
                    
                tuple_clip = []
                tuple_clip.append(clip1_r)
                tuple_clip.append(clip2_r)
                tuple_clip.append(clip3_r)
                tuple_clip.append(clip4_r)


                labels = []
                labels.append(self.degree_to_label[degrees[0]])
                labels.append(self.degree_to_label[degrees[1]])
                labels.append(self.degree_to_label[degrees[2]])
                labels.append(self.degree_to_label[degrees[3]])
                return torch.stack(tuple_clip), torch.tensor(int(class_idx)),  labels #torch.tensor(int(0))
                
                
                    
            elif self.args.pretext_mode=='rotation':
                clip1 = self.clip_sampler(videodata)
                class_idx =  random.randint(0, 3)
                degree = self.classes_dict[class_idx]
                clip1_r = self.rotate (clip1, degree)
                
                

                
                if self.transform:
                    clip1_r = self.transform(clip1_r.byte())
                else:
                    clip1_r = torch.tensor(clip1_r)
                  
                    

                
                return clip1_r , torch.tensor(int(class_idx)),  torch.tensor(int(0))
                
                
                
                
 
            
            
            
        
        else:
            # Right now there is no difference between train and test
            
            #clip = self.clip_sampler(videodata) # clip shape : [16, 240, 320, 3] [T, H, W, C]

            if self.args.pretext_mode=='rotation_sorting' and self.args.pretext_videos=='single' and self.args.tuple_len==3  and self.args.pretext_classes==6:
                clip1 = self.clip_sampler(videodata)
                clip2 = self.clip_sampler(videodata)
                clip3 = self.clip_sampler(videodata)
                
                class_idx =  random.randint(0, 5)
                degrees_key = self.class_list[class_idx]
                degrees =  self.sort2degree [degrees_key]
                #print (degrees_key, class_idx, degrees)
                
                clip1_r = self.rotate (clip1, degrees[0])
                clip2_r = self.rotate (clip2, degrees[1])
                clip3_r = self.rotate (clip3, degrees[2])
            
         
                if self.transform:
                    clip1_r = self.transform(clip1_r.byte())
                    clip2_r = self.transform(clip2_r.byte())
                    clip3_r = self.transform(clip3_r.byte())
                
                else:
                    clip1_r = torch.tensor(clip1_r)
                    clip2_r = torch.tensor(clip2_r)
                    clip3_r = torch.tensor(clip3_r)
                    
                    
                tuple_clip = []
                tuple_clip.append(clip1_r)
                tuple_clip.append(clip2_r)
                tuple_clip.append(clip3_r)
                

                

                
                return torch.stack(tuple_clip), torch.tensor(int(class_idx)),  torch.tensor(int(0))
            
            elif (self.args.pretext_mode=='rotation_sorting' or self.args.pretext_mode=='rs_rotation_multi')  and self.args.pretext_videos=='single' and self.args.tuple_len==4:
                clip1 = self.clip_sampler(videodata)
                clip2 = self.clip_sampler(videodata)
                clip3 = self.clip_sampler(videodata)
                clip4 = self.clip_sampler(videodata)
                
                if self.args.pretext_classes==4:
                    class_idx =  random.randint(0, 3)
                elif self.args.pretext_classes==5:
                    class_idx =  random.randint(0, 4)
                elif self.args.pretext_classes==6:
                    class_idx =  random.randint(0, 5)
                elif self.args.pretext_classes==8:
                    class_idx =  random.randint(0, 7)
                elif self.args.pretext_classes==12:
                    class_idx =  random.randint(0, 11)
                elif self.args.pretext_classes==16:
                     class_idx =  random.randint(0, 15)
                    
                degrees_key = self.class_list[class_idx]
                degrees =  self.sort2degree [degrees_key]
                #print (degrees_key, class_idx, degrees)
                
                clip1_r = self.rotate (clip1, degrees[0])
                clip2_r = self.rotate (clip2, degrees[1])
                clip3_r = self.rotate (clip3, degrees[2])
                clip4_r = self.rotate (clip4, degrees[3])
            
         
                if self.transform:
                    clip1_r = self.transform(clip1_r.byte())
                    clip2_r = self.transform(clip2_r.byte())
                    clip3_r = self.transform(clip3_r.byte())
                    clip4_r = self.transform(clip4_r.byte())
                
                else:
                    clip1_r = torch.tensor(clip1_r)
                    clip2_r = torch.tensor(clip2_r)
                    clip3_r = torch.tensor(clip3_r)
                    clip4_r = torch.tensor(clip4_r)
                    
                    
                tuple_clip = []
                tuple_clip.append(clip1_r)
                tuple_clip.append(clip2_r)
                tuple_clip.append(clip3_r)
                tuple_clip.append(clip4_r)

                

                labels = []
                labels.append(self.degree_to_label[degrees[0]])
                labels.append(self.degree_to_label[degrees[1]])
                labels.append(self.degree_to_label[degrees[2]])
                labels.append(self.degree_to_label[degrees[3]])
                return torch.stack(tuple_clip), torch.tensor(int(class_idx)),  labels #torch.tensor(int(0))
                
                
                    
            elif self.args.pretext_mode=='rotation':
                clip1 = self.clip_sampler(videodata)
                class_idx =  random.randint(0, 3)
                degree = self.classes_dict[class_idx]
                clip1_r = self.rotate (clip1, degree)
                
                

                
                if self.transform:
                    clip1_r = self.transform(clip1_r.byte())
                else:
                    clip1_r = torch.tensor(clip1_r)
                  
                    
                

                
                return clip1_r , torch.tensor(int(class_idx)),  torch.tensor(int(0))
                
                
                
                
 
            
            
            
              
                
                
                
 
            
            

        
        
    
    def clip_sampler(self, videodata):
        length, height, width, channel = videodata.shape
        if length >= self.clip_len:
            if self.sampling_mode=='random_skip':
                #print('Sampling Using Random Skip Frames.')
                #https://github.com/sjenni/temporal-ssl
                #https://github.com/sjenni/temporal-ssl/blob/master/PreprocessorVideo.py

                #random.randint(low, high=None, size=None, dtype=int)
                #Return random integers from low (inclusive) to high (exclusive).
                skip_frames=np.random.randint(1,5)
                #print('skip_frames',skip_frames)
                eff_skip = np.amin([(length / self.clip_len), skip_frames])
                eff_skip=int(eff_skip)
                #print ('eff_skip',eff_skip)

                max_frame = length - self.clip_len * eff_skip + 1
                #print('max_frame',max_frame)

                random_offset = int(np.random.uniform(0, max_frame))
                #print('random_offset',random_offset)


                offsets=range(random_offset, random_offset + self.clip_len * eff_skip, eff_skip)

                #print('offsets',offsets)
                #for n in offsets:
                #    print(n)
                
                clip=videodata[offsets]
                return clip
        
            else:
                if self.sampling_mode=='fixed_skip':
                    #print('Sampling Using Fixed Skip Frames.')  # To be implemented
                    skip_frames=self.skip_rate
                    eff_skip = np.amin([(length / self.clip_len), skip_frames])
                    eff_skip=int(eff_skip)
                    max_frame = length - self.clip_len * eff_skip + 1
                    random_offset = int(np.random.uniform(0, max_frame))
                    offsets=range(random_offset, random_offset + self.clip_len * eff_skip, eff_skip)
                    clip=videodata[offsets]
                    return clip
                    
                    
                    
                else:
                    if self.sampling_mode=='sequential':
                        clip_start = random.randint(0, length - self.clip_len)
                        clip = videodata[clip_start: clip_start + self.clip_len]
                        return clip
        else:
            #Repeat some of the frames to pad short videos
            # pad left, only sample once
            # https://github.com/TengdaHan/CoCLR/blob/main/dataset/lmdb_dataset.py
            sequence = np.arange(self.clip_len)
            seq_idx = np.zeros_like(sequence)
            sequence = sequence[sequence < length]
            seq_idx[-len(sequence)::] = sequence
            clip=videodata[seq_idx]
            return clip
        
    def rotate (self, clip, degree):
        # takes a numpy of shape [T,H,W,C] 
        #convert it to a tensor of shape ([T,C,H,W]) and outputs PIL images in a list of length T
        
        clip_tensor = torch.from_numpy(clip)
        clip_tensor = clip_tensor.permute(0,3,1,2)
        topil = transforms.ToPILImage()
        imgmap = [topil(i) for i in clip_tensor]      #imgmap list of Images
        
        imgmap = [i.rotate(degree, expand=True) for i in imgmap]
        
        totensor = transforms.ToTensor() #the range [0.0, 1.0]
        clip_tensor_list = [totensor(i) for i in imgmap]                   #<class 'list'> of lenght T of many tensors torch.Size([C, H, W])
        clip_tensor=torch.stack(clip_tensor_list).permute(0,2,3,1)         #torch.Size([T,H,W,C])
        
        ##clip_tensor = clip_tensor.numpy()
        clip_tensor = clip_tensor * 255                                    # undo the scaling that is done by transforms.ToTensor() operation
        
        return clip_tensor
        
    def save_video(self, video, name):
        video = video.numpy()
        ###video = video * 255
        torchvision.io.write_video(name, video, 25 )
        

        


def print_permutations():
    #https://www.geeksforgeeks.org/permutation-and-combination-in-python/
    
    # Get all permutations of [90, 180, 270]
    perm = permutations([0,90, 180, 270])
    idx = 1
    # Print the obtained permutations
    for i in list(perm):
        print (idx, ' - ' ,i , type(i), np.argsort(np.array(i)) )
        
        idx+=1
    
        
       

        
def parse_args():
    parser = argparse.ArgumentParser(description='Video Classification') 
    parser.add_argument('--split', type=str, default='1', help='dataset split 1,2,3')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--sampling_mode', type=str, default='fixed_skip', help='random_skip/fixed_skip/sequential.')
    parser.add_argument('--skip_rate', type=int, default=4, help='1..4.') 

    
    parser.add_argument('--tuple_len', type=int, default=4, help='the number of clips sampled in each tuple.')
    parser.add_argument('--pretext_mode', type=str, default='rotation_sorting', help='rotation_sorting/rotation/rs_rotation_multi')
    parser.add_argument('--pretext_videos', type=str, default='single', help='single/multi')
    parser.add_argument('--pretext_classes', type=int, default=16, help='')
    
    args = parser.parse_args()
    return args       
         
if __name__ == '__main__':
    print ('Running')
    root_dir=r'C:\Users\.......'
    args = parse_args()
    
    train_transforms =torchvision.transforms.Compose([T.ToTensorVideo(),
                                               
   
                                            A.RandomSizedCrop(112, interpolation=Image.BICUBIC, consistent=False, p=1.0, seq_len=args.cl, bottom_area=0.2),
                                            A.RandomHorizontalFlip(consistent=False, command=None, seq_len=args.cl),
                                          
    ])
    
    
    data_set =  RSUCF101Dataset(args,root_dir, True, train_transforms)
    train_dataloader = DataLoader(data_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    i, train_bar = 1, tqdm(train_dataloader)
    for clips, labels, degrees  in train_bar:
        print (clips.shape, labels)
        print(degrees)
        

        break

            