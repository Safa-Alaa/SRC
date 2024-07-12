
"""
The code incorporates elements from the following GitHub repositories. We express our gratitude to the authors, acknowledge their contributions, and give them the appropriate credit for their open-source code.
https://github.com/xudejing/video-clip-order-prediction/blob/master/datasets/hmdb51.py

"""

"""Dataset utils for NN."""

import os
import random
from glob import glob
from pprint import pprint
import uuid
import tempfile

import numpy as np
import ffmpeg
import skvideo.io
import pandas as pd
from skvideo.io import ffprobe
skvideo.setFFmpegPath('C:\ProgramData\Anaconda3\Lib\site-packages\skvideo\io')

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class HMDB51Dataset(Dataset):
    """HMDB51 dataset for recognition. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """
    def __init__(self, args, root_dir, train=True, transform=None, test_sample_num=10):
        
        print('Useing :: HMDB51 Dateset')
        
        self.sampling_mode = args.sampling_mode
        self.test_mode = args.test_mode
        self.skip_rate = args.skip_rate
        self.root_dir = root_dir
        self.clip_len = args.cl
        self.split = args.split
        self.train = train
        self.transform= transform
        self.test_sample_num = test_sample_num
        
        
        self.toPIL = transforms.ToPILImage()
        class_idx_path = os.path.join(root_dir, 'Splits', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]   #<class 'pandas.core.frame.DataFrame'> #Reads The class idxs and labels
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]   #The original classInd.txt and the videos txt splits has to be modified to start at idx=0 

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
            class_idx (tensor): class index, [0-5]
        """
        

        
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        ##################################################################
        class_idx = self.class_label2idx[videoname[:videoname.find("\\")]]     # Windows Version
        # class_idx = self.class_label2idx[videoname[:videoname.find("/")]]    # Mac Version
        
        filename = os.path.join(self.root_dir, 'Videos', videoname)
        
        #ndarray of dimension (T, M, N, C), where T is the number of frames, M is the height, N is width, and C is depth.
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape
        
        # random select a clip for train
        if self.train:
            
            clip = self.clip_sampler(videodata)
            #clip_start = random.randint(0, length - self.clip_len)
            #clip = videodata[clip_start: clip_start + self.clip_len]

            if self.transform:
                #buffer = self.transform(torch.from_numpy(buffer).byte())
                r"""trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                clip = torch.stack(trans_clip).permute([1, 0, 2, 3]) """
                
                clip = self.transform(torch.from_numpy(clip).byte())
            else:
                clip = torch.tensor(clip)

            return clip, torch.tensor(int(class_idx))
        
        
        
        
        # sample several clips for test
        else:

            all_clips, all_idx = self.test_clip_sampler(videodata, class_idx)
            return torch.stack(all_clips), torch.tensor(int(class_idx))
        
        
        

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
                    ###print('Sampling Using Fixed Skip Frames.')  # To be implemented
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
        
        
        
        
        
    def test_clip_sampler(self, videodata, class_idx):
        
        length, height, width, channel = videodata.shape
        if self.test_mode=='ten_clips':
            if length >= self.clip_len:
                if self.sampling_mode=='sequential':
                    # print('ten clips - sequential testing')
                    # https://github.com/xudejing/video-clip-order-prediction/blob/master/datasets/ucf101.py
                    all_clips = []
                    all_idx = []
                    for i in np.linspace(self.clip_len/2, length-self.clip_len/2, self.test_sample_num):
                        clip_start = int(i - self.clip_len/2)
                        clip = videodata[clip_start: clip_start + self.clip_len]
                        if self.transform:
                            clip = self.transform(torch.from_numpy(clip).byte())
                        else:
                            clip = torch.tensor(clip)
                                
                        all_clips.append(clip)
                        all_idx.append(torch.tensor(int(class_idx)))
                            
                    return all_clips, all_idx
                
                
                
                
                else:
                    # random skip or fixed skip sampling with 10 clips
                    # https://github.com/sjenni/temporal-ssl
                    # https://github.com/sjenni/temporal-ssl/blob/master/PreprocessorVideo.py

                    if self.sampling_mode=='random_skip':    
                        #random.randint(low, high=None, size=None, dtype=int)
                        #Return random integers from low (inclusive) to high (exclusive).
                        skip_frames = np.random.randint(1,5)
                        #print('ten clips - random skip testing')
                    else:
                        if self.sampling_mode=='fixed_skip':
                            skip_frames = self.skip_rate
                            #print('ten clips - fixed skip testing')
                        
            
                    #print('skip_frames',skip_frames)
            
                    eff_skip = np.amin([(length / self.clip_len), skip_frames])
                    eff_skip=int(eff_skip)
                    #print ('eff_skip',eff_skip)
                    
                    max_frame = length - (self.clip_len * eff_skip)
                    #print('max_frame',max_frame)
            
                    if self.test_sample_num is None:
                        start_inds = range(0, max_frame)
                    else:
                        start_inds=np.linspace(0, max_frame, self.test_sample_num)
               

            
                    inds_all_sub = [range (int(i), int(i) + self.clip_len * eff_skip, eff_skip) for i in start_inds]   


                    
                    all_clips = []
                    all_idx = []
                    for clip_inds in inds_all_sub:
                        clip = videodata [clip_inds]
                        if self.transform:
                            clip = self.transform(torch.from_numpy(clip).byte())
                        else:
                            clip = torch.tensor(clip)
                                
                        all_clips.append(clip)
                        all_idx.append(torch.tensor(int(class_idx)))
                    return all_clips, all_idx
                                
            else:
                #print('the video is too short, padding will be used')
                #Repeat some of the frames to pad short videos
                # pad left, only sample once
                # https://github.com/TengdaHan/CoCLR/blob/main/dataset/lmdb_dataset.py
                sequence = np.arange(self.clip_len)
                seq_idx = np.zeros_like(sequence)
                sequence = sequence[sequence < length]
                seq_idx[-len(sequence)::] = sequence
                clip = videodata[seq_idx]
                
                if self.transform:
                    clip = self.transform(torch.from_numpy(clip).byte())
                else:
                    clip = torch.tensor(clip)
                
               
                
                # return the same clip with different transform ten times
                all_clips = []
                all_idx = []
                for x in range(0, self.test_sample_num,1):
                    all_clips.append(clip)
                    all_idx.append(torch.tensor(int(class_idx)))
                return all_clips, all_idx
                
            
        
        else:
            if self.test_mode=='all_seq':
                
                num_test_seq=32
                if length >= self.clip_len:
                    
                    #https://github.com/sjenni/temporal-ssl
                    #https://github.com/sjenni/temporal-ssl/blob/master/PreprocessorVideo.py

                    if self.sampling_mode=='random_skip':    
                        #random.randint(low, high=None, size=None, dtype=int)
                        #Return random integers from low (inclusive) to high (exclusive).
                        skip_frames = np.random.randint(1,5)
                        #print('all sequences - random skip testing')
                    else:
                        if self.sampling_mode=='fixed_skip':
                            skip_frames = self.skip_rate
                            #print('all sequences - fixed skip testing')
                        else:
                            if self.sampling_mode=='sequential':
                                skip_frames = 1
                                #print('all sequences - sequential testing')
                                    
                        
                        

            
                    eff_skip = np.amin([(length / self.clip_len), skip_frames])
                    eff_skip=int(eff_skip)
                    #print ('eff_skip',eff_skip)

                    max_frame = length - self.clip_len * eff_skip + 1
                    #print('max_frame',max_frame)
            
                    if num_test_seq is None:
                        start_inds = range(0, max_frame)
                    else:
                        start_inds = range(0, max_frame, np.amax([(max_frame//num_test_seq), 1]))
            

            
                    inds_all_sub = [range (i, i + self.clip_len * eff_skip, eff_skip) for i in start_inds]   


                        
                    all_clips = []
                    all_idx = []
                    for clip_inds in inds_all_sub:
                        clip = videodata [clip_inds]
                        if self.transform:
                            clip = self.transform(torch.from_numpy(clip).byte())
                        else:
                            clip = torch.tensor(clip)
                                
                        all_clips.append(clip)
                        all_idx.append(torch.tensor(int(class_idx)))
                    return all_clips, all_idx
                        
                else:
                    #print('the video is too short, padding will be used')
                    #Repeat some of the frames to pad short videos
                    # pad left, only sample once
                    # https://github.com/TengdaHan/CoCLR/blob/main/dataset/lmdb_dataset.py
                    sequence = np.arange(self.clip_len)
                    seq_idx = np.zeros_like(sequence)
                    sequence = sequence[sequence < length]
                    seq_idx[-len(sequence)::] = sequence
                    clip=videodata[seq_idx]
                   
                   
                    
                    all_clips = []
                    all_idx = []
                    if self.transform:
                        clip = self.transform(torch.from_numpy(clip).byte())
                    else:
                        clip = torch.tensor(clip)
                    all_clips.append(clip)
                    all_idx.append(torch.tensor(int(class_idx)))
                    return all_clips, all_idx
            