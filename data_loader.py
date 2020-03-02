import numpy as np

from common.arguments import parse_args
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import pickle

from common.camera import *
from time import time
from common.utils import deterministic_random


class PoseDataset(data.Dataset):
    def __init__(self, opt, split='train'):

        print('Creating {} data loader\n'.format(split))

        self.opt = opt
        self.split = split
        dataset_path = 'data/dataset_{}_gt.pickle'.format(split)

        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
        f.close()

        self.cameras, self.poses_2d, self.poses_3d = self.merge_videos()

        self.n_samples = self.poses_2d.shape[0]

    def merge_videos(self):
        subject_map = {'S1': 1, 'S5': 5, 'S6': 6, 'S7': 7, 'S8': 8, 'S9': 9, 'S11': 11}

        # keys = ['subject', 'cameras', 'poses_3d', 'poses_2d']

        n_vids = len(self.dataset['subject'])

        subjects = np.tile(np.asarray(subject_map[self.dataset['subject'][0]]),
                           (self.dataset['poses_2d'][0].shape[0], 1))
        cameras = np.tile(self.dataset['cameras'][0], (self.dataset['poses_2d'][0].shape[0], 1))
        poses_2d = self.dataset['poses_2d'][0]
        poses_3d = self.dataset['poses_3d'][0]

        for i in range(1, n_vids):
            vid_samples = self.dataset['poses_2d'][i].shape[0]

            subjects = np.append(subjects, np.tile(np.asarray(
                subject_map[self.dataset['subject'][i]]), (vid_samples, 1)), axis=0)
            cameras = np.append(cameras, np.tile(self.dataset['cameras'][i], (vid_samples, 1)), axis=0)
            poses_2d = np.append(poses_2d, self.dataset['poses_2d'][i], axis=0)
            poses_3d = np.append(poses_3d, self.dataset['poses_3d'][i], axis=0)

        return cameras, poses_2d, poses_3d

    def __getitem__(self, idx):
        pose_2d = self.poses_2d[idx].reshape(-1)
        pose_3d_rel = self.poses_3d[idx][1:, :].reshape(-1)
        pose_root = self.poses_3d[idx][0, :].reshape(-1)

        return pose_2d, pose_3d_rel, pose_root

    def __len__(self):
        return self.n_samples

