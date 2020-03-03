# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.optim as optim
import os
import sys
from tqdm import tqdm
from time import time
from data_loader import H36M
from common.visualizer import show_pred_pose


class PoseModel(nn.Module):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 dropout=0.25, channels=1024):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__()

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.expand_layer = nn.Linear(num_joints_in * in_features, channels, bias=False)
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Linear(channels, num_joints_out * 3)

        # First block
        self.b1_l1 = nn.Linear(channels, channels, bias=False)
        self.b1_bn1 = nn.BatchNorm1d(channels, momentum=0.1)
        self.b1_l2 = nn.Linear(channels, channels, bias=False)
        self.b1_bn2 = nn.BatchNorm1d(channels, momentum=0.1)

        # Second block
        self.b2_l1 = nn.Linear(channels, channels, bias=False)
        self.b2_bn1 = nn.BatchNorm1d(channels, momentum=0.1)
        self.b2_l2 = nn.Linear(channels, channels, bias=False)
        self.b2_bn2 = nn.BatchNorm1d(channels, momentum=0.1)

    def forward(self, x):

        assert len(x.shape) == 2
        assert x.shape[-1] == self.num_joints_in * self.in_features

        x = self.drop(self.relu(self.expand_bn(self.expand_layer(x))))

        # First block
        res_1 = x
        x = self.drop(self.relu(self.b1_bn1(self.b1_l1(x))))
        x = self.drop(self.relu(self.b1_bn2(self.b1_l2(x))))
        x = x + res_1
        
        # Second block
        res_2 = x
        x = self.drop(self.relu(self.b2_bn1(self.b2_l1(x))))
        x = self.drop(self.relu(self.b2_bn2(self.b2_l2(x))))
        x = x + res_2
        
        x = self.shrink(x)
        return x


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))


def run_epoch(epoch, args, model_pose, data_loader, optimiser, mode='train', viz_pred=False):

    if mode == 'train':
        model_pose.train()
    else:
        model_pose.eval()

    start_time = time()
    epoch_loss_3d = 0
    mpjpe_epoch = 0.
    N = 0

    for batch_idx, (pose_2d, pose_3d_rel, _) in enumerate(tqdm(data_loader, ascii=True)):

        pose_2d = pose_2d.float()
        pose_3d_rel = pose_3d_rel.float()

        if torch.cuda.is_available():
            pose_2d = pose_2d.cuda()
            pose_3d_rel = pose_3d_rel.cuda()

        optimiser.zero_grad()

        # Predict 3D poses
        predicted_3d_pos = model_pose(pose_2d)

        if viz_pred is True:
            pred_3d_viz = np.zeros((1, 17, 3), dtype='float32')
            pred_3d_viz[:, 1:, :] = predicted_3d_pos[0].detach().cpu().view(16, 3).numpy() * 1000
            pose_2d_viz = pose_2d[0].cpu().view(1, 17, 2).numpy()

            show_pred_pose(pose_2d_viz, pred_3d_viz, n_joints=17)

        loss_3d_pos = mpjpe(predicted_3d_pos, pose_3d_rel)
        epoch_loss_3d += pose_3d_rel.shape[0] * loss_3d_pos.item()

        mpjpe_batch = mpjpe(predicted_3d_pos.detach().view(-1, 16, 3), pose_3d_rel.view(-1, 16, 3))
        mpjpe_epoch = mpjpe_epoch + pose_3d_rel.shape[0] * mpjpe_batch

        N += pose_3d_rel.shape[0]

        loss_total = loss_3d_pos

        if mode == 'train':
            loss_total.backward()
            optimiser.step()

    mpjpe_epoch = mpjpe_epoch / N
    epoch_loss_3d = epoch_loss_3d / N
    elapsed = (time() - start_time) / 60

    print('{}: [{:d}] time {:.2f} 3d_loss {:.3f} Mpjpe {:2f}'.format(
        mode, epoch, elapsed, epoch_loss_3d, mpjpe_epoch * 1000))

    result = dict()
    result['mpjpe'] = mpjpe_epoch * 1000

    return result


def main():
    args = parse_args()
    # print(args)

    tr_dataset = H36M(args, split='train')
    te_dataset = H36M(args, split='test')

    train_loader = torch.utils.data.DataLoader(
        tr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
        # worker_init_fn=worker_init_fn
    )

    test_loader = torch.utils.data.DataLoader(
        te_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        # worker_init_fn=worker_init_fn
    )

    num_joints_in = tr_dataset.poses_2d.shape[1]
    in_features = tr_dataset.poses_2d.shape[2]
    num_joints_out = tr_dataset.poses_3d.shape[1] - 1
    n_blocks = 2

    model_pose = PoseModel(num_joints_in, in_features, num_joints_out,
                           dropout=0.25, channels=1024)

    if torch.cuda.is_available():
        model_pose = model_pose.cuda()

    optimizer = optim.Adam(model_pose.parameters(), lr=args.learning_rate, amsgrad=True)
    schedular = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    epoch = 1
    min_mpjpe = 999.0
    while epoch < args.epochs:
        _ = run_epoch(epoch, args, model_pose, train_loader, optimizer, mode='train')
        result_test = run_epoch(epoch, args, model_pose, test_loader, optimizer, mode='test')

        if result_test['mpjpe'] < min_mpjpe:
            min_mpjpe = result_test['mpjpe']

            torch.save(model_pose.state_dict(), 'checkpoint/model_best.pth')

        epoch = epoch + 1

        schedular.step()

    run_epoch(1, args, model_pose, test_loader, optimizer, mode='test', viz_pred=True)


if __name__ == '__main__':
    main()
