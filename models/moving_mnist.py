import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from pointnet2 import *
from modules.pointnet2_utils import QueryAndGroup, furthest_point_sample, gather_operation
from modules.pointnet2_modules import PointnetFPModule
from modules.pointrnn_cell_impl import PointRNNCell, PointGRUCell, PointLSTMCell
from modules.point_local_transformer_rnn_cell_impl import PointLocalTransformerRNNCell

class PointRNN(nn.Module):
    def __init__(self, radius=4.0, num_samples=4, subsampling=2):
        super(PointRNN, self).__init__()

        self.radius = radius
        self.num_samples = num_samples
        self.subsampling = subsampling

        self.sample_and_group2 = QueryAndGroup(radius=2*radius/4+1e-6, nsample=num_samples, use_xyz=False)
        self.sample_and_group3 = QueryAndGroup(radius=4*radius/4+1e-6, nsample=num_samples, use_xyz=False)

        self.en_cell1 = PointRNNCell(radius=1*radius+1e-6, nsamples=3*num_samples, in_channels=0, out_channels=64)
        self.en_cell2 = PointRNNCell(radius=2*radius+1e-6, nsamples=2*num_samples, in_channels=64, out_channels=128)
        self.en_cell3 = PointRNNCell(radius=3*radius+1e-6, nsamples=1*num_samples, in_channels=128, out_channels=256)

        self.de_cell1 = PointRNNCell(radius=1*radius+1e-6, nsamples=3*num_samples, in_channels=0, out_channels=64)
        self.de_cell2 = PointRNNCell(radius=2*radius+1e-6, nsamples=2*num_samples, in_channels=64, out_channels=128)
        self.de_cell3 = PointRNNCell(radius=3*radius+1e-6, nsamples=1*num_samples, in_channels=128, out_channels=256)

        self.fp1 = PointnetFPModule(mlp=[128], bn=False)
        self.fp2 = PointnetFPModule(mlp=[128], bn=False)
        self.fp3 = PointnetFPModule(mlp=[128], bn=False)

        self.mlp = nn.Sequential(torch.nn.Conv1d(in_channels=448, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
                                 nn.ReLU(True),
                                 torch.nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'))

    def forward(self, xyzs):
        B = xyzs.size(0)
        L = xyzs.size(1)
        N = xyzs.size(2)

        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]

        # context
        states1 = None
        states2 = None
        states3 = None

        for t in range(int(L/2)):
            # 1
            xyz1_idx = furthest_point_sample(xyzs[t], N//self.subsampling)                                                          # (B, N//subsampling)
            xyz1 = gather_operation(xyzs[t].transpose(1, 2).contiguous(), xyz1_idx).transpose(1, 2).contiguous()                    # (B, N//self.subsampling, 3)
            states1 = self.en_cell1((xyz1, None), states1)
            s_xyz1, s_feat1 = states1
            #torch.max(input=spatial_feature, dim=-1, keepdim=False)

            # 2
            xyz2_idx = furthest_point_sample(s_xyz1, N//self.subsampling//self.subsampling)
            xyz2 = gather_operation(s_xyz1.transpose(1, 2).contiguous(), xyz2_idx).transpose(1, 2).contiguous()
            feat2 = self.sample_and_group2(s_xyz1, xyz2, s_feat1)
            feat2 = torch.max(input=feat2, dim=-1, keepdim=False)[0]
            states2 = self.en_cell2((xyz2, feat2), states2)
            s_xyz2, s_feat2 = states2

            # 3
            xyz3_idx = furthest_point_sample(s_xyz2, N//self.subsampling//self.subsampling//self.subsampling)
            xyz3 = gather_operation(s_xyz2.transpose(1, 2).contiguous(), xyz3_idx).transpose(1, 2).contiguous()
            feat3 = self.sample_and_group3(s_xyz2, xyz3, s_feat2)
            feat3 = torch.max(input=feat3, dim=-1, keepdim=False)[0]
            states3 = self.en_cell3((xyz3, feat3), states3)

        # prediction
        predicted_motions = []
        predicted_frames = []
        input_frame = xyzs[int(L/2)-1]
        for t in range(int(L/2), L):
            # 1
            xyz1_idx = furthest_point_sample(input_frame, N//self.subsampling)                                                      # (B, N//subsampling)
            xyz1 = gather_operation(input_frame.transpose(1, 2).contiguous(), xyz1_idx).transpose(1, 2).contiguous()                # (B, N//self.subsampling, 3)

            states1 = self.de_cell1((xyz1, None), states1)
            s_xyz1, s_feat1 = states1

            # 2
            xyz2_idx = furthest_point_sample(s_xyz1, N//self.subsampling//self.subsampling)
            xyz2 = gather_operation(s_xyz1.transpose(1, 2).contiguous(), xyz2_idx).transpose(1, 2).contiguous()
            feat2 = self.sample_and_group2(s_xyz1, xyz2, s_feat1)
            feat2 = torch.max(input=feat2, dim=-1, keepdim=False)[0]
            states2 = self.de_cell2((xyz2, feat2), states2)
            s_xyz2, s_feat2 = states2

            # 3
            xyz3_idx = furthest_point_sample(s_xyz2, N//self.subsampling//self.subsampling//self.subsampling)
            xyz3 = gather_operation(s_xyz2.transpose(1, 2).contiguous(), xyz3_idx).transpose(1, 2).contiguous()
            feat3 = self.sample_and_group3(s_xyz2, xyz3, s_feat2)
            feat3 = torch.max(input=feat3, dim=-1, keepdim=False)[0]
            states3 = self.de_cell3((xyz3, feat3), states3)
            s_xyz3, s_feat3 = states3

            l3_feat = self.fp3(s_xyz2, s_xyz3, s_feat2, s_feat3)
            l2_feat = self.fp2(s_xyz1, s_xyz2, s_feat1, l3_feat)
            l1_feat = self.fp1(input_frame, s_xyz1, None, l2_feat)

            predicted_motion = self.mlp(l1_feat).transpose(1, 2)
            predicted_motions.append(predicted_motion)
            predicted_frames.append(input_frame+predicted_motion)

            if self.training:
                #input_frame = xyzs[t]
                input_frame += predicted_motion
            else:
                input_frame += predicted_motion

        predicted_frames = torch.stack(tensors=predicted_frames, dim=1)
        return predicted_frames

class PointGRU(nn.Module):
    def __init__(self, radius=4.0, num_samples=4, subsampling=2):
        super(PointGRU, self).__init__()

        self.radius = radius
        self.num_samples = num_samples
        self.subsampling = subsampling

        self.sample_and_group2 = QueryAndGroup(radius=2*radius/4+1e-6, nsample=num_samples, use_xyz=False)
        self.sample_and_group3 = QueryAndGroup(radius=4*radius/4+1e-6, nsample=num_samples, use_xyz=False)

        self.en_cell1 = PointGRUCell(radius=1*radius+1e-6, nsamples=3*num_samples, in_channels=0, out_channels=64)
        self.en_cell2 = PointGRUCell(radius=2*radius+1e-6, nsamples=2*num_samples, in_channels=64, out_channels=128)
        self.en_cell3 = PointGRUCell(radius=3*radius+1e-6, nsamples=1*num_samples, in_channels=128, out_channels=256)

        self.de_cell1 = PointGRUCell(radius=1*radius+1e-6, nsamples=3*num_samples, in_channels=0, out_channels=64)
        self.de_cell2 = PointGRUCell(radius=2*radius+1e-6, nsamples=2*num_samples, in_channels=64, out_channels=128)
        self.de_cell3 = PointGRUCell(radius=3*radius+1e-6, nsamples=1*num_samples, in_channels=128, out_channels=256)

        self.fp1 = PointnetFPModule(mlp=[128], bn=False)
        self.fp2 = PointnetFPModule(mlp=[128], bn=False)
        self.fp3 = PointnetFPModule(mlp=[128], bn=False)

        self.mlp = nn.Sequential(torch.nn.Conv1d(in_channels=448, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
                                 nn.ReLU(True),
                                 torch.nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'))

    def forward(self, xyzs):
        B = xyzs.size(0)
        L = xyzs.size(1)
        N = xyzs.size(2)

        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]

        # context
        states1 = None
        states2 = None
        states3 = None

        for t in range(int(L/2)):
            # 1
            xyz1_idx = furthest_point_sample(xyzs[t], N//self.subsampling)                                                          # (B, N//subsampling)
            xyz1 = gather_operation(xyzs[t].transpose(1, 2).contiguous(), xyz1_idx).transpose(1, 2).contiguous()                    # (B, N//self.subsampling, 3)
            states1 = self.en_cell1((xyz1, None), states1)
            s_xyz1, s_feat1 = states1
            #torch.max(input=spatial_feature, dim=-1, keepdim=False)

            # 2
            xyz2_idx = furthest_point_sample(s_xyz1, N//self.subsampling//self.subsampling)
            xyz2 = gather_operation(s_xyz1.transpose(1, 2).contiguous(), xyz2_idx).transpose(1, 2).contiguous()
            feat2 = self.sample_and_group2(s_xyz1, xyz2, s_feat1)
            feat2 = torch.max(input=feat2, dim=-1, keepdim=False)[0]
            states2 = self.en_cell2((xyz2, feat2), states2)
            s_xyz2, s_feat2 = states2

            # 3
            xyz3_idx = furthest_point_sample(s_xyz2, N//self.subsampling//self.subsampling//self.subsampling)
            xyz3 = gather_operation(s_xyz2.transpose(1, 2).contiguous(), xyz3_idx).transpose(1, 2).contiguous()
            feat3 = self.sample_and_group3(s_xyz2, xyz3, s_feat2)
            feat3 = torch.max(input=feat3, dim=-1, keepdim=False)[0]
            states3 = self.en_cell3((xyz3, feat3), states3)

        # prediction
        predicted_motions = []
        predicted_frames = []
        input_frame = xyzs[int(L/2)-1]
        for t in range(int(L/2), L):
            # 1
            xyz1_idx = furthest_point_sample(input_frame, N//self.subsampling)                                                      # (B, N//subsampling)
            xyz1 = gather_operation(input_frame.transpose(1, 2).contiguous(), xyz1_idx).transpose(1, 2).contiguous()                # (B, N//self.subsampling, 3)

            states1 = self.de_cell1((xyz1, None), states1)
            s_xyz1, s_feat1 = states1

            # 2
            xyz2_idx = furthest_point_sample(s_xyz1, N//self.subsampling//self.subsampling)
            xyz2 = gather_operation(s_xyz1.transpose(1, 2).contiguous(), xyz2_idx).transpose(1, 2).contiguous()
            feat2 = self.sample_and_group2(s_xyz1, xyz2, s_feat1)
            feat2 = torch.max(input=feat2, dim=-1, keepdim=False)[0]
            states2 = self.de_cell2((xyz2, feat2), states2)
            s_xyz2, s_feat2 = states2

            # 3
            xyz3_idx = furthest_point_sample(s_xyz2, N//self.subsampling//self.subsampling//self.subsampling)
            xyz3 = gather_operation(s_xyz2.transpose(1, 2).contiguous(), xyz3_idx).transpose(1, 2).contiguous()
            feat3 = self.sample_and_group3(s_xyz2, xyz3, s_feat2)
            feat3 = torch.max(input=feat3, dim=-1, keepdim=False)[0]
            states3 = self.de_cell3((xyz3, feat3), states3)
            s_xyz3, s_feat3 = states3

            l3_feat = self.fp3(s_xyz2, s_xyz3, s_feat2, s_feat3)
            l2_feat = self.fp2(s_xyz1, s_xyz2, s_feat1, l3_feat)
            l1_feat = self.fp1(input_frame, s_xyz1, None, l2_feat)

            predicted_motion = self.mlp(l1_feat).transpose(1, 2)
            predicted_motions.append(predicted_motion)
            predicted_frames.append(input_frame+predicted_motion)

            if self.training:
                input_frame = xyzs[t]
            else:
                input_frame += predicted_motion

        predicted_frames = torch.stack(tensors=predicted_frames, dim=1)
        return predicted_frames

class PointLSTM(nn.Module):
    def __init__(self, radius=4.0, num_samples=4, subsampling=2):
        super(PointLSTM, self).__init__()

        self.radius = radius
        self.num_samples = num_samples
        self.subsampling = subsampling

        self.sample_and_group2 = QueryAndGroup(radius=2*radius/4+1e-6, nsample=num_samples, use_xyz=False)
        self.sample_and_group3 = QueryAndGroup(radius=4*radius/4+1e-6, nsample=num_samples, use_xyz=False)

        self.en_cell1 = PointLSTMCell(radius=1*radius+1e-6, nsamples=3*num_samples, in_channels=0, out_channels=64)
        self.en_cell2 = PointLSTMCell(radius=2*radius+1e-6, nsamples=2*num_samples, in_channels=64, out_channels=128)
        self.en_cell3 = PointLSTMCell(radius=3*radius+1e-6, nsamples=1*num_samples, in_channels=128, out_channels=256)

        self.de_cell1 = PointLSTMCell(radius=1*radius+1e-6, nsamples=3*num_samples, in_channels=0, out_channels=64)
        self.de_cell2 = PointLSTMCell(radius=2*radius+1e-6, nsamples=2*num_samples, in_channels=64, out_channels=128)
        self.de_cell3 = PointLSTMCell(radius=3*radius+1e-6, nsamples=1*num_samples, in_channels=128, out_channels=256)

        self.fp1 = PointnetFPModule(mlp=[128], bn=False)
        self.fp2 = PointnetFPModule(mlp=[128], bn=False)
        self.fp3 = PointnetFPModule(mlp=[128], bn=False)

        self.mlp = nn.Sequential(torch.nn.Conv1d(in_channels=448, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
                                 nn.ReLU(True),
                                 torch.nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'))

    def forward(self, xyzs):
        B = xyzs.size(0)
        L = xyzs.size(1)
        N = xyzs.size(2)

        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]

        # context
        states1 = None
        states2 = None
        states3 = None

        for t in range(int(L/2)):
            # 1
            xyz1_idx = furthest_point_sample(xyzs[t], N//self.subsampling)                                                          # (B, N//subsampling)
            xyz1 = gather_operation(xyzs[t].transpose(1, 2).contiguous(), xyz1_idx).transpose(1, 2).contiguous()                    # (B, N//self.subsampling, 3)
            states1 = self.en_cell1((xyz1, None), states1)
            s_xyz1, h_feat1, _ = states1
            #torch.max(input=spatial_feature, dim=-1, keepdim=False)

            # 2
            xyz2_idx = furthest_point_sample(s_xyz1, N//self.subsampling//self.subsampling)
            xyz2 = gather_operation(s_xyz1.transpose(1, 2).contiguous(), xyz2_idx).transpose(1, 2).contiguous()
            feat2 = self.sample_and_group2(s_xyz1, xyz2, h_feat1)
            feat2 = torch.max(input=feat2, dim=-1, keepdim=False)[0]
            states2 = self.en_cell2((xyz2, feat2), states2)
            s_xyz2, h_feat2, _ = states2

            # 3
            xyz3_idx = furthest_point_sample(s_xyz2, N//self.subsampling//self.subsampling//self.subsampling)
            xyz3 = gather_operation(s_xyz2.transpose(1, 2).contiguous(), xyz3_idx).transpose(1, 2).contiguous()
            feat3 = self.sample_and_group3(s_xyz2, xyz3, h_feat2)
            feat3 = torch.max(input=feat3, dim=-1, keepdim=False)[0]
            states3 = self.en_cell3((xyz3, feat3), states3)

        # prediction
        predicted_motions = []
        predicted_frames = []
        input_frame = xyzs[int(L/2)-1]
        for t in range(int(L/2), L):
            # 1
            xyz1_idx = furthest_point_sample(input_frame, N//self.subsampling)                                                      # (B, N//subsampling)
            xyz1 = gather_operation(input_frame.transpose(1, 2).contiguous(), xyz1_idx).transpose(1, 2).contiguous()                # (B, N//self.subsampling, 3)

            states1 = self.de_cell1((xyz1, None), states1)
            s_xyz1, h_feat1, _ = states1

            # 2
            xyz2_idx = furthest_point_sample(s_xyz1, N//self.subsampling//self.subsampling)
            xyz2 = gather_operation(s_xyz1.transpose(1, 2).contiguous(), xyz2_idx).transpose(1, 2).contiguous()
            feat2 = self.sample_and_group2(s_xyz1, xyz2, h_feat1)
            feat2 = torch.max(input=feat2, dim=-1, keepdim=False)[0]
            states2 = self.de_cell2((xyz2, feat2), states2)
            s_xyz2, h_feat2, _ = states2

            # 3
            xyz3_idx = furthest_point_sample(s_xyz2, N//self.subsampling//self.subsampling//self.subsampling)
            xyz3 = gather_operation(s_xyz2.transpose(1, 2).contiguous(), xyz3_idx).transpose(1, 2).contiguous()
            feat3 = self.sample_and_group3(s_xyz2, xyz3, h_feat2)
            feat3 = torch.max(input=feat3, dim=-1, keepdim=False)[0]
            states3 = self.de_cell3((xyz3, feat3), states3)
            s_xyz3, h_feat3, _ = states3

            l3_feat = self.fp3(s_xyz2, s_xyz3, h_feat2, h_feat3)
            l2_feat = self.fp2(s_xyz1, s_xyz2, h_feat1, l3_feat)
            l1_feat = self.fp1(input_frame, s_xyz1, None, l2_feat)

            predicted_motion = self.mlp(l1_feat).transpose(1, 2)
            predicted_motions.append(predicted_motion)
            predicted_frames.append(input_frame+predicted_motion)

            if self.training:
                input_frame = xyzs[t]
            else:
                input_frame += predicted_motion

        predicted_frames = torch.stack(tensors=predicted_frames, dim=1)
        return predicted_frames

if __name__ == '__main__':
    rnn = PointRNN(radius=4.0, num_samples=4, subsampling=2).to('cuda')
    video = torch.zeros([16, 10, 128, 3], dtype=torch.float32).to('cuda')
    prediction = rnn(video)
    print(prediction.shape)
