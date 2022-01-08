import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import pointnet2_utils
import pytorch_utils as pt_utils

from typing import List

class PointSpatioTemporalCorrelation(nn.Module):
    def __init__(
            self,
            radius: float,
            nsamples: int,
            in_channels: int,
            out_channels: int
    ):
        super().__init__()

        self.radius = radius
        self.nsamples = nsamples
        self.in_channels = in_channels

        self.fc = pt_utils.Conv2d(in_size=in_channels+out_channels+3, out_size=out_channels, activation=nn.ReLU(inplace=True), bn=None)

    def forward(self, P1: torch.Tensor, P2: torch.Tensor, X1: torch.Tensor, S2: torch.Tensor) -> (torch.Tensor):
        r"""
        Parameters
        ----------
        P1:     (B, N, 3)
        P2:     (B, N, 3)
        X1:     (B, C, N)
        S2:     (B, C, N)

        Returns
        -------
        S1:     (B, C, N)
        """
        # 1. Sample points
        idx = pointnet2_utils.ball_query(self.radius, self.nsamples, P2, P1)                            # (B, npoint, nsample)

        # 2.1 Group P2 points
        P2_flipped = P2.transpose(1, 2).contiguous()                                                    # (B, 3, npoint)
        P2_grouped = pointnet2_utils.grouping_operation(P2_flipped, idx)                                # (B, 3, npoint, nsample)
        # 2.2 Group P2 states
        S2_grouped = pointnet2_utils.grouping_operation(S2, idx)                                        # (B, C, npoint, nsample)

        # 3. Calcaulate displacements
        P1_flipped = P1.transpose(1, 2).contiguous()                                                    # (B, 3, npoint)
        P1_expanded = torch.unsqueeze(P1_flipped, 3)                                                    # (B, 3, npoint, 1)
        displacement = P2_grouped - P1_expanded                                                         # (B, 3, npoint, nsample)
        # 4. Concatenate X1, S2 and displacement
        if self.in_channels != 0:
            X1_expanded = torch.unsqueeze(X1, 3)                                                        # (B, C, npoint, 1)
            X1_repeated = X1_expanded.repeat(1, 1, 1, self.nsamples)
            correlation = torch.cat(tensors=(S2_grouped, X1_repeated, displacement), dim=1)
        else:
            correlation = torch.cat(tensors=(S2_grouped, displacement), dim=1)

        # 5. Fully-connected layer (the only parameters)
        S1 = self.fc(correlation)

        # 6. Pooling
        S1 = torch.max(input=S1, dim=-1, keepdim=False)[0]

        return S1


class PointRNNCell(nn.Module):
    def __init__(
            self,
            radius: float,
            nsamples: int,
            in_channels: int,
            out_channels: int
    ):
        super().__init__()

        self.out_channels = out_channels
        self.corr = PointSpatioTemporalCorrelation(radius, nsamples, in_channels, out_channels)

    def init_state(self, inputs: (torch.Tensor, torch.Tensor)):
        P, _ = inputs

        inferred_batch_size = P.size(0)
        inferred_npoints = P.size(1)

        device = P.get_device()

        P = torch.zeros([inferred_batch_size, inferred_npoints, 3], dtype=torch.float32, device=device)
        S = torch.zeros([inferred_batch_size, self.out_channels, inferred_npoints], dtype=torch.float32, device=device)

        return P, S

    def forward(self, inputs: (torch.Tensor, torch.Tensor), states: (torch.Tensor, torch.Tensor)=None) -> (torch.Tensor, torch.Tensor):
        if states is None:
            states = self.init_state(inputs)

        P1, X1 = inputs
        P2, S2 = states

        S1 = self.corr(P1, P2, X1, S2)

        return P1, S1

class PointGRUCell(nn.Module):
    def __init__(
            self,
            radius: float,
            nsamples: int,
            in_channels: int,
            out_channels: int
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_corr = PointSpatioTemporalCorrelation(radius, nsamples, in_channels, out_channels)
        self.r_corr = PointSpatioTemporalCorrelation(radius, nsamples, in_channels, out_channels)
        self.s_corr = PointSpatioTemporalCorrelation(radius, nsamples, 0, out_channels)

        self.sigmoid = nn.Sigmoid()

        self.fc = pt_utils.Conv1d(in_size=in_channels+out_channels, out_size=out_channels, activation=None, bn=None)
        self.tanh = nn.Tanh()

    def init_state(self, inputs: (torch.Tensor, torch.Tensor)):
        P, _ = inputs

        inferred_batch_size = P.size(0)
        inferred_npoints = P.size(1)

        device = P.get_device()

        P = torch.zeros([inferred_batch_size, inferred_npoints, 3], dtype=torch.float32, device=device)
        S = torch.zeros([inferred_batch_size, self.out_channels, inferred_npoints], dtype=torch.float32, device=device)

        return P, S

    def forward(self, inputs: (torch.Tensor, torch.Tensor), states: (torch.Tensor, torch.Tensor)=None) -> (torch.Tensor, torch.Tensor):
        if states is None:
            states = self.init_state(inputs)

        P1, X1 = inputs
        P2, S2 = states

        Z = self.z_corr(P1, P2, X1, S2)
        R = self.r_corr(P1, P2, X1, S2)
        Z = self.sigmoid(Z)
        R = self.sigmoid(R)

        S_old = self.s_corr(P1, P2, None, S2)

        if self.in_channels == 0:
            S_new = R*S_old
        else:
            S_new = torch.cat(tensors=[X1, R*S_old], dim=1)

        S_new = self.fc(S_new)

        S_new = self.tanh(S_new)

        S1 = Z * S_old + (1 - Z) * S_new

        return P1, S1

class PointLSTMCell(nn.Module):
    def __init__(
            self,
            radius: float,
            nsamples: int,
            in_channels: int,
            out_channels: int
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.i_corr = PointSpatioTemporalCorrelation(radius, nsamples, in_channels, out_channels)
        self.f_corr = PointSpatioTemporalCorrelation(radius, nsamples, in_channels, out_channels)
        self.o_corr = PointSpatioTemporalCorrelation(radius, nsamples, in_channels, out_channels)

        self.c_corr_new = PointSpatioTemporalCorrelation(radius, nsamples, in_channels, out_channels)
        self.c_corr_old = PointSpatioTemporalCorrelation(radius, nsamples, 0, out_channels)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def init_state(self, inputs: (torch.Tensor, torch.Tensor)):
        P, _ = inputs

        inferred_batch_size = P.size(0)
        inferred_npoints = P.size(1)

        device = P.get_device()

        P = torch.zeros([inferred_batch_size, inferred_npoints, 3], dtype=torch.float32, device=device)
        H = torch.zeros([inferred_batch_size, self.out_channels, inferred_npoints], dtype=torch.float32, device=device)
        C = torch.zeros([inferred_batch_size, self.out_channels, inferred_npoints], dtype=torch.float32, device=device)

        return P, H, C

    def forward(self, inputs: (torch.Tensor, torch.Tensor), states: (torch.Tensor, torch.Tensor, torch.Tensor)=None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        if states is None:
            states = self.init_state(inputs)

        P1, X1 = inputs
        P2, H2, C2 = states

        I = self.i_corr(P1, P2, X1, H2)
        F = self.f_corr(P1, P2, X1, H2)
        O = self.o_corr(P1, P2, X1, H2)
        C_new = self.c_corr_new(P1, P2, X1, H2)
        C_old = self.c_corr_old(P1, P2, None, C2)

        I = self.sigmoid(I)
        F = self.sigmoid(F)
        O = self.sigmoid(O)
        C_new = self.tanh(C_new)

        C1 = F * C_old + I * C_new
        H1 = O * self.tanh(C1)

        return P1, H1, C1

if __name__ == '__main__':
    radius = 1
    nsamples = 4
    in_channels = 128
    out_channels = 256

    lstm = PointLSTMCell(radius, nsamples, in_channels, out_channels).to('cuda')

    batch_size = 32
    npoints = 1024
    P1 = torch.zeros([batch_size, npoints, 3], dtype=torch.float32).to('cuda')
    X1 = torch.zeros([batch_size, in_channels, npoints], dtype=torch.float32).to('cuda')
    P2 = torch.zeros([batch_size, npoints, 3], dtype=torch.float32).to('cuda')
    H2 = torch.zeros([batch_size, out_channels, npoints], dtype=torch.float32).to('cuda')
    C2 = torch.zeros([batch_size, out_channels, npoints], dtype=torch.float32).to('cuda')

    P1, H1, C1 = lstm((P1, X1), (P2, H2, C2))
    print(P1.shape)
    print(H1.shape)
    print(C1.shape)

