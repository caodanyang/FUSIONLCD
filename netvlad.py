import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetVLAD(nn.Module):
    def __init__(self, fea_size=128, num_clusters=16):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.conv = nn.Conv2d(fea_size, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.randn(num_clusters, fea_size))
        self.relu = nn.ReLU(inplace=True)
        # self.linear=nn.Linear(fea_size*num_clusters,256)

    def forward(self, x):
        """
        x: B, C, H, W,W=1
        """
        soft_assign = self.conv(x)  # (B, num_clusters, H, W)
        soft_assign = self.relu(soft_assign)
        soft_assign = torch.nn.functional.softmax(soft_assign, dim=1)  # (B, num_clusters, H, W)

        # reshape for broadcasting
        B, C, H, W = x.shape
        soft_assign = soft_assign.view(B, self.num_clusters, -1)  # (B, num_clusters, H * W)
        x_flatten = x.view(B, C, -1)  # (B, C, H * W)

        # compute residuals
        x_flatten1 = x_flatten.unsqueeze(1).permute(0, 1, 3, 2)  # (B, 1, H*W, C)
        centroids = self.centroids.unsqueeze(0).unsqueeze(2)  # (1, num_cluster, 1, C)
        residual = x_flatten1 - centroids  # (B, num_clusters, H * W, C)

        residual *= soft_assign.unsqueeze(-1)  # (B, num_clusters, H * W, C)

        # sum residuals and assign
        vlad = residual.sum(dim=-2)  # (B, num_clusters, C)
        vlad = nn.functional.normalize(vlad, p=2, dim=2)  # (B, num_clusters, C)
        # vlad = vlad.view(B, -1)  # (B, num_clusters * C)
        # vlad = torch.max(vlad,dim=1)[0]
        # vlad=F.max_pool1d(vlad,kernel_size=2)
        vlad = vlad.view(B, -1)
        vlad = nn.functional.normalize(vlad, p=2, dim=1)  # (B, num_clusters * C)
        # vlad=self.linear(vlad)
        return vlad


class NetVLADLoupe(nn.Module):
    """
    Original Tensorflow implementation: https://github.com/antoine77340/LOUPE
    """

    def __init__(self, feature_size, cluster_size, output_dim,
                 gating=True, add_norm=True, is_training=True, normalization='batch'):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        # output_dim=cluster_size * feature_size
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_norm
        self.cluster_size = cluster_size
        if normalization == 'instance':
            norm = lambda x: nn.LayerNorm(x)
        elif normalization == 'group':
            norm = lambda x: nn.GroupNorm(8, x)
        else:
            norm = lambda x: nn.BatchNorm1d(x)
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))
        if add_norm:
            self.cluster_biases = None
            self.bn1 = norm(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = norm(output_dim)

        if gating:
            self.context_gating = GatingContext(output_dim, add_batch_norm=add_norm, normalization=normalization)

    def forward(self, x):
        """
        x: B N C
        """
        max_samples = x.shape[1]
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1, max_samples, self.cluster_size)
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1).contiguous()
        vlad0 = vlad - a

        vlad1 = F.normalize(vlad0, dim=1, p=2, eps=1e-6)
        vlad2 = vlad1.view((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad2, dim=1, p=2, eps=1e-6)

        vlad = torch.matmul(vlad, self.hidden1_weights)

        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)
        # vlad = vlad / vlad.norm(dim=1, keepdim=True)
        return vlad


class GatingContext(nn.Module):
    """
    Original Tensorflow implementation: https://github.com/antoine77340/LOUPE
    """

    def __init__(self, dim, add_batch_norm=True, normalization='batch'):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        if normalization == 'instance':
            norm = lambda x: nn.LayerNorm(x)
        elif normalization == 'group':
            norm = lambda x: nn.GroupNorm(8, x)
        else:
            norm = lambda x: nn.BatchNorm1d(x)
        self.gating_weights = nn.Parameter(torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = norm(dim)
        else:
            self.gating_biases = nn.Parameter(torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation
