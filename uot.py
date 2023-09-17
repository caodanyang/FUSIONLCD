import math
import torch
import torch._utils
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from typing import Optional, Callable


def compute_rigid_transform(points1, points2, weights):
    """Compute rigid transforms between two point clouds via weighted SVD.
       Adapted from https://github.com/yewzijian/RPMNet/
    Args:
        points1 (torch.Tensor): (B, M, 3) coordinates of the first point cloud
        points2 (torch.Tensor): (B, N, 3) coordinates of the second point cloud
        weights (torch.Tensor): (B, M)
    Returns:
        Transform T (B, 3, 4) to get from points1 to points2, i.e. T*points1 = points2
    """

    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + 1e-5)
    centroid_a = torch.sum(points1 * weights_normalized, dim=1)
    centroid_b = torch.sum(points2 * weights_normalized, dim=1)
    a_centered = points1 - centroid_a[:, None, :]
    b_centered = points2 - centroid_b[:, None, :]
    cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

    transform = torch.cat((rot_mat, translation), dim=2)
    return transform


def sinkhorn_unbalanced(feature1, feature2, epsilon, gamma, max_iter, matrix='cosine'):
    """
    Sinkhorn algorithm for Unbalanced Optimal Transport.
    Modified from https://github.com/valeoai/FLOT/
    Args:
        feature1 (torch.Tensor):
            (B, N, C) Point-wise features for points cloud 1.
        feature2 (torch.Tensor):
            (B, M, C) Point-wise features for points cloud 2.
        epsilon (torch.Tensor):
            Entropic regularization.
        gamma (torch.Tensor):
            Mass regularization.
        max_iter (int):
            Number of iteration of the Sinkhorn algorithm.
    Returns:
        T (torch.Tensor):
            (B, N, M) Transport plan between point cloud 1 and 2.
    """
    if matrix == 'cosine':
        # Transport cost matrix
        feature1 = feature1 / torch.sqrt(torch.sum(feature1 ** 2, -1, keepdim=True) + 1e-8)
        feature2 = feature2 / torch.sqrt(torch.sum(feature2 ** 2, -1, keepdim=True) + 1e-8)
        C = 1.0 - torch.bmm(feature1, feature2.transpose(1, 2))
    elif matrix == 'euclidean':
        distance_matrix = torch.sum(feature1 ** 2, -1, keepdim=True)
        distance_matrix = distance_matrix + torch.sum(feature2 ** 2, -1, keepdim=True).transpose(1, 2)
        distance_matrix = distance_matrix - 2 * torch.bmm(feature1, feature2.transpose(1, 2))  # c^2=a^2+b^2-2abcos
        distance_matrix = distance_matrix ** 0.5
        # d_max, _ = torch.max(distance_matrix, dim=2, keepdim=True)
        C = distance_matrix

    # Entropic regularisation
    K = torch.exp(-C / epsilon)  # * support

    # Early return if no iteration
    if max_iter == 0:
        return K

    # Init. of Sinkhorn algorithm
    power = gamma / (gamma + epsilon + 1e-8)
    a = (torch.ones((K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype) / K.shape[1])
    prob1 = (torch.ones((K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype) / K.shape[1])
    prob2 = (torch.ones((K.shape[0], K.shape[2], 1), device=feature2.device, dtype=feature2.dtype) / K.shape[2])

    # Sinkhorn algorithm
    for _ in range(max_iter):
        # Update b
        KTa = torch.bmm(K.transpose(1, 2), a)
        b = torch.pow(prob2 / (KTa + 1e-8), power)
        # Update a
        Kb = torch.bmm(K, b)
        a = torch.pow(prob1 / (Kb + 1e-8), power)

    # Transportation map
    T = torch.mul(torch.mul(a, K), b.transpose(1, 2))
    return T


class UOTHead(nn.Module):

    def __init__(self, nb_iter=5,name='original'):
        super().__init__()
        self.epsilon = torch.nn.Parameter(torch.zeros(1))  # Entropic regularisation
        self.gamma = torch.nn.Parameter(torch.zeros(1))  # Mass regularisation
        self.nb_iter = nb_iter
        self.name=name

    def forward(self, batch_dict, src_coords=None, mode='pairs'):

        feats = batch_dict['fea_kpt_'+self.name].squeeze(-1)

        B, C, NUM = feats.shape

        assert B % 2 == 0, "Batch size must be multiple of 2: B anchor + B positive samples"
        B = B // 2
        feat1 = feats[:B]
        feat2 = feats[B:]

        coords = batch_dict['key_points']
        coords1 = coords[:B, :, 0:3]
        coords2 = coords[B:, :, 0:3]

        correspondences_feature = sinkhorn_unbalanced(
            feat1.permute(0, 2, 1),
            feat2.permute(0, 2, 1),
            epsilon=torch.exp(self.epsilon) + 0.03,
            gamma=torch.exp(self.gamma),
            max_iter=self.nb_iter,
            matrix='cosine',
        )

        feature_corr_sum = correspondences_feature.sum(-1, keepdim=True)
        project_kpts = (correspondences_feature @ coords2) / (feature_corr_sum + 1e-8)
        project_feas = (correspondences_feature @ feat2.permute(0, 2, 1)) / (feature_corr_sum + 1e-8)

        batch_dict['project_kpts_'+self.name] = project_kpts
        batch_dict['project_feas_'+self.name] = project_feas.permute(0, 2, 1)
        # batch_dict['project_coord_kpts'] = project_coord_kpts
        batch_dict['correspondences_feature_'+self.name] = correspondences_feature
        # batch_dict['correspondences_coord'] = correspondences_coord

        transformation = compute_rigid_transform(coords1, project_kpts, feature_corr_sum.squeeze(-1))
        batch_dict['transformation_'+self.name] = transformation

        return batch_dict