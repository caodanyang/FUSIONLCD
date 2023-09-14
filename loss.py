import torch
import tools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import distances


def tr_loss(batch_dict,key):
    loss1 = (batch_dict[key][:, 0:3, 3] -
             batch_dict['pose_to_frame'][:, 0:3, 3]).norm(dim=1).mean()
    loss2 = (torch.acos(torch.clip(batch_dict[key][:, 0, 0].view(-1, 1), -1, 1)) -
             torch.acos(torch.clip(batch_dict['pose_to_frame'][:, 0, 0].view(-1, 1), -1, 1))).norm(dim=1).mean() / 3.1415 * 180
    return loss1, loss2


def gen_points_loss(batch_dict):
    key_points_gen = batch_dict['key_points_gen']
    key_points = batch_dict['key_points']
    key_points_gen1 = torch.cat((key_points_gen, key_points_gen * 0), dim=2)
    key_points_gen1[:, :, 3] = 1
    # pose_query=batch_dict['pose_query']
    # pose_positive=batch_dict['pose_positive']
    # poses=torch.cat((pose_query,pose_positive),dim=0)
    # key_points_gen2=torch.bmm(poses,key_points_gen1.permute(0,2,1)).permute(0,2,1)
    # key_points2=torch.bmm(poses,key_points.permute(0,2,1)).permute(0,2,1)
    # loss_gpo=(key_points_gen2[:,:,:2]-key_points2[:,:,:2]).norm(p=1,dim=2).mean()
    pose_to_frame = batch_dict['pose_to_frame']
    B = pose_to_frame.shape[0]
    src_pts = key_points[:B]
    tgt_pts = key_points[B:]
    src_pts_gen = key_points_gen1[:B]
    tgt_pts_gen = key_points_gen1[B:]
    srcs = torch.cat((src_pts, src_pts_gen), dim=0)
    tgts = torch.cat((tgt_pts_gen, tgt_pts), dim=0)
    pose_to_frame1 = torch.cat((pose_to_frame, pose_to_frame), dim=0)
    srcs1 = torch.bmm(pose_to_frame1, srcs.permute(0, 2, 1)).permute(0, 2, 1)
    loss = torch.mean(torch.abs(srcs1[:, :, :2] - tgts[:, :, :2]))
    return loss


def rand_dis(x, y):
    assert len(x.shape)==2 and len(y.shape)==2,'x and y must be 2 dim'
    N, N = x.size()
    ids=torch.arange(N).to(x.device)
    idx = ids.view(1, N).repeat(N, 1)
    mask = ~(idx == idx.transpose(0, 1))
    idx1 = idx[mask].view(N, N - 1)
    random_indices = torch.randint(N - 1, size=(N,)).to(x.device)
    rand_idx = torch.gather(idx1, 1, random_indices.view(-1, 1))
    rand_idx1 = torch.cat([ids.view(N, 1), rand_idx], dim=1)
    diag = ids.view(N, 1).repeat(1, 2)
    x1 = x[rand_idx1[:, 0], rand_idx1[:, 1]]
    x2 = x1*0
    x3 = torch.cat([x1.view(N, 1), x2.view(N, 1)], dim=1)
    y1 = y[rand_idx1[:, 0], rand_idx1[:, 1]]
    y2 = y[diag[:, 0], diag[:, 1]]
    y3 = torch.cat([y1.view(N, 1), y2.view(N, 1)], dim=1)
    dis=torch.abs(x3-y3).mean()+F.relu(0.2-torch.abs(y1)).mean()
    return dis


def gen_feature_loss(batch_dict):
    #BCN
    fea_pt_dual_gen = batch_dict['fea_pt_dual_gen']
    fea_pl_dual_gen = batch_dict['fea_pl_dual_gen']
    fea_kpt_original_gen = batch_dict['fea_kpt_original_gen']
    # fea_kpt_gen_gen=batch_dict['fea_kpt_gen_gen']
    fea_pt_dual = batch_dict['fea_pt_dual']
    fea_pl_dual = batch_dict['fea_pl_dual']
    fea_kpt_original = batch_dict['fea_kpt_original']
    # fea_pt_dual = batch_dict['fea_pt_dual'].detach()
    # fea_pl_dual = batch_dict['fea_pl_dual'].detach()
    # fea_kpt_original = batch_dict['fea_kpt_original'].detach()

    b = fea_pl_dual.shape[0]
    loss0 = 0
    loss1 = 0
    loss2 = 0
    loss3 = 0
    relation = batch_dict['relation']
    nums=0
    for i in range(b):
        cnt = torch.sum((relation[i, :, -1, 0] > 0) & (relation[i, :, -1, 1] > 0))
        nums+=cnt
        fea_pt_dual1 = fea_pt_dual[i, :, :cnt]  # 匹配点云特征，CN
        fea_pt_dual_gen1 = fea_pt_dual_gen[i, :, :cnt]  # 匹配点云特征，生成于图像
        fea_pl_dual1 = fea_pl_dual[i, :, :cnt]  # 匹配图像特征
        fea_pl_dual_gen1 = fea_pl_dual_gen[i, :, :cnt]  # 匹配图像特征，生成于点云

        # loss0 = loss0 + torch.abs(fea_pt_dual1 - fea_pt_dual_gen1).mean()
        loss0 = loss0 + (1 - F.cosine_similarity(fea_pt_dual1,fea_pt_dual_gen1,dim=0)).mean()
        # loss0 = loss0 + F.mse_loss(fea_pt_dual1, fea_pt_dual_gen1)
        # loss0 = loss0 + ((fea_pt_dual1 - fea_pt_dual_gen1).norm(p=2, dim=0)).mean()
        # sims00=tools.batch_distance(fea_pt_dual1.unsqueeze(0).permute(0,2,1),fea_pt_dual1.unsqueeze(0).permute(0,2,1),'cosine')
        # sims01=tools.batch_distance(fea_pt_dual_gen1.unsqueeze(0).permute(0,2,1),fea_pt_dual_gen1.unsqueeze(0).permute(0,2,1),'cosine')
        # loss0 = loss0 + torch.abs(sims00-sims01).mean()
        
        # loss1 = loss1 + torch.abs(fea_pl_dual1 - fea_pl_dual_gen1).mean()
        loss1 = loss1 + (1 - F.cosine_similarity(fea_pl_dual1,fea_pl_dual_gen1,dim=0)).mean()
        # loss1 = loss1 + F.mse_loss(fea_pl_dual1, fea_pl_dual_gen1)
        # loss1 = loss1 + ((fea_pl_dual1 - fea_pl_dual_gen1).norm(p=2, dim=0)).mean()
        # sims10=tools.batch_distance(fea_pl_dual1.unsqueeze(0).permute(0,2,1),fea_pl_dual1.unsqueeze(0).permute(0,2,1),'cosine')
        # sims11=tools.batch_distance(fea_pl_dual_gen1.unsqueeze(0).permute(0,2,1),fea_pl_dual_gen1.unsqueeze(0).permute(0,2,1),'cosine')
        # loss1 = loss1 + torch.abs(sims10-sims11).mean()

        #全景特征生成模块损失计算
        # loss2 = loss2 + torch.abs(fea_kpt_original[i] - fea_kpt_original_gen[i]).mean()
        loss2= loss2 + (1-F.cosine_similarity(fea_kpt_original[i], fea_kpt_original_gen[i],dim=0)).mean()
        # loss2 = loss2 +  F.mse_loss(fea_kpt_original[i], fea_kpt_original_gen[i])
        # loss2 = loss2 + ((fea_kpt_original[i] - fea_kpt_original_gen[i]).norm(p=2, dim=0)).mean()
        # sims20=tools.batch_distance(fea_kpt_original[i:i+1].permute(0,2,1),fea_kpt_original[i:i+1].permute(0,2,1),'cosine')
        # sims21=tools.batch_distance(fea_kpt_original_gen[i:i+1].permute(0,2,1),fea_kpt_original_gen[i:i+1].permute(0,2,1),'cosine')
        # loss2 = loss2 + torch.abs(sims20-sims21).mean()
    loss0 = loss0 / b
    loss1 = loss1 / b
    loss2 = loss2 / b
    return loss0, loss1, loss2, loss3


def sinkhorn_matches_loss(batch_dict,key):
    project_kpts = batch_dict[key]  # calculated from corrspondence of kpts
    src_coords = batch_dict['key_points']
    pose_to_frame = batch_dict['pose_to_frame']
    src_coords = src_coords.clone().view(batch_dict['batch_size'], -1, 4)
    B, N_POINT, _ = src_coords.shape
    B = B // 2
    src_coords = src_coords[:B, :, [0, 1, 2, 3]]
    src_coords[:, :, -1] = 1.
    gt_dst_coords = torch.bmm(pose_to_frame, src_coords.permute(0, 2, 1))  # True project kpts
    gt_dst_coords = gt_dst_coords.permute(0, 2, 1)[:, :, :3]
    loss = (gt_dst_coords - project_kpts).norm(dim=2).mean()
    return loss



def score_loss(batch_dict):
    score = batch_dict['score_bev']
    label_score = batch_dict['label_score']
    label_score = torch.cat([label_score[:, :, :, 0], label_score[:, :, :, 1]], dim=0)
    mask1 = score > 1e-8
    # mask2 = label_score > 1e-8
    # mask = mask1 | mask2
    score = score[mask1]
    label_score = label_score[mask1]
    loss = nn.functional.mse_loss(score, label_score)

    return loss


def pose_loss(batch_dict,key):
    src_coords = batch_dict['key_points']
    src_coords = src_coords.clone().view(batch_dict['batch_size'], -1, 4)
    delta_pose = batch_dict['pose_to_frame']
    B, N_POINT, _ = src_coords.shape
    B = B // 2
    src_coords = src_coords[:B]
    gt_dst_coords = torch.bmm(delta_pose, src_coords.permute(0, 2, 1)).float()
    gt_dst_coords = gt_dst_coords.permute(0, 2, 1)[:, :, :3]
    
    transformation = batch_dict[key]
    pred_dst_coords = torch.bmm(transformation, src_coords.permute(0, 2, 1))
    pred_dst_coords = pred_dst_coords.permute(0, 2, 1)[:, :, :3]
    loss = torch.mean(torch.abs(pred_dst_coords - gt_dst_coords))
    return loss


def get_all_triplets(dist_mat, pos_mask, neg_mask, is_inverted=False, margin=0.5, different_embedding=False):
    if not different_embedding:
        pos_mask = torch.triu(pos_mask, 1)
    triplets = pos_mask.unsqueeze(2) * neg_mask.unsqueeze(1)
    return torch.where(triplets)


def hardest_negative_selector(dist_mat, pos_mask, neg_mask, is_inverted, margin=0.5, different_embedding=False):
    if not different_embedding:
        pos_mask = torch.triu(pos_mask, 1)
    a, p = torch.where(pos_mask)
    if neg_mask.sum() == 0:
        return a, p, None
    if is_inverted:
        dist_neg = dist_mat * neg_mask
        n = torch.max(dist_neg, dim=1)
    else:
        dist_neg = dist_mat.clone()
        dist_neg[~neg_mask] = dist_neg.max() + 1.
        _, n = torch.min(dist_neg, dim=1)
    n = n[a]
    return a, p, n


def random_negative_selector(dist_mat, pos_mask, neg_mask, is_inverted, margin=0.5, different_embedding=False):
    if not different_embedding:
        pos_mask = torch.triu(pos_mask, 1)
    a, p = torch.where(pos_mask)
    selected_negs = []
    for i in range(a.shape[0]):
        possible_negs = torch.where(neg_mask[a[i]])[0]
        if len(possible_negs) == 0:
            return a, p, None

        dist_neg = dist_mat[a[i], possible_negs]
        if is_inverted:
            curr_loss = -dist_mat[a[i], p[i]] + dist_neg + margin
        else:
            curr_loss = dist_mat[a[i], p[i]] - dist_neg + margin

        if len(possible_negs[curr_loss > 0]) > 0:
            possible_negs = possible_negs[curr_loss > 0]
        random_neg = np.random.choice(possible_negs.cpu().numpy())
        selected_negs.append(random_neg)
    n = torch.tensor(selected_negs, dtype=a.dtype, device=a.device)
    return a, p, n


def semihard_negative_selector(dist_mat, pos_mask, neg_mask, is_inverted, margin=0.5, different_embedding=False):
    if not different_embedding:
        pos_mask = torch.triu(pos_mask, 1)
    a, p = torch.where(pos_mask)
    selected_negs = []
    for i in range(a.shape[0]):
        possible_negs = torch.where(neg_mask[a[i]])[0]
        if len(possible_negs) == 0:
            return a, p, None

        dist_neg = dist_mat[a[i], possible_negs]
        if is_inverted:
            curr_loss = -dist_mat[a[i], p[i]] + dist_neg + margin
        else:
            curr_loss = dist_mat[a[i], p[i]] - dist_neg + margin

        semihard_idxs = (curr_loss > 0) & (curr_loss < margin)
        if len(possible_negs[semihard_idxs]) > 0:
            possible_negs = possible_negs[semihard_idxs]
        random_neg = np.random.choice(possible_negs.cpu().numpy())
        selected_negs.append(random_neg)
    n = torch.tensor(selected_negs, dtype=a.dtype, device=a.device)
    return a, p, n


class TripletLoss(nn.Module):
    def __init__(self, margin: float, triplet_selector, distance: distances.BaseDistance):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
        self.distance = distance

    def forward(self, embeddings, pos_mask, neg_mask, other_embeddings=None):
        if other_embeddings is None:
            other_embeddings = embeddings
        dist_mat = self.distance(embeddings, other_embeddings)
        triplets = self.triplet_selector(
            dist_mat, pos_mask, neg_mask, self.distance.is_inverted)
        distance_positive = dist_mat[triplets[0], triplets[1]]
        if triplets[-1] is None:
            if self.distance.is_inverted:
                return F.relu(1 - distance_positive).mean()
            else:
                return F.relu(distance_positive).mean()
        distance_negative = dist_mat[triplets[0], triplets[2]]
        curr_margin = self.distance.margin(
            distance_positive, distance_negative)
        loss = F.relu(curr_margin + self.margin)
        return loss.mean()


def _pairwise_distance(x, squared=False, eps=1e-16):
    # Compute the 2D matrix of distances between all the embeddings.

    cor_mat = torch.matmul(x, x.t())
    norm_mat = cor_mat.diag()
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
    distances = F.relu(distances)

    if not squared:
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


class TotalLoss(nn.Module):
    def __init__(self, cfg):
        super(TotalLoss, self).__init__()
        if 'hardest' == cfg['negetative_selsector']:
            neg_selector = hardest_negative_selector
        elif 'semihard' == cfg['negetative_selsector']:
            neg_selector = semihard_negative_selector
        else:
            neg_selector = random_negative_selector
        self.trip_fun = TripletLoss(margin=cfg['trip_margin'], triplet_selector=neg_selector, distance=distances.LpDistance())
        self.negetative_distcance = 50

    def forward(self, batch_dict):
        l_pose=l_score=l_match=l_tra=l_rot=l_gb=l_gi=l_gpa=l_gpo=l_kpl = 0
        if 'key_points' in batch_dict.keys():
            l_score = score_loss(batch_dict)
        l_match1,l_pose1,l_match2,l_pose2,l_tra1,l_rot1,l_tra2,l_rot2=0,0,0,0,0,0,0,0
        
        if 'transformation_original' in batch_dict.keys():
            l_match1 = sinkhorn_matches_loss(batch_dict,'project_kpts_original')
            l_tra1, l_rot1 = tr_loss(batch_dict,'transformation_original')
            l_pose1 = pose_loss(batch_dict,'transformation_original')
        if  'transformation_fusion' in batch_dict.keys():
            l_match2 = sinkhorn_matches_loss(batch_dict,'project_kpts_fusion')
            l_tra2, l_rot2 = tr_loss(batch_dict,'transformation_fusion')
            l_pose2 = pose_loss(batch_dict,'transformation_fusion')
        cnt=1
        if min(l_rot1,l_rot2)>0:
            cnt=2
        l_match=(l_match1+l_match2)/cnt
        l_pose=(l_pose1+l_pose2)/cnt
        l_tra=(l_tra1+l_tra2)/cnt
        l_rot=(l_rot1+l_rot2)/cnt
        if ('fea_pt_dual_gen' in batch_dict.keys()) or ('fea_pl_dual_gen' in batch_dict.keys()):
            l_gb, l_gi, l_gpa,l_kpl = gen_feature_loss(batch_dict)
        if 'key_points_gen' in batch_dict.keys():
            l_gpo = gen_points_loss(batch_dict)
        if 'sequence' in batch_dict:
            neg_mask = batch_dict['sequence'].view(1, -1) != batch_dict['sequence'].view(-1, 1)
        else:
            neg_mask = torch.zeros((batch_dict['pose_query'].shape[0] * 2, batch_dict['pose_query'].shape[0] * 2), dtype=torch.bool)
        pair_dist = _pairwise_distance(batch_dict['pose_query'][:, 0:3, 3])
        neg_mask = ((pair_dist > self.negetative_distcance) | neg_mask.to(pair_dist.device))
        neg_mask = neg_mask.repeat(2, 2)
        batch_size = batch_dict['batch_size']
        pos_mask = torch.zeros((batch_size, batch_size), device=neg_mask.device)

        for i in range(batch_size // 2):
            pos_mask[i, i + batch_size // 2] = 1
            pos_mask[i + batch_size // 2, i] = 1
        l_triplet = self.trip_fun(batch_dict['vlads'], pos_mask, neg_mask)
        l_total = l_score + l_pose + 0.05 * l_match + l_triplet + (l_gb + l_gi + l_gpa + l_kpl)
        loss = [l_total, l_pose, l_score, l_match, l_triplet, l_tra, l_rot, l_gb, l_gi, l_gpa, l_gpo,l_kpl]
        for i in range(len(loss)):
            if loss[i]==0:
                loss[i]=loss[0]*0
        batch_dict['loss']=loss
        return batch_dict