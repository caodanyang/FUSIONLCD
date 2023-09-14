import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch._utils
import torch.nn as nn
import torch.nn.functional as F
from uot import UOTHead,GenUOTHead
from netvlad import NetVLAD, NetVLADLoupe
from ALIKE.alike import configs
from ALIKE.alnet import ALNet
from BEVNet import RICNN, RECNN, EncodePosition, RIAvgpool2d, RIMaxpool2d
import tools


def simple_nms(scores, nms_radius=2, itertation=2, mode='1'):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)
    if mode == 'ri':
        max_pool = RIMaxpool2d(kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)
    else:
        max_pool = nn.MaxPool2d(kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)

    for _ in range(itertation):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


class BEVHead(nn.Module):
    def __init__(self, alnet='alike-n', iter=5, num_kpt=100, cluster_num=16, vlad_size=256):
        super(BEVHead, self).__init__()
        cfg = configs[alnet]
        self.feature_extractor = ALNet(c1=cfg['c1'], c2=cfg['c2'], c3=cfg['c3'], c4=cfg['c4'], dim=cfg['dim'],
                                       single_head=cfg['single_head'])
        self.feature_size = int(self.feature_extractor.feature_size)
        self.select = 'maxpool'
        self.num_kpt = num_kpt
        self.ep = EncodePosition(feature_size=self.feature_size)
        self.uot = UOTHead(nb_iter=iter,name='original')
        self.netvlad_bev = NetVLAD(self.feature_size, cluster_num)
        # state_dict=torch.load('/data4/caodanyang/results/FUSIONLCD/bev_07250/models/checkpoint_049.pth.tar', map_location='cpu')['model']
        # state_dict_new={}
        # for k,v in state_dict.items():
        #     state_dict_new[k[4:]]=v
        # self.load_state_dict(state_dict_new)
        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, batch_dict):
        assert type(batch_dict) is dict, 'Input should be a dict'
        bev = batch_dict['bev']
        guider = (bev[:, 2:3] > 0).float()
        b, c, h_bev, w_bev = bev.shape
        x = bev[:, 0:3, :, :]
        points = bev[:, 3:7, :, :]  # xyzi
        points[:, 2] = 0
        points[:, 3] = 1

        score_bev, feature_bev = self.feature_extractor(x)
        score_bev = score_bev * guider

        if self.select == 'avgpool':
            avgpool = RIAvgpool2d(kernel_size=5, stride=4, padding=1)
            grid = np.array(np.meshgrid(np.arange(h_bev), np.arange(w_bev))).swapaxes(0, 2)
            grid = torch.from_numpy(grid).to(x.device).permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1)
            score_bev_avg = avgpool(score_bev)
            grid_avg = avgpool(grid.float() * score_bev) / (score_bev_avg + 1e-8)
            grid_avg = torch.round(grid_avg).long().permute(0, 2, 3, 1)
            points_avg = avgpool(score_bev * points) / (score_bev_avg + 1e-8)
            feature_bev_avg = avgpool(feature_bev * score_bev) / (score_bev_avg + 1e-8)
            score_bev = score_bev.view(b, h_bev, w_bev)
            score_bev_avg = score_bev_avg.squeeze(1)
            kpts = []
            feas_kpt = []
            pixels_kpt = []
            # cnt=0
            for i in range(b):
                uv = list(torch.where(score_bev_avg[i] > 0))
                num_kpt = int(self.num_kpt)
                if num_kpt == 0:
                    print('NO BEV key point')
                    exit()
                while len(uv[0]) < num_kpt:
                    uv[0] = torch.cat([uv[0], uv[0][:(num_kpt - len(uv[0]))]])
                    uv[1] = torch.cat([uv[1], uv[1][:(num_kpt - len(uv[1]))]])
                score_bev0 = score_bev_avg[i, uv[0], uv[1]]
                score_bev1, idx = torch.topk(score_bev0, k=self.num_kpt)
                # cnt=max(cnt,len(uv[0]))
                # idx=torch.arange(len(uv[0])).to(x.device)
                pc = points_avg[i, :, uv[0], uv[1]].permute(1, 0)
                # pc = torch.cat([pc, pc * 0], dim=1)
                kpt = pc[idx]
                fea_kpt = feature_bev_avg[i, :, uv[0][idx], uv[1][idx]]
                pixel_kpt = grid_avg[i, uv[0][idx], uv[1][idx]]
                pixels_kpt.append(pixel_kpt)
                kpts.append(kpt.unsqueeze(0))
                feas_kpt.append(fea_kpt.unsqueeze(0))
        else:
            score_bev_max = simple_nms(score_bev, nms_radius=3)
            score_bev = score_bev.view(b, h_bev, w_bev)
            score_bev_max = score_bev_max.view(b, h_bev, w_bev)
            kpts = []
            feas_kpt = []
            pixels_kpt = []
            for i in range(b):
                uv = list(torch.where((score_bev[i] == score_bev_max[i]) & (score_bev[i] > 0)))
                num_kpt = int(self.num_kpt)
                if num_kpt == 0:
                    print('NO BEV key point')
                    exit()
                while len(uv[0]) < num_kpt:
                    uv[0] = torch.cat([uv[0], uv[0][:(num_kpt - len(uv[0]))]])
                    uv[1] = torch.cat([uv[1], uv[1][:(num_kpt - len(uv[1]))]])
                score_bev0 = score_bev[i, uv[0], uv[1]]
                # sc0 = score_bev0.cpu().detach().numpy()
                score_bev1, idx = torch.topk(score_bev0, k=self.num_kpt)
                pc = points[i, :, uv[0], uv[1]].permute(1, 0)
                # pc = torch.cat([pc, pc * 0], dim=1)
                kpt = pc[idx]
                fea_kpt = feature_bev[i, :, uv[0][idx], uv[1][idx]]
                pixel_kpt = torch.cat([uv[0][idx], uv[1][idx]]).view(2, -1).T
                pixels_kpt.append(pixel_kpt.unsqueeze(0))
                kpts.append(kpt.unsqueeze(0))
                feas_kpt.append(fea_kpt.unsqueeze(0))

        # kpts1=torch.zeros((b,cnt,kpt.shape[1])).to(x.device)
        # feas_kpt1=torch.zeros((b,fea_kpt.shape[0],cnt)).to(x.device)
        # for i in range(b):
        #     kpts1[i,:kpts[i].shape[1]]=kpts[i].squeeze(0)
        #     feas_kpt1[i,:,:feas_kpt[i].shape[2]]=feas_kpt[i].squeeze(0)

        kpts = torch.cat(kpts)
        feas_kpt = torch.cat(feas_kpt)
        pixels_kpt = torch.cat(pixels_kpt)
        if hasattr(self, 'ep'):
            feas_kpt = self.ep(kpts, feas_kpt)
        batch_dict['pixels_kpt'] = pixels_kpt
        batch_dict['score_bev'] = score_bev
        batch_dict['fea_kpt_original'] = feas_kpt
        batch_dict['fea_bev'] = feature_bev
        batch_dict['key_points'] = kpts
        if hasattr(self, 'netvlad_bev'):
            try:
                vlad_bev = self.netvlad_bev(feas_kpt.transpose(1, 2).contiguous())
            except:
                vlad_bev = self.netvlad_bev(feas_kpt.unsqueeze(3))
            batch_dict['vlad_bev'] = vlad_bev
        if ('pose_to_frame' in batch_dict.keys()) and (hasattr(self, 'uot')):
            self.uot(batch_dict)
        ####################################    show bev and kpt     ############################################
        if 0:
            for i in range(b):
                bevshow = x[i].permute(1, 2, 0).cpu().detach().numpy()
                bevshow = np.ascontiguousarray(bevshow[:, :, 0:3] * 255, dtype=np.uint8)
                bevshow1 = bevshow.copy()
                bevshow1[:, 1] = [255, 255, 255]
                for j in range(kpt.shape[0]):
                    center = (int(uv[1][idx[j]].cpu().detach().numpy()), int(uv[0][idx[j]].cpu().detach().numpy()))
                    cv2.circle(bevshow1, center, 2, (0, 0, 255), -1, cv2.LINE_AA)
                bevshow2 = np.hstack((bevshow, bevshow1))
                # cv2.namedWindow('2', cv2.WINDOW_NORMAL)
                # cv2.imshow('2', bevshow2)
                # cv2.waitKey(0)
                fig = plt.figure()
                plt.imshow(bevshow2)
                plt.show()
        #########################################################################################################

        ####################################        show match       ############################################
        if 0:
            for i in range(b // 2):
                kpt1 = kpts[i]
                pose_to_frame = batch_dict['pose_to_frame'][i]
                # pose_to_frame = batch_dict['transformation'][i]
                # pose_to_frame = torch.cat((pose_to_frame, torch.tensor([0, 0, 0, 1]).view(1, 4).to(pose_to_frame.device)))
                kpt1 = (pose_to_frame @ kpt1.permute(1, 0)).permute(1, 0)
                kpt2 = kpts[i + b // 2]
                bev1 = batch_dict['bev'][i][0:3].permute(1, 2, 0)
                bev1 = np.ascontiguousarray(bev1.cpu().detach().numpy() * 255, dtype=np.uint8)
                bev2 = batch_dict['bev'][i + b // 2][0:3].permute(1, 2, 0)
                bev2 = np.ascontiguousarray(bev2.cpu().detach().numpy() * 255, dtype=np.uint8)
                pixel1 = pixels_kpt[i].cpu().detach().numpy()
                pixel2 = pixels_kpt[i + b // 2].cpu().detach().numpy()
                fea1 = feas_kpt[i].permute(1, 0).cpu().detach().numpy()
                fea2 = feas_kpt[i + b // 2].permute(1, 0).cpu().detach().numpy()
                idx1, idx2, dis = tools.nn_match(fea1, fea2, 'cosine')
                # idx11, idx21, dis1 = tools.nn_match(kpt1, kpt2, 'euclidean')
                # idx1 = idx1[dis < 0.1]
                # idx2 = idx2[dis < 0.1]
                h, w, _ = bev1.shape
                img = np.hstack((bev1, bev2))
                img[:, w] = [255, 255, 255]
                tp = 0
                img1 = img.copy()
                for j in range(len(pixel1)):
                    center1 = (int(pixel1[j, 1]), int(pixel1[j, 0]))
                    center2 = (int(pixel2[j, 1]) + w, int(pixel2[j, 0]))
                    cv2.circle(img, center1, 2, (155, 155, 155), -1, cv2.LINE_AA)
                    cv2.circle(img, center2, 2, (155, 155, 155), -1, cv2.LINE_AA)
                for j in range(len(idx1)):
                    center1 = (int(pixel1[idx1[j], 1]), int(pixel1[idx1[j], 0]))
                    center2 = (int(pixel2[idx2[j], 1]) + w, int(pixel2[idx2[j], 0]))
                    dis_kpt = (kpt1[idx1[j]] - kpt2[idx2[j]]).norm(p=2)
                    if dis_kpt < 2:
                        tp = tp + 1
                        cv2.line(img, center1, center2, (0, 166, 0), 1, cv2.LINE_AA)
                    else:
                        cv2.line(img, center1, center2, (0, 0, 188), 1, cv2.LINE_AA)
                    cv2.circle(img, center1, 2, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(img, center2, 2, (255, 255, 255), -1, cv2.LINE_AA)
                # print(np.arccos(pose_to_frame.cpu().detach().numpy()[0, 0]) / np.pi * 180, (tp / len(idx1)))
                img2 = np.vstack((img1, img))
                img2[h, :] = [255, 255, 255]
                cv2.namedWindow('bev match %.3f,%.1fdeg' % (tp / len(idx1), np.arccos(pose_to_frame.cpu().detach().numpy()[0, 0]) / np.pi * 180))
                cv2.imshow('bev match %.3f,%.1fdeg' % (tp / len(idx1), np.arccos(pose_to_frame.cpu().detach().numpy()[0, 0]) / np.pi * 180), img2)
                cv2.waitKey(0)
        #####################################################################################################

        ############################################  ICP  ##################################################
        if 0:
            import open3d as o3d
            for i in range(b // 2):
                pose_to_frame = batch_dict['pose_to_frame'][i].cpu().detach().numpy()
                print('angle', np.arccos(pose_to_frame[0, 0]) / 3.14 * 180)
                transformation = batch_dict['transformation'][i].cpu().detach().numpy()
                transformation = np.vstack((transformation, [0, 0, 0, 1]))
                scan1 = batch_dict['scan_query'][i].cpu().detach().numpy()
                scan2 = batch_dict['scan_positive'][i].cpu().detach().numpy()

                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(scan1[:, :3])
                pcd1.colors = o3d.utility.Vector3dVector([[0, 0, 1] for i in range(len(pcd1.points))])

                pcd11 = o3d.geometry.PointCloud()
                pcd11.points = o3d.utility.Vector3dVector(scan1[:, :3])
                pcd11.colors = o3d.utility.Vector3dVector([[0, 0, 1] for i in range(len(pcd1.points))])

                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(scan2[:, :3])
                pcd2.colors = o3d.utility.Vector3dVector([[0, 1, 0] for i in range(len(pcd2.points))])

                icp_config = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200, relative_fitness=1e-6,
                                                                               relative_rmse=1e-6)
                trans_init = transformation
                threshold = 2
                estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
                registration_result = o3d.pipelines.registration.registration_icp(pcd1, pcd2, threshold, trans_init,
                                                                                  estimation_method, icp_config)

                # 将待配准点云应用变换
                pcd1.transform(registration_result.transformation)

                vis1 = o3d.visualization.Visualizer()
                vis1.create_window(window_name='registration', width=600, height=600)  # 创建窗口
                render_option: o3d.visualization.RenderOption = vis1.get_render_option()  # 设置点云渲染参数
                render_option.background_color = np.array([1, 1, 1])  # 设置背景色（这里为黑色）
                render_option.point_size = 2  # 设置渲染点的大小
                vis1.add_geometry(pcd11)
                vis1.run()

                vis2 = o3d.visualization.Visualizer()
                vis2.create_window(window_name='registration', width=600, height=600)  # 创建窗口
                render_option: o3d.visualization.RenderOption = vis2.get_render_option()  # 设置点云渲染参数
                render_option.background_color = np.array([1, 1, 1])  # 设置背景色（这里为黑色）
                render_option.point_size = 2  # 设置渲染点的大小
                vis2.add_geometry(pcd2)
                vis2.run()

                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name='registration', width=600, height=600)  # 创建窗口
                render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
                render_option.background_color = np.array([1, 1, 1])  # 设置背景色（这里为黑色）
                render_option.point_size = 2  # 设置渲染点的大小
                vis.add_geometry(pcd1)
                vis.add_geometry(pcd2)
                vis.run()
        #######################################################################################################

        return batch_dict


class ImgHead(nn.Module):
    def __init__(self, alnet='alike-n', num_kpt=150, cluster_num=0,vlad_size=256):
        super(ImgHead, self).__init__()
        cfg = configs[alnet]
        self.feature_extractor = ALNet(c1=cfg['c1'], c2=cfg['c2'], c3=cfg['c3'], c4=cfg['c4'], dim=cfg['dim'],
                                       single_head=cfg['single_head'])
        self.feature_size = int(self.feature_extractor.feature_size)
        # try:
        #     model_path = cfg['model_path']
        # except:
        #     model_path = ''
        # if model_path != '':
        #     state_dict = torch.load(model_path)
        #     self.feature_extractor.load_state_dict(state_dict)
            # for param in self.feature_extractor.parameters():
            #     param.requires_grad = False
        if num_kpt>0:
            self.num_kpt = num_kpt

    def forward(self, batch_dict):
        x = batch_dict['img'][:, 0:3].float() / 255.0
        # x=x[:,:,:,384:768,]
        # pixels = batch_dict['img'][:, 3:5]
        b, c, h, w = x.shape
        pixel_features = []
        kpts = []
        scores = []
        score_img, feature_img = self.feature_extractor(x)
        # feature_img=feature_img*0
        if hasattr(self,'num_kpt') :
            score_img = simple_nms(score_img, 2, 2)
            s_thr = 0.1
            for i in range(b):
                score_global1 = score_img[i, 0]
                values, indices = torch.topk(score_global1.view(-1), k=self.num_kpt, dim=0, largest=True)
                if torch.max(values) < s_thr:
                    print('0 pixel')
                    exit()
                num_low_value = torch.sum(values < s_thr)
                if num_low_value > 0:
                    indices1 = indices.clone()
                    indices1[(self.num_kpt - num_low_value):] = indices[:num_low_value]
                    indices = indices1
                row = torch.div(indices, score_global1.shape[1], rounding_mode='trunc')
                col = indices % score_global1.shape[1]
                pixel_feature = feature_img[i:i + 1, :, row, col]
                pixel_features.append(pixel_feature)
                kpts.append(torch.cat([row.view(1, -1, 1), col.view(1, -1, 1)], dim=2))
                scores.append(values.view(1, -1))
            pixel_features = torch.cat(pixel_features)
            kpts = torch.cat(kpts)
            scores = torch.cat(scores)
        ####################################        show match       ############################################
        if 0:
            for i in range(b // 2):
                img1 = batch_dict['img'][i][0:3].permute(1, 2, 0)
                img1 = np.ascontiguousarray(img1.cpu().detach().numpy(), dtype=np.uint8)
                img2 = batch_dict['img'][i + b // 2][0:3].permute(1, 2, 0)
                img2 = np.ascontiguousarray(img2.cpu().detach().numpy(), dtype=np.uint8)
                pixel1 = kpts[i].cpu().detach().numpy()
                pixel2 = kpts[i + b // 2].cpu().detach().numpy()
                fea1 = pixel_features[i].permute(1, 0).cpu().detach().numpy()
                fea2 = pixel_features[i + b // 2].permute(1, 0).cpu().detach().numpy()
                idx1, idx2, dis = tools.nn_match(fea1, fea2, 'euclidean')
                idx1 = idx1[dis < 10]
                idx2 = idx2[dis < 10]
                h, w, _ = img1.shape
                img = np.vstack((img1, img2))
                img[h, :] = [255, 255, 255]
                for i in range(len(idx1)):
                    center1 = (int(pixel1[idx1[i], 1]), int(pixel1[idx1[i], 0]))
                    center2 = (int(pixel2[idx2[i], 1]), int(pixel2[idx2[i], 0] + h))
                    cv2.line(img, center1, center2, (0, 188, 0), 1, cv2.LINE_AA)
                    cv2.circle(img, center1, 2, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(img, center2, 2, (0, 0, 255), -1, cv2.LINE_AA)
                fig = plt.figure()
                plt.imshow(img[:, :, [2, 1, 0]])
                plt.show()
                # cv2.namedWindow('img match')
                # cv2.imshow('img match', img)
                # cv2.waitKey(0)
        
        #########################################################################################################
        batch_dict['key_pixels'] = kpts
        batch_dict['fea_kpl'] = pixel_features
        batch_dict['fea_img'] = feature_img
        batch_dict['score_img'] = score_img
        if hasattr(self, 'netvlad_img'):
            vlad = self.netvlad_img(pixel_features.transpose(1, 2).contiguous())
            batch_dict['vlad_img'] = vlad

        return batch_dict

class LocalPool(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv1 = nn.Conv2d(100, 10, 1, 1, 0, bias=True)
        self.mp=nn.MaxPool2d((1, 10))

    def forward(self, x):
        b, c, n, k = x.shape #k=100
        x1 = x.permute(0, 3, 2, 1)  # b,k,n,c
        x2=self.conv1(x1)
        x3=x2.permute(0,3,2,1)
        x4=self.mp(x3)
        return x4  # bcn1

class TransformerEncoder(nn.Module):
    def __init__(self, in_c=128, num_heads=4, dropout=0.1, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_c, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        y = self.encoder(x)
        return y



class Attention(nn.Module):
    def __init__(self, d_model):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fnn = nn.Linear(d_model, d_model)
        # self.dp=nn.Dropout(0.1)

    def forward(self, q, k=None, v=None):
        proj_q = self.w_q(q)  # BNC
        proj_k = self.w_k(k)
        proj_w = self.w_v(v)
        # proj_q=self.dp(proj_q)
        # proj_k=self.dp(proj_k)
        # proj_w=self.dp(proj_w)
        weights = nn.functional.softmax(torch.matmul(proj_q, proj_k.transpose(-2, -1)) / (self.d_model ** 0.5), dim=-1)
        attn_output = torch.matmul(weights, proj_w).contiguous()
        output = self.fnn(attn_output)
        return output, weights

# class Generator(nn.Module):
#     def __init__(self, in_c=128, num=150):
#         super().__init__()
#         self.mha = nn.MultiheadAttention(embed_dim=in_c, num_heads=1, dropout=0.1, bias=True, batch_first=True)
#         self.conv1 = nn.Sequential(
#             nn.ConvTranspose1d(in_c, in_c, kernel_size=3, stride=3, padding=0),
#             nn.AdaptiveMaxPool1d(num)
#         )

#     def forward(self, x):
#         b, c, n = x.shape
#         # x=x.detach()
#         x1 = x.permute(0, 2, 1)  # BNC
#         x2, _ = self.mha(x1, x1, x1)
#         x2 = x2.permute(0, 2, 1)
#         x3 = self.conv1(x2)
#         return x3


# class Converter(nn.Module):
#     def __init__(self, in_c=128):
#         super().__init__()
#         self.mha = nn.MultiheadAttention(embed_dim=in_c, num_heads=1, dropout=0.1, bias=True, batch_first=True)
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(in_c, in_c, kernel_size=1, stride=1, padding=0),# nn.BatchNorm1d(in_c), nn.ReLU(),
#             nn.Conv1d(in_c, in_c // 4, kernel_size=1, stride=1, padding=0),# nn.BatchNorm1d(in_c // 4), nn.ReLU(),
#             nn.Conv1d(in_c // 4, in_c // 8, kernel_size=1, stride=1, padding=0),# nn.BatchNorm1d(in_c // 8), nn.ReLU(),
#             nn.Conv1d(in_c // 8, in_c // 4, kernel_size=1, stride=1, padding=0),# nn.BatchNorm1d(in_c // 4), nn.ReLU(),
#             nn.Conv1d(in_c // 4, in_c, kernel_size=1, stride=1, padding=0),# nn.BatchNorm1d(in_c), nn.ReLU(),
#             nn.Conv1d(in_c, in_c, kernel_size=1, stride=1, padding=0)
#         )
#         self.conv2 = nn.Conv1d(in_c * 2, in_c, 1, 1, 0, bias=False)

#     def forward(self, x):
#         # return x
#         b, c, n = x.shape
#         # x=x.detach()
#         mask = (x == 0).all(dim=1)
#         x1 = x.permute(0, 2, 1)  # BNC
#         x2, _ = self.mha(x1, x1, x1, mask)
#         x2 = x2.permute(0, 2, 1)
#         x3 = self.conv1(x)
#         x4=torch.cat([x2,x3],dim=1)
#         x5=self.conv2(x4)
#         x5 = x5.masked_fill(mask.unsqueeze(1), 0)
#         return x5

# class FusionHead(nn.Module):
#     def __init__(self, in_c=128):
#         super().__init__()
#         self.mha1 = nn.MultiheadAttention(in_c, 1, 0.1)
#         self.mha2 = nn.MultiheadAttention(in_c, 1, 0.1)
#         self.conv1 = nn.Conv1d(in_c * 2, in_c, 1)

#     def forward(self, x):
#         fea_kpt = x[:, :, 0]
#         fea_kpt_pan_gen = x[:, :, 1]
#         fea_kpt_gen_gen = x[:, :, 2]
#         fea_kpl_gen = x[:, :, 3]
#         B, C, K, N = x.shape
#         x1 = x[:, :, :3]  # BC3N
#         x2 = x1.permute(2, 0, 3, 1).contiguous()
#         x3 = x2.view(3, B * N, C)
#         x4, _ = self.mha1(x3, x3, x3)
#         x5 = torch.max(x4, dim=0)[0]
#         x6 = x5.view(B, N, C).permute(1, 0, 2)
#         x7, _ = self.mha2(x6, fea_kpl_gen.permute(2, 0, 1), fea_kpl_gen.permute(2, 0, 1))
#         x7 = x7.permute(1, 2, 0)
#         x8 = torch.cat([fea_kpt, x7] ,dim=1)
#         x9 = self.conv1(x8)
#         return x9
    

class Generator(nn.Module):
    def __init__(self, in_c=128, num=150):
        super().__init__()
        self.mha = Attention(in_c)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(in_c, in_c, kernel_size=3, stride=3, padding=0),
            nn.AdaptiveMaxPool1d(num)
        )

    def forward(self, x):
        b, c, n = x.shape
        # x=x.detach()
        x1 = x.permute(0, 2, 1)  # BNC
        x2, _ = self.mha(x1, x1, x1)
        x2 = x2.permute(0, 2, 1)
        x3 = self.conv1(x2)
        return x3


class Converter(nn.Module):
    def __init__(self, in_c=128):
        super().__init__()
        self.mha = Attention(in_c)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_c, in_c, kernel_size=1, stride=1, padding=0),# nn.BatchNorm1d(in_c), nn.ReLU(),
            nn.Conv1d(in_c, in_c // 4, kernel_size=1, stride=1, padding=0),# nn.BatchNorm1d(in_c // 4), nn.ReLU(),
            nn.Conv1d(in_c // 4, in_c // 8, kernel_size=1, stride=1, padding=0),# nn.BatchNorm1d(in_c // 8), nn.ReLU(),
            nn.Conv1d(in_c // 8, in_c // 4, kernel_size=1, stride=1, padding=0),# nn.BatchNorm1d(in_c // 4), nn.ReLU(),
            nn.Conv1d(in_c // 4, in_c, kernel_size=1, stride=1, padding=0),# nn.BatchNorm1d(in_c), nn.ReLU(),
            nn.Conv1d(in_c, in_c, kernel_size=1, stride=1, padding=0)
        )
        self.conv2 = nn.Conv1d(in_c * 2, in_c, 1, 1, 0, bias=False)

    def forward(self, x):
        # return x
        b, c, n = x.shape
        # x=x.detach()
        mask = (x == 0).all(dim=1)
        x1 = x.permute(0, 2, 1)  # BNC
        x2, _ = self.mha(x1, x1, x1)
        x2 = x2.permute(0, 2, 1)
        x3 = self.conv1(x)
        x4=torch.cat([x2,x3],dim=1)
        x5=self.conv2(x4)
        x5 = x5.masked_fill(mask.unsqueeze(1), 0)
        return x5

class FusionHead(nn.Module):
    def __init__(self, in_c=128):
        super().__init__()
        self.mha1 = Attention(in_c)
        self.mha2 = Attention(in_c)
        self.conv1 = nn.Conv1d(in_c * 2, in_c, 1)

    def forward(self, x):
        fea_kpt = x[:, :, 0]
        fea_kpl_gen = x[:, :, 3]
        B, C, K, N = x.shape
        x1 = x[:, :, :3]  # BC3N
        x2 = x1.permute(0, 3, 2, 1).contiguous()#BN3C
        x3 = x2.view(B * N, 3, C)
        x4, _ = self.mha1(x3, x3, x3)
        x5 = torch.max(x4, dim=1)[0]#B*N 3 C
        x6=x5.view(B,N,C)
        x7, _ = self.mha2(x6, fea_kpl_gen.permute(0, 2, 1), fea_kpl_gen.permute(0, 2, 1))
        x7 = x7.permute(0, 2, 1)
        x8 = torch.cat([fea_kpt, x7] ,dim=1)
        x9 = self.conv1(x8)
        return x9


def cosine_similarity(feature1, feature2):
    # BNC
    feature1 = feature1 / torch.sqrt(torch.sum(feature1 ** 2, -1, keepdim=True) + 1e-8)
    feature2 = feature2 / torch.sqrt(torch.sum(feature2 ** 2, -1, keepdim=True) + 1e-8)
    C = torch.bmm(feature1, feature2.transpose(1, 2))
    # distance_matrix = torch.sum(feature1 ** 2, -1, keepdim=True)
    # distance_matrix = distance_matrix + torch.sum(feature2 ** 2, -1, keepdim=True).transpose(1, 2)
    # distance_matrix = distance_matrix - 2 * torch.bmm(feature1, feature2.transpose(1, 2))  # c^2=a^2+b^2-2abcos
    # C = distance_matrix ** 0.5
    return C


class Fusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        flag = cfg['flag']
        self.flag = flag
        if flag == 'fusion':
            self.img = ImgHead(alnet='alike-n', num_kpt=cfg['kpts_number_img'],
                               cluster_num=cfg['cluster_num_img'], vlad_size=cfg['vlad_size'])
            self.bev = BEVHead(alnet='alike-n', iter=cfg['sinkhorn_iter'],
                               num_kpt=cfg['kpts_number_bev'], cluster_num=cfg['cluster_num_bev'], vlad_size=cfg['vlad_size'])
            assert self.img.feature_size == self.bev.feature_size, 'img feature and image feature should be the same size'
            feature_size = self.img.feature_size
            self.localpool = LocalPool(feature_size)
            self.cvt_img = Converter(feature_size)
            self.cvt_bev = Converter(feature_size)
            self.gen_pan = Generator(feature_size, cfg['kpts_number_bev'])
            self.att_fusion = FusionHead(feature_size)
            # self.netvlad_fusion = NetVLADLoupe(feature_size, cfg['cluster_num_fusion'], cfg['vlad_size'])
            self.netvlad_fusion = NetVLAD(feature_size, cfg['cluster_num_fusion'])
            self.uot = UOTHead(nb_iter=cfg['sinkhorn_iter'],name='fusion')
            self.vlad='fusion'
            self.w= torch.nn.Parameter(torch.zeros(1))

        if flag == 'bev':
            self.bev = BEVHead(alnet='alike-n',iter=cfg['sinkhorn_iter'], num_kpt=cfg['kpts_number_bev'], cluster_num=cfg['cluster_num_bev'], vlad_size=256)
        if flag == 'img':
            self.img = ImgHead(alnet='alike-n', num_kpt=cfg['kpts_number_img'], cluster_num=cfg['cluster_num_img'], vlad_size=cfg['vlad_size'])

    def forward(self, batch_dict):
        if self.flag == 'fusion':
            batch_dict = self.img(batch_dict)
            batch_dict = self.bev(batch_dict)
            fea_img = batch_dict['fea_img']
            fea_bev = batch_dict['fea_bev']
            relation = batch_dict['relation']
            fea_kpt_original = batch_dict['fea_kpt_original']
            # fea_kpl = batch_dict['fea_kpl']
            # pixel_kpt = batch_dict['pixels_kpt']
            b, n1, n2, _ = relation.shape
            n2 = n2 - 1
            # ns=torch.sum((relation[:,:,-1]>0).all(dim=2),dim=1)
            # n_least=torch.min(ns)
            # n_least=min(n_least,256)
            # relation1=[]
            # for i in range(b):
            #     idx=torch.randperm(ns[i])[:n_least].to(relation.device)
            #     relation1.append(relation[i:i+1,idx])
            # relation1=torch.cat(relation1)
            # relation=relation1  
            pixel_img = relation[:, :, 0:n2].clone()
            grid_img = pixel_img[:, :, :, [1, 0]].float() / torch.tensor([fea_img.shape[3] - 1, fea_img.shape[2] - 1]).to(fea_img.device).float() * 2 - 1
            fea_pl_dual = F.grid_sample(fea_img, grid_img, align_corners=True, mode='bilinear', padding_mode='zeros')
            fea_pl_dual = self.localpool(fea_pl_dual).squeeze(3)
            fea_pt_dual_gen = self.cvt_bev(fea_pl_dual)

            if 'pose_to_frame' in batch_dict.keys() and hasattr(self, 'uot'):
                pixel_bev = relation[:, :, n2:n2 + 1, 0:2].clone()
                grid_bev = pixel_bev[:, :, :, [1, 0]].float() / torch.tensor([fea_bev.shape[3] - 1, fea_bev.shape[2] - 1]).to(fea_bev.device).float() * 2 - 1
                fea_pt_dual = (F.grid_sample(fea_bev, grid_bev, align_corners=True, mode='bilinear', padding_mode='zeros')).squeeze(3)
                fea_pl_dual_gen = self.cvt_img(fea_pt_dual)
                batch_dict['fea_pt_dual_gen'] = fea_pt_dual_gen
                batch_dict['fea_pl_dual_gen'] = fea_pl_dual_gen
                batch_dict['fea_pt_dual'] = fea_pt_dual
                batch_dict['fea_pl_dual'] = fea_pl_dual

            fea_kpt_original_gen = self.gen_pan(fea_pt_dual_gen)
            batch_dict['fea_kpt_original_gen'] = fea_kpt_original_gen
            fea_kpl_gen = self.cvt_img(fea_kpt_original)
            fea_kpt_gen_gen = self.cvt_bev(fea_kpl_gen)
            batch_dict['fea_kpt_gen_gen'] = fea_kpt_gen_gen
            batch_dict['fea_kpl_gen']=fea_kpl_gen
            fea_kpts = torch.cat([fea_kpt_original.unsqueeze(2), fea_kpt_original_gen.unsqueeze(2), fea_kpt_gen_gen.unsqueeze(2), fea_kpl_gen.unsqueeze(2)], dim=2)
            fea_kpt_fusion = self.att_fusion(fea_kpts)
            batch_dict['fea_kpt_fusion'] = fea_kpt_original

            # sim10 = cosine_similarity(fea_pt_dual.permute(0, 2, 1), fea_pt_dual.permute(0, 2, 1))[0].cpu().detach().numpy()
            # sim11 = cosine_similarity(fea_pt_dual_gen.permute(0, 2, 1), fea_pt_dual_gen.permute(0, 2, 1))[0].cpu().detach().numpy()
            # sim20 = cosine_similarity(fea_pl_dual.permute(0, 2, 1), fea_pl_dual.permute(0, 2, 1))[0].cpu().detach().numpy()
            # sim21 = cosine_similarity(fea_pl_dual_gen.permute(0, 2, 1), fea_pl_dual_gen.permute(0, 2, 1))[0].cpu().detach().numpy()
            # sim30 = cosine_similarity(fea_kpt_original.permute(0, 2, 1), fea_kpt_original.permute(0, 2, 1))[0].cpu().detach().numpy()
            # sim31 = cosine_similarity(fea_kpt_original_gen.permute(0, 2, 1), fea_kpt_original_gen.permute(0, 2, 1))[0].cpu().detach().numpy()
            # sim32 = cosine_similarity(fea_kpt_gen_gen.permute(0, 2, 1), fea_kpt_gen_gen.permute(0, 2, 1))[0].cpu().detach().numpy()
            # fig=plt.figure()
            # plt.subplot(2, 4, 1), plt.imshow(sim10), plt.title('points')
            # plt.subplot(2, 4, 5), plt.imshow(sim11), plt.title('gen points')
            # plt.subplot(2, 4, 2), plt.imshow(sim20), plt.title('pixel')
            # plt.subplot(2, 4, 6), plt.imshow(sim21), plt.title('gen pixel')
            # plt.subplot(2, 4, 3), plt.imshow(sim30), plt.title('kpt orig')
            # plt.subplot(2, 4, 7), plt.imshow(sim31), plt.title('pan kpt')
            # plt.subplot(2, 4, 4), plt.imshow(sim30), plt.title('kpt orig')
            # plt.subplot(2, 4, 8), plt.imshow(sim32), plt.title('kpt gen gen')
            # plt.show()

            if 'pose_to_frame' in batch_dict.keys() and hasattr(self, 'uot'):
                self.uot(batch_dict)

            vlad_fusion = self.netvlad_fusion(fea_kpt_fusion.unsqueeze(3))
            if self.vlad=='bev':
                batch_dict['vlads']=batch_dict['vlad_bev']
            if self.vlad=='fusion':
                if 'vlad_bev' in batch_dict.keys():
                    batch_dict['vlads']=torch.sigmoid(self.w)*vlad_fusion + (1-torch.sigmoid(self.w))*batch_dict['vlad_bev']
                else:
                    batch_dict['vlads']=vlad_fusion
        if self.flag == 'bev':
            batch_dict = self.bev(batch_dict)
            batch_dict['vlads'] = batch_dict['vlad_bev']
        if self.flag == 'img':
            batch_dict = self.img(batch_dict)
            batch_dict['vlads'] = batch_dict['vlad_img']
        return batch_dict


if __name__ == '__main__':
    b=BEVHead()