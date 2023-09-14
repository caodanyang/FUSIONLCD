import math
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'

import time
import torch
import numpy as np
from tqdm import tqdm
import yaml
import net
import tools
from dataset import KittiTotalLoader
from skimage.measure import ransac
from skimage.transform import SimilarityTransform,EuclideanTransform
import warnings
warnings.filterwarnings("ignore")


def npto_XYZRPY(rotmatrix):
    '''
    Usa mathutils per trasformare una matrice di trasformazione omogenea in xyzrpy
    https://docs.blender.org/api/master/mathutils.html#
    WARNING: funziona in 32bits quando le variabili numpy sono a 64 bit

    :param rotmatrix: np array
    :return: np array with the xyzrpy
    '''

    # qui sotto corrisponde a
    # quat2eul([ 0.997785  -0.0381564  0.0358964  0.041007 ],'XYZ')
    roll = math.atan2(-rotmatrix[1, 2], rotmatrix[2, 2])
    pitch = math.asin(rotmatrix[0, 2])
    yaw = math.atan2(-rotmatrix[0, 1], rotmatrix[0, 0])
    x = rotmatrix[:3, 3][0]
    y = rotmatrix[:3, 3][1]
    z = rotmatrix[:3, 3][2]

    return np.array([x, y, z, roll, pitch, yaw])


def yt_error(pose1, pose2):
    distance = np.linalg.norm((pose1 - pose2)[0:3, 3])
    yaw1 = npto_XYZRPY(pose1)[-1]
    yaw2 = npto_XYZRPY(pose2)[-1]
    yaw1 = yaw1 % (2 * np.pi)
    yaw2 = yaw2 % (2 * np.pi)
    dyaw = abs(yaw1 - yaw2) % (2 * np.pi)
    angle = dyaw * 180 / np.pi
    if angle > 180:
        angle = 360 - angle
    return distance, angle


def rt_error(pose1, pose2):
    r0 = pose1[:3, :3]
    r1 = pose2[:3, :3]
    dr = np.linalg.inv(r0) @ r1
    angle = np.arccos(np.clip((np.trace(dr) - 1) / 2, -1, 1)) * 180 / np.pi
    distance = np.sqrt(np.sum(np.square(pose1[:3, -1] - pose2[:3, -1])))
    return distance, angle


def pose_err(vlads=None, dataset=None, positive_distance=4., kpts=None, feas=None, model=None, num_cand=10):
    poses = dataset.poses
    error_ransac = []
    error_uot = []
    gt = dataset.gt
    pairs = []
    num_reverse = 0
    for i in range(len(gt)):
        sample = gt[i]
        idx_query = sample['idx']
        positive_idxs = sample['positive_idxs']
        for j in range(len(positive_idxs)):
            idx_positive = positive_idxs[j]
            if idx_query < idx_positive:
                pairs.append([idx_query, idx_positive])
                _, r = yt_error(poses[idx_query], poses[idx_positive])
                if r > 90:
                    num_reverse = num_reverse + 1
    # print('Total %d frames, %d pair of loops, %d[%.3f] reverse'%(len(poses),len(pairs),num_reverse,num_reverse/len(pairs)))
    cnt_ransca = 0
    pairs = np.asarray(pairs)
    idx = np.argsort(pairs[:, 1])
    pairs = pairs[idx]
    times_uot=[]
    times_ransac=[]
    for i in tqdm(range(len(pairs)),ncols=60):
        fea_query = feas[pairs[i][0]]
        p1 = kpts[pairs[i][0]].cpu().detach().numpy()
        fea_cand = feas[pairs[i][1]]
        p2 = kpts[pairs[i][1]].cpu().detach().numpy()
        pose_to_frame = np.matmul(np.linalg.inv(poses[pairs[i][1]]), poses[pairs[i][0]])

        st_ransac=time.time()
        idx1, idx2, dis = tools.nn_match(fea_query, fea_cand, 'cosine')
        ransac_flag=False
        if len(idx1) >= 20:
            idx1 = idx1.cpu().detach().numpy()
            idx2 = idx2.cpu().detach().numpy()
            try:
                result_ransac, inliers = ransac((p1[idx1, 0:3], p2[idx2, 0:3]), model_class=EuclideanTransform, min_samples=15, max_trials=3, residual_threshold=1.7)
                num_inlier = np.sum(inliers)
                if num_inlier > 30:  # ransac存在一致的点且无缩
                    cnt_ransca = cnt_ransca + 1
                    error_ransac.append(yt_error(pose_to_frame, result_ransac.params))
                    ransac_flag=True
            except:
                pass
        # if not ransac_flag:
        #     try:
        #         result_ransac, inliers = ransac((p1[idx1, 0:3], p2[idx2, 0:3]), model_class=EuclideanTransform,min_samples=5,max_trials=10,residual_threshold=5)
        #         num_inlier = np.sum(inliers)
        #         cnt_ransca = cnt_ransca + 1
        #         error_ransac.append(yt_error(pose_to_frame, result_ransac.params))
        #     except:
        #         pass
        times_ransac.append(time.time()-st_ransac)

        fea1 = feas[pairs[i], :, :].to(model.epsilon.device).permute(0, 2, 1)
        kpts1 = kpts[pairs[i], :, :].to(model.epsilon.device)
        bd = {'fea_kpt': fea1, 'key_points': kpts1}
        st_uot=time.time()
        bd = model(bd)
        times_uot.append(time.time()-st_uot)
        pose_estimate1 = bd['transformation'].squeeze(0).cpu().detach().numpy()
        pose_estimate1 = np.vstack((pose_estimate1, [0, 0, 0, 1]))
        error_uot.append(yt_error(pose_to_frame, pose_estimate1))

    ransac_rate = cnt_ransca / len(pairs)
    error_ransac = np.asarray(error_ransac)
    error_uot = np.asarray(error_uot)
    # np.save('error_ransac_%02d.npy' % dataset.sequence, error_ransac)
    # np.save('error_uot_%02d.npy' % dataset.sequence, error_uot)
    et_ransac = np.mean(error_ransac[:, 0])
    er_ransac = np.mean(error_ransac[:, 1])
    et_uot = np.mean(error_uot[:, 0])
    er_uot = np.mean(error_uot[:, 1])
    print('uot time: ',np.array(times_uot).mean())
    print('ransac time: ',np.array(times_ransac).mean())
    return et_ransac, er_ransac, et_uot, er_uot, ransac_rate


def estimate_pose(database):
    try:
        with open(os.path.join(os.getcwd(), "config.yaml"), "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        print('Loading config file from %s' % os.path.join(os.getcwd(), "config.yaml"))
    except:
        with open(os.path.join(os.getcwd(), "project/BevNvLcd/config.yaml"), "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        print('Loading config file from %s' % os.path.join(os.getcwd(), "project/BevNvLcd/config.yaml"))
    cfg = cfg['experiment']
    path_result = cfg['path_result']
    _, _, loader_test = KittiTotalLoader(cfg)
    model = net.Fusion(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    checkpoint = torch.load(tools.path_join(path_result, 'models', cfg['last_model']))
    model.load_state_dict(checkpoint['model'])
    uot = model.uot
    # uot = model.bev.uot

    vlads = database['vlads']
    key_points = database['key_points']
    fea_kpt = database['fea_kpt']
    sequences = database['sequences']
    print()
    print('****************************************************************************')
    end = 0
    for i in range(len(loader_test.dataset.datasets)):
        start = end
        end = start + len(loader_test.dataset.datasets[i])
        # end=0
        # start=0
        et_ransac, er_ransac, et_uot, er_uot, rate = pose_err(vlads=vlads[start:end], dataset=loader_test.dataset.datasets[i],
                                                              kpts=key_points[start:end], feas=fea_kpt[start:end], model=uot)
        print('Sequence %02d' % (torch.unique(sequences)[i]))
        print('ransac rate:%.4f, translation error:%.4f[m], rototion error:%.4f[deg]' % (rate, float(et_ransac), float(er_ransac)))
        print('uot translation error:%.4f[m], uot rototion error:%.4f[deg],' % (float(et_uot), float(er_uot)))
    print('****************************************************************************')


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=5 python evaluate_pose.py
    # CUDA_VISIBLE_DEVICES=2 nohup python -u evaluate_pose.py >03090.log 2>&1 &
    # fuser /dev/nvidia*
    database = torch.load('/data4/caodanyang/results/FUSIONLCD/07030/database/database_fusion.pth.tar')
    estimate_pose(database)
