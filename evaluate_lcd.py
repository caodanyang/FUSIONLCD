import os
import time
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import numpy as np
import torch
from skimage.measure import ransac
from skimage.transform import EuclideanTransform
import tools
from tqdm import tqdm
from sklearn.metrics import auc
from sklearn.neighbors import KDTree
import warnings

warnings.filterwarnings("ignore")


def recall_with_candidates(vlads, poses, sequence, recall_num=25, positive_distance=4):
    recall_at_k = [0] * recall_num
    num_with_loop = 0
    if __name__ == '__main__':
        flag = False
    else:
        flag = True
    for i in tqdm(range(0, len(vlads)), disable=flag, ncols=60, desc='Recall@k'):
        valid_idx = list(set(range(0, len(vlads))) - set(range(max(0, i - 50), min(len(vlads), i + 50))))
        valid_idx = torch.tensor(valid_idx).to(vlads.device)
        vlad_query = vlads[i].view(1, -1)
        vlad_valid = vlads[valid_idx]
        dis_valid = torch.linalg.norm((poses[i:i + 1, 0:3, 3] - poses[valid_idx, 0:3, 3]), dim=1)
        min_dis = torch.min(dis_valid)
        if min_dis > positive_distance:
            continue
        num_with_loop = num_with_loop + 1
        # global feature to query quickly
        dis_vlad = torch.cdist(vlad_query, vlad_valid).view(-1, )
        dis, idx_cand = torch.topk(dis_vlad, recall_num, largest=False)
        idx_cand = valid_idx[idx_cand]
        for j in range(recall_num):
            idx_cand1 = idx_cand[j]
            dis = torch.linalg.norm((poses[i:i + 1, 0:3, 3] - poses[idx_cand1, 0:3, 3]), dim=1)
            if dis <= positive_distance:
                recall_at_k[j] = recall_at_k[j] + 1
                break
    time.sleep(1)
    recall_at_k = np.cumsum(recall_at_k) / float(num_with_loop)
    print('Sequence %02d, Recall@' % sequence, end='')
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 19, 24]:
        if i == (len(recall_at_k) - 1):
            print('%d[%.3f]' % (i + 1, recall_at_k[i]))
        else:
            print('%d[%.3f]' % (i + 1, recall_at_k[i]), end=', ')
    return recall_at_k


def retrieve(vlads, feas, kpts, poses, num_cand=1, verify='ransac'):
    loops = []
    if __name__ == '__main__':
        flag = False
    else:
        flag = True
    ts = []
    for i in tqdm(range(0, len(feas)), disable=flag, ncols=60, desc='retrieve loop'):
        t0 = time.time()
        valid_idx = list(set(range(0, len(feas))) - set(range(max(0, i - 50), min(len(feas), i + 50))))
        valid_idx = torch.tensor(valid_idx).to(vlads.device)
        vlad_query = vlads[i].view(1, -1)
        vlad_valid = vlads[valid_idx]
        # global feature to query quickly
        dis_vlad = torch.cdist(vlad_query, vlad_valid).view(-1, )
        dis, idx_cand = torch.topk(dis_vlad, num_cand, largest=False)
        t_retrieve = time.time() - t0
        idx_cand = valid_idx[idx_cand]
        # local feature to qverify
        fea_query = feas[i]
        if verify == 'ransac':
            p1 = kpts[i]
            p1 = p1.cpu().detach().numpy()
            min_dis = torch.tensor([9999])
            idx_detect = idx_cand[0]
            dis_truth = torch.tensor([9999])
            for idx_cand1 in idx_cand:
                fea_cand1 = feas[idx_cand1]
                p2 = kpts[idx_cand1].cpu().detach().numpy()
                idx1, idx2, dis = tools.nn_match(fea_query, fea_cand1, 'cosine')
                if len(idx1) < 31:
                    continue
                idx1 = idx1.cpu().detach().numpy()
                idx2 = idx2.cpu().detach().numpy()
                try:
                    model, inliers = ransac((p1[idx1, 0:2], p2[idx2, 0:2]), model_class=EuclideanTransform, min_samples=15, max_trials=3, residual_threshold=1)
                    num_inlier = np.sum(inliers)
                    # r = model.params[0:2, 0:2]
                    dis_estimate = np.linalg.norm(model.params[0:2, 2])
                    # rot = model.rotation
                    if num_inlier > 30:  # ransac存在足够内点
                        if min_dis > dis_estimate:
                            min_dis = dis_estimate
                            idx_detect = idx_cand1
                            dis_truth = torch.linalg.norm((poses[i, 0:3, 3] - poses[idx_detect, 0:3, 3]))
                except:
                    pass
            loops.append([i, idx_detect.item(), min_dis.item(), dis_truth.item()])

        else:
            idx_detect = idx_cand[0]
            dis_truth = torch.linalg.norm(poses[i, 0:3, 3] - poses[idx_detect, 0:3, 3])
            loops.append([i, idx_detect.item(), dis[0].item(), dis_truth.item()])
        t_verify = time.time() - t0 - t_retrieve

        ts.append([t_retrieve, t_verify])
    #     if loops[-1][2] < 4 and loops[-1][1] < i:
    #         loop1.append(loops[-1][1])
    # x = poses[:, 0, 3]
    # y = poses[:, 1, 3]
    # x1 = x[loop1]
    # y1 = y[loop1]
    # plt.plot(x, y, 'b.', markersize=1)
    # plt.plot(x1, y1, 'ro', markersize=2, markerfacecolor='none')
    # plt.axis('equal')
    # plt.show()
    ts = np.array(ts) * 1000
    # np.savetxt('times.txt', ts)
    # x=np.arange(len(ts))
    # plt.plot(x,ts[:,0],'b.')
    # plt.plot(x,ts[:,1],'r.')
    # plt.show()
    loops = np.array(loops)
    return loops


def pr_curve(poses, loops, sequence, positive_distance=4):
    

    map_tree_poses = KDTree(poses[:, 0:3, 3])
    reverse_loops = []
    real_loop = []
    for i in range(0,len(poses)):
        min_range = max(0, i - 50)
        max_range = min(i + 50, poses.shape[0])
        current_pose = poses[i]
        indices = map_tree_poses.query_radius(np.expand_dims(current_pose[0:3, 3], 0), positive_distance)
        valid_idxs = list(set(indices[0]) - set(range(min_range, max_range)))
        valid_idxs = np.array(valid_idxs)
        if len(valid_idxs) > 0:
            # dis = np.linalg.norm(current_pose[0:3, 3]-poses[valid_idxs,0:3,3],axis=1)
            real_loop.append(1)
            r0 = poses[i, :3, :3]
            rs = poses[valid_idxs, :3, :3]
            dr = np.linalg.inv(r0) @ rs.swapaxes(0, 2)
            angle = np.arccos(np.clip((np.trace(dr) - 1) / 2, -1, 1))
            angle = angle * 180 / np.pi
            if np.min(angle) > 90:
                reverse_loops.append(1)
            else:
                reverse_loops.append(0)
        else:
            real_loop.append(0)
            reverse_loops.append(0)
    reverse_loops = np.array(reverse_loops)
    real_loop = np.array(real_loop)
    # loops=np.hstack((loops,real_loop.reshape(-1,1)))
    # np.savetxt('loops_bev%02d.txt'%sequence,loops,fmt='%.6f')
    # print('sequence %d, %d frames, %d loops, %d reverse loops' % (sequence,len(real_loop), np.sum(real_loop), np.sum(reverse_loops)))
    # # return 0
    distances = loops[:, 3]
    detected_loop = loops[:, 2]
    precision2 = [1]
    recall2 = [0]
    for thr in np.unique(detected_loop):
        tp = detected_loop <= thr
        tp = tp & real_loop
        tp = tp & (distances <= positive_distance)
        tp = tp.sum()
        fp = (detected_loop <= thr).sum() - tp
        fn = (real_loop.sum()) - tp
        if (tp + fp) > 0.:
            precision2.append(tp / (tp + fp))
        else:
            precision2.append(1.)

        recall2.append(tp / (tp + fn))
    f1s = []
    for i in range(len(recall2)):
        f1s.append((2 * precision2[i] * recall2[i]) / (precision2[i] + recall2[i]))
    f1 = max(f1s)
    recall_p1 = np.max(np.array(recall2)[np.array(precision2) == 1])
    # plt.plot(recall2, precision2, 'b-')
    # plt.show()
    pr = np.array(precision2 + recall2).reshape(2, -1).T
    # np.save('fusion_pr_%02d.npy' % sequence, pr)
    ap = auc(recall2, precision2)
    idx=loops[:,2]<9999
    loops1=loops[idx]
    rp=np.sum(np.abs(loops1[:,2]-loops1[:,3])<2)/len(loops1)

    print('Sequence %02d, AP %.3f, Recall@100 %.3f, F1 %.3f, RP %.3f/%d' % (sequence, ap, recall_p1, f1, rp, len(loops1)))
    # if ap<0.1:
    #     exit()
    return ap, recall_p1, f1


def lcd(data):
    vlads = data['vlads'].cuda()
    kpts = data['key_points']
    sequences = data['sequences']
    poses = data['pose_query'].cuda()
    feas = data['fea_kpt'].cuda()
    # feas = feas / torch.sqrt(torch.sum(feas ** 2, -1, keepdim=True) + 1e-8)
    result = []
    recall_at_ks = []
    recall_at_k=[]
    for s in torch.unique(sequences):
        # if s==54:
        #     continue
        mask = sequences == s

        vlads1 = vlads[mask]
        feas1 = feas[mask]
        kpts1 = kpts[mask]
        poses1 = poses[mask]
        poses2 = poses1.cpu().detach().numpy()
        # recall_at_k = recall_with_candidates(vlads1, poses1, s)
        # idx=np.arange(len(vlads1)//2)
        # idx=np.tile(idx, 2)
        # vlads1, feas1, kpts1, poses1 =vlads1[idx], feas1[idx], kpts1[idx], poses1[idx]
        loops = retrieve(vlads1, feas1, kpts1, poses1, 1, 'ransac')
        ap, recall_p1, f1 = pr_curve(poses2, loops, s, 4)
        recall_at_ks.append(recall_at_k)
        result.append([ap, recall_p1, f1])
        
    return result, recall_at_ks


if __name__ == '__main__':
    np.random.seed(123)
    data = torch.load('/data4/caodanyang/results/FUSIONLCD/07030/database/database_bev.pth.tar')
    lcd(data)
    print('----------------------------------------------------------------------')
    data= torch.load('/data4/caodanyang/results/FUSIONLCD/07030/database/database_bevp.pth.tar')
    lcd(data)
    print('----------------------------------------------------------------------')
    data=torch.load('/data4/caodanyang/results/FUSIONLCD/07030/database/database_fusion.pth.tar')
    lcd(data)

