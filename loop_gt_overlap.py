import argparse
import glob
import os
import setproctitle

# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import numba
import torch
from torch.utils.data import Dataset, DataLoader
import numba as nb
from sklearn.neighbors import KDTree
import pickle
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

import tools

skip_frame = 50


@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(points,
                                    voxel_size,
                                    coors_range,
                                    num_points_per_voxel,
                                    coor_to_voxelidx,
                                    voxels,
                                    coors,
                                    max_points,
                                    max_voxels, voxel_idx_empty, voxel_mamiz):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3,), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]  # 0-15000
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxel_idx_empty[voxelidx, num_points_per_voxel[voxelidx]] = 1
            if points[i, 2] > voxel_mamiz[voxelidx, 0]:
                voxel_mamiz[voxelidx, 0] = points[i, 2]
            if points[i, 2] < voxel_mamiz[voxelidx, 1]:
                voxel_mamiz[voxelidx, 1] = points[i, 2]
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1

    return voxel_num


def points_to_voxel(points,
                    voxel_size,
                    coors_range,
                    max_points=35,
                    reverse_index=True,
                    max_voxels=20000):
    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud)
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    voxel_idx_empty = np.zeros((max_voxels, max_points), dtype=np.int32)
    voxel_mamiz = np.zeros((max_voxels, 2), dtype=points.dtype)
    voxel_mamiz[:, 0] = -99
    voxel_mamiz[:, 1] = 99
    voxel_num = _points_to_voxel_reverse_kernel(
        points, voxel_size, coors_range, num_points_per_voxel,
        coor_to_voxelidx, voxels, coors, max_points, max_voxels, voxel_idx_empty, voxel_mamiz)

    # coors = coors[:voxel_num]
    # voxels = voxels[:voxel_num]
    # num_points_per_voxel = num_points_per_voxel[:voxel_num]
    # voxels[:, :, -3:] = voxels[:, :, :3] - \
    #     voxels[:, :, :3].sum(axis=1, keepdims=True)/num_points_per_voxel.reshape(-1, 1, 1)
    return voxels, coors, num_points_per_voxel, coor_to_voxelidx, voxel_idx_empty, voxel_mamiz


def pointcloud_decoder(pointcloud=None, cfg=None):
    try:
        resolution = cfg['bev_resolution']
        pointcloud_range = [float(x) for x in tools.read_cfg(cfg['bev_range'])]
        voxel_max_points = cfg['voxel_max_points']
        voxel_num = cfg['voxel_num']
        voxel_sample = cfg['voxel_sample']
    except:
        resolution = 0.2
        pointcloud_range = [-32, -32, -10, 32, 32, 10]
        voxel_max_points = 100
        voxel_num = 15000
        voxel_sample = 'random'

    pc_filter = (pointcloud[:, 0] > pointcloud_range[0]) & (pointcloud[:, 0] < pointcloud_range[3]) \
                & (pointcloud[:, 1] > pointcloud_range[1]) & (pointcloud[:, 1] < pointcloud_range[4]) \
                & (pointcloud[:, 2] > pointcloud_range[2]) & (pointcloud[:, 2] < pointcloud_range[5])
    pointcloud = pointcloud[pc_filter]
    if voxel_sample == 'top':
        idx = np.argsort(-pointcloud[:, 2])
        pointcloud = pointcloud[idx]
    else:
        idx = np.arange(len(pointcloud))
        np.random.shuffle(idx)
        pointcloud = pointcloud[idx]
    resolution_z = pointcloud_range[5] - pointcloud_range[2]
    voxels, coors, num_points_per_voxel, coor_to_voxelidx, voxel_idx_empty, voxel_mamiz = points_to_voxel(pointcloud,
                                                                                                          voxel_size=[
                                                                                                              resolution, resolution, resolution_z],
                                                                                                          coors_range=[
                                                                                                              *pointcloud_range],
                                                                                                          max_points=voxel_max_points,
                                                                                                          max_voxels=voxel_num)

    coor_to_voxelidx = np.squeeze(coor_to_voxelidx)
    voxel_idx_empty = voxel_idx_empty.astype(np.bool_)

    voxels_center = np.sum(voxels, axis=1)
    voxels_center[:, 0:4] = voxels_center[:, 0:4] / (num_points_per_voxel.reshape(-1, 1) + 1e-8)

    max_z = voxel_mamiz[:, 0]
    min_z = voxel_mamiz[:, 1]

    dz = max_z - min_z
    idx_not_ground = (dz > 0.05) & (num_points_per_voxel > 1)
    voxels_center = voxels_center[idx_not_ground].astype(np.float32)
    return voxels_center


class KITTITruthPositives(Dataset):

    def __init__(self, dir, sequence, poses, positive_range=4.0, use_overlap=True):
        super(KITTITruthPositives, self).__init__()

        self.positive_range = positive_range
        self.use_overlap = use_overlap
        self.dir = dir
        self.sequence = sequence
        path_velo = os.path.join(dir, 'sequences/%02d' % sequence, 'velodyne', "*.bin")
        scans = glob.glob(path_velo)
        scans.sort()
        self.scans = scans
        if int(sequence) > 21:
            self.poses = np.load(poses).astype(np.float32)
        else:
            calib = np.genfromtxt(os.path.join(dir, 'sequences/%02d' % sequence, 'calib.txt'))[:, 1:]
            T_cam_velo = np.reshape(calib[4], (3, 4))
            T_cam_velo = np.vstack([T_cam_velo, [0, 0, 0, 1]])
            poses2 = []
            with open(poses, 'r') as f:
                for x in f:
                    x = x.strip().split()
                    x = [float(v) for v in x]
                    pose = np.zeros((4, 4))
                    pose[0, 0:4] = np.array(x[0:4])
                    pose[1, 0:4] = np.array(x[4:8])
                    pose[2, 0:4] = np.array(x[8:12])
                    pose[3, 3] = 1.0
                    pose = np.linalg.inv(T_cam_velo) @ (pose @ T_cam_velo)
                    poses2.append(pose)
            self.poses = np.stack(poses2).astype(np.float32)
        self.kdtree = KDTree(self.poses[:, :3, 3])

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):

        x = self.poses[idx, 0, 3]
        y = self.poses[idx, 1, 3]
        z = self.poses[idx, 2, 3]
        r0 = self.poses[idx, :3, :3]
        rs = self.poses[:, :3, :3]
        dr = np.linalg.inv(r0) @ rs.swapaxes(0, 2)
        angle = np.arccos(np.clip((np.trace(dr) - 1) / 2, -1, 1))
        angle = angle * 180 / np.pi

        idx_angle = np.where(angle < 99999)[0]
        anchor_pose = torch.tensor([x, y, z])

        indices, dis = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.positive_range, sort_results=True, return_distance=True)
        indices = indices[0]
        min_range = max(0, idx - skip_frame)
        max_range = min(idx + skip_frame, len(self.poses))
        positive_idxs = list(set(indices) & set(idx_angle) - set(range(min_range, max_range)))
        overlaps = []
        if self.use_overlap:
            scan0 = np.fromfile(self.scans[idx], dtype=np.float32).reshape((-1, 4))
            voxel_center0 = pointcloud_decoder(scan0)
            voxel_center0[:, 2] = 0
            voxel_center0[:, 3] = 1
            pose0 = self.poses[idx]
            # voxel_center0 = np.matmul(pose0, voxel_center0.T).T
            voxel_center0 = torch.matmul(torch.from_numpy(pose0), torch.from_numpy(voxel_center0.T)).numpy().T
            voxel_center0 = voxel_center0[:, 0:2]
            positive_idxs1 = []
            for i in positive_idxs:
                scan1 = np.fromfile(self.scans[i], dtype=np.float32).reshape((-1, 4))
                pose1 = self.poses[i]
                voxel_center1 = pointcloud_decoder(scan1)
                voxel_center1[:, 2] = 0
                voxel_center1[:, 3] = 1
                # voxel_center1 = np.matmul(pose1, voxel_center1.T).T
                voxel_center1 = torch.matmul(torch.from_numpy(pose1), torch.from_numpy(voxel_center1.T)).numpy().T
                voxel_center1 = voxel_center1[:, 0:2]
                # idx1, idx2, dis = tools.nn_match(voxel_center0, voxel_center1, 'euclidean')
                # dis = (torch.from_numpy(voxel_center0).unsqueeze(0) - torch.from_numpy(voxel_center1).unsqueeze(1)).norm(p=2, dim=2)
                # dis = torch.cdist(torch.from_numpy(voxel_center0).cuda(), torch.from_numpy(voxel_center1).cuda(), p=2)
                # idx1, idx2, dis = tools.nn_match(voxel_center0, voxel_center1, 'euclidean')
                # dis = (torch.from_numpy(voxel_center0).unsqueeze(0) - torch.from_numpy(voxel_center1).unsqueeze(1)).norm(p=2, dim=2)
                # dis = torch.cdist(torch.from_numpy(voxel_center0).cuda(), torch.from_numpy(voxel_center1).cuda(), p=2)

                voxel_center = np.mean(np.vstack((voxel_center0, voxel_center1)), axis=0, keepdims=True)
                voxel_center0 = voxel_center0 - voxel_center
                voxel_center1 = voxel_center1 - voxel_center
                af = torch.sum(torch.from_numpy(voxel_center0).cuda() ** 2, -1, keepdim=True)
                bf = torch.sum(torch.from_numpy(voxel_center1).cuda() ** 2, -1, keepdim=True).transpose(0, 1)
                cf = af + bf - 2 * torch.mm(torch.from_numpy(voxel_center0).cuda(), torch.from_numpy(voxel_center1).cuda().transpose(0, 1))  # c^2=a^2+b^2-2abcos
                c = torch.sqrt(cf)
                dis = c
                voxel_center0 = voxel_center0 + voxel_center
                voxel_center1 = voxel_center1 + voxel_center

                dis1 = torch.min(dis, dim=0)[0]
                dis2 = torch.min(dis, dim=1)[0]
                idx1 = torch.where(dis1 < 0.5)[0]
                idx2 = torch.where(dis2 < 0.5)[0]
                overlap = ((len(idx1) + len(idx2)) / 2) / (len(voxel_center0) + len(voxel_center1) - (len(idx1) + len(idx2)) / 2)
                dis_pose = np.linalg.norm(self.poses[idx][0:3, 3] - self.poses[i][0:3, 3])
                print('Sequence:%08d, %06d, %06d, overlap:%.3f, dis_pose:%.3f' % (int(self.sequence), idx, i, overlap, dis_pose))
                if overlap > 0.3:
                    overlaps.append(int(overlap * 1000))
                    positive_idxs1.append(i)
                # plt.plot(voxel_center0[:, 0], voxel_center0[:, 1], 'b.', markersize=1)
                # plt.plot(voxel_center1[:, 0], voxel_center1[:, 1], 'r.', markersize=1)
                # plt.xticks([])
                # plt.yticks([])
                # plt.axis('off')
                # plt.axis('equal')
                # plt.title('overlap:%.3f, dis_pose:%.3f' % (overlap, dis_pose))
                # # if not os.path.exists(os.path.join(self.dir, 'sequences/%02d' % sequence, 'overlap')):
                # #     os.makedirs(os.path.join(self.dir, 'sequences/%02d' % sequence, 'overlap'))
                # # path_fig = os.path.join(self.dir, 'sequences/%02d' % sequence, 'overlap', "%06d_%06d.png" % (idx, i))
                # # plt.savefig(path_fig, dpi=100, bbox_inches='tight', pad_inches=0)
                # plt.show()
                # plt.close()
            positive_idxs = positive_idxs1

        loop_angle = angle[positive_idxs]
        reverse = 0
        if len(loop_angle) > 0:
            reverse = np.sum(loop_angle > 90)
            if min(loop_angle) > 90:
                reverse = -1 * reverse
        num_loop = len(positive_idxs)

        return num_loop, positive_idxs, reverse, overlaps


if __name__ == '__main__':
    setproctitle.setproctitle('lgtovp')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--root_folder', default='E:\work\Project\dataset\FUSION', help='dataset directory')
    parser.add_argument('--root_folder', default='/data2/caodanyang/project/dataset/FUSION', help='dataset directory')
    parser.add_argument('--sequences', default=None)
    args = parser.parse_args()
    base_dir = args.root_folder
    strs = []
    if args.sequences is None:
        # sequences=[0, 5, 6, 7, 8, 9, 50, 59, 120205, 130405]
        sequences = [0]
    else:
        sequences = [int(args.sequences)]
    for sequence in sequences:
        if int(sequence) >= 50:
            poses_file = base_dir + "/sequences/%02d" % sequence + "/poses.npy"
        else:
            poses_file = base_dir + "/sequences/%02d" % sequence + "/poses.txt"
        dataset = KITTITruthPositives(base_dir, sequence, poses_file, 32, use_overlap=True)
        data_loader = DataLoader(dataset, 1, shuffle=False, num_workers=0)

        lc_gt = []
        lc_gt_file = os.path.join(base_dir, 'sequences', '%02d' % sequence, 'loop_GT_overlap.pickle')
        loop_pairs = []
        loop_files = []
        for i, data in (enumerate(data_loader)):
            num_loop, positive_idxs, reverse, overlaps = data
            if num_loop > 0.:
                positive_idxs = [p.item() for p in positive_idxs]
                overlaps = [o.item() for o in overlaps]
                loop_files.append([i, reverse.item()])
                sample_dict = {'idx': i, 'positive_idxs': positive_idxs, 'overlaps': overlaps}
                for p in positive_idxs:
                    if i < p:
                        loop_pairs.append([i, p])
                lc_gt.append(sample_dict)
        loop_files = np.array(loop_files)
        num_reverse_file = int(np.sum(loop_files[:, 1] < 0))
        num_reverse_pairs = int(np.sum(np.abs(loop_files[:, 1]))) / 2
        with open(lc_gt_file, 'wb') as f:
            pickle.dump(lc_gt, f)
        strs.append('Sequence %02d done,%05d files, %05d files with loop, %05d[%.4f] files only has reverse loop, %05d loop pairs, %05d[%.4f] reverse loop' %
                    (int(sequence), len(dataset), len(loop_files), num_reverse_file, num_reverse_file / len(loop_files), len(loop_pairs), num_reverse_pairs,
                     num_reverse_pairs / len(loop_pairs)))
    for str1 in strs:
        print(str1)

# overlap
# Sequence 00 done,04541 files, 02234 files with loop, 00058[0.0260] files only has reverse loop, 77070 loop pairs, 16666[0.2162] reverse loop
# Sequence 05 done,02761 files, 01357 files with loop, 00048[0.0354] files only has reverse loop, 45038 loop pairs, 02995[0.0665] reverse loop
# Sequence 06 done,01101 files, 01034 files with loop, 00449[0.4342] files only has reverse loop, 26046 loop pairs, 14567[0.5593] reverse loop
# Sequence 07 done,01101 files, 00418 files with loop, 00026[0.0622] files only has reverse loop, 10841 loop pairs, 00447[0.0412] reverse loop
# Sequence 08 done,04071 files, 01583 files with loop, 00739[0.4668] files only has reverse loop, 33647 loop pairs, 23441[0.6967] reverse loop
# Sequence 09 done,01591 files, 00088 files with loop, 00000[0.0000] files only has reverse loop, 01504 loop pairs, 00000[0.0000] reverse loop
# Sequence 50 done,10514 files, 07786 files with loop, 02894[0.3717] files only has reverse loop, 295232 loop pairs, 198852[0.6735] reverse loop
# Sequence 59 done,13247 files, 10657 files with loop, 02374[0.2228] files only has reverse loop, 496296 loop pairs, 205760[0.4146] reverse loop
# Sequence 120205 done,03078 files, 01685 files with loop, 00730[0.4332] files only has reverse loop, 16213 loop pairs, 10789[0.6655] reverse loop
# Sequence 130405 done,02091 files, 00673 files with loop, 00419[0.6226] files only has reverse loop, 05178 loop pairs, 03967[0.7661] reverse loop


# 4m
# Sequence 00 done,04541 files, 01682 files with loop, 00023[0.0137] files only has reverse loop, 10499 loop pairs, 00105[0.0100] reverse loop
# Sequence 05 done,02761 files, 01046 files with loop, 00002[0.0019] files only has reverse loop, 06534 loop pairs, 00027[0.0041] reverse loop
# Sequence 06 done,01101 files, 00563 files with loop, 00000[0.0000] files only has reverse loop, 02138 loop pairs, 00000[0.0000] reverse loop
# Sequence 07 done,01101 files, 00183 files with loop, 00000[0.0000] files only has reverse loop, 02497 loop pairs, 00000[0.0000] reverse loop
# Sequence 08 done,04071 files, 00654 files with loop, 00613[0.9373] files only has reverse loop, 02960 loop pairs, 02825[0.9544] reverse loop
# Sequence 09 done,01591 files, 00048 files with loop, 00000[0.0000] files only has reverse loop, 00252 loop pairs, 00000[0.0000] reverse loop
# Sequence 50 done,10514 files, 04798 files with loop, 03124[0.6511] files only has reverse loop, 24499 loop pairs, 17097[0.6979] reverse loop
# Sequence 59 done,13247 files, 08489 files with loop, 02972[0.3501] files only has reverse loop, 53858 loop pairs, 19110[0.3548] reverse loop
# Sequence 120205 done,03078 files, 01187 files with loop, 00733[0.6175] files only has reverse loop, 02530 loop pairs, 01761[0.6962] reverse loop
# Sequence 130405 done,02091 files, 00500 files with loop, 00382[0.7640] files only has reverse loop, 00903 loop pairs, 00681[0.7547] reverse loop