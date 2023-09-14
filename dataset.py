import glob
import math
import os
import pickle
from functools import reduce
import matplotlib.pylab as plt
import cv2
import numba
import numpy as np
import torch
import yaml
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence

import tools

IMG_HEIGHT = 384
IMG_WIDTH = 1152
EGEG_PROJ = 10
IMAGE_SCALE = 0.5

def euler2mat(z, y, x):
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
            [[cosz, -sinz, 0],
             [sinz, cosz, 0],
             [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
            [[cosy, 0, siny],
             [0, 1, 0],
             [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


def rt_mat(rx, ry, rz, tx, ty, tz):
    rt = np.eye(4, dtype=np.float32)
    r = euler2mat(rz, ry, rx)
    rt[0:3, 0:3] = r
    rt[0:3, 3] = [tx, ty, tz]
    return rt

@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(points,
                                    voxel_size,
                                    coors_range,
                                    num_points_per_voxel,
                                    coor_to_voxelidx,
                                    voxels,
                                    coors,
                                    max_points,
                                    max_voxels,voxel_idx_empty,voxel_mamiz):
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
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]#0-15000
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxel_idx_empty[voxelidx,num_points_per_voxel[voxelidx]]=1
            if points[i,2]>voxel_mamiz[voxelidx,0]:
                voxel_mamiz[voxelidx,0]=points[i,2]
            if points[i,2]<voxel_mamiz[voxelidx,1]:
                voxel_mamiz[voxelidx,1]=points[i,2]
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
    voxels = np.zeros(shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    voxel_idx_empty = np.zeros((max_voxels,max_points),dtype=np.int32)
    voxel_mamiz=np.zeros((max_voxels,2),dtype=points.dtype)
    voxel_mamiz[:,0]=-99
    voxel_mamiz[:,1]=99
    voxel_num = _points_to_voxel_reverse_kernel(
        points, voxel_size, coors_range, num_points_per_voxel,
        coor_to_voxelidx, voxels, coors, max_points, max_voxels,voxel_idx_empty,voxel_mamiz)

    # coors = coors[:voxel_num]
    # voxels = voxels[:voxel_num]
    # num_points_per_voxel = num_points_per_voxel[:voxel_num]
    # voxels[:, :, -3:] = voxels[:, :, :3] - \
    #     voxels[:, :, :3].sum(axis=1, keepdims=True)/num_points_per_voxel.reshape(-1, 1, 1)
    return voxels, coors, num_points_per_voxel, coor_to_voxelidx,voxel_idx_empty,voxel_mamiz


@numba.jit(nopython=True)
def pixel_choose(pixel1, pixel2):
    n = pixel1.shape[0]
    k = pixel1.shape[1]
    k1 = pixel2.shape[1]
    for i in range(n):
        idx = []
        for j in range(k):
            if pixel1[i, j, 0] > -1:
                idx.append(j)
        k2 = len(idx)
        idx=np.asarray(idx)
        if k2 >= k1:
            choice = np.random.choice(idx, k1, replace=False)
        else:
            choice = np.random.choice(idx, k1, replace=True)
        for j in range(k1):
            pixel2[i,j] = pixel1[i, choice[j]]

def pointcloud_encoder(pointcloud=None, cfg=None):
    try:
        resolution = cfg['bev_resolution']
        pointcloud_range = [float(x) for x in tools.read_cfg(cfg['bev_range'])]
        voxel_max_points = cfg['voxel_max_points']
        voxel_num = cfg['voxel_num']
        voxel_sample = cfg['voxel_sample']
    except:
        resolution = 0.2
        pointcloud_range = [-40, -40, -2.5, 40, 40, 1.5]
        voxel_max_points = 100
        voxel_num = 15000
        voxel_sample = 'top'

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
    resolution_z=pointcloud_range[5]-pointcloud_range[2]
    voxels, coors, num_points_per_voxel, coor_to_voxelidx,voxel_idx_empty,voxel_mamiz = points_to_voxel(pointcloud,
                                                                            voxel_size=[
                                                                                resolution, resolution,resolution_z],
                                                                            coors_range=[
                                                                                *pointcloud_range],
                                                                            max_points=voxel_max_points,
                                                                            max_voxels=voxel_num)

    
    coor_to_voxelidx = np.squeeze(coor_to_voxelidx)
    voxel_idx_empty=voxel_idx_empty.astype(np.bool_)

    voxels_center = np.sum(voxels, axis=1)
    voxels_center[:, 0:4] = voxels_center[:, 0:4] / (num_points_per_voxel.reshape(-1, 1)+1e-8)
    
    max_z = voxel_mamiz[:,0]
    min_z = voxel_mamiz[:,1]
    max_z1 = max_z / resolution_z
    mean_i = np.sum(voxels[:, :, 3], axis=1) / (num_points_per_voxel+1e-8)
    density = np.log(np.clip(num_points_per_voxel, 1, None)) / np.log(cfg['voxel_max_points'])
    

    dz = max_z - min_z
    idx_not_ground = (dz > 0.05) & (num_points_per_voxel > 1)
    coors1 = coors[idx_not_ground, 1:3]
    relation = 0
    if voxels.shape[2] == 6:  # x,y,z,i,pu,pv
        have_pixel = (voxels[:, :, 4] > 0) & (voxels[:, :, 5] > 0)  # each 3d point has a pixel
        have_pixel = np.bitwise_or.reduce(have_pixel, axis=1)  # each cell of bev has pixels,num of pixels may less than num of 3d points

        # num_voxels_pixel = np.sum(voxels_pixel, axis=1)
        # num_voxels_pixel = np.clip(num_voxels_pixel, 1, None)
        # voxels_center[:, 4:6] = np.int_(voxels_center[:, 4:6] / num_voxels_pixel.reshape(-1, 1))
        feature = np.concatenate([max_z1, mean_i, density,
                                  voxels_center[:, 0], voxels_center[:, 1], voxels_center[:, 2], voxels_center[:, 3]],
                                 axis=0).reshape(7, -1).T
        pixels = voxels[idx_not_ground & have_pixel, :, 4:6]
        # pixels2 = np.zeros([pixels.shape[0], 10, 2], dtype=pixels.dtype)
        # pixel_choose(pixels, pixels2)
        # pixels = pixels2

        coors2 = coors[idx_not_ground & have_pixel, 1:3].reshape(-1, 1, 2)
        relation = np.hstack((pixels, coors2))

    elif voxels.shape[2] == 4:  # x,y,z,i
        feature = np.concatenate([max_z1, mean_i, density,
                                  voxels_center[:, 0], voxels_center[:, 1], voxels_center[:, 2], voxels_center[:, 3]],
                                 axis=0).reshape(7, -1).T
    else:
        print('ERROR VOXEL')
        exit()
    # eigvalues = voxel_svd(voxels)
    # feature = np.hstack((feature, eigvalues))
    bev = np.zeros([coor_to_voxelidx.shape[1], coor_to_voxelidx.shape[1], feature.shape[1]], dtype=np.float32)
    bev[coors1[:, 0], coors1[:, 1]] = feature[idx_not_ground]
    # bev[coors1[:, 0], coors1[:, 1]] = feature[idx_not_ground]
    # bev_show=np.uint8(bev[:,:,0:3]*255)
    # cv2.imshow('1',bev_show)
    # cv2.waitKey(0)

    return bev, relation


def crop_image(img, height=IMG_HEIGHT, width=IMG_WIDTH):
    # img: the original image, a numpy array of size H*W*C
    # height: the target height
    # width: the target width

    # get the size of the original image
    h, w, _ = img.shape

    # calculate the padding size if necessary
    if h < height:
        pad_top = (height - h) // 2
        pad_bottom = height - h - pad_top
    else:
        pad_top, pad_bottom = 0, 0
    if w < width:
        pad_left = (width - w) // 2
        pad_right = width - w - pad_left
    else:
        pad_left, pad_right = 0, 0

    # pad the original image with black pixels if necessary
    img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # crop the padded image to the target size
    h1 = (img_padded.shape[0] - height) // 2
    h2 = h1 + height
    w1 = (img_padded.shape[1] - width) // 2
    w2 = w1 + width
    img_cropped = img_padded[h1:h2, w1:w2]
    dh = int((height - h) / 2)
    dw = int((width - w) / 2)
    if IMAGE_SCALE<1:
        img_cropped = cv2.resize(img_cropped, None, None, IMAGE_SCALE, IMAGE_SCALE)
        dh = int(dh * IMAGE_SCALE)
        dw = int(dw * IMAGE_SCALE)

    return img_cropped, dh, dw


class KittiDataset(Dataset):
    def __init__(self, cfg, sequence, argument=True, mode='train', flag_bev=True, flag_img=True, flag_fuse=True):
        root_dataset = cfg['path_dataset']
        if (flag_img == False) & (flag_bev == False) & (flag_fuse == False):
            print('No module will be used!')
            exit()
        if (flag_img == False) or (flag_bev == False):
            flag_fuse = False
        self.flag_img = flag_img
        self.flag_bev = flag_bev
        self.flag_fuse = flag_fuse
        self.cfg = cfg
        self.sequence = sequence
        self.mode = mode
        scans = glob.glob(os.path.join(root_dataset, 'sequences', '%02d' % sequence, 'velodyne', "*.bin"))
        images = glob.glob(os.path.join(root_dataset, 'sequences', '%02d' % sequence, 'image_2', "*.png"))
        if int(sequence) >= 50:
            calib = np.loadtxt(os.path.join(root_dataset, 'sequences', '%02d' % sequence, 'calib.txt'))
            poses = os.path.join(os.path.join(root_dataset, 'sequences', '%02d' % sequence, 'poses.npy'))
        else:
            calib = np.genfromtxt(os.path.join(root_dataset, 'sequences', '%02d' % sequence, 'calib.txt'))[:, 1:]
            poses = os.path.join(os.path.join(root_dataset, 'sequences', '%02d' % sequence, 'poses.txt'))
        f_gt = open(os.path.join(root_dataset, 'sequences', '%02d' % sequence, cfg['loop_file'] + '.pickle'), 'rb')
        scans.sort()
        images.sort()
        self.scans = scans
        self.images = images
        if int(sequence) >= 50:
            cam0_to_velo = np.reshape(calib, (3, 4))
            cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
            cam0_to_velo = np.linalg.inv(cam0_to_velo)
            self.cam0_to_velo = cam0_to_velo.astype(np.float32)
            k = [552.554261, 0.000000, 682.049453, 0.000000, 
                 0.000000, 552.554261, 238.769549, 0.000000, 
                 0.000000, 0.000000, 1.000000, 0.000000]
            k = np.array(k).reshape([3, 4])
            p2 = np.eye(4)
            p2[:3] = k
            self.p2 = p2.astype(np.float32)
            poses2 = np.load(poses)
        else:
            p2 = np.reshape(calib[2], (3, 4))
            p2 = np.vstack([p2, [0, 0, 0, 1]])
            self.p2 = p2.astype(np.float32)
            cam0_to_velo = np.reshape(calib[4], (3, 4))
            cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
            self.cam0_to_velo = cam0_to_velo.astype(np.float32)
            cam0_to_velo = torch.tensor(cam0_to_velo)
            poses2 = []
            with open(poses, 'r') as f:
                for x in f:
                    x = x.strip().split()
                    x = [float(v) for v in x]
                    pose = torch.zeros((4, 4), dtype=torch.float64)
                    pose[0, 0:4] = torch.tensor(x[0:4])
                    pose[1, 0:4] = torch.tensor(x[4:8])
                    pose[2, 0:4] = torch.tensor(x[8:12])
                    pose[3, 3] = 1.0
                    pose = cam0_to_velo.inverse() @ (pose @ cam0_to_velo)  #
                    poses2.append(pose.float().numpy())
            poses2 = np.stack(poses2)
        # for i in range(12):
        #     plt.subplot(3, 4, i + 1), plt.plot(np.arange(len(poses2)), poses2[:, i // 4, i % 4])
        # plt.show()
        self.poses = poses2

        gt = pickle.load(f_gt)
        self.gt=gt
        # gt_new=[]
        # for i in range(len(gt)):
        #     idx=gt[i]['idx']
        #     positive_idxs=gt[i]['positive_idxs']
        #     for j in positive_idxs:
        #         sample={'idx':idx,'positive_idxs':[j]}
        #         gt_new.append(sample)
        # self.gt=gt_new
        self.argument = argument

    def __len__(self):
        if self.mode == 'test':
            return len(self.poses)
        else:
            return int(len(self.gt))

    def __getitem__(self, idx):
        if self.mode == 'test':
            idx_query = idx
            pose_query = self.poses[idx_query]
            image_query, scan_query, bev_query, dw, dh, W, H, relation_query = 0, 0, 0, 0, 0, 0, 0, 0
            if self.flag_img:
                image_query = cv2.imread(self.images[idx_query])
                # image_query = cv2.GaussianBlur(image_query, (15,15),0)
                image_query, dh, dw = crop_image(image_query, IMG_HEIGHT, IMG_WIDTH)
            if self.flag_bev:
                scan_query = np.fromfile(self.scans[idx_query], dtype=np.float32).reshape((-1, 4))
                # idx = np.random.choice(len(scan_query), int(len(scan_query) /4), replace=False)
                # scan_query = scan_query[idx]
                if self.flag_bev & self.flag_img & self.flag_fuse:
                    # mat_proj = np.matmul(self.p2, self.cam0_to_velo)
                    mat_proj = torch.matmul(torch.from_numpy(self.p2), torch.from_numpy(self.cam0_to_velo)).numpy()
                    pts_query = scan_query.copy()
                    pts_query[:, 3] = 1
                    # pts_proj_query = np.matmul(mat_proj, pts_query.T).T
                    pts_proj_query = torch.matmul(torch.from_numpy(mat_proj), torch.from_numpy(pts_query.T)).numpy().T
                    z = pts_proj_query[:, 2:3]
                    pts_proj_query = pts_proj_query / z * IMAGE_SCALE 
                    pts_proj_query[:, 0:2] = pts_proj_query[:, 0:2] + [dw, dh]
                    H, W, _ = image_query.shape
                    mask_query = (pts_proj_query[:, 0] >= EGEG_PROJ) & (pts_proj_query[:, 0] < W - EGEG_PROJ) & (
                            pts_proj_query[:, 1] >= EGEG_PROJ) & (pts_proj_query[:, 1] < H - EGEG_PROJ) & (z[:, 0] >= 0)
                    pts_proj_query[~mask_query] = -1
                    pixel_query = pts_proj_query[:, 0:2]
                    pixel_query = pixel_query[:, [1, 0]]
                    scan_query = np.hstack((scan_query, pixel_query))
                bev_query, relation_query = pointcloud_encoder(scan_query, self.cfg)
            sample = {
                'sequence': self.sequence,
                'id_query': idx_query,
                'bev_query': bev_query,
                'img_query': image_query,
                'pose_query': pose_query,
                'relation_query': relation_query
            }
        else:
            gt = self.gt[idx]
            idx_query = gt['idx']
            idx_ps = gt['positive_idxs']
            idx_positive = np.random.choice(idx_ps)
            pose_query = self.poses[idx_query]
            pose_positive = self.poses[idx_positive]
            image_query, scan_query, bev_query, image_positive, scan_query, bev_positive, dw, dh, W, H, pose_to_frame, \
                label_score, relation_query, relation_positive, = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            if self.flag_img:
                image_query = cv2.imread(self.images[idx_query])
                image_query, dh, dw = crop_image(image_query)
                image_positive = cv2.imread(self.images[idx_positive])
                image_positive, _, _ = crop_image(image_positive)
            if self.flag_bev:
                scan_query = np.fromfile(self.scans[idx_query], dtype=np.float32).reshape((-1, 4))

                scan_positive = np.fromfile(self.scans[idx_positive], dtype=np.float32).reshape((-1, 4))

                # return {'1':np.zeros(5)}
                # import open3d as o3d
                # pcd1 = o3d.geometry.PointCloud()
                # sc1 = scan_query.copy()
                # sc1[:, 3] = 1
                # sc1 = np.matmul(pose_query, sc1.T).T
                # pcd1.points = o3d.utility.Vector3dVector(sc1[:, :3])
                # pcd1.colors = o3d.utility.Vector3dVector([[0, 0, 1] for i in range(len(pcd1.points))])
                # pcd2 = o3d.geometry.PointCloud()
                # sc2 = scan_positive.copy()
                # sc2[:, 3] = 1
                # sc2 = np.matmul(pose_positive, sc2.T).T
                # pcd2.points = o3d.utility.Vector3dVector(sc2[:, :3])
                # pcd2.colors = o3d.utility.Vector3dVector([[0, 1, 0] for i in range(len(pcd2.points))])
                # vis1 = o3d.visualization.Visualizer()
                # vis1.create_window(window_name='registration', width=600, height=600)  # 创建窗口
                # render_option: o3d.visualization.RenderOption = vis1.get_render_option()  # 设置点云渲染参数
                # render_option.background_color = np.array([1, 1, 1])  # 设置背景色（这里为黑色）
                # render_option.point_size = 2  # 设置渲染点的大小
                # vis1.add_geometry(pcd1)
                # vis1.add_geometry(pcd2)
                # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=pose_query[0:3,3])
                # vis1.add_geometry(coord_frame)
                # vis1.run()

                if self.argument:
                    # Rt1 = np.eye(4)
                    # idx1=np.random.random((len(scan_query),))
                    # idx1=idx1>0.1
                    # scan_query=scan_query[idx1]
                    # idx2=np.random.random((len(scan_positive),))
                    # idx2=idx2>0.1
                    # scan_positive=scan_positive[idx2]
                    rand = np.random.random(6) * 2 - 1
                    Rt = rt_mat(rand[0] * 3 / 180 * np.pi,
                                rand[1] * 3 / 180 * np.pi,
                                rand[2] * 180 / 180 * np.pi,
                                rand[3] * 3, rand[4] * 3, rand[5] * 0.3)
                    Rt1 = torch.from_numpy(Rt).inverse().numpy()
                    ints = scan_positive[:, 3].copy()
                    scan_positive[:, 3] = 1
                    # scan_positive = np.matmul(Rt, scan_positive.T).T
                    scan_positive = torch.matmul(torch.from_numpy(Rt),torch.from_numpy(scan_positive.T)).numpy().T
                    scan_positive[:, 3] = ints
                    # pose_positive = np.matmul(pose_positive, np.linalg.inv(Rt))
                    pose_positive=torch.matmul(torch.from_numpy(pose_positive),torch.from_numpy(Rt).inverse()).numpy()
                else:
                    Rt1 = np.eye(4).astype(np.float32)
                if self.flag_fuse:
                    # mat_proj = np.matmul(self.p2, self.cam0_to_velo)
                    mat_proj = torch.matmul(torch.from_numpy(self.p2), torch.from_numpy(self.cam0_to_velo)).numpy()
                    pts_query = scan_query.copy()
                    pts_query[:, 3] = 1
                    # pts_proj_query = np.matmul(mat_proj, pts_query.T).T
                    pts_proj_query = torch.matmul(torch.from_numpy(mat_proj), torch.from_numpy(pts_query.T)).numpy().T
                    z = pts_proj_query[:, 2:3]
                    pts_proj_query = pts_proj_query / z * IMAGE_SCALE 
                    pts_proj_query[:, 0:2] = pts_proj_query[:, 0:2] + [dw, dh]
                    H, W, _ = image_query.shape
                    mask_query = (pts_proj_query[:, 0] >= EGEG_PROJ) & (pts_proj_query[:, 0] < W - EGEG_PROJ) & (
                            pts_proj_query[:, 1] >= EGEG_PROJ) & (pts_proj_query[:, 1] < H - EGEG_PROJ) & (z[:, 0] >= 0)
                    pts_proj_query[~mask_query] = -1
                    pixel_query = pts_proj_query[:, 0:2]
                    pixel_query = pixel_query[:, [1, 0]]  # h,w
                    scan_query = np.hstack((scan_query, pixel_query))

                    # fig = plt.figure()
                    # plt.subplot(2, 1, 1), plt.imshow(image_query[:, :, [2, 1, 0]])
                    # plt.subplot(2, 1, 2), plt.imshow(image_query[:, :, [2, 1, 0]])
                    # plt.scatter(pixel_query[mask_query, 1], pixel_query[mask_query, 0], c=z[mask_query], cmap='jet', alpha=0.5, s=1)
                    # plt.show()

                    # mat_proj1 = self.p2.dot(self.cam0_to_velo).dot(Rt1)
                    mat_proj1 = torch.matmul(torch.matmul(torch.from_numpy(self.p2), torch.from_numpy(self.cam0_to_velo)), torch.from_numpy(Rt1)).numpy()
                    pts_positive = scan_positive.copy()
                    pts_positive[:, 3] = 1
                    # pts_proj_positive = np.matmul(mat_proj1, pts_positive.T).T
                    pts_proj_positive = torch.matmul(torch.from_numpy(mat_proj1), torch.from_numpy(pts_positive.T)).numpy().T
                    z = pts_proj_positive[:, 2:3]
                    pts_proj_positive = pts_proj_positive / z * IMAGE_SCALE 
                    pts_proj_positive[:, 0:2] = pts_proj_positive[:, 0:2] + [dw, dh]
                    mask_positive = (pts_proj_positive[:, 0] >= EGEG_PROJ) & (pts_proj_positive[:, 0] < W - EGEG_PROJ) & (
                            pts_proj_positive[:, 1] >= 0 - EGEG_PROJ) & (pts_proj_positive[:, 1] < H - EGEG_PROJ) & (z[:, 0] >= 0)
                    pts_proj_positive[~mask_positive] = -1
                    pixel_positive = pts_proj_positive[:, 0:2]
                    pixel_positive = pixel_positive[:, [1, 0]]
                    scan_positive = np.hstack((scan_positive, pixel_positive))

                # pose_to_frame = np.matmul(np.linalg.inv(pose_positive), pose_query)
                pose_to_frame=torch.matmul(torch.from_numpy(pose_positive).inverse(),torch.from_numpy(pose_query)).numpy()
                # scan_query1=scan_query.copy()
                # scan_query1[:,3]=1
                # scan_query1=np.matmul(pose_to_frame,scan_query1.T).T
                # scan_query[:,0:3]=scan_query1[:,0:3]
                # scan_query[:, 3] = 1
                # scan_query = np.matmul(pose_query, scan_query.T).T
                # scan_positive[:, 3] = 1
                # scan_positive = np.matmul(pose_positive, scan_positive.T).T
                # plt.subplot(1, 2, 1), plt.plot(scan_query[:, 0], scan_query[:, 1], 'b.', markersize=1),plt.axis([-60,60,-60,60])
                # plt.subplot(1, 2, 2), plt.plot(scan_positive[:, 0], scan_positive[:, 1], 'b.', markersize=1),plt.axis([-60,60,-60,60])
                # plt.show()
                
                bev_query, relation_query = pointcloud_encoder(scan_query, self.cfg)
                bev_positive, relation_positive = pointcloud_encoder(scan_positive, self.cfg)
                # if self.argument:
                #     rand = np.random.randint(0, 9, [2, ])
                #     if rand[0] > 4:
                #         bev_query1 = np.rot90(bev_query, 2, axes=(0, 1))
                #         bev_query = bev_query1.copy()
                #     if rand[1] > 4:
                #         bev_positive1 = np.rot90(bev_positive, 2, axes=(0, 1))
                #         bev_positive = bev_positive1.copy()
                h_bev, w_bev, _ = bev_query.shape
                label_score = np.zeros_like(bev_positive[:, :, :2])
                grid = np.array(np.meshgrid(np.arange(h_bev), np.arange(w_bev))).swapaxes(0, 2)
                scan_query_sample = bev_query[:, :, 3:5]
                mask_query = scan_query_sample != 0
                mask_query = mask_query[:, :, 0] | mask_query[:, :, 1]
                grid_query = grid[mask_query]
                scan_query_sample = scan_query_sample[mask_query]
                scan_positive_sample = bev_positive[:, :, 3:5]
                mask_positive = scan_positive_sample != 0
                mask_positive = mask_positive[:, :, 0] | mask_positive[:, :, 1]
                grid_positive = grid[mask_positive]
                scan_positive_sample = scan_positive_sample[mask_positive]
                scan_query_sample1 = np.hstack((scan_query_sample, scan_query_sample * 0))
                scan_query_sample1[:, 3] = 1
                # scan_query_sample1 = np.matmul(pose_to_frame, scan_query_sample1.T).T
                scan_query_sample1 = torch.matmul(torch.from_numpy(pose_to_frame), torch.from_numpy(scan_query_sample1.T)).numpy().T

                idx1, idx2, dis = tools.nn_match(scan_query_sample1[:, 0:2], scan_positive_sample[:, 0:2], 'euclidean')
                if len(dis) > 50:
                    th1 = max([2, dis[min([256, int(len(dis) * 0.3)])]])
                    idx1 = idx1[dis < th1]
                    idx2 = idx2[dis < th1]
                else:
                    dis = cdist(scan_query_sample1[:, 0:2], scan_positive_sample)
                    min1 = np.min(dis, axis=1)
                    min2 = np.min(dis, axis=0)
                    min11 = np.sort(min1)
                    th1 = max([0.2, min11[min([256, int(len(min11) * 0.2)])]])
                    min21 = np.sort(min2)
                    th2 = max([0.2, min21[min([256, int(len(min21) * 0.2)])]])
                    idx1 = np.arange(len(scan_query_sample))[min1 < th1]
                    idx2 = np.arange(len(scan_positive_sample))[min2 < th2]

                # points1=scan_query_sample1[:, 0:2]
                # points2=scan_positive_sample[:, 0:2]
                # points = np.mean(np.vstack((points1, points2)), axis=0, keepdims=True)
                # points1 = points1 - points
                # points2 = points2 - points
                # af = torch.sum(torch.from_numpy(points1) ** 2, -1, keepdim=True)
                # bf = torch.sum(torch.from_numpy(points2) ** 2, -1, keepdim=True).transpose(0, 1)
                # cf = af + bf - 2 * torch.mm(torch.from_numpy(points1), torch.from_numpy(points2).transpose(0, 1))  # c^2=a^2+b^2-2abcos
                # c = torch.sqrt(cf)
                # dis = c
                # dis1 = torch.min(dis, dim=1)[0]
                # dis2 = torch.min(dis, dim=0)[0]
                # idx1 = torch.where(dis1 < 0.5)[0].numpy()
                # idx2 = torch.where(dis2 < 0.5)[0].numpy()
                # dis1=dis1.numpy()
                # dis2= dis2.numpy()

                grid_query = grid_query[idx1]
                grid_positive = grid_positive[idx2]
                label_score[grid_query[:, 0], grid_query[:, 1], 0] = 1
                label_score[grid_positive[:, 0], grid_positive[:, 1], 1] = 1
                # fig, ax = plt.subplots(2, 2)
                # ax[0,0].imshow(bev_query[:, :, 0:3])
                # ax[0,1].imshow(label_score[:, :, 0])
                # ax[1,0].imshow(bev_positive[:, :, 0:3])
                # ax[1,1].imshow(label_score[:, :, 1])
                # plt.savefig('1.png')
                # plt.show()

            sample = {
                'sequence': self.sequence,
                'id_query': idx_query,
                'bev_query': bev_query,
                'img_query': image_query,
                'pose_query': pose_query,
                'id_positive': idx_positive,
                'bev_positive': bev_positive,
                'img_positive': image_positive,
                'pose_positive': pose_positive,
                'pose_to_frame': pose_to_frame,
                'label_score': label_score,
                'relation_query': relation_query,
                'relation_positive': relation_positive
            }
        return sample


def collate(samples):
    relation_query = []
    relation_positive = []
    samples2 = {key: default_collate([d[key] for d in samples]) for key in samples[0]
                if key != 'relation_query' and key != 'relation_positive'}

    for single_sample in samples:
        try:
            relation_query.append(torch.from_numpy(single_sample['relation_query']))
        except:
            pass
        try:
            relation_positive.append(torch.from_numpy(single_sample['relation_positive']))
        except:
            pass

    relation = relation_query + relation_positive
    if len(relation) > 0:
        relation1 = pad_sequence(relation, batch_first=True, padding_value=-1)
        relation1 = relation1.float()
        samples2['relation'] = relation1

    return samples2


def KittiTotalLoader(cfg):
    flag = cfg['flag']
    bev = False
    img = False
    fuse = False
    if flag == 'fusion':
        bev = True
        img = True
        fuse = True
    elif flag == 'img':
        img = True
    else:
        bev = True
    sequence_train = [int(x) for x in tools.read_cfg(cfg['train'])]
    sequence_val = [int(x) for x in tools.read_cfg(cfg['validate'])]
    sequence_test = [int(x) for x in tools.read_cfg(cfg['test'])]
    dataset_list = []
    for sequence in sequence_train:
        single_dataset = KittiDataset(cfg, sequence, flag_bev=bev, flag_img=img, flag_fuse=fuse, argument=True, mode='train')
        print('===Trainloader add: sequence %02d, %04d files, %04d frames with loop' % (sequence, len(single_dataset.poses), len(single_dataset.gt)))
        dataset_list.append(single_dataset)
    dataset_train = ConcatDataset(dataset_list)
    dataset_list = []
    for sequence in sequence_val:
        single_dataset = KittiDataset(cfg, sequence, flag_bev=bev, flag_img=img, flag_fuse=fuse, argument=False, mode='train')
        print('===Validationloader add: sequence %02d, %04d files, %04d frames with loop' % (sequence, len(single_dataset.poses), len(single_dataset.gt)))
        dataset_list.append(single_dataset)
    dataset_val = ConcatDataset(dataset_list)
    dataset_list = []
    for sequence in sequence_test:
        single_dataset = KittiDataset(cfg, sequence, flag_bev=bev, flag_img=img, flag_fuse=fuse, argument=False, mode='test')
        print('===Testloader add: sequence %02d, %04d files' % (sequence, len(single_dataset.poses)))
        dataset_list.append(single_dataset)
    dataset_test = ConcatDataset(dataset_list)
    loader_train = DataLoader(dataset_train, batch_size=cfg['batchsize'], shuffle=True, num_workers=6, collate_fn=collate)
    loader_val = DataLoader(dataset_val, batch_size=cfg['batchsize'], shuffle=True, num_workers=6, collate_fn=collate)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=6, collate_fn=collate)
    return loader_train, loader_val, loader_test


if __name__ == '__main__':
    root_dir = '/media/ubuntu/Workshop/caodanyang/Project_CDY/results/mylcd'
    # sequence = 0
    # dataset = KittiDataset(root_dir, sequence)
    # dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)
    # t = tools.Timer(name='Loading', number=len(dataloader))
    # for data in dataloader:
    #     t.update()
    try:
        with open(os.path.join(os.getcwd(), "config.yaml"), "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        print('Loading config file from %s' % os.path.join(os.getcwd(), "config.yaml"))
    except:
        with open(os.path.join(os.getcwd(), "project/BevNvLcd/config.yaml"), "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        print('Loading config file from %s' % os.path.join(os.getcwd(), "project/BevNvLcd/config.yaml"))
    cfg = cfg['experiment']
    path_dataset = cfg['path_dataset']
    path_result = cfg['path_result']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = KittiTotalLoader(cfg)
    t = tools.Timer(name='Loading')
    ds = []
    for i, data in enumerate(train):
        # ds.append(d)
        t.update(i)

    # t = tools.Timer(name='Loading', number=len(val))
    # for i, data in enumerate(val):
    #     t.update()
