import os
import time

import torch
import numpy as np
import tqdm


def k3602k(k, k360):
    for src in [0, 3, 4, 5, 6, 7, 9, 10]:
        tgt = src + 50
        path_pose = k360 + '/data_poses/2013_05_28_drive_%04d_sync/cam0_to_world.txt' % src
        path_velo = k360 + '/data_3d_raw/2013_05_28_drive_%04d_sync/velodyne_points/data' % src
        path_calib = k360 + '/calibration/calib_cam_to_velo.txt'
        path_img = k360 + '/data_2d_raw/2013_05_28_drive_%04d_sync/image_00/data_rect' % src

        path_pose1 = k + '/data_odometry_poses/dataset/poses/%02d.npy' % tgt
        path_velo1 = k + '/data_odometry_velodyne/dataset/sequences/%02d/velodyne' % tgt
        path_calib1 = k + '/data_odometry_calib/dataset/sequences/%02d' % tgt
        path_img1 = k + '/data_odometry_color/dataset/sequences/%02d/image_2/' % tgt

        if not os.path.exists(path_velo1):
            os.makedirs(path_velo1)
        if not os.path.exists(path_img1):
            os.makedirs(path_img1)
        if not os.path.exists(path_calib1):
            os.makedirs(path_calib1)

        if not os.path.exists(path_calib1 + '/calib.txt'):
            os.symlink(path_calib, path_calib1 + '/calib.txt')

        with open(path_calib, 'r') as f:
            for line in f.readlines():
                data = np.array([float(x) for x in line.split()])
        cam0_to_velo = np.reshape(data, (3, 4))
        cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
        cam0_to_velo = torch.tensor(cam0_to_velo)
        poses2 = []
        ids = []
        with open(path_pose, 'r') as f:
            for x in f:
                x = x.strip().split()
                x = [float(v) for v in x]
                ids.append(int(x[0]))
                pose = torch.zeros((4, 4), dtype=torch.float64)
                pose[0, 0:4] = torch.tensor(x[1:5])
                pose[1, 0:4] = torch.tensor(x[5:9])
                pose[2, 0:4] = torch.tensor(x[9:13])
                pose[3, 3] = 1.0
                pose = pose @ cam0_to_velo.inverse()
                poses2.append(pose.float().numpy())
        pose = np.stack(poses2)
        np.save(path_pose1, pose)

        cnt = 0
        for i in tqdm.tqdm(ids, desc='%02d:' % src):
            path_velo_now = os.path.join(path_velo, '0000%06d.bin' % i)
            path_img_now = os.path.join(path_img, '0000%06d.png' % i)
            if os.path.exists(path_velo_now) and os.path.exists(path_img_now):
                pass
            else:
                break
            path_velo_now1 = os.path.join(path_velo1, '%06d.bin' % cnt)
            path_img_now1 = os.path.join(path_img1, '%06d.png' % cnt)
            if not os.path.exists(path_velo_now1):
                os.symlink(path_velo_now, path_velo_now1)
            if not os.path.exists(path_img_now1):
                os.symlink(path_img_now, path_img_now1)
            cnt = cnt + 1


def todataset(kitti_root, dataset_root):
    sequences = [0, 5, 6, 7, 8, 9, 50, 54, 55, 56, 59]
    for s in sequences:
        if s >= 50:
            suffix = '.npy'
        else:
            suffix = '.txt'
        kitti_velo_dir = kitti_root + '/data_odometry_velodyne/dataset/sequences/%02d/velodyne' % s
        kitti_img_dir = kitti_root + '/data_odometry_color/dataset/sequences/%02d/image_2' % s
        kitti_pose_dir = kitti_root + '/data_odometry_poses/dataset/poses/%02d' % s + suffix
        kitti_calib_dir = kitti_root + '/data_odometry_calib/dataset/sequences/%02d/calib.txt' % s
        dataset_path = dataset_root + '/%02d' % s
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        dataset_velo_dir = dataset_path + '/velodyne'
        dataset_img_dir = dataset_path + '/image_2'
        dataset_pose_dir = dataset_path + '/poses' + suffix
        dataset_calib_dir = dataset_path + '/calib.txt'
        if not os.path.exists(dataset_velo_dir):
            os.symlink(kitti_velo_dir, dataset_velo_dir)
        if not os.path.exists(dataset_img_dir):
            os.symlink(kitti_img_dir, dataset_img_dir)
        if not os.path.exists(dataset_pose_dir):
            os.symlink(kitti_pose_dir, dataset_pose_dir)
        if not os.path.exists(dataset_calib_dir):
            os.symlink(kitti_calib_dir, dataset_calib_dir)


if __name__ == '__main__':
    k360 = '/data4/caodanyang/dataset/KITTI-360'
    k = '/data4/caodanyang/dataset/kitti/odometry'
    k3602k(k, k360)
    kitti_root = '/data4/caodanyang/dataset/kitti/odometry'
    dataset_root = '/data2/caodanyang/project/dataset/FUSION/sequences'
    todataset(kitti_root, dataset_root)
