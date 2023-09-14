import argparse
import torch
from torch.utils.data import Dataset
import pykitti
import os
from sklearn.neighbors import KDTree
import pickle
import numpy as np
skip_frame=50

class KITTILoader3DPosesOnlyLoopPositives(Dataset):

    def __init__(self, dir, sequence, poses, positive_range=5., negative_range=25., hard_range=None):
        super(KITTILoader3DPosesOnlyLoopPositives, self).__init__()

        self.positive_range = positive_range
        self.negative_range = negative_range
        self.hard_range = hard_range
        self.dir = dir
        self.sequence = sequence

        if int(sequence) > 21:
            self.poses = np.load(poses)
        else:
            calib = np.genfromtxt(os.path.join(dir, 'sequences', sequence, 'calib.txt'))[:, 1:]
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
            self.poses = np.stack(poses2)
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

        indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.positive_range, sort_results=True, return_distance=True)
        indices = [indices[0][0], indices[1][0]]
        min_range = max(0, idx - skip_frame)
        max_range = min(idx + skip_frame, len(self.poses))
        positive_idxs = list(set(indices[0]) & set(idx_angle) - set(range(min_range, max_range)))
        loop_angle = angle[positive_idxs]
        reverse = 0
        if len(loop_angle) > 0:
            reverse=np.sum(loop_angle>90)
            if min(loop_angle) > 90:
                reverse = -1*reverse
        positive_idxs.sort()
        num_loop = len(positive_idxs)

        indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.negative_range)
        indices = set(indices[0])
        negative_idxs = set(range(len(self.poses))) - indices
        negative_idxs = list(negative_idxs)
        negative_idxs.sort()

        hard_idxs = None
        if self.hard_range is not None:
            inner_indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.hard_range[0])
            outer_indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.hard_range[1])
            hard_idxs = set(outer_indices[0]) - set(inner_indices[0])
            pass

        return num_loop, positive_idxs, negative_idxs, hard_idxs, reverse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', default='/data2/caodanyang/project/dataset/FUSION',
                        help='dataset directory')
    args = parser.parse_args()
    base_dir = args.root_folder
    for sequence in ['54']:#'00', '05', '06', '07', '08', '09', '50', '54', '55', '56', '59', '120205','130405'
        if int(sequence) < 50:
            poses_file = base_dir + "/sequences/" + sequence + "/poses.txt"
        elif int(sequence)<100:
            poses_file = base_dir + "/sequences/" + sequence + "/poses.npy"
        else:
            pass
            
        
        dataset = KITTILoader3DPosesOnlyLoopPositives(base_dir, sequence, poses_file, 4, 15, [8, 15])
        lc_gt = []
        lc_gt_file = os.path.join(base_dir, 'sequences', sequence, 'loop_GT_4m.pickle')
        loop_pairs = []
        loop_files = []
        for i in range(len(dataset)):
            sample, pos, neg, hard, reverse = dataset[i]
            if sample > 0.:
                loop_files.append([i, reverse])
                sample_dict = {}
                sample_dict['idx'] = i
                sample_dict['positive_idxs'] = pos
                for p in pos:
                    if i < p:
                        loop_pairs.append([i, p])
                # sample_dict['negative_idxs'] = neg
                # sample_dict['hard_idxs'] = hard
                lc_gt.append(sample_dict)
        loop_files = np.array(loop_files)
        num_reverse_file = int(np.sum(loop_files[:, 1]<0))
        num_reverse_pairs = int(np.sum(np.abs(loop_files[:, 1])))/2
        # with open(lc_gt_file, 'wb') as f:
        #     pickle.dump(lc_gt, f)
        print('Sequence %02d done,%05d files, %05d files with loop, %05d[%.4f] files only has reverse loop, %05d loop pairs, %05d[%.4f] reverse loop' %
              (int(sequence), len(dataset), len(loop_files), num_reverse_file, num_reverse_file / len(loop_files), len(loop_pairs),num_reverse_pairs,num_reverse_pairs/len(loop_pairs)))
