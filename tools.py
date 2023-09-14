import threading
import torch
import os
import time


def farthest_point_sample(xyz, npoint):
    """Iterative farthest point sampling

    Args:
        xyz: pointcloud data_loader, [B, N, C]
        npoint: number of samples
    Returns:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def batch_distance(feature1,feature2,mode='cosine'):
    if mode == 'cosine':
        # Transport cost matrix
        feature1 = feature1 / torch.sqrt(torch.sum(feature1 ** 2, -1, keepdim=True) + 1e-8)
        feature2 = feature2 / torch.sqrt(torch.sum(feature2 ** 2, -1, keepdim=True) + 1e-8)
        dis = 1.0 - torch.bmm(feature1, feature2.transpose(1, 2))
    elif mode == 'euclidean':
        feature=torch.cat([feature1,feature2],dim=1)
        feature_mean=torch.mean(feature,dim=1,keepdim=True)
        feature1=feature1-feature_mean
        feature2=feature2-feature_mean
        distance_matrix = torch.sum(feature1 ** 2, -1, keepdim=True)
        distance_matrix = distance_matrix + torch.sum(feature2 ** 2, -1, keepdim=True).transpose(1, 2)
        distance_matrix = distance_matrix - 2 * torch.bmm(feature1, feature2.transpose(1, 2))  # c^2=a^2+b^2-2abcos
        distance_matrix = distance_matrix ** 0.5
        dis = distance_matrix
    return dis

def nn_match(fea1, fea2, matrix='cosine'):
    assert len(fea1.shape) == 2 and len(fea2.shape) == 2, 'nnmatch error'
    if not isinstance(fea1, torch.Tensor):
        fea1 = torch.tensor(fea1)
    if not isinstance(fea2, torch.Tensor):
        fea2 = torch.tensor(fea2)
    if matrix == 'cosine':
        # Transport cost matrix
        fea1 = fea1 / torch.sqrt(torch.sum(fea1 ** 2, -1, keepdim=True) + 1e-8)
        fea2 = fea2 / torch.sqrt(torch.sum(fea2 ** 2, -1, keepdim=True) + 1e-8)
        dis = 1.0 - torch.mm(fea1, fea2.transpose(0, 1))
    elif matrix == 'euclidean':
        distance_matrix = torch.sum(fea1 ** 2, -1, keepdim=True)
        distance_matrix = distance_matrix + torch.sum(fea2 ** 2, -1, keepdim=True).transpose(0, 1)
        distance_matrix = distance_matrix - 2 * torch.mm(fea1, fea2.transpose(0, 1))  # c^2=a^2+b^2-2abcos
        dis = distance_matrix ** 0.5
    else:
        dis = 0
        print('Invalid matrix')
    idx0_min = torch.argmin(dis, dim=0)
    idx1_min = torch.argmin(dis, dim=1)
    ids1 = torch.arange(0, dis.shape[1]).to(fea1.device)
    idx = idx1_min[idx0_min]
    idx_match = ids1 == idx
    idx1 = ids1[idx_match]
    idx2 = idx0_min[idx_match]
    dis_min = dis[idx2, idx1]

    return idx2, idx1, dis_min


def path_join(*args):
    names = list(args)
    path = names[0]
    for i in range(len(names) - 1):
        path = os.path.join(path, names[i + 1])
    path = list(path)
    while "\\" in path:
        idx = path.index("\\")
        path[idx] = "/"
    path = ''.join(path)
    return path


def make_save_path(*args):
    path = path_join(*args)
    if not os.path.exists(path):
        os.makedirs(path)
    return path



def read_cfg(data):
    if type(data) is int:
        result = [data]
    else:
        result = data.split(',')
    return result


class Timer:
    """A module to record the program running time"""

    def __init__(self, name="Now"):
        self.strat = time.time()
        self.cnt = 0
        self.end = time.time()
        self.avg = 0
        self.all = 0
        self.now = 0
        self.name = name
        time_now = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        print('Init timer: ',time_now)

    def update(self, name=None):
        if name is not None:
            self.name = name
        self.cnt = self.cnt + 1
        self.end = time.time()
        self.avg = (self.end - self.strat) / self.cnt
        self.now = self.end - self.all - self.strat
        self.all = self.end - self.strat
        time_now = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        if self.avg<1:
            print("%s | %s | using %d | each %.3f" %
                (time_now, self.name, self.all, self.now))
        elif self.avg<10:
            print("%s | %s | using %d | each %.2f" %
                (time_now, self.name, self.all, self.now))
        elif self.avg<100:
            print("%s | %s | using %d | each %.1f" %
                (time_now, self.name, self.all, self.now))
        else:
            print("%s | %s | using %d | each %d" %
                (time_now, self.name, self.all, self.now))


if __name__ == '__main__':
    # draw_trace()
    pass
