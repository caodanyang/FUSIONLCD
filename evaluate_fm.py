import os
import time
import torch
import tools
import numpy as np
import yaml
from dataset import KittiTotalLoader
from tqdm import tqdm

def match_err(fea1,fea2,pose1,pose2,kpt1,kpt2):
    pose_to_frame=torch.matmul(pose2.inverse(),pose1)
    kpt1 = (pose_to_frame @ kpt1.permute(1, 0)).permute(1, 0)
    kpt1[:,2]=0
    kpt2[:,2]=0
    idxp1,idxp2,dis_kpt=tools.nn_match(kpt1,kpt2,'euclidean')
    idxf1,idxf2,dis_fea=tools.nn_match(fea1,fea2,'cosine')
    kpt11=kpt1[idxf1]
    kpt21=kpt2[idxf2]
    dis_kpt1=(kpt11-kpt21).norm(p=2,dim=1)
    mas=[]
    mss=[]
    reps=[]
    for thr in [0.3,0.5,1,2,3]:
        mas.append((torch.sum(dis_kpt1<=thr)/(len(idxf1)+1e-8)).item())
        mss.append((torch.sum(dis_kpt1<=thr)/(torch.sum(dis_kpt<=thr)+1e-8)).item())
        reps.append((torch.sum(dis_kpt<=thr)).item()/(len(fea1)+len(fea2))*2)
    return mas,mss,reps

def feature_match(loader_val=None,data=None):
    if loader_val is None:
        try:
            with open(os.path.join(os.getcwd(), "config.yaml"), "r") as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
            print('Loading config file from %s' % os.path.join(os.getcwd(), "config.yaml"))
        except:
            with open(os.path.join(os.getcwd(), "project/BevNvLcd/config.yaml"), "r") as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
            print('Loading config file from %s' % os.path.join(os.getcwd(), "project/BevNvLcd/config.yaml"))
        cfg = cfg['experiment']
        _, loader_val, _ = KittiTotalLoader(cfg)
    sequences=data['sequences']
    fea_kpt=data['fea_kpt'].cuda()
    poses=data['pose_query'].cuda()
    kpts=data['key_points'].cuda()
    seq=torch.unique(sequences)
    if __name__ == '__main__':
        flag = False
    else:
        flag = True
    for i in range(len(seq)):
        idx=sequences==seq[i]
        fea_kpt1=fea_kpt[idx]
        poses1=poses[idx]
        kpts1=kpts[idx]
        gt=loader_val.dataset.datasets[i].gt
        ms=[]
        for j in tqdm(range(0, len(gt)), disable=flag, ncols=60, desc='feature match'):
            query=gt[j]['idx']
            p_idxs=gt[j]['positive_idxs']
            for p in p_idxs:
                a,s,reps=match_err(fea_kpt1[query],fea_kpt1[p],poses1[query],poses1[p],kpts1[query],kpts1[p])
                ms.append(a+s+reps)
        ms=np.asarray(ms)
        ms1=np.mean(ms,axis=0)
        print('Feature matching, sequence %02d'%(seq[i]))
        print('MA@0.3:%.3f, MA@0.5:%.3f, MA@1:%.3f, MA@2:%.3f, MA@3:%.3f' %(ms1[0], ms1[1], ms1[2], ms1[3], ms1[4]))
        print('MS@0.3:%.3f, MS@0.5:%.3f, MS@1:%.3f, MS@2:%.3f, MS@3:%.3f' %(ms1[5], ms1[6], ms1[7], ms1[8], ms1[9]))
        print('RP@0.3:%.3f, RP@0.5:%.3f, RP@1:%.3f, RP@2:%.3f, RP@3:%.3f' %(ms1[10], ms1[11], ms1[12], ms1[13], ms1[14]))
        time.sleep(1)

        




if __name__ == '__main__':
    # data = torch.load("/mnt/data2/datasets/cdy/results/FUSIONLCD/05230/database/database_149_b0.pth.tar")
    # data = torch.load("/mnt/data2/datasets/cdy/results/BEVLCD/ricnn03202/database/database_xyz_000.pth.tar")
    data= torch.load("/data4/caodanyang/results/FUSIONLCD/bev_07190/database/database_all.pth.tar")
    feature_match(data=data)
   