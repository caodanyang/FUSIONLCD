import argparse
import os
import time

import numpy as np
import torch
import torch.optim as optim
import yaml

import net
import tools
from dataset import KittiTotalLoader
from evaluate_lcd import lcd
from loss import TotalLoss

test_step = 10

def save_checkpoint(model, optimizer, loss_total_fun, epoch, iter_train, path_result):
    if (epoch + 1) % test_step == 0 and epoch+1>=test_step:

        time_now = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        checkpoint = {'time': time_now,
                      'epoch': epoch,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        savepath = tools.make_save_path(path_result, 'models')
        torch.save(checkpoint, savepath + '/checkpoint_%03d.pth.tar' % epoch)
        print(savepath + '/checkpoint_%03d.pth.tar is saved' % epoch)

class log_result():
    def __init__(self,path_result):
        self.path=path_result
        if not os.path.exists(path_result):
            with open(path_result, 'w') as file:
                file.write('Time           Sequence Epoch  AP    R100  F1    R@1   R@2   R@3   R@4   R@5')
                file.write('   R@6   R@7   R@8   R@9   R@10  R@15  R@20  R@25\n')
                for i in range(300):
                    file.write('\n')
    def write(self,seq,epoch,row,x):
        with open(self.path, 'r') as file:
            lines = file.readlines()
        time_now = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        new_content='%s %08d %06d'%(time_now,seq,epoch)
        for x1 in x:
            new_content=new_content + ' %.3f'%x1
        
        lines[row] = new_content+'\n'
        with open(self.path, 'w') as file:
            file.writelines(lines)

def train(model, optimizer, loss_total_fun, data, device):
    model.train()
    sequences = data['sequence']
    id_query = data['id_query']
    id_positive = data['id_positive']
    batchsize = len(id_query)

    bev_query = data['bev_query'].to(device)
    bev_positive = data['bev_positive'].to(device)

    pose_query = data['pose_query'].to(device)
    pose_positive = data['pose_positive'].to(device)

    pose_to_frame = data['pose_to_frame'].to(device)
    label_score = data['label_score'].to(device)

    img_query = data['img_query'].to(device)
    img_positive = data['img_positive'].to(device)
    try:
        bev = torch.cat([bev_query, bev_positive], dim=0)
        bev = bev.permute(0, 3, 1, 2)
    except:
        bev = 0
    try:
        img = torch.cat([img_query, img_positive], dim=0)
        img = img.permute(0, 3, 1, 2)
    except:
        img = 0
    try:
        relation = data['relation'].to(device)
    except:
        relation = 0

    batch_dict = {'bev': bev,
                  'label_score': label_score,
                  'img': img,
                  'relation': relation,
                  'id_query': id_query,
                  'sequence': sequences,
                  'id_positive': id_positive,
                  'pose_to_frame': pose_to_frame,
                  'pose_query': pose_query,
                  'pose_positive': pose_positive,
                  'batch_size': int(batchsize * 2)}

    model(batch_dict)

    loss_total_fun(batch_dict)
    l_total = batch_dict['loss'][0]

    optimizer.zero_grad()
    l_total.backward()
    optimizer.step()
    for p in model.parameters():
        if torch.isnan(p).any():
            print('Model NAN, ', p.shape)
            exit()
    return batch_dict


def validate(model, loss_total_fun, data, device):
    model.eval()
    with torch.no_grad():
        sequences = data['sequence']
        id_query = data['id_query']
        id_positive = data['id_positive']
        batchsize = len(id_query)

        bev_query = data['bev_query'].to(device)
        bev_positive = data['bev_positive'].to(device)

        pose_query = data['pose_query'].to(device)
        pose_positive = data['pose_positive'].to(device)

        pose_to_frame = data['pose_to_frame'].to(device)
        label_score = data['label_score'].to(device)

        img_query = data['img_query'].to(device)
        img_positive = data['img_positive'].to(device)
        try:
            bev = torch.cat([bev_query, bev_positive], dim=0)
            bev = bev.permute(0, 3, 1, 2)
        except:
            bev = 0
        try:
            img = torch.cat([img_query, img_positive], dim=0)
            img = img.permute(0, 3, 1, 2)
        except:
            img = 0
        try:
            relation = data['relation'].to(device)
        except:
            relation = 0

        batch_dict = {'bev': bev,
                      'label_score': label_score,
                      'img': img,
                      'relation': relation,
                      'id_query': id_query,
                      'sequence': sequences,
                      'id_positive': id_positive,
                      'pose_to_frame': pose_to_frame,
                      'pose_query': pose_query,
                      'pose_positive': pose_positive,
                      'batch_size': int(batchsize * 2)}
        model(batch_dict)
        loss_total_fun(batch_dict)

    return batch_dict


def test(model, data, device):
    model.eval()
    with torch.no_grad():
        sequences = data['sequence']
        id_query = data['id_query']
        batchsize = len(id_query)
        bev_query = data['bev_query'].to(device)
        pose_query = data['pose_query'].to(device)

        img_query = data['img_query'].to(device)
        try:
            bev = bev_query
            bev = bev.permute(0, 3, 1, 2)
        except:
            bev = 0
        try:
            img = img_query
            img = img.permute(0, 3, 1, 2)
        except:
            img = 0
        try:
            relation = data['relation'].to(device)
        except:
            relation = 0

        batch_dict = {'bev': bev,
                      'img': img,
                      'relation': relation,
                      'id_query': id_query,
                      'sequence': sequences,
                      'pose_query': pose_query,
                      'batch_size': int(batchsize * 2)}

        model(batch_dict)

    return batch_dict


def main(args):
    try:
        with open(os.path.join(os.getcwd(), "config.yaml"), "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        print('Loading config file from %s' % os.path.join(os.getcwd(), "config.yaml"))
    except:
        with open(os.path.join(os.getcwd(), "project/FUSIONLCD/config.yaml"), "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        print('Loading config file from %s' % os.path.join(os.getcwd(), "project/BevNvLcd/config.yaml"))
    cfg = cfg['experiment']
    for k, v in cfg.items():
        print(k, ':', v)
    path_result = os.path.join(cfg['path_result'],args.result_name)
    lres=log_result(os.path.join(os.getcwd(),'result',args.result_name+'.txt'))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg['cuda'] else "cpu")
    start_epoch = 0
    iter_train = 0
    epochs = cfg['epochs']
    model = net.Fusion(cfg)
    print(model)
    model = model.to(device)
    loss_total_fun = TotalLoss(cfg).to(device)
    print("Model params: %.6fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'], betas=(cfg['beta1'], cfg['beta2']), eps=cfg['eps'], weight_decay=cfg['weight_decay'])
    # optimizer = optim.Adam([{'params': model.bev.parameters(), 'lr': 0.0002},
    #                         {'params': model.img.parameters(), 'lr': 0.0001},
    #                         {'params': model.vlad_fusion_layer.parameters(), 'lr': 0.0001}],
    #                        betas=(cfg['beta1'], cfg['beta2']), eps=cfg['eps'], weight_decay=cfg['weight_decay'])
    # print(optimizer)
    loader_train, loader_val, loader_test = KittiTotalLoader(cfg)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 3, 5, 10, 50, 100], gamma=0.5, last_epoch=start_epoch - 1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.99)
    # scheduler = warmup(optimizer, 5, 1e-6, cfg['learning_rate'])
    # writer = SummaryWriter(tools.make_save_path(path_result, 'tensorboard_log'))
    t = tools.Timer()
    test_best = np.zeros([len(loader_test.dataset.datasets), 3])
    if cfg['load_model']:
        checkpoint = torch.load((cfg['last_model']))
        start_epoch = checkpoint['epoch'] + 1 * cfg['train_flag']
        state_dict_saved = checkpoint['model']
        model.load_state_dict(state_dict_saved)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('loaded %s' % cfg['last_model'])
    if not cfg['train_flag']:
        print_frequency = 1e9
    else:
        print_frequency = 1
    for epoch in range(start_epoch, epochs):
        torch.cuda.empty_cache()
        '''
        ============================== train ===============================
        '''
        if cfg['train_flag']:
            if epoch - start_epoch == 0:
                pf = print_frequency
                print_frequency = min(len(loader_train), print_frequency * 10)
            else:
                print_frequency = pf
            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10,l11 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            step_print = max(1, int(len(loader_train) / print_frequency))
            step_now = 0
            optimizer.zero_grad()
            for id_sample, data in enumerate(loader_train):
                batch_dict = train(model, optimizer, loss_total_fun, data, device)

                # if (id_sample+1)%4==0 or (id_sample+1)==len(loader_train):
                #     optimizer.step()
                #     optimizer.zero_grad()
                if step_now < step_print:
                    step_now = step_now + 1
                    l0 = l0 + batch_dict['loss'][0]
                    l1 = l1 + batch_dict['loss'][1]
                    l2 = l2 + batch_dict['loss'][2]
                    l3 = l3 + batch_dict['loss'][3]
                    l4 = l4 + batch_dict['loss'][4]
                    l5 = l5 + batch_dict['loss'][5]
                    l6 = l6 + batch_dict['loss'][6]
                    l7 = l7 + batch_dict['loss'][7]
                    l8 = l8 + batch_dict['loss'][8]
                    l9 = l9 + batch_dict['loss'][9]
                    l10 = l10 + batch_dict['loss'][10]
                    l11 = l11 + batch_dict['loss'][11]
                if step_now == step_print:
                    step_now = 0
                    info = 'loss a%.3f p%.3f s%.3f m%.3f t%.3f tr%.3f_%.1f genb%.3f geni%.3f genpa%.3f genpo%.3f genkpl%.3f' % (
                        l0 / step_print, l1 / step_print, l2 / step_print, l3 / step_print,
                        l4 / step_print, l5 / step_print, l6 / step_print, l7 / step_print,
                        l8 / step_print, l9 / step_print, l10 / step_print, l11 / step_print)
                    t.update("Epoch %03d | train %04d/%04d | %s" %
                             (epoch, id_sample, len(loader_train) - 1, info))
                    l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            save_checkpoint(model, optimizer, loss_total_fun, epoch, iter_train, path_result)
            scheduler.step()
        '''
        ============================= validate =============================
        '''
        if cfg['validate_flag'] and (epoch + 1) % test_step == 0:
            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10,l11 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            step_print = max(1, int(len(loader_val) / print_frequency))
            step_now = 0
            for id_sample, data in enumerate(loader_val):
                batch_dict = validate(model, loss_total_fun, data, device)
                if step_now < step_print:
                    step_now = step_now + 1
                    l0 = l0 + batch_dict['loss'][0]
                    l1 = l1 + batch_dict['loss'][1]
                    l2 = l2 + batch_dict['loss'][2]
                    l3 = l3 + batch_dict['loss'][3]
                    l4 = l4 + batch_dict['loss'][4]
                    l5 = l5 + batch_dict['loss'][5]
                    l6 = l6 + batch_dict['loss'][6]
                    l7 = l7 + batch_dict['loss'][7]
                    l8 = l8 + batch_dict['loss'][8]
                    l9 = l9 + batch_dict['loss'][9]
                    l10 = l10 + batch_dict['loss'][10]
                    l11 = l11 + batch_dict['loss'][11]
                if step_now == step_print:
                    step_now = 0
                    info = 'loss a%.3f p%.3f s%.3f m%.3f t%.3f tr%.3f_%.1f genb%.3f geni%.3f genpa%.3f genpo%.3f genkpl%.3f' % (
                        l0 / step_print, l1 / step_print, l2 / step_print, l3 / step_print,
                        l4 / step_print, l5 / step_print, l6 / step_print, l7 / step_print,
                        l8 / step_print, l9 / step_print, l10 / step_print, l11 / step_print)
                    t.update("Epoch %03d | validate %04d/%04d | %s" %
                             (epoch, id_sample, len(loader_val) - 1, info))
                    l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10,l11 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        '''
        ============================== test ================================
        '''
        if cfg['test_flag'] and (epoch + 1) % test_step == 0:
            step_print = max(1, int(len(loader_test) / print_frequency))
            step_now = 0
            vlads = []
            kpts = []
            feas_original = []
            feas_fusion = []
            sequences = []
            poses = []
            for id_sample, data in enumerate(loader_test):
                batch_dict = test(model, data, device)
                # save_figure(batch_dict, epoch, path_result, cfg)
                sequences.append((batch_dict['sequence']).detach().cpu())
                vlads.append(batch_dict['vlads'].detach().cpu())
                poses.append(batch_dict['pose_query'].detach().cpu())
                kpts.append(batch_dict['key_points'].detach().cpu())
                if 'fea_kpt_fusion' in batch_dict.keys():
                    feas_fusion.append(batch_dict['fea_kpt_fusion'].detach().cpu().permute(0, 2, 1))
                feas_original.append(batch_dict['fea_kpt_original'].detach().cpu().permute(0, 2, 1))

                if step_now < step_print:
                    step_now = step_now + 1
                if step_now == step_print:
                    step_now = 0
                    t.update("Epoch %03d | test %05d/%05d" % (epoch, id_sample, len(loader_test)))
            vlads = torch.cat(vlads)
            kpts = torch.cat(kpts)
            feas_original = torch.cat(feas_original)
            if 'fea_kpt_fusion' in batch_dict.keys():
                feas_fusion = torch.cat(feas_fusion)
            else:
                feas_fusion=feas_original

            poses = torch.cat(poses)
            sequences = torch.cat(sequences)

            database = {'vlads': vlads,
                        'key_points': kpts,
                        'fea_kpt_original': feas_original,
                        'fea_kpt_fusion': feas_fusion,
                        'fea_kpt': feas_fusion,
                        'sequences': sequences,
                        'pose_query': poses}
            savepath = tools.make_save_path(path_result, 'database')

            torch.save(database, savepath + '/database_bevp.pth.tar')
            # print('save ' + savepath + '/database_%03d.pth.tar' % epoch)
            # exit()
            # database = torch.load('/data4/caodanyang/results/FUSIONLCD/07250/database/database_159.pth.tar')
            print()
            print('***************************************************************************************************************************************')
            print('Epoch %03d' % epoch)
            # feature_match(loader_val,database)
            result,recall_at_k = lcd(database)
            seq = torch.unique(sequences)
            for i in range((test_best.shape[0])):
                recall_at_k1=recall_at_k[i]
                for j in range((test_best.shape[1])):
                    test_best[i, j] = max([test_best[i, j], result[i][j]])
                print('Best, sequence %02d, AP=%.3f, R100=%.3f, F1=%.3f' % (seq[i],  test_best[i, 0], test_best[i, 1],test_best[i, 2]))
                lres.write(seq[i],epoch,(epoch+1)//test_step+i*(epochs)//test_step,
                           [result[i][0],result[i][1],result[i][2],recall_at_k1[0],recall_at_k1[1],recall_at_k1[2],recall_at_k1[3],recall_at_k1[4],
                            recall_at_k1[5],recall_at_k1[6],recall_at_k1[7],recall_at_k1[8],recall_at_k1[9],recall_at_k1[14],recall_at_k1[19],recall_at_k1[24]
                            ])
            #     print('Sequence %02d, AP=%.3f[%.3f], R100=%.3f[%.3f], F1=%.3f[%.3f], Recall@1[%.3f] 2[%.3f] 5[%.3f] 10[%.3f] 15[%.3f] 25[%.3f]' %
            #           (sequences[i], result[i][0], test_best[i, 0], result[i][1], test_best[i, 1], result[i][2], test_best[i, 2],
            #            recall_at_k1[0],recall_at_k1[1],recall_at_k1[4],recall_at_k1[9],recall_at_k1[14],recall_at_k1[24]))
            print('***************************************************************************************************************************************')
            print()
            if cfg['train_flag']:
                pass
            else:
                exit()
            # exit()

if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --result_name=08280 --info=cosim >log/08280.log 2>&1 &
    # fuser /dev/nvidia*
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_name', type=str,default='07030',help='log name of result')
    parser.add_argument('--pro_name', type=str,default='python',help='name of process')
    parser.add_argument('--info', type=str,default='python',help='name of process')
    args = parser.parse_args()
    print(args.info)
    try:
        print("Using GPU device:", os.environ["CUDA_VISIBLE_DEVICES"])
    except:
        pass
    main(args)