import faiss
import torch
import torch.utils.data
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import SimilarityTransform
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from sklearn.neighbors import KDTree
import tools


def evaluate_model_with_emb(emb_list, datasets_list, positive_distance=5.):
    recall_sum = 0.
    start_idx = 0
    cont = 0
    F1_sum = 0.
    auc_sum = 0.
    auc_sum2 = 0.
    emb_list = emb_list.cpu().numpy()
    recall_p1s = []

    for dataset in datasets_list:
        poses = dataset.poses
        samples_num = len(dataset)
        finish_idx = start_idx + samples_num
        emb_sublist = emb_list[start_idx:finish_idx]

        recall, maxF1, wrong_auc, real_auc, recall_p1 = compute_recall(emb_sublist, poses, dataset, positive_distance)
        recall_sum = recall_sum + recall
        F1_sum += maxF1
        auc_sum += wrong_auc
        auc_sum2 += real_auc
        recall_p1s.append(recall_p1)
        start_idx = finish_idx
        cont += 1

    final_recall = recall_sum / cont
    return final_recall, F1_sum / cont, auc_sum / cont, auc_sum2 / cont, recall_p1s


def compute_recall(emb_list=None, dataset=None, positive_distance=5., key_points=None, kpt_features=None, model=None, varify=False, mode='ransac'):
    have_matches = []
    poses = dataset.poses
    for gt in dataset.gt:
        have_matches.append(gt['idx'])
    num_neighbors = 25
    recall_at_k = [0] * num_neighbors

    num_evaluated = 0
    emb_list = np.asarray(emb_list)

    for i in range(len(emb_list)):
        if hasattr(dataset, 'frames_with_gt'):
            if dataset.frames_with_gt[i] not in have_matches:
                continue
        elif i not in have_matches:
            continue
        min_range = max(0, i - 100)
        max_range = min(i + 100, len(emb_list))
        ignored_idxs = set(range(min_range, max_range))
        valid_idx = set(range(len(emb_list))) - ignored_idxs
        valid_idx = list(valid_idx)

        # tr = KDTree(emb_list[valid_idx])

        index = faiss.IndexFlatL2(emb_list.shape[1])
        index.add(emb_list[valid_idx])

        x = poses[i][0, 3]
        y = poses[i][1, 3]
        z = poses[i][2, 3]
        anchor_pose = torch.tensor([x, y, z])
        num_evaluated += 1
        # distances, indices = tr.query(np.array([emb_list[i]]), k=num_neighbors)

        D, I = index.search(emb_list[i:i + 1], num_neighbors)

        indices = I[0]
        for j in range(len(indices)):

            m = valid_idx[indices[j]]
            x = poses[m][0, 3]
            y = poses[m][1, 3]
            z = poses[m][2, 3]
            possible_match_pose = torch.tensor([x, y, z])
            distance = torch.norm(anchor_pose - possible_match_pose)
            if distance <= positive_distance:
                # if j == 0:
                # similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                # top1_similarity_score.append(similarity)
                recall_at_k[j] += 1
                break
    recall_at_k = (np.cumsum(recall_at_k) / float(num_evaluated)+1e-8) * 100

    map_tree_poses = KDTree(np.stack(poses)[:, :3, 3])

    index = faiss.IndexFlatL2(emb_list.shape[1])
    index.add(emb_list[:50])

    real_loop = []
    detected_loop = []
    distances = []
    total_frame = 0
    for i in range(100, emb_list.shape[0]):
        min_range = max(0, i - 50)  # Scan Context
        current_pose = torch.tensor(poses[i][:3, 3])
        indices = map_tree_poses.query_radius(np.expand_dims(current_pose, 0), positive_distance)
        valid_idxs = list(set(indices[0]) - set(range(min_range, emb_list.shape[0])))
        if len(valid_idxs) > 0:
            real_loop.append(1)
        else:
            real_loop.append(0)

        index.add(emb_list[i - 50:i - 49])
        nearest = index.search(emb_list[i:i + 1], 1)
        total_frame += 1

        if varify:
            id_cand = nearest[1][0][0]
            fea_query = kpt_features[i]
            fea_cand = kpt_features[id_cand]
            kpt_query = key_points[i]
            kpt_cand = key_points[id_cand]
            fea = torch.cat([fea_query.unsqueeze(0), fea_cand.unsqueeze(0)], dim=0).to(model.gamma.device)
            kpt = torch.cat([kpt_query.unsqueeze(0), kpt_cand.unsqueeze(0)], dim=0).to(model.gamma.device)
            dis_between_frame = 999
            if mode == 'uot':
                bd = {'kpt_features': fea,
                      'key_points': kpt}
                bd = model(bd)
                transformation = bd['transformation']
                dis_between_frame = torch.linalg.norm(transformation[0, :, 3]).cpu()
            elif mode == 'ransac':
                fea_query1 = fea_query.permute(1, 0).detach().cpu().numpy()
                fea_cand = fea_cand.permute(1, 0).detach().cpu().numpy()
                kpt_query1 = kpt_query.detach().cpu().numpy()
                kpt_cand = kpt_cand.detach().cpu().numpy()
                matches = match_descriptors(fea_query1, fea_cand, metric='cosine')
                # kpt_cand=bd['project_kpts'].squeeze().detach().cpu().numpy()
                try:
                    transf, inliers = ransac((kpt_query1[matches[:, 0], 0:3],
                                              kpt_cand[matches[:, 1], 0:3]),
                                             model_class=SimilarityTransform,
                                             min_samples=min(7, int(len(matches) * 0.5)),
                                             max_trials=20,
                                             residual_threshold=2)
                    if 0.95 < transf.scale < 1.05:
                        dis_between_frame = np.linalg.norm(transf.params[0:3, 3])
                except:
                    dis_between_frame = 999
            else:
                print('Input varify mode error')
                exit()
            detected_loop.append(-dis_between_frame)
        else:
            detected_loop.append(-nearest[0][0][0])

        candidate_pose = torch.tensor(poses[nearest[1][0][0]][:3, 3])
        distances.append((current_pose - candidate_pose).norm())

    precision, recall, _ = precision_recall_curve(real_loop, detected_loop)

    wrong_auc = average_precision_score(real_loop, detected_loop)
    F1 = [2 * ((precision[i] * recall[i]) / max((precision[i] + recall[i]), 1e-8)) for i in range(len(precision))]

    distances = np.array(distances)
    detected_loop = -np.array(detected_loop)
    real_loop = np.array(real_loop)
    precision2 = []
    recall2 = []
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
    if 1 in precision2:
        recall_p1 = np.max(np.array(recall2)[np.array(precision2) == 1])
    else:
        recall_p1 = 0
    real_auc = auc(recall2, precision2)
    # plt.subplot(1, 2, 1), plt.plot(recall, precision, 'b.', markersize=1)
    # plt.subplot(1, 2, 2), plt.plot(recall2, precision2, 'b.', markersize=1)
    # plt.show()
    return recall_at_k, np.array(F1).max(), wrong_auc, real_auc, recall_p1


if __name__ == '__main__':
    pass
