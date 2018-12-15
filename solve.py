import numpy as np
import torch
import random
import sys

num_centers = 1024
chunk_size = 16384

device = torch.device('cuda')

if __name__ == '__main__':
    dataset_numpy = np.load('dataset.npy')
    dataset = torch.from_numpy(dataset_numpy).to(device)
    num_points = dataset.size()[0]
    dimension = dataset.size()[1]
    centers = torch.zeros(num_centers, dimension, dtype=torch.float).to(device)
    used = torch.zeros(num_points, dtype=torch.long)
    for i in range(num_centers):
        while True:
            cur_id = random.randint(0, num_points - 1)
            if used[cur_id] > 0:
                continue
            used[cur_id] = 1
            centers[i] = dataset[cur_id]
            break
    centers = torch.transpose(centers, 0, 1)
    new_centers = torch.zeros(num_centers, dimension, dtype=torch.float).to(device)
    cnt = torch.zeros(num_centers, dtype=torch.float).to(device)
    all_ones = torch.ones(chunk_size, dtype=torch.float).to(device)
    if num_points % chunk_size != 0:
        all_ones_last = torch.ones(num_points % chunk_size, dtype=torch.float).to(device)
    all_ones_cnt = torch.ones(num_centers, dtype=torch.float).to(device)
    while True:
        centers_norms = torch.sum(centers ** 2, dim=0).view(1, -1)
        new_centers.fill_(0.0)
        cnt.fill_(0.0)
        for i in range(0, num_points, chunk_size):
            begin = i
            end = min(i + chunk_size, num_points)
            dataset_piece = dataset[begin:end, :]
            dataset_norms = torch.sum(dataset_piece ** 2, dim=1).view(-1, 1)
            distances = torch.mm(dataset_piece, centers)
            distances *= -2
            distances += dataset_norms
            distances += centers_norms
            _, min_ind = torch.min(distances, dim=1)
            new_centers.scatter_add_(0, min_ind.view(-1, 1).expand(-1, dimension), dataset_piece)
            if end - begin == chunk_size:
                cnt.scatter_add_(0, min_ind, all_ones)
            else:
                cnt.scatter_add_(0, min_ind, all_ones_last)
        sorted_cnt = torch.sort(cnt)[0]
        for i in range(num_centers):
            sys.stdout.write('%d ' % sorted_cnt[i])
        sys.stdout.write('\n==========\n')
        cnt = torch.where(cnt > 1e-3, cnt, all_ones_cnt)
        new_centers /= cnt.view(-1, 1)
        centers = torch.transpose(new_centers, 0, 1).clone()
