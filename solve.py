import numpy as np
import torch
import random
import sys

num_centers = 1024
chunk_size = 1024

if __name__ == '__main__':
    dataset_numpy = np.load('dataset.npy')
    dataset = torch.from_numpy(dataset_numpy)
    num_points = dataset.size()[0]
    dimension = dataset.size()[1]
    centers = torch.zeros(num_centers, dimension, dtype=torch.float)
    for i in range(num_centers):
        cur_id = random.randint(0, num_points - 1)
        centers[i] = dataset[cur_id]
    centers = torch.transpose(centers, 0, 1)
    new_centers = torch.zeros(num_centers, dimension, dtype=torch.float)
    cnt = torch.zeros(num_centers, dtype=torch.long)
    while True:
        print(centers.size())
        centers_norms = torch.sum(centers ** 2, dim=0).view(1, -1)
        new_centers.fill_(0.0)
        cnt.fill_(0)
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
            for j in range(begin, end):
                cnt[min_ind[j - begin]] += 1
                new_centers[min_ind[j - begin]] += dataset[j]
        for i in range(num_centers):
            if cnt[i] != 0:
                new_centers[i] /= cnt[i]
            else:
                cur_id = random.randint(0, num_points - 1)
                new_centers[i] = dataset[cur_id]
        centers = torch.transpose(new_centers, 0, 1).clone()
        print(torch.sort(cnt))
