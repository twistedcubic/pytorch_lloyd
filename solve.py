import numpy as np
import torch
import random
import sys

num_centers = 65536
chunk_size = 8192
num_iterations = 20
k = 10
device = torch.device('cuda')
device_cpu = torch.device('cpu')

def eval_kmeans(queries, centers, centers_norms, codes ):
    queries_norms = torch.sum(queries ** 2, dim=1).view(-1, 1)
    distances = torch.mm(queries, centers)
    distances *= -2.0
    distances += queries_norms
    distances += centers_norms
    codes = codes.to(device_cpu)
    cnt = torch.zeros(num_centers, dtype=torch.long)
    bins = [[]] * num_centers
    for i in range(num_points):
        cnt[codes[i]] += 1
        bins[codes[i]].append(i)
    num_queries = answers.size()[0]
    for num_probes in range(1, num_centers + 1):
        _, probes = torch.topk(distances, num_probes, dim=1, largest=False)
        probes = probes.to(device_cpu)
        total_score = 0
        total_candidates = 0
        for i in range(num_queries):
            candidates = []
            tmp = set()
            for j in range(num_probes):
                candidates.append(cnt[probes[i, j]])
                tmp.add(int(probes[i, j]))
            overall_candidates = sum(candidates)
            score = 0
            for j in range(k):
                if int(codes[answers[i, j]]) in tmp:
                    score += 1
            total_score += score
            total_candidates += overall_candidates 
        print(num_probes, float(total_score) / float(k * num_queries), float(total_candidates) / float(num_queries))

if __name__ == '__main__':
    
    dataset_numpy = np.load('dataset.npy')
    queries_numpy = np.load('queries.npy')
    answers_numpy = np.load('answers.npy')
    
    dataset = torch.from_numpy(dataset_numpy).to(device)
    queries = torch.from_numpy(queries_numpy).to(device)
    answers = torch.from_numpy(answers_numpy)
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
    codes = torch.zeros(num_points, dtype=torch.long).to(device)
    for it in range(num_iterations):
        centers_norms = torch.sum(centers ** 2, dim=0).view(1, -1)
        new_centers.fill_(0.0)
        cnt.fill_(0.0)
        for i in range(0, num_points, chunk_size):
            begin = i
            end = min(i + chunk_size, num_points)
            dataset_piece = dataset[begin:end, :]
            dataset_norms = torch.sum(dataset_piece ** 2, dim=1).view(-1, 1)
            distances = torch.mm(dataset_piece, centers)
            distances *= -2.0
            distances += dataset_norms
            distances += centers_norms
            _, min_ind = torch.min(distances, dim=1)
            codes[begin:end] = min_ind
            new_centers.scatter_add_(0, min_ind.view(-1, 1).expand(-1, dimension), dataset_piece)
            if end - begin == chunk_size:
                cnt.scatter_add_(0, min_ind, all_ones)
            else:
                cnt.scatter_add_(0, min_ind, all_ones_last)
        print('Iteration %d is done' % it)
        if it + 1 == num_iterations:
            break
        cnt = torch.where(cnt > 1e-3, cnt, all_ones_cnt)
        new_centers /= cnt.view(-1, 1)
        centers = torch.transpose(new_centers, 0, 1).clone()
    eval_kmeans(queries, centers, centers_norms, codes)
