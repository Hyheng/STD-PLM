import numpy as np
import torch.nn as nn
import time
import os
from typing import Any, Dict, Optional, Tuple, Union
import logging
from matplotlib import pyplot as plt
import torch
import torchcde  
import networkx as nx

def draw_mape_node(mape_node, save_path):
    plt.figure()

    plt.plot(range(len(mape_node)),mape_node,label='mape')
    
    plt.ylabel('mape')
    plt.xlabel('node')
    #plt.legend()

    plt.savefig(save_path)
    plt.close()    

def draw_loss_line(train_loss_line, val_loss_line, save_path):
    plt.figure()

    plt.plot(train_loss_line['x'],train_loss_line['y'],label='train loss')

    plt.plot(val_loss_line['x'],val_loss_line['y'],label='val loss')
    
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()

    plt.savefig(save_path)
    plt.close()

def check_dir(path:str ,mkdir=False):
    if os.path.exists(path):
        return True
    elif mkdir:
        os.mkdir(path)
        return True
    
    return False


def print_conf(config: Dict,logger: Optional[logging.Logger]=None):
    if logger is not None:
        output = logger.info
    else :
        output = print
    output('*'*10+'config'+'*'*10)
    for section in config.keys():
        output(f'[{section}]')
        data = config[section]
        for key in data:
            output(f'{key} = {data[key]}\n')
    output('*'*26)

def get_time_str():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

def init_model(model: nn.Module, filter):
    for p in model.parameters():
        if filter is not None and not filter(p):
            continue
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

def lap_eig(adj_matrix):
    
    number_of_nodes , _ = adj_matrix.shape

    

    D = np.sum(adj_matrix,axis=1)

    D = np.divide(1,np.sqrt(D),out=np.zeros_like(D))
    D = np.nan_to_num(D,nan=0)
    D = np.diag(D)
    
    I = np.eye(number_of_nodes)

    L = I - D @ adj_matrix @ D


    eigval, eigvec = np.linalg.eig(L)

    return np.real(eigvec) , np.real(eigval)  # [N, N (channels)]  [N (channels),]

def norm_Adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    normalized Adj matrix: (D^hat)^{-1} A^hat; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N)  # 为邻接矩阵加上自连接
    D = np.diag(1.0/np.sum(W, axis=1))
    norm_Adj_matrix = np.dot(D, W)

    return norm_Adj_matrix


def topological_sort(adj_mx):
    N = adj_mx.shape[0]
    A = adj_mx*(1-np.eye(N)).astype(np.int32)
    d_in = adj_mx.sum(axis=0)

    node_order_list = []
    list_len = 0

    while list_len!=N:
        idx = np.argmin(d_in)
        node_order_list.append(idx)
        list_len+=1

        edge_out = A[idx,:]
        d_in -= edge_out
        d_in[idx] = np.inf

    return node_order_list, list(np.argsort(node_order_list))


def cal_shortest_path_length(adj_mx, distance_mx):
    N = adj_mx.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(0,N))

    for edge in np.argwhere(adj_mx!=0):
        i,j = edge
        if i>j:
            continue
        G.add_edge(i,j,weight=distance_mx[i,j])
        
    s = nx.shortest_path_length(G,weight='weight')
    d_mx = np.zeros_like(distance_mx)

    for tmp in s:
        u,path_dict = tmp
        dis_array = list(path_dict.items())
        dis_array.sort(key=lambda v:v[0])
        dis_array = list(zip(*dis_array))[1]
        d_mx[u] = np.array(dis_array)
    
    d_mx = np.where(np.logical_or(d_mx!=0,np.eye(N=N)!=0) ,d_mx,np.inf)
    return d_mx

def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)
        distaneA = adj_mx
        return adj_mx, distaneA

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {str(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()  # 略过表头那一行
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = str(row[0]), str(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:  # distance file中的id直接从0开始

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {str(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()  # 略过表头那一行
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = str(row[0]), str(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
                    distaneA[id_dict[j], id_dict[i]] = distance
            return A, distaneA

        else:  # distance file中的id直接从0开始

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    A[j, i] = 1
                    distaneA[i, j] = distance
                    distaneA[j, i] = distance
            return A, distaneA

def get_randmask(observed_mask, min_miss_ratio=0., max_miss_ratio=1.):
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    rand_for_mask = rand_for_mask.reshape(-1)
    sample_ratio = np.random.rand()
    sample_ratio = sample_ratio * (max_miss_ratio-min_miss_ratio) + min_miss_ratio
    num_observed = observed_mask.sum().item()
    num_masked = round(num_observed * sample_ratio)
    rand_for_mask[rand_for_mask.topk(num_masked).indices] = -1

    cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
    return cond_mask


def get_block_mask(observed_mask, target_strategy='hybrid',min_seq = 3,max_seq = 12):
    rand_sensor_mask = torch.rand_like(observed_mask)
    randint = np.random.randint
    sample_ratio = np.random.rand()
    sample_ratio = sample_ratio * 0.15
    mask = rand_sensor_mask < sample_ratio
    
    for col in range(observed_mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, observed_mask.shape[0] - 1)
        mask[idxs, col] = True
    rand_base_mask = torch.rand_like(observed_mask) < 0.05
    reverse_mask = mask | rand_base_mask
    block_mask = 1 - reverse_mask.to(torch.float32)

    cond_mask = observed_mask.clone()
    mask_choice = np.random.rand()
    if target_strategy == "hybrid" and mask_choice > 0.7:
        cond_mask = get_randmask(observed_mask, 0., 1.)
    else:
        cond_mask = block_mask * cond_mask

    return cond_mask

def linear_interpolate(data):
    #(B,L,C)
    return torchcde.linear_interpolation_coeffs(data)