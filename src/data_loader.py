import pickle
import os
import numpy as np
import os
from tqdm import tqdm
import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn.conv import MessagePassing
import pdb
import copy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import random
import utils
from time import time
import networkx as nx
import matplotlib.pyplot as plt
# import vf3py as vf3
from networkx.algorithms import isomorphism

class update_min_dist(MessagePassing): # batch_confirmed
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'min')
        super(update_min_dist, self).__init__(node_dim=0, **kwargs)
        
    def forward(self, x, edge_index):
        propagated = self.propagate(edge_index.flip([0]), x=x)
        return propagated
        
    def message(self, x_j):
        return x_j+1
    



    

def get_t01(subckt_dict):
    if 'inv' in subckt_dict.keys():
        inverter = subckt_dict['inv']
        buffer_list = []
        for i in range(len(inverter)):
            for j in range(i, len(inverter)):
                if inverter[i][0][1] == inverter[j][0][0]:
                    buffer_list.append([[inverter[i][0][0], inverter[j][0][1]], list(set(inverter[i][1]).union(set(inverter[j][1]))), list(set(inverter[i][2]).union(set(inverter[j][2])))])            
                if inverter[i][0][0] == inverter[j][0][1]:
                    buffer_list.append([[inverter[j][0][0], inverter[i][0][1]], list(set(inverter[i][1]).union(set(inverter[j][1]))), list(set(inverter[i][2]).union(set(inverter[j][2])))])    
        
        if len(buffer_list) == 0:
            return subckt_dict
        subckt_dict['t01'] = buffer_list

 
    return subckt_dict
    
    
    raise NotImplementedError
    # get all inv
    # if two invs are connected, then add it.


def get_t02(subckt_dict, node_names, entire_edge_list):
    gnd_or_vsource = ['vddh!', 'gnd!', 'vbb!', 'vdd!', 'vddh2!', '0']
    if 'io' in subckt_dict.keys():
        io_list = subckt_dict['io']
        t02_list = []
        for i in range(len(io_list)):
            io_sample = io_list[i]
            io_node_names = []
            io_boundary_names = []
            for boundary in io_sample[0]:
                io_boundary_names.append(node_names[boundary])
            for node_idx in io_sample[1]:
                io_node_names.append(node_names[node_idx])
            non_boundary_non_vsource = list(set(io_node_names) -set(io_boundary_names) -set(gnd_or_vsource))
            non_bv_parsed = [i.split('/') for i in non_boundary_non_vsource]
            MM12_list = []
            MM1_list = []
            XI161_307_related = []
            for component in non_bv_parsed:
                if component[-1] == 'MM12':
                    MM12_list.append(component)
                elif component[-1] == 'MM1':
                    MM1_list.append(component)
                if len(component) > 1:
                    if component[-2] in ['XI161', 'XI307']:
                        XI161_307_related.append('/'.join(component))
            smallest_index_MM12 = MM12_list.index(min(MM12_list, key=len))
            circuit_name = '/'.join(MM12_list[smallest_index_MM12][:-1])
            MM12_index_in_entire = node_names.index('/'.join(MM12_list[smallest_index_MM12]))
            smallest_index_MM1 = MM1_list.index(min(MM1_list, key=len))
            MM1_index_in_entire = node_names.index('/'.join(MM1_list[smallest_index_MM1]))
            MM1_within_edge_list_indices = [i for i, x in enumerate(entire_edge_list[0]) if x == MM1_index_in_entire]
            MM12_within_edge_list_indices = [i for i, x in enumerate(entire_edge_list[0]) if x == MM12_index_in_entire]
            MM1_io_candidate = [entire_edge_list[1][i] for i in MM1_within_edge_list_indices]
            MM12_io_candidate = [entire_edge_list[1][i] for i in MM12_within_edge_list_indices]
            io_index = list(set(MM1_io_candidate).intersection(MM12_io_candidate))[0]
            new_boundary = ['net101', 'net107', 'net93' ]
            rel_node_names = ['net52', 'net107', 'net93', 'net101', 'net63', 'net95', 'MM0', 'MM1', 'MM2', 'MM12' ]
            if circuit_name != '':
                related_node_names = ['{}/{}'.format(circuit_name,i) for i in rel_node_names] + XI161_307_related + ['gnd!', 'vbb!', 'vdd!', 'vddh!'] + [node_names[io_index]]
                new_boundary_names = ['{}/{}'.format(circuit_name,i) for i in new_boundary] + [node_names[io_index]]
            else: 
                related_node_names = ['{}'.format(i) for i in rel_node_names] + XI161_307_related + ['gnd!', 'vbb!', 'vdd!', 'vddh!'] + [node_names[io_index]]
                new_boundary_names = ['{}'.format(i) for i in new_boundary] + [node_names[io_index]]
            node_idx_list = []
            boundary_idx_list = []
            for rel_node in related_node_names:
                node_idx_list.append(io_sample[1][io_node_names.index(rel_node)])
            for b_node in new_boundary_names:
                boundary_idx_list.append(io_sample[1][io_node_names.index(b_node)])
            node_idx_list = sorted(node_idx_list)
            boundary_idx_list = sorted(boundary_idx_list)
            np_edge_list = np.asarray(entire_edge_list)
            io_related_edges = np_edge_list.T[io_sample[2]].T.tolist()
            new_edge_list = []
            for edge_idx in range(len(io_related_edges[0])):
                if io_related_edges[0][edge_idx] in node_idx_list:
                    if io_related_edges[1][edge_idx] in node_idx_list:
                        new_edge_list.append(edge_idx)
            new_edge_idx = np.asarray(io_sample[2])[new_edge_list].tolist()
            t02_list.append([boundary_idx_list, node_idx_list, new_edge_idx])
        if len(t02_list) == 0:
            return subckt_dict
        subckt_dict['t02'] = t02_list
    return subckt_dict
    
    
    # get all io
    # find indices that corresponds to t02.
    
    
    
    
    
class Data_Manager:
    def __init__(self, args):
        self.args = args
        self.target = args.target_circuit
        self.entire = args.entire_circuit
        self.root = args.root
        self.dataset = args.dataset
        self.proc_dir = self.root + '/processed_circuit/{}/'.format(self.dataset)
        self.src = self.root + '/src/'
        self.device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
        
        
        self.split_dir_name = '_'.join([str(int_to_str) for int_to_str in [self.args.num_repeat, self.args.num_positive, self.args.num_partial, \
                self.args.num_mutation, self.args.num_others, self.args.num_random, self.args.num_positive_test, self.args.num_partial_test, \
                    self.args.num_mutation_test, self.args.num_others_test, self.args.num_random_test]])
        # split_dir 이름 : split 개수 + positive 개수 + partial 개수 + mutation 개수 + others 개수 + random 개수 + positive_test 개수 + partial_test 개수 + mutation_test 개수 + others_test개수 + random_test 개수

        if args.diameter:
            self.k_option = 'diameter'
        if args.radius:
            self.k_option = 'radius'
        if args.diameter and args.radius:
            self.k_option = 'both'
        
        self.split_dir = self.proc_dir+'splits/{}/'.format(self.split_dir_name)
    
    
    
    
    def generate_sample_with_option_file(self): 
        # This code is not tested yet.
        opened_file = open(self.src+'embedder_train_option/train_targets.txt', 'r')
        target_ckt_list = opened_file.read().split('\n')[0].split(' ')
        opened_file.close()
        if self.args.train_with_test:
            target_ckt_list.append('t01')
            target_ckt_list.append('t02')
        opened_file = open(self.src+'embedder_train_option/entire_circuits.txt', 'r')
        entire_ckt_list = opened_file.read().split('\n')[0].split(' ')
        opened_file.close()
        for target_ckt in tqdm(target_ckt_list):
            self.target = target_ckt
            for entire_ckt in tqdm(entire_ckt_list):
                self.entire = entire_ckt
                self.generate_train_sample_for_embedder()
        return 


        '''
        (Pdb) entire_graph
        [['a0', 'i_o', 'io', 'io_', 'main_ena', 'pre', 'w_'], 
        ['a0', 'i_o', 'io', 'io_', 'main_ena', 'pre', 'w_', 'vdd!', 'net107', 'net63', 'net83', 'net73', 'vbb!', 'gnd!', 'net70', 'net119', 'net95', 'net111', 'net101', 'net52', 'net93', 'MM0', 'MM12', 'MM1', 'MM2', 'XI210/net20', 'XI210/net21', 'XI210/XI165/MM1', 'XI210/XI165/MM12', 'XI210/MM5', 'XI210/MM2', 'XI210/MM3', 'XI210/MM4', 'XI210/MM0', 'XI210/MM1', 'XI210/MM17', 'XI218/MM0', 'XI218/MM1', 'XI196/MM0', 'XI196/MM1', 'XI164/MM0', 'XI164/MM1', 'XI161/MM0', 'XI161/MM1', 'XI217/MM0', 'XI217/MM1', 'XI224/MM1', 'XI224/MM12', 'XI306/MM1', 'XI306/MM12', 'XI165/MM1', 'XI165/MM12', 'XI307/MM1', 'XI307/MM12', 'XI219/MM1', 'XI219/MM12', 'vddh!', 'XI195/MM4', 'XI195/MM6', 'XI195/MM3', 'CC20', 'CC22'], [6, 6, 6, 6, 6, 6, 6, 3, 7, 7, 7, 7, 2, 1, 7, 7, 7, 7, 7, 7, 7, 9, 9, 8, 8, 7, 7, 8, 9, 9, 9, 9, 9, 8, 8, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 0, 9, 9, 9, 10, 10], ['None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'W=1.4u,L=350.00n,M=1', 'W=1.4u,L=350.00n,M=1', 'W=3.5u,L=350.00n,M=1', 'W=3.5u,L=350.00n,M=1', 'None', 'None', 'W=8u,L=0.35u,M=1', 'W=4u,L=0.35u,M=1', 'W=1.4u,L=350.00n,M=1', 'W=7u,L=350.00n,M=1', 'W=7u,L=350.00n,M=1', 'W=7u,L=350.00n,M=1', 'W=14u,L=350.00n,M=1', 'W=1.4u,L=350.00n,M=1', 'W=14u,L=350.00n,M=1', 'W=3.5u,L=0.35u,M=1', 'W=3.5u,L=0.35u,M=1.0', 'W=3.5u,L=0.35u,M=1', 'W=3.5u,L=0.35u,M=1.0', 'W=3.5u,L=0.35u,M=1', 'W=3.5u,L=0.35u,M=1.0', 'W=3.5u,L=0.35u,M=1', 'W=3.5u,L=0.35u,M=1.0', 'W=3.5u,L=0.35u,M=1', 'W=3.5u,L=0.35u,M=1.0', 'W=2.8u,L=0.35u,M=1', 'W=1.4u,L=0.35u,M=1', 'W=8u,L=0.35u,M=1', 'W=4u,L=0.35u,M=1', 'W=8u,L=0.35u,M=1', 'W=4u,L=0.35u,M=1', 'W=8u,L=0.35u,M=1', 'W=4u,L=0.35u,M=1', 'W=8u,L=0.35u,M=1', 'W=4u,L=0.35u,M=1', 'None', 'W=1.4u,L=350.00n,M=1', 'W=1.4u,L=350.00n,M=1', 'W=1.4u,L=350.00n,M=1', '200f', '200f'], 
        [[19, 21, 8, 21, 13, 12, 1, 22, 20, 22, 19, 22, 12, 1, 23, 18, 23, 9, 23, 7, 9, 24, 8, 24, 7, 7, 25, 27, 5, 27, 7, 7, 25, 28, 5, 28, 13, 12, 11, 29, 5, 29, 14, 29, 12, 14, 30, 2, 30, 26, 30, 12, 11, 31, 3, 31, 26, 31, 12, 26, 32, 4, 32, 13, 12, 7, 14, 33, 11, 33, 7, 14, 34, 25, 34, 11, 34, 7, 7, 11, 35, 14, 35, 7, 10, 36, 17, 36, 11, 36, 12, 10, 37, 0, 37, 11, 37, 7, 16, 38, 0, 38, 3, 38, 12, 16, 39, 15, 39, 3, 39, 7, 16, 40, 15, 40, 2, 40, 12, 16, 41, 0, 41, 2, 41, 7, 1, 42, 18, 42, 16, 42, 12, 1, 43, 20, 43, 16, 43, 7, 10, 44, 0, 44, 14, 44, 12, 10, 45, 17, 45, 14, 45, 7, 8, 46, 10, 46, 7, 7, 8, 47, 10, 47, 13, 12, 18, 48, 6, 48, 7, 7, 18, 49, 6, 49, 13, 12, 15, 50, 0, 50, 7, 7, 15, 51, 0, 51, 13, 12, 20, 52, 18, 52, 7, 7, 20, 53, 18, 53, 13, 12, 17, 54, 0, 54, 7, 7, 17, 55, 0, 55, 13, 12, 56, 5, 57, 2, 57, 12, 56, 5, 58, 3, 58, 12, 2, 59, 5, 59, 3, 59, 12, 3, 60, 13, 2, 61, 13], [21, 19, 21, 8, 21, 21, 22, 1, 22, 20, 22, 19, 22, 23, 1, 23, 18, 23, 9, 23, 24, 9, 24, 8, 24, 24, 27, 25, 27, 5, 27, 27, 28, 25, 28, 5, 28, 28, 29, 11, 29, 5, 29, 14, 29, 30, 14, 30, 2, 30, 26, 30, 31, 11, 31, 3, 31, 26, 31, 32, 26, 32, 4, 32, 32, 33, 33, 14, 33, 11, 33, 34, 14, 34, 25, 34, 11, 34, 35, 35, 11, 35, 14, 35, 36, 10, 36, 17, 36, 11, 36, 37, 10, 37, 0, 37, 11, 37, 38, 16, 38, 0, 38, 3, 38, 39, 16, 39, 15, 39, 3, 39, 40, 16, 40, 15, 40, 2, 40, 41, 16, 41, 0, 41, 2, 41, 42, 1, 42, 18, 42, 16, 42, 43, 1, 43, 20, 43, 16, 43, 44, 10, 44, 0, 44, 14, 44, 45, 10, 45, 17, 45, 14, 45, 46, 8, 46, 10, 46, 46, 47, 8, 47, 10, 47, 47, 48, 18, 48, 6, 48, 48, 49, 18, 49, 6, 49, 49, 50, 15, 50, 0, 50, 50, 51, 15, 51, 0, 51, 51, 52, 20, 52, 18, 52, 52, 53, 20, 53, 18, 53, 53, 54, 17, 54, 0, 54, 54, 55, 17, 55, 0, 55, 55, 57, 57, 5, 57, 2, 57, 58, 58, 5, 58, 3, 58, 59, 2, 59, 5, 59, 3, 59, 60, 3, 60, 61, 2, 61]], 
        [4, 18, 5, 19, 6, 7, 4, 18, 5, 19, 6, 20, 7, 0, 14, 1, 15, 2, 16, 3, 0, 14, 1, 15, 2, 3, 0, 14, 1, 15, 2, 3, 4, 18, 5, 19, 6, 7, 4, 18, 5, 19, 6, 20, 7, 4, 18, 5, 19, 6, 20, 7, 4, 18, 5, 19, 6, 20, 7, 4, 18, 5, 19, 6, 7, 0, 1, 15, 2, 16, 3, 0, 14, 1, 15, 2, 16, 3, 0, 1, 15, 2, 16, 3, 4, 18, 5, 19, 6, 20, 7, 0, 14, 1, 15, 2, 16, 3, 4, 18, 5, 19, 6, 20, 7, 0, 14, 1, 15, 2, 16, 3, 4, 18, 5, 19, 6, 20, 7, 0, 14, 1, 15, 2, 16, 3, 4, 18, 5, 19, 6, 20, 7, 0, 14, 1, 15, 2, 16, 3, 4, 18, 5, 19, 6, 20, 7, 0, 14, 1, 15, 2, 16, 3, 0, 14, 1, 15, 2, 3, 4, 18, 5, 19, 6, 7, 0, 14, 1, 15, 2, 3, 4, 18, 5, 19, 6, 7, 0, 14, 1, 15, 2, 3, 4, 18, 5, 19, 6, 7, 0, 14, 1, 15, 2, 3, 4, 18, 5, 19, 6, 7, 0, 14, 1, 15, 2, 3, 4, 18, 5, 19, 6, 7, 4, 5, 19, 6, 20, 7, 4, 5, 19, 6, 20, 7, 4, 18, 5, 19, 6, 20, 7, 8, 22, 9, 8, 22, 9]]
        
        (pdb) entire_subckt_dict
        {'inv': [[[5, 25], [5, 25, 13, 12, 7, 27, 28], [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]], 
        [[10, 8], [10, 8, 13, 12, 7, 46, 47], [154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165]], 
        [[6, 18], [6, 18, 13, 12, 7, 48, 49], [166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177]], 
        [[0, 15], [0, 15, 13, 12, 7, 50, 51], [178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189]], 
        [[18, 20], [18, 20, 13, 12, 7, 52, 53], [190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201]], 
        [[0, 17], [0, 17, 13, 12, 7, 54, 55], [202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213]]], 
        'main_sense': [[[4, 2, 3, 11, 14, 5], [4, 2, 3, 11, 14, 5, 7, 13, 25, 12, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]]], 
        'ctg': [[[11, 10, 17, 0], [11, 10, 17, 0, 12, 7, 36, 37], [84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97]], 
        [[3, 16, 0, 15], [3, 16, 0, 15, 12, 7, 38, 39], [98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]], 
        [[2, 16, 15, 0], [2, 16, 15, 0, 12, 7, 40, 41], [112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125]], 
        [[16, 1, 18, 20], [16, 1, 18, 20, 12, 7, 42, 43], [126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139]], 
        [[14, 10, 0, 17], [14, 10, 0, 17, 12, 7, 44, 45], [140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153]]], 
        'precharge': [[[2, 3, 5], [2, 3, 5, 56, 12, 57, 58, 59], [214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232]]], 
        'io': [[[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238]]]}
        
        '''

       
    def generate_vf3_labels_with_option_file(self): 
        # This code is not tested yet.
        labels_dict = {}
        dir_flag = 'dir' if self.args.gnd_vsource_directional else 'undir'
        redun_flag = 'redun' if self.args.gnd_vsource_allow_redundancy else 'noredun'

        
        # # read pickle file for dictionary
        # pickle_file_name = self.proc_dir +'graph_files/io_dir_noredun_labels.pkl'
        # # pickle_file_name = self.proc_dir +'graph_files/io_dir_noredun.pkl'
        # with open(pickle_file_name, 'rb') as pickle_file:
        #     subckt_dict = pickle.load(pickle_file) # 해당 target_circuit 안에 어떤 subcircuit이 있는지..
        # pdb.set_trace()
    
        opened_file = open(self.src+'embedder_train_option/train_targets.txt', 'r')
        target_ckt_list = opened_file.read().split('\n')[0].split(' ')
        opened_file.close()
        if self.args.train_with_test:
            target_ckt_list.append('t01')
            target_ckt_list.append('t02')
        opened_file = open(self.src+'embedder_train_option/entire_circuits.txt', 'r')
        entire_ckt_list = opened_file.read().split('\n')[0].split(' ')
        opened_file.close()
        # pdb.set_trace()

        # for entire_ckt in entire_ckt_list:
        for entire_idx, entire_ckt in enumerate(entire_ckt_list):
            # entire_idx = 29
            # entire_ckt = 'ctrl256kbm'
            self.entire = entire_ckt
            entire_graph, prev_entire_dict = self.read_circuit(self.entire)
            # pdb.set_trace()
            labels_dict[entire_ckt] = {}
            # for target_ckt in target_ckt_list:
            for target_idx, target_ckt in enumerate(target_ckt_list):
                # if entire_idx <= target_idx or target_ckt in ['col_sel3_8m', 'deco3_8m', 'decoder4_16', 'atd18m', 'col_sel7_128', 'deco7_128', 'col_sel9_512', 'deco9_512', 'ctrl256kbm']:
                #     continue
                # if entire_ckt =='ctrl256kbm' and target_ckt in ['rl_sel', 'delay6', 'sen_ena', 'delay16','xvald']:
                #     continue    
                self.target = target_ckt
                # if not (self.target in prev_entire_dict.keys()):
                #     continue
                labels = self.get_label_by_vf3()
                if len(labels) != 0:
                    labels_dict[entire_ckt][target_ckt] = labels
            # labels_dict[entire_ckt][entire_ckt] = [[list(range(len(entire_graph[1]))), list(range(len(entire_graph[5])))]]
            # pickle_file_name = self.proc_dir +'graph_files/{}_{}_{}_labels3.pkl'.format(entire_ckt, dir_flag, redun_flag)
            # with open(pickle_file_name, 'wb') as opened_file :
            #     pickle.dump(labels_dict[entire_ckt], opened_file)
        
        return labels_dict
        
        
    def get_label_by_vf3(self):
        # self.entire = 'atd18'
        # self.target = 'delay10'
        # self.entire = 'io'
        # self.target = 'inv'
    
        entire_graph, prev_entire_dict = self.read_circuit(self.entire)
        target_graph, _ = self.read_circuit(self.target)
        # if not (self.target in prev_entire_dict.keys()):
        #     return None
        #colormaps = ['#000000','#000033','#000066','#000099','#0000CC','#0000FF','#330000','#330033','#330066','#330099','#3300CC','#3300FF','#660000']
        colormaps = ['bisque','forestgreen','slategrey', 'darkorange', 'limegreen', 'lightsteelblue', 'burlywood', 'darkgreen', 'cornflowerblue'] #, 'gray', 'black', 'dimgray']
        print()
        print('entire_ckt : {}, target_ckt : {}'.format(self.entire, self.target))
        print('number of previous labels : {}'.format(len(prev_entire_dict.get(self.target, []))))
        
        # create graph with networkx
        # pdb.set_trace()
        nx_target_graph = nx.MultiDiGraph()
        target_node_labels = np.asarray(target_graph[1])
        target_node_types = np.asarray(target_graph[2])
        target_edge_list = np.asarray(target_graph[4])
        target_edge_types = np.asarray(target_graph[5])
        
        """
    #         0: 0, 2: 0, 3: 0, 4: 0,  # 그룹 0
    #         1: 1, 5: 1,              # 그룹 1
    #         6: 2, 7: 2,              # 그룹 2
    #         8: 3,                    # 그룹 3
    #         9: 4,                    # 그룹 4
    #         10: 5,                   # 그룹 5
    #         11: 6,                   # 그룹 6
    #         12: 7,                   # 그룹 7
    #         13: 8                    # 그룹 8
        """

        for node_type in range(13):
            indices = np.where(target_node_types == node_type)[0].tolist() # 해당 node_type을 가진 node_idx 추출
            if len(indices) > 0:
                if node_type == 0 or node_type == 2 or node_type == 3 or node_type == 4:
                    nx_target_graph.add_nodes_from(indices, color=colormaps[0])
                elif node_type == 1 or node_type == 5:
                    nx_target_graph.add_nodes_from(indices, color=colormaps[1])
                elif node_type == 6 or node_type == 7:
                    nx_target_graph.add_nodes_from(indices, color=colormaps[2])
                else:
                    nx_target_graph.add_nodes_from(indices, color=colormaps[node_type-5])

        for edge, edge_type in zip(target_edge_list.T.tolist(), target_edge_types):
            nx_target_graph.add_edge(edge[0], edge[1], feature=edge_type)

        if self.target in ['deco1m', 'col_sel1m']:
            nx_target_graph.remove_nodes_from([4, 8, 12])
            nx_target_graph.remove_edges_from([(4, 12, 0), (8, 12, 0), (12, 4, 0)])

        # # Assign colors to nodes in the target graph based on node type
        # target_labels = {}
        # # pdb.set_trace() 
        # for i, node_type in enumerate(target_node_types):
        #     # target_colors.append(colormaps[node_type])  # Assign color based on node type
        #     target_labels[i] = target_node_labels[i] 

        # target_colors = [nx_target_graph.nodes[node]['color'] for node in nx_target_graph.nodes]
        # # target_colors = [target_colors[n] for n in nx_target_graph.nodes()]
        # # Draw target graph with colors
        # # nx.draw(nx_target_graph, node_color=target_colors, labels=target_labels, with_labels=True)
        # # plt.show()

        '''
        nx_target_graph.add_nodes_from([i for i in range(target_node_types.shape[0])])
        for node in nx_target_graph.nodes:
            if target_node_types[node] == 6:
                nx_target_graph.nodes[node]['color'] = colormaps[7]
            else:
                nx_target_graph.nodes[node]['color'] = colormaps[target_node_types[node]]
        '''
            
            
            
        nx_entire_graph = nx.MultiDiGraph()
        entire_node_labels = np.asarray(entire_graph[1])
        entire_node_types = np.asarray(entire_graph[2])
        entire_edge_list = np.asarray(entire_graph[4])
        entire_edge_types = np.asarray(entire_graph[5])


        for node_type in range(13):
            indices = np.where(entire_node_types == node_type)[0].tolist()
            if len(indices) > 0:
                if node_type == 0 or node_type == 2 or node_type == 3 or node_type == 4:
                    nx_entire_graph.add_nodes_from(indices, color=colormaps[0])
                elif node_type == 1 or node_type == 5:
                    nx_entire_graph.add_nodes_from(indices, color=colormaps[1])
                elif node_type == 6 or node_type == 7:
                    nx_entire_graph.add_nodes_from(indices, color=colormaps[2])
                else:
                    nx_entire_graph.add_nodes_from(indices, color=colormaps[node_type-5])

        for edge, edge_type in zip(entire_edge_list.T.tolist(), entire_edge_types):
            nx_entire_graph.add_edge(edge[0], edge[1], feature=edge_type)



        # Assign colors to nodes in the target graph based on node type
        entire_labels = {}
        # pdb.set_trace() 
        for i, node_type in enumerate(entire_node_types):
            # entire_colors.append(colormaps[node_type])  # Assign color based on node type
            entire_labels[i] = entire_node_labels[i] 

        entire_colors = [nx_entire_graph.nodes[node]['color'] for node in nx_entire_graph.nodes]
        # entire_colors = [entire_colors[n] for n in nx_entire_graph.nodes()]
        # Draw target graph with colors

        # nx.draw(nx_entire_graph, node_color=entire_colors, labels=entire_labels, with_labels=True)
        # plt.show()



        start_time = time()

        # Define a node matcher for vf2 (same as for vf3)
        node_match = nx.algorithms.isomorphism.categorical_node_match("color", None)
        # lambda subgraph_dict, graph_dict: subgraph_dict['color'] == graph_dict['color']
        edge_match = nx.algorithms.isomorphism.categorical_multiedge_match("feature", None)
        # lambda subgraph_dict, graph_dict: subgraph_dict['feature'] == graph_dict['feature']

        # Initialize the vf2 matcher with target and entire graph and node matching rule
        matcher = isomorphism.MultiDiGraphMatcher(nx_entire_graph, nx_target_graph, node_match=node_match, edge_match=edge_match)
        
        # Get all subgraph isomorphisms
        vf2_labels = list(matcher.subgraph_isomorphisms_iter())

        # for subgraph_mapping in matcher.subgraph_isomorphisms_iter():
        #     print("Matching Subgraph Found:")
        #     print(subgraph_mapping)

        # all_matching_list = [list(subgraph_mapping.keys()) for subgraph_mapping in vf2_labels]
        all_matching_list = [sorted(subgraph_mapping, key=lambda k: subgraph_mapping[k]) for subgraph_mapping in vf2_labels]

        all_edge_indices = []

        for subgraph_mapping in vf2_labels:
            # Collect edges by looking at both node pairings and edge connections
            subgraph_edges = []
            for node_in_entire, node_in_target in subgraph_mapping.items():
                for edge in nx_target_graph.edges([node_in_target]):
                    # Translate the target graph's edge back to the entire graph using the mapping
                    mapped_edge = (
                        next((k for k, v in subgraph_mapping.items() if v == edge[0]), None),
                        next((k for k, v in subgraph_mapping.items() if v == edge[1]), None),
                    )
                    # Ensure both nodes in the edge are mapped
                    if mapped_edge[0] is not None and mapped_edge[1] is not None:
                        subgraph_edges.append(mapped_edge)
            
            # Add the subgraph's edge indices (from the entire graph)
            all_edge_indices.append(subgraph_edges)
    

        # all_edge_indices와 entire_graph[4]의 매칭 인덱스를 계산
        matched_indices = []

        # source와 target 리스트로 분리
        entire_source = entire_graph[4][0]
        entire_target = entire_graph[4][1]

        # 전체 엣지를 (source, target) 쌍으로 변환
        entire_edges = list(zip(entire_source, entire_target))

        # 이미 선택된 엣지 인덱스를 추적
        used_indices = set()

        # all_edge_indices 내 각 서브그래프의 엣지들을 검사
        for subgraph_edges in all_edge_indices:
            subgraph_indices = []
            for edge in subgraph_edges:
                # edge가 entire_edges에서 몇 번째 위치에 있는지 찾음
                for i, entire_edge in enumerate(entire_edges):
                    if edge == entire_edge and i not in used_indices:
                        subgraph_indices.append(i)
                        used_indices.add(i)  # 선택된 인덱스를 사용된 집합에 추가
                        break  # 같은 엣지를 여러 번 선택하지 않도록 중단
            # 서브그래프의 인덱스를 정렬된 형태로 matched_indices에 추가
            matched_indices.append(sorted(subgraph_indices))

        final_results = [[node_list, edge_list] for node_list, edge_list in zip(all_matching_list, matched_indices)]

        end_time = time()
        print('Number of vf2 labels : {}'.format(len(vf2_labels)))
        print('Time taken for getting labels : {}'.format(end_time - start_time))
        # pdb.set_trace()
        # pdb.set_trace() 
        # vf3_labels = vf3.get_subgraph_isomorphisms(
        #     subgraph=nx_target_graph,
        #     graph=nx_entire_graph,
        #     node_match = lambda subgraph_dict, graph_dict: subgraph_dict['color'] == graph_dict['color']
        # )
        # end_time = time()
        # print('number of vf3 labels : {}'.format(len(vf3_labels)))
        # print('time took for getting labels : {}'.format(end_time-start_time))
        # #pdb.set_trace()
        return final_results
        
    
    def get_graph_data(self):
        self.entire_ckt = self.args.entire_circuit
        entire_graph, entire_subckt_dict = self.read_circuit(self.entire_ckt)
        entire_node_names  = entire_graph[1]
        entire_edge_list = torch.LongTensor(entire_graph[4])
        entire_node_types = torch.LongTensor(np.asarray(entire_graph[2]))
        entire_edge_types = torch.LongTensor(np.asarray(entire_graph[5]))

        group_mapping = {0: 0, 2: 0, 3: 0, 4: 0,  # Group 0
                                    1: 1, 5: 1,              # Group 1
                                    6: 2, 7: 2,              # Group 2
                                    8: 3,                    # Group 3
                                    9: 4,                    # Group 4
                                    10: 5,                   # Group 5
                                    11: 6,                   # Group 6
                                    12: 7}                    # Group 7

        grouped_node_types = torch.tensor([group_mapping[node_type.item()] for node_type in entire_node_types])
        entire_x = torch.nn.functional.one_hot(grouped_node_types, num_classes=8).float()#.to(self.device)
        entire_data =  Data(x=entire_x, edge_index=entire_edge_list, edge_type=entire_edge_types)

        self.target_ckt = self.args.target_circuit
        target_graph, _ = self.read_circuit(self.target_ckt) 
        target_node_names  = target_graph[1]
        target_edge_list = torch.LongTensor(target_graph[4])
        target_node_types = torch.LongTensor(np.asarray(target_graph[2]))
        target_edge_types = torch.LongTensor(np.asarray(target_graph[5]))

        group_mapping = {0: 0, 2: 0, 3: 0, 4: 0,  # Group 0
                                    1: 1, 5: 1,              # Group 1
                                    6: 2, 7: 2,              # Group 2
                                    8: 3,                    # Group 3
                                    9: 4,                    # Group 4
                                    10: 5,                   # Group 5
                                    11: 6,                   # Group 6
                                    12: 7}                    # Group 7

        grouped_node_types = torch.tensor([group_mapping[node_type.item()] for node_type in target_node_types])
        target_x = torch.nn.functional.one_hot(grouped_node_types, num_classes=8).float()#.to(self.device)
        target_data =  Data(x=target_x, edge_index=target_edge_list, edge_type=target_edge_types)

        target_vsource_index = np.arange(len(target_graph[1]))[np.asarray(target_graph[2])<6].tolist()
        diameter, radius = target_detect_k([i for i in range(len(target_graph[1]))], target_graph[4], target_graph[4], 'diameter', target_vsource_index)

        return target_data, entire_data, radius
        

    def label_count(self):
        entire = self.args.entire_circuit
        target = self.args.target_circuit

        entire_pickle_file = "../processed_circuit/company/graph_files/"+ entire +"_dir_noredun_labels3.pkl"
        with open(entire_pickle_file, 'rb') as pickle_file:
            subckt_dict = pickle.load(pickle_file) # 해당 target_circuit 안에 어떤 subcircuit이 있는지..
            if entire == 'ctrl256kbm' and target == 'atd1':
                return 2
            if entire == 'ctrl256kbm' and target == 'col_sel1m':
                return 1024
            if entire == 'ctrl256kbm' and target == 'decoder2_4':
                return 2
            if entire == 'ctrl256kbm' and target == 'deco1m':
                return 1024
            if entire == 'ctrl256kbm' and target == 'delay4':
                return 440
            if entire == 'deco9_512' and target == 'deco1m':
                return 512
            if entire == 'deco9_512' and target == 'col_sel1m':
                return 512
            if entire == 'col_sel9_512' and target == 'col_sel1m':
                return 512
            return len(subckt_dict[target])




    def get_all_k_hop(self, get_label=False):
        self.entire_ckt = self.args.entire_circuit
        self.target_ckt = self.args.target_circuit
        entire_graph, entire_subckt_dict = self.read_circuit(self.entire_ckt)
        # pdb.set_trace()
        target_graph, _ = self.read_circuit(self.target_ckt)
        entire_node_names  = entire_graph[1]
        entire_edge_list = entire_graph[4]
        entire_node_types = np.asarray(entire_graph[2])
        entire_edge_types = np.asarray(entire_graph[5])
        gnd_vsource_index = np.arange(len(entire_node_names))[entire_node_types<6].tolist()
        target_vsource_index = np.arange(len(target_graph[1]))[np.asarray(target_graph[2])<6].tolist()
        diameter, radius = target_detect_k([i for i in range(len(target_graph[1]))], target_graph[4], target_graph[4], 'diameter', target_vsource_index)
        graphs = []
        labels = []
        raw_graphs = []
        num_target_node = len(target_graph[1])
        for node_idx in range(len(entire_graph[1])):
            node_idx_map, new_edge_index, mapping, edge_mask = k_hop_subgraph(torch.LongTensor([node_idx]), radius, torch.LongTensor(entire_edge_list).flip([0]), relabel_nodes=True, flow="target_to_source")
            new_edge_index = new_edge_index.flip([0])#.to(self.device)
            selected_edge_index = ((edge_mask == True).nonzero(as_tuple=True)[0])
            if num_target_node >= node_idx_map.size(0):
                continue
            node_types = torch.from_numpy(entire_node_types[node_idx_map.tolist()])#.to(self.device)
            edge_types = torch.from_numpy(entire_edge_types[selected_edge_index.tolist()])#.to(self.device)
            
            group_mapping = {0: 0, 2: 0, 3: 0, 4: 0,  # Group 0
                                    1: 1, 5: 1,              # Group 1
                                    6: 2, 7: 2,              # Group 2
                                    8: 3,                    # Group 3
                                    9: 4,                    # Group 4
                                    10: 5,                   # Group 5
                                    11: 6,                   # Group 6
                                    12: 7}                    # Group 7

            grouped_node_types = torch.tensor([group_mapping[node_type.item()] for node_type in node_types])
            x = torch.nn.functional.one_hot(grouped_node_types, num_classes=8).float()#.to(self.device)
            
            # x = torch.nn.functional.one_hot(node_types, num_classes=14).float().to(self.device)
            data = Data(x=x, edge_index=new_edge_index, edge_type=edge_types)
            if get_label:
                is_positive = False
                for positive_samples in entire_subckt_dict[self.target_ckt]:
                    if set(positive_samples[0]).issubset(set(node_idx_map.tolist())):
                        if set(positive_samples[1]).issubset(set(selected_edge_index.tolist())):
                            is_positive = True 
                if is_positive: 
                    labels.append(1.0)
                else: 
                    labels.append(0.0)
            graphs.append(data)
            raw_graphs.append([node_idx_map.tolist(), selected_edge_index.tolist()])
        edge_types = torch.LongTensor(target_graph[5])#.to(self.device)
        node_types = torch.LongTensor(target_graph[2])#.to(self.device)

        group_mapping = {0: 0, 2: 0, 3: 0, 4: 0,  # Group 0
                                    1: 1, 5: 1,              # Group 1
                                    6: 2, 7: 2,              # Group 2
                                    8: 3,                    # Group 3
                                    9: 4,                    # Group 4
                                    10: 5,                   # Group 5
                                    11: 6,                   # Group 6
                                    12: 7}                    # Group 7

        grouped_node_types = torch.tensor([group_mapping[node_type.item()] for node_type in node_types])
        x = torch.nn.functional.one_hot(grouped_node_types, num_classes=8).float()#.to(self.device)

        # x = torch.nn.functional.one_hot(node_types, num_classes=14).float().to(self.device)
        edge_list_target = torch.LongTensor(target_graph[4])#.to(self.device)
        target =  Data(x=x, edge_index=edge_list_target, edge_type=edge_types)
        if get_label:
            return target, graphs, raw_graphs, torch.Tensor(labels)
        else: 
            return target, graphs, raw_graphs
    
    
        
        
        # positive, partial, mutation, target, random 생성
    def generate_train_sample_for_embedder(self):
        k_hop_samples = {'radius': {'positive':[], 'partial':[], 'mutation':[], 'target':[], 'random':[]}, 'diameter':{'positive':[], 'partial':[], 'mutation':[], 'target':[],  'random':[]}}
        entire_graph, entire_subckt_dict = self.read_circuit(self.entire)
        entire_node_names  = entire_graph[1]
        entire_edge_list = entire_graph[4]
        entire_node_types = np.asarray(entire_graph[2])
        entire_edge_types = np.asarray(entire_graph[5])
        gnd_vsource_index = np.arange(len(entire_node_names))[entire_node_types<6].tolist()
        if self.target == 't01':
            entire_subckt_dict = get_t01(entire_subckt_dict) 
        elif self.target == 't02':
            entire_subckt_dict = get_t02(entire_subckt_dict, entire_node_names, entire_edge_list)
            
            
        # need to add random
        if self.target not in entire_subckt_dict.keys():
            #exit(-1)
            return
        #assert len(entire_subckt_dict[self.target]) > 0 , "no target in entire graph, function : Model_Trainer.during_development"
        if len(entire_subckt_dict[self.target]) <= 0:
            return
        center_list = [[],[]] # store center nodes that were already used as seed. To avoid selecting sample that was used in positive, partial, mutation, target when creating random sample.
        for target_idx in tqdm(range(len(entire_subckt_dict[self.target]))):
        #for target_idx in range(len(entire_subckt_dict[self.target])):
            target_nodes = entire_subckt_dict[self.target][target_idx][0] 
            target_edge_list = entire_subckt_dict[self.target][target_idx][1]
            # pdb.set_trace()
            center_list, sample = get_k_hop_samples(self.args.num_mutation+self.args.num_mutation_test, self.args.num_partial+self.args.num_partial_test, center_list, entire_edge_list, target_nodes, target_edge_list, entire_node_types, entire_edge_types, self.k_option, gnd_vsource_index)
            if self.args.radius:
                for k,v in sample['radius'].items():
                    k_hop_samples['radius'][k] = k_hop_samples['radius'][k] + v
            if self.args.diameter:
                for k,v in sample['diameter'].items():
                    k_hop_samples['diameter'][k] = k_hop_samples['diameter'][k] + v  
        # get random_sample.
        random_sample = get_k_hop_random(self.args.num_random+self.args.num_random_test, center_list, entire_edge_list, target_nodes, target_edge_list, entire_node_types, entire_edge_types, self.k_option, gnd_vsource_index) 
        if self.args.radius:
            for k,v in random_sample['radius'].items():
                k_hop_samples['radius'][k] = k_hop_samples['radius'][k] + v
        if self.args.diameter:
            for k,v in random_sample['diameter'].items():
                k_hop_samples['diameter'][k] = k_hop_samples['diameter'][k] + v  
        if self.args.radius:
            for sample_type in ['positive', 'partial', 'mutation', 'target', 'random']:
                file_name = self.proc_dir+'embedder_k_hop/radius/{}_in_{}_{}.txt'.format(self.target, self.entire, sample_type)
                write_text = ''
                for sample_id in range(len(k_hop_samples['radius'][sample_type])):
                    one_sample = k_hop_samples['radius'][sample_type][sample_id]
                    #rewrite this
                    write_text += ' '.join([str(int_to_text) for int_to_text in one_sample[0].tolist()]) + '\t' + \
                        ' '.join([str(int_to_text) for int_to_text in one_sample[1][0].tolist()]) + \
                            '\t' + ' '.join([str(int_to_text) for int_to_text in one_sample[1][1].tolist()]) + \
                                '\t' + ' '.join([str(int_to_text) for int_to_text in one_sample[2]]) + \
                                    '\t' + ' '.join([str(int_to_text) for int_to_text in one_sample[3].tolist()]) + \
                                        '\t' + ' '.join([str(int_to_text) for int_to_text in one_sample[4].tolist()]) + '\n'
                opened = open(file_name,'w')
                opened.write(write_text)
                opened.close() 
        if self.args.diameter:
            for sample_type in ['positive', 'partial', 'mutation', 'target', 'random']:
                file_name = self.proc_dir+'embedder_k_hop/diameter/{}_in_{}_{}.txt'.format(self.target, self.entire, sample_type)
                write_text = ''
                for sample_id in range(len(k_hop_samples['diameter'][sample_type])):
                    one_sample = k_hop_samples['diameter'][sample_type][sample_id]
                    write_text += ' '.join([str(int_to_text) for int_to_text in one_sample[0].tolist()]) + '\t' + \
                        ' '.join([str(int_to_text) for int_to_text in one_sample[1][0].tolist()]) + \
                            '\t' + ' '.join([str(int_to_text) for int_to_text in one_sample[1][1].tolist()]) + \
                                '\t' + ' '.join([str(int_to_text) for int_to_text in one_sample[2]]) + \
                                    '\t' + ' '.join([str(int_to_text) for int_to_text in one_sample[3].tolist()]) + \
                                        '\t' + ' '.join([str(int_to_text) for int_to_text in one_sample[4].tolist()]) + '\n'
                opened = open(file_name,'w')
                opened.write(write_text)
                opened.close()  
                
        return
    
    
    
    
    
    def read_circuit(self, ckt_name='None'):
            # description : function for just reading a file
            # return : graph and dictionary
            # graph : [boundary, node_names, node_types, node_params, edge_list, edge_types]
            # dictionary : single pyhton dictionary holding subckt position informations
            graph = []
            
            # file name to read
            dir_flag = 'dir' if self.args.gnd_vsource_directional else 'undir'
            redun_flag = 'redun' if self.args.gnd_vsource_allow_redundancy else 'noredun'
            graph_file_name = self.proc_dir +'graph_files/{}_{}_{}.txt'.format(ckt_name, dir_flag, redun_flag)
            pickle_file_name = self.proc_dir +'graph_files/{}_{}_{}_labels3.pkl'.format(ckt_name, dir_flag, redun_flag)
                
            # check whether file exist
            assert os.path.isfile(graph_file_name)
            assert os.path.isfile(pickle_file_name)
            
            # read graph file
            graph_opened = open(graph_file_name, 'r')
            graph_file = graph_opened.read().split('\n')[:-1]
            graph_opened.close()
            graph.append(graph_file[0].split(' ')) # boundary
            graph.append(graph_file[1].split(' ')) # node_names
            graph.append([int(i) for i in graph_file[2].split(' ')]) # node_types
            graph.append(graph_file[3].split(' ')) # node_params
            graph.append([[int(i) for i in graph_file[4].split(' ')], [int(i) for i in graph_file[5].split(' ')]]) # edge_list
            graph.append([int(i) for i in graph_file[6].split(' ')]) # edge_type

            # read pickle file for dictionary
            with open(pickle_file_name, 'rb') as pickle_file:
                subckt_dict = pickle.load(pickle_file) # 해당 target_circuit 안에 어떤 subcircuit이 있는지..
            
            return graph, subckt_dict
        
        
        
        
        
        
    def load_data_from_file(self, k_hop_file_path):
        data_list = []
        with open(k_hop_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split('\t')
                node_list = list(map(int, parts[0].split()))
                edge_list_1 = list(map(int, parts[1].split()))
                edge_list_2 = list(map(int, parts[2].split()))
                new_edge_list = torch.tensor([edge_list_1, edge_list_2])#, device=self.device) 
                center_nodes = torch.tensor(list(map(int, parts[3].split()))).tolist()
                node_types = torch.tensor(list(map(int, parts[4].split())))#, device=self.device) 
                edge_types = torch.tensor(list(map(int, parts[5].split())))#, device=self.device) 

                group_mapping = {0: 0, 2: 0, 3: 0, 4: 0,  # Group 0
                                    1: 1, 5: 1,              # Group 1
                                    6: 2, 7: 2,              # Group 2
                                    8: 3,                    # Group 3
                                    9: 4,                    # Group 4
                                    10: 5,                   # Group 5
                                    11: 6,                   # Group 6
                                    12: 7}                    # Group 7

                grouped_node_types = torch.tensor([group_mapping[node_type.item()] for node_type in node_types])
                x = torch.nn.functional.one_hot(grouped_node_types, num_classes=8).float()#.to(self.device)
                # try:
                new_center_nodes = torch.tensor([node_list.index(center_node) for center_node in center_nodes if center_node in node_list])
                # except:
                #     pdb.set_trace()
                # x = torch.nn.functional.one_hot(node_types, num_classes=14).float().to(self.device)
                data = Data(x=x, edge_index=new_edge_list, edge_type=edge_types, center_node=new_center_nodes)
                data_list.append(data)
        return data_list # center_list
    
    
    
    
    def load_splits(self):
        # load predefined splits
        entire_split = []
        for split_idx in range(self.args.num_repeat):
            split_file_name = 'split_{}.pkl'.format(str(split_idx))
            with open(self.split_dir+split_file_name, 'rb') as f:
                split = pickle.load(f)
                entire_split.append(split)
        return entire_split
    
    
    def generate_splits(self, target_and_k_hop):
        assert os.path.exists(self.split_dir)
        if self.args.radius:
            save_path_others = self.proc_dir+'temp/{}/target_and_k_hop_include_others.pkl'.format('radius')
        elif self.args.diameter:
            save_path_others = self.proc_dir+'temp/{}/target_and_k_hop_include_others.pkl'.format('diameter')
            
        if self.args.use_predefined_split:    
            splits_all_exists = True
            for i in range(self.args.num_repeat): # check whether all split files exist.
                splits_all_exists = splits_all_exists and os.path.isfile(self.split_dir+'split_{}.pkl'.format(str(i)))
            if splits_all_exists: # we have predefined split. Now lets load it.
                return self.load_splits() 
            
            
        # creating split process & saving splits
        opened_file = open(self.src+'embedder_train_option/train_targets.txt', 'r')
        self.target_ckt_list = opened_file.read().split('\n')[0].split(' ')
        if self.args.train_with_test:
            self.target_ckt_list.append('t01')
            self.target_ckt_list.append('t02')
        opened_file.close()
        
        if target_and_k_hop == None: 
            with open(save_path_others, 'rb') as f:
                target_and_k_hop = pickle.load(f)
            print("Loaded target_and_k_hop_include_others from saved file.")      
        
        entire_split=[]
        data_type = ["target", "positive", "partial", "mutation", "random", "others"]
        train_sample_numbers = [1, self.args.num_positive, self.args.num_partial, self.args.num_mutation, self.args.num_random, self.args.num_others]
        test_sample_numbers = [1, self.args.num_positive_test, self.args.num_partial_test, self.args.num_mutation_test, self.args.num_random_test, self.args.num_others_test]
        for repeat_idx in range(self.args.num_repeat):
            utils.set_seed(repeat_idx+1)
            split = {'train':{}, 'test':{}}  # hierarchy of split: [train,test] --> circuit_type --> data_type(positive,partial, etc..)
        # train sample
            for target_idx, target_ckt in enumerate(self.target_ckt_list):
                split['train'][target_ckt] = {}
                split['test'][target_ckt] = {}
                for type_idx, dt in enumerate(data_type):
                    if type_idx == 0: # target
                        split['train'][target_ckt][dt] = target_and_k_hop[target_ckt][type_idx]
                        split['test'][target_ckt][dt] = target_and_k_hop[target_ckt][type_idx]
                        continue
                        
                    if type_idx == 5: # others
                        samples_extracted = target_and_k_hop[target_ckt][type_idx][repeat_idx]
                    else: # positive ~ random
                        samples_extracted = target_and_k_hop[target_ckt][type_idx]
                        
                    random.shuffle(samples_extracted)
                    if type_idx in [2, 4] and len(samples_extracted) <= train_sample_numbers[type_idx]: #and len(samples_extracted)+len(target_and_k_hop[target_ckt][2]) >= train_sample_numbers[type_idx]+train_sample_numbers[2]:
                        split['train'][target_ckt][dt] = samples_extracted[:train_sample_numbers[type_idx]]
                        split['train'][target_ckt][dt].extend(target_and_k_hop[target_ckt][3][train_sample_numbers[3]:train_sample_numbers[3]+train_sample_numbers[type_idx]-len(samples_extracted)])
                    # if type_idx == 4 and len(samples_extracted) <= train_sample_numbers[type_idx] and len(samples_extracted)+len(target_and_k_hop[target_ckt][2]) < train_sample_numbers[type_idx]+train_sample_numbers[2]:
                    #     split['train'][target_ckt][dt].extend(target_and_k_hop[target_ckt][2][:train_sample_numbers[type_idx]])
                    #     split['train'][target_ckt][dt].extend(target_and_k_hop[target_ckt][2][:train_sample_numbers[type_idx]-len(samples_extracted)])
                    
                    elif type_idx == 1 and len(samples_extracted) < train_sample_numbers[type_idx] + test_sample_numbers[type_idx]:
                        split['train'][target_ckt][dt] = samples_extracted[:int(len(samples_extracted)*2/3)]

                    else:    
                        split['train'][target_ckt][dt] = samples_extracted[:train_sample_numbers[type_idx]]
                    


                    if test_sample_numbers[type_idx] == 0:
                        split['test'][target_ckt][dt] = []
                    elif type_idx == 1 and len(samples_extracted) < train_sample_numbers[type_idx] + test_sample_numbers[type_idx]:
                        split['test'][target_ckt][dt] = samples_extracted[-int(len(samples_extracted)*1/3):]
                    elif type_idx not in [1, 4] and len(samples_extracted) <= train_sample_numbers[type_idx]:
                        split['test'][target_ckt][dt] = []       
                    elif type_idx == 4 and len(samples_extracted) <= train_sample_numbers[type_idx]:
                        split['test'][target_ckt][dt] = samples_extracted[-test_sample_numbers[type_idx]:] 
                        split['test'][target_ckt][dt].extend(target_and_k_hop[target_ckt][3][-test_sample_numbers[type_idx]+len(samples_extracted):])                                       
                    elif type_idx not in [1, 4] and len(samples_extracted) <= train_sample_numbers[type_idx] + test_sample_numbers[type_idx]:
                        split['test'][target_ckt][dt] = samples_extracted[-len(samples_extracted)+train_sample_numbers[type_idx]:]
                    # elif type_idx == 4 and len(samples_extracted) <= train_sample_numbers[type_idx] + test_sample_numbers[type_idx]:
                        # 이 경우 없음
                    else: #elif len(samples_extracted) > train_sample_numbers[type_idx] + test_sample_numbers[type_idx]:
                        split['test'][target_ckt][dt] = samples_extracted[-test_sample_numbers[type_idx]:] # if test_sample_numbers[type_idx] > 0 and len(samples_extracted) > train_sample_numbers[type_idx] else []
                    # split['test'][target_ckt][dt] = samples_extracted[-test_sample_numbers[type_idx]:]
            entire_split.append(split)
            
        print("It's time to save split!")
        for split_idx in range(len(entire_split)):
            split_file_name = 'split_{}.pkl'.format(str(split_idx))
            with open(self.split_dir+split_file_name, 'wb') as f:
                pickle.dump(entire_split[split_idx], f)
            print("Saved to {}".format(split_file_name))
        print("Save finished")

        return entire_split
    
    
    
    def generate_k_hop_embedding_other_targets(self, target_and_k_hop):
        # function description : add negative_other_targets
        if self.args.radius:
            save_path_others = self.proc_dir+'temp/{}/target_and_k_hop_include_others.pkl'.format('radius')
            save_path = self.proc_dir+'temp/{}/target_and_k_hop.pkl'.format('radius')
        elif self.args.diameter:
            save_path_others = self.proc_dir+'temp/{}/target_and_k_hop_include_others.pkl'.format('diameter')
            save_path = self.proc_dir+'temp/{}/target_and_k_hop.pkl'.format('diameter')
            
        if os.path.exists(save_path_others):
            # file already exists. Proceed to generating splits
            with open(save_path_others, 'rb') as f:
                target_and_k_hop = pickle.load(f)
            print("Loaded target_and_k_hop_include_others from saved file.")
            return self.generate_splits(target_and_k_hop)
            
        if target_and_k_hop == None: # load target_and_k_hop (which does not include 'others' yet)
            with open(save_path, 'rb') as f:
                target_and_k_hop = pickle.load(f)
            print("Loaded target_and_k_hop from saved file.")


        opened_file = open(self.src+'embedder_train_option/train_targets.txt', 'r')
        self.target_ckt_list = opened_file.read().split('\n')[0].split(' ')
        if self.args.train_with_test:
            self.target_ckt_list.append('t01')
            self.target_ckt_list.append('t02')
        opened_file.close()
        
        utils.set_seed(0)
        num_required_minimum = int((self.args.num_others+self.args.num_others_test)/(len(self.target_ckt_list)-1)+1)*2#*self.args.num_repeat
        entire_positive = []
        for target_idx, target_ckt in enumerate(self.target_ckt_list):
            target_and_k_hop[target_ckt].append([])
            entire_positive.append(target_and_k_hop[target_ckt][1])
        
        for repeat_idx in range(self.args.num_repeat):
            utils.set_seed(repeat_idx+1)
            current_split_others = [[] for i in range(len(self.target_ckt_list))]
            for target_idx, target_ckt in enumerate(self.target_ckt_list):
                target_positive = entire_positive[target_idx]
                for shuffle_num_idx in range(len(self.target_ckt_list)):
                    if target_idx == shuffle_num_idx:
                        continue
                    random.shuffle(target_positive)
                    current_split_others[shuffle_num_idx].extend(target_positive[:num_required_minimum])
            for target_idx, target_ckt in enumerate(self.target_ckt_list):      
                target_and_k_hop[target_ckt][-1].append(current_split_others[target_idx])
            
        print("It's time to save target_and_k_hop!")
        with open(save_path_others, 'wb') as f:
            pickle.dump(target_and_k_hop, f)
        print("Saved target_and_k_hop_include_others to file.")
        
        return self.generate_splits(target_and_k_hop)
            
            
            
            
            
            
    
    def generate_k_hop_embedding(self): # 여기에서 dimension 고쳐야함
        # create embeddings
        
        # pdb.set_trace()
        # generate splits
        if self.args.use_predefined_split:    
            if os.path.exists(self.split_dir):
                splits_all_exists = True
                for i in range(self.args.num_repeat):
                    splits_all_exists = splits_all_exists and os.path.isfile(self.split_dir+'split_{}.pkl'.format(str(i)))
                if splits_all_exists:
                    return self.generate_splits(None)
            else:
                os.mkdir(self.split_dir)

        opened_file = open(self.src+'embedder_train_option/train_targets.txt', 'r')
        self.target_ckt_list = opened_file.read().split('\n')[0].split(' ')
        if self.args.train_with_test:
            self.target_ckt_list.append('t01')
            self.target_ckt_list.append('t02')
        opened_file.close()
        opened_file = open(self.src+'embedder_train_option/entire_circuits.txt', 'r')
        self.entire_ckt_list = opened_file.read().split('\n')[0].split(' ')
        opened_file.close()
        if self.args.radius:
            save_path_others = self.proc_dir+'temp/{}/target_and_k_hop_include_others.pkl'.format('radius')
            save_path = self.proc_dir+'temp/{}/target_and_k_hop.pkl'.format('radius')
            directory_path = self.proc_dir+'embedder_k_hop/{}/'.format('radius')
        elif self.args.diameter:
            save_path_others = self.proc_dir+'temp/{}/target_and_k_hop_include_others.pkl'.format('diameter')
            save_path = self.proc_dir+'temp/{}/target_and_k_hop.pkl'.format('diameter')
            directory_path = self.proc_dir+'embedder_k_hop/{}/'.format('diameter')
        else:
            raise NotImplementedError
        target_and_k_hop = {}
        
        # Check if the save file exists

        if os.path.exists(save_path_others):
            return self.generate_splits(None)
        
        if os.path.exists(save_path):
            return self.generate_k_hop_embedding_other_targets(None)
            with open(save_path, 'rb') as f:
                target_and_k_hop = pickle.load(f)
            print("Loaded target_and_k_hop from saved file.")
        else:
            data_type = ["positive", "partial", "mutation", "random"]

            for target_ckt in tqdm(self.target_ckt_list):
                if target_ckt not in target_and_k_hop.keys():
                    target_graph, _ = self.read_circuit(target_ckt)
                    target_node_types = torch.tensor(target_graph[2])#, device=self.device) 
                    target_edge_list = torch.tensor(target_graph[4])#, device=self.device) 
                    target_edge_type = torch.tensor(target_graph[5])#, device=self.device)

                    group_mapping = {0: 0, 2: 0, 3: 0, 4: 0,  # Group 0
                                    1: 1, 5: 1,              # Group 1
                                    6: 2, 7: 2,              # Group 2
                                    8: 3,                    # Group 3
                                    9: 4,                    # Group 4
                                    10: 5,                   # Group 5
                                    11: 6,                   # Group 6
                                    12: 7}                    # Group 7
                    grouped_target_node_types = torch.tensor([group_mapping[node_type.item()] for node_type in target_node_types])
                    x = torch.nn.functional.one_hot(grouped_target_node_types, num_classes=8).float()#.to(self.device)
                    # x = torch.nn.functional.one_hot(target_node_types, num_classes=14).float().to(self.device) 
                    data = [Data(x=x, edge_index=target_edge_list, edge_type=target_edge_type),[],[],[],[]]
                    target_and_k_hop[target_ckt] = data

            target_bar = tqdm(self.target_ckt_list)
            entire_bar = tqdm(self.entire_ckt_list)
            for entire_ckt in entire_bar:
                entire_bar.set_description("In entire : {}".format(entire_ckt))
                for target_ckt in target_bar:
                    target_bar.set_description('target : {}'.format(target_ckt))
                    for dt_idx, dt in enumerate(data_type):
                        file_name = '_'.join([target_ckt, "in", entire_ckt, "{}.txt".format(dt)])
                        if os.path.isfile(directory_path+file_name):
                            print(file_name)
                            graphs_list = self.load_data_from_file(directory_path+file_name)
                            target_and_k_hop[target_ckt][dt_idx+1].extend(graphs_list)
             
            # Save the target_and_k_hop data to a file for future uses
            print("It's time to save target_and_k_hop!")
            with open(save_path, 'wb') as f:
                pickle.dump(target_and_k_hop, f)
            print("Saved target_and_k_hop to file.")
            
        return self.generate_k_hop_embedding_other_targets(target_and_k_hop)
        
    
    
    
    
    
    
    
    
    
def target_detect_k(nodes, entire_edge_list, edge_list, option='None', gnd_vsource_index= []):
    '''
    decide 'k' based on the option (one of diameter or radius)
    return k, center, target_distance_vector
    target_distance_vector : vector holding for minimum distance to every nodes in the circuit
    '''
    inf_nodes = nodes
    target_distance_vector = torch.full((len(nodes), len(nodes)), len(nodes))
    gnd_vsource_mask = []
    for i in range(len(nodes)):
        if nodes[i] in gnd_vsource_index:
            gnd_vsource_mask.append(i)
        target_distance_vector[i,i] = 0
    normal_nodes = list(set([i for i in range(len(nodes))]) - set(gnd_vsource_mask))

    if len(entire_edge_list) == len(edge_list):
        selected_edge_list = torch.LongTensor(edge_list)
    else: 
        selected_edge_list = torch.LongTensor(entire_edge_list)[:, edge_list]
        for i in range(len(nodes)):
            selected_edge_list[selected_edge_list == nodes[i]] = -i-1
        selected_edge_list = -selected_edge_list-1
        
    upd = update_min_dist()
    while(True):
        got_upd = upd(target_distance_vector, selected_edge_list)
        target_distance_vector_temp = torch.minimum(got_upd, target_distance_vector)
        if torch.all(target_distance_vector_temp  == target_distance_vector):
            break 
        target_distance_vector = target_distance_vector_temp
    target_distance_vector[target_distance_vector == len(nodes)] = -1
    diameter = target_distance_vector.max().item()+1 # reason for adding 1 : if diameter path contains two vsource/gnd, then the diameter would have 1 less value.
    radius = max(target_distance_vector[normal_nodes,:].max(1).values.min().item()+1, (diameter+1)//2) #(diameter+1)//2
    #max_dist_of_each_node = target_distance_vector.max(-1).values
    #center = ((max_dist_of_each_node == radius).nonzero(as_tuple=True)[0])
    #k = diameter if option=='diameter' else radius
    return diameter, radius
    #return k, center, target_distance_vector






def get_k_hop_samples(num_mutation, num_partial, center_list, edge_list, target_nodes, target_edge_list, entire_node_types, entire_edge_types, option='None', gnd_vsource_index=[]):
    '''
    return positive, negative_partial, negative_mutation, negative_target
    positive : literally
    negative_partial : k_hop subgraph that contains only part of the target
    negative_mutation : change of connection of target circuit
    negative_target : other targets as negative sample  --> decided not to create those samples here
    
    gnd_vsource_index : node index of entire circuit that is gnd or vsource. 
                        It is important in implementation since those nodes will never be reachable via message passing from other nodes.
                        when performing target_detect_k, this must be considered.
    '''
    positive = []
    negative_partial = []
    negative_mutation = []
    negative_target = []
    
    #diameter, center_idx, target_distance_vector = target_detect_k(target_nodes, edge_list, target_edge_list, 'diameter', gnd_vsource_index)
    diameter, radius = target_detect_k(target_nodes, edge_list, target_edge_list, 'diameter', gnd_vsource_index)
    #centers = torch.LongTensor(target_nodes)[center_idx].tolist()
    target_circuit_size = len(target_nodes)
    k_hop_sample={}
    ### [node_idx_map, new_edge_list, p_center, node_types, edge_types]
    if option in ['radius','both']:
        radius_center_list = []
        radius_subgraph_str = []
        radius_dict={}
        positive_radius = []
        negative_partial_radius = []
        negative_mutation_radius = []
        negative_target_radius = []
        pos_idx_list = []
        neg_idx_list = []
        # when finding seed, we exclude vsource or gnd from target_nodes before using them in k_hop_subgraph function.
        # If not, seeds will include most of the nodes in entire circuit as many components are reachable from vsource or gnd.
        target_nodes_non_vsource_ground = np.asarray(target_nodes)[entire_node_types[target_nodes]>=6].tolist()
        possible_center_seeds, reduced_edge_index, mapping, edge_mask = k_hop_subgraph(target_nodes_non_vsource_ground, radius, torch.LongTensor(edge_list).flip([0]), relabel_nodes=False, flow="source_to_target") # flow "source_to_target" finds source nodes that can reach target node
        for p_seed_idx, p_center in enumerate(possible_center_seeds):
            radius_center_list.append(p_center.item())
            node_idx_map, new_edge_index, mapping, edge_mask = k_hop_subgraph(torch.LongTensor([p_center]), radius, torch.LongTensor(edge_list).flip([0]), relabel_nodes=True, flow="target_to_source")
            node_idx_map_to_text = ' '.join([str(int_to_str) for int_to_str in node_idx_map.tolist()])
            edge_idx_map_to_text = ' '.join([str(int_to_str) for int_to_str in new_edge_index.flatten().tolist()])
            subgraph_text = node_idx_map_to_text + '\t' + edge_idx_map_to_text
            if subgraph_text in radius_subgraph_str: # to not allow subgraph duplicate
                duplicate_idx = radius_subgraph_str.index(subgraph_text)
                if duplicate_idx in pos_idx_list:
                    pos_idx = pos_idx_list.index(duplicate_idx)
                    positive_radius[pos_idx][2].append(p_center.item())
                    radius_subgraph_str.append('pos_Duplicate')
                elif duplicate_idx in neg_idx_list:
                    neg_idx = neg_idx_list.index(duplicate_idx)
                    negative_partial_radius[neg_idx][2].append(p_center.item())
                    radius_subgraph_str.append('partial_Duplicate')
                else: # when node_idx_map.size(0) < target_circuit_size
                    continue
                continue
            if node_idx_map.size(0) < target_circuit_size:
                radius_subgraph_str.append('Too small')
                continue
            radius_subgraph_str.append(subgraph_text)
            new_edge_index = new_edge_index.flip([0])
            if set(target_nodes).issubset(set(node_idx_map.tolist())): #positive sample
                positive_radius.append([node_idx_map, new_edge_index, [p_center.item()], torch.LongTensor(entire_node_types[node_idx_map]),torch.LongTensor(entire_edge_types)[edge_mask]])
                pos_idx_list.append(p_seed_idx)
            else: # partial
                intersections = list(set(target_nodes).intersection(set(node_idx_map.tolist())))
                if len(intersections) <= len(target_nodes)*0.5:
                    negative_partial_radius.append([node_idx_map, new_edge_index, [p_center.item()], torch.LongTensor(entire_node_types[node_idx_map]),torch.LongTensor(entire_edge_types)[edge_mask]])
                    neg_idx_list.append(p_seed_idx)

        #negative_mutation
        count_pmos_mutations = 0
        count_nmos_mutations = 0
        for p_sample in positive_radius:
            # pmos : node type 8, nmos : node type 9
            # pmos_drain : edge type 0(in), 14(out),     pmos_gate : edge_type 1,15,     pmos_source 2, 16      , pmos_base 3, 17
            # nmos_drain : edge type 4(in), 18(out),     pmos_gate : edge_type 5,19,     pmos_source 6, 20      , pmos_base 7, 21
            node_idx_map = p_sample[0]
            new_edge_index = p_sample[1]
            node_types = p_sample[3]
            edge_types = p_sample[4]
            pmos_nodes = (node_types == 8).nonzero(as_tuple=True)[0] 
            nmos_nodes = (node_types == 9).nonzero(as_tuple=True)[0]
            for node_idx in pmos_nodes:
                if count_pmos_mutations >= num_mutation:
                    continue
                # one pmos-nmos type change
                if node_idx_map[node_idx] not in target_nodes: # the pmos must be one of target_nodes
                    continue
                negative_mutation_radius.append(copy.deepcopy(p_sample))
                negative_mutation_radius[-1][3][node_idx] = 9 # change node_types
                related_edges =  (new_edge_index == node_idx).nonzero(as_tuple = True)[1]
                negative_mutation_radius[-1][4][related_edges] += 4 # change edge_types
                # one drain-source change
                negative_mutation_radius.append(copy.deepcopy(p_sample))
                related_edges_types = negative_mutation_radius[-1][4][related_edges]
                related_edges = related_edges[(related_edges_types%14%2 == 0).nonzero(as_tuple = True)[0]] #select only drain and source
                negative_mutation_radius[-1][4][related_edges] += 2 * (1 - negative_mutation_radius[-1][4][related_edges]%14)  # change edge_types
                count_pmos_mutations += 2
                
            for node_idx in nmos_nodes:
                if count_nmos_mutations >= num_mutation:
                    continue
                if node_idx_map[node_idx] not in target_nodes: # the nmos must be one of target_nodes
                    continue
                negative_mutation_radius.append(copy.deepcopy(p_sample))
                negative_mutation_radius[-1][3][node_idx] = 8 # change node_types
                related_edges =  (new_edge_index == node_idx).nonzero(as_tuple = True)[1]
                negative_mutation_radius[-1][4][related_edges] -= 4 # change edge_types
                # one drain-source change
                negative_mutation_radius.append(copy.deepcopy(p_sample))
                related_edges_types = negative_mutation_radius[-1][4][related_edges]
                related_edges = related_edges[((related_edges_types-4)%14%2 == 0).nonzero(as_tuple = True)[0]] #select only drain and source
                negative_mutation_radius[-1][4][related_edges] += 2 * (5 - negative_mutation_radius[-1][4][related_edges]%14)  # change edge_types
                count_nmos_mutations += 2
                
        
        
        radius_dict['positive'] = positive_radius
        radius_dict['partial'] = negative_partial_radius[:int(num_partial*1.5)]
        radius_dict['mutation'] = negative_mutation_radius
        radius_dict['target'] = negative_target_radius
        k_hop_sample['radius'] = radius_dict
        
    if option in ['diameter','both']:
        diameter_center_list = []
        diameter_subgraph_str = []
        diameter_dict = {}
        positive_diameter = []
        negative_partial_diameter = []
        negative_mutation_diameter = []
        negative_target_diameter = []
        pos_idx_list = []
        neg_idx_list = []
        target_nodes_non_vsource_ground = np.asarray(target_nodes)[entire_node_types[target_nodes]>=6].tolist()
        possible_center_seeds, reduced_edge_index, mapping, edge_mask = k_hop_subgraph(target_nodes_non_vsource_ground, diameter, torch.LongTensor(edge_list).flip([0]), relabel_nodes=False, flow="source_to_target") # flow "source_to_target" finds source nodes that can reach target node
        # p_center: original index
        # new_edge_index : edge_index with new index
        # node_idx_map : new_index to original_index map for index of nodes.
        for p_seed_idx, p_center in enumerate(possible_center_seeds):
            diameter_center_list.append(p_center.item())
            node_idx_map, new_edge_index, mapping, edge_mask = k_hop_subgraph(torch.LongTensor([p_center]), diameter, torch.LongTensor(edge_list).flip([0]), relabel_nodes=True, flow="target_to_source")
            node_idx_map_to_text = ' '.join([str(int_to_str) for int_to_str in node_idx_map.tolist()])
            edge_idx_map_to_text = ' '.join([str(int_to_str) for int_to_str in new_edge_index.flatten().tolist()])
            subgraph_text = node_idx_map_to_text + '\t' + edge_idx_map_to_text
            if subgraph_text in diameter_subgraph_str: # to not allow subgraph duplicate
                duplicate_idx = diameter_subgraph_str.index(subgraph_text)
                if duplicate_idx in pos_idx_list:
                    pos_idx = pos_idx_list.index(duplicate_idx)
                    positive_diameter[pos_idx][2].append(p_center.item())
                    diameter_subgraph_str.append('pos_Duplicate')
                else: 
                    neg_idx = neg_idx_list.index(duplicate_idx)
                    negative_partial_diameter[neg_idx][2].append(p_center.item())
                    diameter_subgraph_str.append('partial_Duplicate')
                continue
            if node_idx_map.size(0) < target_circuit_size:
                diameter_subgraph_str.append('Too small')
                continue
            diameter_subgraph_str.append(subgraph_text)
            if set(target_nodes).issubset(set(node_idx_map.tolist())): #positive sample
                positive_diameter.append([node_idx_map, new_edge_index.flip([0]), [p_center.item()], torch.LongTensor(entire_node_types[node_idx_map]),torch.LongTensor(entire_edge_types)[edge_mask]])
                pos_idx_list.append(p_seed_idx)
            else: 
                negative_partial_diameter.append([node_idx_map, new_edge_index.flip([0]), [p_center.item()], torch.LongTensor(entire_node_types[node_idx_map]),torch.LongTensor(entire_edge_types)[edge_mask]])
                neg_idx_list.append(p_seed_idx)
                
        #negative_mutation
        for p_sample in positive_diameter:
            # pmos : node type 8, nmos : node type 9
            # pmos_drain : edge type 0(in), 14(out),     pmos_gate : edge_type 1,15,     pmos_source 2, 16      , pmos_base 3, 17
            # nmos_drain : edge type 4(in), 18(out),     pmos_gate : edge_type 5,19,     pmos_source 6, 20      , pmos_base 7, 21
            node_idx_map = p_sample[0]
            new_edge_index = p_sample[1]
            p_center = p_sample[2]
            node_types = p_sample[3]
            edge_types = p_sample[4]
            pmos_nodes = (node_types == 8).nonzero(as_tuple=True)[0] 
            nmos_nodes = (node_types == 9).nonzero(as_tuple=True)[0]
            for node_idx in pmos_nodes:
                if node_idx_map[node_idx] not in target_nodes: # the pmos must be one of target_nodes
                    continue
                negative_mutation_diameter.append(copy.deepcopy(p_sample))
                negative_mutation_diameter[-1][3][node_idx] = 9 # change node_types
                related_edges =  (new_edge_index == node_idx).nonzero(as_tuple = True)[1]
                negative_mutation_diameter[-1][4][related_edges] += 4 # change edge_types
                # one drain-source change
                negative_mutation_diameter.append(copy.deepcopy(p_sample))
                related_edges_types = negative_mutation_diameter[-1][4][related_edges]
                related_edges = related_edges[(related_edges_types%14%2 == 0).nonzero(as_tuple = True)[0]] #select only drain and source
                negative_mutation_diameter[-1][4][related_edges] += 2 * (1 - negative_mutation_diameter[-1][4][related_edges]%14)  # change edge_types
            for node_idx in nmos_nodes:
                if node_idx_map[node_idx] not in target_nodes: # the nmos must be one of target_nodes
                    continue
                negative_mutation_diameter.append(copy.deepcopy(p_sample))
                negative_mutation_diameter[-1][3][node_idx] = 8 # change node_types
                related_edges =  (new_edge_index == node_idx).nonzero(as_tuple = True)[1]
                negative_mutation_diameter[-1][4][related_edges] -= 4 # change edge_types
                # one drain-source change
                negative_mutation_diameter.append(copy.deepcopy(p_sample))
                related_edges_types = negative_mutation_diameter[-1][4][related_edges]
                related_edges = related_edges[((related_edges_types-4)%14%2 == 0).nonzero(as_tuple = True)[0]] #select only drain and source
                negative_mutation_diameter[-1][4][related_edges] += 2 * (5 - negative_mutation_diameter[-1][4][related_edges]%14)  # change edge_types
        
        diameter_dict['positive'] = positive_diameter
        diameter_dict['partial'] = negative_partial_diameter
        diameter_dict['mutation'] = negative_mutation_diameter
        diameter_dict['target'] = negative_target_diameter
        k_hop_sample['diameter'] = diameter_dict

    if option in ['radius', 'both']:
        center_list[0] += radius_center_list
    if option in ['diameter','both']:
        center_list[1] += diameter_center_list
    center_list = [list(sorted(set(center_list[0]))), list(sorted(set(center_list[1])))]

    return center_list, k_hop_sample






def get_k_hop_random(num_random, center_list, edge_list, target_nodes, target_edge_list, entire_node_types, entire_edge_types, option='None', gnd_vsource_index=[]):
    '''
    return positive, negative_partial, negative_mutation, negative_target
    positive : literally
    negative_partial : k_hop subgraph that contains only part of the target
    negative_mutation : change of connection of target circuit
    negative_target : other targets as negative sample  --> decided not to create those samples here
    
    gnd_vsource_index : node index of entire circuit that is gnd or vsource. 
                        It is important in implementation since those nodes will never be reachable via message passing from other nodes.
                        when performing target_detect_k, this must be considered.
    '''
    
    #diameter, center_idx, target_distance_vector = target_detect_k(target_nodes, edge_list, target_edge_list, 'diameter', gnd_vsource_index)
    diameter, radius = target_detect_k(target_nodes, edge_list, target_edge_list, 'diameter', gnd_vsource_index)
    #centers = torch.LongTensor(target_nodes)[center_idx].tolist()
    target_circuit_size = len(target_nodes)
    
    k_hop_sample={}
    ### [node_idx_map, new_edge_list, p_center, node_types, edge_types]
    possible_seeds = np.arange(len(entire_node_types))
    possible_seeds = possible_seeds[entire_node_types >=6]
    
    
    radius_subgraph_str = []
    diameter_subgraph_str = []
    radius_dict = {}
    diameter_dict = {}
    
    if option in ['radius','both']:
        radius_possible_seeds = list((set(possible_seeds).difference(set(center_list[0]))))[:int(num_random * 1.5)] # random center node는 partial, mutation 등에서 사용되는 않은 노드들만 선택
        random_idx_list = []
        negative_random_radius = []
        for p_seed_idx, p_center in tqdm(enumerate(radius_possible_seeds)):
            node_idx_map, new_edge_index, _, edge_mask = k_hop_subgraph(torch.LongTensor([p_center]), radius, torch.LongTensor(edge_list).flip([0]), relabel_nodes=True, flow="target_to_source")
            node_idx_map_to_text = ' '.join([str(int_to_str) for int_to_str in node_idx_map.tolist()])
            edge_idx_map_to_text = ' '.join([str(int_to_str) for int_to_str in new_edge_index.flatten().tolist()])
            subgraph_text = node_idx_map_to_text + '\t' + edge_idx_map_to_text
            if subgraph_text in radius_subgraph_str: # to not allow subgraph duplicate
                duplicate_idx = radius_subgraph_str.index(subgraph_text)
                if duplicate_idx in random_idx_list:
                    rand_idx = random_idx_list.index(duplicate_idx)
                    negative_random_radius[rand_idx][2].append(p_center.item())
                    radius_subgraph_str.append('random_Duplicate')
                else: # when node_idx_map.size(0) < target_circuit_size
                    raise NotImplementedError
                continue
            if node_idx_map.size(0) < target_circuit_size:
                radius_subgraph_str.append('Too small')
                continue
            radius_subgraph_str.append(subgraph_text)
            new_edge_index = new_edge_index.flip([0])
            if not set(target_nodes).issubset(set(node_idx_map.tolist())): 
                negative_random_radius.append([node_idx_map, new_edge_index, [p_center.item()], torch.LongTensor(entire_node_types[node_idx_map]),torch.LongTensor(entire_edge_types)[edge_mask]])
                random_idx_list.append(p_seed_idx)
            else: 
                raise NotImplementedError
        radius_dict['random'] = negative_random_radius
        k_hop_sample['radius'] = radius_dict
        
        
    if option in ['diameter','both']:
        diameter_possible_seeds = list((set(possible_seeds).difference(set(center_list[1]))))
        random_idx_list = []
        negative_random_diameter = []
        for p_seed_idx, p_center in tqdm(enumerate(diameter_possible_seeds)):
            node_idx_map, new_edge_index, _, edge_mask = k_hop_subgraph(torch.LongTensor([p_center]), diameter, torch.LongTensor(edge_list).flip([0]), relabel_nodes=True, flow="target_to_source")
            node_idx_map_to_text = ' '.join([str(int_to_str) for int_to_str in node_idx_map.tolist()])
            edge_idx_map_to_text = ' '.join([str(int_to_str) for int_to_str in new_edge_index.flatten().tolist()])
            subgraph_text = node_idx_map_to_text + '\t' + edge_idx_map_to_text
            if subgraph_text in diameter_subgraph_str: # to not allow subgraph duplicate
                duplicate_idx = diameter_subgraph_str.index(subgraph_text)
                if duplicate_idx in random_idx_list:
                    rand_idx = random_idx_list.index(duplicate_idx)
                    negative_random_diameter[rand_idx][2].append(p_center.item())
                    diameter_subgraph_str.append('random_Duplicate')
                else: # when node_idx_map.size(0) < target_circuit_size
                    raise NotImplementedError
                continue
            if node_idx_map.size(0) < target_circuit_size:
                diameter_subgraph_str.append('Too small')
                continue
            diameter_subgraph_str.append(subgraph_text)
            new_edge_index = new_edge_index.flip([0])
            if not set(target_nodes).issubset(set(node_idx_map.tolist())): 
                negative_random_diameter.append([node_idx_map, new_edge_index, [p_center.item()], torch.LongTensor(entire_node_types[node_idx_map]),torch.LongTensor(entire_edge_types)[edge_mask]])
                random_idx_list.append(p_seed_idx)
            else: 
                raise NotImplementedError
        diameter_dict['random'] = negative_random_diameter
        k_hop_sample['diameter'] = diameter_dict
        
    return k_hop_sample


