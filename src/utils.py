import argparse
import pdb
import torch
from torch_geometric.nn.conv import MessagePassing
import numpy as np
from torch_geometric.utils import k_hop_subgraph
import copy
from tqdm import tqdm
import random
import os

def parse_args():
    # Parses the node2vec arguments.
    parser = argparse.ArgumentParser(description="circuit parser")
    parser.add_argument('--dataset', default='company', choices=['company'], help='select dataset : company or GANA') 
    parser.add_argument('--entire_circuit', type=str, default='main_sense', help='The largest possible circuit that you want to investigate') 
    parser.add_argument('--target_circuit', type=str, default='inv', help='target circuit') 
    parser.add_argument('--train_with_test', action='store_true', help='add test circuits to training set') 
    parser.add_argument('--tune_with_test', action='store_true', help='after training, tune with test circuits') 
    parser.add_argument('--diameter', action='store_true', help='use diameter as k-hop decision') 
    parser.add_argument('--radius', action='store_true', help='use radius as k-hop decision') 
    parser.add_argument('--device', type=int, default=6, help='gpu number') 
    parser.add_argument('--train_verbose', action = 'store_true', help='allow program to print temporary outputs during training') 
    parser.add_argument('--test_verbose', action = 'store_true', help='allow program to print temporary outputs during testing') 
    parser.add_argument('--verbose', action = 'store_true', help='allow program to print temporary outputs during training and testing') 
    parser.add_argument("--gnd_vsource_directional", action = "store_true", help = "When set as True, the edges that are connected to ground or voltage source become directional. (gnd or source --> net) is added but not (net --> gnd or source)")
    parser.add_argument("--gnd_vsource_allow_redundancy", action = "store_true", help = "gnd or voltage source as one node, or allow redundancy")
    parser.add_argument("--preprocess_data", action='store_true', help='preprocess data')
    parser.add_argument("--manual_preprocess_data", action='store_true', help='preprocess data')
    parser.add_argument("--train_embedder", action='store_true', help='train the embedder as well as classifier')
    parser.add_argument("--load_parameter_file_name", type=str, default = 'best',help='checkpoint file name to be loaded. format : epoch_timestamp')
    parser.add_argument("--save_best", action='store_true', help='when training is complete, save it to checkpoint/saved_parameter directory to deploy our model.')
    parser.add_argument("--epoch", type=int, default=1000, help='number of epochs used for training')
    parser.add_argument("--batch_size", type=int, default=32, help='batch_size')
        
    parser.add_argument("--num_positive", type=int, default=200, help='number of positive samples per target circuit')
    parser.add_argument("--num_partial",type=int, default=100, help='number of partial samples per target circuit')
    parser.add_argument("--num_mutation", type=int, default=50, help='number of mutation samples per target circuit')
    parser.add_argument("--num_others", type=int, default=50, help='number of other samples per target circuit')
    parser.add_argument("--num_random",type=int, default=300, help='number of random samples per target circuit')
    parser.add_argument("--num_positive_test", type=int, default=100, help='number of positive samples per target circuit')
    parser.add_argument("--num_partial_test",type=int, default=0, help='number of partial samples per target circuit') # 20
    parser.add_argument("--num_mutation_test", type=int, default=0, help='number of mutation samples per target circuit') # 10
    parser.add_argument("--num_others_test", type=int, default=0, help='number of other samples per target circuit') # 10
    parser.add_argument("--num_random_test",type=int, default=100, help='number of random samples per target circuit') # 60
    
    parser.add_argument("--num_repeat", type=int, default=10, help='number of repeat')
    parser.add_argument("--use_predefined_split", action ='store_true', help='use predefined split, rather than creating split.')
    parser.add_argument("--use_only_first_split", action ='store_true', help='use first split only (repeat 1)')
    parser.add_argument("--decision_threshold", type=float, default=0.5, help='between 0 and 1. The threshold for binary classification')
    parser.add_argument("--bb_merge_threshold", type=float, default=0.2, help='between 0 and 1. The threshold for merging positive and partial as candidate bounding box')
    parser.add_argument("--pt_name", type=str, default='', help='name to be added to checkpoint')
    
    parser.add_argument("--label_vf3", action ='store_true', help='use vf3 to get labels.')
    
    args = parser.parse_args()
    if args.verbose:
        args.train_verbose = True
        args.test_verbose = True
    assert args.diameter or args.radius, "one of diameter or radius must be set true"
    #assert not (args.diameter and args.radius), "at most one of diameter or radius must be set true"
    return args


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def extract_k_hop_from_boundary(entire_edge_list, k, option='None'):
    '''
    purpose : extract nodes that are within k-hop from boundary. used when combining different subckts.
    will be implemented later
    '''
    raise NotImplementedError


def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union
 