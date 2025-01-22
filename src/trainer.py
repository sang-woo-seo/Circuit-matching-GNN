import torch 
import numpy as np 
import pdb
import utils
import pickle
import os
import data_loader
from tqdm import tqdm
import pdb
import embedder
import sys

class Model_Trainer:
    def __init__(self, args):
        self.args = args
        self.target = args.target_circuit
        self.entire = args.entire_circuit
        self.root = args.root
        self.dataset = args.dataset
        self.proc_dir = self.root + '/processed_circuit/{}/'.format(self.dataset)
        self.src = self.root + '/src/'
        if args.diameter:
            self.k_option = 'diameter'
        if args.radius:
            self.k_option = 'radius'
        if args.diameter and args.radius:
            self.k_option = 'both'
            
        # preprocessing data when needed
        self.data_manager = data_loader.Data_Manager(args)
        
        
        # label generation by vf3py
        if self.args.label_vf3:
            self.data_manager.generate_vf3_labels_with_option_file()
            exit()
        
        # positive, negative, partial
        if self.args.preprocess_data:
            self.data_manager.generate_sample_with_option_file() # txt 파일내 회로마다 generate_train_sample_for_embedder 동작
            exit()
        if self.args.manual_preprocess_data:
            self.data_manager.generate_train_sample_for_embedder() # positive, partial, mutation, target, random 형성
            exit()

        

        self.embedder = embedder.GNN_embedder(args)
        if self.args.train_embedder: #or self.args.feature_matching:
            self.data = self.data_manager.generate_k_hop_embedding()  # target circuit 내에서 모든 k-hop embedding 뽑음 # txt파일 사용

    
    def load_embedder(self):
        self.embedder.load_embedder()
        return self.embedder
    
    
    def train_embedder(self):
        self.embedder.train_embedder(self.data)
        return self.embedder
    
    def feature_matching(self):
        self.embedder.feature_matching(self.data)
        return self.embedder

    
    def test_embedder_research(self, perform_analysis = False): # manual
        target, graphs, raw_graphs, labels = self.data_manager.get_all_k_hop(get_label=True)
        self.entire_ckt = self.args.entire_circuit
        entire_graph, entire_subckt_dict = self.data_manager.read_circuit(self.entire_ckt)
        self.embedder.test_embedder_research(target, graphs, labels, entire_graph, entire_subckt_dict, raw_graphs, perform_analysis)
        return
    

    # txt 파일 구현할 때에는 test_embedder_research를 iteration

    
    def get_bb(self):
        target, graphs, raw_graphs = self.data_manager.get_all_k_hop(get_label=False)
        return self.embedder.get_bb(target, graphs, raw_graphs)
    
    
    def test_bb(self):
        target_graph, entire_graph, radius = self.data_manager.get_graph_data()
        num_labels = self.data_manager.label_count()
        self.embedder.test_bb(target_graph, entire_graph, radius, num_labels)

        # target, graphs, raw_graphs = self.data_manager.get_all_k_hop(get_label=False)
        # candidates, raw_candidates = self.embedder.get_bb(target, graphs, raw_graphs) # 위 함수 아님. embedder.py에 존재
        # self.entire_ckt = self.args.entire_circuit
        # entire_graph, entire_subckt_dict = self.data_manager.read_circuit(self.entire_ckt)
        # self.embedder.feature_matching_bb(target, candidates, raw_candidates, entire_graph, entire_subckt_dict )
        return
    


    def detect_within_circuit(self):
        raise NotImplementedError
    
    def detect_between_circuits(self):
        raise NotImplementedError
    
    