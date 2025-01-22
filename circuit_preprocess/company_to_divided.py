import os
import sys
import pdb
import argparse
import tqdm
import numpy as np
import preprocess_utils
import pickle
ROOT_DIR = os.path.dirname(os.getcwd())



class circuit_divider():
    def __init__(self, args):
        self.args = args
        self.dataset = self.args.dataset
        self.raw_dir = ROOT_DIR + '/raw_circuit/{}/raw/'.format(self.dataset)
        self.proc_dir = ROOT_DIR + '/raw_circuit/{}/'.format(self.dataset)
        
    
    def divide_ckt(self):
        origin_file = self.raw_dir+'{}.cir'.format(self.args.file_name)
        origin_opened = open(origin_file, 'r')
        origin_read = origin_opened.read()
        origin_opened.close()
        cut_by_subckt = origin_read.split('.SUBCKT')[1:]
        cut_by_ends = [ckt.split('.ENDS')[0] for ckt in cut_by_subckt]
        circuit_names = [ckt_names.split(' ')[1] for ckt_names in cut_by_ends]
        for i, circuit_name in enumerate(circuit_names):
            new_file = open(self.proc_dir+'{}.cir'.format(circuit_name), 'w')
            new_file.write('.SUBCKT'+cut_by_ends[i]+'.ENDS\n')
            new_file.close()
        new_file = open(self.raw_dir+'circuit_sequence.txt', 'w')
        new_file.write(' '.join(circuit_names))
        new_file.close()
        
    
def parse_args():
    # Parses the node2vec arguments.
    parser = argparse.ArgumentParser(description="circuit parser")
    parser.add_argument('--dataset', default='company', choices=['company'], help='select dataset : company or GANA') 
    parser.add_argument('--file_name', type=str, default = 'entire', help='file name that needs to be divided into separate subckt netlist files') 
    args = parser.parse_args()
    return args
    
if __name__ == '__main__' : 
    
    args = parse_args()
    cd = circuit_divider(args)
    cd.divide_ckt()
    