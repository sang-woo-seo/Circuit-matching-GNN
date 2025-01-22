import numpy
import os
import sys

class circuit_graph():
    def __init__(self, name):
        self.circuit_name = name
        self.subckt_hierarchy = {} # each key will store direct subckts as values
        self.nodes = None # will be list of nodes
        self.edges = None # will be edge_index
        self.edge_types = None # will be list of types of edges
        self.node_types = None # will be list of types of nodes
        
    
    def node_name_map(self):
        pass 
    
    def node_idx_map(self):
        pass 
    
    def idx_name_map(self):
        pass
    
    def node_type(self):
        pass
    
    def num_nodes(self): 
        pass
    
    def edge_index(self):
        pass 
    
    def edge_type(self):
        pass 
    
    def merge_subcircuit(self, subcircuit):
        
        raise NotImplementedError
        return 