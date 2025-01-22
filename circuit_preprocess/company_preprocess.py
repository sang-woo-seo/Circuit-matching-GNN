import os
import sys
import pdb
import argparse
import tqdm
import numpy as np
import preprocess_utils
import pickle
import networkx as nx 
from pyvis.network import Network

ROOT_DIR = os.path.dirname(os.getcwd())


def netlist_basic_preprocessing(netlist, circuit_name, redundancy_allow_flag): 
    # remove comments and line management (ex : '+' needs to merge lines of netlist)
    # manage nets(wires)
    total_len = len(netlist)
    prev_arranged_netlist = []
    arranged_netlist = []
    first_line = netlist[0].split(' ')
    assert first_line[0] == '.SUBCKT'
    assert first_line[1] == circuit_name
    first_sentence = netlist[0]
    for i, netline in enumerate(netlist):
        if i == 0 :
            continue 
        if netline[0] == '+':
            first_sentence += ' '+netline[1:]
        else: 
            break
    first_sentence_idx = i
    boundaries = first_sentence.split(' ')[2:] # net(wires) that can be connected to other circuit. The boundary net(wire) of the circuit
    net_stat = []
    node_list = []
    

    for i, netline in enumerate(netlist):
        if i < first_sentence_idx:
            continue
        if netline == '': # empty line (ex :  rl_sel.cir)
            continue
        netline = netline.split(' ')

        for c_i, component in enumerate(netline):
            if component == '':
                continue
            if component[0] in ['*', '$']: # remove comment
                netline = netline[:c_i]
                break
        if len(netline) > 0:
            if netline[0][0] == "+": # continued to previous line
                assert len(prev_arranged_netlist) > 0
                netline[0] = netline[0][1:]
                prev_arranged_netlist[-1] = prev_arranged_netlist[-1]+' '+' '.join(netline)
            else: 
                prev_arranged_netlist.append(' '.join(netline))

    for i, netline in enumerate(prev_arranged_netlist): 
        # manage nets(wires) : list the nets(wires) in the netlist.
        # circuit components (MOSFET, resistor, etc...) will not be handled here.
        netline = netline.split(' ')
        
        if netline[0][0] in ["M", "m"]: # MOSFET
            if netline[1] == '':
                node_list += netline[2:6]
                arranged_netlist.append(' '.join([netline[0]]+netline[2:]))
            else: 
                node_list += netline[1:5]
                arranged_netlist.append(' '.join(netline))
            
        elif netline[0][0] in ["C", "c"]: # Capacitor
            if netline[1] == '':
                node_list += netline[2:4]
                arranged_netlist.append(' '.join([netline[0]]+netline[2:]))
            else: 
                node_list += netline[1:3]
                arranged_netlist.append(' '.join(netline))
        
        elif netline[0][0] in ["X", "x"]: # subckt
            for net_arg_index in range(1, len(netline)):
                if netline[net_arg_index] == '':
                    break
                node_list.append(netline[net_arg_index])
            arranged_netlist.append(' '.join(netline))
        
        elif netline[0][0] in ["R", "r"]: # Resistor
            if netline[1] == '':
                node_list += netline[2:4]
                arranged_netlist.append(' '.join([netline[0]]+netline[2:]))
            else: 
                node_list += netline[1:3]
                arranged_netlist.append(' '.join(netline))
        
        elif netline[0][0] in ["L", "l"]: # Inductor
            if netline[1] == '':
                node_list += netline[2:4]
                arranged_netlist.append(' '.join([netline[0]]+netline[2:]))
            else: 
                node_list += netline[1:3]
                arranged_netlist.append(' '.join(netline))
        
        else: 
            print(netline)
            raise NotImplementedError
        

    if not redundancy_allow_flag:
        node_list = boundaries + list(set(node_list) - set(boundaries))
    else: 
        raise NotImplementedError
    net_stat = [node_list, None]
    #pdb.set_trace()
    return boundaries, arranged_netlist, net_stat


class circuit_processor():
    def __init__(self, args):
        self.args = args
        self.dataset = self.args.dataset
        self.ckt = self.args.circuit
        self.src_dir = ROOT_DIR + '/circuit_preprocess/'
        self.raw_dir = ROOT_DIR + '/raw_circuit/{}/'.format(self.dataset)
        self.proc_dir = ROOT_DIR + '/processed_circuit/{}/'.format(self.dataset)
        
        self.gnd_or_vsource = ['vddh!', 'gnd!', 'vbb!', 'vdd!', 'vddh2!', '0'] # fixed, do not change
        self.node_names = []
        self.node_types = [] 
        # 0 : vddh!, 1 : gnd!, 2 : vbb!, 3 : vdd!, 4 : vddh2!, 5 : 0, 6: boundary, 7: edge(circuit lines), 8: PMOS, 9: NMOS, 
        # 10: Capacitor, 11 : Resistor, 12: Inductor
        self.edge_types = [] 
        # 0~13 : in (going in to the circuit component)
        # 14~27 : out (going out from circuit component == going out to the wires)
        # 0 : PMOS_drain, 1: PMOS_gate, 2: PMOS_source, 3: PMOS_base, 
        # 4 : NMOS_drain, 5: NMOS_gate, 6: NMOS_source, 7: NMOS_base, 
        # 8 : C+, 9: C-, 10: R+, 11: R-, 12: I+, 13: I-
        self.edge_list = []
        self.boundary = [] # nodes that will be connnected to other circuits
        self.net_stat = []
        self.node_params = [] # parameters for nodes. for example, resistance, Wn, Wp,  values
        self.subckt_dict = {} # dictionary holding list of subcircuits. format example : {"inv" : [ [[boundary indices for inv_1],[node indices for inv_1],[indices of related edges in edge_list for inv_1]], ..., [[node indices for inv_k],[indices of related edges in edge_list for inv_k]] ]}
                              # with node indices, we can also retrieve node_types, node_names, node_params, edge_index, edge_types using indices.
                              # this dictionary must be updated whenever adding new circuit component as well as adding (merging) other subckt.
        
        # only during development
        
        self.raw_cir_file_name = self.raw_dir+"{}.cir".format(self.ckt)
        
        
        
    def parse_ckt(self):
        read_file = open(self.raw_cir_file_name, 'r')
        splitted_file = read_file.read().split('\n')[:-2]
        read_file.close()
        
        # remove comments, extract nets(wires) and boundary of circuit
        self.boundary, ckt_lines, net_stat = netlist_basic_preprocessing(splitted_file, self.ckt, self.args.gnd_vsource_allow_redundancy)
        
        # assign proper node types for nets(wires)
        self.node_names = net_stat[0]
        for i, node in enumerate(self.node_names):
            if node in self.boundary: # boundary
                self.node_types.append(6)
            elif node in self.gnd_or_vsource: # globals
                self.node_types.append(self.gnd_or_vsource.index(node))
            else: # boundary (regular edge(wires) of circuits)
                self.node_types.append(7)
            self.node_params.append('None')
            
        # now handle each components of circuit
        for line_idx, netline in enumerate(ckt_lines):
            netline = netline.split(' ')
            if netline[0][0] in ["M", "m"]: # MOSFET
                self.node_names.append(netline[0])
                self.node_params.append(','.join(netline[6:]))
                
                if netline[5] == "PMOS":
                    self.node_types.append(8)
                    target_node_idx = self.node_names.index(netline[0])
                    for port_idx in range(1,5):
                        source_node_idx = self.node_names.index(netline[port_idx])
                        self.edge_list.append([source_node_idx, target_node_idx])
                        self.edge_types.append(port_idx-1)
                        if netline[port_idx] in self.gnd_or_vsource:
                            if self.args.gnd_vsource_directional:
                                continue
                        self.edge_list.append([target_node_idx, source_node_idx])
                        self.edge_types.append(port_idx-1+14)
                        
                elif netline[5] == "NMOS":
                    self.node_types.append(9)
                    target_node_idx = self.node_names.index(netline[0])
                    for port_idx in range(1,5):
                        source_node_idx = self.node_names.index(netline[port_idx])
                        self.edge_list.append([source_node_idx, target_node_idx])
                        self.edge_types.append(port_idx-1+4)
                        if netline[port_idx] in self.gnd_or_vsource:
                            if self.args.gnd_vsource_directional:
                                continue
                        self.edge_list.append([target_node_idx, source_node_idx])
                        self.edge_types.append(port_idx-1+4+14)
                else: 
                    raise NotImplementedError

            elif netline[0][0] in ["C", "c"]: # Capacitor
                self.node_names.append(netline[0])
                self.node_params.append(','.join(netline[3:]))
                self.node_types.append(10)
                target_node_idx = self.node_names.index(netline[0])
                for port_idx in range(1,3):
                    source_node_idx = self.node_names.index(netline[port_idx])
                    self.edge_list.append([source_node_idx, target_node_idx])
                    self.edge_types.append(port_idx-1+8)
                    if netline[port_idx] in self.gnd_or_vsource:
                        if self.args.gnd_vsource_directional:
                            continue
                    self.edge_list.append([target_node_idx, source_node_idx])
                    self.edge_types.append(port_idx-1+8+14)
            
            elif netline[0][0] in ["X", "x"]: # subckt
                for net_arg_index in range(1, len(netline)):
                    if netline[net_arg_index] == '':
                        break 
                subckt_id = netline[0]
                subckt_found_boundaries = netline[1:net_arg_index]
                if netline[net_arg_index+1] == '':
                    subckt_type = netline[net_arg_index+2]
                    subckt_params = netline[net_arg_index+3:]
                else: 
                    subckt_type = netline[net_arg_index+1]
                    subckt_params = netline[net_arg_index+2:]
                
                # no need to add edge_list or edge_types as we are not adding new edges. Instead, edges must be managed
                self.manage_subckt(subckt_id, subckt_found_boundaries, subckt_type, subckt_params)
            
            elif netline[0][0] in ["R", "r"]: # Resistor
                self.node_names.append(netline[0])
                self.node_params.append(','.join(netline[3:]))
                self.node_types.append(11)
                target_node_idx = self.node_names.index(netline[0])
                for port_idx in range(1,3):
                    source_node_idx = self.node_names.index(netline[port_idx])
                    self.edge_list.append([source_node_idx, target_node_idx])
                    self.edge_types.append(port_idx-1+10)
                    if netline[port_idx] in self.gnd_or_vsource:
                        if self.args.gnd_vsource_directional:
                            continue
                    self.edge_list.append([target_node_idx, source_node_idx])
                    self.edge_types.append(port_idx-1+10+14)
            
            elif netline[0][0] in ["L", "l"]: # Inductor
                self.node_names.append(netline[0])
                self.node_params.append(','.join(netline[3:]))
                self.node_types.append(12)
                target_node_idx = self.node_names.index(netline[0])
                for port_idx in range(1,3):
                    source_node_idx = self.node_names.index(netline[port_idx])
                    self.edge_list.append([source_node_idx, target_node_idx])
                    self.edge_types.append(port_idx-1+12)
                    if netline[port_idx] in self.gnd_or_vsource:
                        if self.args.gnd_vsource_directional:
                            continue
                    self.edge_list.append([target_node_idx, source_node_idx])
                    self.edge_types.append(port_idx-1+12+14)
            
            else: 
                print(netline)
                raise NotImplementedError
        if self.ckt not in self.subckt_dict.keys():
            self.subckt_dict[self.ckt] = []
        self.subckt_dict[self.ckt].append([[self.node_names.index(wire) for wire in self.boundary],[i for i in range(len(self.node_types))], [i for i in range(len(self.edge_types))]])
        #pdb.set_trace()
        self.save_ckt()
        
    def save_ckt(self):
        # save : edge_list, edge_types, boundary, node_names, node_types, node_params, subckt_dict
        # txt_file : boundary \n node_names \n node_types \n node_params \n edge_list_1(source node) \n edge_list_2(target node) \n edge_types
        # pickle file : subckt_dict
        text = '' # empty string.
        self.edge_list = np.asarray(self.edge_list).T.tolist() # convert nx2 matrix to 2xn matrix (python list type)
        
        dir_flag = 'dir' if self.args.gnd_vsource_directional else 'undir'
        redun_flag = 'redun' if self.args.gnd_vsource_allow_redundancy else 'noredun'
        graph_file_name = self.proc_dir +'graph_files/{}_{}_{}.txt'.format(self.ckt, dir_flag, redun_flag)
        pickle_file_name = self.proc_dir +'graph_files/{}_{}_{}.pkl'.format(self.ckt, dir_flag, redun_flag)
        
        # txt file
        text += ' '.join(self.boundary) + '\n'
        text += ' '.join(self.node_names) + '\n'
        text += ' '.join([str(i) for i in self.node_types]) + '\n'
        text += ' '.join(self.node_params) + '\n'
        text += ' '.join([str(i) for i in self.edge_list[0]]) + '\n'
        text += ' '.join([str(i) for i in self.edge_list[1]]) + '\n'
        text += ' '.join([str(i) for i in self.edge_types]) + '\n'
        graph_file_opened = open(graph_file_name, 'w')
        graph_file_opened.write(text)
        graph_file_opened.close()

        # pickle file
        with open(pickle_file_name, 'wb') as opened_file :
            pickle.dump(self.subckt_dict, opened_file)

        return 
    
    def manage_subckt(self, ckt_id, ckt_found_boundaries, ckt_name, ckt_params):
        # function for handling data after reading file. use self.read_subckt_file(ckt_name) function to read 
        
        # loading subcircuit file
        subckt_graph, subckt_dict = self.read_subckt_file(ckt_name)
        subckt_boundary = subckt_graph[0]
        subckt_node_names = subckt_graph[1]
        subckt_node_types = subckt_graph[2]
        subckt_node_params = subckt_graph[3]
        subckt_edge_list = subckt_graph[4]
        subckt_edge_type = subckt_graph[5]
        
        # manage params
        if len(ckt_params) > 0:
            # first, parse ckt_params
            ckt_param_type = [aa.split('=')[0] for aa in ckt_params]
            ckt_param_values = [aa.split('=')[1] for aa in ckt_params]
            for subckt_param_idx in range(len(subckt_node_params)):
                if subckt_node_params[subckt_param_idx] == 'None':
                    continue
                # parse subckt_params
                subckt_params = subckt_node_params[subckt_param_idx].split(',')
                subckt_param_type = [aa.split('=')[0] for aa in subckt_params]
                subckt_param_values = [aa.split('=')[1] for aa in subckt_params]
                # match subckt and ckt params 
                for subckt_param_value_idx in range(len(subckt_param_values)):
                    if subckt_param_values[subckt_param_value_idx] in ckt_param_type:
                        ckt_param_idx = ckt_param_type.index(subckt_param_values[subckt_param_value_idx])
                        subckt_param_values[subckt_param_value_idx] = ckt_param_values[ckt_param_idx]
                subckt_node_params[subckt_param_idx] = ','.join(['{}={}'.format(subckt_param_type[merge_idx], subckt_param_values[merge_idx]) for merge_idx in range(len(subckt_param_type))])
            
        # The hardest part. Be careful
        ckt_found_boundaries_idx = []
        gnd_vsource_idx = []
        if not self.args.gnd_vsource_allow_redundancy:
            for gv_idx, gv_type in enumerate(self.gnd_or_vsource):
                if gv_type in self.node_names:
                    gnd_vsource_idx.append(self.node_names.index(gv_type))
                else: 
                    gnd_vsource_idx.append(-1)
        for bound_idx in range(len(ckt_found_boundaries)):
            ckt_found_boundaries_idx.append(self.node_names.index(ckt_found_boundaries[bound_idx]))
        start_idx_non_boundary = len(ckt_found_boundaries) # starting index of the nodes in subckt that are not boundary.
        node_idx_map = [] # convert node index of subckt to the index in ckt. (merging)
        node_idx_map += ckt_found_boundaries_idx
        new_indices = 0
        to_be_added = subckt_node_names[start_idx_non_boundary:]
        for new_index_map_idx in range(len(to_be_added)):
            if (not self.args.gnd_vsource_allow_redundancy) and (to_be_added[new_index_map_idx] in self.gnd_or_vsource) and (gnd_vsource_idx[self.gnd_or_vsource.index(to_be_added[new_index_map_idx])] > 0):
                node_idx_map.append(gnd_vsource_idx[self.gnd_or_vsource.index(to_be_added[new_index_map_idx])])
            else: 
                node_idx_map.append(len(self.node_names)+new_indices)
                new_indices += 1
        #node_idx_map += [len(self.node_names)+new_indices for new_indices in range(len(subckt_node_names[start_idx_non_boundary:]))]

        ckt_node_max_idx = len(self.node_names) - 1
        new_elems = (np.arange(len(node_idx_map))[np.asarray(node_idx_map) > ckt_node_max_idx]).tolist()
        for old_node_idx in new_elems: 
            if subckt_node_names[old_node_idx] in self.gnd_or_vsource:
                self.node_names.append(subckt_node_names[old_node_idx])
            else: 
                self.node_names.append(ckt_id+'/'+subckt_node_names[old_node_idx])
        self.node_params += [subckt_node_params[old_node_idx] for old_node_idx in new_elems]
        self.node_types += [subckt_node_types[old_node_idx] for old_node_idx in new_elems]
        len_edge_list_before_add = len(self.edge_list)
        self.edge_list += np.asarray([ np.asarray(node_idx_map)[subckt_edge_list[0]].tolist(), np.asarray(node_idx_map)[subckt_edge_list[1]].tolist()]).T.tolist()
        self.edge_types += subckt_edge_type
        len_edge_list_after_add = len(self.edge_list)
        # Now handle subckt dictionary
        for k,v in subckt_dict.items():
            for num_subckt_idx in range(len(v)):
                v[num_subckt_idx][0] = np.asarray(node_idx_map)[v[num_subckt_idx][0]].tolist()
                v[num_subckt_idx][1] = np.asarray(node_idx_map)[v[num_subckt_idx][1]].tolist()
                v[num_subckt_idx][2] = np.arange(len_edge_list_before_add, len_edge_list_after_add)[v[num_subckt_idx][2]].tolist()
            if k not in self.subckt_dict.keys():
                self.subckt_dict[k] = v
            else: 
                self.subckt_dict[k] += v

    
    def read_subckt_file(self, ckt_name):
        # description : function for just reading a file
        # return : graph and dictionary
        # graph : [boundary, node_names, node_types, node_params, edge_list, edge_types]
        # dictionary : single pyhton dictionary holding subckt position informations
        graph = []
        
        # file name to read
        dir_flag = 'dir' if self.args.gnd_vsource_directional else 'undir'
        redun_flag = 'redun' if self.args.gnd_vsource_allow_redundancy else 'noredun'
        graph_file_name = self.proc_dir +'graph_files/{}_{}_{}.txt'.format(ckt_name, dir_flag, redun_flag)
        pickle_file_name = self.proc_dir +'graph_files/{}_{}_{}.pkl'.format(ckt_name, dir_flag, redun_flag)
            
        # check whether file exist
        #pdb.set_trace()
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
            subckt_dict = pickle.load(pickle_file)
        
        return graph, subckt_dict
    
    
    def visualize_graph(self):
        ckt_name = self.args.circuit
        graph, subckt_dict = self.read_subckt_file(ckt_name)
        node_names = graph[1]
        node_types = graph[2]
        edge_list = graph[4]
        edge_type = graph[5]
        #pdb.set_trace()
        
        # setting color definitions
        ### For node type
        # 0 : vddh!, 1 : gnd!, 2 : vbb!, 3 : vdd!, 4 : vddh2!, 5 : 0, 6: boundary, 7: edge(circuit lines), 8: PMOS, 9: NMOS, 
        # 10: Capacitor, 11 : Resistor, 12: Inductor
        ## ground_or_voltage_source : yellow
        ## boundary : green
        ## wires(nets) : gray
        ## PMOS, NMOS : red, blue
        ## etc : white
        node_colors = ['yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'green', 'gray', 'red', 'blue', 'white', 'white', 'white']
        
        
        ### For edge type
        # 0~13 : in (going in to the circuit component)
        # 14~27 : out (going out from circuit component == going out to the wires)
        # 0 : PMOS_drain, 1: PMOS_gate, 2: PMOS_source, 3: PMOS_base, 
        # 4 : NMOS_drain, 5: NMOS_gate, 6: NMOS_source, 7: PMOS_base, 
        # 8 : C+, 9: C-, 10: R+, 11: R-, 12: I+, 13: I-
        # drain :red,  gate : blue, source : green, base : yellow, capacitor/inductor/resistor related: gray
        edge_colors = ['red','blue','green','yellow', 'red','blue','green','yellow', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'red','blue','green','yellow', 'red','blue','green','yellow', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray']
        
        
        
        
        nt = Network("2000px", "2000px", directed =True) 
        for idx in range(len(node_names)):
            nt.add_node(idx, label=node_names[idx], color=node_colors[node_types[idx]])
        edge_list = np.asarray(edge_list).T.tolist()
        for idx in range(len(edge_list)):
            nt.add_edge(edge_list[idx][0], edge_list[idx][1], color=edge_colors[edge_type[idx]])
        nt.show("{}.html".format(ckt_name))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # what is needed : subckt handler - subckt node index merger(handler), edge_index merger(handler), subckt_dict handler, node_name handler(be careful with boundaries)
    # handling arguments like Wn Wp is also needed
    
    
    
    
    
    
def parse_args():
    # Parses the node2vec arguments.
    parser = argparse.ArgumentParser(description="circuit parser")
    parser.add_argument('--dataset', default='company', choices=['company'], help='select dataset : company or GANA') 
    parser.add_argument('--circuit', type=str, default='inv', help='select dataset : company or GANA') 
    #parser.add_argument('--model', type=str, default='hgnn')
    #parser.add_argument('--lam1',type = float, default = 1.0, help='lam_1 for phenomnn' )
    parser.add_argument("--check_data", action = "store_true", help = "show tqdm bar")
    parser.add_argument("--visualize", action = "store_true", help = "visualize graph")
    parser.add_argument("--gnd_vsource_directional", action = "store_true", help = "When set as True, the edges that are connected to ground or voltage source become directional. (gnd or source --> net) is added but not (net --> gnd or source)")
    parser.add_argument("--gnd_vsource_allow_redundancy", action = "store_true", help = "gnd or voltage source as one node, or allow redundancy")
    parser.add_argument("--timer", action = "store_true", help = "show tqdm bar")
    args = parser.parse_args()
    return args
    
if __name__ == '__main__' : 
    
    args = parse_args()
    cp = circuit_processor(args)
    if args.visualize:
        cp.visualize_graph()
    else:
        cp.parse_ckt()
    