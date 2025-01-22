import torch 
import numpy as np 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool, global_add_pool
import pdb
import pickle
import random
import torch.optim.lr_scheduler as lr_scheduler
import datetime
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import collections 
from torch_geometric.utils import k_hop_subgraph
import networkx as nx
from networkx.algorithms import isomorphism
from time import time
from data_loader import target_detect_k


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, num_relations):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(in_channels, in_channels, num_relations, num_bases=30)
        self.conv2 = self.conv1
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2 = self.conv1
        return

    def forward(self, x, edge_index, edge_type, batch):
        node_features1 = self.conv1(x, edge_index, edge_type)
        x = F.relu(node_features1)
        node_features2 = self.conv2(x, edge_index, edge_type)
        graph_feature = global_add_pool(node_features2, batch)  # Perform average pooling
        return graph_feature, node_features1, node_features2 


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 1)
    
    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        return
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x



class GNN_embedder:
    def __init__(self, args):
        self.args = args
        self.root = args.root
        self.dataset = args.dataset
        self.proc_dir = self.root + '/processed_circuit/{}/'.format(self.dataset)
        self.src = self.root + '/src/'
        self.device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')  # CUDA 설정

        self.in_channels = 8  
        self.num_relations = 28  

        # Initialize the RGCN model
        self.model = RGCN(self.in_channels, self.num_relations).to(self.device) 
        ##self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        # self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

        # Initialize the MLP model
        self.mlp = MLP(input_dim=self.in_channels * 6).to(self.device).to(self.device)    # Concatenation of two embeddings
        ##self.mlp_optimizer = torch.optim.Adam(self.mlp.parameters(), lr=0.01)
        # self.mlp_scheduler = lr_scheduler.StepLR(self.mlp_optimizer, step_size=10, gamma=0.5)        

        # Number of epochs
        self.epochs = self.args.epoch

    def train_embedder(self, data):
        if self.args.use_only_first_split:
            self.args.num_repeat = 1
        negative_type_list = ["mutation", "partial", "others", "random"]
        
        for repeat_idx in tqdm(range(self.args.num_repeat)): # 각 repeat_idx마다 epoch이 돌아감
            self.model.reset_parameters()
            self.mlp.reset_parameters()
            #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
            #self.mlp_optimizer = torch.optim.Adam(self.mlp.parameters(), lr=0.01)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.mlp_optimizer = torch.optim.Adam(self.mlp.parameters(), lr=0.001)
            
            # ## load_model
            # checkpoint_dir = self.root+'/checkpoint/'
            # model_path = checkpoint_dir+'best_model_epoch_86_20241212_002849.pt'
            # mlp_path = checkpoint_dir+'best_mlp_epoch_86_20241212_002849.pt' 

            # self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            # self.mlp.load_state_dict(torch.load(mlp_path, map_location=self.device))
            # self.model.conv2 = self.model.conv1

            best_recall = 0.0  # Initialize the best accuracy variable
            best_model_path = None  # To store the best model path
            best_mlp_path = None  # To store the best model path
            
            # load split
            # pdb.set_trace()
            train_split = data[repeat_idx]['train'] 
            # train_split.keys() = dict_keys(['ctg', 'inv', 'delay10', 'nand2', 'nor3', 'nand5', 'nor4', 'nand3'])
            # test_split['ctg'].keys() = dict_keys(['target', 'positive', 'partial', 'mutation', 'random', 'others'])


            test_split = data[repeat_idx]['test']
            for epoch in tqdm(range(1, self.args.epoch + 1)):
                per_target_correct = {}
                per_target_total = {}
                per_target_predictions = {}
                per_target_labels = {}
                per_target_scores = {}
                for target_ckt in train_split.keys():
                    per_target_correct[target_ckt] = [0,0,0,0,0] # positive & mutation & partial & others & random
                    per_target_total[target_ckt] = [0,0,0,0,0]
                    per_target_predictions[target_ckt] = []
                    per_target_labels[target_ckt] = []
                    per_target_scores[target_ckt] = []
                epoch_loss = 0
                self.model.train()
                self.mlp.train() ################################################################## 이거 안했음
            
                for target_ckt in train_split.keys():
                    target_embedding = None
                    losses = []
                    target_data = train_split[target_ckt]['target'] # This is not a list, just one Data()
                    target_data.to(self.device)

                    target_graph_embedding, target_node_features1, target_node_features2  = self.model(target_data.x, target_data.edge_index, target_data.edge_type, None)
                    
                    target_initial_embedding = global_add_pool(target_data.x, None) 
                    target_one_hop_embedding = global_add_pool(target_node_features1, None) 
                    target_two_hop_embedding = global_add_pool(target_node_features2, None) 
                    target_embedding = torch.cat((target_initial_embedding, target_one_hop_embedding, target_two_hop_embedding), dim=1)
                   
                    _, radius = target_detect_k([i for i in range(len(target_data.x))], target_data.edge_index.tolist(), target_data.edge_index.tolist(), 'diameter', (target_data.x[:, :2] == 1).nonzero(as_tuple=True)[0].tolist())
                    for positive_sample in train_split[target_ckt]["positive"]:
                        positive_loader = DataLoader([positive_sample], batch_size=self.args.batch_size)

                        for positive_data in positive_loader:
                            positive_data.to(self.device)
                            positive_graph_embedding, positive_node_features1, positive_node_features2 = self.model(positive_data.x, positive_data.edge_index, positive_data.edge_type, positive_data.batch)
                            
                            node_idx_map2, _, _, _ = k_hop_subgraph(positive_data.center_node, radius-1, positive_data.edge_index.flip([0]), relabel_nodes=True, flow="target_to_source")
                            node_idx_batch2 = torch.zeros(positive_data.x.shape[0], device=self.device, dtype=torch.long).scatter_(0, node_idx_map2, 1)
                            node_idx_map3, _, _, _ = k_hop_subgraph(positive_data.center_node, radius-2, positive_data.edge_index.flip([0]), relabel_nodes=True, flow="target_to_source")
                            node_idx_batch3 = torch.zeros(positive_data.x.shape[0], device=self.device, dtype=torch.long).scatter_(0, node_idx_map3, 1)

                            positive_initial_embedding = positive_data.x.sum(dim=0).unsqueeze(0) 
                            positive_one_hop_embedding = positive_node_features1[node_idx_batch2 == 1].sum(dim=0).unsqueeze(0)
                            positive_two_hop_embedding = positive_node_features2[node_idx_batch3 == 1].sum(dim=0).unsqueeze(0)                              

                            positive_embedding = torch.cat((positive_initial_embedding, positive_one_hop_embedding, positive_two_hop_embedding), dim=1)
                        
                            concatenated_positive = torch.cat([target_embedding, positive_embedding], dim=1)
                            output_positive = self.mlp(concatenated_positive)
                            label_positive = torch.tensor([1.0], device=self.device)   # Positive sample label
                            loss_positive = F.binary_cross_entropy(output_positive, label_positive.unsqueeze(1))
                            
                            losses.append(loss_positive)
                            
                            # Training accuracy 계산
                            prediction = 1.0 if output_positive >= self.args.decision_threshold else 0.0
                            per_target_correct[target_ckt][0] += int(prediction)
                            per_target_total[target_ckt][0] += 1
                            per_target_predictions[target_ckt].append(prediction)  # 예측 결과 저장
                            per_target_labels[target_ckt].append(1.0)  # 실제 레이블 저장
                            per_target_scores[target_ckt].append(output_positive.cpu().detach().item())

                    
                    for n_type_idx, negative_sample_type in enumerate(negative_type_list): # currently, not using random

                        for negative_sample in train_split[target_ckt][negative_sample_type]:
                            negative_loader = DataLoader([negative_sample], batch_size=self.args.batch_size)
                            for negative_data in negative_loader:
                                negative_data.to(self.device)
                                negative_graph_embedding, negative_node_features1, negative_node_features2 = self.model(negative_data.x, negative_data.edge_index, negative_data.edge_type, negative_data.batch)
                                
                                node_idx_map2, _, _, _ = k_hop_subgraph(negative_data.center_node, radius-1, negative_data.edge_index.flip([0]), relabel_nodes=True, flow="target_to_source")
                                node_idx_batch2 = torch.zeros(negative_data.x.shape[0], device=self.device, dtype=torch.long).scatter_(0, node_idx_map2, 1)
                                node_idx_map3, _, _, _ = k_hop_subgraph(negative_data.center_node, radius-2, negative_data.edge_index.flip([0]), relabel_nodes=True, flow="target_to_source")
                                node_idx_batch3 = torch.zeros(negative_data.x.shape[0], device=self.device, dtype=torch.long).scatter_(0, node_idx_map3, 1)

                                negative_initial_embedding = negative_data.x.sum(dim=0).unsqueeze(0) 
                                negative_one_hop_embedding = negative_node_features1[node_idx_batch2 == 1].sum(dim=0).unsqueeze(0)
                                negative_two_hop_embedding = negative_node_features2[node_idx_batch3 == 1].sum(dim=0).unsqueeze(0)  
                                 
                                negative_embedding = torch.cat((negative_initial_embedding, negative_one_hop_embedding, negative_two_hop_embedding), dim=1)
                            

                                concatenated_negative = torch.cat([target_embedding, negative_embedding], dim=1)
                                output_negative = self.mlp(concatenated_negative)
                                label_negative = torch.tensor([0.0], device=self.device)   # Negative sample label
                                loss_negative = F.binary_cross_entropy(output_negative, label_negative.unsqueeze(1))
                            
                                losses.append(loss_negative)
                                
                                # Training accuracy 계산
                                prediction = 1.0 if output_negative >= self.args.decision_threshold else 0.0
                                per_target_correct[target_ckt][n_type_idx + 1] += 1-int(prediction)
                                per_target_total[target_ckt][n_type_idx + 1] += 1
                                per_target_predictions[target_ckt].append(prediction)  # 예측 결과 저장
                                per_target_labels[target_ckt].append(0.0)  # 실제 레이블 저장
                                per_target_scores[target_ckt].append(output_negative.cpu().detach().item())


                    if losses:
                        num_positive = len(train_split[target_ckt]['positive'])
                        num_negative = 0
                        for dt in negative_type_list:
                            num_negative += len(train_split[target_ckt][dt])
                        #total_loss = sum(losses[:num_positive])/num_positive + sum(losses[num_positive:])/num_negative
                        total_loss = sum(losses)
                        
                        self.mlp_optimizer.zero_grad()
                        self.optimizer.zero_grad()
                        
                        total_loss.backward()
                        
                        self.mlp_optimizer.step()
                        self.optimizer.step()
                        
                        epoch_loss += total_loss.item()
                
                # statistics, accuracy, precision, recall for each circuit type
                if self.args.train_verbose:
                    print(f"======================================  TRAIN epoch : {epoch} ====================================== \n")
                    for target_ckt in train_split.keys():
                        print("ABOUT : {}".format(target_ckt))
                        print("            \tpositive \tmutation \tpartial \tothers  \trandom")
                        print("NUM SAMPLE  \t{}".format('       \t'.join([str(int_to_str) for int_to_str in per_target_total[target_ckt]])))
                        print("NUM CORRECT \t{}".format('       \t'.join([str(int_to_str) for int_to_str in per_target_correct[target_ckt]])))
                        print("NUM WRONG   \t{}".format('       \t'.join([str(int_to_str) for int_to_str in (np.asarray(per_target_total[target_ckt]) - np.asarray(per_target_correct[target_ckt])).tolist()])))
                        print()
                        ckt_accuracy = sum(per_target_correct[target_ckt])/sum(per_target_total[target_ckt])*100
                        ckt_precision = precision_score(per_target_labels[target_ckt], per_target_predictions[target_ckt], zero_division=0)
                        ckt_recall = recall_score(per_target_labels[target_ckt], per_target_predictions[target_ckt])
                        ckt_auroc = roc_auc_score(per_target_labels[target_ckt], per_target_scores[target_ckt], average = None)
                        print(f"accuracy : {ckt_accuracy:.4f}  ||  precision : {ckt_precision:.4f}  ||  recall : {ckt_recall:.4f}  ||  auroc : {ckt_auroc:.4f}")
                        
                        print()
                        print()
                
                # statistics, accuracy, precision, recall in global
                correct = []
                total = []
                all_predictions = []
                all_labels = []
                all_scores = []
                for k,v in per_target_total.items():
                    total.append(v)
                for k,v in per_target_correct.items():
                    correct.append(v)
                for k,v in per_target_labels.items():
                    all_labels += v
                for k,v in per_target_predictions.items():
                    all_predictions += v
                for k,v in per_target_scores.items():
                    all_scores += v
                correct = np.asarray(correct).sum(0)
                total = np.asarray(total).sum(0)
                if self.args.train_verbose:
                    print("ABOUT : ALL")
                    print("            \tpositive \tmutation \tpartial \tothers  \trandom")
                    print("NUM SAMPLE  \t{}".format('       \t'.join([str(int_to_str) for int_to_str in total.tolist()])))
                    print("NUM CORRECT \t{}".format('       \t'.join([str(int_to_str) for int_to_str in correct.tolist()])))
                    print("NUM WRONG   \t{}".format('       \t'.join([str(int_to_str) for int_to_str in (total-correct).tolist()])))
                    print()
                accuracy = sum(correct.tolist())/sum(total.tolist())*100
                precision = precision_score(all_labels, all_predictions, zero_division=0)
                recall = recall_score(all_labels, all_predictions)
                auroc = roc_auc_score(all_labels, all_scores, average = None)
                print(f"Epoch {epoch}/{self.epochs}, Loss: {epoch_loss:.4f}")
                print(f"(TRAIN) accuracy : {accuracy:.4f}  ||  precision : {precision:.4f}  ||  recall : {recall:.4f}  ||  auroc : {auroc:.4f}")
                
                
                
                test_recall = self.valid_embedder(test_split, epoch)

                # Save the best model based on test accuracy
                if test_recall > best_recall:
                    best_recall = test_recall
                    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    checkpoint_dir = self.root+'/checkpoint/'
                    best_checkpoint_dir = checkpoint_dir+'saved_parameter/'
                    if not os.path.exists(checkpoint_dir):
                        os.mkdir(checkpoint_dir)
                    if not os.path.exists(best_checkpoint_dir):
                        os.mkdir(best_checkpoint_dir)
                    model_filename = f"{checkpoint_dir}best_model_epoch_{epoch}_{now}.pt"
                    mlp_filename = f"{checkpoint_dir}best_mlp_epoch_{epoch}_{now}.pt"
                    torch.save(self.model.state_dict(), model_filename)
                    torch.save(self.mlp.state_dict(), mlp_filename)
                    if self.args.save_best:
                        best_model_filename = f"{best_checkpoint_dir}best_model.pt"
                        best_mlp_filename = f"{best_checkpoint_dir}best_mlp.pt"
                        torch.save(self.model.state_dict(), best_model_filename)
                        torch.save(self.mlp.state_dict(), best_mlp_filename)
                    best_model_path = model_filename
                    best_mlp_path = mlp_filename
                    if self.args.verbose:
                        print(f"New best model saved as {model_filename}, {mlp_filename} with recall {best_recall:.4f}")

            if best_model_path:
                print(f"Best model during training was saved as: {best_model_path}, {best_mlp_path}")
                
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            self.mlp.load_state_dict(torch.load(best_mlp_path, map_location=self.device))
        
        
        return
    

    def valid_embedder(self, test_split, epoch):
        negative_type_list = ["mutation", "partial", "others", "random"]
        self.model.eval()
        self.mlp.eval()
        per_target_correct = {}
        per_target_total = {}
        per_target_predictions = {}
        per_target_labels = {}
        per_target_scores = {}
        for target_ckt in test_split.keys():
            per_target_correct[target_ckt] = [0,0,0,0,0] # positive & mutation & partial & others & random
            per_target_total[target_ckt] = [0,0,0,0,0]
            per_target_predictions[target_ckt] = []
            per_target_labels[target_ckt] = []
            per_target_scores[target_ckt] = []



        with torch.no_grad():
            for target_ckt in test_split.keys():
                target_data = test_split[target_ckt]['target']
                target_data.to(self.device)

                target_graph_embedding, target_node_features1, target_node_features2  = self.model(target_data.x, target_data.edge_index, target_data.edge_type, None)
                  
                target_initial_embedding = global_add_pool(target_data.x, None) 
                target_one_hop_embedding = global_add_pool(target_node_features1, None) 
                target_two_hop_embedding = global_add_pool(target_node_features2, None) 
                target_embedding = torch.cat((target_initial_embedding, target_one_hop_embedding, target_two_hop_embedding), dim=1) 

                _, radius = target_detect_k([i for i in range(len(target_data.x))], target_data.edge_index.tolist(), target_data.edge_index.tolist(), 'diameter', (target_data.x[:, :2] == 1).nonzero(as_tuple=True)[0].tolist())

                # Fixed test positive and negative samples 사용
                for positive_sample in test_split[target_ckt]["positive"]:
                    positive_loader = DataLoader([positive_sample], batch_size=self.args.batch_size)

                    for positive_data in positive_loader:
                        positive_data.to(self.device)
                        positive_graph_embedding, positive_node_features1, positive_node_features2 = self.model(positive_data.x, positive_data.edge_index, positive_data.edge_type, positive_data.batch)
                        
                        node_idx_map2, _, _, _ = k_hop_subgraph(positive_data.center_node, radius-1, positive_data.edge_index.flip([0]), relabel_nodes=True, flow="target_to_source")
                        node_idx_batch2 = torch.zeros(positive_data.x.shape[0], device=self.device, dtype=torch.long).scatter_(0, node_idx_map2, 1)
                        node_idx_map3, _, _, _ = k_hop_subgraph(positive_data.center_node, radius-2, positive_data.edge_index.flip([0]), relabel_nodes=True, flow="target_to_source")
                        node_idx_batch3 = torch.zeros(positive_data.x.shape[0], device=self.device, dtype=torch.long).scatter_(0, node_idx_map3, 1)
                                                  
                       
                        positive_initial_embedding = positive_data.x.sum(dim=0).unsqueeze(0) 
                        positive_one_hop_embedding = positive_node_features1[node_idx_batch2 == 1].sum(dim=0).unsqueeze(0)
                        positive_two_hop_embedding = positive_node_features2[node_idx_batch3 == 1].sum(dim=0).unsqueeze(0)                              
                        positive_embedding = torch.cat((positive_initial_embedding, positive_one_hop_embedding, positive_two_hop_embedding), dim=1)
                    
                        concatenated_positive = torch.cat([target_embedding, positive_embedding], dim=1).to(self.device)
                        output_positive = self.mlp(concatenated_positive)

                        prediction = 1.0 if output_positive >= self.args.decision_threshold else 0.0
                        per_target_correct[target_ckt][0] += int(prediction)
                        per_target_total[target_ckt][0] += 1
                        per_target_predictions[target_ckt].append(prediction)  # 예측 결과 저장
                        per_target_labels[target_ckt].append(1.0)  # 실제 레이블 저장
                        per_target_scores[target_ckt].append(output_positive.cpu().detach().item())
                    

                for n_type_idx, negative_sample_type in enumerate(negative_type_list): # currently, not using random

                    for negative_sample in test_split[target_ckt][negative_sample_type]:
                        negative_loader = DataLoader([negative_sample], batch_size=self.args.batch_size)
                        for negative_data in negative_loader:
                            negative_data.to(self.device)
                            negative_graph_embedding, negative_node_features1, negative_node_features2 = self.model(negative_data.x, negative_data.edge_index, negative_data.edge_type, negative_data.batch)
                            
                            node_idx_map2, _, _, _ = k_hop_subgraph(negative_data.center_node, radius-1, negative_data.edge_index.flip([0]), relabel_nodes=True, flow="target_to_source")
                            node_idx_batch2 = torch.zeros(negative_data.x.shape[0], device=self.device, dtype=torch.long).scatter_(0, node_idx_map2, 1)
                            node_idx_map3, _, _, _ = k_hop_subgraph(negative_data.center_node, radius-2, negative_data.edge_index.flip([0]), relabel_nodes=True, flow="target_to_source")
                            node_idx_batch3 = torch.zeros(negative_data.x.shape[0], device=self.device, dtype=torch.long).scatter_(0, node_idx_map3, 1)
                      
                            negative_initial_embedding = negative_data.x.sum(dim=0).unsqueeze(0) 
                            negative_one_hop_embedding = negative_node_features1[node_idx_batch2 == 1].sum(dim=0).unsqueeze(0)
                            negative_two_hop_embedding = negative_node_features2[node_idx_batch3 == 1].sum(dim=0).unsqueeze(0)  
                            negative_embedding = torch.cat((negative_initial_embedding, negative_one_hop_embedding, negative_two_hop_embedding), dim=1)
                        
                            concatenated_negative = torch.cat([target_embedding, negative_embedding], dim=1).to(self.device)
                            output_negative = self.mlp(concatenated_negative)
                        
                            prediction = 1.0 if output_negative >= self.args.decision_threshold else 0.0                          
                            per_target_correct[target_ckt][n_type_idx + 1] += 1-int(prediction)
                            per_target_total[target_ckt][n_type_idx + 1] += 1
                            per_target_predictions[target_ckt].append(prediction)  # 예측 결과 저장
                            per_target_labels[target_ckt].append(0.0)  # 실제 레이블 저장     
                            per_target_scores[target_ckt].append(output_negative.cpu().detach().item())           

            if self.args.test_verbose:
                print(f"\n======================================  TEST epoch : {epoch} ====================================== \n")
                for target_ckt in test_split.keys():
                    print("ABOUT : {}".format(target_ckt))
                    print("            \tpositive \tmutation \tpartial \tothers  \trandom")
                    print("NUM SAMPLE  \t{}".format('       \t'.join([str(int_to_str) for int_to_str in per_target_total[target_ckt]])))
                    print("NUM CORRECT \t{}".format('       \t'.join([str(int_to_str) for int_to_str in per_target_correct[target_ckt]])))
                    print("NUM WRONG   \t{}".format('       \t'.join([str(int_to_str) for int_to_str in (np.asarray(per_target_total[target_ckt]) - np.asarray(per_target_correct[target_ckt])).tolist()])))
                    print()
                    ckt_accuracy = sum(per_target_correct[target_ckt])/sum(per_target_total[target_ckt])*100
                    ckt_precision = precision_score(per_target_labels[target_ckt], per_target_predictions[target_ckt], zero_division=0)
                    ckt_recall = recall_score(per_target_labels[target_ckt], per_target_predictions[target_ckt])
                    ckt_auroc = roc_auc_score(per_target_labels[target_ckt], per_target_scores[target_ckt], average = None)
                    print(f"accuracy : {ckt_accuracy:.4f}  ||  precision : {ckt_precision:.4f}  ||  recall : {ckt_recall:.4f}  ||  auroc : {ckt_auroc:.4f}")
                    
                    print()
                
            # statistics, accuracy, precision, recall in global
            correct = []
            total = []
            all_predictions = []
            all_labels = []
            all_scores = []
            for k,v in per_target_total.items():
                total.append(v)
            for k,v in per_target_correct.items():
                correct.append(v)
            for k,v in per_target_labels.items():
                all_labels += v
            for k,v in per_target_predictions.items():
                all_predictions += v
            for k,v in per_target_scores.items():
                all_scores += v
            correct = np.asarray(correct).sum(0)
            total = np.asarray(total).sum(0)
            if self.args.test_verbose:
                print("ABOUT : ALL")
                print("            \tpositive \tmutation \tpartial \tothers  \trandom")
                print("NUM SAMPLE  \t{}".format('       \t'.join([str(int_to_str) for int_to_str in total.tolist()])))
                print("NUM CORRECT \t{}".format('       \t'.join([str(int_to_str) for int_to_str in correct.tolist()])))
                print("NUM WRONG   \t{}".format('       \t'.join([str(int_to_str) for int_to_str in (total-correct).tolist()])))
                print()
            accuracy = sum(correct.tolist())/sum(total.tolist())*100
            precision = precision_score(all_labels, all_predictions, zero_division=0)
            recall = recall_score(all_labels, all_predictions)
            auroc = roc_auc_score(all_labels, all_scores, average = None)
            print(f"(TEST)  accuracy : {accuracy:.4f}  ||  precision : {precision:.4f}  ||  recall : {recall:.4f}  ||  auroc : {auroc:.4f}")
            print()
            
        return auroc
        #return recall  # Return test accuracy
        

    def feature_matching(self, data): 

        if self.args.use_only_first_split:
            self.args.num_repeat = 1
        negative_type_list = ["mutation", "partial", "others", "random"]
        
        for repeat_idx in tqdm(range(self.args.num_repeat)):
            # train_split = data[repeat_idx]['train'] 
            
            # # train_split.keys() = dict_keys(['ctg', 'inv', 'delay10', 'nand2', 'nor3', 'nand5', 'nor4', 'nand3'])
            # # test_split['ctg'].keys() = dict_keys(['target', 'positive', 'partial', 'mutation', 'random', 'others'])

            # per_target_correct = {}
            # per_target_total = {}
            # per_target_predictions = {}
            # per_target_labels = {}
            # for target_ckt in train_split.keys():
            #     per_target_correct[target_ckt] = [0,0,0,0,0] # positive & mutation & partial & others & random
            #     per_target_total[target_ckt] = [0,0,0,0,0]
            #     per_target_predictions[target_ckt] = []
            #     per_target_labels[target_ckt] = []
            # epoch_loss = 0

            self.model.eval()
            per_target_correct = {}
            per_target_total = {}
            per_target_predictions = {}
            per_target_labels = {}
            per_positive_node_features = {}  # positive node features 저장
            per_target_node_features = {}  # target node features 저장

            test_split = data[repeat_idx]['test']
            for target_ckt in test_split.keys():
                per_target_correct[target_ckt] = [0,0,0,0,0] # positive & mutation & partial & others & random
                per_target_total[target_ckt] = [0,0,0,0,0]
                per_target_predictions[target_ckt] = []
                per_target_labels[target_ckt] = []

                per_positive_node_features[target_ckt] = []  # 각 target에 대한 positive node features 저장
                per_target_node_features[target_ckt] = None  # target node features 저장 초기화


            with torch.no_grad():
                for target_ckt in test_split.keys():
                    target_data = test_split[target_ckt]['target']
                    
                    target_loader = DataLoader([target_data], batch_size=1)
                    target_embedding, target_node_features = self.get_graph_embedding(target_loader)

                    # target_node_features 저장
                    per_target_node_features[target_ckt] = target_node_features
                    
                    # Fixed test positive and negative samples 사용
                    for positive_sample in test_split[target_ckt]["positive"]:
                        positive_loader = DataLoader([positive_sample], batch_size=1)
                        positive_embedding, positive_node_features = self.get_graph_embedding(positive_loader)
                        
                        # positive_node_features 저장
                        per_positive_node_features[target_ckt].append(positive_node_features)

                        target_node_features = self.normalize_features(target_node_features)
                        positive_node_features = self.normalize_features(positive_node_features)
                        similarity = torch.cdist(target_node_features, positive_node_features)  

                        subg_idx1 = []
                        subg_idx2 = []

                        while len(subg_idx2) < target_node_features.shape[0]:
                            min_idx = torch.argmin(similarity).item() 
                            row_idx = min_idx // len(positive_node_features)
                            col_idx = min_idx % len(positive_node_features) 

                            similarity[row_idx] = 1e9
                            similarity[:, col_idx] = 1e9
                            
                            subg_idx1.append(row_idx)
                            subg_idx2.append(col_idx)  

                        pdb.set_trace()

                        # concatenated_positive = torch.cat([target_embedding, positive_embedding], dim=1).to(self.device)
                        # output_positive = self.mlp(concatenated_positive)
                        
                        # prediction = torch.round(output_positive)
                        # per_target_correct[target_ckt][0] += (prediction == 1.0).sum().item()
                        # per_target_total[target_ckt][0] += 1
                        # per_target_predictions[target_ckt].append(prediction.item())  # 예측 결과 저장
                        # per_target_labels[target_ckt].append(1.0)  # 실제 레이블 저장
                                
                    # for subgraph in loop: 
                    #     subgraph = subgraph.to(device)
                    #     similarity = torch.cdist(subgraph, pattern)  

                    #     subg_idx = []
                    #     while len(subg_idx) < K:
                    #         min_idx = torch.argmin(similarity).item() 
                    #         row_idx = min_idx // len(pattern)
                    #         col_idx = min_idx % len(pattern) 

                    #         similarity[row_idx] = 1e9
                    #         similarity[:, col_idx] = 1e9
                    #         subg_idx.append(row_idx)  


    def normalize_features(self, features):
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True) + 1e-9  # 분산이 0인 경우를 방지하기 위해 작은 값을 추가
        return (features - mean) / std
    
    def load_embedder(self):
        checkpoint_dir = self.root+'/checkpoint/'
        if self.args.save_best or self.args.load_parameter_file_name == 'best':

            best_checkpoint_dir = checkpoint_dir+'saved_parameter/'
            best_model_path = f"{best_checkpoint_dir}best_model.pt"
            best_mlp_path = f"{best_checkpoint_dir}best_mlp.pt"
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            self.mlp.load_state_dict(torch.load(best_mlp_path, map_location=self.device))
            self.model.conv2 = self.model.conv1
        else: 
            file_name = self.args.load_parameter_file_name
            model_filename = f"{checkpoint_dir}best_model_epoch_{file_name}.pt"
            mlp_filename = f"{checkpoint_dir}best_mlp_epoch_{file_name}.pt"
            self.model.load_state_dict(torch.load(model_filename, map_location=self.device))
            self.mlp.load_state_dict(torch.load(mlp_filename, map_location=self.device))
            self.model.conv2 = self.model.conv1
        return 
    
    
    def test_embedder_research(self, target, graphs, labels, entire_graph, entire_subckt_dict, raw_graphs, perform_analysis): 
        # Measure performance in realistic manner (bounding box text). 
        raw_outputs = []
        predictions = []
        target_loader = DataLoader([target], batch_size=1)
        target_emb, _ = self.get_graph_embedding(target_loader)
        for graph in graphs:
            graph_loader = DataLoader([graph], batch_size=1)
            graph_emb, _ = self.get_graph_embedding(graph_loader)
            concat_emb = torch.cat([target_emb, graph_emb], dim=1)
            classifier_out = self.mlp(concat_emb)
            raw_outputs.append(classifier_out.item())
            if classifier_out.item() >= self.args.decision_threshold:
                predictions.append(1.0)
            else: 
                predictions.append(0.0)
        ckt_accuracy = (1-(labels.long()+torch.LongTensor(predictions))%2).sum().item()/labels.size(0)*100
        ckt_precision = precision_score(labels.tolist(), predictions, zero_division=0)
        ckt_recall = recall_score(labels.tolist(), predictions)
        print(f"(TEST)  accuracy : {ckt_accuracy:.4f}  ||  precision : {ckt_precision:.4f}  ||  recall : {ckt_recall:.4f}")
        
        
        if perform_analysis:
            self.fig_dir = self.root + '/figures/test_embedder_research/'
            print("size of circuit (number of nodes) : {}".format(str(target.x.size(0))))
            sample_size = []
            for sample_idx in range(len(raw_outputs)):
                sample_size.append(len(raw_graphs[sample_idx][1]))
            sample_size = torch.Tensor(sample_size)
            raw_outputs = torch.Tensor(raw_outputs)
            positive_idx = ((labels == 1.0).nonzero(as_tuple=True)[0])
            negative_idx = ((labels == 0.0).nonzero(as_tuple=True)[0])
            correctness = (1-(labels.long()+torch.LongTensor(predictions))%2)
            #positive dist
            pos_correct_idx = ((correctness[positive_idx] == 1).nonzero(as_tuple=True)[0])
            pos_wrong_idx = ((correctness[positive_idx] == 0).nonzero(as_tuple=True)[0])
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(sample_size[positive_idx[pos_correct_idx]], raw_outputs[positive_idx[pos_correct_idx]], c='b', marker="o", label='correct')
            ax1.scatter(sample_size[positive_idx[pos_wrong_idx]], raw_outputs[positive_idx[pos_wrong_idx]], c='r', marker="o", label='wrong')
            plt.legend(loc='upper left')
            fig.savefig(self.fig_dir+'size_by_confidence/svg/{}_in_{}_positive.svg'.format(self.args.target_circuit, self.args.entire_circuit)) 
            fig.savefig(self.fig_dir+'size_by_confidence/png/{}_in_{}_positive.png'.format(self.args.target_circuit, self.args.entire_circuit)) 
            plt.clf()
            
            # neg dist
            neg_correct_idx = ((correctness[negative_idx] == 1).nonzero(as_tuple=True)[0])
            neg_wrong_idx = ((correctness[negative_idx] == 0).nonzero(as_tuple=True)[0])
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(sample_size[negative_idx[neg_correct_idx]], raw_outputs[negative_idx[neg_correct_idx]], c='b', marker="o", label='correct')
            ax1.scatter(sample_size[negative_idx[neg_wrong_idx]], raw_outputs[negative_idx[neg_wrong_idx]], c='r', marker="o", label='wrong')
            plt.legend(loc='upper left')
            fig.savefig(self.fig_dir+'size_by_confidence/svg/{}_in_{}_negative.svg'.format(self.args.target_circuit, self.args.entire_circuit)) 
            fig.savefig(self.fig_dir+'size_by_confidence/png/{}_in_{}_negative.png'.format(self.args.target_circuit, self.args.entire_circuit)) 
            plt.clf()
            
            pos_correct_idx = positive_idx[pos_correct_idx]
            pos_wrong_idx = positive_idx[pos_wrong_idx]
            neg_correct_idx = negative_idx[neg_correct_idx]
            neg_wrong_idx = negative_idx[neg_wrong_idx]
            graph_size_list = sample_size[pos_correct_idx].tolist()+sample_size[pos_wrong_idx].tolist() + sample_size[neg_correct_idx].tolist() + sample_size[neg_wrong_idx].tolist()
            graph_type_list = ['pos_correct' for i in range(sample_size[pos_correct_idx].size(0))] +\
                ['pos_wrong' for i in range(sample_size[pos_wrong_idx].size(0))] +\
                    ['neg_correct' for i in range(sample_size[neg_correct_idx].size(0))]+\
                        ['neg_wrong' for i in range(sample_size[neg_wrong_idx].size(0))]
            data = pd.DataFrame({'case': graph_type_list, 'size': graph_size_list})
            fig = sns.violinplot(x="case", y="size", data=data).get_figure()
            fig.savefig(self.fig_dir+'case_by_size/svg/{}_in_{}.svg'.format(self.args.target_circuit, self.args.entire_circuit)) 
            fig.savefig(self.fig_dir+'case_by_size/png/{}_in_{}.png'.format(self.args.target_circuit, self.args.entire_circuit)) 
            fig.clf()



            target_places = [i[1] for i in entire_subckt_dict[self.args.target_circuit]]
            jaccards = []
            intersection_ratios = []
            for rg in raw_graphs:
                raw_nodes_list = rg[0]
                max_jaccard = 0
                max_intersection_ratio = 0
                for tp in target_places:
                    jaccard_value = utils.jaccard_similarity(raw_nodes_list, tp)
                    intersection_ratio = len(set(tp).intersection(set(raw_nodes_list)))/target.x.size(0)
                    if max_jaccard < jaccard_value:
                        max_jaccard = jaccard_value
                    if max_intersection_ratio < intersection_ratio:
                        max_intersection_ratio = intersection_ratio
                intersection_ratios.append(max_intersection_ratio)
                jaccards.append(max_jaccard)
            intersection_ratios = torch.Tensor(intersection_ratios)    
            jaccards = torch.Tensor(jaccards)
            
            # jaccard
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(jaccards[positive_idx], raw_outputs[positive_idx], c='b', marker="o", label='positive(1)')
            ax1.scatter(jaccards[negative_idx], raw_outputs[negative_idx], c='r', marker="o", label='negative(0)')
            plt.legend(loc='upper left')
            fig.savefig(self.fig_dir+'jaccard_confidence_intersection/svg/{}_in_{}_jaccard.svg'.format(self.args.target_circuit, self.args.entire_circuit)) 
            fig.savefig(self.fig_dir+'jaccard_confidence_intersection/png/{}_in_{}_jaccard.png'.format(self.args.target_circuit, self.args.entire_circuit)) 
            plt.clf()
            
            # intersection
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(intersection_ratios[positive_idx], raw_outputs[positive_idx], c='b', marker="o", label='positive(1)')
            ax1.scatter(intersection_ratios[negative_idx], raw_outputs[negative_idx], c='r', marker="o", label='negative(0)')
            plt.legend(loc='upper left')
            fig.savefig(self.fig_dir+'jaccard_confidence_intersection/svg/{}_in_{}_intersection.svg'.format(self.args.target_circuit, self.args.entire_circuit)) 
            fig.savefig(self.fig_dir+'jaccard_confidence_intersection/png/{}_in_{}_intersection.png'.format(self.args.target_circuit, self.args.entire_circuit)) 
            plt.clf()
            
            '''
            pdb.set_trace()
            fig = plt.figure()
            ax1 = fig.add_subplot(projection='3d')
            ax1.scatter(jaccards[positive_idx], intersection_ratios[positive_idx], raw_outputs[positive_idx], c='b', marker="o", label='positive(1)')
            ax1.scatter(jaccards[negative_idx], intersection_ratios[negative_idx], raw_outputs[negative_idx], c='r', marker="o", label='negative(0)')
            ax1.set_xlabel('jaccard')
            ax1.set_ylabel('intersection')
            ax1.set_zlabel('score')
            #plt.show()
            plt.clf()
            '''
            
            '''
            fig = plt.figure()
            cmap = plt.cm.bwr
            ax1 = fig.add_subplot(111)
            ax1.scatter(jaccards, intersection_ratios, c=raw_outputs, marker="o", cmap=cmap)
            cbar = fig.colorbar()
            fig.savefig(self.fig_dir+'jaccard_confidence_intersection/svg/{}_in_{}_jaccard_intersection.svg'.format(self.args.target_circuit, self.args.entire_circuit)) 
            fig.savefig(self.fig_dir+'jaccard_confidence_intersection/png/{}_in_{}_jaccard_intersection.png'.format(self.args.target_circuit, self.args.entire_circuit)) 
            plt.clf()
            '''
            
            
            
        return


    def test_bb(self, target_graph, entire_graph, radius, num_labels):
        
        target_graph.to(self.device)
        entire_graph.to(self.device)
        self.model.eval()
        self.mlp.eval()

        nx_target_graph = nx.MultiDiGraph()
        for i in range(target_graph.x.shape[0]):
            nx_target_graph.add_node(i, feature=torch.argmax(target_graph.x[i]).item())  # 각 노드에 대한 특성 저장

        for edge, edge_type in zip(target_graph.edge_index.T.tolist(), target_graph.edge_type):
            nx_target_graph.add_edge(edge[0], edge[1], feature=edge_type.item())

        if self.args.target_circuit in ['deco1m', 'col_sel1m']:
            nx_target_graph.remove_nodes_from([4, 8, 12])
            nx_target_graph.remove_edges_from([(4, 12, 0), (8, 12, 0), (12, 4, 0)])

        start_time = time()

        print("타겟 회로 모델 적용 시작")
        time1=time()
        _, target_node_features1, target_node_features2  = self.model(target_graph.x, target_graph.edge_index, target_graph.edge_type, None)

        target_initial_embedding = global_add_pool(target_graph.x, None) 
        target_one_hop_embedding = global_add_pool(target_node_features1, None) 
        target_two_hop_embedding = global_add_pool(target_node_features2, None) 
        target_embedding = torch.cat((target_initial_embedding, target_one_hop_embedding, target_two_hop_embedding), dim=1)
        
        _, entire_node_features1, entire_node_features2  = self.model(entire_graph.x, entire_graph.edge_index, entire_graph.edge_type, None)
        time2=time()
        print("시간 : ", time2-time1)
        
        print("전체 회로 모델 적용 시작")
        time3=time()        
        raw_outputs = []
        center_nodes = []        
        for node_idx in range(entire_graph.x.shape[0]):
            node_idx_map1, _, _, _ = k_hop_subgraph(torch.LongTensor([node_idx]), radius, entire_graph.edge_index.flip([0]), relabel_nodes=True, flow="target_to_source")
            node_idx_batch1 = torch.zeros(entire_graph.x.shape[0], device=self.device, dtype=torch.long).scatter_(0, node_idx_map1, 1)
            if target_graph.x.shape[0] >= node_idx_map1.size(0):
                continue
            node_idx_map2, _, _, _ = k_hop_subgraph(torch.LongTensor([node_idx]), radius-1, entire_graph.edge_index.flip([0]), relabel_nodes=True, flow="target_to_source")
            node_idx_batch2 = torch.zeros(entire_graph.x.shape[0], device=self.device, dtype=torch.long).scatter_(0, node_idx_map2, 1)
            node_idx_map3, _, _, _ = k_hop_subgraph(torch.LongTensor([node_idx]), radius-2, entire_graph.edge_index.flip([0]), relabel_nodes=True, flow="target_to_source")
            node_idx_batch3 = torch.zeros(entire_graph.x.shape[0], device=self.device, dtype=torch.long).scatter_(0, node_idx_map3, 1)
    

            entire_initial_embedding = entire_graph.x[node_idx_batch1 == 1].sum(dim=0).unsqueeze(0) 
            entire_one_hop_embedding = entire_node_features1[node_idx_batch2 == 1].sum(dim=0).unsqueeze(0)
            entire_two_hop_embedding = entire_node_features2[node_idx_batch3 == 1].sum(dim=0).unsqueeze(0)
            entire_embedding = torch.cat((entire_initial_embedding, entire_one_hop_embedding, entire_two_hop_embedding), dim=1)
            concat_emb = torch.cat([target_embedding, entire_embedding], dim=1)
            classifier_out = self.mlp(concat_emb)
            raw_outputs.append(classifier_out.item())            
            center_nodes.append(node_idx)

        descending_arg_index = np.argsort(np.asarray(raw_outputs)).tolist()
        descending_arg_index.reverse()
        # descending_raw_outputs = np.asarray(raw_outputs)[descending_arg_index]
        descending_center_nodes = np.asarray(center_nodes)[descending_arg_index]
        time4=time()        
        print("시간 : ", time4-time3)

        # print("매칭 시작")     
        cumulative_matched_nodes = set()    
        first_time_print = False    
        for idx, node_idx in enumerate(descending_center_nodes): # batch_size + A^K 미리 설정
            if idx % 100 == 0:
                print(idx)    
                end_time3 = time()
                print('Time taken : {}'.format(end_time3 - start_time))    
            node_idx_map, new_edge_index, mapping, edge_mask = k_hop_subgraph(torch.LongTensor([node_idx]), radius, entire_graph.edge_index.flip([0]), relabel_nodes=True, flow="target_to_source")
            nx_candidate_graph = nx.MultiDiGraph()
            for node in node_idx_map.tolist():
                nx_candidate_graph.add_node(node, feature=torch.argmax(entire_graph.x[node]).item()) 
            new_edge_index_mapped = node_idx_map[new_edge_index].flip(0)
            new_edge_type = entire_graph.edge_type[edge_mask]
            for edge, edge_type in zip(new_edge_index_mapped.T.tolist(), new_edge_type):
                nx_candidate_graph.add_edge(edge[0], edge[1], feature=edge_type.item())
            
            node_match = nx.algorithms.isomorphism.categorical_node_match("feature", None)
            edge_match = nx.algorithms.isomorphism.categorical_multiedge_match("feature", None)
            matcher = isomorphism.MultiDiGraphMatcher(nx_candidate_graph, nx_target_graph, node_match=node_match, edge_match=edge_match)
            vf2_labels = list(matcher.subgraph_isomorphisms_iter())
            all_matching_list = [tuple(sorted(subgraph_mapping, key=lambda k: subgraph_mapping[k])) for subgraph_mapping in vf2_labels]

            for matching_nodes in all_matching_list:
                if matching_nodes not in cumulative_matched_nodes:
                    cumulative_matched_nodes.add(matching_nodes)
                    print(f"Total unique matches: {len(cumulative_matched_nodes)}")
                    if len(cumulative_matched_nodes) >= 1 and first_time_print==False: 
                        end_time2 = time()
                        print('Time taken : {}'.format(end_time2 - start_time))
                        first_time_print=True

            if len(cumulative_matched_nodes) == num_labels:
                end_time = time()
                print('Time taken : {}'.format(end_time - start_time))
                break

        
        return len(cumulative_matched_nodes)          



    def get_bb(self, target, graphs, raw_graphs): ######################### need more, raw data
        candidate = None
        raw_outputs = []
        predictions = []
        pdb.set_trace()
        target.to(self.device)

        target_graph_embedding, target_node_features1, target_node_features2  = self.model(target.x, target.edge_index, target.edge_type, None)
        
        target_initial_embedding = global_add_pool(target.x, None) 
        target_one_hop_embedding = global_add_pool(target_node_features1, None) 
        target_two_hop_embedding = global_add_pool(target_node_features2, None) 
        target_embedding = torch.cat((target_initial_embedding, target_one_hop_embedding, target_two_hop_embedding), dim=1)

        graph_loader = DataLoader(graphs, batch_size=self.args.batch_size)
        for graph_data in graph_loader:
            graph_data.to(self.device)
            graph_embedding, node_features1, node_features2 = self.model(graph_data.x, graph_data.edge_index, graph_data.edge_type, graph_data.batch)
            
            graph_loader = DataLoader([graph], batch_size=1)
            graph_emb, _ = self.get_graph_embedding(graph_loader)
            concat_emb = torch.cat([target_emb, graph_emb], dim=1)
            classifier_out = self.mlp(concat_emb)
            raw_outputs.append(classifier_out.item())

        descending_arg_index = np.argsort(np.asarray(raw_outputs)).tolist()
        descending_arg_index.reverse()
        descending_raw_output = np.asarray(raw_outputs)[descending_arg_index]
       
       
        # pdb.set_trace()
        topk = int(1*len(descending_raw_output))  # 원하는 상위 개수를 self.args.topk로 지정
        topk = min(topk, len(descending_raw_output))  # raw_outputs보다 큰 경우 방지
        active_raw_output = descending_raw_output[:topk]
        active_arg_index = descending_arg_index[:topk]  
            
       
        # number_of_positives = np.sum(descending_raw_output >= self.args.decision_threshold)
        # active_raw_output = descending_raw_output[:number_of_positives]
        # active_arg_index = descending_arg_index[:number_of_positives]


        # # jaccard_matrix = np.zeros((number_of_positives, number_of_positives))
        # # for j1 in range(number_of_positives):
        # #     for j2 in range(j1,number_of_positives):
        # #         graph1 = graphs[descending_arg_index[j1]] # graphs : k-hop 뽑은 그래프
        # #         graph2 = graphs[descending_arg_index[j2]]  ######### Not this, need to use raw data to get graph
        # #         jaccard_matrix[j1][j2] = utils.jaccard_similarity(graph1, graph2)
        

        # pdb.set_trace()
        candidate = [graphs[i] for i in active_arg_index]
        raw_candidate = [raw_graphs[i] for i in active_arg_index]
        
        # hierarchical clustering? --> No
        ## get clusters with jaccard similarity larger than threshold
        ## get statistics of nodes, select top k(proportional to target ckt size)
        ## select graphs that contains most of them.
        
        # self.args.bb_merge_threshold == 0.0 # do note merge
        return candidate, raw_candidate

    def feature_matching_bb(self, target, candidates, raw_candidates, entire_graph, entire_subckt_dict):
        pdb.set_trace()
        target_loader = DataLoader([target], batch_size=1)
        self.model.eval()
        target_embedding, target_node_features = self.model(target.x, target.edge_index, target.edge_type, target.batch)
        # self.get_graph_embedding(target_loader)    

        per_target_correct = {}
        per_target_total = {}
        per_target_predictions = {}
        per_target_labels = {}
        per_positive_node_features = {}  # positive node features 저장
        per_target_node_features = {}  # target node features 저장
        prediction = 0

        node_labels = [sublist[1] for sublist in entire_subckt_dict[self.args.target_circuit]]
        label_idx_set = set(range(len(node_labels)))

        # Fixed test positive and negative samples 사용
        for cand_idx, candidate in enumerate(candidates): # for cand_idx, candidate in enumerate(candidates[1:], start=1):
            print('cand_idx = ', cand_idx)
            candidate_embedding, candidate_features = self.model(candidate.x, candidate.edge_index, candidate.edge_type, candidate.batch)

            distance = torch.cdist(target_node_features, candidate_features)

            # normalized_target_node_features = self.normalize_features(target_node_features)
            # normalized_candidate_features = self.normalize_features(candidate_features)
            # similarity = torch.cdist(normalized_target_node_features, normalized_candidate_features)  

            subg_idx1 = []
            subg_idx2 = []

            # 임시로 제외된 노드 쌍을 추적하기 위한 리스트
            disconnected_pairs = []
            initial_skipped_pairs = []
            # while len(subg_idx2) < target_node_features.shape[0]:
            #     min_idx = torch.argmin(distance).item() 
            #     row_idx = min_idx // len(candidate_features)
            #     col_idx = min_idx % len(candidate_features) 


            #     distance[row_idx] = 1e9
            #     distance[:, col_idx] = 1e9
                
            #     subg_idx1.append(row_idx)
            #     subg_idx2.append(col_idx)
              

            while len(subg_idx2) < target_node_features.shape[0]:
                found_match = False
                # pdb.set_trace()
                
                count = 0
                while not found_match:
                    min_idx = torch.argmin(distance).item()
                    row_idx = min_idx // len(candidate_features)
                    col_idx = min_idx % len(candidate_features)



                    # if not subg_idx2:
                    #     row_idx = 5
                    #     col_idx = 81         

                    # if torch.all(target.x[row_idx] == candidate.x[col_idx]):
                    #     found_match = True
                    # pdb.set_trace()

                    # x 값이 같은지 확인
                    target_group = (target.x[row_idx] == 1).nonzero(as_tuple=True)[0].item() # self.map_feature_to_group(target.x[row_idx])
                    candidate_group = (candidate.x[col_idx] == 1).nonzero(as_tuple=True)[0].item() # self.map_feature_to_group(candidate.x[col_idx])


                    if not subg_idx1 and target_group in [0, 1]:
                        # 해당 쌍을 건너뛰고 나중에 다시 시도 가능하도록 리스트에 추가
                        initial_skipped_pairs.append((row_idx, col_idx))
                        # 거리 값을 높게 설정하여 다음 최소 값 선택 시 건너뛰도록 함
                        distance[row_idx, col_idx] = 1e9
                        
                        
                        if (distance == 1.0000e+09).all():
                            break

                        continue

                    # target의 연결된 노드 (one-hop) 찾기
                    target_connected_indices = (target.edge_index[0] == row_idx) | (target.edge_index[1] == row_idx)
                    target_connected_edges = target.edge_index[:, target_connected_indices]
                    target_one_hop_nodes = torch.where(target_connected_edges[0] == row_idx, target_connected_edges[1], target_connected_edges[0])
                    target_connected_edge_types = target.edge_type[target_connected_indices]
                    target_one_hop_group_count = self.count_nodes_by_group(target_one_hop_nodes, target_connected_edge_types, target) #, target_connected_edges , target_one_hop_edge_types)

                    candidate_connected_indices = (candidate.edge_index[0] == col_idx) | (candidate.edge_index[1] == col_idx)
                    candidate_connected_edges = candidate.edge_index[:, candidate_connected_indices]
                    candidate_one_hop_nodes  = torch.where(candidate_connected_edges[0] == col_idx, candidate_connected_edges[1], candidate_connected_edges[0])
                    candidate_connected_edge_types = candidate.edge_type[candidate_connected_indices]
                    candidate_one_hop_group_count = self.count_nodes_by_group(candidate_one_hop_nodes, candidate_connected_edge_types, candidate)

                    # # 이제 각 노드에 대한 edge type을 매핑해야 하므로, target_connected_edges에서 노드와 연결된 edge type을 zip으로 묶음
                    # connected_node_edge_list = []

                    # # 각 연결된 엣지를 순회하며, row_idx가 아닌 다른 노드를 찾아서 엣지 타입과 함께 저장
                    # for i, edge in enumerate(target_connected_edges.T):  # target_connected_edges의 전치를 통해 edge 쌍을 얻음
                    #     if edge[0].item() == row_idx:
                    #         connected_node_edge_list.append((edge[1].item(), target_connected_edge_types[i].item()))
                    #     else:
                    #         connected_node_edge_list.append((edge[0].item(), target_connected_edge_types[i].item()))

                    # # 결과: [(노드, edge_type), (노드, edge_type), ...]
                    # print(connected_node_edge_list)
                                        
                    
                    
                    # target_one_hop_nodes = target_connected_edges[target_connected_edges != row_idx].unique()
                    # target_one_hop_edge_types = list(zip(target_one_hop_nodes.tolist(), target_connected_edge_types.tolist()))
                    # target_one_hop_nodes = target_one_hop_nodes[(target.x[target_one_hop_nodes] != 0).any(dim=1) & (target.x[target_one_hop_nodes] != 1).any(dim=1)]

                    # # candidate의 연결된 노드 (one-hop) 찾기
                    # candidate_connected_indices = (candidate.edge_index[0] == col_idx) | (candidate.edge_index[1] == col_idx)
                    # candidate_connected_edges = candidate.edge_index[:, candidate_connected_indices]
                    # candidate_one_hop_nodes  = candidate_connected_edges[candidate_connected_edges != col_idx].unique()
                    # # candidate_one_hop_nodes = candidate_one_hop_nodes[(candidate.x[candidate_one_hop_nodes] != 0).any(dim=1) & (candidate.x[candidate_one_hop_nodes] != 1).any(dim=1)]


                    # target의 two-hop 이웃 찾기
                    target_two_hop_nodes = []
                    target_two_hop_edge_types = []
                    for node in target_one_hop_nodes.unique():
                        # 각 one-hop 노드의 이웃 노드를 찾기
                        indices = (target.edge_index[0] == node) | (target.edge_index[1] == node)
                        edges = target.edge_index[:, indices]
                        # node가 edges의 첫 번째 열에 있으면 두 번째 열의 값을, 두 번째 열에 있으면 첫 번째 열의 값을 추출
                        neighbors = torch.where(edges[0] == node, edges[1], edges[0]) # 조건, 참, 거짓

                        edge_types = target.edge_type[indices]
                        # neighbors = neighbors[(target.x[neighbors] != 0).any(dim=1) & (target.x[neighbors] != 1).any(dim=1)]
                        target_two_hop_nodes.append(neighbors)
                        target_two_hop_edge_types.append(edge_types)

                    # Flatten and make unique
                    target_two_hop_nodes = torch.cat(target_two_hop_nodes)
                    target_two_hop_edge_types = torch.cat(target_two_hop_edge_types)
                    target_two_hop_group_count = self.count_nodes_by_group(target_two_hop_nodes, target_two_hop_edge_types, target)
                
                    # candidate의 two-hop 이웃 찾기
                    candidate_two_hop_nodes = []
                    candidate_two_hop_edge_types = []
                    for node in candidate_one_hop_nodes.unique():
                        # 각 one-hop 노드의 이웃 노드를 찾기
                        indices = (candidate.edge_index[0] == node) | (candidate.edge_index[1] == node)
                        edges = candidate.edge_index[:, indices]
                        # node가 edges의 첫 번째 열에 있으면 두 번째 열의 값을, 두 번째 열에 있으면 첫 번째 열의 값을 추출
                        neighbors = torch.where(edges[0] == node, edges[1], edges[0]) # 조건, 참, 거짓
                        
                        edge_types = candidate.edge_type[indices]
                        # neighbors = neighbors[(candidate.x[neighbors] != 0).any(dim=1) & (candidate.x[neighbors] != 1).any(dim=1)]
                        candidate_two_hop_nodes.append(neighbors)
                        candidate_two_hop_edge_types.append(edge_types)
                    
                    # Flatten and make unique
                    candidate_two_hop_nodes = torch.cat(candidate_two_hop_nodes)
                    candidate_two_hop_edge_types = torch.cat(candidate_two_hop_edge_types)
                    candidate_two_hop_group_count = self.count_nodes_by_group(candidate_two_hop_nodes, candidate_two_hop_edge_types, candidate)
                
                    # target의 three-hop 이웃 찾기
                    target_three_hop_nodes = []
                    target_three_hop_edge_types = []


                    for node in target_two_hop_nodes.unique():
                        # 각 two-hop 노드의 이웃 노드를 찾기
                        indices = (target.edge_index[0] == node) | (target.edge_index[1] == node)
                        edges = target.edge_index[:, indices]
                        # node가 edges의 첫 번째 열에 있으면 두 번째 열의 값을, 두 번째 열에 있으면 첫 번째 열의 값을 추출
                        neighbors = torch.where(edges[0] == node, edges[1], edges[0]) # 조건, 참, 거짓
                        # neighbors = edges[edges != node]

                        edge_types = target.edge_type[indices]
                        # neighbors = neighbors[(target.x[neighbors] != 0).any(dim=1) & (target.x[neighbors] != 1).any(dim=1)]
                        target_three_hop_nodes.append(neighbors)
                        target_three_hop_edge_types.append(edge_types)

                    # Flatten and make unique
                    target_three_hop_nodes = torch.cat(target_three_hop_nodes)
                    target_three_hop_edge_types = torch.cat(target_three_hop_edge_types)
                    target_three_hop_group_count = self.count_nodes_by_group(target_three_hop_nodes, target_three_hop_edge_types, target)
                
                    # candidate의 three-hop 이웃 찾기
                    candidate_three_hop_nodes = []
                    candidate_three_hop_edge_types = []
                    
                    for node in candidate_two_hop_nodes.unique():
                        # 각 two-hop 노드의 이웃 노드를 찾기
                        indices = (candidate.edge_index[0] == node) | (candidate.edge_index[1] == node)
                        edges = candidate.edge_index[:, indices]
                        # node가 edges의 첫 번째 열에 있으면 두 번째 열의 값을, 두 번째 열에 있으면 첫 번째 열의 값을 추출
                        neighbors = torch.where(edges[0] == node, edges[1], edges[0]) # 조건, 참, 거짓
                        # neighbors = edges[edges != node]
                        
                        edge_types = candidate.edge_type[indices]
                        # neighbors = neighbors[(candidate.x[neighbors] != 0).any(dim=1) & (candidate.x[neighbors] != 1).any(dim=1)]
                        candidate_three_hop_nodes.append(neighbors)
                        candidate_three_hop_edge_types.append(edge_types)

                    # Flatten and make unique
                    candidate_three_hop_nodes = torch.cat(candidate_three_hop_nodes)
                    candidate_three_hop_edge_types = torch.cat(candidate_three_hop_edge_types)
                    candidate_three_hop_group_count = self.count_nodes_by_group(candidate_three_hop_nodes, candidate_three_hop_edge_types, candidate)
                



                    
                    # # one-hop과 two-hop 노드들 모두 합침
                    # target_all_hop_nodes = torch.cat([target_one_hop_nodes, target_two_hop_nodes]).unique()
                    # candidate_all_hop_nodes = torch.cat([candidate_one_hop_nodes, candidate_two_hop_nodes]).unique()

                    # # target과 candidate의 이웃 노드들을 그룹화하고 각 그룹의 카운트를 계산
                    # target_group_count = self.count_nodes_by_group(target_all_hop_nodes, target)
                    # candidate_group_count = self.count_nodes_by_group(candidate_all_hop_nodes, candidate)

                    # # target_group_count가 candidate_group_count에 포함되는지 확인
                    # is_included = all(candidate_group_count[group] >= count for group, count in target_group_count.items())

                    # pdb.set_trace()
                    # target과 candidate의 연결된 노드들을 그룹화하고 각 그룹의 카운트를 계산
                    # target_one_hop_group_count = self.count_nodes_by_group(target_one_hop_nodes, target_connected_edge_types, target) #, target_connected_edges , target_one_hop_edge_types)
                    # candidate_one_hop_group_count = self.count_nodes_by_group(candidate_one_hop_nodes, candidate) #, candidate_connected_edges, candidate.edge_type)
                    # target_two_hop_group_count = self.count_nodes_by_group(target_two_hop_nodes, target)
                    # candidate_two_hop_group_count = self.count_nodes_by_group(candidate_two_hop_nodes, candidate)
                    # target_three_hop_group_count = self.count_nodes_by_group(target_three_hop_nodes, target)
                    # candidate_three_hop_group_count = self.count_nodes_by_group(candidate_three_hop_nodes, candidate)

                    # pdb.set_trace()
                    # target_group_count가 candidate_group_count에 포함되는지 확인
                    # pdb.set_trace()
                    one_hop_is_included = self.is_group_count_included(target_one_hop_group_count, candidate_one_hop_group_count)
                    two_hop_is_included = self.is_group_count_included(target_two_hop_group_count, candidate_two_hop_group_count)
                    three_hop_is_included = self.is_group_count_included(target_three_hop_group_count, candidate_three_hop_group_count)

                    # one_hop_is_included = all(candidate_one_hop_group_count[group] >= count for group, count in target_one_hop_group_count.items())
                    # two_hop_is_included = all(candidate_two_hop_group_count[group] >= count for group, count in target_two_hop_group_count.items())
                    # three_hop_is_included = all(candidate_three_hop_group_count[group] >= count for group, count in target_three_hop_group_count.items())


                    if subg_idx1 and subg_idx2:
                        target_is_connected = any(
                            ((target.edge_index[0] == row_idx) & (target.edge_index[1] == prev_idx)).any().item() |
                            ((target.edge_index[1] == row_idx) & (target.edge_index[0] == prev_idx)).any().item()
                            for prev_idx in subg_idx1 
                            if (target.x[prev_idx] == 1).nonzero(as_tuple=True)[0].item() not in [0, 1]
                        )

                        candidate_is_connected = any(
                            ((candidate.edge_index[0] == col_idx) & (candidate.edge_index[1] == prev_idx)).any().item() |
                            ((candidate.edge_index[1] == col_idx) & (candidate.edge_index[0] == prev_idx)).any().item()
                            for prev_idx in subg_idx2
                            if (candidate.x[prev_idx] == 1).nonzero(as_tuple=True)[0].item() not in [0, 1] 
                        )


                        # candidate_is_connected = any(
                        #     torch.any((candidate.edge_index[0] == col_idx) & (candidate.edge_index[1] == prev_idx)).item() |
                        #     torch.any((candidate.edge_index[1] == col_idx) & (candidate.edge_index[0] == prev_idx)).item()
                        #     for prev_idx in subg_idx2
                        # )
                        
                    else:
                        target_is_connected = True  # 첫 번째 선택은 연결 확인이 필요 없음
                        candidate_is_connected = True

                    # if cand_idx == 2:
                        # if row_idx == 0 and  raw_candidates[cand_idx][0][col_idx] ==  6:
                        #     pdb.set_trace()
                        # print('row_idx = ', row_idx, ', raw_candidates[cand_idx][0][col_idx] = ', raw_candidates[cand_idx][0][col_idx])
                    
                    # if row_idx == 5 and raw_candidates[cand_idx][0][col_idx] == 33:
                    #     pdb.set_trace()

                    # 그룹화된 feature 비교
                    if target_group == candidate_group and one_hop_is_included and two_hop_is_included and three_hop_is_included:
                        if target_is_connected and candidate_is_connected:
                            # pdb.set_trace()
                            print('row_idx = ', row_idx, ', col_idx = ', col_idx, ', raw_idx = ', raw_candidates[cand_idx][0][col_idx])

                            # if row_idx == 22:
                            #     pdb.set_trace()
                            # pdb.set_trace()
                            
                            # print('target_group_count = ', target_group_count)
                            # print('candidate_group_count = ', candidate_group_count)
                            found_match = True   
                            # pdb.set_trace()
                        else:
                            distance[row_idx, col_idx] = 1e9
                            disconnected_pairs.append((row_idx, col_idx))
                            # print('row_idx = ', row_idx)
                                      

                    else:
                        # 동일하지 않으면, 해당 쌍을 제외하고 다음 최소 거리 쌍으로 진행
                        distance[row_idx, col_idx] = 1e9

                        if (distance == 1.0000e+09).all():
                            break
                        # if (distance[1] == 1000000000).all():
                        #     pdb.set_trace()
                
                ### 여기까지가 while 문 끝
                # print(row_idx)


                # if (distance[5] == 1000000000).all():
                #     pdb.set_trace()
                
                subg_idx1.append(row_idx)
                subg_idx2.append(col_idx)

                for r_idx, c_idx in disconnected_pairs:
                    if distance[r_idx, c_idx] == 1e9:
                        # 다시 원래의 거리를 계산해서 복구
                        distance[r_idx, c_idx] = torch.cdist(target_node_features[r_idx].unsqueeze(0), candidate_features[c_idx].unsqueeze(0)).item()

                if initial_skipped_pairs:
                    for r_idx, c_idx in initial_skipped_pairs:
                        if distance[r_idx, c_idx] == 1e9:
                            # 다시 원래의 거리를 계산해서 복구
                            distance[r_idx, c_idx] = torch.cdist(target_node_features[r_idx].unsqueeze(0), candidate_features[c_idx].unsqueeze(0)).item()

                # 일치하는 경우에만 업데이트
                distance[row_idx] = 1e9
                distance[:, col_idx] = 1e9

                # 복원 후 리스트 초기화
                disconnected_pairs = []
                initial_skipped_pairs = []


            #     # 중복 확인 후 다음 cand_idx로 넘어감
            #     if len(subg_idx1) != len(set(subg_idx1)) or len(subg_idx2) != len(set(subg_idx2)):
            #         break  # 내부 while 루프 탈출

            # # 중복이 발생하여 break로 탈출했다면, 다음 cand_idx로 넘어감
            # if len(subg_idx1) != len(set(subg_idx1)) or len(subg_idx2) != len(set(subg_idx2)):
            #     # print(cand_idx)
            #     continue

            # pdb.set_trace()
            matching_nodes_list = [raw_candidates[cand_idx][0][i] for i in subg_idx2]
            matching_nodes_set = set(matching_nodes_list)
            
            candidate_nodes_set = set(raw_candidates[cand_idx][0])

            # print('cand_idx = ', cand_idx)
            for idx, label in enumerate(node_labels):
                # print('label_idx = ', idx)
                # print('intersection = ', len(set(label).intersection(candidate_nodes_set)))
                # if set(label).issubset(candidate_nodes_set) and idx in label_idx_set:
                #     print('label_idx = ', idx)
                #     print('intersection = ', len(set(label).intersection(matching_nodes_set)))
                #     # label_idx_set.remove(idx)
                #     prediction += 1
                #     # pdb.set_trace()
                #     # break  

                if set(label) == matching_nodes_set and idx in label_idx_set: 
                    print('label_idx = ', idx)
                    print('intersection = ', len(set(label).intersection(matching_nodes_set)))
                    label_idx_set.remove(idx)
                    prediction += 1
                    break

            print("--------------------------------------------------")
            # pdb.set_trace()
                    
        total_positive_labels = len(node_labels)

            # recall 계산
        recall = prediction / total_positive_labels if total_positive_labels > 0 else 0.0

        print(recall)
        pdb.set_trace()

        return recall


    # def map_feature_to_group(self, one_hot_vector):
    #     # One-hot 벡터의 인덱스 추출
    #     feature_index = torch.argmax(one_hot_vector).item()
    #     index_group_map = {
    #         0: 0, 2: 0, 3: 0, 4: 0,  # 그룹 0
    #         1: 1, 5: 1,              # 그룹 1
    #         6: 2, 7: 2,              # 그룹 2
    #         8: 3,                    # 그룹 3
    #         9: 4,                    # 그룹 4
    #         10: 5,                   # 그룹 5
    #         11: 6,                   # 그룹 6
    #         12: 7,                   # 그룹 7
    #         13: 8                    # 그룹 8
    #     }
    #     # 그룹 매핑 적용
    #     return index_group_map[feature_index]


    # def count_nodes_by_group(self, nodes, data):
    #     # 각 노드의 그룹을 찾고 카운트 (target.x[row_idx] == 1).nonzero(as_tuple=True)[0].item()
    #     groups = [(data.x[node] == 1).nonzero(as_tuple=True)[0].item() for node in nodes] # [self.map_feature_to_group(data.x[node]) for node in nodes]
    #     return collections.Counter(groups)
    

    def count_nodes_by_group(self, nodes, edge_types, target):
        group_count = {}
        
        for node, edge_type in zip(nodes, edge_types):
            edge_type = edge_type.item()
            # Get the feature of the node (assuming 'x' holds the node features in target)
            node_feature = (target.x[node] == 1).nonzero(as_tuple=True)[0].item() # Assuming the node feature is scalar or use tuple(node_feature) if it's a vector
            
            # Initialize the edge type if not present in the dictionary
            if edge_type not in group_count:
                group_count[edge_type] = {}
            
            # Initialize the feature count if not present
            if node_feature not in group_count[edge_type]:
                group_count[edge_type][node_feature] = 0
            
            # Increment the count for this node feature
            group_count[edge_type][node_feature] += 1
        
        return group_count


    def is_group_count_included(self, target_group_count, candidate_group_count):
        # target_group_count가 candidate_group_count에 포함되는지 확인
        for edge_type, feature_counts in target_group_count.items():
            if edge_type not in candidate_group_count:
                return False
            
            # 각 feature와 해당 개수 확인
            for feature, count in feature_counts.items():
                if feature not in candidate_group_count[edge_type]:
                    return False
                if candidate_group_count[edge_type][feature] < count:
                    return False
        
        return True



    # def count_nodes_by_group(self, nodes, data):
    #     # 각 노드의 그룹을 찾고 카운트 (target.x[row_idx] == 1).nonzero(as_tuple=True)[0].item()
    #     groups = [(data.x[node] == 1).nonzero(as_tuple=True)[0].item() for node in nodes] # [self.map_feature_to_group(data.x[node]) for node in nodes]
    #     return collections.Counter(groups)



    # def count_nodes_by_group(self, nodes, data, edges, edge_types):
    #     # edge별로 구분된 이웃 노드 feature count
    #     edge_group_count = collections.defaultdict(lambda: collections.defaultdict(int))
        
    #     for node in nodes:
    #         # 각 노드에 연결된 edge 찾기
    #         connected_edge_indices = (edges[0] == node) | (edges[1] == node)
    #         connected_edges = edges[:, connected_edge_indices]
    #         pdb.set_trace()
    #         connected_edge_types = edge_types[connected_edge_indices]
            
    #         # 해당 노드의 feature group
    #         feature_group = (data.x[node] == 1).nonzero(as_tuple=True)[0].item()
            
    #         # 각 edge type별로 count
    #         for edge_type in connected_edge_types.unique():
    #             edge_group_count[edge_type.item()][feature_group] += 1

    #     return edge_group_count




    def bb_test(self):
    
    
        return 


    def get_graph_embedding(self, loader):
        for data in loader:
            data = data.to(self.device) 
            graph_embedding, node_features1, node_features2  = self.model(data.x, data.edge_index, data.edge_type, data.batch)
        return graph_embedding, node_features1, node_features2
