# -----------------------------------------------------------
# Dual Semantic Relations Attention Network (DSRAN) implementation
# "Learning Dual Semantic Relations with Graph Attention for Image-Text Matching"
# Keyu Wen, Xiaodong Gu, and Qingrong Cheng
# IEEE Transactions on Circuits and Systems for Video Technology, 2020
# Writen by Keyu Wen, 2020
# ------------------------------------------------------------

import math
import torch
from torch import nn
import torch.nn.functional as F
import copy

class GATopt(object):
    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = 8
        self.hidden_dropout_prob = 0.2
        self.attention_probs_dropout_prob = 0.2


class GAT(nn.Module):
    def __init__(self, config_gat):
        super(GAT, self).__init__()
        layer = GATLayer(config_gat)
        self.encoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(config_gat.num_layers)])

    def forward(self, input_graph, agl=None):
        hidden_states = input_graph  # input_graph就是resnet152提取的特征  128 49 1024    50,32,1024
        for layer_module in self.encoder:
            hidden_states = layer_module(hidden_states,agl)
        return hidden_states  # B, seq_len, D


class GAT_MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(GAT_MultiHeadAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.WGs = nn.ModuleList([nn.Linear(16, 1, bias=True) for _ in range(8)])#80

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # x是128 49 1024, 输出new_x_shape: 128 49 8 128
        x = x.view(*new_x_shape) # 128 49 8 128
        return x.permute(0, 2, 1, 3) # 128, 8, 49, 128

    def forward(self, input_graph, agl=None):
        nodes_q = self.query(input_graph) # 128 49 1024    50,32,1024
        nodes_k = self.key(input_graph) # 128 49 1024    50,32,1024
        nodes_v = self.value(input_graph)# 128 49 1024    50,32,1024

        nodes_q_t = self.transpose_for_scores(nodes_q) # 128 8 49 128; 50,8,32,128
        nodes_k_t = self.transpose_for_scores(nodes_k)# 128 8 49 128; 50,8,32,128
        nodes_v_t = self.transpose_for_scores(nodes_v)# 128 8 49 128; 50,8,32,128

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(nodes_q_t, nodes_k_t.transpose(-1, -2)) # Q 乘以K的转置。nodes_k_t.transpose(-1, -2)结果为128 8 128 49。最后输出128 8 49 49
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # 128 8 49 49; 50,8,32,32
        # Apply the attention mask is (precomputed for all layers in GATModel forward() function)
        attention_scores = attention_scores

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs) # 128 8 49 49
        # view相当于reshape，重新定义矩阵的形状
        nodes_new = torch.matmul(attention_probs, nodes_v_t) # 128 8 49 128;50,8,32,128
        nodes_new = nodes_new.permute(0, 2, 1, 3).contiguous() # 128 49 8 128; 50,32,8,128
        new_nodes_shape = nodes_new.size()[:-2] + (self.all_head_size,) #  50,32,1024    第一项结果是128, 49, 第二项是1024。最后结果是128,49,1024
        nodes_new = nodes_new.view(*new_nodes_shape) # 128 49 1024;
        return nodes_new


class GATLayer(nn.Module):
    def __init__(self, config):
        super(GATLayer, self).__init__()
        self.mha = GAT_MultiHeadAttention(config)

        self.fc_in = nn.Linear(config.hidden_size, config.hidden_size)
        self.bn_in = nn.BatchNorm1d(config.hidden_size)
        self.dropout_in = nn.Dropout(config.hidden_dropout_prob)

        self.fc_int = nn.Linear(config.hidden_size, config.hidden_size)

        self.fc_out = nn.Linear(config.hidden_size, config.hidden_size)
        self.bn_out = nn.BatchNorm1d(config.hidden_size)
        self.dropout_out = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_graph, agl=None):
        attention_output = self.mha(input_graph, agl) # multi-head attention   128 49 1024
        attention_output = self.fc_in(attention_output) # 经过1024 1024fc层，得 128 49 1024
        attention_output = self.dropout_in(attention_output) # 128 49 1024
        attention_output = self.bn_in((attention_output + input_graph).permute(0, 2, 1)).permute(0, 2, 1) # 128 49 1024
        # attention_output=attention_output + input_graph
        # attention_output = F.relu(attention_output+input_graph)




        # intermediate_output = self.fc_int(attention_output)# 128 49 1024
        # intermediate_output = F.relu(intermediate_output)# 128 49 1024
        # intermediate_output = self.fc_out(intermediate_output)# 128 49 1024
        # intermediate_output = self.dropout_out(intermediate_output)# 128 49 1024
        # graph_output = self.bn_out((intermediate_output + attention_output).permute(0, 2, 1)).permute(0, 2, 1)# 128 49 1024
        # return graph_output
        return attention_output