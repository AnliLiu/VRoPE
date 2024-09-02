from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention,MultiHeadGeometryAttention
from ..resnet import resnet152
import numpy as np
from ..GAT.GAT import *
import copy
from models.transformer.utils import *
from models.transformer.grid_aug import BoxRelationalEmbedding

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadGeometryAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, input_gl=None, memory=None,
                geometry=None, isencoder=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights, input_gl=input_gl, memory=memory,
                         geometry=geometry, isencoder=isencoder)  # 10 36 512
        ff = self.pwff(att)  # 10 36 512
        return ff


class EncoderLayer_gl_lo(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=4, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer_gl_lo, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadGeometryAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, input_gl=None, memory=None,
                geometry=None,isencoder=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights, input_gl=input_gl, memory=memory,
                         geometry=geometry, isencoder=isencoder)  # 10 36 512
        ff = self.pwff(att)  # 10 36 512
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])

        # self.gl_layer = EncoderLayer_gl_lo(d_model, d_k, d_v, 4, d_ff, dropout,
        #                                    identity_map_reordering=identity_map_reordering,
        #                                    attention_module=None,
        #                                    attention_module_kwargs=attention_module_kwargs)
        #
        # self.lo_layer = EncoderLayer_gl_lo(d_model, d_k, d_v, 4, d_ff, dropout,
        #                                    identity_map_reordering=identity_map_reordering,
        #                                    attention_module=None,
        #                                    attention_module_kwargs=attention_module_kwargs)

        self.padding_idx = padding_idx

        self.WGs = nn.ModuleList([nn.Linear(80, 1, bias=True) for _ in range(h)])#80

    def forward(self, input, input_gl=None,rois=None,isencoder=None, attention_weights=None):
        #######################
        attention_mask_lo = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)

        out_lo=input
        out_gl=input_gl
        #######################
        memory = torch.matmul(out_lo, out_gl.permute(0, 2, 1)) / np.sqrt(self.d_model)
        # memory = torch.softmax(memory.sum(dim=-1), -2)
        memory = torch.softmax(memory, -2).sum(dim=-1)
        memory = 0
        #######################
        out = out_lo
        outs = []

        # grid geometry embedding
        relative_geometry_embeddings = BoxRelationalEmbedding(rois)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 80)#80
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [layer(flatten_relative_geometry_embeddings).view(box_size_per_head) for layer in self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        for l in self.layers:
            out = l(out, out, out, attention_mask_lo, attention_weights, input_gl=out_gl, memory=memory,
                    geometry=relative_geometry_weights, isencoder=isencoder)  # 10 36 512
            outs.append(out.unsqueeze(1))  # 10 1 36 512

        outs = torch.cat(outs, 1)  # 10 3 36 512 ;  3是三层编码器
        return outs, attention_mask_lo



class MemoryAugmentedEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(MemoryAugmentedEncoder, self).__init__(N, padding_idx, **kwargs)
        config_img_lo = GATopt(512, 1)
        config_img_gl = GATopt(512, 1)
        self.gat_1 = GAT(config_img_lo)  # local feature
        #self.gat_2 = GAT(config_img_gl)  # global feature
        self.AAP = nn.AdaptiveAvgPool2d(1)
        self.fc_lo = nn.Linear(256, self.d_model)  # 映射到512维度
        self.fc_gl = nn.Linear(2048, self.d_model)
        self.dropout_lo = nn.Dropout(p=self.dropout)
        self.dropout_gl = nn.Dropout(p=self.dropout)
        self.layer_norm_lo = nn.LayerNorm(self.d_model)
        self.layer_norm_gl = nn.LayerNorm(self.d_model)

    def forward(self, input, input_gl=None, rois=None, isencoder=None, attention_weights=None):
        # # sydney+UCM
        pooled_tensor = self.AAP(input)
        flatten_lo = torch.flatten(pooled_tensor, 2)
        lo = F.relu(self.fc_lo(flatten_lo))
        # lo = F.relu(self.fc_lo(input))
        lo = self.dropout_lo(lo)
        lo = self.layer_norm_lo(lo)  # 50,32,512

        gl = F.relu(self.fc_gl(input_gl))
        gl = self.dropout_gl(gl)
        gl = self.layer_norm_gl(gl)  # 50,196,512

        agl = rois[:, :, 5]

        lo = self.gat_1(lo, agl)  # 10 50 2048   50,32,1024


        rois = rois

        return super(MemoryAugmentedEncoder, self).forward(lo, input_gl=gl,rois=rois,isencoder=isencoder,
                                                           attention_weights=attention_weights)

