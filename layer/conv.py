import math
from typing import Union, Tuple, Optional
from torch import Tensor, cat
from torch.nn import init, Parameter, Linear, LayerNorm
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.typing import OptPairTensor, Adj, OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from utils import ccorr
from torch_scatter import scatter_mean, scatter_max
import torch
import os
import torch.nn as nn   
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



from layer.feature_attention import FeatureDimSelfAttention, AllDimAttention, CrossFeatureAttention, FeatureDimSelfAttentionNoFFN

class HypergraphTransformer(MessagePassing):
    """
    Hypergraph Transformer layer, including relation transformation, edge fusion (including time fusion), self-attention mechanism and gated residual connections (or skip connections).
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        attn_heads: int = 4,
        residual_beta: Optional[float] = None,
        learn_beta: bool = False,
        dropout: float = 0.,
        negative_slope: float = 0.2,
        bias: bool = True,
        trans_method: str = 'add',
        edge_fusion_mode: str = 'add',
        time_fusion_mode: str = None,
        head_fusion_mode: str = 'concat',
        residual_fusion_mode: str = None,
        edge_dim: int = None,
        rel_embed_dim: int = None,
        time_embed_dim: int = 0,
        dist_embed_dim: int = 0,
        normalize: bool = True,
        message_mode: str = 'node_edge',
        have_query_feature: bool = False,
        feature_attention_dropout: float=0.1,
        feature_residual_alpha: float=0.5,
        **kwargs
    ):
        # Initialize MessagePassing's aggregation method and node dimension
        super(HypergraphTransformer, self).__init__(aggr='add', node_dim=0, **kwargs)

        # Set input/output channels and number of heads
        self.in_channels = in_channels   # in out channels: embedding_dim
        self.out_channels = out_channels
        self.attn_heads = attn_heads

        # Set whether to learn residual weight beta
        self.learn_beta = learn_beta
        self.residual_beta = residual_beta

        # Set dropout probability, LeakyReLU negative slope, etc.
        self.dropout = dropout
        self.negative_slope = negative_slope

        # Relation transformation, edge fusion, time fusion, head fusion modes, etc.
        self.trans_method = trans_method
        self.edge_fusion_mode = edge_fusion_mode
        self.time_fusion_mode = time_fusion_mode
        self.head_fusion_mode = head_fusion_mode
        self.residual_fusion_mode = residual_fusion_mode

        # Set edge dimension, time embedding dimension, distance embedding dimension, etc.
        self.time_embed_dim = time_embed_dim
        self.dist_embed_dim = dist_embed_dim
        if self.time_fusion_mode == 'concat':
            self.rel_embed_dim = rel_embed_dim + self.time_embed_dim + self.dist_embed_dim
            self.edge_dim = edge_dim + self.time_embed_dim + self.dist_embed_dim
        else:
            self.rel_embed_dim = rel_embed_dim
            self.edge_dim = edge_dim

        # Control whether to normalize output
        self.normalize = normalize

        # Message passing mode (node-to-edge message passing or other methods)
        self.message_mode = message_mode

        self.trans_flag = False # Whether linear transformation is needed

        # Whether to include query features
        self.have_query_feature = have_query_feature

        # If input channels is integer, convert to tuple form
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
            self.in_channels = in_channels

        # If input and output channels are different and using additive transformation, linear transformation is needed
        if in_channels[0] != out_channels and self.trans_method == 'add':
            self.trans_flag = True
            self.lin_trans_x = Linear(in_channels[0], in_channels[1])

        # Define self-attention weight parameters
        if not self.have_query_feature:
            self.att_r = Parameter(Tensor(1, attn_heads, out_channels))

        # Check attention input and output dimensions
        self.attn_in_dim, self.attn_out_dim = self._check_attn_dim(in_channels[1], out_channels)

        # # Define linear transformation layers for key, query, value
        # self.lin_key = Linear(self.attn_in_dim, attn_heads * out_channels)
        # self.lin_query = Linear(in_channels[1], attn_heads * out_channels)
        # if self.message_mode == 'node_edge':
        #     self.lin_value = Linear(self.attn_in_dim, attn_heads * out_channels)
        # else:
        #     self.lin_value = Linear(in_channels[1], attn_heads * out_channels)

        # Define FFN (feed-forward network) or skip connection layers under different residual fusion modes
        if self.residual_fusion_mode == 'concat':
            self.lin_ffn_0 = Linear(in_channels[1] + self.attn_out_dim, out_channels + 128)
            self.lin_ffn_1 = Linear(out_channels + 128, out_channels)
        elif residual_fusion_mode == 'add':
            if head_fusion_mode == 'concat':
                self.lin_ffn_1 = Linear(attn_heads * out_channels, out_channels, bias=bias)
                self.lin_skip = Linear(in_channels[0], attn_heads * out_channels, bias=bias)
                if learn_beta:
                    self.lin_beta = Linear(3 * attn_heads * out_channels, 1, bias=False)
            else:
                self.lin_skip = Linear(in_channels[0], out_channels, bias=bias)
                if learn_beta:
                    self.lin_beta = Linear(3 * out_channels, 1, bias=False)
        else:
            self.lin_ffn_0 = Linear(self.attn_out_dim, out_channels + 128)
            self.lin_ffn_1 = Linear(out_channels + 128, out_channels)
            if self.head_fusion_mode == 'add':
                self.layer_norm = LayerNorm(out_channels)
            else:
                self.layer_norm = LayerNorm(out_channels * attn_heads)

        # Define feature dimension attention mechanism
        feature_num=6
        if self.time_fusion_mode == 'concat' and self.time_embed_dim and self.dist_embed_dim:
            feature_num+=2
        if self.edge_fusion_mode == 'concat' and self.edge_dim is not None:
            feature_num+=1
        # self.feature_attention = FeatureDimSelfAttention(128*feature_num, 128, feature_residual_alpha, feature_attention_dropout, ffn=True)
        # self.feature_attention_source = FeatureDimSelfAttention(128*feature_num, 128, 
        #                                                         feature_residual_alpha, 
        #                                                         feature_attention_dropout, 
        #                                                         ffn=False)
        # self.feature_attention_target = FeatureDimSelfAttention(128*feature_num, 128, 
        #                                                         feature_residual_alpha, 
        #                                                         feature_attention_dropout, 
        #                                                         ffn=False)
        
        self.feature_attention = FeatureDimSelfAttentionNoFFN(128*feature_num, 128, 
                                                                feature_residual_alpha, 
                                                                feature_attention_dropout)
        # # self.feature_attention = AllDimAttention(out_channels, 1, feature_residual_alpha, feature_attention_dropout)

        # Define cross feature attention
        # self.feature_attention = CrossFeatureAttention(128, feature_attention_dropout, feature_residual_alpha)

        # Low-rank projection matrices
        rank = 256
        self.U_key = nn.Parameter(torch.randn(self.attn_in_dim, rank))  # [attn_in_dim, r]
        self.V_key = nn.Parameter(torch.randn(rank, attn_heads * out_channels))  # [r, heads*out_channels]
        
        if have_query_feature:
            self.U_query = nn.Parameter(torch.randn(in_channels[1], rank))  # [in_channels[1], r]
            self.V_query = nn.Parameter(torch.randn(rank, attn_heads * out_channels))  # [r, heads*out_channels]
        
        if self.message_mode == 'node_edge':
            self.U_value = nn.Parameter(torch.randn(self.attn_in_dim, rank))  # [attn_in_dim, r]
        else:
            self.U_value = nn.Parameter(torch.randn(in_channels[1], rank))  # [in_channels[1], r]
        self.V_value = nn.Parameter(torch.randn(rank, attn_heads * out_channels))  # [r, heads*out_channels]

        # self.W_residual = nn.Parameter(torch.randn(in_channels[1], attn_heads * out_channels))
        
        # Parameter initialization
        nn.init.xavier_uniform_(self.U_key)
        nn.init.xavier_uniform_(self.V_key)
        if have_query_feature:
            nn.init.xavier_uniform_(self.U_query)
            nn.init.xavier_uniform_(self.V_query)
        nn.init.xavier_uniform_(self.U_value)
        nn.init.xavier_uniform_(self.V_value)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Reset model parameters"""
        if self.trans_flag:
            self.lin_trans_x.reset_parameters()
        # self.lin_key.reset_parameters()
        # self.lin_query.reset_parameters()
        # self.lin_value.reset_parameters()
        if not self.have_query_feature:
            init.xavier_uniform_(self.att_r)
        if self.residual_fusion_mode == 'add':
            self.lin_skip.reset_parameters()
            if self.head_fusion_mode == 'concat':
                self.lin_ffn_1.reset_parameters()
            if self.learn_beta:
                self.lin_beta.reset_parameters()
        else:
            self.lin_ffn_0.reset_parameters()
            self.lin_ffn_1.reset_parameters()
            if not self.residual_fusion_mode:
                self.layer_norm.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr_embed: Tensor,
        edge_time_embed: Tensor,
        edge_dist_embed: Tensor,
        edge_type_embed: Tensor,
    ):
        """
        Forward propagation function
        x: node features + target features, for ci2traj is node features + target node features, for traj2traj is node features + traj features
        edge_index: edge index, containing source, target node indices and edge indices
        edge_time_embed: time embedding
        edge_dist_embed: distance embedding
        edge_type_embed: type embedding
        edge_attr_embed: edge attribute embedding, None for ci2traj, source size, target size, similarity for traj2traj
        """
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)  #  shape = [N, in_channels]

        # Message passing based on edge index type, if it's sparse tensor, it means
        if isinstance(edge_index, SparseTensor):
            out = self.propagate(
                edge_index,
                x=(x[0][edge_index.storage.col()], x[1][edge_index.storage.row()]),
                edge_attr_embed=edge_attr_embed,
                edge_time_embed=edge_time_embed,
                edge_dist_embed=edge_dist_embed,
                edge_type_embed=edge_type_embed,
                have_query_feature=self.have_query_feature,
                size=None
            )  # shape = [E, attn_heads, out_channels]
        else:
            out = self.propagate(
                edge_index,
                x=(x[0][edge_index[0]], x[1][edge_index[1]]),
                edge_attr_embed=edge_attr_embed,
                edge_time_embed=edge_time_embed,
                edge_dist_embed=edge_dist_embed,
                edge_type_embed=edge_type_embed,
                have_query_feature=self.have_query_feature,
                size=None
            )

        # If no query features, add self-attention weights
        if not self.have_query_feature:
            out += self.att_r  # shape = [N, attn_heads, out_channels]

        # Adjust output based on head fusion mode
        if self.head_fusion_mode == 'concat':
            out = out.view(-1, self.attn_heads * self.out_channels)  # shape = [N, attn_heads * out_channels]
        else:
            out = out.mean(dim=1)  # shape = [N, out_channels]

        # Handle different residual fusion modes
        if self.residual_fusion_mode == 'concat':
            out = cat([out, x[1]], dim=-1)   # shape = [N, out_channels + in_channels]
            out = self.lin_ffn_0(out)    
            out = F.relu(out)
            out = self.lin_ffn_1(out)   # shape = [N, out_channels]
        elif self.residual_fusion_mode == 'add':
            x_skip = self.lin_skip(x[1]) 

            # If learning residual weight beta
            if self.learn_beta:
                beta = self.lin_beta(cat([out, x_skip, out - x_skip], -1))
                beta = beta.sigmoid()
                out = beta * x_skip + (1 - beta) * out
            else:
                if self.residual_beta is not None:
                    out = self.residual_beta * x_skip + (1 - self.residual_beta) * out
                else:
                    out += x_skip
            if self.head_fusion_mode == 'concat':
                out = self.lin_ffn_1(out)
        else:
            out = self.layer_norm(out)
            out = self.lin_ffn_0(out)
            out = F.relu(out)
            out = self.lin_ffn_1(out)

        # # Apply feature-wise attention to out
        # out,weight = self.feature_attention(out)

        # Normalize if needed
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out  # shape = [N, out_channels]

    def message_original(
        self,
        x: OptPairTensor,
        edge_attr_embed: Tensor,
        edge_time_embed: Tensor,
        edge_dist_embed: Tensor,
        edge_type_embed: Tensor,
        index: Tensor,
        ptr: OptTensor,
        have_query_feature: bool,
        size_i: Optional[int]
    ) -> Tensor:
        x_j, x_i = x  #shape = [E, out_channels]
        # Check if x_j and x_i are all zeros
        # print(torch.sum(x_j),torch.sum(x_i))

        
        """
        Message passing function, including key, query, value computation and edge fusion operations.
        """
        # In graph convolution, information is passed along edges, x_i is target node features, x_j is source node features, so their dimensions are [E, out_channels]

        # x_i,_ = self.feature_attention(x_i)  # shape = [E, out_channels]
        # x_j,_ = self.feature_attention(x_j)  # shape = [E, out_channels]

        # If relation transformation is needed, perform linear transformation
        if self.trans_flag:
            if have_query_feature:
                x_i = self.lin_trans_x(x_i)
            x_j_raw = self.lin_trans_x(x_j)
            x_j = self.lin_trans_x(x_j)
        else:
            x_j_raw = x_j       # shape = [E, out_channels]

        # Edge type embedding fusion
        if edge_type_embed is not None:
            x_j = self.rel_transform(x_j, edge_type_embed)

        # Edge attribute embedding fusion
        if edge_attr_embed is not None:
            if self.edge_fusion_mode == 'concat':
                x_j = cat([x_j, edge_attr_embed], dim=-1)
            else:
                x_j += edge_attr_embed

        # Edge time and distance embedding fusion
        if self.time_fusion_mode == 'concat':
            x_j = cat([x_j, edge_time_embed, edge_dist_embed], dim=-1)
        elif self.time_fusion_mode == 'add':
            # x_j += edge_time_embed + edge_dist_embed
            x_j = x_j + edge_time_embed + edge_dist_embed

        # Add feature dimension attention mechanism
        # x_j,_ = self.feature_attention(x_j)  # shape = [E, out_channels]
        # x_j, _ = self.feature_attention(x_i, x_j)  # shape = [E, out_channels]

        # x_i,_ = self.feature_attention_target(x_i)  # shape = [E, out_channels]
        # x_j,_ = self.feature_attention_source(x_j)  # shape = [E, out_channels]
        if have_query_feature:
            x_i,_ = self.feature_attention(x_i)  # shape = [E, out_channels]
        x_j,_ = self.feature_attention(x_j)  # shape = [E, out_channels]


        # Compute key and query, and calculate attention weights
        key = self.lin_key(x_j).view(-1, self.attn_heads, self.out_channels)   # shape = [E, attn_heads, out_channels]
        if not have_query_feature:
            query = self.att_r  # shape = [1, attn_heads, out_channels]
            alpha = (key * query).sum(dim=-1)    # shape = [E, attn_heads]
            alpha = F.leaky_relu(alpha, self.negative_slope)  # shape = [E, attn_heads]
        else:
            query = self.lin_query(x_i).view(-1, self.attn_heads, self.out_channels) # shape = [E, attn_heads, out_channels]
            alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels) # shape = [E, attn_heads]

        # print("Index:", index.cpu())
        # print("Ptr:", ptr.cpu())
        # print("Alpha:", alpha.cpu())

        # Perform softmax normalization and dropout
        alpha = softmax(alpha, index, ptr, size_i)  # shape = [E, attn_heads]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training) # shape = [E, attn_heads]

        # Compute output based on message passing mode
        if self.message_mode == 'node_edge':
            out = self.lin_value(x_j).view(-1, self.attn_heads, self.out_channels)  # shape = [E, attn_heads, out_channels]
        else:
            out = self.lin_value(x_j_raw).view(-1, self.attn_heads, self.out_channels)  # shape = [E, attn_heads, out_channels]

        out *= alpha.view(-1, self.attn_heads, 1)  # shape = [E, attn_heads, out_channels]
        return out
    
    def message(
        self,
        x: OptPairTensor,
        edge_attr_embed: Tensor,
        edge_time_embed: Tensor,
        edge_dist_embed: Tensor,
        edge_type_embed: Tensor,
        index: Tensor,
        ptr: OptTensor,
        have_query_feature: bool,
        size_i: Optional[int]
    ) -> Tensor:
        x_j, x_i = x  # shape = [E, out_channels]

        if self.trans_flag:
            if have_query_feature:
                x_i = self.lin_trans_x(x_i)
            x_j_raw = self.lin_trans_x(x_j)
            x_j = self.lin_trans_x(x_j)
        else:
            x_j_raw = x_j

        # Edge type embedding fusion
        if edge_type_embed is not None:
            x_j = self.rel_transform(x_j, edge_type_embed)

        # Edge attribute embedding fusion
        if edge_attr_embed is not None:
            x_j = cat([x_j, edge_attr_embed], dim=-1) if self.edge_fusion_mode == 'concat' else x_j + edge_attr_embed

        # Edge time and distance embedding fusion
        if self.time_fusion_mode == 'concat':
            x_j = cat([x_j, edge_time_embed, edge_dist_embed], dim=-1)
        elif self.time_fusion_mode == 'add':
            x_j = x_j + edge_time_embed + edge_dist_embed

        if have_query_feature:
            x_i, _ = self.feature_attention(x_i)
        x_j, _ = self.feature_attention(x_j)

        # Low-rank projection computation for Key
        low_rank_key = torch.matmul(x_j, self.U_key)  # [E, r]
        key = torch.matmul(low_rank_key, self.V_key)  # [E, heads*out_channels]
        key = F.layer_norm(key, key.shape[-1:])  # normalization
        key = key.view(-1, self.attn_heads, self.out_channels)  # [E, heads, out_channels]
        
        if not have_query_feature:
            query = self.att_r  # [1, heads, out_channels]
            alpha = (key * query).sum(dim=-1)  # [E, heads]
            alpha = F.leaky_relu(alpha, self.negative_slope)
        else:
            # Low-rank projection computation for Query
            low_rank_query = torch.matmul(x_i, self.U_query)  # [E, r]
            query = torch.matmul(low_rank_query, self.V_query)  # [E, heads*out_channels]
            query = F.layer_norm(query, query.shape[-1:])  # normalization
            query = query.view(-1, self.attn_heads, self.out_channels)  # [E, heads, out_channels]
            alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)  # [E, heads]
        
        # softmax normalization and dropout
        alpha = softmax(alpha, index, ptr, size_i)  # [E, heads]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Low-rank projection computation for Value
        low_rank_value = torch.matmul(x_j, self.U_value)  # [E, r]
        out = torch.matmul(low_rank_value, self.V_value)  # [E, heads*out_channels]
        out = F.layer_norm(out, out.shape[-1:])  # normalization
        out = out.view(-1, self.attn_heads, self.out_channels)  # [E, heads, out_channels]
        
        out = out * alpha.unsqueeze(-1)  # [E, heads, out_channels]
        return out

    
    def message_debug(
        self,
        x: OptPairTensor,
        edge_attr_embed: Tensor,
        edge_time_embed: Tensor,
        edge_dist_embed: Tensor,
        edge_type_embed: Tensor,
        index: Tensor,
        ptr: OptTensor,
        have_query_feature: bool,
        size_i: Optional[int]
    ) -> Tensor:
        x_j, x_i = x  # shape = [E, out_channels]

        if self.trans_flag:
            if have_query_feature:
                x_i = self.lin_trans_x(x_i)
            x_j_raw = self.lin_trans_x(x_j)
            x_j = self.lin_trans_x(x_j)
        else:
            x_j_raw = x_j  # shape = [E, out_channels]

        # Edge type embedding fusion
        if edge_type_embed is not None:
            x_j = self.rel_transform(x_j, edge_type_embed)

        # Edge attribute embedding fusion
        if edge_attr_embed is not None:
            x_j = cat([x_j, edge_attr_embed], dim=-1) if self.edge_fusion_mode == 'concat' else x_j + edge_attr_embed

        # Edge time and distance embedding fusion
        if self.time_fusion_mode == 'concat':
            x_j = cat([x_j, edge_time_embed, edge_dist_embed], dim=-1)
        elif self.time_fusion_mode == 'add':
            x_j = x_j + edge_time_embed + edge_dist_embed

        if have_query_feature:
            x_i, _ = self.feature_attention(x_i)  # shape = [E, out_channels]
        x_j, _ = self.feature_attention(x_j)  # shape = [E, out_channels]

        # Compute key and query
        key = self.lin_key(x_j).view(-1, self.attn_heads, self.out_channels)  # [E, heads, channels]
        if not have_query_feature:
            query = self.att_r  # [1, heads, channels]
            alpha = (key * query).sum(dim=-1)  # [E, heads]
            alpha = F.leaky_relu(alpha, self.negative_slope)
        else:
            query = self.lin_query(x_i).view(-1, self.attn_heads, self.out_channels)
            alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)  # [E, heads]

        # ✅ **Improvement 1: Check empty alpha to prevent `IndexError`**
        if ptr is not None:
            sparse_alpha = []
            empty_nodes = 0
            total_nodes = len(ptr) - 1

            for i in range(len(ptr) - 1):
                start_idx, end_idx = ptr[i], ptr[i + 1]
                curr_alpha = alpha[start_idx:end_idx]  # [num_neighbors, heads]

                if curr_alpha.size(0) == 0:
                    # If current node has no neighbors, fill with a very small value to prevent `topk()` failure
                    # print(f"Node {i} has no neighbors")
                    # if isinstance(edge_attr_embed, torch.Tensor):  # If there are edge attributes, it means traj2traj layer
                    #     print("This is a trajectory node without neighbors")
                    # else:  # If there are no edge attributes, it means ci2traj layer
                    #     print("This is a check-in node without neighbors")
                    # curr_alpha = torch.full((1, self.attn_heads), -9e15, device=alpha.device)
                    
                    empty_nodes += 1
                    # Skip nodes without neighbors, don't add to sparse_alpha
                    continue

                # Calculate adaptive threshold
                threshold = curr_alpha.mean() + curr_alpha.std()
                mask = curr_alpha >= threshold  # Select high-weight parts

                # **Ensure each attention head retains at least one maximum value**
                for h in range(self.attn_heads):
                    if mask[:, h].sum() == 0:
                        _, top_idx = curr_alpha[:, h].max(dim=0, keepdim=True)
                        mask[top_idx, h] = True

                curr_alpha = curr_alpha.masked_fill(~mask, -9e15)
                sparse_alpha.append(curr_alpha)

            # empty_nodes = sum(1 for i in range(len(ptr)-1) if ptr[i+1]-ptr[i]==0)
            # print(f"Number of empty nodes: {empty_nodes}")
            # print(f"Original alpha rows: {alpha.shape[0]}")
            # print(f"New rows: {empty_nodes} => Final rows: {alpha.shape[0]+empty_nodes}")

            # %% Print empty node information
            # if empty_nodes > 0:
            #     print(f"In current batch, total nodes: {total_nodes}, nodes without neighbors: {empty_nodes}, ratio: {empty_nodes/total_nodes:.2%}")
            #     if isinstance(edge_attr_embed, torch.Tensor):
            #         print("These are trajectory nodes")
            #     else:
            #         print("These are check-in nodes")
            alpha = torch.cat(sparse_alpha, dim=0)
        else:
            # If no `ptr`, perform sparsification based on `index`
            unique_targets = torch.unique(index)
            sparse_alpha = []
            for target in unique_targets:
                mask = index == target
                curr_alpha = alpha[mask]

                if curr_alpha.size(0) == 0:
                    # Handle nodes without connections, fill with default values
                    curr_alpha = torch.full((1, self.attn_heads), -9e15, device=alpha.device)

                # Calculate adaptive threshold
                threshold = curr_alpha.mean() + curr_alpha.std()
                mask = curr_alpha >= threshold  # Select high-weight parts

                # **Ensure at least one selection**
                for h in range(self.attn_heads):
                    if mask[:, h].sum() == 0:
                        _, top_idx = curr_alpha[:, h].max(dim=0, keepdim=True)
                        mask[top_idx, h] = True

                curr_alpha = curr_alpha.masked_fill(~mask, -9e15)
                sparse_alpha.append(curr_alpha)

            # print('shape of total sparse alpha:',torch.cat(sparse_alpha, dim=0).shape)

            alpha = torch.cat(sparse_alpha, dim=0)

        # Calculate original and retained edge counts for each node
        node_sizes = ptr[1:] - ptr[:-1]  # [num_nodes]
        original_edges = node_sizes.sum().item()
        kept_edges = (alpha != -9e15).sum(dim=1).float().mean().item()  # Average number of edges retained per head
        
        print(f"Sparsification statistics - Original edges: {original_edges}, "
            f"Retained edges: {kept_edges:.1f}, "
            f"Sparsity ratio: {kept_edges/original_edges:.2%}")

        # ✅ **Improvement 2: Add temperature scaling to prevent gradient vanishing**
        alpha = softmax(alpha / 0.1, index, ptr, size_i)  # 0.1 as temperature parameter
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Compute output values
        if self.message_mode == 'node_edge':
            out = self.lin_value(x_j).view(-1, self.attn_heads, self.out_channels)
        else:
            out = self.lin_value(x_j_raw).view(-1, self.attn_heads, self.out_channels)

        out *= alpha.view(-1, self.attn_heads, 1)
        return out

    def message_sparse(
        self,
        x: OptPairTensor,
        edge_attr_embed: Tensor,
        edge_time_embed: Tensor,
        edge_dist_embed: Tensor,
        edge_type_embed: Tensor,
        index: Tensor,
        ptr: OptTensor,
        have_query_feature: bool,
        size_i: Optional[int]
    ) -> Tensor:
        x_j, x_i = x  # shape = [E, out_channels]

        if self.trans_flag:
            if have_query_feature:
                x_i = self.lin_trans_x(x_i)
            x_j_raw = self.lin_trans_x(x_j)
            x_j = self.lin_trans_x(x_j)
        else:
            x_j_raw = x_j  # shape = [E, out_channels]

        # Edge type embedding fusion
        if edge_type_embed is not None:
            x_j = self.rel_transform(x_j, edge_type_embed)

        # Edge attribute embedding fusion
        if edge_attr_embed is not None:
            x_j = cat([x_j, edge_attr_embed], dim=-1) if self.edge_fusion_mode == 'concat' else x_j + edge_attr_embed

        # Edge time and distance embedding fusion
        if self.time_fusion_mode == 'concat':
            x_j = cat([x_j, edge_time_embed, edge_dist_embed], dim=-1)
        elif self.time_fusion_mode == 'add':
            x_j = x_j + edge_time_embed + edge_dist_embed

        if have_query_feature:
            x_i, _ = self.feature_attention(x_i)  # shape = [E, out_channels]
        x_j, _ = self.feature_attention(x_j)  # shape = [E, out_channels]

        # Compute key and query
        key = self.lin_key(x_j).view(-1, self.attn_heads, self.out_channels)  # [E, heads, channels]
        if not have_query_feature:
            query = self.att_r  # [1, heads, channels]
            alpha = (key * query).sum(dim=-1)  # [E, heads]
            alpha = F.leaky_relu(alpha, self.negative_slope)
        else:
            query = self.lin_query(x_i).view(-1, self.attn_heads, self.out_channels)
            alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)  # [E, heads]

        if ptr is not None:
            # Calculate neighbor count for each node
            node_sizes = ptr[1:] - ptr[:-1]  # [num_nodes]
            non_empty_mask = node_sizes > 0
            
            # Build node index
            node_idx = torch.arange(len(ptr) - 1, device=alpha.device)
            node_idx_repeat = torch.repeat_interleave(node_idx, node_sizes)  # [num_edges]

            # **Batch compute mean and std**
            means = scatter_mean(alpha, node_idx_repeat, dim=0, dim_size=len(ptr) - 1)  # [num_nodes, heads]
            squared_diff = (alpha - means[node_idx_repeat]) ** 2
            stds = torch.sqrt(scatter_mean(squared_diff, node_idx_repeat, dim=0, dim_size=len(ptr) - 1) + 1e-6)  # Avoid division by zero

            # **Calculate threshold**
            thresholds = means + stds  # [num_nodes, heads]
            edge_thresholds = thresholds[node_idx_repeat]  # [num_edges, heads]

            # **Calculate mask**
            mask = alpha >= edge_thresholds  # [num_edges, heads]

            # **Ensure each node's each head retains at least the maximum value**
            max_values, _ = scatter_max(alpha, node_idx_repeat, dim=0, dim_size=len(ptr) - 1)  # [num_nodes, heads]
            max_values[max_values == float('-inf')] = -9e15  # Avoid invalid values
            max_mask = alpha == max_values[node_idx_repeat]  # [num_edges, heads]
            mask = mask | max_mask

        else:
            # **Handle index method**
            index = index.clamp(max=alpha.size(0) - 1)  # Prevent out of bounds
            means = scatter_mean(alpha, index, dim=0)  # [num_targets, heads]
            squared_diff = (alpha - means[index]) ** 2
            stds = torch.sqrt(scatter_mean(squared_diff, index, dim=0) + 1e-6)  # Avoid division by zero

            # **Calculate threshold**
            thresholds = means[index] + stds[index]  # [num_edges, heads]
            mask = alpha >= thresholds

            # **Ensure each target node's each head retains at least the maximum value**
            max_values, _ = scatter_max(alpha, index, dim=0)  # [num_targets, heads]
            max_values[max_values == float('-inf')] = -9e15
            max_mask = alpha == max_values[index]
            mask = mask | max_mask

        # **Apply mask and perform softmax**
        alpha = alpha.masked_fill(~mask, -9e15)
        alpha = softmax(alpha / 0.1, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)


        # Compute output values
        if self.message_mode == 'node_edge':
            out = self.lin_value(x_j).view(-1, self.attn_heads, self.out_channels)
        else:
            out = self.lin_value(x_j_raw).view(-1, self.attn_heads, self.out_channels)

        out *= alpha.view(-1, self.attn_heads, 1)
        return out


    def rel_transform(self, ent_embed, edge_type_embed):
        """
        Relation transformation function, supporting multiple transformation methods such as correlation, subtraction, multiplication, addition and concatenation.
        """
        if self.trans_method == "corr":
            trans_embed = ccorr(ent_embed, edge_type_embed)
        elif self.trans_method == "sub":
            trans_embed = ent_embed - edge_type_embed
        elif self.trans_method == "multi":
            trans_embed = ent_embed * edge_type_embed
        elif self.trans_method == 'add':
            trans_embed = ent_embed + edge_type_embed
        elif self.trans_method == 'concat':
            trans_embed = cat([ent_embed, edge_type_embed], dim=1)
        else:
            raise NotImplementedError
        return trans_embed

    def _check_attn_dim(self, in_channels, out_channels):
        """
        Check if attention input and output dimensions match.
        """
        attn_in_dim = in_channels
        attn_out_dim = out_channels * self.attn_heads if self.head_fusion_mode == 'concat' else out_channels

        # Handle relation embedding transformation
        if self.trans_method == 'concat':
            attn_in_dim += self.rel_embed_dim
        else:
            assert attn_in_dim == self.rel_embed_dim, \
                "[HypergraphTransformer >> Translation Error] Node embedding dimension {} is not equal with relation " \
                "embedding dimension {} when you are using '{}' translation method" \
                ".".format(attn_in_dim, self.rel_embed_dim, self.trans_method)

        # Handle edge embedding transformation
        if self.edge_fusion_mode == 'concat' and self.edge_dim is not None:
            attn_in_dim += self.edge_dim
        else:
            if self.edge_dim is not None:
                assert attn_in_dim == self.edge_dim, \
                    "[HypergraphTransformer >> Edge Fusion Error] Edge embedding dimension {} is " \
                    "not equal with translation result embedding dimension {} when you are using '{}' " \
                    "edge fusion mode.".format(self.edge_dim, attn_in_dim, self.edge_fusion_mode)
        
                # Handle time embedding transformation
        if self.time_fusion_mode:
            if self.time_fusion_mode == 'concat':
                attn_in_dim += self.time_embed_dim + self.dist_embed_dim
            else:
                assert attn_in_dim == self.time_embed_dim, \
                    "[HypergraphTransformer >> Time Fusion Error] Time embedding dimension {} is " \
                    "not equal with edge fusion result embedding dimension {} when you are using '{}' " \
                    "time fusion mode.".format(self.time_embed_dim, attn_in_dim, self.time_fusion_mode)
                assert attn_in_dim == self.dist_embed_dim, \
                    "[HypergraphTransformer >> Time Fusion Error] Time embedding dimension {} is " \
                    "not equal with edge fusion result embedding dimension {} when you are using '{}' " \
                    "time fusion mode.".format(self.dist_embed_dim, attn_in_dim, self.time_fusion_mode)

        return attn_in_dim, attn_out_dim

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, attn_heads={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.attn_heads)
