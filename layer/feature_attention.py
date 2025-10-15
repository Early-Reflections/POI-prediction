import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class HeadDimAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, residual_alpha, dropout):
        super(AllDimAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim_per_head = embed_dim // num_heads

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.residual_alpha = residual_alpha  

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)

        self.layer_norm.reset_parameters()


    def forward(self, x):
        # x: [N, embed_dim]
        query = self.query_proj(x).view(-1, self.num_heads, self.embed_dim_per_head)  # [N, num_heads, embed_dim_per_head]
        key = self.key_proj(x).view(-1, self.num_heads, self.embed_dim_per_head)      # [N, num_heads, embed_dim_per_head]
        value = self.value_proj(x).view(-1, self.num_heads, self.embed_dim_per_head)  # [N, num_heads, embed_dim_per_head]

        # Feature-wise attention
        attention_score = torch.einsum('bhd,bhd->bh', query, key) / math.sqrt(self.embed_dim_per_head)  # [N, num_heads]
        attention_weights = F.softmax(attention_score, dim=-1).unsqueeze(-1)  # [N, num_heads, 1]
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to value
        out = value * attention_weights  # [N, num_heads, embed_dim_per_head]
        out = out.view(x.size(0), -1)  # [N, embed_dim]

        residual = self.residual_alpha * x + (1 - self.residual_alpha) * out
        out = self.layer_norm(residual)

        out = F.relu(out)

        return out,attention_weights
    
class AllDimAttention(nn.Module):
    def __init__(self, embed_dim, num_head, residual_alpha, dropout):
        super(AllDimAttention, self).__init__()
        self.embed_dim = embed_dim

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.residual_alpha = residual_alpha  
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)

        self.layer_norm.reset_parameters()

    def forward(self, x):
        # x: [N, embed_dim]
        N = x.size(0)
        
        # Project query, key, and value to [N, embed_dim]
        query = self.query_proj(x)  # [N, embed_dim]
        key = self.key_proj(x)      # [N, embed_dim]
        value = self.value_proj(x)  # [N, embed_dim]

        # Compute attention scores
        attention_scores = torch.einsum('bd,be->bde', query, key) / math.sqrt(self.embed_dim)  # [N, embed_dim, embed_dim]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [N, embed_dim, embed_dim]
        self.attention_weights = attention_weights
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to value
        out = torch.einsum('bde,be->bd', attention_weights, value)  # [N, embed_dim]

        # Apply residual connection and layer norm
        residual = self.residual_alpha * x + (1 - self.residual_alpha) * out
        out = self.layer_norm(residual)
        out = F.relu(out)

        return out, attention_weights

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class FeatureDimSelfAttentionNoFFN(nn.Module):
    def __init__(self, input_dim, embed_dim, residual_alpha, dropout):
        super(FeatureDimSelfAttentionNoFFN, self).__init__()
        self.embed_dim_per_channel = embed_dim
        self.num_channels = input_dim // self.embed_dim_per_channel

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

        self.query_proj = nn.Linear(self.embed_dim_per_channel, self.embed_dim_per_channel)
        self.key_proj = nn.Linear(self.embed_dim_per_channel, self.embed_dim_per_channel)
        self.value_proj = nn.Linear(self.embed_dim_per_channel, self.embed_dim_per_channel)

        self.residual_alpha = residual_alpha

        self.attention_weights = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        self.layer_norm.reset_parameters()

    def forward(self, x):
        N, embed_dim = x.shape
        x = x.view(N, self.num_channels, self.embed_dim_per_channel)

        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)

        # Compute attention scores and weights
        # attention_scores = torch.einsum('nce,nke->nck', query, key) / math.sqrt(self.embed_dim_per_channel)
        attention_scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(self.embed_dim_per_channel)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        # out = torch.einsum('nck,nke->nce', attention_weights, value)
        out = torch.matmul(attention_weights, value)
        out = out.view(N, embed_dim)

        # Residual connection and layer normalization
        residual = self.residual_alpha * x.view(N, embed_dim) + (1 - self.residual_alpha) * out
        out = self.layer_norm(residual)

        # Activation function
        out = F.relu(out)

        self.attention_weights = attention_weights

        return out, attention_weights
    

class FeatureDimSelfAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, residual_alpha, dropout, ffn=False):
        super(FeatureDimSelfAttention, self).__init__()
        self.embed_dim_per_channel = embed_dim
        self.num_channels = input_dim // self.embed_dim_per_channel

        self.dropout = nn.Dropout(dropout)

        self.query_proj = nn.Linear(self.embed_dim_per_channel, self.embed_dim_per_channel)
        self.key_proj = nn.Linear(self.embed_dim_per_channel, self.embed_dim_per_channel)
        self.value_proj = nn.Linear(self.embed_dim_per_channel, self.embed_dim_per_channel)        

        self.residual_alpha = residual_alpha

        self.attention_weights = None

        self.ffn = ffn
        if self.ffn:
            self.ffn = nn.Sequential(
                nn.Linear(self.embed_dim_per_channel, self.embed_dim_per_channel),
                nn.ReLU(),
                nn.Dropout(dropout),    
                nn.Linear(self.embed_dim_per_channel, self.embed_dim_per_channel)
            )
            self.layer_norm_channel = nn.LayerNorm(self.embed_dim_per_channel)

        self.layer_norm_output = nn.LayerNorm(input_dim)


        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        self.layer_norm_output.reset_parameters()

        if self.ffn:
            self.ffn[0].reset_parameters()
            self.ffn[3].reset_parameters()
            self.layer_norm_channel.reset_parameters()



    def forward(self, x):
        N, embed_dim = x.shape
        x = x.view(N, self.num_channels, self.embed_dim_per_channel)

        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)

        # Compute attention scores and weights
        attention_scores = torch.einsum('nce,nke->nck', query, key) / math.sqrt(self.embed_dim_per_channel)
        attention_weights = F.softmax(attention_scores, dim=-1)
        self.attention_weights = attention_weights

        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        out = torch.einsum('nck,nke->nce', attention_weights, value)
        # out = out.view(N, embed_dim)
        if self.ffn:
            # Residual connection and layer normalization
            residual = self.residual_alpha * x.view(N, self.num_channels, self.embed_dim_per_channel) + (1 - self.residual_alpha) * out
            out = self.layer_norm_channel(residual)

            # Activation function
            # out = F.relu(out)
            ffn_out = self.ffn(out)
            # ffn_out = self.dropout(ffn_out)

            # Residual connection and layer normalization
            residual = self.residual_alpha * out.view(N, embed_dim) + (1 - self.residual_alpha) * ffn_out.view(N, embed_dim)
            out = self.layer_norm_output(residual)
        else:
            # Residual connection and layer normalization
            residual = self.residual_alpha * x.view(N, embed_dim) + (1 - self.residual_alpha) * out.view(N, embed_dim)
            out = self.layer_norm_output(residual)
            # out = F.relu(out)

        return out, attention_weights


class CrossFeatureAttention(nn.Module):
    def __init__(self, embed_dim=128, residual_alpha=0.5, dropout=0.1):
        """
        embed_dim: Embedding dimension for each feature
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.query_transform = nn.Linear(embed_dim, embed_dim)  # Query projection
        self.key_transform = nn.Linear(embed_dim, embed_dim)    # Key projection
        self.value_transform = nn.Linear(embed_dim, embed_dim)  # Value projection

        self.residual_alpha=residual_alpha
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_i, x_j):
        """
        x_i: Target node feature (E, feature_num*embed_dim)
        x_j: Source node feature (E, feature_num*embed_dim)
        """
        # reshape x_i, x_j
        x_i = x_i.view(x_i.size(0), -1, self.embed_dim)  # (E, feature_num, embed_dim)
        x_j = x_j.view(x_j.size(0), -1, self.embed_dim)  # (E, feature_num, embed_dim)
        # Projection Query, Key, Value
        q = self.query_transform(x_i)  # (E, feature_num, embed_dim)
        k = self.key_transform(x_j)    # (E, feature_num, embed_dim)
        v = self.value_transform(x_j)  # (E, feature_num, embed_dim)

        # Calculate attention scores
        scores = torch.einsum("efh,efh->ef", q, k) / math.sqrt(q.size(-1))  # (E, feature_num)
        attn_weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (E, feature_num, 1)
        self.attention_weights = attn_weights

        attn_weights = self.dropout(attn_weights)

        # Weighted Value
        attn_output = attn_weights * v  # (E, feature_num, embed_dim)
        
        # residual = self.residual_alpha * x_i + (1 - self.residual_alpha) * attn_output
        # out = self.layer_norm(residual)

        # flatten
        out = attn_output.view(x_i.size(0), -1)
        
        return out, attn_weights