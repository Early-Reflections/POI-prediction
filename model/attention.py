import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import BertCheckinEmbedding, TimeEncoder, DistanceEncoderSimple


class TrajectoryAttention(nn.Module):
    """
    轨迹相似度 Attention 计算模块
    """
    def __init__(self, embed_size, num_heads=4, dropout=0.1):
        super(TrajectoryAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_size)
    
    def forward(self, query_traj, key_traj, traj_sim):
        """
        query_traj: 目标轨迹 [B, D]
        key_traj: 相似轨迹 [B, K, D]
        traj_sim: 轨迹相似度 [B, K]
        """
        attn_weights = F.softmax(traj_sim, dim=-1)  # [B, K]
        attn_output = torch.einsum("bk,bkd->bd", attn_weights, key_traj)  # [B, D]
        return self.norm(query_traj + attn_output)  # 残差连接


class STChannelAttention(nn.Module):
    """
    时空通道注意力机制，用于融合时间、空间信息
    """
    def __init__(self, embed_size):
        super(STChannelAttention, self).__init__()
        self.time_weight = nn.Parameter(torch.randn(embed_size))
        self.space_weight = nn.Parameter(torch.randn(embed_size))
    
    def forward(self, traj_embedding, time_embedding, distance_embedding):
        """
        traj_embedding: 轨迹嵌入 [B, D]
        time_embedding: 时间嵌入 [B, D]
        distance_embedding: 空间嵌入 [B, D]
        """
        return traj_embedding + self.time_weight * time_embedding + self.space_weight * distance_embedding


class STHGCN(nn.Module):
    def __init__(self, cfg):
        super(STHGCN, self).__init__()
        self.device = cfg.run_args.device
        self.batch_size = cfg.run_args.batch_size
        self.num_poi = cfg.dataset_args.num_poi
        self.embed_size = cfg.model_args.embed_size
        
        # POI & Check-in Embedding
        self.checkin_embedding_layer = BertCheckinEmbedding(
            embed_size=self.embed_size,
            fusion_type=cfg.model_args.embed_fusion_type,
            dataset_args=cfg.dataset_args
        )
        
        # 时间 & 距离编码器
        self.time_encoder = TimeEncoder(cfg.model_args, self.embed_size)
        self.distance_encoder = DistanceEncoderSimple(cfg.model_args, self.embed_size, cfg.dataset_args.spatial_slots)
        
        # 轨迹相似度 Attention 计算
        self.traj_attention = TrajectoryAttention(self.embed_size)
        self.st_channel_attn = STChannelAttention(self.embed_size)
        
        # 输出层
        self.linear = nn.Linear(self.embed_size, self.num_poi)
        self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, data, label=None, mode='train'):
        input_x = data['x']  # 获取输入特征
        split_idx = data['split_index']  # 分界点
        check_in_x = input_x[split_idx + 1:]  # 获取签到点特征
        checkin_feature = self.checkin_embedding_layer(check_in_x)  # 计算签到嵌入
        trajectory_feature = torch.zeros(split_idx + 1, checkin_feature.shape[-1], device=checkin_feature.device)
        
        # 合并轨迹 & 签到特征
        x = torch.cat([trajectory_feature, checkin_feature], dim=0)
        
        # 计算时间 & 空间嵌入
        edge_time_embed = self.time_encoder(data['delta_ts'][0] / (60 * 60))
        edge_distance_embed = self.distance_encoder(data['delta_ss'][0])

        x_for_time_filter = self.traj_attention(x, x, data['edge_attr'][0])
        x_for_time_filter = self.st_channel_attn(x_for_time_filter, edge_time_embed, edge_distance_embed)
        x_for_time_filter = self.act(x_for_time_filter)
        x_for_time_filter = self.dropout_for_time_filter(x_for_time_filter)
        
        # 更新 x
        x = x_for_time_filter
        
        # 计算 ci2traj 信息传播
        ci2traj_edge_index = data['edge_index'][0]
        row_ci, col_ci, _ = ci2traj_edge_index.coo()
        ci2traj_edge_attr = data['edge_attr'][0]
        query_poi = x[row_ci]  # 目标签到点
        key_traj = x[col_ci]  # 对应轨迹
        x[split_idx + 1:] = self.traj_attention(query_poi, key_traj, ci2traj_edge_attr)
        
        # 计算 traj2traj 信息传播
        traj2traj_edge_index = data['edge_index'][1]
        row_tj, col_tj, _ = traj2traj_edge_index.coo()
        query_traj = x[row_tj]  # 目标轨迹
        key_traj = x[col_tj]  # 相似轨迹
        traj_sim = data['edge_attr'][1]  # 轨迹相似度
        x[:split_idx + 1] = self.traj_attention(query_traj, key_traj, traj_sim)
        
        # 融合时间 & 空间信息
        x[split_idx + 1:] = self.st_channel_attn(x[split_idx + 1:], edge_time_embed, edge_distance_embed)
        x[:split_idx + 1] = self.st_channel_attn(x[:split_idx + 1], edge_time_embed, edge_distance_embed)
        
        # 计算最终签到点预测
        logits = self.linear(x[split_idx + 1:])
        loss = self.loss_func(logits, label.long())
        return logits, loss

