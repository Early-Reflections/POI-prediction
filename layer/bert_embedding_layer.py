import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import logging
import pandas as pd
import os.path as osp
import numpy as np
from utils import get_root_dir, construct_slots
from layer.st_encoder import AbsoluteLatEncoder, AbsoluteLonEncoder, AbsoluteUnixTimeEncoder

class AddressFusion(nn.Module):
    def __init__(self, input_dim=768, output_dim=128):
        super(AddressFusion, self).__init__()
        self.reducer = nn.Linear(input_dim, output_dim)
        # init the weights and bias
        nn.init.xavier_uniform_(self.reducer.weight)
        nn.init.zeros_(self.reducer.bias)
    def forward(self, embeddings):
        # embeddings shape: [batch_size, sequence_length, input_dim]
        reduced_embeddings = self.reducer(embeddings)  # output shape: [batch_size, sequence_length, output_dim]
        return reduced_embeddings

class BertCheckinEmbedding(nn.Module):
    def __init__(
        self,
        embed_size,
        fusion_type,
        dataset_args
    ):
        super(BertCheckinEmbedding, self).__init__()

        # read the POI database
        # D:\Projects\Spatio-Temporal-Hypergraph-Model\data\tky\preprocessed\POI_database.csv
        root=osp.join(get_root_dir(), 'data', dataset_args.dataset_name, 'preprocessed')
        self.pretrain_address_embedding = torch.load(osp.join(root, "bert_address_embedding.pt"))
        self.pretrain_address_embedding = nn.Embedding.from_pretrained(self.pretrain_address_embedding, freeze=True)
        self.address_fusion = AddressFusion(input_dim=768, output_dim=embed_size)
        
        self.embed_size = embed_size
        self.fusion_type = fusion_type
        self.user_embedding = nn.Embedding(
            dataset_args.num_user + 1,
            self.embed_size,
            padding_idx=dataset_args.padding_user_id
        )
        self.poi_embedding = nn.Embedding(
            dataset_args.num_poi + 1,
            self.embed_size,
            padding_idx=dataset_args.padding_poi_id
        )
        self.category_embedding = nn.Embedding(
            dataset_args.num_category + 1,
            self.embed_size,
            padding_idx=dataset_args.padding_poi_category
        )
        self.dayofweek_embedding = nn.Embedding(8, self.embed_size, padding_idx=dataset_args.padding_weekday_id)
        self.hourofday_embedding = nn.Embedding(25, self.embed_size, padding_idx=dataset_args.padding_hour_id)

        if self.fusion_type == 'concat':
            self.output_embed_size = 6 * self.embed_size
        elif self.fusion_type == 'add':
            self.output_embed_size = embed_size
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")

    def forward(self, data):
            # Checkin feature columns
    # checkin_feature_columns = [
    #     'UserId',
    #     'PoiId',
    #     'PoiCategoryId',
    #     'UTCTimeOffsetEpoch',
    #     'Longitude',
    #     'Latitude',
    #     'UTCTimeOffsetWeekday',
    #     'UTCTimeOffsetHour'
    # ]
        embedding_list = [
            self.user_embedding(data[..., 0].long()),
            self.poi_embedding(data[..., 1].long()),
            self.category_embedding(data[..., 2].long()),
            self.dayofweek_embedding(data[..., 6].long()),
            self.hourofday_embedding(data[..., 7].long()),
            self.address_fusion(self.pretrain_address_embedding(data[..., 1].long()))
        ]
        if self.fusion_type == 'concat':
            self.output_embed_size = len(embedding_list) * self.embed_size
            return torch.cat(embedding_list, -1)
        elif self.fusion_type == 'add':
            return torch.squeeze(sum(embedding_list))
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")


class EdgeEmbedding(torch.nn.Module):
    def __init__(self, embed_size, fusion_type, num_edge_type):
        super(EdgeEmbedding, self).__init__()
        self.embed_size = embed_size
        self.fusion_type = fusion_type
        self.edge_type_embedding = nn.Embedding(num_edge_type, self.embed_size)
        self.output_embed_size = self.embed_size

    def forward(self, data):
        embedding_list = [self.edge_type_embedding(data.long())]

        if self.fusion_type == 'concat':
            self.output_embed_size = len(embedding_list) * self.embed_size
            return torch.cat(embedding_list, -1)
        elif self.fusion_type == 'add':
            return sum(embedding_list)
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")

class TrajFeatureEmbedding(nn.Module):
    def __init__(
        self,
        embed_size,
        fusion_type,
        dataset_args
    ):
        super(TrajFeatureEmbedding, self).__init__()
        
        # traj_stat[['size', 'mean_lon', 'mean_lat', 'mean_time', 'start_time', 'end_time']]
        self.embed_size = embed_size
        self.fusion_type = fusion_type
        if dataset_args.dataset_name == 'tky':
            maxsize = 239
        elif dataset_args.dataset_name == 'nyc':
            maxsize = 438
        elif dataset_args.dataset_name == 'ca':
            maxsize = 520
        self.size_embedding = nn.Embedding(maxsize, self.embed_size)

        self.mean_lon_embedding = AbsoluteLonEncoder(self.embed_size)
        self.mean_lat_embedding = AbsoluteLatEncoder(self.embed_size)
        self.mean_time_embedding = AbsoluteUnixTimeEncoder(self.embed_size)
        self.start_time_embedding = AbsoluteUnixTimeEncoder(self.embed_size)
        self.end_time_embedding = AbsoluteUnixTimeEncoder(self.embed_size)
        


        if self.fusion_type == 'concat':
            self.output_embed_size = 6 * self.embed_size
        elif self.fusion_type == 'add':
            self.output_embed_size = embed_size
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")

    def forward(self, data):
            # Trajectory feature columns: size, mean_lon, mean_lat, mean_time, start_time, end_time
        embedding_list = [
            self.size_embedding(data[..., 0].long()),
            self.mean_lon_embedding(data[..., 1].long()),
            self.mean_lat_embedding(data[..., 2].long()),
            self.mean_time_embedding(data[..., 3].long()),
            self.start_time_embedding(data[..., 4].long()),
            self.end_time_embedding(data[..., 5].long())
        ]
        if self.fusion_type == 'concat':
            self.output_embed_size = len(embedding_list) * self.embed_size
            return torch.cat(embedding_list, -1)
        elif self.fusion_type == 'add':
            return torch.squeeze(sum(embedding_list))
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")