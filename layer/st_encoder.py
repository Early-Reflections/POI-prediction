import math
import torch
from torch import nn
import numpy as np
from utils import cal_slot_distance_batch


class PositionEncoder(nn.Module):
    def __init__(self, d_model, device, dropout=0.1, max_len=500):
        super(PositionEncoder, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (- math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class TimeEncoder(nn.Module):
    r"""
    This is a trainable encoder to map continuous time value into a low-dimension time vector.
    Ref: https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs/blob/master/module.py

    The input of ts should be like [E, 1] with all time interval as values.
    """

    def __init__(self, args, embedding_dim):
        super(TimeEncoder, self).__init__()
        self.time_dim = embedding_dim
        self.expand_dim = self.time_dim
        self.factor = args.phase_factor
        self.use_linear_trans = args.use_linear_trans

        self.basis_freq = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
        self.phase = nn.Parameter(torch.zeros(self.time_dim).float())
        if self.use_linear_trans:
            self.dense = nn.Linear(self.time_dim, self.expand_dim, bias=False)
            nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):
        if ts.dim() == 1:
            dim = 1
            edge_len = ts.size().numel()
        else:
            edge_len, dim = ts.size()
        ts = ts.view(edge_len, dim)
        map_ts = ts * self.basis_freq.view(1, -1)
        map_ts += self.phase.view(1, -1)
        harmonic = torch.cos(map_ts)
        if self.use_linear_trans:
            harmonic = harmonic.type(self.dense.weight.dtype)
            harmonic = self.dense(harmonic)
        return harmonic
    
class AbsoluteLatEncoder(nn.Module):
    """
    Trainable encoder to map continuous latitude values into low-dimensional embeddings.
    """

    def __init__(self, embedding_dim, lat_range=(-90, 90)):
        """
        Args:
            embedding_dim (int): Dimension of the embedding.
            lat_range (tuple): Range of latitude values, typically (-90, 90).
        """
        super(AbsoluteLatEncoder, self).__init__()
        self.embed_dim = embedding_dim
        self.lat_range = lat_range
        self.lat_proj = nn.Linear(1, embedding_dim)  # Learnable linear projection

    def forward(self, lat):
        """
        Forward pass for latitude encoding.

        Args:
            lat (Tensor): Latitude values. Shape: [N].

        Returns:
            Tensor: Latitude embeddings. Shape: [N, embedding_dim].
        """
        # Normalize latitude to [0, 1]
        lat_norm = (lat - self.lat_range[0]) / (self.lat_range[1] - self.lat_range[0])

        # Apply sine function to capture periodic properties
        lat_enc = torch.sin(lat_norm.unsqueeze(1) * math.pi)  # Shape: [N, 1]

        # Project to embedding dimension
        lat_embedding = self.lat_proj(lat_enc)  # Shape: [N, embedding_dim]

        return lat_embedding
    
class AbsoluteLonEncoder(nn.Module):
    """
    Trainable encoder to map continuous longitude values into low-dimensional embeddings.
    """

    def __init__(self, embedding_dim, lon_range=(-180, 180)):
        """
        Args:
            embedding_dim (int): Dimension of the embedding.
            lon_range (tuple): Range of longitude values, typically (-180, 180).
        """
        super(AbsoluteLonEncoder, self).__init__()
        self.embed_dim = embedding_dim
        self.lon_range = lon_range
        self.lon_proj = nn.Linear(1, embedding_dim)  # Learnable linear projection

    def forward(self, lon):
        """
        Forward pass for longitude encoding.

        Args:
            lon (Tensor): Longitude values. Shape: [N].

        Returns:
            Tensor: Longitude embeddings. Shape: [N, embedding_dim].
        """
        # Normalize longitude to [0, 1]
        lon_norm = (lon - self.lon_range[0]) / (self.lon_range[1] - self.lon_range[0])

        # Apply cosine function to capture periodic properties
        lon_enc = torch.cos(lon_norm.unsqueeze(1) * math.pi)  # Shape: [N, 1]

        # Project to embedding dimension
        lon_embedding = self.lon_proj(lon_enc)  # Shape: [N, embedding_dim]

        return lon_embedding


class AbsoluteUnixTimeEncoder(nn.Module):
    """
    Trainable module to encode Unix timestamps into low-dimensional sinusoidal embeddings.
    """

    def __init__(self, embedding_dim, scale=3600):
        """
        Args:
            embedding_dim (int): Dimension of the embedding (must be even).
            scale (float): Scaling factor for time normalization 
                           (e.g., 3600 for hours, 86400 for days).
        """
        super(AbsoluteUnixTimeEncoder, self).__init__()
        assert embedding_dim % 2 == 0, "Embedding dimension must be even."
        self.embedding_dim = embedding_dim
        self.scale = scale

    def forward(self, unix_time):
        """
        Forward pass for Unix timestamp encoding.

        Args:
            unix_time (Tensor): Unix timestamps (e.g., seconds since 1970). Shape: [N].

        Returns:
            Tensor: Time encoding. Shape: [N, embedding_dim].
        """
        # Normalize timestamps
        position = unix_time / self.scale  # Shape: [N]

        # Compute sine and cosine embeddings
        dim = torch.arange(self.embedding_dim // 2, dtype=torch.float32, device=unix_time.device)
        div_term = 10000 ** (2 * dim / self.embedding_dim)

        # Allocate space for embedding
        time_enc = torch.zeros((unix_time.size(0), self.embedding_dim), device=unix_time.device)
        time_enc[:, 0::2] = torch.sin(position.unsqueeze(1) / div_term)  # Even indices: sin
        time_enc[:, 1::2] = torch.cos(position.unsqueeze(1) / div_term)  # Odd indices: cos

        return time_enc


    
class RotaryTimeEncoder(nn.Module):
    r"""
    Rotary Position Embedding (RoPE) for time encoding.
    Ref: https://arxiv.org/abs/2104.09864 (Su et al., RoFormer).
    
    This implementation maps continuous time values into a rotary embedding.
    """

    def __init__(self, embedding_dim):
        super(RotaryTimeEncoder, self).__init__()
        self.time_dim = embedding_dim
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, self.time_dim, 2).float() / self.time_dim))

    def forward(self, ts):
        """
        Args:
            ts: A tensor of shape [E, 1] or [E] containing continuous time values.

        Returns:
            A tensor of shape [E, time_dim] with rotary position encoding applied.
        """
        if ts.dim() == 1:
            ts = ts.unsqueeze(-1)  # Ensure shape [E, 1]

        # Compute scaled time values
        sinusoid_inp = ts * self.inv_freq.view(1, -1)  # [E, time_dim / 2]

        # Compute sine and cosine embeddings
        sin_embed = torch.sin(sinusoid_inp)  # [E, time_dim / 2]
        cos_embed = torch.cos(sinusoid_inp)  # [E, time_dim / 2]

        # Concatenate along the last dimension
        rotary_embed = torch.cat([sin_embed, cos_embed], dim=-1)  # [E, time_dim]

        return rotary_embed



class DistanceEncoderHSTLSTM(nn.Module):
    r"""
    This is a trainable encoder to map continuous distance value into a low-dimension vector.
    Ref: HST-LSTM

    First determine the position of diffrent slot bins, and do linear interpolation within different slots
    with the embedding of the slots as a trainable parameters.
    """

    def __init__(self, args, embedding_dim, spatial_slots):
        super(DistanceEncoderHSTLSTM, self).__init__()
        self.dist_dim = embedding_dim
        self.spatial_slots = spatial_slots
        self.embed_q = nn.Embedding(len(spatial_slots), self.dist_dim)
        self.device = args.gpu

    def place_parameters(self, ld, hd, l, h):
        if self.device == 'cpu':
            ld = torch.from_numpy(np.array(ld)).type(torch.FloatTensor)
            hd = torch.from_numpy(np.array(hd)).type(torch.FloatTensor)
            l = torch.from_numpy(np.array(l)).type(torch.LongTensor)
            h = torch.from_numpy(np.array(h)).type(torch.LongTensor)
        else:     
            ld = torch.from_numpy(np.array(ld, dtype=np.float16)).type(torch.FloatTensor).to(self.device)
            hd = torch.from_numpy(np.array(hd, dtype=np.float16)).type(torch.FloatTensor).to(self.device)
            l = torch.from_numpy(np.array(l, dtype=np.float16)).type(torch.LongTensor).to(self.device)
            h = torch.from_numpy(np.array(h, dtype=np.float16)).type(torch.LongTensor).to(self.device)
        return ld, hd, l, h

    def cal_inter(self, ld, hd, l, h, embed):
        """
        Calculate a linear interpolation.
        :param ld: Distances to lower bound, shape (batch_size, step)
        :param hd: Distances to higher bound, shape (batch_size, step)
        :param l: Lower bound indexes, shape (batch_size, step)
        :param h: Higher bound indexes, shape (batch_size, step)
        """
        # Fetch the embed of higher and lower bound.
        # Each result shape (batch_size, step, input_size)
        l_embed = embed(l)
        h_embed = embed(h)
        return torch.stack([hd], -1) * l_embed + torch.stack([ld], -1) * h_embed

    def forward(self, dist):
        self.spatial_slots = sorted(self.spatial_slots)
        d_ld, d_hd, d_l, d_h = self.place_parameters(*cal_slot_distance_batch(dist, self.spatial_slots))
        batch_q = self.cal_inter(d_ld, d_hd, d_l, d_h, self.embed_q)
        return batch_q


class DistanceEncoderSTAN(nn.Module):
    r"""
    This is a trainable encoder to map continuous distance value into a low-dimension vector.
    Ref: STAN

    Interpolating between min and max distance value, only need to initial minimum distance embedding and maximum
    distance embedding.
    """

    def __init__(self, args, embedding_dim, spatial_slots):
        super(DistanceEncoderSTAN, self).__init__()
        self.dist_dim = embedding_dim
        self.min_d, self.max_d_ch2tj, self.max_d_tj2tj = spatial_slots
        self.embed_min = nn.Embedding(1, self.dist_dim)
        self.embed_max = nn.Embedding(1, self.dist_dim)
        self.embed_max_traj = nn.Embedding(1, self.dist_dim)
        self.quantile = args.quantile

    def forward(self, dist, dist_type):
        if dist_type == 'ch2tj':
            emb_low, emb_high = self.embed_min.weight, self.embed_max.weight
            max_d = self.max_d_ch2tj
        else:
            emb_low, emb_high = self.embed_min.weight, self.embed_max_traj.weight
            max_d = self.max_d_tj2tj

        # if you want to clip in case of outlier maxmimum exist, please uncomment the line below
        # max_d = torch.quantile(dist, self.quantile)
        dist = dist.clip(0, max_d)
        vsl = (dist - self.min_d).unsqueeze(-1).expand(-1, self.dist_dim)
        vsu = (max_d - dist).unsqueeze(-1).expand(-1, self.dist_dim)

        space_interval = (emb_low * vsu + emb_high * vsl) / (max_d - self.min_d)
        return space_interval


class DistanceEncoderSimple(nn.Module):
    r"""
    This is a trainable encoder to map continuous distance value into a low-dimension vector.

    Only need to initial just on embedding, and directly do scalar*vector multiply.
    """
    def __init__(self, args, embedding_dim, spatial_slots):
        super(DistanceEncoderSimple, self).__init__()
        self.args = args
        self.dist_dim = embedding_dim
        self.min_d, self.max_d, self.max_d_traj = spatial_slots
        self.embed_unit = nn.Embedding(1, self.dist_dim)

    def forward(self, dist):
        dist = dist.unsqueeze(-1).expand(-1, self.dist_dim)
        return dist * self.embed_unit.weight
