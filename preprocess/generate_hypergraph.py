from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from utils import haversine
import os
import os.path as osp
import logging


def generate_hypergraph_from_file(input_file, output_path, args):
    """
    从输入文件构建[签到 -> 轨迹]的关联矩阵和[轨迹 -> 轨迹]的邻接列表。
    edge_index 表示：
        [[ -签到- ]
        [ -轨迹(超边)- ]]
    以及
        [[ -轨迹(超边)- ]
        [ -轨迹(超边)- ]]
    使用的txt文件列:
        UserId, check_ins_id, PoiId, Latitude, Longitude, PoiCategoryId, UTCTimeOffsetEpoch,
        pseudo_session_trajectory_id, UTCTimeOffsetWeekday, UTCTimeOffsetHour。

    :param input_file: 超图原始数据文件路径
    :param output_path: 输出文件目录
    :param args: 解析输入参数
    :return: None

    这里的超图中的节点实际上包含了签到和轨迹两种节点，并且签到节点和轨迹节点之间存在关联矩阵，
    而轨迹节点和轨迹节点之间存在邻接矩阵。感觉把两个节点类型分开处理会更好。
    """
    usecols = [
        'UserId', 'PoiId', 'PoiCategoryId', 'Latitude', 'Longitude', 'UTCTimeOffsetEpoch', 'UTCTimeOffsetWeekday',
        'UTCTimeOffsetHour', 'check_ins_id', 'trajectory_id'
    ]
    threshold = args.threshold
    filter_mode = args.filter_mode
    data = pd.read_csv(input_file, usecols=usecols)

    traj_column = 'trajectory_id'

    # If True, Shift traj_id with offset #check_ins_id before saving to pyg data, which means idx of checkin are
    # in range [0, #checkin_id - 1], and idx of trajectory are in range [#checkin, #trajectory+#checkin-1]
    # 为 trajectory_id 设置偏移量，检查数据是否按顺序排列
    traj_offset = True
    if traj_offset:
        checkin_offset = torch.as_tensor([data.check_ins_id.max() + 1], dtype=torch.long)
    else:
        checkin_offset = torch.as_tensor([0], dtype=torch.long)

    # 生成轨迹（超边）统计信息
    traj_stat = generate_hyperedge_stat(data, traj_column) # (207148, 9)
    # 生成签到到轨迹的pyg数据
    ci2traj_pyg_data = generate_ci2traj_pyg_data(data, traj_stat, traj_column, checkin_offset)
     # 生成轨迹到轨迹的intra user pyg数据
    traj2traj_intra_u_data = generate_traj2traj_data(
        data,
        traj_stat,
        traj_column,
        threshold,
        filter_mode=filter_mode,
        relation_type='intra'
    )
    # 生成轨迹到轨迹的inter user pyg数据
    traj2traj_inter_u_data = generate_traj2traj_data(
        data,
        traj_stat,
        traj_column,
        threshold,
        filter_mode=filter_mode,
        relation_type='inter'
    )
    # 合并轨迹到轨迹的pyg数据
    traj2traj_pyg_data = merge_traj2traj_data(traj_stat, traj2traj_intra_u_data, traj2traj_inter_u_data, checkin_offset)

    # save pyg data
    if not osp.isdir(output_path):
        os.makedirs(output_path)
    ci2traj_out_file = osp.join(output_path, 'ci2traj_pyg_data.pt')
    traj2traj_out_file = osp.join(output_path, 'traj2traj_pyg_data.pt')
    torch.save(ci2traj_pyg_data, ci2traj_out_file)
    torch.save(traj2traj_pyg_data, traj2traj_out_file)

    logging.info(
        f'[Preprocess - Generate Hypergraph] Done saving checkin2trajectory pyg data to {ci2traj_out_file}'
        f' and trajectory2trajectory pyg data to {traj2traj_out_file}.'
    )
    return


def generate_hyperedge_stat(data, traj_column):
    """
    生成轨迹（超边）的统计信息，例如大小、中心经纬度、时间范围等。

    :param data: 原始伪会话轨迹数据
    :param traj_column: 轨迹列名
    :return: traj_stat - 包含超边统计信息的 DataFrame
    """
    traj_stat = pd.DataFrame()
    traj_stat['size'] = data.groupby(traj_column)['UTCTimeOffsetEpoch'].apply(len)
    traj_stat['mean_lon'] = data.groupby(traj_column)['Longitude'].apply(sum) / traj_stat['size']
    traj_stat['mean_lat'] = data.groupby(traj_column)['Latitude'].apply(sum) / traj_stat['size']
    traj_stat[['last_lon', 'last_lat']] = \
        data.sort_values([traj_column, 'UTCTimeOffsetEpoch']).groupby(traj_column).last()[['Longitude', 'Latitude']]

    # 计算轨迹的开始时间和结束时间和时间窗口
    traj_stat['start_time'] = data.groupby(traj_column)['UTCTimeOffsetEpoch'].apply(min)
    traj_stat['end_time'] = data.groupby(traj_column)['UTCTimeOffsetEpoch'].apply(max)
    traj_stat['mean_time'] = data.groupby(traj_column)['UTCTimeOffsetEpoch'].apply(sum) / traj_stat['size']
    traj_stat['time_window_in_hour'] = (traj_stat.end_time - traj_stat.start_time) / (60*60)
    logging.info(f'[Preprocess - Generate Hypergraph] Number of hyperedges(trajectory): {traj_stat.shape[0]}.')
    logging.info(
        f'[Preprocess - Generate Hypergraph] The min, mean, max size of hyperedges are: '
        f'{traj_stat["size"].min()}, {traj_stat["size"].mean()}, {traj_stat["size"].max()}.'
    )
    logging.info(
        f'[Preprocess - Generate Hypergraph] The min, mean, max time window of hyperedges are:'
        f'{traj_stat.time_window_in_hour.min()}, {traj_stat.time_window_in_hour.mean()}, '
        f'{traj_stat.time_window_in_hour.max()}.'
    )
    return traj_stat


def generate_ci2traj_pyg_data(data, traj_stat, traj_column, checkin_offset):
    """
    生成签到到轨迹的关联矩阵和签到特征矩阵。

    :param data: 原始轨迹数据
    :param traj_stat: 超边（轨迹）统计数据
    :param traj_column: 轨迹列名
    :param checkin_offset: 最大签到索引加1
    :return: 包含关联矩阵、签到特征矩阵和其他边信息的 PyG 数据
    """
    # 签到特征列
    checkin_feature_columns = [
        'UserId',
        'PoiId',
        'PoiCategoryId',
        'UTCTimeOffsetEpoch',
        'Longitude',
        'Latitude',
        'UTCTimeOffsetWeekday',
        'UTCTimeOffsetHour'
    ]
    checkin_feature = data.sort_values('check_ins_id')[checkin_feature_columns].to_numpy()
    assert data.check_ins_id.unique().shape[0] == data.check_ins_id.max() + 1, \
        'check_ins_id is not chronological order in raw data'

    # Calculate distance between trajectory's last poi location and curren poi location
    # 计算轨迹的最后一个 POI 位置到当前 POI 位置的距离
    delta_s_in_traj = data.join(traj_stat, on=traj_column, how='left')[
        ['Longitude', 'Latitude', 'last_lon', 'last_lat']
    ]
    delta_s_in_traj['distance_km'] = haversine(
        delta_s_in_traj.Longitude,
        delta_s_in_traj.Latitude,
        delta_s_in_traj.last_lon,
        delta_s_in_traj.last_lat
    )

    # Create incidence matrix for check-in -> trajectory
    # 這裏row是traj_id, col是checkin_id, value是checkin在traj中的index
    # 创建签到到轨迹的关联矩阵
    ci2traj_adj_t = SparseTensor(
        row=torch.as_tensor(data[traj_column].tolist(), dtype=torch.long),
        col=torch.as_tensor(data.check_ins_id.tolist(), dtype=torch.long),
        value=torch.as_tensor(range(0, data.shape[0]), dtype=torch.long)
    )   # shape: (num_trajectories, num_checkins) 以sparse格式存储

    # checkin到轨迹的边(也就是checkin行为发生的时刻)特征：签到时间、时间差和空间差
    perm = ci2traj_adj_t.storage.value()  # 获取非零元素的索引
    ci2traj_edge_t = torch.tensor(data.UTCTimeOffsetEpoch.tolist())[perm]  # 获取签到时间 
    ci2traj_edge_delta_t = torch.tensor(
        traj_stat.end_time[data[traj_column].tolist()].values - data.UTCTimeOffsetEpoch.values
    )[perm]  # 获取轨迹结束时间与签到时间的时间差，也就是这个checkin行为持续的有效时间
    ci2traj_edge_delta_s = torch.tensor(delta_s_in_traj.distance_km.tolist())[perm]  # 获取轨迹结束点到当前签到点的距离

    # 生成边索引，轨迹ID偏移量用于区分轨迹和签到，这里的row是checkin_id，col是traj_id，此时checkin_id是source，traj_id是target
    ci2traj_edge_index = torch.stack([ci2traj_adj_t.storage.col(), ci2traj_adj_t.storage.row() + checkin_offset])

    # 构建 PyG 数据对象
    ci2traj_pyg_data = Data(
        edge_index=ci2traj_edge_index,
        x=torch.tensor(checkin_feature),   # shape: (num_checkins, 8)
        edge_t=ci2traj_edge_t,           # 签到时间（签到-轨迹的边存在的时刻）
        edge_delta_t=ci2traj_edge_delta_t,  # 轨迹结束时间与签到时间的时间差
        edge_delta_s=ci2traj_edge_delta_s  # 轨迹结束点到当前签到点的距离
    )
    ci2traj_pyg_data.num_hyperedges = traj_stat.shape[0]
    return ci2traj_pyg_data


def generate_traj2traj_data(
        data,
        traj_stat,
        traj_column,
        threshold=0.02,
        filter_mode='min size',
        chunk_num=10,
        relation_type='intra'
):
    """
    生成轨迹到轨迹（超边到超边）的动态关系。

    :param data: 原始轨迹数据
    :param traj_stat: 超边（轨迹）统计数据
    :param traj_column: 轨迹列名
    :param threshold: 过滤噪声关系的阈值
    :param filter_mode: 用于过滤噪声关系的过滤模式
    :param chunk_num: 分块数量，防止 OOM
    :param relation_type: intra（同用户）或 inter（跨用户）
    :return: 轨迹到轨迹的关系数据（边索引、边类型、时间差和空间差）
    """
    traj2traj_original_metric = None
    # First create sparse matrix for trajectory -> poi, then generate inter-user adjacency list
    # one trajectory may have multiple identical poi_id, we drop the duplicate ones first
    # 生成稀疏矩阵表示的轨迹到POI的映射
    traj_user_map = data[['UserId', traj_column]].drop_duplicates().set_index(traj_column)  # 轨迹-用户映射
    traj_size_adjust = None
    if relation_type == 'inter':
        traj_poi_map = data[['PoiId', traj_column]].drop_duplicates()  # 轨迹-POI映射
        traj2node = coo_matrix((
            np.ones(traj_poi_map.shape[0]),
            (np.array(traj_poi_map['PoiId'], dtype=np.int64), np.array(traj_poi_map[traj_column], dtype=np.int64))
        )).tocsr()  # 轨迹-POI的稀疏邻接矩阵 shape: (num_trajectories, num_POIs)

        # 根据新的轨迹-POI映射调整轨迹ID的大小
        traj_size_adjust = traj_poi_map.groupby(traj_column).apply(len).tolist()
    else:
        traj2node = coo_matrix((
            np.ones(traj_user_map.shape[0]),
            (np.array(traj_user_map['UserId'], dtype=np.int64), np.array(traj_user_map.index, dtype=np.int64))
        )).tocsr()  # 轨迹-用户的稀疏邻接矩阵 shape: (num_trajectories, num_users)

    # 生成轨迹间的稀疏邻接矩阵
    node2traj = traj2node.T
    traj2traj = node2traj * traj2node # 这里*是矩阵的哈达玛积，也就是对应元素相乘，从而保留了具有相同poi或者user的轨迹之间的连接
    traj2traj = traj2traj.tocoo()
    # 此时的traj2traj应该是对称的

    # for inter-user type, save the original similarity metric# 跨用户模式下，保存原始相似度
    if relation_type == 'inter':
        row_filtered, col_filtered, data_filtered = filter_chunk(
            row=traj2traj.row,
            col=traj2traj.col,
            data=traj2traj.data,
            chunk_num=chunk_num,
            he_size=traj_size_adjust,
            threshold=0,
            filter_mode=filter_mode
        )
        traj2traj_original_metric = coo_matrix((data_filtered, (row_filtered, col_filtered)), shape=traj2traj.shape)  
        # 跨用户模式下，保存过滤之后的相似度矩阵， shape: (num_trajectories, num_trajectories)

    # Filter 1: filter based on pre-define conditions
    # 1. different trajectory 2. source_endtime <= target_starttime # 过滤条件：不同轨迹、时间约束和不同用户
    mask_1 = traj2traj.row != traj2traj.col # 去掉自环
    mask_2 = traj_stat.end_time[traj2traj.col].values <= traj_stat.start_time[traj2traj.row].values # 源节点的结束时间小于目标节点的开始时间，这里col是源节点，row是目标节点
    mask = mask_1 & mask_2
    if relation_type == 'inter':
        # 3. diffrent user
        mask_3 = traj_user_map['UserId'][traj2traj.row].values != traj_user_map['UserId'][traj2traj.col].values # 过滤掉不同用户
        mask = mask & mask_3

    traj2traj.row = traj2traj.row[mask]
    traj2traj.col = traj2traj.col[mask]
    traj2traj.data = traj2traj.data[mask]

    # 跨用户模式下进一步基于阈值过滤
    if relation_type == 'inter':
        # Filter 2: filter based on pre-define metric threshold
        row_filtered, col_filtered, data_filtered = filter_chunk(
            row=traj2traj.row,
            col=traj2traj.col,
            data=traj2traj.data,
            chunk_num=chunk_num,
            he_size=traj_size_adjust,
            threshold=threshold,
            filter_mode=filter_mode
        )
        traj2traj.row = row_filtered
        traj2traj.col = col_filtered
        traj2traj.data = data_filtered
        edge_type = np.ones_like(traj2traj.row) # 设置边类型为1，表示是inter-user的边
    else:
        edge_type = np.zeros_like(traj2traj.row) # 设置边类型为0，表示是intra-user的边

    # Calculate edge_delta_t and edge_delta_s
    # 计算轨迹-轨迹的边的时间差和空间差
    edge_delta_t = traj_stat.mean_time[traj2traj.row].values - traj_stat.mean_time[traj2traj.col].values # 轨迹-轨迹的边平均时间差
    edge_delta_s = np.stack([
        traj_stat.mean_lon[traj2traj.row].values,
        traj_stat.mean_lat[traj2traj.row].values,
        traj_stat.mean_lon[traj2traj.col].values,
        traj_stat.mean_lat[traj2traj.col].values],
        axis=1
    )

    edge_delta_s = torch.tensor(edge_delta_s)
    edge_delta_s = haversine(edge_delta_s[:, 0], edge_delta_s[:, 1], edge_delta_s[:, 2], edge_delta_s[:, 3]) # 轨迹-轨迹的边的中心的平均距离

    logging.info(
        f"[Preprocess - Generate Hypergraph] Number of {relation_type}-user hyperedge2hyperedge(traj2traj) "
        f"relation has been generated: {traj2traj.row.shape[0]}, while threshold={threshold} and mode={filter_mode}."
    )

    return traj2traj, traj2traj_original_metric, edge_type, edge_delta_t, edge_delta_s.numpy()


def merge_traj2traj_data(traj_stat, intra_u_data, inter_u_data, checkin_offset):
    """
    Merge intra-user and inter-user hyperedge2hyperedge(traj2traj) dynamic relation.
    Merge intra-user and inter-user hyperedge2hyperedge(traj2traj) dynamic relation.
    Merge intra-user and inter-user hyperedge2hyperedge(traj2traj) dynamic relation.
    Merge intra-user and inter-user hyperedge2hyperedge(traj2traj) dynamic relation.    
    Merge intra-user and inter-user hyperedge2hyperedge(traj2traj) dynamic relation.

    :param traj_stat: hyperedge(trajectory) statistics;
    :param intra_u_data: hyperedge2hyperedge(traj2traj) relation between the same user, composited of tuple with
        edge_index(coo), edge_attr(np.array), edge_type(np.array), edge_delta_t(np.array), edge_delta_s(np.array);
    :param inter_u_data: hyperedge2hyperedge(traj2traj) relation between different users, composited of tuple like
        intra_u_data.
    :param checkin_offset: max checkin index plus 1;
    :return: pyg data of traj2traj
    """
    """
    合并同用户和不同用户之间的轨迹到轨迹（超边到超边）关系。

    :param traj_stat: 超边（轨迹）统计数据
    :param intra_u_data: 同用户的轨迹到轨迹关系，包含 edge_index、edge_type、edge_delta_t 和 edge_delta_s
    :param inter_u_data: 跨用户的轨迹到轨迹关系，包含类似 intra_u_data 的数据
    :param checkin_offset: 最大签到索引加1
    :return: PyG 数据表示的轨迹到轨迹数据
    """
    traj_feature = traj_stat[['size', 'mean_lon', 'mean_lat', 'mean_time', 'start_time', 'end_time']].to_numpy()

    # add two extra feature column to make sure traj feature has the same dimension size with ci feature
    # 补充填充特征列，使轨迹特征维度与签到特征一致
    padding_feature = np.zeros([traj_feature.shape[0], 2])
    traj_feature = np.concatenate([traj_feature, padding_feature], axis=1)

    intra_edge_index, _, intra_edge_type, intra_edge_delta_t, intra_edge_delta_s = intra_u_data
    inter_edge_index, traj2traj_orginal_metric, inter_edge_type, inter_edge_delta_t, inter_edge_delta_s = inter_u_data
    row = np.concatenate([intra_edge_index.row, inter_edge_index.row])  
    col = np.concatenate([intra_edge_index.col, inter_edge_index.col])

    # replace data with metric value
    # 将原始矩阵的相似度加上一个很小的值，其余依旧为0
    metric_data = coo_matrix((np.ones(row.shape[0]), (row, col)), shape=traj2traj_orginal_metric.shape)
    epsilon = coo_matrix((np.zeros(row.shape[0]) + 1e-6, (row, col)), shape=traj2traj_orginal_metric.shape)
    metric_data = metric_data.multiply(traj2traj_orginal_metric)
    metric_data += epsilon

    # 构建traj2traj的稀疏矩阵，此时row是target，col是source
    adj_t = SparseTensor(
        row=torch.as_tensor(row, dtype=torch.long),
        col=torch.as_tensor(col, dtype=torch.long),
        value=torch.as_tensor(range(0, row.shape[0]), dtype=torch.long)
    )
    perm = adj_t.storage.value() # 获取非零元素的索引

    x = torch.tensor(traj_feature) # x是轨迹的统计特征
    edge_type = torch.tensor(np.concatenate([intra_edge_type, inter_edge_type]))[perm] # 获取边类型
    edge_delta_t = torch.tensor(np.concatenate([intra_edge_delta_t, inter_edge_delta_t]))[perm] # 获取时间差
    edge_delta_s = torch.tensor(np.concatenate([intra_edge_delta_s, inter_edge_delta_s]))[perm] # 获取空间差

    edge_index = torch.stack([
        adj_t.storage.col() + checkin_offset,
        adj_t.storage.row() + checkin_offset
    ]) # 获取边索引，这里的row是target，col是source

    # edge_attr: source_size, target_size, jaccard_similarity
    # 边属性：源节点大小、目标节点大小和相似度
    # 其中节点大小被定义为轨迹中签到的数量除以最大签到数量
    # 节点大小经过归一化处理
    source_size = x[edge_index[0] - checkin_offset][:, 0] / x[:, 0].max() # 获取源节点大小 
    target_size = x[edge_index[1] - checkin_offset][:, 0] / x[:, 0].max() # 获取目标节点大小
    edge_attr = torch.stack([source_size, target_size, torch.tensor(metric_data.data)], dim=1) # 获取边属性，包括源节点大小、目标节点大小和节点与节点之间的相似度

    traj2traj_pyg_data = Data(
        edge_index=edge_index,
        x=x,
        edge_attr=edge_attr,
        edge_type=edge_type,
        edge_delta_t=edge_delta_t,
        edge_delta_s=edge_delta_s
    )
    return traj2traj_pyg_data


def filter_chunk(row, col, data, he_size, chunk_num=10, threshold=0.02, filter_mode='min size'):
    """
    Filter noise hyperedge2hyperedge connection based on metric threshold

    :param row: row, hyperedge2hyperedge scipy.sparse coo matrix
    :param col: col, hyperedge2hyperedge scipy.sparse coo matrix
    :param data: data, hyperedge2hyperedge scipy.sparse coo matrix
    :param he_size: hyperedge size list (drop duplicates)
    :param chunk_num: number of chunk to prevent from oom issue
    :param threshold: metric threshold, relation will be kept only if metric value is greater than threshold
    :param filter_mode: min_size - propotional to minmum size, 'jaccard' - jaccard similarity
        min_size, E2E_{ij} keeps when E2E_{ij} \ge \theta\min(|\mathcal{E}_i|,|\mathcal{E}_j|)
        jaccard, E2E_{ij} keeps when \frac{E2E_{ij}}{|\mathcal{E}_i|+|\mathcal{E}_j| - E2E_{ij}} \ge \theta
    :return:
    """
    """
    根据度量阈值过滤噪声超边到超边的连接。

    :param row: 行索引，表示超边到超边的稀疏矩阵
    :param col: 列索引，表示超边到超边的稀疏矩阵
    :param data: 数据值，表示超边到超边的稀疏矩阵
    :param he_size: 超边大小列表（去重）
    :param chunk_num: 分块数量，防止 OOM
    :param threshold: 度量阈值，关系保留的条件
    :param filter_mode: 过滤模式，支持 'min size' 或 'jaccard'
    :return: 过滤后的行、列和数据值
    """
    # Split the data to multiple chunks for large data
    chunk_bin = np.linspace(0, row.shape[0], chunk_num, dtype=np.int64)
    rows, cols, datas = [], [], []
    for i in tqdm(range(len(chunk_bin) - 1)):
        row_chunk = row[chunk_bin[i]:chunk_bin[i + 1]]
        col_chunk = col[chunk_bin[i]:chunk_bin[i + 1]]
        data_chunk = data[chunk_bin[i]:chunk_bin[i + 1]]
        source_size = np.array(list(map(he_size.__getitem__, row_chunk.tolist())))
        target_size = np.array(list(map(he_size.__getitem__, col_chunk.tolist())))
        if filter_mode == 'min size':
            # propotional to minimum size
            metric = data_chunk / np.minimum(source_size, target_size)
        else:
            # jaccard similarity
            metric = data_chunk / (source_size + target_size - data_chunk)
        filter_mask = metric >= threshold
        rows.append(row_chunk[filter_mask])
        cols.append(col_chunk[filter_mask])
        datas.append(metric[filter_mask])

    return np.concatenate(rows), np.concatenate(cols), np.concatenate(datas)
