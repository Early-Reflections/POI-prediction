import os
import pickle
import pandas as pd
from datetime import datetime
import os.path as osp
from utils import Cfg, get_root_dir
from preprocess import (
    ignore_first,
    only_keep_last,
    id_encode,
    remove_unseen_user_poi,
    FileReader,
    generate_hypergraph_from_file
)
import logging

import numpy as np
from geopy.distance import geodesic

def calculate_distance(row):
    if pd.isna(row['last_latitude']) or pd.isna(row['last_longitude']):
        return np.nan
    return geodesic((row['Latitude'], row['Longitude']), (row['last_latitude'], row['last_longitude'])).meters

def preprocess_nyc(cfg: Cfg, path: bytes, preprocessed_path: bytes) -> pd.DataFrame:
    raw_path = osp.join(path, 'raw')

    df_train = pd.read_csv(osp.join(raw_path, 'NYC_train.csv'))
    df_val = pd.read_csv(osp.join(raw_path, 'NYC_val.csv'))
    df_test = pd.read_csv(osp.join(raw_path, 'NYC_test.csv'))
    df_train['SplitTag'] = 'train'
    df_val['SplitTag'] = 'validation'
    df_test['SplitTag'] = 'test'
    df = pd.concat([df_train, df_val, df_test])
    df.columns = [
        'UserId', 'PoiId', 'PoiCategoryId', 'PoiCategoryCode', 'PoiCategoryName', 'Latitude', 'Longitude',
        'TimezoneOffset', 'UTCTime', 'UTCTimeOffset', 'UTCTimeOffsetWeekday', 'UTCTimeOffsetNormInDayTime',
        'pseudo_session_trajectory_id', 'UTCTimeOffsetNormDayShift', 'UTCTimeOffsetNormRelativeTime', 'SplitTag'
    ]

    # data transformation
    df['UTCTimeOffset'] = df['UTCTimeOffset'].apply(lambda x: datetime.strptime(x[:19], "%Y-%m-%d %H:%M:%S"))
    df['UTCTimeOffsetEpoch'] = df['UTCTimeOffset'].apply(lambda x: x.timestamp())
    df['UTCTimeOffsetWeekday'] = df['UTCTimeOffset'].apply(lambda x: x.weekday())
    df['UTCTimeOffsetHour'] = df['UTCTimeOffset'].apply(lambda x: x.hour)
    df['UTCTimeOffsetDay'] = df['UTCTimeOffset'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df['UserRank'] = df.groupby('UserId')['UTCTimeOffset'].rank(method='first')
    df = df.sort_values(by=['UserId', 'UTCTimeOffset'], ascending=True)

    if cfg.dataset_args.group_type == 'trajectory':
        df['trajectory_id'] = df['pseudo_session_trajectory_id']
    elif cfg.dataset_args.group_type == 'staypoint':
        # extract stay point using time and distance delta
        time_delta = 48 * 60 * 60  # 12 hours
        dist_delta = 3000 # 2000 meters

        df['last_checkin_epoch_time'] = df['UTCTimeOffsetEpoch'].shift(1)
        
        df['last_latitude'] = df['Latitude'].shift(1)
        df['last_longitude'] = df['Longitude'].shift(1)
        
        df['time_to_last'] = df['UTCTimeOffsetEpoch'] - df['last_checkin_epoch_time']
        df['distance_to_last'] = df.apply(calculate_distance, axis=1)

        # get the mean of the distance and time
        mean_time = df['time_to_last'].mean()
        mean_dist = df['distance_to_last'].mean()
        print(f"Mean time: {mean_time}, Mean distance: {mean_dist}")

        # Identifying stay points
        df['is_staypoint'] = (df['time_to_last'] <= time_delta) & (df['distance_to_last'] <= dist_delta)
        # the first check-in is always a stay point
        df.loc[df['UserRank'] == 1, 'is_staypoint'] = True

        df['user_change'] = df['UserId'].ne(df['UserId'].shift())  # Check if UserId changes
        df['staypoint_change'] = df['is_staypoint'].ne(df['is_staypoint'].shift()) | df['user_change']  # Add user change as a new staypoint start

        # According to staypoint_change to assign Stay Point ID
        df['staypoint_id'] = df['staypoint_change'].cumsum()

        # Only keep the data marked as staypoint
        df['staypoint_id'] = np.where(df['is_staypoint'], df['staypoint_id'], np.nan)
        df = df.dropna(subset=['staypoint_id'])
        
        print(f"User count: {df['UserId'].nunique()}, Staypoint count: {df['staypoint_id'].nunique()}")

        # only keep the trjectory with more than 1 staypoint
        # df = df.groupby('staypoint_id').filter(lambda x: len(x) > 1)
        # print(f"User count after filter: {df['UserId'].nunique()}, Staypoint count after filter: {df['staypoint_id'].nunique()}")

        # Check the number of staypoints for each user
        user_staypoint_count = df.groupby('UserId')['staypoint_id'].nunique()
        print(f" Min staypoint count: {user_staypoint_count.min()}, Max staypoint count: {user_staypoint_count.max()}")

        print(f"User count: {df['UserId'].nunique()}, Staypoint count: {df['staypoint_id'].nunique()}")
        # # filter out the users only have one staypoint
        # df = df[df['UserId'].isin(user_staypoint_count[user_staypoint_count > 1].index)]
        # print(f"User count after filter: {df['UserId'].nunique()}, Staypoint count after filter: {df['staypoint_id'].nunique()}")

        # Remap staypoint_id to start from 0 and be continuous
        df['staypoint_id'] = pd.factorize(df['staypoint_id'])[0]

        # Assign a continuous trajectory ID to each staypoint
        df['trajectory_id'] = df['staypoint_id']
    else:
        raise ValueError(f'Wrong group type: {cfg.dataset_args.group_type}')

    # id encoding
    df['check_ins_id'] = df['UTCTimeOffset'].rank(ascending=True, method='first') - 1
    traj_id_le, padding_traj_id = id_encode(df, df, 'trajectory_id')
    # traj_id_le, padding_traj_id = id_encode(df, df, 'pseudo_session_trajectory_id')

    df_train = df[df['SplitTag'] == 'train']
    poi_id_le, padding_poi_id = id_encode(df_train, df, 'PoiId')
    poi_category_le, padding_poi_category = id_encode(df_train, df, 'PoiCategoryId')
    user_id_le, padding_user_id = id_encode(df_train, df, 'UserId')
    hour_id_le, padding_hour_id = id_encode(df_train, df, 'UTCTimeOffsetHour')
    weekday_id_le, padding_weekday_id = id_encode(df_train, df, 'UTCTimeOffsetWeekday')

    # save mapping logic
    with open(osp.join(preprocessed_path, 'label_encoding.pkl'), 'wb') as f:
        pickle.dump([
            poi_id_le, poi_category_le, user_id_le, hour_id_le, weekday_id_le,
            padding_poi_id, padding_poi_category, padding_user_id, padding_hour_id, padding_weekday_id
        ], f)

    # ignore the first for train/validate/test and keep the last for validata/test
    df = ignore_first(df)
    df = only_keep_last(df)
    return df


def preprocess_tky_ca(cfg: Cfg, path: bytes) -> pd.DataFrame:
    if cfg.dataset_args.dataset_name == 'tky':
        raw_file = 'dataset_TSMC2014_TKY.txt'
    else:
        raw_file = 'dataset_gowalla_ca_ne.csv'

    FileReader.root_path = path
    data = FileReader.read_dataset(raw_file, cfg.dataset_args.dataset_name)

    # for ca dataset, after one round of filter, there still be many low frequency pois and users, so do twice
    if cfg.dataset_args.dataset_name == 'ca':
        data = FileReader.do_filter(data, cfg.dataset_args.min_poi_freq, cfg.dataset_args.min_user_freq)
        data = FileReader.do_filter(data, cfg.dataset_args.min_poi_freq, cfg.dataset_args.min_user_freq)
        data = FileReader.split_train_test(data)
    else:
        data = FileReader.do_filter(data, cfg.dataset_args.min_poi_freq, cfg.dataset_args.min_user_freq)
        data = FileReader.split_train_test(data)

    if cfg.dataset_args.group_type == 'trajectory':
        data = FileReader.generate_id(
            data,
            cfg.dataset_args.session_time_interval,
            cfg.dataset_args.do_label_encode,
            cfg.dataset_args.only_last_metric
        )
    elif cfg.dataset_args.group_type == 'staypoint':
        data = FileReader.generate_id_stay_point(
            data,
            cfg.dataset_args.session_time_interval,
            # cfg.dataset_args.session_distance_interval,
            cfg.dataset_args.do_label_encode,
            cfg.dataset_args.only_last_metric
        )
    return data


def preprocess(cfg: Cfg):
    root_path = get_root_dir()
    dataset_name = cfg.dataset_args.dataset_name
    data_path = osp.join(root_path, 'data', dataset_name)
    preprocessed_path = osp.join(data_path, 'preprocessed')
    sample_file = osp.join(preprocessed_path, 'sample.csv')
    train_file = osp.join(preprocessed_path, 'train_sample.csv')
    validate_file = osp.join(preprocessed_path, 'validate_sample.csv')
    test_file = osp.join(preprocessed_path, 'test_sample.csv')

    keep_cols = [
        'check_ins_id', 'UTCTimeOffset', 'UTCTimeOffsetEpoch', 'trajectory_id',
        'query_trajectory_id', 'UserId', 'Latitude', 'Longitude', 'PoiId', 'PoiCategoryId',
        'PoiCategoryName', 'last_checkin_epoch_time'
    ]

    if not osp.exists(preprocessed_path):
        os.makedirs(preprocessed_path)

    # Step 1. preprocess raw files and create sample files including
    # 1. data transformation; 2. id encoding; 3.train/validate/test splitting; 4. remove unseen user or poi
    if not osp.exists(sample_file):
        if 'nyc' == dataset_name:
            keep_cols += ['trajectory_id']
            preprocessed_data = preprocess_nyc(cfg, data_path, preprocessed_path)
        elif 'tky' == dataset_name or 'ca' == dataset_name:
            preprocessed_data = preprocess_tky_ca(cfg, data_path)
        else:
            raise ValueError(f'Wrong dataset name: {dataset_name} ')
        preprocessed_result = remove_unseen_user_poi(preprocessed_data)
        preprocessed_result['sample'].to_csv(sample_file, index=False)
        preprocessed_result['train_sample'][keep_cols].to_csv(train_file, index=False)
        preprocessed_result['validate_sample'][keep_cols].to_csv(validate_file, index=False)
        preprocessed_result['test_sample'][keep_cols].to_csv(test_file, index=False)

    # Step 2. generate hypergraph related data
    if not osp.exists(osp.join(preprocessed_path, 'ci2traj_pyg_data.pt')):
        generate_hypergraph_from_file(sample_file, preprocessed_path, cfg.dataset_args)

    logging.info('[Preprocess] Done preprocessing.')

def preprocess_address(cfg: Cfg):
    root_path = get_root_dir()
    dataset_name = cfg.dataset_args.dataset_name
    data_path = osp.join(root_path, 'data', dataset_name)
    preprocessed_path = osp.join(data_path, 'preprocessed')
    sample_file = osp.join(preprocessed_path, 'sample_address.csv')
    train_file = osp.join(preprocessed_path, 'train_sample_address.csv')
    validate_file = osp.join(preprocessed_path, 'validate_sample_address.csv')
    test_file = osp.join(preprocessed_path, 'test_sample_address.csv')

    keep_cols = [
        'check_ins_id', 'UTCTimeOffset', 'UTCTimeOffsetEpoch', 'pseudo_session_trajectory_id',
        'query_pseudo_session_trajectory_id', 'UserId', 'Latitude', 'Longitude', 'PoiId', 'PoiCategoryId',
        'PoiCategoryName', 'last_checkin_epoch_time','address'
    ]

    if not osp.exists(preprocessed_path):
        os.makedirs(preprocessed_path)

    # Step 1. preprocess raw files and create sample files including
    # 1. data transformation; 2. id encoding; 3.train/validate/test splitting; 4. remove unseen user or poi
    if not osp.exists(sample_file):
        if 'nyc' == dataset_name:
            keep_cols += ['trajectory_id']
            preprocessed_data = preprocess_nyc(data_path, preprocessed_path)
        elif 'tky' == dataset_name or 'ca' == dataset_name:
            preprocessed_data = preprocess_tky_ca(cfg, data_path)
        else:
            raise ValueError(f'Wrong dataset name: {dataset_name} ')

        preprocessed_result = remove_unseen_user_poi(preprocessed_data)
        preprocessed_result['sample'].to_csv(sample_file, index=False)
        preprocessed_result['train_sample'][keep_cols].to_csv(train_file, index=False)
        preprocessed_result['validate_sample'][keep_cols].to_csv(validate_file, index=False)
        preprocessed_result['test_sample'][keep_cols].to_csv(test_file, index=False)

    # Step 2. generate hypergraph related data
    if not osp.exists(osp.join(preprocessed_path, 'ci2traj_pyg_data_address.pt')):
        generate_hypergraph_from_file(sample_file, preprocessed_path, cfg.dataset_args)

    logging.info('[Preprocess] Done preprocessing.')