import logging
import os
import os.path as osp
import datetime
import torch
import pandas as pd
import random
import argparse
from preprocess import preprocess
from utils import seed_torch, set_logger, Cfg, count_parameters, test_step_test
from layer import NeighborSampler
from dataset import LBSNDataset
from model import STHGCN, SequentialTransformer
from tqdm import tqdm
import logging
import torch
import pandas as pd
import os.path as osp
from utils import get_root_dir, construct_slots
import numpy as np


def user_split(dataset_name, split_ratio=0.3):
    """
    Split the users into training and testing set
    """

    # read from data\nyc\preprocessed\sample.csv
    sample_path = osp.join('data', dataset_name, 'preprocessed', 'train_sample.csv')
    sample_df = pd.read_csv(sample_path)

    # group by user_id, trajectory_id, and get the count of trajectory_id
    user_traj_count = sample_df.groupby('UserId')['trajectory_id'].count().reset_index()
    user_traj_count.columns = ['UserId', 'TrajectoryCount']
    sorted_user_traj_count = user_traj_count.sort_values(by='TrajectoryCount', ascending=False)

    # get the top 30% of the users
    top_user = sorted_user_traj_count.iloc[:int(len(sorted_user_traj_count) * split_ratio)]['UserId'].values

    # get the bottom 30% of the users
    bottom_user = sorted_user_traj_count.iloc[int(len(sorted_user_traj_count) * (1-split_ratio)):]['UserId'].values

    # get middle 40% of the users
    middle_user = sorted_user_traj_count.iloc[int(len(sorted_user_traj_count) * split_ratio):int(len(sorted_user_traj_count) * (1-split_ratio))]['UserId'].values

    # read test_sample.csv
    test_sample_path = osp.join('data', dataset_name, 'preprocessed', 'test_sample.csv')
    test_sample_df = pd.read_csv(test_sample_path)

    # valid_sample_path = osp.join('data', dataset_name, 'preprocessed', 'validate_sample.csv')
    # valid_sample_df = pd.read_csv(valid_sample_path)

    # # concat the test_sample_df and valid_sample_df
    # test_sample_df = pd.concat([test_sample_df, valid_sample_df], ignore_index=True)

    # assign the split tag to the test_sample_df
    # top user
    test_sample_df_top = test_sample_df.loc[test_sample_df['UserId'].isin(top_user)]
    test_sample_df_middle = test_sample_df.loc[test_sample_df['UserId'].isin(middle_user)]
    test_sample_df_bottom = test_sample_df.loc[test_sample_df['UserId'].isin(bottom_user)]

    # write the top, middle, bottom user into csv file
    top_path = osp.join('data', dataset_name, 'preprocessed', 'test_sample_active.csv')
    test_sample_df_top.to_csv(top_path, index=False)

    middle_path = osp.join('data', dataset_name, 'preprocessed', 'test_sample_normal.csv')
    test_sample_df_middle.to_csv(middle_path, index=False)

    bottom_path = osp.join('data', dataset_name, 'preprocessed', 'test_sample_inactive.csv')
    test_sample_df_bottom.to_csv(bottom_path, index=False)





if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--yaml_file', help='The configuration file.', required=True)
    parser.add_argument('--multi_run_mode', help='Run multiple experiments with the same config.', action='store_true')
    args = parser.parse_args()
    conf_file = args.yaml_file

    cfg = Cfg(conf_file)

    sizes = [int(i) for i in cfg.model_args.sizes.split('-')]
    cfg.model_args.sizes = sizes

    # cuda setting
    if int(cfg.run_args.gpu) >= 0:
        device = 'cuda:' + str(cfg.run_args.gpu)
    else:
        device = 'cpu'
    cfg.run_args.device = device

    # for multiple runs, seed is replaced with random value
    if args.multi_run_mode:
        cfg.run_args.seed = None
    if cfg.run_args.seed is None:
        seed = random.randint(0, 100000000)
    else:
        seed = int(cfg.run_args.seed)

    seed_torch(seed)

    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    cfg.run_args.log_path = f'log/{current_time}/{cfg.dataset_args.dataset_name}'

    if not osp.isdir(cfg.run_args.log_path):
        os.makedirs(cfg.run_args.log_path)

    set_logger(cfg.run_args)

    hparam_dict = {}
    for group, hparam in cfg.__dict__.items():
        hparam_dict.update(hparam.__dict__)
    hparam_dict['seed'] = seed
    hparam_dict['sizes'] = '-'.join([str(item) for item in cfg.model_args.sizes])

    # Preprocess data
    # preprocess(cfg)
    # preprocess_address(cfg)

    user_split(cfg.dataset_args.dataset_name)

    # Initialize dataset
    lbsn_dataset = LBSNDataset(cfg)
    cfg.dataset_args.spatial_slots = lbsn_dataset.spatial_slots
    cfg.dataset_args.num_user = lbsn_dataset.num_user
    cfg.dataset_args.num_poi = lbsn_dataset.num_poi
    cfg.dataset_args.num_category = lbsn_dataset.num_category
    cfg.dataset_args.padding_poi_id = lbsn_dataset.padding_poi_id
    cfg.dataset_args.padding_user_id = lbsn_dataset.padding_user_id
    cfg.dataset_args.padding_poi_category = lbsn_dataset.padding_poi_category
    cfg.dataset_args.padding_hour_id = lbsn_dataset.padding_hour_id
    cfg.dataset_args.padding_weekday_id = lbsn_dataset.padding_weekday_id

    # Initialize neighbor sampler(dataloader)
    sampler_train, sampler_validate, sampler_test = None, None, None
    # sampler_train = NeighborSampler(
    #     lbsn_dataset.x,
    #     lbsn_dataset.edge_index,
    #     lbsn_dataset.edge_attr,
    #     intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
    #     inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
    #     edge_t=lbsn_dataset.edge_t,
    #     edge_delta_t=lbsn_dataset.edge_delta_t,
    #     edge_type=lbsn_dataset.edge_type,
    #     sizes=sizes,
    #     sample_idx=lbsn_dataset.sample_idx_train,
    #     node_idx=lbsn_dataset.node_idx_train,
    #     edge_delta_s=lbsn_dataset.edge_delta_s,
    #     max_time=lbsn_dataset.max_time_train,
    #     label=lbsn_dataset.label_train,
    #     batch_size=cfg.run_args.batch_size,
    #     num_workers=0 if device == 'cpu' else cfg.run_args.num_workers,
    #     shuffle=True,
    #     pin_memory=True
    # )


    # sampler_validate = NeighborSampler(
    #     lbsn_dataset.x,
    #     lbsn_dataset.edge_index,
    #     lbsn_dataset.edge_attr,
    #     intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
    #     inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
    #     edge_t=lbsn_dataset.edge_t,
    #     edge_delta_t=lbsn_dataset.edge_delta_t,
    #     edge_type=lbsn_dataset.edge_type,
    #     sizes=sizes,
    #     sample_idx=lbsn_dataset.sample_idx_valid,
    #     node_idx=lbsn_dataset.node_idx_valid,
    #     edge_delta_s=lbsn_dataset.edge_delta_s,
    #     max_time=lbsn_dataset.max_time_valid,
    #     label=lbsn_dataset.label_valid,
    #     batch_size=cfg.run_args.eval_batch_size,
    #     num_workers=0 if device == 'cpu' else cfg.run_args.num_workers,
    #     shuffle=False,
    #     pin_memory=True
    # )

    if cfg.run_args.do_test:
        sampler_test = NeighborSampler(
            lbsn_dataset.x,
            lbsn_dataset.edge_index,
            lbsn_dataset.edge_attr,
            intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
            inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
            edge_t=lbsn_dataset.edge_t,
            edge_delta_t=lbsn_dataset.edge_delta_t,
            edge_type=lbsn_dataset.edge_type,
            sizes=sizes,
            sample_idx=lbsn_dataset.sample_idx_test,
            node_idx=lbsn_dataset.node_idx_test,
            edge_delta_s=lbsn_dataset.edge_delta_s,
            max_time=lbsn_dataset.max_time_test,
            label=lbsn_dataset.label_test,
            batch_size=cfg.run_args.eval_batch_size,
            num_workers=0 if device == 'cpu' else cfg.run_args.num_workers,
            shuffle=False,
            pin_memory=True
        )

    # # check the distribution of the label in the lbsn_dataset


    # # get the label=data.y[:, 0] of all the samples in the training, validation, and test set
    # train_label = []   
    # for data in tqdm(sampler_train):
    #     label=data.y[:, 0]
    #     train_label.append(label)

    # valid_label = []
    # for data in tqdm(sampler_validate):
    #     label=data.y[:, 0]
    #     valid_label.append(label)

    # test_label = []
    # for data in tqdm(sampler_test):
    #     label=data.y[:, 0]
    #     test_label.append(label)
        
    # train_label = torch.cat(train_label).tolist()
    # valid_label = torch.cat(valid_label).tolist()
    # test_label = torch.cat(test_label).tolist()

    # all_label = train_label + valid_label + test_label

    # from collections import Counter
    # label_count = Counter(all_label)

    # # get the count of original label from lbsn_dataset
    # original_label = lbsn_dataset.label_train[:, 0].tolist() + lbsn_dataset.label_valid[:, 0].tolist() + lbsn_dataset.label_test[:, 0].tolist()
    # original_label_count = Counter(original_label)


    if cfg.model_args.model_name == 'sthgcn':
        model = STHGCN(cfg)
    elif cfg.model_args.model_name == 'seq_transformer':
        model = SequentialTransformer(cfg)
    else:
        raise NotImplementedError(
            f'[Training] Model {cfg.model_args.name}, please choose from ["sthgcn", "seq_transformer"]'
        )

    model = model.to(device)

    if cfg.run_args.do_test:
        logging.info('[Evaluating] Start evaluating on test set...')

        checkpoint = torch.load(osp.join(cfg.run_args.init_checkpoint, 'checkpoint.pt'))
        logging.info(f'[Evaluating] Load checkpoint from {cfg.run_args.init_checkpoint}')
        logging.info(f'Cold Start Evaluation on Top dataset')
        model.load_state_dict(checkpoint['model_state_dict'])
        num_params = count_parameters(model)
        logging.info(f'hparam/num_params: {num_params}')

        # Create a function to save prediction results
        def save_prediction_results(sampler, result_file_name):
            all_predictions = []
            all_labels = []
            all_user_ids = []
            
            for batch in tqdm(sampler, desc=f"Collecting predictions for {result_file_name}"):
                batch = batch.to(device)
                with torch.no_grad():
                    out = model(batch)
                    predictions = out.cpu().numpy()
                    labels = batch.y[:, 0].cpu().numpy()
                    user_ids = batch.x[batch.node_idx][:, 0].cpu().numpy()
                    
                    all_predictions.append(predictions)
                    all_labels.append(labels)
                    all_user_ids.append(user_ids)
            
            # Convert to appropriate format and save
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            all_user_ids = np.concatenate(all_user_ids, axis=0)
            
            results_df = pd.DataFrame({
                'user_id': all_user_ids,
                'true_label': all_labels,
            })
            
            # Add top-k predictions as columns
            top_k = 20  # Adjust as needed
            top_indices = np.argsort(-all_predictions, axis=1)[:, :top_k]
            
            for k in range(top_k):
                results_df[f'pred_{k+1}'] = top_indices[:, k]
                results_df[f'pred_{k+1}_score'] = [all_predictions[i, idx] for i, idx in enumerate(top_indices[:, k])]
            
            # Save to CSV
            results_path = osp.join(cfg.run_args.log_path, result_file_name)
            results_df.to_csv(results_path, index=False)
            logging.info(f'Saved prediction results to {results_path}')
            
            return results_df

        # Save test predictions
        test_results = save_prediction_results(sampler_test, 'test_predictions.csv')

        recall_res, ndcg_res, map_res, mrr_res, loss, valid_indices, class_correct, class_total, class_accuracy = test_step_test(model, sampler_test)

        metric_dict = {
            'hparam/Recall@1': recall_res[1],
            'hparam/Recall@5': recall_res[5],
            'hparam/Recall@10': recall_res[10],
            'hparam/Recall@20': recall_res[20],
            'hparam/NDCG@1': ndcg_res[1],
            'hparam/NDCG@5': ndcg_res[5],
            'hparam/NDCG@10': ndcg_res[10],
            'hparam/NDCG@20': ndcg_res[20],
            'hparam/MAP@1': map_res[1],
            'hparam/MAP@5': map_res[5],
            'hparam/MAP@10': map_res[10],
            'hparam/MAP@20': map_res[20],
            'hparam/MRR': mrr_res,
        }
        logging.info(f'[Evaluating] Test evaluation result : {metric_dict}')

        print('Recall:', recall_res)
        print('NDCG:', ndcg_res)
        print('MAP:', map_res)
        print('MRR:', mrr_res)

        # write to cold_start_result.txt file
        with open('cold_start_result.txt', 'a+') as file:
            file.write(f'Cold Start Evaluation on normal dataset\n')
            file.write(f'num_params: {num_params}\n')
            file.write(f'Test evaluation result : {metric_dict}\n')
            file.write('\n')
        
        # Save detailed metrics with predictions
        metrics_df = pd.DataFrame({
            'valid_indices': valid_indices,
            'class_correct': class_correct,
            'class_total': class_total,
            'class_accuracy': class_accuracy
        })
        metrics_df.to_csv(osp.join(cfg.run_args.log_path, 'class_metrics.csv'), index=False)