import logging
import os
import os.path as osp
import datetime
import torch
import pandas as pd
import numpy as np
import random
import argparse
from torch.utils.tensorboard import SummaryWriter
from preprocess import preprocess
from utils import seed_torch, set_logger, Cfg, count_parameters, test_step, visualize_channel_weight, visualize_step, get_root_dir
from layer import NeighborSampler
from dataset import LBSNDataset
from model import STHGCN, SequentialTransformer
from tqdm import tqdm


def _find_latest_checkpoint_dir(dataset_name: str):
    base = 'tensorboard'
    if not osp.isdir(base):
        return None
    candidates = []
    for ts in os.listdir(base):
        d = osp.join(base, ts, dataset_name)
        if osp.isfile(osp.join(d, 'checkpoint.pt')):
            candidates.append(d)
    if not candidates:
        return None
    candidates.sort(key=lambda p: osp.getmtime(p), reverse=True)
    return candidates[0]



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

    preprocess(cfg)

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
    sampler_train = NeighborSampler(
        lbsn_dataset.x,
        lbsn_dataset.edge_index,
        lbsn_dataset.edge_attr,
        intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
        inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
        edge_t=lbsn_dataset.edge_t,
        edge_delta_t=lbsn_dataset.edge_delta_t,
        edge_type=lbsn_dataset.edge_type,
        sizes=sizes,
        sample_idx=lbsn_dataset.sample_idx_train,
        node_idx=lbsn_dataset.node_idx_train,
        edge_delta_s=lbsn_dataset.edge_delta_s,
        max_time=lbsn_dataset.max_time_train,
        label=lbsn_dataset.label_train,
        batch_size=cfg.run_args.batch_size,
        num_workers=0 if device == 'cpu' else cfg.run_args.num_workers,
        shuffle=True,
        pin_memory=True
    )


    sampler_validate = NeighborSampler(
        lbsn_dataset.x,
        lbsn_dataset.edge_index,
        lbsn_dataset.edge_attr,
        intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
        inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
        edge_t=lbsn_dataset.edge_t,
        edge_delta_t=lbsn_dataset.edge_delta_t,
        edge_type=lbsn_dataset.edge_type,
        sizes=sizes,
        sample_idx=lbsn_dataset.sample_idx_valid,
        node_idx=lbsn_dataset.node_idx_valid,
        edge_delta_s=lbsn_dataset.edge_delta_s,
        max_time=lbsn_dataset.max_time_valid,
        label=lbsn_dataset.label_valid,
        batch_size=cfg.run_args.eval_batch_size,
        num_workers=0 if device == 'cpu' else cfg.run_args.num_workers,
        shuffle=False,
        pin_memory=True
    )

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

    # check the distribution of the label in the lbsn_dataset


    # get the label=data.y[:, 0] of all the samples in the training, validation, and test set
    train_label = []   
    for data in tqdm(sampler_train):
        label=data.y[:, 0]
        train_label.append(label)

    valid_label = []
    for data in tqdm(sampler_validate):
        label=data.y[:, 0]
        valid_label.append(label)

    test_label = []
    for data in tqdm(sampler_test):
        label=data.y[:, 0]
        test_label.append(label)
        
    train_label = torch.cat(train_label).tolist()
    valid_label = torch.cat(valid_label).tolist()
    test_label = torch.cat(test_label).tolist()

    all_label = train_label + valid_label + test_label

    from collections import Counter
    label_count = Counter(all_label)

    # get the count of original label from lbsn_dataset
    original_label = lbsn_dataset.label_train[:, 0].tolist() + lbsn_dataset.label_valid[:, 0].tolist() + lbsn_dataset.label_test[:, 0].tolist()
    original_label_count = Counter(original_label)


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

        ckpt_dir = cfg.run_args.init_checkpoint or _find_latest_checkpoint_dir(cfg.dataset_args.dataset_name)
        if ckpt_dir is None:
            raise FileNotFoundError("No checkpoint found. Set run_args.init_checkpoint in YAML or train a model first.")
        checkpoint = torch.load(osp.join(ckpt_dir, 'checkpoint.pt'))
        logging.info(f'[Evaluating] Load checkpoint from {ckpt_dir}')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        loss_list = []
        pred_list = []
        label_list = []
        # top_k = min(20, cfg.dataset_args.num_poi)
        top_k = 20

        with torch.no_grad():
            for row in tqdm(sampler_test):
                split_index = torch.max(row.adjs_t[1].storage.row()).tolist()
                row = row.to(model.device)

                input_data = {
                    'x': row.x,
                    'edge_index': row.adjs_t,
                    'edge_attr': row.edge_attrs,
                    'split_index': split_index,
                    'delta_ts': row.edge_delta_ts,
                    'delta_ss': row.edge_delta_ss,
                    'edge_type': row.edge_types
                }

                out, loss = model(input_data, label=row.y[:, 0], mode='test')
                loss_list.append(loss.cpu().detach().numpy().tolist())
                topk_indices = torch.topk(out, k=top_k, dim=1, largest=True, sorted=True)[1]
                pred_list.append(topk_indices.cpu().detach())
                label_list.append(row.y[:, :1].cpu())
        pred_ = torch.cat(pred_list, dim=0)
        label_ = torch.cat(label_list, dim=0)
        recalls, NDCGs, MAPs = {}, {}, {}
        logging.info(f"[Evaluating] Average loss: {np.mean(loss_list)}")

        test_sample_path = osp.join(
            get_root_dir(),
            'data',
            cfg.dataset_args.dataset_name,
            'preprocessed',
            'test_sample.csv'
        )

        # save the pred_ and label_ into csv file
        if osp.exists(test_sample_path):
            test_df = pd.read_csv(test_sample_path).reset_index(drop=True)
            if len(test_df) != pred_.size(0):
                logging.warning(
                    '[Evaluating] Length mismatch between test samples and predictions '
                    f'({len(test_df)} vs {pred_.size(0)}). Skipping combined export.'
                )
            else:
                for idx in range(top_k):
                    test_df[f'pred_top_{idx + 1}'] = pred_[:, idx].tolist()
                combined_path = osp.join(cfg.run_args.log_path, f'test_predictions_top{top_k}.csv')
                test_df.to_csv(combined_path, index=False)
                logging.info(f'[Evaluating] Saved top-{top_k} predictions with original data to {combined_path}')
        else:
            logging.warning(
                f'[Evaluating] test_sample.csv not found at {test_sample_path}, unable to store predictions with original data.'
            )
        # # %% visulize the channel weight
        # attention_weights_0, attention_weights_1, attention_weights_2= visualize_step(model, sampler_test)
        # visualize_channel_weight([attention_weights_0, attention_weights_1, attention_weights_2], 'attention_weights')