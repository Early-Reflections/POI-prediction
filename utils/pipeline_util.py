import json
import torch
import logging
import numpy as np
from tqdm import tqdm
import os.path as osp
from metric import (
    recall,
    ndcg,
    map_k,
    mrr,
    class_acc
)


def save_model(model, optimizer, save_variable_list, run_args, argparse_dict):
    """
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    """
    with open(osp.join(run_args.log_path, 'config.json'), 'w') as fjson:
        for key, value in argparse_dict.items():
            if isinstance(value, torch.Tensor):
                argparse_dict[key] = value.numpy().tolist()
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        osp.join(run_args.save_path, 'checkpoint.pt')
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_step_test(model, data, ks=(1, 5, 10, 20)):
    model.eval()
    loss_list = []
    pred_list = []
    label_list = []
    with torch.no_grad():
        for row in tqdm(data):
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
            ranking = torch.sort(out, descending=True)[1]
            pred_list.append(ranking.cpu().detach())
            label_list.append(row.y[:, :1].cpu())
    pred_ = torch.cat(pred_list, dim=0)
    label_ = torch.cat(label_list, dim=0)
    recalls, NDCGs, MAPs = {}, {}, {}
    logging.info(f"[Evaluating] Average loss: {np.mean(loss_list)}")
    for k_ in ks:
        recalls[k_] = recall(label_, pred_, k_).cpu().detach().numpy().tolist()
        NDCGs[k_] = ndcg(label_, pred_, k_).cpu().detach().numpy().tolist()
        MAPs[k_] = map_k(label_, pred_, k_).cpu().detach().numpy().tolist()
        logging.info(f"[Evaluating] Recall@{k_} : {recalls[k_]},\tNDCG@{k_} : {NDCGs[k_]},\tMAP@{k_} : {MAPs[k_]}")
    mrr_res = mrr(label_, pred_).cpu().detach().numpy().tolist()
    class_correct, class_total, class_accuracy = class_acc(torch.squeeze(label_), pred_[:,0].T, pred_.shape[1])
    valid_indices = torch.nonzero(class_total > 0).squeeze().cpu().detach().numpy()
    class_correct = class_correct.cpu().detach().numpy()
    class_total = class_total.cpu().detach().numpy()
    class_accuracy = class_accuracy.cpu().detach().numpy()
    # only keep the index and acc if cls_total is not 0
    class_correct = class_correct[valid_indices]
    class_total = class_total[valid_indices]
    class_accuracy = class_accuracy[valid_indices]
    
    logging.info(f"[Evaluating] MRR : {mrr_res}")
    logging.info(f"[Evaluating] Class Average : {np.mean(class_accuracy)}")
    # return recalls, NDCGs, MAPs, mrr_res, np.mean(loss_list)
    return recalls, NDCGs, MAPs, mrr_res, np.mean(loss_list), valid_indices, class_correct, class_total, class_accuracy


def test_step(model, data, ks=(1, 5, 10, 20)):
    model.eval()
    loss_list = []
    pred_list = []
    label_list = []
    with torch.no_grad():
        for row in tqdm(data):
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
            ranking = torch.sort(out, descending=True)[1]
            pred_list.append(ranking.cpu().detach())
            label_list.append(row.y[:, :1].cpu())
    pred_ = torch.cat(pred_list, dim=0)
    label_ = torch.cat(label_list, dim=0)
    recalls, NDCGs, MAPs = {}, {}, {}
    logging.info(f"[Evaluating] Average loss: {np.mean(loss_list)}")
    for k_ in ks:
        recalls[k_] = recall(label_, pred_, k_).cpu().detach().numpy().tolist()
        NDCGs[k_] = ndcg(label_, pred_, k_).cpu().detach().numpy().tolist()
        MAPs[k_] = map_k(label_, pred_, k_).cpu().detach().numpy().tolist()
        logging.info(f"[Evaluating] Recall@{k_} : {recalls[k_]},\tNDCG@{k_} : {NDCGs[k_]},\tMAP@{k_} : {MAPs[k_]}")
    mrr_res = mrr(label_, pred_).cpu().detach().numpy().tolist()
    logging.info(f"[Evaluating] MRR : {mrr_res}")
    return recalls, NDCGs, MAPs, mrr_res, np.mean(loss_list)


def visualize_step(model, data):
    model.eval()
    attention_weights_0 = []
    attention_weights_1 = []
    attention_weights_2 = []
    with torch.no_grad():
        for row in tqdm(data):
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
            out, _ = model(input_data, label=row.y[:, 0], mode='test')
            # visualize the attention weights
            # get the attention weights of the 3 layers
            
            attention_weights_0.append(model.conv_for_time_filter.feature_attention.attention_weights.cpu().detach())
            attention_weights_1.append(model.conv_list[0].feature_attention.attention_weights.cpu().detach())
            attention_weights_2.append(model.conv_list[1].feature_attention.attention_weights.cpu().detach())

    return attention_weights_0, attention_weights_1, attention_weights_2