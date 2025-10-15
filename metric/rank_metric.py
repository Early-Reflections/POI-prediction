import torch


def recall(lab, prd, k):
    return torch.sum(torch.sum(lab == prd[:, :k], dim=1)) / lab.shape[0]


def ndcg(lab, prd, k):
    exist_pos = torch.nonzero(prd[:, :k] == lab, as_tuple=False)[:, 1] + 1
    dcg = 1 / torch.log2(exist_pos.float() + 1)
    return torch.sum(dcg) / lab.shape[0]


def map_k(lab, prd, k):
    exist_pos = torch.nonzero(prd[:, :k] == lab, as_tuple=False)[:, 1] + 1
    map_tmp = 1 / exist_pos
    return torch.sum(map_tmp) / lab.shape[0]


def mrr(lab, prd):
    exist_pos = torch.nonzero(prd == lab, as_tuple=False)[:, 1] + 1
    mrr_tmp = 1 / exist_pos
    return torch.sum(mrr_tmp) / lab.shape[0]

def class_acc(labels, predictions, num_classes):
    correct = predictions == labels
    class_correct = torch.zeros(num_classes)
    class_totals = torch.zeros(num_classes)

    # Accumulate correct predictions and totals per class
    for i in range(num_classes):
        class_mask = labels == i
        class_correct[i] = correct[class_mask].sum()
        class_totals[i] = class_mask.sum()

    # Avoid division by zero
    class_acc = class_correct / torch.where(class_totals == 0, torch.ones_like(class_totals), class_totals)
    return class_correct, class_totals, class_acc
