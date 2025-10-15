import numpy as np
import pandas as pd
import os
import json
# get the sub folders from the root folder 'log'

def get_sub_folders(root):
    sub_folders = []
    for folder in os.listdir(root):
        for sub_folder in os.listdir(os.path.join(root, folder)):
            sub_folders.append(os.path.join(folder, sub_folder))
    return sub_folders

# get the last 10 folders where end with 'tky'
def get_last_tky_folders(root, end_with='tky', n=10):
    sub_folders = get_sub_folders(root)
    tky_folders = [folder for folder in sub_folders if folder.endswith(end_with)]
    return tky_folders[-n:]

def get_last_nyc_folders(root, end_with='nyc', n=10):
    sub_folders = get_sub_folders(root)
    tky_folders = [folder for folder in sub_folders if folder.endswith(end_with)]
    return tky_folders[-n:]

def get_last_ca_folders(root, end_with='ca', n=10):
    sub_folders = get_sub_folders(root)
    tky_folders = [folder for folder in sub_folders if folder.endswith(end_with)]
    return tky_folders[-n:]


# get the last line 2024-06-18 06:20:10 INFO     [Evaluating] Test evaluation result : {'hparam/num_params': 41535641, 'hparam/Recall@1': 0.29099175333976746, 'hparam/Recall@5': 0.5191816091537476, 'hparam/Recall@10': 0.6078431606292725, 'hparam/Recall@20': 0.6710713505744934, 'hparam/NDCG@1': 0.29099175333976746, 'hparam/NDCG@5': 0.41296133399009705, 'hparam/NDCG@10': 0.44171908497810364, 'hparam/NDCG@20': 0.4576885998249054, 'hparam/MAP@1': 0.29099175333976746, 'hparam/MAP@5': 0.37755754590034485, 'hparam/MAP@10': 0.38947567343711853, 'hparam/MAP@20': 0.3938499987125397, 'hparam/MRR': 0.3971254229545593}
def get_last_line(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines[-1]

def log_acc(dataset='tky',n=10):
    # get all the last lines from the last 10 folders for currernt log file
    # get current directory
    root = os.getcwd()
    root = os.path.join(root, 'log')
    last_lines = []
    if dataset == 'tky':
        tky_folders = get_last_tky_folders(root,n=n)
    elif dataset == 'nyc':
        tky_folders = get_last_nyc_folders(root,n=n)
    elif dataset == 'ca':
        tky_folders = get_last_ca_folders(root,n=n)
    else:
        raise ValueError('dataset should be tky or nyc')
    for folder in tky_folders:
        file_path = os.path.join(root, folder, 'train.log')
        last_line = get_last_line(file_path)
        last_lines.append(last_line)

    acc1, acc5, acc10, acc20, mrr = [], [], [], [], []
    for line in last_lines:
        if 'INFO     [Evaluating] Test evaluation result : ' not in line:
            continue
        json_str = line.split('INFO     [Evaluating] Test evaluation result : ')[1]
        json_str = json_str.replace("'", '"')

        data = json.loads(json_str)
        acc1.append(data['hparam/Recall@1'])
        acc5.append(data['hparam/Recall@5'])
        acc10.append(data['hparam/Recall@10'])
        acc20.append(data['hparam/Recall@20'])
        mrr.append(data['hparam/MRR'])

    print(acc1)
    print(acc5)
    print(acc10)
    print(acc20)
    print(mrr)

    means = [np.mean(acc1), np.mean(acc5), np.mean(acc10), np.mean(acc20), np.mean(mrr)]
    maxs = [np.max(acc1), np.max(acc5), np.max(acc10), np.max(acc20), np.max(mrr)]
    mins = [np.min(acc1), np.min(acc5), np.min(acc10), np.min(acc20), np.min(mrr)]


    print([round(m, 4) for m in means])
    print([round(m, 4) for m in maxs])
    print([round(m, 4) for m in mins])

    bias1 = [maxs[i] - means[i] for i in range(5)]

    bias2 = [means[i] - mins[i] for i in range(5)]

    bias = [(bias1[i] + bias2[i])/2 for i in range(5)]

    print ([round(b, 4) for b in bias])

    comment = 'source:channel+ffn, target:channel, dropout 0.4 residual 0.5, attn low rank 248'
    write_path = ('./acc.txt')
    with open(write_path, 'a+') as file:
        # record the last 10 folders
        file.write('*'*20)
        file.write('last 10 folders: ')
        file.write('*'*20)
        file.write('\n')
        file.write('\n'.join(tky_folders))
        # file.write('\n')
        # write max, min, mean, bias
        file.write('\n')
        file.write('max:')
        file.write('\n')
        file.write(str(maxs))
        file.write('\n')
        file.write('min:')
        file.write('\n')
        file.write(str(mins))
        file.write('\n')
        file.write('mean:')
        file.write('\n')
        file.write(str(means))
        file.write('\n')
        file.write('bias:')
        file.write('\n')
        file.write(str(bias))
        file.write('\n')
        file.write(comment)
        file.write('\n')

def search_best(dataset='tky'):
    root = os.getcwd()
    root = os.path.join(root, 'log')
    last_lines = []
    if dataset == 'tky':
        tky_folders = get_last_tky_folders(root, n=30)
    elif dataset == 'nyc':
        tky_folders = get_last_nyc_folders(root, n=30)
    for folder in tky_folders:
        file_path = os.path.join(root, folder, 'train.log')
        last_line = get_last_line(file_path)
        last_lines.append(last_line)

    acc1, acc5, acc10, acc20, mrr = [], [], [], [], []
    for line in last_lines:
        if 'INFO     [Evaluating] Test evaluation result : ' not in line:
            continue
        json_str = line.split('INFO     [Evaluating] Test evaluation result : ')[1]
        json_str = json_str.replace("'", '"')

        data = json.loads(json_str)
        acc1.append(data['hparam/Recall@1'])
        acc5.append(data['hparam/Recall@5'])
        acc10.append(data['hparam/Recall@10'])
        acc20.append(data['hparam/Recall@20'])
        mrr.append(data['hparam/MRR'])

    # get the best acc
    best_idx = np.argmax(acc1)
    best_folder = tky_folders[best_idx]
    print(best_folder)

if __name__ == '__main__':
    log_acc(dataset='nyc', n=10)
    # search_best('nyc')
