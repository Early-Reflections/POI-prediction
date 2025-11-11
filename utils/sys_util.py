import os
import random
import logging
import torch
import numpy as np
import os.path as osp

import matplotlib  # Use a non-interactive backend for headless/server environments
matplotlib.use('Agg')  # Force Agg to avoid Tkinter main-loop errors during training/cleanup
import matplotlib.pyplot as plt  # Import pyplot after backend selection
import seaborn as sns

from pathlib import Path

def get_root_dir(marker_files=("setup.py", ".git", "requirements.txt")):
    path = Path.cwd()
    for parent in [path] + list(path.parents):
        if any((parent / marker).exists() for marker in marker_files):
            return str(parent)
    raise FileNotFoundError("Project root not found.")

def set_logger(args):
    """
    Write logs to checkpoint and console
    """
    if args.do_train:
        log_file = osp.join(args.log_path or args.init_checkpoint, 'train.log')
    else:
        log_file = osp.join(args.log_path or args.init_checkpoint, 'test.log')

    # Remove all handlers associated with the root logger object
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w+'
    )


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def visualize_channel_weight(batch_attention_weights_list, writer, epoch, figure_names):
    """
    传入一个包含多个 batch_attention_weights 的列表，以及对应的图名列表，将每个 attention weight 可视化。
    """
    num_plots = len(batch_attention_weights_list)
    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))

    for idx, (batch_attention_weights, figure_name) in enumerate(zip(batch_attention_weights_list, figure_names)):
        # 计算平均通道注意力
        test_attention_scores = torch.cat(batch_attention_weights, dim=0)  # [total_samples, num_channels, num_channels]
        mean_attention_scores = test_attention_scores.mean(dim=0).numpy()  # [num_channels, num_channels]

        # 绘制热力图
        ax = axes[idx] if num_plots > 1 else axes  # 处理单一图像的情况
        sns.heatmap(mean_attention_scores, annot=True, cmap='viridis', cbar=True, square=True, ax=ax)
        ax.set_title(figure_name)


    # fig.show()
    # fig.savefig(osp.join(get_root_dir(), 'attention_weights.png'))
    writer.add_figure(figure_name, fig, epoch)  # Log figure to TensorBoard
    plt.close(fig)  # Close the figure to free resources and prevent Tkinter warnings

