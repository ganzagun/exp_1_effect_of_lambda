import argparse

import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
import csv
import os
import numpy as np
from models import Model
from dataloader import load_data, load_logits_t, load_out_t, load_out_emb_t
from utils import (
    get_logger,
    get_evaluator,
    set_seed,
    get_training_config,
    check_writable,
    check_readable,
    compute_min_cut_loss,
    graph_split,
)
from train_and_eval import (
    distill_run_transductive, 
    distill_run_inductive,
    compute_ece,
    compute_adaptive_ece,
    compute_brier_score
)
import networkx as nx
from position_encoding import DeepWalk
import dgl
import matplotlib.pyplot as plt
import pandas as pd
import os


def get_csv_path(dataset_name, base_dir="results"):
    """Get the path for the CSV file based on dataset name."""
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{dataset_name}_results.csv")


def create_or_append_to_csv(config, metrics, dataset_name, base_dir="results"):
    """
    Create or append results to a dataset-specific CSV file.
    
    Args:
        config (dict): Configuration parameters
        metrics (dict): Dictionary containing metrics (means and standard deviations)
        dataset_name (str): Name of the dataset
        base_dir (str): Base directory for results
    """
    csv_path = get_csv_path(dataset_name, base_dir)
    file_exists = os.path.exists(csv_path)
    
    # Prepare the row data
    row_data = {
        # Key parameters that might be varied
        'lamb': config['lamb'],
        'temperature': config['temperature'],
        'feature_noise': config['feature_noise'],
        'hidden_dim': config['hidden_dim'],
        'num_layers': config['num_layers'],
        'learning_rate': config['learning_rate'],
        'weight_decay': config['weight_decay'],
        
        # Model identification (constant across runs)
        'teacher': config['teacher'],
        'student': config['student'],
        'dataset': config['dataset'],
        'exp_setting': config['exp_setting'],
        
        # Feature distillation settings
        'feat_distill': config.get('feat_distill', False),
        'feat_distill_weight': config.get('feat_distill_weight', None),
        'teacher_emb_layers': '_'.join(map(str, config.get('teacher_emb_layers', [-1]))),
        'student_emb_layers': '_'.join(map(str, config.get('student_emb_layers', [-1]))),
        'feat_distill_weights': '_'.join(map(str, config.get('feat_distill_weights', [-1]))),
    }
    # Add metrics
    row_data.update(metrics)
    
    # Write to CSV
    with open(csv_path, mode='a' if file_exists else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def compute_metrics_statistics(metrics_list):
    """
    Compute mean and standard deviation for each metric across multiple runs.
    
    Args:
        metrics_list (list): List of dictionaries containing metrics from each run
    
    Returns:
        dict: Dictionary containing mean and std for each metric
    """
    metrics_arrays = {
        key: np.array([run_metrics[key] for run_metrics in metrics_list])
        for key in metrics_list[0].keys()
    }
    
    stats = {}
    for key, values in metrics_arrays.items():
        stats[f'{key}_mean'] = float(np.mean(values))
        stats[f'{key}_std'] = float(np.std(values))
    
    return stats


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--device", type=int, default=-1, help="CUDA device, -1 means CPU")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
        help="Logger levels for run {10: DEBUG, 20: INFO, 30: WARNING}",
    )
    parser.add_argument(
        "--console_log",
        action="store_true",
        help="Set to True to display log info in console",
    )
    parser.add_argument(
        "--output_path", type=str, default="outputs", help="Path to save outputs"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results", help="Directory to save CSV results"
    )
    parser.add_argument(
        "--num_exp", type=int, default=1, help="Repeat how many experiments"
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default="tran",
        help="Experiment setting, one of [tran, ind]",
    )
    parser.add_argument(
        "--eval_interval", type=int, default=1, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Set to True to save the loss curves, trained model, and min-cut loss for the transductive setting",
    )
    parser.add_argument(
        "--teacher_logits",
        default=None,
        type=int
    )

    """Dataset"""
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data")
    parser.add_argument(
        "--labelrate_train",
        type=int,
        default=20,
        help="How many labeled data per class as train set",
    )
    parser.add_argument(
        "--labelrate_val",
        type=int,
        default=30,
        help="How many labeled data per class in valid set",
    )
    parser.add_argument(
        "--split_idx",
        type=int,
        default=0,
        help="For Non-Homo datasets only, one of [0,1,2,3,4]",
    )

    """Model"""
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=".conf.yaml",
        help="Path to model configeration",
    )
    parser.add_argument("--teacher", type=str, default="SAGE", help="Teacher model")
    parser.add_argument("--student", type=str, default="MLP", help="Student model")
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Student model number of layers"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Student model hidden layer dimensions",
    )
    # parser.add_argument("--dropout_ratio", type=float, default=0)
    parser.add_argument(
        "--norm_type", type=str, default="none", help="One of [none, batch, layer]"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature of teacher logits for student training"
    )

    """SAGE Specific"""
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--fan_out",
        type=str,
        default="5,5",
        help="Number of samples for each layer in SAGE. Length = num_layers",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for sampler"
    )

    """Optimization"""
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument(
        "--max_epoch", type=int, default=500, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stop is the score on validation set does not improve for how many epochs",
    )

    """Ablation"""

    parser.add_argument(
        "--feature_noise",
        type=float,
        default=0,
        help="add white noise to features for analysis, value in [0, 1] for noise level",
    )
    parser.add_argument(
        "--split_rate",
        type=float,
        default=0.2,
        help="Rate for graph split, see comment of graph_split for more details",
    )
    parser.add_argument(
        "--compute_min_cut",
        action="store_true",
        help="Set to True to compute and store the min-cut loss",
    )

    """Distiall"""
    parser.add_argument(
        "--lamb",
        type=float,
        default=1,
        help="Parameter balances loss from hard labels and teacher outputs, take values in [0, 1]",
    )
    parser.add_argument(
        "--out_t_path", type=str, default="outputs", help="Path to load teacher outputs"
    )

    # add-up
    parser.add_argument(
        "--dw",
        action="store_true",
        help="Set to True to include deepwalk positional encoding",
    )
    parser.add_argument(
        "--feat_distill",
        action="store_true",
        help="Set to True to include feature distillation loss",
    )
    parser.add_argument(
        "--adv",
        action="store_true",
        help="Set to True to include adversarial feature learning",
    )

    """parameter sensitivity"""
    parser.add_argument(
        "--sensitivity_adv_eps",
        type=float,
        default=-1,
        help="adv_eps for parameter sensitivity",
    )
    parser.add_argument(
        "--sensitivity_dw_emb_size",
        type=int,
        default=-1,
        help="dw_emb_size for parameter sensitivity",
    )
    parser.add_argument(
        "--sensitivity_feat_distill_weight",
        type=float,
        default=-1,
        help="feat_distill_weight for parameter sensitivity",
    )
    parser.add_argument(
        "--teacher_emb_layers",
        nargs="*",
        default=[-2],
        type=int,
        help="Layers for which teacher embeddings are saved (0 indexed)"
    )
    parser.add_argument(
        "--student_emb_layers",
        nargs="*",
        default=[-1],
        type=int,
        help="Layers for which student embeddings are to be distilled (0 indexed; must be of same size as teacher_emb_layers)"
    )
    parser.add_argument(
        "--feat_distill_weights",
        nargs="*",
        default=[-1],
        type=float,
        help="feat_distill_weights for distillation of corresponding teacher and student layer representations (must be of same size as teacher_emb_layers)"
    )

    args = parser.parse_args()
    return args


global_trans_dw_feature = None

def get_features_dw(adj, device, is_transductive, args):
    if args.dataset == 'ogbn-products' or args.dataset == 'ogbn-arxiv':
        print('getting dw for ogbn-arxiv/ogbn-products ...')
        G = adj
    else:
        adj = np.asarray(adj.cpu())
        G = nx.Graph(adj)

    model_emb = DeepWalk(G, walk_length=args.dw_walk_length, num_walks=args.dw_num_walks, workers=4)
    model_emb.train(window_size=args.dw_window_size, iter=args.dw_iter, embed_size=args.dw_emb_size, workers=4)

    emb = model_emb.get_embeddings()  # get embedding vectors
    embeddings = []
    for i in range(len(emb)):
        embeddings.append(emb[i])
    embeddings = np.array(embeddings)
    embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
    if is_transductive:
        global global_trans_dw_feature
        global_trans_dw_feature = embeddings
    else:  # inductive
        pass  # we don't have global_ind_dw_feature since each time seed (data split) is different.
    return embeddings


def run(args):
    """
    Returns:
    score_lst: a list of evaluation results on test set.
    len(score_lst) = 1 for the transductive setting.
    len(score_lst) = 2 for the inductive/production setting.
    """

    """ Set seed, device, and logger """
    set_seed(args.seed)
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = "cpu"

    if args.feature_noise != 0:
        if "noisy_features" not in str(args.output_path):
            args.output_path = Path.cwd().joinpath(
                args.output_path, "noisy_features", f"noise_{args.feature_noise}"
            )
        # Teacher is assumed to be trained on the same noisy features as well.
        # args.out_t_path = args.output_path

    if args.exp_setting == "tran":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "transductive",
            args.dataset,
            f"{args.teacher}_{args.student}",
            f"seed_{args.seed}",
        )
        dw_emb_path = Path.cwd().joinpath(
            args.output_path,
            "transductive",
            args.dataset,
            f"{args.teacher}_{args.student}",
            # "dw_emb.pt"
        )
        out_t_dir = Path.cwd().joinpath(
            args.out_t_path,
            "transductive",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )
    elif args.exp_setting == "ind":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "inductive",
            f"split_rate_{args.split_rate}",
            args.dataset,
            f"{args.teacher}_{args.student}",
            f"seed_{args.seed}",
        )
        out_t_dir = Path.cwd().joinpath(
            args.out_t_path,
            "inductive",
            f"split_rate_{args.split_rate}",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )

    else:
        raise ValueError(f"Unknown experiment setting! {args.exp_setting}")
    args.output_dir = output_dir

    check_writable(output_dir, overwrite=False)
    check_readable(out_t_dir)

    logger = get_logger(output_dir.joinpath("log"), args.console_log, args.log_level)
    logger.info(f"output_dir: {output_dir}")
    logger.info(f"out_t_dir: {out_t_dir}")

    """ Load data and model config"""
    g, labels, idx_train, idx_val, idx_test = load_data(
        args.dataset,
        args.data_path,
        split_idx=args.split_idx,
        seed=args.seed,
        labelrate_train=args.labelrate_train,
        labelrate_val=args.labelrate_val,
    )

    logger.info(f"Total {g.number_of_nodes()} nodes.")
    logger.info(f"Total {g.number_of_edges()} edges.")

    g = g.to(device)
    feats = g.ndata["feat"]
    args.feat_dim = g.ndata["feat"].shape[1]
    args.label_dim = labels.int().max().item() + 1

    if 0 < args.feature_noise <= 1:
        feats = (
                        1 - args.feature_noise
                ) * feats + args.feature_noise * torch.randn_like(feats)

    """ Model config """
    conf = {}
    if args.model_config_path is not None:
        conf = get_training_config(
            # args.model_config_path, args.student, args.dataset
            args.exp_setting + args.model_config_path, args.student, args.dataset, args.teacher, True
        )  # Note: student config
    conf = dict(args.__dict__, **conf)
    conf["device"] = device
    conf["num_distill_layers"] = len(args.student_emb_layers)
    conf["num_layers"] = args.num_layers
    conf["fan_out"] = args.fan_out
    logger.info(f"conf: {conf}")
    # print('conf: ', conf)

    # use parameters from conf
    if 'dw_walk_length' in conf and 'dw_walk_length' not in args:
        args.dw_walk_length = conf['dw_walk_length']
    if 'dw_num_walks' in conf and 'dw_num_walks' not in args:
        args.dw_num_walks = conf['dw_num_walks']
    if 'dw_window_size' in conf and 'dw_window_size' not in args:
        args.dw_window_size = conf['dw_window_size']
    if 'dw_iter' in conf and 'dw_iter' not in args:
        args.dw_iter = conf['dw_iter']
    if 'dw_emb_size' in conf and 'dw_emb_size' not in args:
        args.dw_emb_size = conf['dw_emb_size']
    if args.adv and 'adv_eps' in conf and 'adv_eps' not in args:
        args.adv_eps = conf['adv_eps']
    if args.feat_distill and 'feat_distill_weight' in conf and 'feat_distill_weight' not in args:
        args.feat_distill_weight = conf['feat_distill_weight']

    # parameter sensitivity
    if args.adv and args.sensitivity_adv_eps > 0:
        args.adv_eps = args.sensitivity_adv_eps
    if args.dw and args.sensitivity_dw_emb_size > 0:
        args.dw_emb_size = args.sensitivity_dw_emb_size
    if args.feat_distill and args.sensitivity_feat_distill_weight > 0:
        args.feat_distill_weight = args.sensitivity_feat_distill_weight

    len_position_feature = 0
    if args.exp_setting == "tran":
        idx_l = idx_train
        idx_t = torch.cat([idx_train, idx_val, idx_test])
        distill_indices = (idx_l, idx_t, idx_val, idx_test)

        # position feature (tran)
        if args.dw:
            if args.dataset == 'ogbn-products' or args.dataset == 'ogbn-arxiv':
                dw_emb_path = dw_emb_path.joinpath("dw_emb.pt")
                try:
                    loaded_dw_emb = torch.load(dw_emb_path).to(device)
                    print('load dw_emb successfully!', flush=True)
                    position_feature = loaded_dw_emb
                    len_position_feature = position_feature.shape[-1]
                    feats = torch.cat([feats, position_feature], dim=1)
                except:
                    print('cannot load dw_emb, now try to calculate it ...... ', flush=True)
                    network_g = g.cpu()
                    network_g = network_g.to_networkx()
                    print('done with network_g')
                    dw_emb = get_features_dw(network_g, device, is_transductive=True, args=args)
                    torch.save(dw_emb, dw_emb_path)
                    print('save dw_emb successfully')
                    position_feature = global_trans_dw_feature
                    len_position_feature = position_feature.shape[-1]
                    feats = torch.cat([feats, position_feature], dim=1)

            # cpf datasets
            else:
                if args.cal_dw_flag:
                    adj = g.adj().to_dense()
                    get_features_dw(adj, device, is_transductive=True, args=args)

                position_feature = global_trans_dw_feature
                len_position_feature = position_feature.shape[-1]
                feats = torch.cat([feats, position_feature], dim=1)

    elif args.exp_setting == "ind":
        # Create inductive split
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = graph_split(
            idx_train, idx_val, idx_test, args.split_rate, args.seed
        )
        obs_idx_l = obs_idx_train
        obs_idx_t = torch.cat([obs_idx_train, obs_idx_val, obs_idx_test])
        distill_indices = (
            obs_idx_l,
            obs_idx_t,
            obs_idx_val,
            obs_idx_test,
            idx_obs,
            idx_test_ind,
        )

        # position feature (ind)
        if args.dw:  # We need to run it every time since seed (data split) is different.
            # computation optimized for large datasets.
            if args.dataset == 'ogbn-products':
                dw_emb_path = output_dir.joinpath("dw_emb.pt")  # need to include the seed in the path
                # subgraph
                trained_grapah = dgl.node_subgraph(g, idx_obs.to(device))
                network_g = trained_grapah.cpu()
                network_g = network_g.to_networkx()
                # print('done with network_g')
                position_feature_obs = get_features_dw(network_g, device, is_transductive=True, args=args)
                torch.save(position_feature_obs, dw_emb_path)
                # print('save dw_emb successfully')
                position_feature_obs = position_feature_obs.cpu()

                # change the order of position_feature_obs
                idx_position_feature = idx_obs.tolist()
                position_feature_list_correct_order = [[] for i in range(g.num_nodes())]
                for idx_from_zero, idx_p_f in enumerate(idx_position_feature):
                    temp_position_feature = position_feature_obs[idx_from_zero]
                    position_feature_list_correct_order[idx_p_f].extend(temp_position_feature)

                # get the neighbor for every node
                src_node, dst_node = g.edges()
                src_node = src_node.cpu().tolist()
                dst_node = dst_node.cpu().tolist()
                assert len(src_node) == len(dst_node)
                idx_test_ind_neighbor_dict = {}
                idx_test_ind_list = idx_test_ind.tolist()
                for i in range(len(src_node)):
                    src_node_i = src_node[i]
                    dst_node_i = dst_node[i]
                    if src_node_i not in idx_test_ind_neighbor_dict:
                        idx_test_ind_neighbor_dict[src_node_i] = []
                    idx_test_ind_neighbor_dict[src_node_i].append(dst_node_i)
                    if dst_node_i not in idx_test_ind_neighbor_dict:
                        idx_test_ind_neighbor_dict[dst_node_i] = []
                    idx_test_ind_neighbor_dict[dst_node_i].append(src_node_i)

                # get the dw for test nodes
                for idx_cur_node_id in idx_test_ind_list:
                    try:
                        idx_cur_node_id_neighbor = idx_test_ind_neighbor_dict[idx_cur_node_id]
                        if len(idx_cur_node_id_neighbor):
                            temp_position_feature = torch.mean(position_feature_obs[idx_cur_node_id_neighbor, :], dim=0)
                        else:
                            temp_position_feature = np.zeros(position_feature_obs.shape[-1])
                    except:
                        temp_position_feature = np.zeros(position_feature_obs.shape[-1])

                    position_feature_obs[idx_cur_node_id] = torch.tensor(temp_position_feature, dtype=torch.float32)

                position_feature = position_feature_obs.to(device)
                len_position_feature = position_feature.shape[-1]
                feats = torch.cat([feats, position_feature], dim=1)
                del position_feature_obs, position_feature  # save memory

            # not computation-friendly for large datasets (e.g., ogbn-products).
            elif args.dataset == 'ogbn-arxiv':
                dw_emb_path = output_dir.joinpath("dw_emb.pt")  # include the seed in the path

                # subgraph
                trained_grapah = dgl.node_subgraph(g, idx_obs.to(device))
                network_g = trained_grapah.cpu()
                network_g = network_g.to_networkx()
                # print('done with network_g')
                position_feature_obs = get_features_dw(network_g, device, is_transductive=True, args=args)
                torch.save(position_feature_obs, dw_emb_path)
                # print('save dw_emb successfully')
                position_feature_obs = position_feature_obs.cpu()

                # change the order of position_feature_obs
                idx_position_feature = idx_obs.tolist()
                position_feature_list_correct_order = [[] for i in range(g.num_nodes())]
                for idx_from_zero, idx_p_f in enumerate(idx_position_feature):  # tqdm(
                    temp_position_feature = position_feature_obs[idx_from_zero]
                    position_feature_list_correct_order[idx_p_f].extend(temp_position_feature)

                # get the dw for test nodes
                for idx_cur_node_id in idx_test_ind.tolist():  # tqdm(
                    temp_position_feature = None
                    counter_neighbor_in_obs = 0
                    _, idx_one_in_cur_node = g.out_edges(idx_cur_node_id)
                    idx_one_in_cur_node = idx_one_in_cur_node.tolist()
                    for idx_j in idx_one_in_cur_node:
                        if idx_j not in idx_position_feature:
                            continue
                        if temp_position_feature is None:
                            temp_position_feature = np.asarray(position_feature_list_correct_order[idx_j])
                        else:
                            temp_position_feature += np.asarray(position_feature_list_correct_order[idx_j])
                        counter_neighbor_in_obs += 1
                    # for those we could not find a neighbor
                    if temp_position_feature is None:
                        temp_position_feature = np.zeros(position_feature_obs.shape[-1])
                    else:
                        temp_position_feature /= counter_neighbor_in_obs
                    position_feature_list_correct_order[idx_cur_node_id].extend(temp_position_feature)

                position_feature = torch.tensor(position_feature_list_correct_order, dtype=torch.float32).to(device)
                len_position_feature = position_feature.shape[-1]
                feats = torch.cat([feats, position_feature], dim=1)

            # cpf dataset
            else:
                adj = g.adj().to_dense()
                adj_obs = adj[idx_obs, :][:, idx_obs]

                # take dw from neighbors
                position_feature_obs = get_features_dw(adj_obs, device, is_transductive=False, args=args).cpu()

                idx_position_feature = idx_obs.tolist()
                # change the order of position_feature_obs
                position_feature_list_correct_order = [[] for i in range(len(adj))]
                for idx_from_zero, idx_p_f in enumerate(idx_position_feature):
                    temp_position_feature = position_feature_obs[idx_from_zero]
                    position_feature_list_correct_order[idx_p_f].extend(temp_position_feature)

                # fill in the dw for test nodes
                adj_numpy = adj.cpu().numpy()
                for idx_cur_node_id in idx_test_ind.tolist():
                    temp_position_feature = None
                    counter_neighbor_in_obs = 0
                    idx_one_in_cur_node = np.where(adj_numpy[idx_cur_node_id] == 1)[0]
                    idx_one_in_cur_node = idx_one_in_cur_node.tolist()
                    for idx_j in idx_one_in_cur_node:
                        if idx_j not in idx_position_feature:
                            continue
                        if temp_position_feature is None:
                            temp_position_feature = np.asarray(position_feature_list_correct_order[idx_j])
                        else:
                            temp_position_feature += np.asarray(position_feature_list_correct_order[idx_j])
                        counter_neighbor_in_obs += 1
                    # for those we could not find a neighbor
                    if temp_position_feature is None:
                        temp_position_feature = np.zeros(position_feature_obs.shape[-1])
                    else:
                        temp_position_feature /= counter_neighbor_in_obs
                    position_feature_list_correct_order[idx_cur_node_id].extend(temp_position_feature)

                position_feature = torch.tensor(position_feature_list_correct_order, dtype=torch.float32).to(device)
                len_position_feature = position_feature.shape[-1]
                feats = torch.cat([feats, position_feature], dim=1)

    """ Model init """
    model = Model(conf, args, len_position_feature)

    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion_l = torch.nn.NLLLoss()
    criterion_t = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    evaluator = get_evaluator(conf["dataset"])

    """Load teacher model output"""
    logits_t = load_logits_t(out_t_dir)
    logits_temp_t = logits_t/args.temperature
    out_t = load_out_t(out_t_dir)
    out_emb_t = []

    if args.teacher_emb_layers != [-2]:
        print("here")
        for layer_num in args.teacher_emb_layers:
            out_emb_t.append(load_out_emb_t(out_t_dir, layer_num).to(device))

    # Calculate teacher scores and metrics
    teacher_test_score = evaluator(out_t[idx_test], labels[idx_test])
    teacher_test_ece = compute_ece(torch.softmax(out_t[idx_test], dim=1), labels[idx_test])
    teacher_test_ace = compute_adaptive_ece(torch.softmax(out_t[idx_test], dim=1), labels[idx_test])
    teacher_test_brier = compute_brier_score(torch.softmax(out_t[idx_test], dim=1), labels[idx_test], args.label_dim)
    
    logger.info(f"Teacher metrics on test data:")
    logger.info(f"  Score: {teacher_test_score:.4f}")
    logger.info(f"  ECE: {teacher_test_ece:.4f}")
    logger.info(f"  ACE: {teacher_test_ace:.4f}")
    logger.info(f"  Brier: {teacher_test_brier:.4f}")

    """Data split and run"""
    loss_and_score = []
    if args.exp_setting == "tran":
        out, score_test, ece_test, ace_test, brier_test = distill_run_transductive(
            conf,
            model,
            feats,
            labels,
            logits_temp_t,
            out_emb_t,
            distill_indices,
            criterion_l,
            criterion_t,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
            g,
            args
        )


    elif args.exp_setting == "ind":
        out, score_test, ece_test, ace_test, brier_test = distill_run_inductive(
            conf,
            model,
            feats,
            labels,
            logits_temp_t,
            out_emb_t,
            distill_indices,
            criterion_l,
            criterion_t,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
            g,
            args
        )


    logger.info(
        f"num_layers: {conf['num_layers']}. hidden_dim: {conf['hidden_dim']}. dropout_ratio: {conf['dropout_ratio']}"
    )
    logger.info(f"# params {sum(p.numel() for p in model.parameters())}")

    """ Saving student outputs """
    out_np = out.detach().cpu().numpy()
    np.savez(output_dir.joinpath("out"), out_np)

    """ Saving loss curve and model """
    if args.save_results:
        # Loss curves
        loss_and_score = np.array(loss_and_score)
        np.savez(output_dir.joinpath("loss_and_score"), loss_and_score)

        # Model
        torch.save(model.state_dict(), output_dir.joinpath("model.pth"))

    """ Saving min-cut loss"""
    if args.exp_setting == "tran" and args.compute_min_cut:
        min_cut = compute_min_cut_loss(g, out)
        # with open(output_dir.parent.joinpath("min_cut_loss"), "a+") as f:
        #     f.write(f"{min_cut :.4f}\n")
        print('min_cut: ', min_cut, flush=True)

    return score_test, ece_test, ace_test, brier_test


def repeat_run(args):
    # Lists for student metrics
    scores = []
    eces = []
    aces = []
    briers = []
    
    # Lists for teacher metrics
    teacher_scores = []
    teacher_eces = []
    teacher_aces = []
    teacher_briers = []
    
    for seed in range(args.num_exp):
        if seed == 0:
            cal_dw_flag = True
        else:
            cal_dw_flag = False
        args.cal_dw_flag = cal_dw_flag
        args.seed = seed
        temp_score, temp_ece, temp_ace, temp_brier = run(args)
        
        # Student metrics
        scores.append(temp_score)
        eces.append(temp_ece)
        aces.append(temp_ace)
        briers.append(temp_brier)
        
        # Load teacher output for this seed
        if args.exp_setting == "tran":
            out_t_dir = Path.cwd().joinpath(
                args.out_t_path,
                "transductive",
                args.dataset,
                args.teacher,
                f"seed_{args.seed}",
            )
        else:
            out_t_dir = Path.cwd().joinpath(
                args.out_t_path,
                "inductive",
                f"split_rate_{args.split_rate}",
                args.dataset,
                args.teacher,
                f"seed_{args.seed}",
            )
            
        # Load and evaluate teacher
        out_t = load_out_t(out_t_dir)
        _, labels, idx_train, idx_val, idx_test = load_data(
            args.dataset,
            args.data_path,
            split_idx=args.split_idx,
            seed=args.seed,
            labelrate_train=args.labelrate_train,
            labelrate_val=args.labelrate_val,
        )
        
        if torch.cuda.is_available() and args.device >= 0:
            device = torch.device("cuda:" + str(args.device))
            labels = labels.to(device)
            out_t = out_t.to(device)
        
        evaluator = get_evaluator(args.dataset)
        teacher_test_score = evaluator(out_t[idx_test], labels[idx_test])
        teacher_test_ece = compute_ece(torch.softmax(out_t[idx_test], dim=1), labels[idx_test])
        teacher_test_ace = compute_adaptive_ece(torch.softmax(out_t[idx_test], dim=1), labels[idx_test])
        teacher_test_brier = compute_brier_score(torch.softmax(out_t[idx_test], dim=1), labels[idx_test], args.label_dim)
        
        teacher_scores.append(teacher_test_score)
        teacher_eces.append(teacher_test_ece)
        teacher_aces.append(teacher_test_ace)
        teacher_briers.append(teacher_test_brier)

    # Convert to numpy arrays and compute statistics
    scores_np = np.array(scores)
    eces_np = np.array(eces)
    aces_np = np.array(aces)
    briers_np = np.array(briers)
    
    teacher_scores_np = np.array(teacher_scores)
    teacher_eces_np = np.array(teacher_eces)
    teacher_aces_np = np.array(teacher_aces)
    teacher_briers_np = np.array(teacher_briers)
    
    return (
        # Student metrics
        scores_np.mean(axis=0),
        scores_np.std(axis=0),
        eces_np.mean(axis=0),
        eces_np.std(axis=0),
        aces_np.mean(axis=0),
        aces_np.std(axis=0),
        briers_np.mean(axis=0),
        briers_np.std(axis=0),
        # Teacher metrics
        teacher_scores_np.mean(),
        teacher_scores_np.std(),
        teacher_eces_np.mean(),
        teacher_eces_np.std(),
        teacher_aces_np.mean(),
        teacher_aces_np.std(),
        teacher_briers_np.mean(),
        teacher_briers_np.std()
    )
    
def main():
    args = get_args()
    assert args.num_exp > 1

    (
        # Student metrics
        score_mean,
        score_std,
        ece_mean,
        ece_std,
        ace_mean,
        ace_std,
        brier_mean,
        brier_std,
        # Teacher metrics
        teacher_score_mean,
        teacher_score_std,
        teacher_ece_mean,
        teacher_ece_std,
        teacher_ace_mean,
        teacher_ace_std,
        teacher_brier_mean,
        teacher_brier_std
    ) = repeat_run(args)

    # Print results
    print("\nTeacher Results:")
    print(f"Score Mean and Std: {teacher_score_mean:.4f} ± {teacher_score_std:.4f}")
    print(f"ECE Mean and Std: {teacher_ece_mean:.4f} ± {teacher_ece_std:.4f}")
    print(f"ACE Mean and Std: {teacher_ace_mean:.4f} ± {teacher_ace_std:.4f}")
    print(f"Brier Mean and Std: {teacher_brier_mean:.4f} ± {teacher_brier_std:.4f}")
    
    print(f"\nStudent Results ({args.exp_setting}):")
    print(f"Score Mean and Std: {score_mean:.4f} ± {score_std:.4f}")
    print(f"ECE Mean and Std: {ece_mean:.4f} ± {ece_std:.4f}")
    print(f"ACE Mean and Std: {ace_mean:.4f} ± {ace_std:.4f}")
    print(f"Brier Mean and Std: {brier_mean:.4f} ± {brier_std:.4f}")

    # Prepare metrics dictionary (averages across seeds)
    metrics = {
        # Student metrics (averaged over seeds)
        'accuracy': float(score_mean),
        'accuracy_std': float(score_std),
        'ece': float(ece_mean),
        'ece_std': float(ece_std),
        'ace': float(ace_mean),
        'ace_std': float(ace_std),
        'brier': float(brier_mean),
        'brier_std': float(brier_std),
        # Teacher metrics (averaged over seeds)
        'teacher_accuracy': float(teacher_score_mean),
        'teacher_accuracy_std': float(teacher_score_std),
        'teacher_ece': float(teacher_ece_mean),
        'teacher_ece_std': float(teacher_ece_std),
        'teacher_ace': float(teacher_ace_mean),
        'teacher_ace_std': float(teacher_ace_std),
        'teacher_brier': float(teacher_brier_mean),
        'teacher_brier_std': float(teacher_brier_std),
        # Number of seeds used for averaging
        'num_seeds': args.num_exp
    }

    # Prepare config dictionary with all hyperparameters
    config = {
        # Experiment settings
        'dataset': args.dataset,
        'exp_setting': args.exp_setting,
        'seed': args.seed,
        
        # Model architecture
        'teacher': args.teacher,
        'student': args.student,
        'num_layers': args.num_layers,
        'hidden_dim': args.hidden_dim,
        
        # Training hyperparameters
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'temperature': args.temperature,
        'lamb': args.lamb,
        'patience': args.patience,
        'max_epoch': args.max_epoch,
        
        # Feature parameters
        'feature_noise': args.feature_noise,
        'feat_distill': args.feat_distill,
        'feat_distill_weight': getattr(args, 'feat_distill_weight', None),
        
        # Layer configurations
        'teacher_emb_layers': args.teacher_emb_layers,
        'student_emb_layers': args.student_emb_layers,
        'feat_distill_weights': args.feat_distill_weights,
        
        # Additional parameters
        'norm_type': args.norm_type,
        'batch_size': args.batch_size,
        'labelrate_train': args.labelrate_train,
        'labelrate_val': args.labelrate_val,
        'teacher_logits': args.teacher_logits,
        
        # DW parameters
        'dw': args.dw,
        'dw_walk_length': getattr(args, 'dw_walk_length', None),
        'dw_num_walks': getattr(args, 'dw_num_walks', None),
        'dw_window_size': getattr(args, 'dw_window_size', None),
        'dw_iter': getattr(args, 'dw_iter', None),
        'dw_emb_size': getattr(args, 'dw_emb_size', None)
    }

    # Save to dataset-specific CSV
    create_or_append_to_csv(
        config=config,
        metrics=metrics,
        dataset_name=args.dataset,
        base_dir=args.results_dir
    )

    # Keep the original exp_results file for backward compatibility
    score_str = "".join([f"{score_mean:.4f}\t"] + [f"{score_std:.4f}\t"])
    with open(args.output_dir.parent.joinpath("exp_results"), "a+") as f:
        f.write(f"{score_str}\n")


if __name__ == "__main__":
    args = get_args()
    main()
