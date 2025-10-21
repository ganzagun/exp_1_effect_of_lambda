import argparse
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from models import Model
from dataloader import load_data
from utils import (
    get_logger,
    get_evaluator,
    set_seed,
    get_training_config,
    check_writable,
    compute_min_cut_loss,
    graph_split,
)
from train_and_eval import run_transductive, run_inductive
import matplotlib.pyplot as plt
import pandas as pd


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
        # default="./tran.conf.yaml",
        default=".conf.yaml",
        help="Path to model configeration",
    )
    parser.add_argument("--teacher", type=str, default="SAGE", help="Teacher model")
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Model number of layers"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Model hidden layer dimensions"
    )
    parser.add_argument("--dropout_ratio", type=float, default=0)
    parser.add_argument(
        "--norm_type", type=str, default="none", help="One of [none, batch, layer]"
    )
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=[1.0], help="Temperatures of teacher logits for student training"
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

    """Layer Distillation"""
    parser.add_argument(
        "--teacher_emb_layers", type=int, nargs="*", default=[-1], help="Layers for which teacher embeddings are to be saved (0 indexed)"
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

    args = parser.parse_args()
    return args


def run(args):
    """
    Returns:
    score_lst: a list of evaluation results on test set.
    len(score_lst) = 1 for the transductive setting.
    len(score_lst) = 2 for the inductive/production setting.
    """

    """ Set seed, device, and logger """
    # print('args.seed: ', args.seed)
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

    if args.exp_setting == "tran":
        output_dir = Path.cwd().joinpath(
            args.output_path,
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
            args.teacher,
            f"seed_{args.seed}",
        )
    else:
        raise ValueError(f"Unknown experiment setting! {args.exp_setting}")
    args.output_dir = output_dir

    check_writable(output_dir, overwrite=False)
    logger = get_logger(output_dir.joinpath("log"), args.console_log, args.log_level)
    logger.info(f"output_dir: {output_dir}")

    """ Load data """
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
        conf = get_training_config(args.exp_setting + args.model_config_path, args.teacher, args.dataset)
    conf = dict(args.__dict__, **conf)
    conf["device"] = device
    conf["num_layers"] = args.num_layers
    conf["teacher_hidden_dim"] = None
    conf["num_distill_layers"] = None
    conf["fan_out"] = args.fan_out
    logger.info(f"conf: {conf}")

    """ Model init """
    model = Model(conf, args)
    optimizer = optim.Adam(
        model.parameters(), lr=conf["learning_rate"], weight_decay=conf["weight_decay"]
    )
    criterion = torch.nn.NLLLoss()
    evaluator = get_evaluator(conf["dataset"])

    """ Data split and run """
    loss_and_score = []
    if args.exp_setting == "tran":
        indices = (idx_train, idx_val, idx_test)
        out, score_test, emb_list, ece_test, ace_test, brier_test, logits = run_transductive(
            conf,
            model,
            g,
            feats,
            labels,
            indices,
            criterion,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
            args
        )

    elif args.exp_setting == "ind":
        indices = graph_split(idx_train, idx_val, idx_test, args.split_rate, args.seed)

        out, score_test, emb_list, ece_test, ace_test, brier_test, logits = run_inductive(
            conf,
            model,
            g,
            feats,
            labels,
            indices,
            criterion,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
            args
        )

    logger.info(
        f"num_layers: {conf['num_layers']}. hidden_dim: {conf['hidden_dim']}. dropout_ratio: {conf['dropout_ratio']}"
    )
    logger.info(f"# params {sum(p.numel() for p in model.parameters())}")

    """ Saving teacher outputs """
    if not args.compute_min_cut:
        if 'MLP' not in model.model_name:
            logits_np = logits.detach().cpu().numpy()
            np.savez(output_dir.joinpath("logits"), logits_np)
            out_np = out.detach().cpu().numpy()
            np.savez(output_dir.joinpath("out"), out_np)
            for layer_num in args.teacher_emb_layers:
                out_emb_list = emb_list[layer_num].detach().cpu().numpy()
                np.savez(output_dir.joinpath("out_emb_list" + str(layer_num)), out_emb_list)

        """ Saving loss curve and model """
        if args.save_results:
            # Loss curves
            loss_and_score = np.array(loss_and_score)
            np.savez(output_dir.joinpath("loss_and_score"), loss_and_score)

            # Model
            torch.save(model.state_dict(), output_dir.joinpath("model.pth"))

    """ Saving min-cut loss """
    if args.exp_setting == "tran" and args.compute_min_cut:
        min_cut = compute_min_cut_loss(g, out)
        # with open(output_dir.parent.joinpath("min_cut_loss"), "a+") as f:
        #     f.write(f"{min_cut :.4f}\n")
        print('min_cut: ', min_cut)
    
    return score_test, ece_test, ace_test, brier_test


def repeat_run(args):
    scores = []
    eces = []
    aces = []
    briers = []
    for seed in range(args.num_exp):
        args.seed = seed
        temp_score, temp_ece, temp_ace, temp_brier = run(args)
        scores.append(temp_score)
        eces.append(temp_ece)
        aces.append(temp_ace)
        briers.append(temp_brier)
    scores_np = np.array(scores)
    eces_np = np.array(eces)
    aces_np = np.array(aces)
    briers_np = np.array(briers)
    return (
        scores_np.mean(axis=0),
        scores_np.std(axis=0),
        eces_np.mean(axis=0),
        eces_np.std(axis=0),
        aces_np.mean(axis=0),
        aces_np.std(axis=0),
        briers_np.mean(axis=0),
        briers_np.std(axis=0)
    )

def main():
    args = get_args()
    assert args.num_exp > 1

    (
        score_mean,
        score_std,
        ece_mean,
        ece_std,
        ace_mean,
        ace_std,
        brier_mean,
        brier_std
    ) = repeat_run(args)

    score_str = "".join(
        [f"{score_mean : .4f}\t"] + [f"{score_std : .4f}\t"]
    )
    if args.exp_setting == "tran":
        assert len(ece_mean) == len(args.temperatures)
        print("Score Mean and Std: ", score_mean, ", ", score_std)
        print("ECE Mean and Std: ", ece_mean, ", ", ece_std)
        print("ACE Mean and Std: ", ace_mean, ", ", ace_std)
        print("Brier Mean and Std: ", brier_mean, ", ", brier_std)

        data = []
        for i, temperature in enumerate(args.temperatures):
            data.append(
                [i+1, temperature, score_mean, ece_mean[i], ace_mean[i], brier_mean[i]]
            )
        
        df = pd.DataFrame(data, columns=["Teacher Logits", "Temperature", "Score", "ECE", "ACE", "Brier"])
        df.to_csv(args.output_dir.parent.joinpath("output.csv"), index=False)
    elif args.exp_setting == "ind":
        assert len(ece_mean) == len(args.temperatures)
        print("Score Mean and Std (test_ind): ", score_mean, ", ", score_std)
        print("ECE Mean and Std (test_ind): ", ece_mean, ", ", ece_std)
        print("ACE Mean and Std (test_ind): ", ace_mean, ", ", ace_std)
        print("Brier Mean and Std (test_ind): ", brier_mean, ", ", brier_std)
        

        data = []
        for i, temperature in enumerate(args.temperatures):
            data.append(
                [i+1, temperature, score_mean, ece_mean[i], ace_mean[i], brier_mean[i]]
            )
        
        df = pd.DataFrame(data, columns=["Teacher Logits", "Temperature", "Score(test_ind)", "ECE (test_ind)", "ACE (test_ind)", "Brier (test_ind)"])
        df.to_csv(args.output_dir.parent.joinpath("output.csv"), index=False)

    with open(args.output_dir.parent.joinpath("exp_results"), "a+") as f:
        f.write(f"{score_str}\n")


if __name__ == "__main__":
    main()
