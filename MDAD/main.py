import os
import sys
import random
import argparse
import configparser
import numpy as np
import torch
from torch.backends import cudnn

from solver import Solver


class Logger(object):
    def __init__(self, filename="default.log", add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        mode = "a+" if self.add_flag else "w"
        with open(self.filename, mode, encoding="utf-8") as log:
            self.terminal.write(message)
            log.write(message)

    def flush(self):
        self.terminal.flush()


def main(config):
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path, exist_ok=True)

    solver = Solver(vars(config))

    if config.mode == "train":
        solver.train()
    elif config.mode == "test":
        solver.test()
    else:
        solver.train()
        solver.test()

    return solver


if __name__ == "__main__":
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, required=True, help="configuration file path")
    base_args, remaining = base_parser.parse_known_args()

    fileconfig = configparser.ConfigParser()
    fileconfig.read(base_args.config, encoding="utf-8")

    def cfg_get(section, option, fallback):
        if fileconfig.has_option(section, option):
            return fileconfig.get(section, option)
        if section == "param" and option == "lambda_weight":
            if fileconfig.has_option(section, "causal_loss_weight"):
                return fileconfig.get(section, "causal_loss_weight")
        return str(fallback)

    def cfg_cast(section, option, cast, fallback):
        return cast(cfg_get(section, option, fallback))

    def cfg_optional_cast(section, option, cast):
        if fileconfig.has_option(section, option):
            return cast(fileconfig.get(section, option))
        return None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=base_args.config,
        help="configuration file path",
    )

    arg_specs = [
        ("lr", float, "train", "lr", 1e-4),
        ("gpu", int, "train", "gpu", 0),
        ("num_epochs", int, "train", "epoch", 1),
        ("anormly_ratio", float, "train", "ar", 1.0),
        ("batch_size", int, "train", "bs", 128),
        ("seed", int, "train", "seed", 1),
        ("lradj", str, "train", "lradj", "type1"),
        ("win_size", int, "data", "ws", 100),
        ("input_c", int, "data", "ic", 1),
        ("dataset", str, "data", "ds", "MSL"),
        ("data_path", str, "data", "dp", "./dataset/MSL"),
        ("d_model", int, "param", "d_model", 64),
        ("dr", float, "param", "dropout", 0.1),
        ("wavelet_name", str, "param", "wavelet_name", "db2"),
        ("wavelet_level", int, "param", "wavelet_level", 3),
        ("num_prototypes", int, "param", "num_prototypes", 8),
        ("proto_top_k", int, "param", "proto_top_k", 1),
        ("n_heads", int, "param", "n_heads", 4),
        ("use_revin", int, "param", "use_revin", 1),
        ("lambda_weight", float, "param", "lambda_weight", 1.0),
        ("scale_patch_len", int, "param", "scale_patch_len", 8),
        ("scale_patch_stride_mode", str, "param", "scale_patch_stride_mode", "half"),
        ("encoder_layers", int, "param", "encoder_layers", 2),
        ("model_name", str, "model", "model_name", "pmkt_raw"),
        ("mode", str, "model", "mode", "all"),
        ("model_save_path", str, "model", "msp", "./checkpoints"),
        ("result_dir", str, "model", "result_dir", "result"),
    ]

    for name, value_type, section, option, fallback in arg_specs:
        parser.add_argument(
            f"--{name}",
            type=value_type,
            default=cfg_cast(section, option, value_type, fallback),
        )

    parser.add_argument(
        "--causal_loss_weight",
        dest="lambda_weight",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--proto_loss_weight",
        type=float,
        default=cfg_optional_cast("param", "proto_loss_weight", float),
    )

    config = parser.parse_args(remaining)
    args = vars(config)

    # random seed
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    # mkdir for log
    os.makedirs(config.result_dir, exist_ok=True)
    os.makedirs(config.model_save_path, exist_ok=True)

    # redirect stdout to logger
    log_file = os.path.join(config.result_dir, f"{config.dataset}.log")
    sys.stdout = Logger(filename=log_file, add_flag=True, stream=sys.stdout)

    print("------------ Options -------------")
    for k, v in sorted(args.items()):
        print(f"{str(k)}: {str(v)}")
    print("-------------- End ----------------")

    main(config)
