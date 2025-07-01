import torch
import argparse
import os
import pandas as pd
from main import get_device, get_dataset
from multitasking import get_current_gpu_utilization
from model import shallowPLRNN
from bptt import load_from_path, read_hypers
import multiprocessing
from pathlib import Path

torch.set_num_threads(4)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained hierarchical PLRNN model."
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to pre trained model"
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default=None,
        help="Path to evaluation data. Must be specified if not in hypers.txt",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./results/experiment",
        help="Path to save results. Default: ./results/experiment",
    )

    parser.add_argument(
        "--mse_steps",
        type=int,
        default=100,
        help="Number of steps to calculate MSE over. Default: 100",
    )

    parser.add_argument(
        "--use_gpu", action="store_true", help="Use GPU. Default: False"
    )
    parser.add_argument(
        "--compile", action="store_true", help="Compile model. Default: False"
    )

    parser.add_argument(
        "--kl_bins",
        type=int,
        default=None,
        help="Number of bins for KL divergence. Default: None (uses the same number of bins as at train time)",
    )
    parser.add_argument(
        "--pse_smooth",
        type=int,
        default=None,
        help="Smoothing sigma for PSE. Default: None (see kl_bins)",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of subprocesses that are spawned. Default: 1",
    )

    parser.add_argument(
        "--finetune", action="store_true", help="finetune a pre trained model."
    )

    parser.add_argument(
        "--num_shared_objects",
        type=int,
        default=1,
        help="Number of different systems that share a feature vector",
    )
    return parser.parse_args()


def handle_path(args):
    stack = []
    stack.append(args.model_path)

    runs = []
    while len(stack) > 0:
        path = stack.pop()
        if path.split("/")[-1] in [str(i).zfill(3) for i in range(1, 1000)]:
            runs.append(path)
        else:
            for p in os.listdir(path):
                stack.append(os.path.join(path, p))
    return runs


def additional_paths(args):
    """
    Finds the model paths for subsystems of a shared model.
    """
    stack = []
    if args.num_shared_objects > 1:
        stack += [args.model_path + f"_{i}" for i in range(0, args.num_shared_objects)]
    runs = []
    while len(stack) > 0:
        path = stack.pop()
        if path.split("/")[-1] in [str(i).zfill(3) for i in range(1, 1000)]:
            runs.append(path)
        else:
            for p in os.listdir(path):
                stack.append(os.path.join(path, p))
    return runs


def evaluate_model(args_p):
    """Evaluate a single model at path args.model_path."""
    print("\n Evaluating model at path ", args_p[1])
    args, p = args_p
    args.model_path = p
    modelargs = read_hypers(args)
    # overwrite some arguments
    modelargs.use_gpu = args.use_gpu
    modelargs.compile = args.compile
    # get the GPU with the most free memory
    if args.use_gpu:
        _, mem_dict = get_current_gpu_utilization()
        min_mem = 100
        device_id = None
        for g in args.free_gpus:
            if mem_dict[g] < min_mem:
                min_mem = mem_dict[g]
                device_id = g
        modelargs.device_id = device_id
    # change pse smoothing and kl bins if necessary
    if args.kl_bins is not None:
        modelargs.kl_bins = args.kl_bins
    if args.pse_smooth is not None:
        modelargs.pse_smooth = args.pse_smooth
    modelargs = get_device(modelargs)
    dataset = get_dataset(modelargs)
    model = shallowPLRNN(modelargs, dataset)
    load_from_path(model, args)
    if modelargs.compile:
        model = torch.compile(model)
    model.eval()
    model.evaluator.compute_expensive(["dstsp", "pse"])
    dstsp = model.evaluator.get_state_space_divergence().cpu().squeeze().numpy()
    pse = model.evaluator.get_pse().squeeze()
    return args.model_path, dstsp, pse


def main():
    args = parse_args()
    if args.model_path is None:
        raise ValueError("Model path must be specified.")
    # get all free GPUs
    if args.use_gpu:
        util_dict, mem_dict = get_current_gpu_utilization()
        args.free_gpus = [
            int(g)
            for g in util_dict.keys()
            if util_dict[g] < 0.05 and mem_dict[g] < 0.1
        ]
        print(
            "Can use GPUs: ",
            args.free_gpus,
            ". Make sure to not spawn too many workers for this amount as to not run into memory issues.",
        )
    paths = handle_path(args)
    subsystem_paths = additional_paths(args)
    subsystem_args = args.__dict__.copy()
    subsystem_args.update({"num_shared_objects": 1})
    print("Found the following paths to evaluate: ", paths + subsystem_paths)
    # split up the evaluation into multiple processes
    pool = multiprocessing.Pool(args.num_workers)
    results_list = pool.map(
        evaluate_model,
        [(args, p) for p in paths] + [(subsystem_args, p) for p in subsystem_paths],
        chunksize=1,
    )
    pool.close()
    pool.join()

    # store the results in a dictionary
    results = {}
    for p, dstsp, pse in results_list:
        for s, (kl, hel) in enumerate(zip(dstsp, pse)):
            results[(p, s)] = {"dstsp": kl, "pse": hel}

    # save results
    df = pd.DataFrame(results)
    # generate folders if necessary
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    df.to_csv(os.path.join(args.save_path, "results.csv"))
    print("Results saved to ", os.path.join(args.save_path, "results.csv"))


if __name__ == "__main__":
    main()
