from problems.tsp.problem_tsp import TSP
import argparse
from utils.heatmap_dataset import HeatmapDataset
from utils.export_heatmap import generate_heatmaps
from utils.eval import _eval_dataset
import torch
from training.main_tsp import train
import math
from models.gcn_model import ResidualGatedGCNModel
from torch import nn
from utils.config import *
from models.prep_wrapper import PrepWrapResidualGatedGCNModel
from copy import deepcopy
from matplotlib import pyplot as plt
from utils.visualize import plot
from itertools import count
import time
import os


def main(opts):
    start_time = time.time()
    torch.set_num_threads(opts.num_cores)

    config = get_config(opts.config)

    make_dataset_kwargs = {"size": opts.num_nodes}  # {'normalize': False} for cvrp?

    current_best_test_cost = math.inf

    test_dataset = TSP.make_dataset(
        filename=None,
        num_samples=opts.test_size,
        offset=opts.offset,
        **make_dataset_kwargs
    )

    if torch.cuda.is_available():
        print("CUDA available, using GPU")
        dtypeFloat = torch.cuda.FloatTensor
        dtypeLong = torch.cuda.LongTensor
        torch.cuda.manual_seed(1)
    else:
        print("CUDA not available")
        dtypeFloat = torch.FloatTensor
        dtypeLong = torch.LongTensor
        torch.manual_seed(1)

    # Instantiate the network
    model = ResidualGatedGCNModel(config, dtypeFloat, dtypeLong)
    model = PrepWrapResidualGatedGCNModel(model)
    net = nn.DataParallel(model)
    best_net = net

    for i in count():
        dataset = TSP.make_dataset(
            filename=None,
            num_samples=opts.train_size,
            offset=opts.offset,
            **make_dataset_kwargs
        )

        results = evaluate_dataset(best_net, dataset, opts, config)

        net = train(config, dataset, results)

        test_results = evaluate_dataset(net, test_dataset, opts, config)

        average_test_cost = sum(r[0] for r in test_results) / len(test_results)
        print("achieved a new total test cost of {}".format(average_test_cost))

        if average_test_cost < current_best_test_cost:
            best_net = deepcopy(net)
            current_best_test_cost = average_test_cost

            dir_name = "tests/{}/{}".format(start_time, i)
            os.makedirs(dir_name, exist_ok=True)

            for j in range(10):
                fig, ax = plt.subplots()
                plot(ax, test_dataset[j], solution=test_results[j][1])
                plt.savefig("{}/{}.png".format(dir_name, j), bbox_inches="tight")
                plt.close()


def evaluate_dataset(net, dataset, opts, config):
    heatmaps = generate_heatmaps(net, dataset, opts, config)
    heatmap_dataset = HeatmapDataset(dataset, heatmaps)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not opts.no_cuda else "cpu"
    )

    return _eval_dataset(
        TSP,
        heatmap_dataset,
        opts.beam_size,
        opts,
        device,
        no_progress_bar=opts.no_progress_bar,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", action="store_true", help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument(
        "--train_size",
        type=int,
        default=5000,
        help="Number of instances used for reporting validation performance",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=500,
        help="Number of instances used for reporting validation performance",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset where to start in dataset (default 0)",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        help="Size of beam to use for beam search/DP",
        default=100,
    )
    parser.add_argument(
        "--decode_strategy",
        type=str,
        help="Deep Policy Dynamic Programming (dpdp) or Deep Policy Beam Search (dpbs)",
        default="dpdp",
    )
    parser.add_argument(
        "--score_function",
        type=str,
        default="heatmap_potential",
        help="Policy/score function to use to select beam: 'cost', 'heatmap' or 'heatmap_potential'",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--no_progress_bar", action="store_true", help="Disable progress bar"
    )
    parser.add_argument("--verbose", action="store_true", help="Set to show statistics")
    parser.add_argument(
        "--results_dir", default="results", help="Name of results directory"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use per device (cpu or gpu).",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=20,
        help="Number of nodes in TSP problems.",
    )
    # When providing a heatmap, will sparsify the input
    parser.add_argument("--heatmap", default=None, help="Heatmaps to use")
    parser.add_argument(
        "--heatmap_threshold",
        type=float,
        default=None,
        help="Use sparse graph based on heatmap treshold",
    )
    parser.add_argument("--knn", type=int, default=None, help="Use sparse knn graph")
    parser.add_argument(
        "--kthvalue_method",
        type=str,
        default="sort",
        help="Which kthvalue method to use for dpdp ('auto' = auto determine)",
    )
    parser.add_argument(
        "--skip_oom", action="store_true", help="Skip batch when out of memory"
    )
    parser.add_argument(
        "-c", "--config", type=str, default="configs/reinforcement.json"
    )
    parser.add_argument("--problem", type=str, default="tsp")
    parser.add_argument("--checkpoint", type=str)  # , required=True)
    parser.add_argument("--instances", type=str)  # , required=True)
    parser.add_argument("-o2", "--output_filename", type=str, default="results/output")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument(
        "--no_prepwrap", action="store_true", help="For backwards compatibility"
    )
    parser.add_argument(
        "--num_cores", type=int, help="number of cpu cores to use", default=6
    )

    opts = parser.parse_args()
    assert opts.o is None or (
        len(opts.datasets) == 1 and len(opts.beam_size) <= 1
    ), "Cannot specify result filename with more than one dataset or more than one beam_size"
    assert (
        opts.heatmap is None or len(opts.datasets) == 1
    ), "With heatmap can only run one (corresponding) dataset"

    main(opts)
