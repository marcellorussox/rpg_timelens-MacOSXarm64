import argparse
import glob
import os
import time
from os.path import join, basename, dirname

import cv2
import numpy as np
import tqdm
import yaml
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from task_manager import TaskManager


def psnr(gt, pred):
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max() - gt.min())


def ssim(gt, pred):
    multichannel = len(gt.shape) == 3 and gt.shape[2] == 3
    return structural_similarity(gt, pred, data_range=gt.max() - gt.min(), multichannel=multichannel,
                                 gaussian_weights=True)


METRICS = {"PSNR": psnr, "SSIM": ssim}


def tm_worker(gt_path, pred_path, w, metric_name, metric, grayscale):
    gt_img = cv2.imread(gt_path)
    pred_img = cv2.imread(pred_path)

    # Debug: verifica lettura immagini
    if gt_img is None:
        print(f"Errore: Impossibile leggere immagine GT {gt_path}")
    if pred_img is None:
        print(f"Errore: Impossibile leggere immagine predetta {pred_path}")

    # Debug: verifica shape immagini
    print(
        f"GT shape: {None if gt_img is None else gt_img.shape}, Pred shape: {None if pred_img is None else pred_img.shape}")

    if grayscale:
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)

    # Debug: range pixel
    print(f"GT range: {gt_img.min() if gt_img is not None else 'N/A'}, {gt_img.max() if gt_img is not None else 'N/A'}")
    print(
        f"Pred range: {pred_img.min() if pred_img is not None else 'N/A'}, {pred_img.max() if pred_img is not None else 'N/A'}")

    return metric(gt_img, pred_img), metric_name, w


def evaluate(method_files, summary, weights, num_processes, grayscale=False):
    print(f"method_files: {method_files}")
    assert "GT" in method_files
    gt_files = method_files.pop("GT")

    for method, files in method_files.items():
        with tqdm.tqdm(total=len(files) * len(METRICS)) as pbar:
            pbar.set_description(f"Dataset: {curr_dataset} Method: {method}")
            tm = TaskManager(num_processes, 4, lambda x: summary[method][x[1]][x[2]].append(x[0]))

            for i, (pred_path, gt_path) in enumerate(zip(files, gt_files)):
                if i % (1 + len(weights)) == 0:
                    pbar.update(1)
                    continue

                w = weights[(i % (1 + len(weights))) - 1]

                if not os.path.exists(gt_path):
                    pbar.update(1)
                    print(f"Image not Found {gt_path}")
                    continue

                if not os.path.exists(pred_path):
                    pbar.update(1)
                    print(f"Image not Found {pred_path}")
                    continue

                for metric_name, metric in METRICS.items():
                    tm.new_task(tm_worker,
                                gt_path,
                                pred_path,
                                w, metric_name,
                                metric, grayscale)
                    pbar.update(1)

            tm.close()


def convert_leaves_to_statistics(data):
    if type(data) is dict:
        return {k: convert_leaves_to_statistics(v) for k, v in data.items()}
    return get_statistics(data)


def fuse_dicts(args):
    if type(args[0]) is dict:
        keys = list(args[0].keys())
        return {k: fuse_dicts([a[k] for a in args]) for k in keys}
    return sum(args, [])


def aggregate_lowest_level(data):
    if type(data) is dict:
        if len(data) == 0:
            return []
        if type(list(data.values())[0]) is dict:
            return {k: aggregate_lowest_level(v) for k, v in data.items()}
        return sum(data.values(), [])


def get_statistics(data):
    std = float(np.std(data))
    mean = float(np.mean(data))
    return {"std": std, "mean": mean}


def print_nice(data):
    print("Dataset\tMethod\tSSIM\tPSNR")
    print("---------------------------")
    methods = None
    datasets = sorted(list(data.keys()))
    for dataset in datasets:
        if methods is None:
            methods = sorted(list(data[dataset].keys()))
            global_averages = {m: {"SSIM": [], "PSNR": []} for m in methods}
        for method in methods:
            subdata = data[dataset][method]
            print(
                f"\t\t{dataset}\t{method}\t{subdata['SSIM']['mean']}+-{subdata['SSIM']['std']}\t{subdata['PSNR']['mean']}+-{subdata['PSNR']['std']}")
            global_averages[method]['SSIM'] += [subdata['SSIM']['mean']]
            global_averages[method]['PSNR'] += [subdata['PSNR']['mean']]

    print("GLOBAL")
    for m, data in global_averages.items():
        for metric, d in data.items():
            print(f"Method: {m}\t Metric: {metric}: {np.mean(d)}+-{np.std(d)}")
    print("---------------------------")


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser("""Given a meta folder, run evaluation""")
    parser.add_argument("results_folder")
    parser.add_argument('-n', "--num_processes", type=int, default=0)
    parser.add_argument('-o', "--output_file")
    parser.add_argument('-g', "--grayscale", action="store_true")
    parser.add_argument('-s', "--num_skips", type=int, default=7)
    args = parser.parse_args()

    num_processes = args.num_processes

    data = {}

    weights = np.linspace(0, 1, args.num_skips + 2)[1:-1]

    # Walk through results_folder to find dataset directories
    for path, subdirs, files in os.walk(args.results_folder):
        if len(files) == 0:  # Skip empty directories
            continue

        curr_dataset = basename(path)  # Use the current directory name as the dataset name
        if curr_dataset in data:  # Avoid processing the same dataset twice
            continue

        # Collect all dataset directories (e.g., baloon_popping)
        dataset_folders = [d for d in sorted(glob.glob(join(args.results_folder, "*"))) if os.path.isdir(d)]
        datasets = [basename(d) for d in dataset_folders]  # Extract dataset names

        # Prepare method_files to map methods like 'GT' and 'Timelens' to their files
        method_files = {}
        for dataset_folder in dataset_folders:
            methods_in_dataset = [d for d in sorted(glob.glob(join(dataset_folder, "*"))) if os.path.isdir(d)]
            method_files[basename(dataset_folder)] = {basename(m): sorted(glob.glob(join(m, "*"))) for m in
                                                      methods_in_dataset}

        # Debug: print the results
        print(f"Dataset folders: {dataset_folders}")
        print(f"Datasets: {datasets}")
        print(f"Method files: {method_files}")

        # Prepare data dict for the current dataset
        if curr_dataset in method_files:
            methods = list(method_files[curr_dataset].keys())
            summary = {method: {metric: {w: [] for w in weights} for metric in METRICS} for method in methods}
            data[curr_dataset] = summary

            # Fill data[key] with evaluations for methods and samples
            print(f"Evaluating methods {methods} on {curr_dataset}")
            evaluate(method_files[curr_dataset], data[curr_dataset], weights, num_processes, args.grayscale)

    all_data = {}
    print("Computing per_frame_per_dataset")
    all_data['per_frame_per_dataset'] = convert_leaves_to_statistics(data)
    print("Computing per_dataset")
    all_data['per_dataset'] = convert_leaves_to_statistics(aggregate_lowest_level(data))
    print("Computing per_frame")
    all_data['per_frame'] = convert_leaves_to_statistics(fuse_dicts(list(data.values())))
    print("Computing total_average")
    all_data['total_average'] = convert_leaves_to_statistics(aggregate_lowest_level(fuse_dicts(list(data.values()))))

    from pprint import pprint

    pprint(all_data)

    print_nice(all_data['per_dataset'])

    with open(args.output_file, 'w') as fh:
        yaml.dump(all_data, fh, default_flow_style=False)
