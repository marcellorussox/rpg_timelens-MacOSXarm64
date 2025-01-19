#!/usr/bin/env python3
"""
evaluation.py

Script Python per calcolare PSNR e SSIM su immagini in un dataset organizzato in subfolder,
ognuno contenente una cartella GT e una cartella di un metodo (es. Timelens).
Applica una logica di skip basata su i % (num_skips + 1) != 0 (quindi i=0 viene sempre processato).
Permette anche l'opzione di calcolo in scala di grigi.
"""

import argparse
import glob
import os
from os.path import join, basename
import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import lpips  # Libreria per calcolare LPIPS
import torch  # Necessario per LPIPS

# Inizializza il calcolo di LPIPS
loss_fn = lpips.LPIPS(net='alex')  # Puoi usare 'vgg' per un approccio più tradizionale


def compute_mse(gt_img, pred_img):
    """Calcola il Mean Squared Error (MSE)."""
    return np.mean((gt_img.astype(float) - pred_img.astype(float)) ** 2)


def compute_psnr(gt_img, pred_img):
    """Calcola il PSNR."""
    mse = compute_mse(gt_img, pred_img)
    data_range = gt_img.max() - gt_img.min()
    return 10 * np.log10((data_range ** 2) / mse)


def compute_ssim(gt_img, pred_img):
    """Calcola SSIM."""
    return structural_similarity(
        gt_img, pred_img,
        data_range=gt_img.max() - gt_img.min(),
        gaussian_weights=True,
        channel_axis=-1
    )


def compute_lpips(gt_img, pred_img):
    """Calcola LPIPS utilizzando la libreria esterna."""
    # Converti le immagini in formato Torch Tensor e normalizzale a [-1, 1]
    gt_tensor = torch.tensor(gt_img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
    pred_tensor = torch.tensor(pred_img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
    # Calcola LPIPS
    return float(loss_fn(gt_tensor, pred_tensor).item())


def process_dataset(gt_files, pred_files, num_skips, grayscale):
    """
    Confronta gt_files e pred_files (lista di path corrispondenti).
    Restituisce i valori di MSE, PSNR, SSIM, LPIPS.
    """
    list_mse = []
    list_psnr = []
    list_ssim = []
    list_lpips = []

    for i, (gt_path, pred_path) in enumerate(zip(gt_files, pred_files)):
        if i % (num_skips + 1) != 0:
            continue

        gt_img = cv2.imread(gt_path)
        pred_img = cv2.imread(pred_path)

        if gt_img is None or pred_img is None:
            print(f"Immagine non valida: GT={gt_path}, Pred={pred_path}")
            continue

        if gt_img.shape != pred_img.shape:
            print(f"Dimensioni diverse: GT={gt_img.shape}, Pred={pred_img.shape}")
            continue

        if grayscale:
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)

        try:
            val_mse = compute_mse(gt_img, pred_img)
            val_psnr = compute_psnr(gt_img, pred_img)
            val_ssim = compute_ssim(gt_img, pred_img)
            val_lpips = compute_lpips(gt_img, pred_img)
        except Exception as e:
            print(f"Errore calcolo metriche: {e}")
            continue

        list_mse.append(val_mse)
        list_psnr.append(val_psnr)
        list_ssim.append(val_ssim)
        list_lpips.append(val_lpips)

    return list_mse, list_psnr, list_ssim, list_lpips


def print_statistics(method_name, mse_values, psnr_values, ssim_values, lpips_values):
    """Stampa media e std di tutte le metriche."""
    mse_values = np.array(mse_values)
    psnr_values = np.array(psnr_values)
    ssim_values = np.array(ssim_values)
    lpips_values = np.array(lpips_values)

    if len(mse_values) == 0:
        print(f"Metodo {method_name}: Nessun frame processato.\n")
        return

    mse_mean, mse_std = np.mean(mse_values), np.std(mse_values)
    psnr_mean, psnr_std = np.mean(psnr_values), np.std(psnr_values)
    ssim_mean, ssim_std = np.mean(ssim_values), np.std(ssim_values)
    lpips_mean, lpips_std = np.mean(lpips_values), np.std(lpips_values)

    print(f"Metodo: {method_name}")
    print(f"  MSE:   {mse_mean:.4f} ± {mse_std:.4f}")
    print(f"  PSNR:  {psnr_mean:.4f} ± {psnr_std:.4f}")
    print(f"  SSIM:  {ssim_mean:.4f} ± {ssim_std:.4f}")
    print(f"  LPIPS: {lpips_mean:.4f} ± {lpips_std:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Valutazione MSE, PSNR, SSIM e LPIPS.")
    parser.add_argument("ev_test", help="Cartella principale con i dataset.")
    parser.add_argument("--num_skips", type=int, default=1, help="Numero di frame da saltare.")
    parser.add_argument("--grayscale", action="store_true", help="Converte le immagini in scala di grigi.")
    args = parser.parse_args()

    dataset_folders = [d for d in sorted(glob.glob(join(args.results_folder, "*"))) if os.path.isdir(d)]
    if not dataset_folders:
        print(f"Nessun dataset trovato in {args.results_folder}")
        return

    for dataset_path in dataset_folders:
        dataset_name = basename(dataset_path)
        print(f"\n--- Dataset: {dataset_name} ---")

        subfolders = [f for f in sorted(glob.glob(join(dataset_path, "*"))) if os.path.isdir(f)]
        method_files = {basename(sf): sorted(glob.glob(join(sf, "*.png"))) for sf in subfolders}

        if "GT" not in method_files:
            print(f"Manca cartella GT in {dataset_name}, skipping.")
            continue

        gt_files = method_files.pop("GT")
        for method_name, pred_files in method_files.items():
            if len(gt_files) != len(pred_files):
                print(f"Mismatch numero file: GT={len(gt_files)} vs {method_name}={len(pred_files)}. Skipping.")
                continue

            mse_vals, psnr_vals, ssim_vals, lpips_vals = process_dataset(gt_files, pred_files, args.num_skips,
                                                                         args.grayscale)
            print_statistics(method_name, mse_vals, psnr_vals, ssim_vals, lpips_vals)


if __name__ == "__main__":
    main()
