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


def compute_psnr(gt_img, pred_img):
    """Restituisce il valore di PSNR tra gt_img e pred_img."""
    mse = np.mean((gt_img.astype(float) - pred_img.astype(float)) ** 2)
    eps = 1e-10  # soglia per evitare errore di divisione
    mse = max(mse, eps)
    data_range = gt_img.max() - gt_img.min()
    return 10 * np.log10((data_range ** 2) / mse)


def compute_ssim(gt_img, pred_img):
    # Usa channel_axis=-1 (l'ultimo asse è il canale)
    return structural_similarity(
        gt_img, pred_img,
        data_range=gt_img.max() - gt_img.min(),
        gaussian_weights=True,
        channel_axis=-1  # invece di multichannel=True
    )


def process_dataset(gt_files, pred_files, num_skips, grayscale):
    """
    Confronta gt_files e pred_files (lista di path corrispondenti),
    saltando i frame per cui i % (num_skips + 1) != 0.
    Ritorna i valori (list_psnr, list_ssim).
    """
    list_psnr = []
    list_ssim = []

    for i, (gt_path, pred_path) in enumerate(zip(gt_files, pred_files)):
        # Logica di skip
        if i % (num_skips + 1) == 0:
            # Saltiamo questo frame
            continue

        gt_img = cv2.imread(gt_path)
        pred_img = cv2.imread(pred_path)

        # Se lettura fallisce, saltiamo
        if gt_img is None or pred_img is None:
            print(f"Immagine non valida o non trovata per i={i} -> GT: {gt_path}, PRED: {pred_path}")
            continue

        # Se dimensioni diverse, saltiamo
        if gt_img.shape != pred_img.shape:
            print(f"Dimensioni diverse per i={i} -> GT: {gt_img.shape}, PRED: {pred_img.shape}")
            continue

        # Se richiesto, conversione in scala di grigi
        if grayscale:
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)

        # Calcolo metriche
        try:
            val_psnr = compute_psnr(gt_img, pred_img)
            val_ssim = compute_ssim(gt_img, pred_img)
        except Exception as e:
            print(f"Errore calcolo metriche per i={i}, eccezione: {e}")
            continue

        list_psnr.append(val_psnr)
        list_ssim.append(val_ssim)

    return list_psnr, list_ssim


def print_statistics(method_name, psnr_values, ssim_values):
    """Stampa media e std di PSNR/SSIM per un determinato metodo."""
    psnr_values = np.array(psnr_values)
    ssim_values = np.array(ssim_values)

    if len(psnr_values) == 0:
        print(f"Metodo {method_name} -> nessun frame processato o valori vuoti.\n")
        return

    # Calcolo di media e std
    psnr_mean = float(np.mean(psnr_values))
    psnr_std = float(np.std(psnr_values))
    ssim_mean = float(np.mean(ssim_values))
    ssim_std = float(np.std(ssim_values))

    print(f"Metodo: {method_name}")
    print(f"  PSNR: {psnr_mean:.4f} ± {psnr_std:.4f}")
    print(f"  SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
    print("")


def main():
    parser = argparse.ArgumentParser(description="Calcola PSNR e SSIM su subfolder di dataset, con skip frames.")
    parser.add_argument("results_folder", help="Cartella principale contenente i subfolder di dataset.")
    parser.add_argument("--num_skips", type=int, default=1,
                        help="Numero di frame da saltare; i frame processati rispettano i % (num_skips+1) == 0.")
    parser.add_argument("--grayscale", action="store_true",
                        help="Se presente, converte le immagini in scala di grigi prima di calcolare le metriche.")
    args = parser.parse_args()

    results_folder = args.results_folder
    num_skips = args.num_skips
    grayscale = args.grayscale

    # Trova le sottocartelle (dataset) in results_folder
    dataset_folders = [d for d in sorted(glob.glob(join(results_folder, "*"))) if os.path.isdir(d)]
    if not dataset_folders:
        print(f"Nessun dataset trovato in {results_folder}")
        return

    # Per ogni dataset, troviamo GT e eventuali metodi
    for dataset_path in dataset_folders:
        dataset_name = basename(dataset_path)
        print(f"\n--- Dataset: {dataset_name} ---")

        subfolders = [f for f in sorted(glob.glob(join(dataset_path, "*"))) if os.path.isdir(f)]
        # subfolders, ad esempio, [dataset_path/GT, dataset_path/Timelens, ...]
        method_files = {}
        for sf in subfolders:
            method_name = basename(sf)
            # Raccogli i file PNG dentro la cartella
            png_files = sorted(glob.glob(join(sf, "*.png")))
            method_files[method_name] = png_files

        # Controlla esistenza di "GT"
        if "GT" not in method_files:
            print(f"Manca cartella GT in {dataset_name}, skipping.")
            continue

        gt_files = method_files.pop("GT")  # Rimuove la key "GT" e restituisce i file
        # A questo punto method_files contiene { "Timelens": [...], "AltroMetodo": [...] }

        for method_name, pred_files in method_files.items():
            if len(gt_files) != len(pred_files):
                print(f"Mismatch numero file: GT({len(gt_files)}) vs {method_name}({len(pred_files)}). Skipping.")
                continue

            psnr_vals, ssim_vals = process_dataset(gt_files, pred_files, num_skips, grayscale)
            print_statistics(method_name, psnr_vals, ssim_vals)


if __name__ == "__main__":
    main()
