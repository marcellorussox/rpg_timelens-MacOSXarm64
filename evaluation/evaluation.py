#!/usr/bin/env python3
"""
evaluation.py

Script Python per calcolare MSE, PSNR, SSIM e LPIPS su immagini in un dataset
organizzato in subfolder (GT e Timelens). Accetta come input una cartella "parent"
nella quale si trovano i vari "test" (es. ev_test1, ev_test2, ...).
All'interno di ognuno di questi test ci sono i dataset (es. datasetA, datasetB, ...)
che contengono la cartella GT e la cartella Timelens.

Funziona sia come script standalone (con argparse) sia importando le funzioni
da un notebook Jupyter.

Il file CSV generato ha la struttura:

    test,dataset,mse,psnr,ssim,lpips

dove:
- "test" è il nome della sottocartella di "parent_folder" (es. ev_test1),
- "dataset" è il nome del dataset (es. datasetA),
- mse, psnr, ssim, lpips sono le medie delle metriche sul set di frame considerati.
"""

import argparse
import glob
import os
import csv
from os.path import join, basename
import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import lpips  # Libreria per calcolare LPIPS
import torch  # Necessario per LPIPS

# Inizializza il calcolo di LPIPS solo una volta, globalmente
loss_fn = lpips.LPIPS(net='alex')  # Puoi usare 'vgg' se preferisci


def compute_mse(gt_img, pred_img):
    """Calcola il Mean Squared Error (MSE)."""
    return np.mean((gt_img.astype(float) - pred_img.astype(float)) ** 2)


def compute_psnr(gt_img, pred_img):
    """Calcola il PSNR."""
    mse = compute_mse(gt_img, pred_img)
    if mse == 0:
        return float('inf')
    data_range = gt_img.max() - gt_img.min()
    return 10 * np.log10((data_range ** 2) / mse)


def compute_ssim(gt_img, pred_img):
    """Calcola SSIM utilizzando scikit-image."""
    return structural_similarity(
        gt_img,
        pred_img,
        data_range=gt_img.max() - gt_img.min(),
        gaussian_weights=True,
        channel_axis=-1
    )


def compute_lpips(gt_img, pred_img):
    """Calcola LPIPS utilizzando la libreria esterna LPIPS."""
    # Converti le immagini in formato Torch Tensor e normalizzale a [-1, 1]
    gt_tensor = torch.tensor(gt_img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
    pred_tensor = torch.tensor(pred_img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
    # Calcola LPIPS
    return float(loss_fn(gt_tensor, pred_tensor).item())


def process_dataset(gt_files, pred_files, num_skips, grayscale):
    """
    Confronta gt_files e pred_files (lista di path corrispondenti).
    Restituisce liste con i valori di MSE, PSNR, SSIM, LPIPS (uno per ogni frame considerato).
    """
    list_mse = []
    list_psnr = []
    list_ssim = []
    list_lpips_vals = []

    for i, (gt_path, pred_path) in enumerate(zip(gt_files, pred_files)):
        # Skip logic
        if i % (num_skips + 1) != 0:
            continue

        gt_img = cv2.imread(gt_path)
        pred_img = cv2.imread(pred_path)

        if gt_img is None or pred_img is None:
            print(f"[Attenzione] Immagine non valida: GT={gt_path}, Pred={pred_path}")
            continue

        if gt_img.shape != pred_img.shape:
            print(f"[Attenzione] Dimensioni diverse: GT={gt_img.shape}, Pred={pred_img.shape}")
            continue

        # Opzionale: conversione in scala di grigi
        if grayscale:
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
            # Per calcolare SSIM/LPIPS su immagini moncanale,
            # dobbiamo riconvertire in 3 canali se vogliamo usare la pipeline originale di LPIPS.
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)

        try:
            val_mse = compute_mse(gt_img, pred_img)
            val_psnr = compute_psnr(gt_img, pred_img)
            val_ssim = compute_ssim(gt_img, pred_img)
            val_lp = compute_lpips(gt_img, pred_img)
        except Exception as e:
            print(f"[Errore] Calcolo metriche: {e}")
            continue

        list_mse.append(val_mse)
        list_psnr.append(val_psnr)
        list_ssim.append(val_ssim)
        list_lpips_vals.append(val_lp)

    return list_mse, list_psnr, list_ssim, list_lpips_vals


def evaluate_tests(test_folder, csv_path, num_skips=1, grayscale=False):
    """
    Scorre tutte le sottocartelle di `test_folder` (i vari "test", es. ev_test1, ev_test2, ...)
    e all'interno di ciascuna scorre i dataset. Per ogni dataset (che deve contenere
    una cartella GT e una cartella Timelens), calcola MSE, PSNR, SSIM, LPIPS e scrive
    i risultati (media delle metriche) su un CSV con intestazione:

        test,dataset,mse,psnr,ssim,lpips

    Inoltre stampa a schermo in modo user-friendly i valori calcolati.
    """
    # Apri (o crea) il file CSV e scrivi l'intestazione
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["test", "dataset", "mse", "psnr", "ssim", "lpips"])

        # Scorri i "test" presenti nella cartella principale
        test_folders = sorted([d for d in os.listdir(test_folder)
                               if os.path.isdir(join(test_folder, d))])
        for test_name in test_folders:
            test_path = join(test_folder, test_name)

            # Sottocartelle del test (i dataset)
            dataset_folders = sorted([d for d in os.listdir(test_path)
                                      if os.path.isdir(join(test_path, d))])

            for dataset_name in dataset_folders:
                dataset_path = join(test_path, dataset_name)
                gt_dir = join(dataset_path, "GT")
                pred_dir = join(dataset_path, "Timelens")

                # Controllo esistenza cartelle GT e Timelens
                if not os.path.isdir(gt_dir) or not os.path.isdir(pred_dir):
                    # Non trovate le cartelle richieste
                    print(f"[Skip] In '{test_name}/{dataset_name}' manca GT o Timelens.")
                    continue

                # Leggi e ordina le immagini
                gt_files = sorted(glob.glob(join(gt_dir, "*.png")))
                pred_files = sorted(glob.glob(join(pred_dir, "*.png")))

                # Controllo che abbiano la stessa lunghezza
                if len(gt_files) != len(pred_files):
                    print(f"[Skip] Test '{test_name}', dataset '{dataset_name}': "
                          f"Mismatch nel numero di file GT({len(gt_files)}) vs Timelens({len(pred_files)}).")
                    continue

                # Calcola le metriche (frame-by-frame, con skip)
                list_mse, list_psnr, list_ssim, list_lpips_vals = process_dataset(
                    gt_files, pred_files, num_skips, grayscale
                )

                if len(list_mse) == 0:
                    print(f"[Info] Nessuna immagine processata per '{test_name}/{dataset_name}'.")
                    continue

                # Calcola media delle metriche
                mse_mean = np.mean(list_mse)
                psnr_mean = np.mean(list_psnr)
                ssim_mean = np.mean(list_ssim)
                lpips_mean = np.mean(list_lpips_vals)

                # Stampa user-friendly
                print(f"\n[Test: {test_name}, Dataset: {dataset_name}]")
                print(f"  MSE:   {mse_mean:.4f}")
                print(f"  PSNR:  {psnr_mean:.4f}")
                print(f"  SSIM:  {ssim_mean:.4f}")
                print(f"  LPIPS: {lpips_mean:.4f}")

                # Scrivi su CSV
                writer.writerow([test_name, dataset_name, mse_mean, psnr_mean, ssim_mean, lpips_mean])


def main():
    parser = argparse.ArgumentParser(description="Valutazione MSE, PSNR, SSIM e LPIPS.")
    parser.add_argument("parent_folder", help="Cartella contenente i test (es. test_folder).")
    parser.add_argument("--csv", type=str, default="metrics_results.csv",
                        help="Path del file CSV di output (default: metrics_results.csv).")
    parser.add_argument("--num_skips", type=int, default=1,
                        help="Numero di frame da saltare (se 1, processa un frame sì e uno no).")
    parser.add_argument("--grayscale", action="store_true",
                        help="Converte le immagini in scala di grigi prima del calcolo delle metriche.")
    args = parser.parse_args()

    # Esegui la valutazione con i parametri da linea di comando
    evaluate_tests(
        test_folder=args.parent_folder,
        csv_path=args.csv,
        num_skips=args.num_skips,
        grayscale=args.grayscale
    )


if __name__ == "__main__":
    main()