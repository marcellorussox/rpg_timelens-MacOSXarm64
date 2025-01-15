#!/bin/bash

skip=3
insert=3
base_path="test"

# Itera su ogni dataset
for dataset in "$base_path"/*; do
    if [ -d "$dataset" ]; then
        echo "Processing dataset: $(basename "$dataset")"

        events_path="$dataset/events"
        images_path="$dataset/images"
        output_path="$dataset/output"

        # Assicurati che esistano le cartelle necessarie
        if [ ! -d "$events_path" ] || [ ! -d "$images_path" ]; then
            echo "Missing events or images folder in $(basename "$dataset"). Skipping..."
            continue
        fi

        # Crea la cartella di output se non esiste
        mkdir -p "$output_path"

        # Esegui il modello
        python -m timelens.run_timelens checkpoint.bin "$events_path" "$images_path" "$output_path" $skip $insert
    fi
done