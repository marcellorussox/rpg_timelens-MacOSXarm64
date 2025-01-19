#!/bin/bash

# Percorso relativo della cartella ev_test
test="test_folder/ev_test2"

# Trova tutte le sottocartelle delle sottocartelle e rinomina i file PNG
find "$test" -mindepth 2 -type d | while read -r folder; do
    echo "Processing folder: $folder"

    # Contatore per i nuovi nomi
    counter=0

    # Rinomina i file PNG nella sottocartella corrente
    for file in "$folder"/*.png; do
        # Controlla se il file PNG esiste
        if [ -f "$file" ]; then
            # Nuovo nome sequenziale (es. 0000.png, 0001.png, ...)
            new_name=$(printf "%06d.png" "$counter")

            # Rinominare il file
            mv "$file" "$folder/$new_name"

            # Incrementare il contatore
            counter=$((counter + 1))
        fi
    done

    if [ "$counter" -gt 0 ]; then
        echo "Renamed $counter files in $folder"
    else
        echo "No PNG files found in $folder. Skipping..."
    fi
done

echo "Rinomina completata!"