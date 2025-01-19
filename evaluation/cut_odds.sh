#!/bin/bash

# Cartella principale
MAIN_FOLDER="test_folder"

# Controlla se la cartella principale esiste
if [ ! -d "$MAIN_FOLDER" ]; then
  echo "La cartella $MAIN_FOLDER non esiste."
  exit 1
fi

# Naviga nelle sottocartelle ev_test e ev_test2 e nelle loro subdir
for subfolder in "$MAIN_FOLDER"/ev_test*; do
  if [ -d "$subfolder" ]; then
    echo "Entrando nella sottocartella: $subfolder"

    for dataset in "$subfolder"/*; do
      if [ -d "$dataset" ]; then
        echo "  Processando dataset: $dataset"

        for method in "$dataset"/*; do
          if [ -d "$method" ]; then
            echo "    Analizzando metodo: $method"

            # Elimina i file con indice dispari
            for file in "$method"/*.png; do
              # Estrai solo il numero dal nome del file (es. 000003.png -> 3)
              base_name=$(basename "$file")
              num=$(echo "$base_name" | grep -o '[0-9]\+')

              # Controlla se il numero è dispari
              num=$((10#$num))  # Converte num in base 10
              if (( num % 2 != 0 )); then
                  echo "Eliminando file dispari: $file"
                  rm "$file"
              fi
            done
          fi
        done
      fi
    done
  fi
done

echo "Operazione completata!"
