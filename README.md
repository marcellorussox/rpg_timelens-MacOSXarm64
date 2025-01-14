# TimeLens: Event-based Video Frame Interpolation - MacOS ARM (CPU-only)

<p align="center">
  <a href="https://youtu.be/dVLyia-ezvo">
    <img src="assets/timelens_yt_thumbnail_icon.png" alt="TimeLens" width="500"/>
  </a>
</p>

Questo repository è una versione adattata per **macOS ARM (Apple Silicon)** del progetto **TimeLens**, descritto nel paper [**TimeLens: Event-based Video Frame Interpolation**](http://rpg.ifi.uzh.ch/docs/CVPR21_Gehrig.pdf), presentato al CVPR 2021 da Stepan Tulyakov*, [Daniel Gehrig*](https://danielgehrig18.github.io/), Stamatios Georgoulis, Julius Erbach, [Mathias Gehrig](https://magehrig.github.io/), Yuanyou Li, e [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html).

Per maggiori informazioni, visita la [pagina del progetto](http://rpg.ifi.uzh.ch/timelens).

---

## Google Colab

Un notebook Google Colab è disponibile [qui](TimeLens.ipynb). È possibile aumentare la frequenza dei frame dei propri video utilizzando i dati salvati su Google Drive.

---

## Gallery

Per ulteriori esempi, visita la [pagina del progetto](http://rpg.ifi.uzh.ch/timelens).

![coke](assets/coke.gif)
![paprika](assets/paprika.gif)
![pouring](assets/pouring.gif)
![water_bomb_floor](assets/water_bomb_floor.gif)

---

## Installazione

Questa versione è ottimizzata per macOS ARM (Apple Silicon) e utilizza Miniforge per la configurazione dell'ambiente Conda. CUDA non è richiesto, e l'esecuzione è basata esclusivamente sulla CPU.

### 1. Installazione di Miniforge

Scaricare e installare Miniforge per macOS ARM dal [sito ufficiale](https://github.com/conda-forge/miniforge/releases). Usare il file:

- `Miniforge3-MacOSX-arm64.sh`

Eseguire il comando per avviare l'installazione:

    bash Miniforge3-MacOSX-arm64.sh

Seguire le istruzioni per completare l'installazione.

### 2. Creazione dell'ambiente Conda

Creare un ambiente Conda e installare le dipendenze richieste:

    conda create -y -n timelens python=3.9
    conda activate timelens
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    conda install -y -c conda-forge opencv scipy tqdm click jupyter
    brew install ffmpeg  # Per la creazione del video finale

### 3. Clonare il repository

Clonare il codice del progetto:

    mkdir ~/timelens/
    cd ~/timelens
    git clone https://github.com/marcellorussox/rpg_timelens-MacOSXarm64.git
    cd rpg_timelens-MacOSXarm64

### 4. Scaricare i dati di esempio e il modello pre-addestrato

Scaricare i dati di esempio e il checkpoint per l’interpolazione:

    wget http://download.ifi.uzh.ch/rpg/web/data/timelens/data2/checkpoint.bin
    wget http://download.ifi.uzh.ch/rpg/web/data/timelens/data2/example_github.zip
    unzip example_github.zip
    rm -rf example_github.zip

---

### Esecuzione

Per eseguire TimeLens con i dati forniti:

    skip=0
    insert=7
    python -m timelens.run_timelens checkpoint.bin example/events example/images example/output $skip $insert

- **`checkpoint.bin`**: Il modello pre-addestrato.
- **`example/events`** e **`example/images`**: I dati di input (eventi e immagini).
- **`example/output`**: La directory in cui verranno salvati i frame interpolati.
- **`skip`** e **`insert`**: Parametri che determinano quanti frame saltare e quanti interpolare. Ad esempio:
  - `insert=7` inserisce 7 frame intermedi per ogni coppia di frame.

---

### Creazione del video

Dopo aver generato i frame interpolati, è possibile combinare le immagini in un video:

    ffmpeg -i example/output/%06d.png timelens.mp4

Il video risultante sarà salvato come `timelens.mp4`.

---

## Dataset

Per test più avanzati, è possibile scaricare il dataset completo dalla [pagina ufficiale del progetto](http://rpg.ifi.uzh.ch/timelens). La struttura del dataset è la seguente:

    .
    ├── close
    │   └── test
    │       ├── baloon_popping
    │       │   ├── events_aligned
    │       │   └── images_corrected
    │       ├── candle
    │       │   ├── events_aligned
    │       │   └── images_corrected
    │       ...
    │
    └── far
        └── test
            ├── bridge_lake_01
            │   ├── events_aligned
            │   └── images_corrected
            ├── bridge_lake_03
            │   ├── events_aligned
            │   └── images_corrected
            ...

Ogni cartella `events_aligned` contiene file `.npz` che rappresentano gli eventi tra due immagini consecutive. Ogni file `.npz` include:

- **`x` e `y`**: Coordinate spaziali dei pixel in cui si sono verificati eventi.
- **`t`**: Timestamps degli eventi.
- **`p`**: Polarità degli eventi (cambiamenti di luminosità positivi o negativi).

Le immagini sono contenute nella directory `images_corrected`. I timestamp corrispondenti sono elencati in `timestamp.txt`.

---

## Modifiche rispetto al progetto originale

1. **Rimosso il supporto CUDA**:
   - Il codice è stato modificato per funzionare esclusivamente su CPU, eliminando i riferimenti a CUDA.
2. **Compatibilità con macOS ARM**:
   - Configurazione ottimizzata per macOS ARM (Apple Silicon) utilizzando Miniforge.
3. **Installazione semplificata**:
   - Procedura aggiornata per includere Miniforge e dipendenze specifiche per macOS.
4. **Inclusione di Jupyter**:
   - Installato Jupyter per una possibile integrazione con notebook e analisi interattive.

---

Se si riscontrano problemi o bug, è possibile aprire un'issue nel repository o contattare i maintainer.
