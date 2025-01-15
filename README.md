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

Per installare Miniforge in modo semplice e gestibile, si consiglia di utilizzare **Homebrew**. Eseguire il seguente comando per installare Miniforge:

    brew install --cask miniforge

#### **Configurare Miniforge per la shell Zsh**
Dopo l'installazione, è necessario configurare Miniforge per funzionare correttamente con la shell Zsh. Esegui il seguente comando per inizializzare Conda e riavvia il terminale:
    
    conda init

Se il comando sopra non risolve il problema, aggiungi manualmente il seguente blocco di configurazione al tuo file `.zshrc`:

    nano ~/.zshrc

Copia e incolla il seguente codice nel file `.zshrc`:

    export PATH="/opt/homebrew/Caskroom/miniforge/base/bin:$PATH"

Salva il file (`CTRL+O`, poi `Invio`) ed esci (`CTRL+X`). Successivamente, ricarica il file `.zshrc` con:

    source ~/.zshrc

Se si riscontrano problemi cancella `export PATH="/opt/homebrew/Caskroom/miniforge/base/bin:$PATH"` e incolla questo:

    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$('/opt/homebrew/Caskroom/miniforge/base/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh" ]; then
            . "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh"
        else
            export PATH="/opt/homebrew/Caskroom/miniforge/base/bin:$PATH"
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<

Salva nuovamente il file e ricarica tramite:

    source ~/.zshrc

#### **Verifica dell'installazione**
Assicurati che Miniforge sia stato configurato correttamente eseguendo:
    
    conda --version

Se tutto è configurato correttamente, dovresti vedere una versione di Conda, ad esempio:
    
    conda 23.x.x

### 2. Creazione dell'ambiente Conda

Creare un ambiente Conda e installare le dipendenze richieste:

    conda create -y -n timelens python=3.9
    conda activate timelens
    conda install pytorch torchvision torchaudio -c pytorch
    conda install -y -c conda-forge opencv scipy tqdm click
    brew install ffmpeg

### 3. Clonare il repository

Clonare il codice del progetto:

    mkdir ~/timelens/
    cd ~/timelens
    git clone https://github.com/marcellorussox/rpg_timelens-MacOSXarm64.git rpg_timelens
    cd rpg_timelens

### 4. Scaricare i dati di esempio e il modello pre-addestrato

Scaricare i dati di esempio e il checkpoint per l’interpolazione:

    brew install wget
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

---

Se si riscontrano problemi o bug, è possibile aprire un'issue nel repository o contattare i maintainer.
