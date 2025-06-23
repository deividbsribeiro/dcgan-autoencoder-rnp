README.txt
==========

Pixel-Art Sprite Generation & Restoration
DCGAN  +  U-Net Autoencoder
Author: Deivid Ribeiro · 2025
------------------------------------------------------------------
End-to-end pipeline to

1. **Generate** 64 × 64 px pixel-art sprites with a DCGAN  
2. **Restore** blurry sprites with a U-Net auto-encoder  
3. **Assemble** ready-to-use sprite-sheets for any 2-D game engine

Main Scripts
------------
| Script                                | Purpose                                                |
| ------------------------------------- | ------------------------------------------------------ |
| `DCGAN_Deivid_Ribeiro_v0.4.py`        | Train the DCGAN and save 8 × 8 sample grids            |
| `AutoEncoders_Deivid_Ribeiro_v0.4.py` | Train the U-Net auto-encoder (2 phases)                |
| `FinalPipeline_ApplyAutoencoder_v0.4.py` | Restore GAN sprites, build triplets & sprite-sheets |

Prerequisites
-------------
Python 3.10+ and the libraries below:

torch torchvision opencv-python pillow matplotlib tqdm

torch>=2.1.0
torchvision>=0.16.0
pillow>=9.5.0
opencv-python>=4.8.0
matplotlib>=3.7.1
tqdm>=4.66.1

Folder Layout
-------------
project_root/
│
├─ data/ ← real sprites for training the DCGAN & AE
│ ├─ 0/ 1/ 2/ 3/ four poses (ImageFolder format, 3648 PNGs)
│
├─ gan_aux/0/ ← optional: extra GAN sprites for AE fine-tune
│
├─ runs/ ← auto-created by the training scripts
│ ├─ YYYYMMDD_HHMMSS/ (DCGAN checkpoints + sample grids)
│ └─ YYYYMMDD_HHMMSS_ae/ (Auto-encoder checkpoints + plots)
│
├─ sprites_split/ ← 64 PNGs 64 × 64 cut from one DCGAN grid
│ └─ sprite_00.png … sprite_63.png
│
├─ gan_restored/ ← final outputs
│ ├─ sprite_00_triplet.png … sprite_63_triplet.png
│ ├─ spritesheet_original.png (8 × 8 originals)
│ ├─ spritesheet_borrada.png (8 × 8 blurred)
│ ├─ spritesheet_restaurada.png (8 × 8 restored)
│ └─ spritesheet_tripla.png (24 × 8 mega-sheet)
│
└─ *.py (the three scripts listed above)

*If your folders differ, edit `DATA_DIR`, `GAN_IMG_DIR`, etc. at the top
 of each script.*

Step-by-Step
------------

### 1 – Train the DCGAN

python DCGAN_Deivid_Ribeiro_v0.4.py

*Input* : `data/0-3/`  
*Output*: `runs/…/samples_epoch_XXX.png`

Cut one 8 × 8 sample grid into 64 PNGs and drop them in `sprites_split/`
(any sprite-sheet slicer will do).

### 2 – Train the U-Net Auto-encoder

python AutoEncoders_Deivid_Ribeiro_v0.4.py

*Phase 1* : real sprites · *Phase 2* (optional) : `gan_aux/0/`  
*Output*  : `runs/…_ae/autoencoder.pth`  + 3 training plots

### 3 – Restore sprites & build sprite-sheets

python FinalPipeline_ApplyAutoencoder_v0.4.py

*Input*  :
  • `sprites_split/` (64 PNG originals)  
  • `autoencoder.pth` from *runs/…_ae/*  

*What it does* :
1. Creates a blurred copy (for comparison only)  
2. Restores each sprite with the AE  
3. Saves `original | blurred | restored` as `sprite_NN_triplet.png`  
4. Auto-builds four sheets:  
   `spritesheet_original.png`, `spritesheet_borrada.png`,  
   `spritesheet_restaurada.png`, `spritesheet_tripla.png`

Tips & Tweaks
-------------
* All hyper-parameters live at the top of each script (`EPOCHS`, `LR`, …)  
* CUDA is used automatically if available (`DEVICE = "cuda"`).  
* Increase `EPOCHS` (DCGAN) or `EPOCHS_FT` (U-Net) for higher quality.  
* PNG sheets import directly into most retro-game engines (rows contiguous).

Happy hacking!  😊