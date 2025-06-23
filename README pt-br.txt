README.txt
==========

Geração e Restauração de Sprites Pixel-Art
DCGAN  +  U-Net Autoencoder
Autor: Deivid Ribeiro · 2025
-------------------------------------------------------------

Este repositório contém uma *pipeline* completa para:

1. **Gerar** sprites 64 × 64 px estilo *pixel-art* via DCGAN.  
2. **Restaurar** (des-blur) esses sprites com um Autoencoder U-Net.  
3. **Organizar** os resultados em folhas (sprite-sheets) prontas para uso.

Arquivos principais
-------------------
| Script                                   | Função                                             |
| ---------------------------------------- | -------------------------------------------------- |
| `DCGAN_Deivid_Ribeiro_v0.4.py`           | Treina a DCGAN e salva grades 8×8 de amostras.     |
| `AutoEncoders_Deivid_Ribeiro_v0.4.py`    | Treina o Autoencoder U-Net (2 fases).             |
| `FinalPipeline_ApplyAutoencoder_v0.4.py` | Restaura sprites da DCGAN e cria sprite-sheets.    |

Pré-requisitos
--------------
Python 3.10 + e as bibliotecas:

torch torchvision opencv-python pillow matplotlib tqdm

torch>=2.1.0
torchvision>=0.16.0
pillow>=9.5.0
opencv-python>=4.8.0
matplotlib>=3.7.1
tqdm>=4.66.1

Estrutura de pastas esperada
----------------------------

project_root/
│
├─ data/ ← sprites reais para treinamento
│ ├─ 0/ 1/ 2/ 3/ quatro poses (ImageFolder)
│ └─ ...
│
├─ gan_aux/0/ ← (opcional) sprites GAN para fine-tune
│
├─ runs/ ← gerado pelos scripts de treino
│ ├─ YYYYMMDD_HHMMSS/ (DCGAN)
│ └─ YYYYMMDD_HHMMSS_ae/ (Autoencoder)
│
├─ sprites_split/ ← 64 PNGs 64×64 extraídos de 1 grade DCGAN
│ └─ sprite_00.png … sprite_63.png
│
├─ gan_restored/ ← resultados finais
│ ├─ sprite_00_triplet.png … sprite_63_triplet.png
│ ├─ spritesheet_original.png (8×8 originais)
│ ├─ spritesheet_borrada.png (8×8 borradas)
│ ├─ spritesheet_restaurada.png (8×8 restauradas)
│ └─ spritesheet_tripla.png (24×8 mega-sheet)
│
└─ *.py (os três scripts acima)

*(Altere variáveis `DATA_DIR`, `GAN_IMG_DIR`, etc. se usar caminhos distintos.)*

Passo-a-Passo
-------------

### 1 ) Treinar a DCGAN

python DCGAN_Deivid_Ribeiro_v0.4.py

*Entrada*: `data/0-3/`  
*Saída*: `runs/YYYYMMDD_HHMMSS/samples_epoch_XXX.png`

>  Corte uma grade 8×8 em 64 PNGs e coloque em `sprites_split/`.  
>  (Pode usar qualquer utilitário de corte; não incluído aqui.)

### 2 ) Treinar o Autoencoder U-Net

python AutoEncoders_Deivid_Ribeiro_v0.4.py

*Fase 1*: sprites reais · *Fase 2* (opcional): `gan_aux/0/`  
*Saída*: `runs/YYYYMMDD_HHMMSS_ae/autoencoder.pth` + gráficos (`loss_*.png`)

### 3 ) Restaurar sprites e gerar sprite-sheets

python FinalPipeline_ApplyAutoencoder_v0.4.py

*Entrada*:  
  • `sprites_split/` (64 PNGs originais)  
  • modelo `.pth` da pasta *runs/..._ae/*

*Processo*:
1. Produz versão borrada (apenas para comparação).  
2. Aplica o U-Net -> gera restaurado.  
3. Salva `original | borrada | restaurada` em `sprite_NN_triplet.png`.  
4. Monta automaticamente quatro folhas:  
   `spritesheet_original.png`, `spritesheet_borrada.png`,  
   `spritesheet_restaurada.png`, `spritesheet_tripla.png`.

Dicas & Ajustes
---------------
*   Hiper-parâmetros editáveis no topo de cada script (`EPOCHS`, `LR`, …).  
*   GPU acelera muito: `DEVICE = "cuda"` é atribuído automaticamente.
*   Aumente `EPOCHS` (DCGAN) ou `EPOCHS_FT` (U-Net) para melhor qualidade.  
*   As folhas PNG podem ser importadas diretamente em engines 2-D retro.

Bom hacking!  😊
