"""
DCGAN_Deivid_Ribeiro_v0.4.py – RNP T3  (Etapa 1 – DCGAN para Sprites Pixel-Art)

Treina uma rede adversarial generativa (DCGAN) para sintetizar personagens
estilo *pixel-art* com resolução \(64\times64\) px, a partir de um vetor de
ruído configurável. O gerador aprende a imitar sprites
reais, enquanto o discriminador tenta distingui-los das amostras sintéticas.
Ao final de cada época, salva uma grade de 64 imagens para análise visual
da convergência.

Caminhos esperados:
• DATA_DIR — pasta com 4 subpastas (\0 \1 \2 e \3) contendo os sprites reais organizados
            para uso com ImageFolder;
• OUT_ROOT — diretório onde serão salvos os checkpoints e imagens geradas;
• O script cria uma subpasta por execução em runs/YYYYMMDD_HHMMSS/.

Parâmetros editáveis:
• EPOCHS            — número total de épocas de treinamento
• BATCH_SIZE        — tamanho do lote por iteração
• IMAGE_SIZE        — resolução das imagens (recomendado pela DCGAN)
• Z_DIM             — dimensão do vetor de ruído \(\mathbf{z}\)
• FEATURE_G         — fator base de canais no gerador (máx: 512)
• FEATURE_D         — fator base de canais no discriminador
• LR                — taxa de aprendizado para ambos os otimizadores
• BETA1             — hiperparâmetro do Adam (beta1)
• NUM_WORKERS       — número de workers no DataLoader (use 0 no Windows)
• SAMPLE_FREQ       — frequência (em épocas) para salvar imagens
• DEVICE            — usa CUDA se disponível, senão CPU

Execute com:
    python DCGAN_Deivid_Ribeiro_v0.4.py

Autor: Deivid Ribeiro (2025)
"""

from __future__ import annotations

import os
import datetime as dt
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
from tqdm import tqdm

# ────────────────────────── PARÂMETROS EDITÁVEIS ─────────────────────────────
DATA_DIR = r"C:\Users\Deivid_Ribeiro\Desktop\RPN_T3\data"  # 4 sub‑pastas (0‑3)
OUT_ROOT = Path("runs")
EPOCHS = 700
BATCH_SIZE = 128
IMAGE_SIZE = 64         # 64×64 recomendado pela DCGAN original
Z_DIM = 1024        # tamanho do vetor de ruído
FEATURE_G = 128        # * 8 = 512 canais máximo no Gerador
FEATURE_D = 128        # * 8 = 512 canais máximo no Discriminador
LR = 8e-5
BETA1 = 0.5        # betas (0.5, 0.999) clássicos
NUM_WORKERS = 0          # WIN: deixe 0 para evitar spawn issues; Linux pode 2‑4
SAMPLE_FREQ = 1          # salvar grade de 64 imagens a cada N épocas
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

torch.backends.cudnn.benchmark = True  # acelera convoluções fixas

# ╭────────────────────────── 1. DATASET ─────────────────────────────────────╮
TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    # 4+ augmentations
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),   # [-1,1]
])

dataset = datasets.ImageFolder(DATA_DIR, transform=TRANSFORM)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)
print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Dataset size: {len(dataset)}  |  Batches/epoch: {len(loader)}")

# ╭────────────────────────── 2. MODELOS ─────────────────────────────────────╮


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # input z_dim × 1 × 1
            nn.ConvTranspose2d(Z_DIM, FEATURE_G*8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(FEATURE_G*8), nn.ReLU(True),

            nn.ConvTranspose2d(FEATURE_G*8, FEATURE_G*4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(FEATURE_G*4), nn.ReLU(True),

            nn.ConvTranspose2d(FEATURE_G*4, FEATURE_G*2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(FEATURE_G*2), nn.ReLU(True),

            nn.ConvTranspose2d(FEATURE_G*2, FEATURE_G, 4, 2, 1, bias=True),
            nn.BatchNorm2d(FEATURE_G), nn.ReLU(True),

            nn.ConvTranspose2d(FEATURE_G, 3, 4, 2, 1, bias=True),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # input 3 × 64 × 64
            nn.Conv2d(3, FEATURE_D, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(FEATURE_D, FEATURE_D*2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(FEATURE_D*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(FEATURE_D*2, FEATURE_D*4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(FEATURE_D*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(FEATURE_D*4, FEATURE_D*8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(FEATURE_D*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(FEATURE_D*8, 1, 4, 1, 0, bias=True),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


# ╭────────────────────────── 3. TREINAMENTO ─────────────────────────────────╮
BCE = nn.BCEWithLogitsLoss()

generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

opt_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
opt_D = torch.optim.Adam(discriminator.parameters(),
                         lr=LR, betas=(BETA1, 0.999))

fixed_noise = torch.randn(64, Z_DIM, 1, 1, device=DEVICE)


def save_samples(epoch: int, out_dir: Path):
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
    grid = vutils.make_grid(fake, nrow=8, normalize=True, value_range=(-1, 1))
    vutils.save_image(grid, out_dir / f"samples_epoch_{epoch:03d}.png")
    generator.train()


def main():
    run_dir = OUT_ROOT / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Out dir: {run_dir}")

    loss_D_epoch = 0
    loss_G_epoch = 0

    for epoch in range(1, EPOCHS+1):
        loop = tqdm(
            loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch", leave=False)
        for real, _ in loop:
            real = real.to(DEVICE, non_blocking=True)
            bsz = real.size(0)

            # ---------- Train Discriminator ----------
            noise = torch.randn(bsz, Z_DIM, 1, 1, device=DEVICE)
            fake = generator(noise)

            d_real = discriminator(real).view(-1)
            d_fake = discriminator(fake.detach()).view(-1)
            loss_D = (BCE(d_real, torch.ones_like(d_real)) +
                      BCE(d_fake, torch.zeros_like(d_fake))) / 2
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ---------- Train Generator --------------
            d_fake = discriminator(fake).view(-1)
            loss_G = BCE(d_fake, torch.ones_like(d_fake))
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            loss_D_epoch += loss_D.item()
            loss_G_epoch += loss_G.item()

            loop.set_postfix({"loss_D": f"{loss_D.item():.4f}",
                              "loss_G": f"{loss_G.item():.4f}"})

        # Média das perdas por época
        avg_loss_D = loss_D_epoch / len(loader)
        avg_loss_G = loss_G_epoch / len(loader)

        print(
            f"Epoch [{epoch}/{EPOCHS}]  Avg Loss_D: {avg_loss_D:.4f}  Avg Loss_G: {avg_loss_G:.4f}")

        # Reset acumuladores
        loss_D_epoch = 0
        loss_G_epoch = 0

        if epoch % SAMPLE_FREQ == 0:
            save_samples(epoch, run_dir)
            torch.save(generator.state_dict(), run_dir / "G.pth")
            torch.save(discriminator.state_dict(), run_dir / "D.pth")

    print("[INFO] Treinamento concluído!")


if __name__ == "__main__":
    main()
