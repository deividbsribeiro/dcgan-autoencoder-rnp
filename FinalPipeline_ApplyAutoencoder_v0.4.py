"""

FinalPipeline_ApplyAutoencoder_v0.4.py – RNP T3 (Pós-processamento – Avaliação Final com Autoencoder)

Aplica um Autoencoder U-Net treinado para restaurar sprites gerados pela DCGAN,
seguindo o mesmo processo de degradação sintética usado durante o treinamento.
Cada sprite original é borrado com filtro gaussiano com os mesmos padrões do utilizado no treinamento da rede
apenas para fins de comparação visual. Essa imagem degradada não é utilizada no pipeline real,
mas permite avaliar a eficácia do modelo frente à mesma perturbação vista em treino.

As três versões — original, borrada e restaurada — são combinadas horizontalmente em uma
imagem tripla e salvas com sufixo *_triplet.png. Também são reconstruídas automaticamente
grades 8x3 para facilitar a exportação em blocos ou integração com motores de jogo.
Além disso, essas imagens tripla são reorganizadas automaticamente em três grades 8×8,
correspondentes aos sprites originais, borrados e restaurados separadamente,
facilitando análises visuais isoladas ou exportação em blocos por categoria.
Por fim uma última imagem é gerada concatenando estas últimas 3, em uma matrix 24x8 sprites.

Caminhos esperados:
• MODEL_PATH      — caminho para o arquivo .pth do Autoencoder treinado;
• GAN_IMG_DIR     — pasta com sprites originais (ex: saídas do DCGAN já cortadas);
• OUT_DIR         — diretório onde serão salvos os arquivos restaurados e reagrupados;
• As imagens geradas terão o sufixo *_triplet.png (original | borrada | restaurada).

Execute com:
    python FinalPipeline_ApplyAutoencoder_v0.4.py

Autor: Deivid Ribeiro (2025)
"""

from __future__ import annotations
from pathlib import Path
from typing import List
import random
import re
import sys

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np
import cv2

# ─────────────── EDITAR AQUI ───────────────
MODEL_PATH = Path(r"runs\ENCODER\20250618_132319_ae\autoencoder.pth")
GAN_IMG_DIR = Path(r"sprites_split")
OUT_DIR = Path(r"gan_restored")
# ────────────────────────────────────────────

IMAGE_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────── Função de borramento (apenas para visualização) ───────


def apply_random_blur(pil_img: Image.Image, sigma_range=(0.5, 1.5)) -> Image.Image:
    sigma = random.uniform(*sigma_range)
    img_np = np.array(pil_img)
    img_blurred = cv2.GaussianBlur(img_np, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return Image.fromarray(img_blurred)

# ─────── Arquitetura do Autoencoder U-Net com ResBlocks ───────


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, 1, 1))

    def forward(self, x): return x + self.conv(x)


class UNetAE(nn.Module):
    def __init__(self, base: int = 64):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(3, base, 4, 2, 1), nn.ReLU(True))
        self.e2 = nn.Sequential(nn.Conv2d(base, base*2, 4, 2, 1),
                                nn.BatchNorm2d(base*2), nn.ReLU(True))
        self.e3 = nn.Sequential(nn.Conv2d(base*2, base*4, 4, 2, 1),
                                nn.BatchNorm2d(base*4), nn.ReLU(True))
        self.e4 = nn.Sequential(nn.Conv2d(base*4, base*8, 4, 2, 1),
                                nn.BatchNorm2d(base*8), nn.ReLU(True))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base*8, base*16, 4, 2, 1), nn.ReLU(True))
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 4, 2, 1)
        self.dec4 = nn.Sequential(nn.Conv2d(base*16, base*8, 3, 1, 1),
                                  nn.BatchNorm2d(base*8), nn.ReLU(True), ResBlock(base*8))
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 4, 2, 1)
        self.dec3 = nn.Sequential(nn.Conv2d(base*8, base*4, 3, 1, 1),
                                  nn.BatchNorm2d(base*4), nn.ReLU(True), ResBlock(base*4))
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 4, 2, 1)
        self.dec2 = nn.Sequential(nn.Conv2d(base*4, base*2, 3, 1, 1),
                                  nn.BatchNorm2d(base*2), nn.ReLU(True), ResBlock(base*2))
        self.up1 = nn.ConvTranspose2d(base*2, base, 4, 2, 1)
        self.dec1 = nn.Sequential(nn.Conv2d(base*2, base, 3, 1, 1),
                                  nn.BatchNorm2d(base), nn.ReLU(True), ResBlock(base))
        self.up0 = nn.ConvTranspose2d(base, base, 4, 2, 1)
        self.out_conv = nn.Conv2d(base, 3, 3, 1, 1)
        self.act = nn.Tanh()

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        b = self.bottleneck(e4)
        d4 = self.dec4(torch.cat([self.up4(b),  e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.act(self.out_conv(self.up0(d1)))


# ─────── Carregar modelo treinado ───────
ae = UNetAE().to(DEVICE)
ae.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=True)
ae.eval()

to_pil = transforms.ToPILImage()
img_paths: List[Path] = sorted(
    p for p in GAN_IMG_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
)

# ─────── Processar sprites individuais ───────
for idx, p in enumerate(img_paths):
    orig = Image.open(p).convert("RGB")
    ow, oh = orig.size

    resized = orig.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    blurred = apply_random_blur(resized)

    # entrada = original (não borrada)
    inp = to_tensor(resized).unsqueeze(0).to(DEVICE) * 2 - 1

    with torch.no_grad():
        out = ae(inp).cpu().squeeze(0).clamp(-1, 1)

    restored = to_pil((out + 1) / 2).resize((ow, oh), Image.BICUBIC)
    blurred_full = blurred.resize((ow, oh), Image.BICUBIC)

    combo = Image.new("RGB", (ow * 3, oh))
    combo.paste(orig, (0, 0))
    combo.paste(blurred_full, (ow, 0))
    combo.paste(restored, (ow * 2, 0))
    combo.save(OUT_DIR / f"sprite_{idx:02}_triplet.png")

print(f"[OK] Sprites restaurados salvos em: {OUT_DIR}")

# ─────── Construir folhas 8×8 ───────
SPRITE_W = SPRITE_H = 64
COLS = ROWS = 8
SHEET_W = SPRITE_W * COLS
SHEET_H = SPRITE_H * ROWS

sheet_orig = Image.new("RGB", (SHEET_W, SHEET_H))
sheet_blur = Image.new("RGB", (SHEET_W, SHEET_H))
sheet_rest = Image.new("RGB", (SHEET_W, SHEET_H))

for idx in range(64):
    path = OUT_DIR / f"sprite_{idx:02}_triplet.png"
    img = Image.open(path).convert("RGB")
    orig = img.crop((0,   0,  64, 64))
    blur = img.crop((64,  0, 128, 64))
    rest = img.crop((128, 0, 192, 64))

    col, row = idx % COLS, idx // COLS
    dest = (col * SPRITE_W, row * SPRITE_H)

    sheet_orig.paste(orig, dest)
    sheet_blur.paste(blur, dest)
    sheet_rest.paste(rest, dest)

sheet_orig.save(OUT_DIR / "spritesheet_original.png")
sheet_blur.save(OUT_DIR / "spritesheet_borrada.png")
sheet_rest.save(OUT_DIR / "spritesheet_restaurada.png")

sheet_tripla = Image.new("RGB", (SHEET_W * 3, SHEET_H))
sheet_tripla.paste(sheet_orig, (0, 0))
sheet_tripla.paste(sheet_blur, (SHEET_W, 0))
sheet_tripla.paste(sheet_rest, (SHEET_W * 2, 0))
sheet_tripla.save(OUT_DIR / "spritesheet_tripla.png")

print("[OK] Folhas 8×8 geradas com sucesso.")
