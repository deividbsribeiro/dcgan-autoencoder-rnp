"""
AutoEncoders_Deivid_Ribeiro_v0.4.py – RNP T3  (Etapa 2 – Autoencoder U-Net para Restauração)

Treina um Autoencoder baseado em U-Net, com blocos residuais no decoder, para remover
borramento e artefatos em sprites no estilo*pixel-art. O treinamento
ocorre em duas fases: (1) com sprites reais e (2) fine-tune opcional com sprites
gerados pela DCGAN. Utiliza perda híbrida com termos L1, VGG perceptual e borda (Sobel),
além de scheduler adaptativo (ReduceLROnPlateau).

Avalia o desempenho com métricas de nitidez (VarLap) e erro por grau de desfoque (σ),
gerando gráficos informativos.

Caminhos esperados:
• DATA_DIR     — pasta com sprites reais organizados por classe (usada com ImageFolder);
• GAN_AUX_DIR  — (opcional) sprites gerados pela DCGAN, salvos na classe 0;
• OUT_ROOT     — diretório onde serão criadas subpastas runs/YYYYMMDD_HHMMSS_ae/ com os resultados.

Parâmetros editáveis:
• IMAGE_SIZE       — resolução esperada das imagens de entrada (padrão: 64)
• BATCH_SIZE       — tamanho do lote por iteração
• EPOCHS_MAIN      — épocas da fase 1 (com dados reais)
• EPOCHS_FT        — épocas da fase 2 (fine-tune com dados GAN)
• LR               — taxa de aprendizado
• BETA1            — hiperparâmetro do Adam (beta1)
• LAMBDA_L1        — peso da perda L1 (erro médio absoluto)
• LAMBDA_PERC      — peso da perda perceptual com VGG
• EDGE_W           — peso da perda de borda (Sobel)
• IDENTITY_P       — probabilidade de usar imagem não-borrada como entrada
• DEVICE           — usa CUDA se disponível, senão CPU

Execute com:
    python AutoEncoders_Deivid_Ribeiro_v0.4.py

Autor: Deivid Ribeiro (2025)
"""

from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms, models
from torchvision.transforms import GaussianBlur
from tqdm import tqdm

# ─────────────────────────── HYPERPARAMETERS ───────────────────────────────
DATA_DIR = Path(r"C:\Users\Deivid_Ribeiro\Desktop\RPN_T3\data")
# colocar sprites GAN aqui (classe 0)
GAN_AUX_DIR = DATA_DIR.parent / "gan_aux"
OUT_ROOT = Path("runs")
IMAGE_SIZE = 64
BATCH_SIZE = 128
EPOCHS_MAIN = 100    # treino inicial
EPOCHS_FT = 20     # fine-tune com dados GAN
LR = 5e-4
BETA1 = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LAMBDA_L1 = 0.8
LAMBDA_PERC = 0.2
EDGE_W = 0.2
IDENTITY_P = 0.20   # aumenta p/ 0.40 no fine-tune

# ───────────────────────────── AUGMENTATIONS ───────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.05),
    transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3),
])

val_tf = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3),
])

# ────────────────────────────── MODEL DEFINITIONS ──────────────────────────


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
        # encoder
        self.e1 = nn.Sequential(
            nn.Conv2d(3, base, 4, 2, 1), nn.ReLU(True))                    # 32
        self.e2 = nn.Sequential(nn.Conv2d(base, base*2, 4, 2, 1),
                                # 16
                                nn.BatchNorm2d(base*2), nn.ReLU(True))
        self.e3 = nn.Sequential(nn.Conv2d(base*2, base*4, 4, 2, 1),
                                # 8
                                nn.BatchNorm2d(base*4), nn.ReLU(True))
        self.e4 = nn.Sequential(nn.Conv2d(base*4, base*8, 4, 2, 1),
                                # 4
                                nn.BatchNorm2d(base*8), nn.ReLU(True))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base*8, base*16, 4, 2, 1), nn.ReLU(True))     # 2

        # decoder (upsample → concat → conv → ResBlock)
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 4, 2, 1)
        self.dec4 = nn.Sequential(nn.Conv2d(base*16, base*8, 3, 1, 1),
                                  nn.BatchNorm2d(base*8), nn.ReLU(True), ResBlock(base*8))
        self.up3 = nn.ConvTranspose2d(base*8,  base*4, 4, 2, 1)
        self.dec3 = nn.Sequential(nn.Conv2d(base*8,  base*4, 3, 1, 1),
                                  nn.BatchNorm2d(base*4), nn.ReLU(True), ResBlock(base*4))
        self.up2 = nn.ConvTranspose2d(base*4,  base*2, 4, 2, 1)
        self.dec2 = nn.Sequential(nn.Conv2d(base*4,  base*2, 3, 1, 1),
                                  nn.BatchNorm2d(base*2), nn.ReLU(True), ResBlock(base*2))
        self.up1 = nn.ConvTranspose2d(base*2,  base,   4, 2, 1)
        self.dec1 = nn.Sequential(nn.Conv2d(base*2,  base,   3, 1, 1),
                                  nn.BatchNorm2d(base),   nn.ReLU(True), ResBlock(base))
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


class VGGPerceptual(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT).features.eval()
        self.slice = nn.Sequential(*list(vgg.children())[:17]).to(DEVICE)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x): return self.slice(x)

# ───────────────────────────── LOSS FUNCTIONS ──────────────────────────────


def perceptual_loss(vgg, x_hat, x):
    def t(z): return transforms.functional.normalize((z+1)/2,
                                                     [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return F.l1_loss(vgg(t(x_hat)), vgg(t(x)))


def edge_loss(x_hat, target):
    sob = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=x_hat.device,
                       dtype=torch.float32).view(1, 1, 3, 3)/8
    sob_x = sob.repeat(3, 1, 1, 1)
    sob_y = sob_x.transpose(2, 3)
    gx = F.conv2d(x_hat, sob_x, padding=1, groups=3) - \
        F.conv2d(target, sob_x, padding=1, groups=3)
    gy = F.conv2d(x_hat, sob_y, padding=1, groups=3) - \
        F.conv2d(target, sob_y, padding=1, groups=3)
    return (gx.abs()+gy.abs()).mean()


def gaussian_blur(img, sigma):  # torchvision blur
    return GaussianBlur(5, sigma=(sigma, sigma))(img)


def variance_of_laplacian(img):
    arr = ((img+1)*127.5).clamp(0, 255).byte().cpu().numpy()
    return float(np.mean([cv2.Laplacian(c, cv2.CV_64F).var() for c in arr]))

# ────────────────────────────── TRAINER UTILS ──────────────────────────────


def evaluate_sigma(model, loader, out):
    model.eval()
    imgs, _ = next(iter(loader))
    imgs = imgs[:16].to(DEVICE)
    sig = np.arange(0.5, 1.51, 0.2)
    blur, loss = [], []
    with torch.no_grad():
        for s in sig:
            x_hat = model(gaussian_blur(imgs, s))
            blur.append(np.mean([variance_of_laplacian(im) for im in x_hat]))
            loss.append(F.l1_loss(x_hat, imgs).item())
    plt.figure()
    plt.plot(sig, blur, 'o-')
    plt.xlabel('Sigma')
    plt.ylabel('VarLap')
    plt.title('Blur vs Sigma')
    plt.xticks(sig)
    plt.savefig(out/'blur_vs_sigma.png')
    plt.close()
    plt.figure()
    plt.plot(sig, loss, 'o-')
    plt.xlabel('Sigma')
    plt.ylabel('Loss')
    plt.title('Loss vs Sigma')
    plt.xticks(sig)
    plt.savefig(out/'loss_vs_sigma.png')
    plt.close()
    model.train()

# ─────────────────────────────── MAIN LOOP ────────────────────────────────


def train_phase(model, loader, val_loader, vgg, opt, sched,
                epochs, identity_p, run_dir, start_ep=1):
    hist_tr, hist_val = [], []
    for ep in range(start_ep, start_ep+epochs):
        model.train()
        run = 0.0
        bar = tqdm(loader, desc=f"Epoch {ep}/{start_ep+epochs-1}", leave=False)
        for x, _ in bar:
            x = x.to(DEVICE)
            if np.random.rand() < identity_p:
                x_in, target = x, x
            else:
                x_in = gaussian_blur(x, float(np.random.uniform(0.5, 1.5)))
                target = x
            x_hat = model(x_in)
            loss = (LAMBDA_L1*F.l1_loss(x_hat, target) +
                    LAMBDA_PERC*perceptual_loss(vgg, x_hat, target) +
                    EDGE_W*edge_loss(x_hat, target))
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        hist_tr.append(run/len(loader))

        # validation
        model.eval()
        vrun = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(DEVICE)
                x_blur = gaussian_blur(x, 1.0)
                x_hat = model(x_blur)
                vrun += (LAMBDA_L1*F.l1_loss(x_hat, x) +
                         LAMBDA_PERC*perceptual_loss(vgg, x_hat, x) +
                         EDGE_W*edge_loss(x_hat, x)).item()
        val = vrun/len(val_loader)
        hist_val.append(val)
        sched.step(val)
        print(f"Epoch {ep:3d}: train={hist_tr[-1]:.4f} | val={val:.4f}")

    # plot loss history
    ep_axis = range(start_ep, start_ep+epochs)
    plt.figure()
    plt.plot(ep_axis, hist_tr, label='Train')
    plt.plot(ep_axis, hist_val, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.savefig(run_dir/'loss_vs_epoch.png')
    plt.close()
    return ep_axis[-1]


def main():
    # datasets
    base_set = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    n_val = int(0.1*len(base_set))
    train_set, val_set = random_split(base_set, [len(base_set)-n_val, n_val])
    # optional GAN sprites
    if GAN_AUX_DIR.exists():
        gan_set = datasets.ImageFolder(GAN_AUX_DIR, transform=train_tf)
        train_set = ConcatDataset([train_set, gan_set])

    val_set.dataset.transform = val_tf
    pin = DEVICE == "cuda"
    train_loader = DataLoader(train_set, BATCH_SIZE,
                              shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_set,  BATCH_SIZE,
                            shuffle=False, pin_memory=pin)

    model = UNetAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(BETA1, 0.999))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                       factor=0.5, patience=5, min_lr=1e-5)
    vgg = VGGPerceptual()
    run_dir = OUT_ROOT/dt.datetime.now().strftime("%Y%m%d_%H%M%S_ae")
    run_dir.mkdir(parents=True)

    # phase 1
    last_ep = train_phase(model, train_loader, val_loader, vgg, opt, sched,
                          EPOCHS_MAIN, IDENTITY_P, run_dir, 1)
    # phase 2 fine-tune (maior IDENTITY_P se gan aux presente)
    if GAN_AUX_DIR.exists():
        train_phase(model, train_loader, val_loader, vgg, opt, sched,
                    EPOCHS_FT, identity_p=0.4, run_dir=run_dir, start_ep=last_ep+1)

    evaluate_sigma(model, val_loader, run_dir)
    torch.save(model.state_dict(), run_dir/'autoencoder.pth')
    print(f"[INFO] Completed. Results in {run_dir}")


if __name__ == "__main__":
    main()
