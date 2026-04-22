import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURACIÓN — ajusta estas rutas
# ─────────────────────────────────────────────
IMG_DIR  = Path("img_align_celeba/img_align_celeba/")   # carpeta con las imágenes
ATTR_CSV = Path("list_attr_celeba.csv")
N_SAMPLE = 500    # imágenes para calcular distribución de píxeles (más rápido)
IMG_SIZE = (64, 64)  # resolución a la que redimensionas para NMF

# ─────────────────────────────────────────────
# 1. MÉTRICAS GENERALES
# ─────────────────────────────────────────────
# Contar imágenes disponibles
img_files = sorted(IMG_DIR.glob("*.jpg"))
n_images  = len(img_files)

# Resolución original de la primera imagen
sample_img = Image.open(img_files[0])
orig_w, orig_h = sample_img.size
orig_pixels    = orig_w * orig_h

# Dimensión m para NMF (píxeles tras redimensionar a escala de grises)
m_nmf = IMG_SIZE[0] * IMG_SIZE[1]

print("=" * 45)
print("MÉTRICAS GENERALES")
print("=" * 45)
print(f"Imágenes totales:       {n_images:,}")
print(f"Resolución original:    {orig_w} × {orig_h} px")
print(f"Píxeles por imagen:     {orig_pixels:,}")
print(f"Resolución NMF ({IMG_SIZE[0]}×{IMG_SIZE[1]}): {m_nmf:,} píxeles")
print(f"Tamaño de X (NMF):      {m_nmf} × {n_images}")

# ─────────────────────────────────────────────
# 2. DISTRIBUCIÓN DE PÍXELES
# ─────────────────────────────────────────────
# Muestreamos N_SAMPLE imágenes aleatorias para ser eficientes
np.random.seed(42)
sample_paths = np.random.choice(img_files, size=N_SAMPLE, replace=False)

pixel_values = []
for path in sample_paths:
    img = Image.open(path).convert("L")        # escala de grises
    img = img.resize(IMG_SIZE)                 # redimensionar
    arr = np.array(img).flatten() / 255.0      # normalizar a [0,1]
    pixel_values.append(arr)

pixel_values = np.concatenate(pixel_values)   # shape: (N_SAMPLE * m_nmf,)

print("\n" + "=" * 45)
print("DISTRIBUCIÓN DE PÍXELES (muestra de %d imgs)" % N_SAMPLE)
print("=" * 45)
print(f"Media:       {pixel_values.mean():.4f}")
print(f"Desviación:  {pixel_values.std():.4f}")
print(f"Mínimo:      {pixel_values.min():.4f}")
print(f"Máximo:      {pixel_values.max():.4f}")
print(f"Percentil 25: {np.percentile(pixel_values, 25):.4f}")
print(f"Percentil 75: {np.percentile(pixel_values, 75):.4f}")

# ─────────────────────────────────────────────
# 3. ATRIBUTOS
# ─────────────────────────────────────────────
attrs = pd.read_csv(ATTR_CSV)

# CelebA usa -1/1; convertimos a 0/1
attr_cols = [c for c in attrs.columns if c != "image_id"]
attrs[attr_cols] = (attrs[attr_cols] == 1).astype(int)

freq = attrs[attr_cols].mean().sort_values(ascending=False)

print("\n" + "=" * 45)
print("TOP 10 ATRIBUTOS MÁS FRECUENTES")
print("=" * 45)
for attr, val in freq.head(10).items():
    print(f"  {attr:<25} {val*100:.1f}%")

print("\nBOTTOM 5 ATRIBUTOS MENOS FRECUENTES")
print("-" * 45)
for attr, val in freq.tail(5).items():
    print(f"  {attr:<25} {val*100:.1f}%")

# ─────────────────────────────────────────────
# 4. SPLIT — estadísticas
# ─────────────────────────────────────────────
n_train = int(0.70 * n_images)
n_val   = int(0.20 * n_images)
n_test  = n_images - n_train - n_val

print("\n" + "=" * 45)
print("SPLIT 70 / 20 / 10")
print("=" * 45)
print(f"  Train: {n_train:,} imágenes")
print(f"  Val:   {n_val:,} imágenes")
print(f"  Test:  {n_test:,} imágenes")
print(f"  Tamaño X_train: {m_nmf} × {n_train}")

# ─────────────────────────────────────────────
# 5. GRÁFICAS EDA
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(16, 7))
fig.patch.set_facecolor("#F4F3EE")
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

BLUE   = "#2156B2"
RED    = "#B83A1A"
PURPLE = "#5C4FAA"
MUTED  = "#7A7870"
DARK   = "#12130F"

# ── 5a. Histograma de píxeles ──────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor("white")
counts, bin_edges = np.histogram(pixel_values, bins=16, range=(0, 1))
centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax1.bar(centers, counts / counts.sum() * 100,
        width=bin_edges[1]-bin_edges[0] - 0.005,
        color=BLUE, alpha=0.9, linewidth=0)
ax1.set_xlabel("Intensidad normalizada [0, 1]", color=MUTED, fontsize=10)
ax1.set_ylabel("Frecuencia (%)", color=MUTED, fontsize=10)
ax1.set_title("Distribución de píxeles", fontsize=12, color=DARK, pad=10)
ax1.tick_params(colors=MUTED)
for spine in ax1.spines.values(): spine.set_color("#D4D2CB")
ax1.set_xlim(0, 1)

# ── 5b. Atributos (top 10 barras horizontales) ─
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor("white")
top10 = freq.head(10)
bars  = ax2.barh(top10.index[::-1], top10.values[::-1] * 100,
                 color=PURPLE, alpha=0.9, linewidth=0, height=0.6)
for bar, val in zip(bars, top10.values[::-1]):
    ax2.text(val * 100 + 0.5, bar.get_y() + bar.get_height()/2,
             f"{val*100:.0f}%", va="center", color=PURPLE, fontsize=9)
ax2.set_xlabel("% de imágenes con el atributo", color=MUTED, fontsize=10)
ax2.set_title("Top 10 atributos más frecuentes", fontsize=12, color=DARK, pad=10)
ax2.tick_params(colors=MUTED)
for spine in ax2.spines.values(): spine.set_color("#D4D2CB")
ax2.set_xlim(0, 105)

# ── 5c. Split visual ───────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor("white")
splits = [n_train, n_val, n_test]
labels = [f"Train\n{n_train:,}\n(70%)", f"Val\n{n_val:,}\n(20%)", f"Test\n{n_test:,}\n(10%)"]
colors = [BLUE, "#1A6B3C", RED]
bars3  = ax3.bar(labels, splits, color=colors, alpha=0.9, linewidth=0, width=0.5)
for bar, val in zip(bars3, splits):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f"{val:,}", ha="center", color=DARK, fontsize=10, fontweight="bold")
ax3.set_ylabel("Número de imágenes", color=MUTED, fontsize=10)
ax3.set_title("Split train / val / test", fontsize=12, color=DARK, pad=10)
ax3.tick_params(colors=MUTED)
for spine in ax3.spines.values(): spine.set_color("#D4D2CB")
ax3.set_ylim(0, max(splits) * 1.15)

plt.suptitle("EDA — CelebA Image NMF", fontsize=16, color=DARK, y=1.03, fontweight="bold")
plt.savefig("eda_celeba_real.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("\nGráfica guardada en eda_celeba_real.png")

# ─────────────────────────────────────────────
# 6. GRÁFICAS INDIVIDUALES
# ─────────────────────────────────────────────

# Gráfica 1: Histograma de píxeles
fig1, ax1 = plt.subplots(figsize=(8, 5))
fig1.patch.set_facecolor("#F4F3EE")
ax1.set_facecolor("white")
counts, bin_edges = np.histogram(pixel_values, bins=16, range=(0, 1))
centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax1.bar(centers, counts / counts.sum() * 100,
    width=bin_edges[1]-bin_edges[0] - 0.005,
    color=BLUE, alpha=0.9, linewidth=0)
ax1.set_xlabel("Intensidad normalizada [0, 1]", color=MUTED, fontsize=10)
ax1.set_ylabel("Frecuencia (%)", color=MUTED, fontsize=10)
ax1.set_title("Distribución de píxeles", fontsize=12, color=DARK, pad=10)
ax1.tick_params(colors=MUTED)
for spine in ax1.spines.values(): spine.set_color("#D4D2CB")
ax1.set_xlim(0, 1)
plt.savefig("eda_1_histogram.png", dpi=150, bbox_inches="tight", facecolor=fig1.get_facecolor())
plt.show()

# Gráfica 2: Atributos top 10
fig2, ax2 = plt.subplots(figsize=(8, 6))
fig2.patch.set_facecolor("#F4F3EE")
ax2.set_facecolor("white")
top10 = freq.head(10)
bars = ax2.barh(top10.index[::-1], top10.values[::-1] * 100,
        color=PURPLE, alpha=0.9, linewidth=0, height=0.6)
for bar, val in zip(bars, top10.values[::-1]):
    ax2.text(val * 100 + 0.5, bar.get_y() + bar.get_height()/2,
         f"{val*100:.0f}%", va="center", color=PURPLE, fontsize=9)
ax2.set_xlabel("% de imágenes con el atributo", color=MUTED, fontsize=10)
ax2.set_title("Top 10 atributos más frecuentes", fontsize=12, color=DARK, pad=10)
ax2.tick_params(colors=MUTED)
for spine in ax2.spines.values(): spine.set_color("#D4D2CB")
ax2.set_xlim(0, 105)
plt.savefig("eda_2_attributes.png", dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.show()

# Gráfica 3: Split train/val/test
fig3, ax3 = plt.subplots(figsize=(8, 5))
fig3.patch.set_facecolor("#F4F3EE")
ax3.set_facecolor("white")
splits = [n_train, n_val, n_test]
labels = [f"Train\n{n_train:,}\n(70%)", f"Val\n{n_val:,}\n(20%)", f"Test\n{n_test:,}\n(10%)"]
colors = [BLUE, "#1A6B3C", RED]
bars3 = ax3.bar(labels, splits, color=colors, alpha=0.9, linewidth=0, width=0.5)
for bar, val in zip(bars3, splits):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
         f"{val:,}", ha="center", color=DARK, fontsize=10, fontweight="bold")
ax3.set_ylabel("Número de imágenes", color=MUTED, fontsize=10)
ax3.set_title("Split train / val / test", fontsize=12, color=DARK, pad=10)
ax3.tick_params(colors=MUTED)
for spine in ax3.spines.values(): spine.set_color("#D4D2CB")
ax3.set_ylim(0, max(splits) * 1.15)
plt.savefig("eda_3_split.png", dpi=150, bbox_inches="tight", facecolor=fig3.get_facecolor())
plt.show()

print("Gráficas individuales guardadas: eda_1_histogram.png, eda_2_attributes.png, eda_3_split.png")