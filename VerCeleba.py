import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IMG_SIZE    = (64, 64)
WEIGHTS_DIR = Path("weights")
DATA_FILE   = Path("celeba_64.npy")
EPS         = 1e-10
TRAIN_SEED  = 42
RUN_ID      = "20260503_183846"   # ver runs_log.csv

from NFMceleba import (
    split_indices, solve_H_best,
    plot_bases, plot_compare_methods, plot_rmse_vs_k,
    DATA_FILE, WEIGHTS_DIR, EPS, IMG_SIZE
)


def plot_reconstructions(W, run_id=RUN_ID, n_show=8,
                         eval_steps=300, restarts=3):
    X_full = np.load(DATA_FILE, mmap_mode="r")
    _, _, te = split_indices(len(X_full), seed=TRAIN_SEED)
    L  = np.linalg.norm(W.T @ W)
    aH = min(1e-2, 1.0 / (L + EPS))

    fig, axes = plt.subplots(n_show, 3, figsize=(7, n_show * 2.2))
    if n_show == 1: axes = axes[np.newaxis, :]
    fig.patch.set_facecolor("#12130F")
    fig.suptitle(f"Reconstrucciones NMF — {run_id}", color="white", fontsize=13)

    for j, c in enumerate(["Original", "Reconstruida", "Diferencia"]):
        axes[0, j].set_title(c, color="#BCBAB3", fontsize=10)

    for i in range(n_show):
        x    = X_full[te[i]].astype(np.float32)
        H    = solve_H_best(W, x.reshape(-1,1), aH, eval_steps, restarts)
        xh   = np.clip((W @ H).flatten(), 0, 1)
        rmse = float(np.sqrt(np.mean((x - xh) ** 2)))
        diff = np.abs(x - xh).reshape(IMG_SIZE)

        axes[i, 0].imshow(x.reshape(IMG_SIZE),  cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow(xh.reshape(IMG_SIZE), cmap="gray", vmin=0, vmax=1)
        axes[i, 2].imshow(diff, cmap="hot", vmin=0, vmax=max(float(diff.max()), 0.01))
        axes[i, 0].set_ylabel(f"RMSE={rmse:.4f}", color="#BCBAB3", fontsize=8)
        for j in range(3): axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(f"recons_{run_id}.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"Guardado: recons_{run_id}.png")


def plot_loss(run_id=RUN_ID):
    loss_h = np.load(WEIGHTS_DIR / f"{run_id}_loss.npy").tolist()
    val_s  = np.load(WEIGHTS_DIR / f"{run_id}_val_steps.npy").tolist()
    val_r  = np.load(WEIGHTS_DIR / f"{run_id}_val_rmse.npy").tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#F4F3EE")
    ax1.plot(range(1, len(loss_h)+1), loss_h, color="#2156B2", lw=1.5)
    ax1.set_title("Pérdida de entrenamiento"); ax1.set_xlabel("Step")
    ax1.set_facecolor("white"); ax1.grid(True, alpha=0.3)
    ax2.plot(val_s, val_r, color="#B83A1A", marker="o", markersize=5, lw=2)
    ax2.set_title("RMSE en validación"); ax2.set_xlabel("Step")
    ax2.set_facecolor("white"); ax2.grid(True, alpha=0.3)
    plt.suptitle(f"Curvas — {run_id}")
    plt.tight_layout()
    plt.savefig(f"loss_{run_id}.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    W_path = WEIGHTS_DIR / f"{RUN_ID}_W.npy"
    if not W_path.exists():
        print(f"No encontrado: {W_path}\nCambia RUN_ID al valor en runs_log.csv")
        exit(1)

    W = np.load(W_path)
    print(f"W: {W.shape}  rango=[{W.min():.4f}, {W.max():.4f}]")

    plot_bases(W, n_show=25, run_id=RUN_ID)
    plot_reconstructions(W, run_id=RUN_ID, n_show=8, eval_steps=300, restarts=3)
    plot_loss(run_id=RUN_ID)