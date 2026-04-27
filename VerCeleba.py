import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
IMG_SIZE    = (64, 64)
M           = IMG_SIZE[0] * IMG_SIZE[1]
WEIGHTS_DIR = Path("weights")
DATA_FILE   = "celeba_64.npy"

# Cambia este run_id al que aparece en runs_log.csv o en weights/
RUN_ID = "20260427_160127"


# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────
def vec_to_img(v):
    return np.clip(v.reshape(IMG_SIZE), 0, 1)


# ─────────────────────────────────────────────
# 1. BASES W
# ─────────────────────────────────────────────
def plot_bases(W, n_show=25, run_id=""):
    k    = W.shape[1]
    cols = int(np.ceil(np.sqrt(n_show)))
    rows = int(np.ceil(n_show / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    fig.patch.set_facecolor("#12130F")
    fig.suptitle(f"Bases W  (k={k})  —  {run_id}", color="white", fontsize=13, y=1.01)

    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < n_show:
            base = vec_to_img(W[:, i])
            base = (base - base.min()) / (base.max() - base.min() + 1e-8)
            ax.imshow(base, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"b{i+1}", fontsize=7, color="#BCBAB3")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"bases_{run_id}.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"Guardado: bases_{run_id}.png")


# ─────────────────────────────────────────────
# 2. RECONSTRUCCIONES
# ─────────────────────────────────────────────
def reconstruct_image(W, x_vec, alpha_H=1e-3, steps=300):
    k = W.shape[1]
    h = np.random.uniform(0, 1/np.sqrt(k), (k, 1)).astype(np.float32)
    for _ in range(steps):
        R  = W @ h - x_vec.reshape(-1, 1)
        gH = W.T @ R
        h  = np.maximum(h - alpha_H * gH, 0)
    return (W @ h).flatten()


def plot_reconstructions(W, X_full, indices, n_show=8, alpha_H=1e-3, run_id=""):
    n_show = min(n_show, len(indices))
    fig, axes = plt.subplots(n_show, 3, figsize=(7, n_show * 2.2))
    fig.patch.set_facecolor("#12130F")
    fig.suptitle(f"Reconstrucciones NMF  —  {run_id}", color="white", fontsize=13, y=1.005)

    for j, col in enumerate(["Original", "Reconstruida", "Diferencia"]):
        axes[0, j].set_title(col, color="#BCBAB3", fontsize=10)

    for i in range(n_show):
        x_vec = X_full[indices[i]].astype(np.float32)
        x_hat = reconstruct_image(W, x_vec, alpha_H)
        diff  = np.abs(x_vec - x_hat).reshape(IMG_SIZE)
        rmse_i = np.sqrt(np.mean((x_vec - x_hat) ** 2))

        axes[i, 0].imshow(vec_to_img(x_vec), cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow(vec_to_img(x_hat), cmap="gray", vmin=0, vmax=1)
        axes[i, 2].imshow(diff,              cmap="hot",  vmin=0, vmax=0.3)
        axes[i, 0].set_ylabel(f"RMSE={rmse_i:.3f}", color="#BCBAB3", fontsize=8)
        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(f"recons_{run_id}.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"Guardado: recons_{run_id}.png")


# ─────────────────────────────────────────────
# 3. CURVAS DE PÉRDIDA
# ─────────────────────────────────────────────
def plot_loss(loss_history, val_steps, val_rmses, run_id=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#F4F3EE")
    BLUE, RED, MUT = "#2156B2", "#B83A1A", "#7A7870"

    ax1.plot(range(1, len(loss_history)+1), loss_history,
             color=BLUE, linewidth=1.5, alpha=0.9)
    ax1.set_xlabel("Step", color=MUT)
    ax1.set_ylabel("Loss (batch)", color=MUT)
    ax1.set_title("Pérdida de entrenamiento", color="#12130F")
    ax1.set_facecolor("white")
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_steps, val_rmses, color=RED, marker="o", markersize=5, linewidth=2)
    ax2.set_xlabel("Step", color=MUT)
    ax2.set_ylabel("RMSE val", color=MUT)
    ax2.set_title("RMSE en validación", color="#12130F")
    ax2.set_facecolor("white")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Curvas de entrenamiento  —  {run_id}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"loss_{run_id}.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"Guardado: loss_{run_id}.png")


# ─────────────────────────────────────────────
# 4. COMPARAR MÉTODOS
# ─────────────────────────────────────────────
def plot_compare_methods(histories: dict):
    """
    histories = {
        "gd":       (loss_list, val_steps, val_rmses),
        "momentum": (loss_list, val_steps, val_rmses),
        "nesterov": (loss_list, val_steps, val_rmses),
    }
    """
    colors = {"gd": "#2156B2", "momentum": "#1A6B3C", "nesterov": "#B83A1A"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor("#F4F3EE")
    MUT = "#7A7870"

    for method, (loss_hist, val_steps, val_rmses) in histories.items():
        col = colors.get(method, "gray")
        ax1.plot(range(1, len(loss_hist)+1), loss_hist,
                 color=col, linewidth=1.8, label=method, alpha=0.9)
        ax2.plot(val_steps, val_rmses, color=col, marker="o",
                 markersize=4, linewidth=2, label=method)

    for ax, title, ylabel in [
        (ax1, "Pérdida de entrenamiento", "Loss"),
        (ax2, "RMSE en validación",       "RMSE val"),
    ]:
        ax.set_title(title, color="#12130F")
        ax.set_xlabel("Step", color=MUT)
        ax.set_ylabel(ylabel, color=MUT)
        ax.legend(fontsize=10)
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Comparación: GD vs Momentum vs Nesterov", fontsize=13)
    plt.tight_layout()
    plt.savefig("compare_methods.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("Guardado: compare_methods.png")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Cargar W
    W_path = WEIGHTS_DIR / f"{RUN_ID}_W.npy"
    if not W_path.exists():
        print(f"No se encontró {W_path}. Verifica el RUN_ID.")
        exit(1)

    W = np.load(W_path)
    print(f"W cargado: {W.shape}  (m={W.shape[0]}, k={W.shape[1]})")

    # 2. Bases aprendidas
    plot_bases(W, n_show=25, run_id=RUN_ID)

    # 3. Reconstrucciones en test
    print("Cargando dataset para reconstrucciones...")
    X_full = np.load(DATA_FILE, mmap_mode="r")

    from NFMceleba import split_indices
    _, _, test_idx = split_indices(len(X_full), seed=42)
    plot_reconstructions(W, X_full, test_idx[:8], run_id=RUN_ID)

    # 4. Curvas de pérdida
    loss_hist  = np.load(WEIGHTS_DIR / f"{RUN_ID}_loss.npy").tolist()
    val_steps  = np.load(WEIGHTS_DIR / f"{RUN_ID}_val_steps.npy").tolist()
    val_rmses  = np.load(WEIGHTS_DIR / f"{RUN_ID}_val_rmse.npy").tolist()
    plot_loss(loss_hist, val_steps, val_rmses, run_id=RUN_ID)