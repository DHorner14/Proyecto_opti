import os
import csv
import time
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
IMG_DIR   = Path("img_align_celeba/img_align_celeba")  # ruta a las imágenes
IMG_SIZE  = (64, 64)
M         = IMG_SIZE[0] * IMG_SIZE[1]   # 4096 píxeles por imagen
LOG_CSV   = Path("runs_log.csv")
WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 1. CARGA DE DATOS EN BATCHES
# ─────────────────────────────────────────────
def load_batch(paths: list[Path]) -> np.ndarray:

    cols = []
    for p in paths:
        img = Image.open(p).convert("L").resize(IMG_SIZE)
        cols.append(np.array(img, dtype=np.float32).flatten() / 255.0)
    return np.column_stack(cols)   # shape: (M, batch_size)


def split_paths(img_dir: Path, train=0.70, val=0.20, seed=42):
    """Divide los paths de imágenes en train/val/test."""
    paths = sorted(img_dir.glob("*.jpg"))[:500]
    np.random.seed(seed)
    idx   = np.random.permutation(len(paths))
    n     = len(paths)
    n_tr  = int(train * n)
    n_val = int(val   * n)
    return (
        [paths[i] for i in idx[:n_tr]],
        [paths[i] for i in idx[n_tr:n_tr+n_val]],
        [paths[i] for i in idx[n_tr+n_val:]]
    )


# ─────────────────────────────────────────────
# 2. PROYECCIÓN NMF
# ─────────────────────────────────────────────
def proj(A: np.ndarray) -> np.ndarray:
    """Proyecta sobre el ortante no-negativo."""
    return np.maximum(A, 0)


# ─────────────────────────────────────────────
# 3. GRADIENTES
# ─────────────────────────────────────────────
def gradients(W: np.ndarray, H: np.ndarray, X: np.ndarray):
    """
    Calcula gradientes de f = ½‖X - WH‖²_F
    ∇_W f = (WH - X) H^T
    ∇_H f = W^T (WH - X)
    """
    R = W @ H - X          # residual (M, n)
    gW = R @ H.T           # (M, k)
    gH = W.T @ R           # (k, n)
    return gW, gH


def loss(W: np.ndarray, H: np.ndarray, X: np.ndarray) -> float:
    R = W @ H - X
    return 0.5 * np.sum(R ** 2)


# ─────────────────────────────────────────────
# 4. PASO DE OPTIMIZACIÓN (GD / Momentum / NAG)
# ─────────────────────────────────────────────
def update_W(W, H, X, vW, alpha, beta, method):
    if method == "gd":
        gW, _ = gradients(W, H, X)
        W = proj(W - alpha * gW)
        return W, vW

    elif method == "momentum":
        gW, _ = gradients(W, H, X)
        vW = beta * vW + gW
        W  = proj(W - alpha * vW)
        return W, vW

    elif method == "nesterov":
        W_look = W - alpha * beta * vW
        W_look = proj(W_look)
        gW, _  = gradients(W_look, H, X)
        vW = beta * vW + gW
        W  = proj(W - alpha * vW)
        return W, vW

    raise ValueError(f"Método desconocido: {method}")


def update_H(W, H, X, vH, alpha, beta, method):
    if method == "gd":
        _, gH = gradients(W, H, X)
        H = proj(H - alpha * gH)
        return H, vH

    elif method == "momentum":
        _, gH = gradients(W, H, X)
        vH = beta * vH + gH
        H  = proj(H - alpha * vH)
        return H, vH

    elif method == "nesterov":
        H_look = H - alpha * beta * vH
        H_look = proj(H_look)
        _, gH  = gradients(W, H_look, X)
        vH = beta * vH + gH
        H  = proj(H - alpha * vH)
        return H, vH

    raise ValueError(f"Método desconocido: {method}")


# ─────────────────────────────────────────────
# 5. EVALUACIÓN EN VAL/TEST
# ─────────────────────────────────────────────
def evaluate(W, val_paths, batch_size, alpha_H, inner_H, seed=0):
    """
    Fija W, resuelve H_val por GD no-negativo, calcula RMSE.
    """
    np.random.seed(seed)
    sq_err = 0.0
    total  = 0

    for i in range(0, len(val_paths), batch_size):
        batch = val_paths[i:i+batch_size]
        X_b   = load_batch(batch)            # (M, b)
        k     = W.shape[1]
        H_b   = np.random.uniform(0, 1/np.sqrt(k), (k, X_b.shape[1])).astype(np.float32)
        vH    = np.zeros_like(H_b)

        for _ in range(inner_H):
            H_b, vH = update_H(W, H_b, X_b, vH, alpha_H, beta=0.0, method="gd")

        sq_err += np.sum((X_b - W @ H_b) ** 2)
        total  += X_b.size

    return float(np.sqrt(sq_err / total))


# ─────────────────────────────────────────────
# 6. LOGGING
# ─────────────────────────────────────────────
def log_run(params: dict, results: dict):
    """Guarda los parámetros y resultados en runs_log.csv."""
    row = {**params, **results, "timestamp": datetime.now().isoformat()}
    file_exists = LOG_CSV.exists()

    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nRun guardado en {LOG_CSV}")


def save_weights(W, H, run_id: str):
    """Guarda W y H como archivos .npy."""
    np.save(WEIGHTS_DIR / f"{run_id}_W.npy", W)
    np.save(WEIGHTS_DIR / f"{run_id}_H.npy", H)
    print(f"Pesos guardados en {WEIGHTS_DIR}/{run_id}_W.npy y _H.npy")


# ─────────────────────────────────────────────
# 7. ENTRENAMIENTO PRINCIPAL (BCGD + BATCHES)
# ─────────────────────────────────────────────
def train(
    k          = 50,
    steps      = 200,
    batch_size = 20,
    alpha_W    = 1e-3,
    alpha_H    = 1e-3,
    beta       = 0.9,
    method     = "gd",   # "gd" | "momentum" | "nesterov"
    inner_W    = 1,
    inner_H    = 1,
    seed       = 42,
    eval_every = 10,           # evaluar en val cada N steps
):
    np.random.seed(seed)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Cargar splits ──────────────────────────
    print("Cargando splits...")
    train_paths, val_paths, test_paths = split_paths(IMG_DIR)
    print(f"  Train: {len(train_paths):,}  Val: {len(val_paths):,}  Test: {len(test_paths):,}")

    # ── Inicialización de W y H ────────────────
    # W se inicializa una vez global; H se reinicia por batch
    W  = np.random.uniform(0, 1/np.sqrt(k), (M, k)).astype(np.float32)
    vW = np.zeros_like(W)

    loss_history = []
    val_rmse_history = []
    t0 = time.time()

    # ── Loop de entrenamiento ──────────────────
    for step in range(1, steps + 1):

        # Muestrear un batch aleatorio del train set
        batch_idx   = np.random.choice(len(train_paths), size=batch_size, replace=False)
        batch_paths = [train_paths[i] for i in batch_idx]
        X_batch     = load_batch(batch_paths)   # (M, batch_size)

        # Inicializar H para este batch
        H_batch = np.random.uniform(0, 1/np.sqrt(k), (k, batch_size)).astype(np.float32)
        vH      = np.zeros_like(H_batch)

        # ── Actualizar H (bloque H) ────────────
        for _ in range(inner_H):
            H_batch, vH = update_H(W, H_batch, X_batch, vH, alpha_H, beta, method)

        # ── Actualizar W (bloque W) ────────────
        for _ in range(inner_W):
            W, vW = update_W(W, H_batch, X_batch, vW, alpha_W, beta, method)

        # ── Loss del batch ─────────────────────
        batch_loss = loss(W, H_batch, X_batch) / batch_size
        loss_history.append(batch_loss)

        # ── Evaluación en val ──────────────────
        if step % eval_every == 0 or step == 1:
            val_rmse = evaluate(W, val_paths[:500], batch_size, alpha_H, inner_H=50)
            val_rmse_history.append((step, val_rmse))
            elapsed = time.time() - t0
            print(f"Step {step:>4}/{steps}  loss={batch_loss:.4f}  val_RMSE={val_rmse:.4f}  t={elapsed:.1f}s")

    # ── Evaluación final en test ───────────────
    print("\nEvaluando en test set...")
    test_rmse = evaluate(W, test_paths[:500], batch_size, alpha_H, inner_H=50)
    print(f"Test RMSE: {test_rmse:.4f}")

    # ── Guardar pesos ──────────────────────────
    # H final = último batch (representativo)
    save_weights(W, H_batch, run_id)

    # ── Logging ───────────────────────────────
    params = dict(
        run_id=run_id, k=k, steps=steps, batch_size=batch_size,
        alpha_W=alpha_W, alpha_H=alpha_H, beta=beta, method=method,
        inner_W=inner_W, inner_H=inner_H, seed=seed,
        img_size=f"{IMG_SIZE[0]}x{IMG_SIZE[1]}"
    )
    results = dict(
        final_loss=round(loss_history[-1], 6),
        val_rmse_final=round(val_rmse_history[-1][1], 6),
        test_rmse=round(test_rmse, 6),
        train_images=len(train_paths),
        val_images=len(val_paths),
        test_images=len(test_paths),
        elapsed_s=round(time.time() - t0, 1)
    )
    log_run(params, results)

    # Plot results
    plot_results(loss_history, val_rmse_history, run_id)

    return W, H_batch, loss_history, val_rmse_history


# ─────────────────────────────────────────────
# 7. GRAPHING
# ─────────────────────────────────────────────
def plot_results(loss_history, val_rmse_history, run_id):
    """Plot training loss and validation RMSE."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(loss_history, linewidth=1.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    
    # Plot validation RMSE
    val_steps, val_rmses = zip(*val_rmse_history)
    ax2.plot(val_steps, val_rmses, linewidth=1.5, marker='o', markersize=4)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Validation RMSE")
    ax2.set_title("Validation RMSE")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"results_{run_id}.png", dpi=100)
    plt.show()


# ─────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    W, H, loss_hist, val_hist = train(
        k          = 50,
        steps      = 200,
        batch_size = 20,
        alpha_W    = 1e-3,
        alpha_H    = 1e-3,
        beta       = 0.9,
        method     = "gd",
        inner_W    = 1,
        inner_H    = 1,
        seed       = 42,
        eval_every = 10,
    )