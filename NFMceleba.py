import os
import csv
import time
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
IMG_DIR     = Path("img_align_celeba/img_align_celeba")
IMG_SIZE    = (64, 64)
M           = IMG_SIZE[0] * IMG_SIZE[1]
DATA_FILE   = "celeba_64.npy"
LOG_CSV     = Path("runs_log.csv")
WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# 1. PREPROCESAMIENTO (SOLO UNA VEZ)
# ─────────────────────────────────────────────
def preprocess_dataset(img_dir, output_file=DATA_FILE, max_images=200000):
    if os.path.exists(output_file):
        print(f"Dataset ya existe: {output_file}")
        return

    paths = sorted(img_dir.glob("*.jpg"))[:max_images]
    print(f"Preprocesando {len(paths)} imágenes...")
    X = np.empty((len(paths), M), dtype=np.float32)

    for i, p in enumerate(paths):
        img = Image.open(p).convert("L").resize(IMG_SIZE)
        X[i] = np.asarray(img, dtype=np.float32).flatten() / 255.0
        if i % 5000 == 0:
            print(f"  {i}/{len(paths)}")

    np.save(output_file, X)
    print(f"Dataset guardado en {output_file}  shape={X.shape}")


# ─────────────────────────────────────────────
# 2. SPLITS SOBRE ÍNDICES
# ─────────────────────────────────────────────
def split_indices(n_samples, train=0.70, val=0.20, seed=42):
    np.random.seed(seed)
    idx     = np.random.permutation(n_samples)
    n_train = int(train * n_samples)
    n_val   = int(val   * n_samples)
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]


# ─────────────────────────────────────────────
# 3. CARGA DE BATCH RÁPIDO
# ─────────────────────────────────────────────
def load_batch(X_full, indices):
    return X_full[indices].T.astype(np.float32)   # (M, batch_size)


# ─────────────────────────────────────────────
# 4. UTILIDADES NMF
# ─────────────────────────────────────────────
def proj(A):
    return np.maximum(A, 0)

def gradients(W, H, X):
    R = (W @ H - X).astype(np.float32)
    return R @ H.T, W.T @ R

def loss(W, H, X):
    R = (W @ H - X).astype(np.float32)
    return 0.5 * np.sum(R ** 2)


# ─────────────────────────────────────────────
# 5. UPDATES
# ─────────────────────────────────────────────
def update_W(W, H, X, vW, alpha, beta, method):
    if method == "gd":
        gW, _ = gradients(W, H, X)
        return proj(W - alpha * gW), vW
    elif method == "momentum":
        gW, _ = gradients(W, H, X)
        vW = beta * vW + gW
        return proj(W - alpha * vW), vW
    elif method == "nesterov":
        W_look = proj(W - alpha * beta * vW)
        gW, _  = gradients(W_look, H, X)
        vW = beta * vW + gW
        return proj(W - alpha * vW), vW
    raise ValueError("Método inválido")

def update_H(W, H, X, vH, alpha, beta, method):
    if method == "gd":
        _, gH = gradients(W, H, X)
        return proj(H - alpha * gH), vH
    elif method == "momentum":
        _, gH = gradients(W, H, X)
        vH = beta * vH + gH
        return proj(H - alpha * vH), vH
    elif method == "nesterov":
        H_look = proj(H - alpha * beta * vH)
        _, gH  = gradients(W, H_look, X)
        vH = beta * vH + gH
        return proj(H - alpha * vH), vH
    raise ValueError("Método inválido")


# ─────────────────────────────────────────────
# 6. EVALUACIÓN
# ─────────────────────────────────────────────
def evaluate(W, X_full, val_idx, batch_size, alpha_H, inner_H=5, seed=0):
    np.random.seed(seed)
    sq_err, total = 0.0, 0

    for i in range(0, len(val_idx), batch_size):
        idx_batch = val_idx[i:i+batch_size]
        X_b = load_batch(X_full, idx_batch)
        k   = W.shape[1]
        H_b = np.random.uniform(0, 1/np.sqrt(k), (k, X_b.shape[1])).astype(np.float32)
        vH  = np.zeros_like(H_b)

        for _ in range(inner_H):
            H_b, vH = update_H(W, H_b, X_b, vH, alpha_H, beta=0.0, method="gd")

        sq_err += np.sum((X_b - W @ H_b) ** 2)
        total  += X_b.size

    return float(np.sqrt(sq_err / total))


# ─────────────────────────────────────────────
# 7. LOGGING Y GUARDADO
# ─────────────────────────────────────────────
def log_run(params: dict, results: dict):
    row = {**params, **results, "timestamp": datetime.now().isoformat()}
    file_exists = LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Run guardado en {LOG_CSV}")

def save_weights(W, H, run_id: str):
    np.save(WEIGHTS_DIR / f"{run_id}_W.npy", W)
    np.save(WEIGHTS_DIR / f"{run_id}_H.npy", H)
    print(f"Pesos guardados en weights/{run_id}_W.npy y _H.npy")


# ─────────────────────────────────────────────
# 8. TRAIN
# ─────────────────────────────────────────────
def train(
    k          = 100,
    steps      = 300,
    batch_size = 1024,
    alpha_W    = 1e-3,
    alpha_H    = 1e-3,
    beta       = 0.9,
    method     = "nesterov",
    inner_W    = 1,
    inner_H    = 1,
    seed       = 42,
    eval_every = 50,
):
    np.random.seed(seed)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    preprocess_dataset(IMG_DIR)
    print("Cargando dataset...")
    X_full = np.load(DATA_FILE, mmap_mode="r")

    train_idx, val_idx, test_idx = split_indices(len(X_full), seed=seed)
    print(f"  Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

    W  = np.random.uniform(0, 1/np.sqrt(k), (M, k)).astype(np.float32)
    vW = np.zeros_like(W)

    loss_history     = []
    val_steps_list   = []
    val_rmse_list    = []
    t0 = time.time()

    for step in range(1, steps + 1):
        batch_idx = np.random.choice(train_idx, size=batch_size, replace=False)
        X_batch   = load_batch(X_full, batch_idx)

        H_batch = np.random.uniform(0, 1/np.sqrt(k), (k, batch_size)).astype(np.float32)
        vH      = np.zeros_like(H_batch)

        for _ in range(inner_H):
            H_batch, vH = update_H(W, H_batch, X_batch, vH, alpha_H, beta, method)
        for _ in range(inner_W):
            W, vW = update_W(W, H_batch, X_batch, vW, alpha_W, beta, method)

        batch_loss = loss(W, H_batch, X_batch) / batch_size
        loss_history.append(float(batch_loss))

        if step % eval_every == 0 or step == 1:
            val_rmse = evaluate(W, X_full, val_idx, batch_size, alpha_H, inner_H=5)
            val_steps_list.append(step)
            val_rmse_list.append(val_rmse)
            elapsed = time.time() - t0
            print(f"Step {step:>4}/{steps}  loss={batch_loss:.4f}  val_RMSE={val_rmse:.4f}  t={elapsed:.1f}s")

    print("Evaluando en test set...")
    test_rmse = evaluate(W, X_full, test_idx, batch_size, alpha_H, inner_H=5)
    print(f"Test RMSE: {test_rmse:.4f}")

    # ── Guardar pesos y curvas ─────────────────
    # val se guarda en 2 arrays separados para evitar dtype=object
    save_weights(W, H_batch, run_id)
    np.save(WEIGHTS_DIR / f"{run_id}_loss.npy",      np.array(loss_history,   dtype=np.float32))
    np.save(WEIGHTS_DIR / f"{run_id}_val_steps.npy", np.array(val_steps_list, dtype=np.int32))
    np.save(WEIGHTS_DIR / f"{run_id}_val_rmse.npy",  np.array(val_rmse_list,  dtype=np.float32))

    # ── Logging ───────────────────────────────
    params = dict(
        run_id=run_id, k=k, steps=steps, batch_size=batch_size,
        alpha_W=alpha_W, alpha_H=alpha_H, beta=beta, method=method,
        inner_W=inner_W, inner_H=inner_H, seed=seed,
        img_size=f"{IMG_SIZE[0]}x{IMG_SIZE[1]}"
    )
    results = dict(
        final_loss=round(loss_history[-1], 6),
        val_rmse_final=round(val_rmse_list[-1], 6),
        test_rmse=round(test_rmse, 6),
        train_images=len(train_idx),
        val_images=len(val_idx),
        test_images=len(test_idx),
        elapsed_s=round(time.time() - t0, 1)
    )
    log_run(params, results)
    plot_results(loss_history, val_steps_list, val_rmse_list, run_id)

    return W


# ─────────────────────────────────────────────
# 9. GRÁFICAS
# ─────────────────────────────────────────────
def plot_results(loss_history, val_steps, val_rmses, run_id):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(loss_history, linewidth=1.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    if val_steps:
        ax2.plot(val_steps, val_rmses, linewidth=1.5, marker="o", markersize=4)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Validation RMSE")
    ax2.set_title("Validation RMSE")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results_{run_id}.png", dpi=100)
    plt.show()
    print(f"Gráfica guardada: results_{run_id}.png")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train(
        k          = 100,
        steps      = 300,
        batch_size = 1024,
        alpha_W    = 1e-3,
        alpha_H    = 1e-3,
        beta       = 0.9,
        method     = "nesterov",
        inner_W    = 1,
        inner_H    = 1,
        seed       = 42,
        eval_every = 50,
    )