import os, csv, time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from PIL import Image

IMG_DIR     = Path("img_align_celeba/img_align_celeba")
IMG_SIZE    = (64, 64)
M           = 64 * 64
DATA_FILE   = Path("celeba_64.npy")
LOG_CSV     = Path("runs_log.csv")
WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)
EPS         = 1e-10


# ── Dataset ───────────────────────────────────────────────
def preprocess(img_dir=IMG_DIR, out=DATA_FILE, n=50000):
    if out.exists() and np.load(out, mmap_mode="r").shape[0] > 0:
        return
    paths = sorted(img_dir.glob("*.jpg"))[:n]
    X = np.empty((len(paths), M), dtype=np.float32)
    for i, p in enumerate(paths):
        X[i] = np.asarray(Image.open(p).convert("L").resize(IMG_SIZE),
                          dtype=np.float32).flatten() / 255.0
        if i % 10000 == 0: print(f"  {i}/{len(paths)}")
    np.save(out, X)

def split_indices(n, train=0.70, val=0.20, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(n)
    nt  = int(train * n); nv = int(val * n)
    return idx[:nt], idx[nt:nt+nv], idx[nt+nv:]

def load_batch(X, idx):
    return np.ascontiguousarray(X[idx].T, dtype=np.float32)


# ── NMF core ──────────────────────────────────────────────
def proj(A):
    np.maximum(A, 0, out=A); return A

def loss(W, H, X, lW=0., lH=0.):
    R = W @ H - X
    return (0.5 * float(np.dot(R.ravel(), R.ravel())) +
            lW/2 * float(np.dot(W.ravel(), W.ravel())) +
            lH/2 * float(np.dot(H.ravel(), H.ravel())))

# Gradientes: ∇_W f = (WH-X)Hᵀ + λW   ∇_H f = Wᵀ(WH-X) + λH
def _clip(g, c=10.):
    n = np.linalg.norm(g); return g * (c/n) if n > c else g

def step_W(W, H, X, vW, a, b, method, lW=0.):
    R  = W @ H - X
    gW = _clip(R @ H.T + lW * W)
    if   method == "gd":        W  = proj(W - a * gW)
    elif method == "momentum":  vW = b*vW + gW;  W = proj(W - a*vW)
    elif method == "nesterov":
        Wl = proj(W - a*b*vW)
        gWl = _clip((Wl@H-X) @ H.T + lW*Wl)
        vW  = b*vW + gWl;  W = proj(W - a*vW)
    return W, vW

def step_H(W, H, X, vH, a, b, method, lH=0.):
    R  = W @ H - X
    gH = _clip(W.T @ R + lH * H)
    if   method == "gd":        H  = proj(H - a * gH)
    elif method == "momentum":  vH = b*vH + gH;  H = proj(H - a*vH)
    elif method == "nesterov":
        Hl  = proj(H - a*b*vH)
        gHl = _clip(W.T @ (W@Hl-X) + lH*Hl)
        vH  = b*vH + gHl;  H = proj(H - a*vH)
    return H, vH


# ── Algorithm 2: solve H dado W ───────────────────────────
def solve_H(W, Xb, a=1e-3, steps=200, lH=0., seed=0):
    np.random.seed(seed)
    k = W.shape[1]
    H = np.random.uniform(0, 1/np.sqrt(k), (k, Xb.shape[1])).astype(np.float32)
    for _ in range(steps):
        H = proj(H - a * _clip(W.T @ (W@H - Xb) + lH*H))
    return H

def solve_H_best(W, Xcol, a=1e-3, steps=500, restarts=5, lH=0.):
    best_H = None; best_e = np.inf
    for i in range(restarts):
        H = solve_H(W, Xcol, a, steps, lH, seed=i*17+3)
        e = float(np.sum((Xcol - W@H)**2))
        if e < best_e: best_e = e; best_H = H.copy()
    return best_H


# ── Evaluación ────────────────────────────────────────────
def evaluate(W, X, idx, bs, a=1e-3, steps=100, lH=0.):
    sq, tot = 0., 0
    for i in range(0, len(idx), bs):
        Xb = load_batch(X, idx[i:i+bs])
        H  = solve_H(W, Xb, a, steps, lH)
        R  = Xb - W@H
        sq += float(np.dot(R.ravel(), R.ravel())); tot += Xb.size
    return float(np.sqrt(sq/tot))


# ── Logging ───────────────────────────────────────────────
def log_run(params, results):
    row = {**params, **results, "ts": datetime.now().isoformat()}
    exists = LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists: w.writeheader()
        w.writerow(row)


# ── TRAIN — Algorithm 1 BCGD ──────────────────────────────
def train(k=100, steps=200, batch_size=1024, method="nesterov",
          alpha_W=1e-2, alpha_H=1e-2, beta=0.9,
          inner_W=1, inner_H=5, seed=42, eval_every=50,
          eval_steps=100, lambda_W=0., lambda_H=0.,
          init="scaled", patience=6, n_images=50000):

    np.random.seed(seed)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    preprocess(n=n_images)
    X_full = np.load(DATA_FILE, mmap_mode="r")
    tr, va, te = split_indices(len(X_full), seed=seed)
    print(f"Train={len(tr):,} Val={len(va):,} Test={len(te):,}  k={k}  method={method}")

    rng = np.random.default_rng(seed)
    W   = (rng.uniform(0, 1, (M,k)) if init=="uniform"
           else rng.uniform(0, 1/np.sqrt(k), (M,k))).astype(np.float32)
    vW  = np.zeros_like(W)

    loss_h = []; val_s = []; val_r = []
    best_rmse = np.inf; best_W = W.copy(); no_imp = 0
    aW = float(alpha_W); aH = float(alpha_H)
    t0 = time.time()

    for s in range(1, steps+1):
        idx  = np.random.choice(tr, batch_size, replace=False)
        Xb   = load_batch(X_full, idx)
        k_   = W.shape[1]
        H    = np.random.uniform(0, 1/np.sqrt(k_), (k_, batch_size)).astype(np.float32)
        vH   = np.zeros_like(H)

        for _ in range(inner_H):
            H, vH = step_H(W, H, Xb, vH, aH, beta, method, lambda_H)
        for _ in range(inner_W):
            W, vW = step_W(W, H, Xb, vW, aW, beta, method, lambda_W)

        if not np.isfinite(W).all():
            W = best_W.copy(); aW *= 0.5; aH *= 0.5; vW = np.zeros_like(W)
            print(f"  ⚠ NaN step {s} → lr={aW:.1e}")

        bl = loss(W, H, Xb, lambda_W, lambda_H) / batch_size
        loss_h.append(float(bl))

        if s % eval_every == 0 or s == 1:
            vrmse = evaluate(W, X_full, va[:1500], batch_size, aH, eval_steps, lambda_H)
            val_s.append(s); val_r.append(vrmse)
            print(f"Step {s:>4}/{steps}  loss={bl:.4f}  val_RMSE={vrmse:.4f}  t={time.time()-t0:.1f}s")
            if vrmse < best_rmse - 1e-5:
                best_rmse = vrmse; best_W = W.copy(); no_imp = 0
            else:
                no_imp += 1
            if no_imp >= patience:
                print(f"  Early stop  best_RMSE={best_rmse:.4f}"); W = best_W; break

    print("Test...")
    test_rmse = evaluate(W, X_full, te, batch_size, aH, eval_steps, lambda_H)
    print(f"Test RMSE: {test_rmse:.4f}")

    np.save(WEIGHTS_DIR/f"{run_id}_W.npy", W)
    np.save(WEIGHTS_DIR/f"{run_id}_loss.npy",      np.array(loss_h, dtype=np.float32))
    np.save(WEIGHTS_DIR/f"{run_id}_val_steps.npy", np.array(val_s,  dtype=np.int32))
    np.save(WEIGHTS_DIR/f"{run_id}_val_rmse.npy",  np.array(val_r,  dtype=np.float32))

    log_run(dict(run_id=run_id, k=k, steps=steps, batch_size=batch_size,
                 method=method, alpha_W=alpha_W, alpha_H=alpha_H, beta=beta,
                 inner_W=inner_W, inner_H=inner_H, seed=seed,
                 lambda_W=lambda_W, lambda_H=lambda_H, init=init),
            dict(final_loss=round(loss_h[-1],6), val_rmse=round(val_r[-1],6),
                 test_rmse=round(test_rmse,6), elapsed=round(time.time()-t0,1)))

    _plot_curves(loss_h, val_s, val_r, run_id, method, k, test_rmse)
    return W, run_id, loss_h, val_s, val_r


# ── Plots ─────────────────────────────────────────────────
def _plot_curves(loss_h, val_s, val_r, run_id, method, k, test_rmse):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    ax1.plot(loss_h, lw=1.5); ax1.set_title("Training Loss")
    ax1.set_xlabel("Step"); ax1.grid(True, alpha=0.3)
    ax2.plot(val_s, val_r, marker="o", lw=1.5); ax2.set_title("Validation RMSE")
    ax2.set_xlabel("Step"); ax2.grid(True, alpha=0.3)
    plt.suptitle(f"{method}  k={k}  Test RMSE={test_rmse:.4f}")
    plt.tight_layout()
    plt.savefig(f"results_{run_id}.png", dpi=100); plt.show()

def plot_bases(W, n_show=25, run_id=""):
    k    = W.shape[1]
    cols = int(np.ceil(np.sqrt(n_show)))
    rows = int(np.ceil(n_show/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.8, rows*1.8))
    fig.patch.set_facecolor("#12130F")
    fig.suptitle(f"Bases W (k={k})", color="white", fontsize=13, y=1.01)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < n_show:
            b = W[:,i].reshape(IMG_SIZE)
            b = (b-b.min())/(b.max()-b.min()+EPS)
            ax.imshow(b, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"b{i+1}", fontsize=7, color="#BCBAB3")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"bases_{run_id}.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor()); plt.show()

def plot_reconstructions(W, run_id="", n_show=8, eval_steps=300, restarts=3, seed=42):
    X_full = np.load(DATA_FILE, mmap_mode="r")
    _, _, te = split_indices(len(X_full), seed=seed)
    L  = np.linalg.norm(W.T @ W)
    aH = min(1e-2, 1.0/(L+EPS))

    fig, axes = plt.subplots(n_show, 3, figsize=(7, n_show*2.2))
    fig.patch.set_facecolor("#12130F")
    fig.suptitle(f"Reconstrucciones NMF — {run_id}", color="white", fontsize=13)
    for j, c in enumerate(["Original","Reconstruida","Diferencia"]):
        axes[0,j].set_title(c, color="#BCBAB3", fontsize=10)

    for i in range(n_show):
        x    = X_full[te[i]].astype(np.float32)
        Xc   = x.reshape(-1,1)
        H    = solve_H_best(W, Xc, aH, eval_steps, restarts)
        xh   = np.clip((W@H).flatten(), 0, 1)
        rmse = float(np.sqrt(np.mean((x-xh)**2)))
        diff = np.abs(x-xh).reshape(IMG_SIZE)
        axes[i,0].imshow(x.reshape(IMG_SIZE), cmap="gray", vmin=0, vmax=1)
        axes[i,1].imshow(xh.reshape(IMG_SIZE), cmap="gray", vmin=0, vmax=1)
        axes[i,2].imshow(diff, cmap="hot", vmin=0, vmax=max(float(diff.max()),0.01))
        axes[i,0].set_ylabel(f"RMSE={rmse:.4f}", color="#BCBAB3", fontsize=8)
        for j in range(3): axes[i,j].axis("off")
    plt.tight_layout()
    plt.savefig(f"recons_{run_id}.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor()); plt.show()

def plot_compare_methods(histories):
    COLORS = {"gd":"#2156B2","momentum":"#1A6B3C","nesterov":"#B83A1A"}
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(13,4))
    for m,(lh,vs,vr) in histories.items():
        c = COLORS.get(m,"gray")
        ax1.plot(range(1,len(lh)+1), lh, color=c, lw=1.8, label=m, alpha=0.9)
        ax2.plot(vs, vr, color=c, marker="o", markersize=4, lw=2, label=m)
    for ax,t,y in [(ax1,"Training Loss","Loss"),(ax2,"Validation RMSE","RMSE")]:
        ax.set_title(t); ax.set_xlabel("Step"); ax.set_ylabel(y)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle("GD vs Momentum vs Nesterov"); plt.tight_layout()
    plt.savefig("compare_methods.png", dpi=120); plt.show()

def plot_rmse_vs_k(results):
    ks  = sorted(results); vrs = [results[k][0] for k in ks]; trs = [results[k][1] for k in ks]
    bk  = ks[int(np.argmin(vrs))]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(ks, vrs, color="#2156B2", marker="o", markersize=7, lw=2.5, label="Val RMSE")
    ax.plot(ks, trs, color="#B83A1A", marker="s", markersize=7, lw=2.5, ls="--", label="Test RMSE")
    ax.axvline(bk, color="#1A6B3C", ls=":", lw=1.5, label=f"mejor k={bk}")
    for k,v in zip(ks,vrs):
        ax.annotate(f"{v:.4f}", (k,v), textcoords="offset points", xytext=(0,10), ha="center", fontsize=9)
    ax.set_xlabel("Rango k",fontsize=12); ax.set_ylabel("RMSE",fontsize=12)
    ax.set_title("Val & Test RMSE vs k",fontsize=13)
    ax.legend(); ax.set_xticks(ks); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("rmse_vs_k.png", dpi=120); plt.show()


# ── Experimentos del proyecto ──────────────────────────────
def compare_methods(k=100, steps=150, **kwargs):
    histories = {}
    for m in ["gd","momentum","nesterov"]:
        print(f"\n── {m}")
        _, rid, lh, vs, vr = train(k=k, steps=steps, method=m, **kwargs)
        histories[m] = (lh, vs, vr)
    plot_compare_methods(histories)
    return histories

def rank_sweep(k_values=[25,50,100,150,200], steps=150, **kwargs):
    results = {}

    for k in k_values:
        print(f"\n── k={k}")
        params = kwargs.copy()
        params["eval_every"] = steps
        W, _, _, _, vr = train(k=k, steps=steps, **params)
        X_full = np.load(DATA_FILE, mmap_mode="r")
        _, _, te = split_indices(len(X_full), seed=params.get("seed", 42))

        bs = params.get("batch_size", 1024)
        aH = params.get("alpha_H", 1e-2)
        test_rmse = evaluate(W, X_full, te, bs, aH, 100)
        val_rmse = vr[-1] if len(vr) > 0 else float("inf")
        results[k] = (val_rmse, test_rmse)
        print(f"k={k} → Val RMSE={val_rmse:.4f} | Test RMSE={test_rmse:.4f}")
    plot_rmse_vs_k(results)

    return results


# ── Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    PARAMS = dict(
        batch_size = 1024,
        alpha_W    = 1e-2,
        alpha_H    = 1e-2,
        beta       = 0.9,
        inner_W    = 1,
        inner_H    = 5,
        seed       = 42,
        eval_every = 50,
        eval_steps = 100,
        patience   = 6,
        n_images   = 200000,
    )

    # 1. Entrenamiento principal
    W, run_id, *_ = train(k=100, steps=200, method="nesterov", **PARAMS)

    # 2. Bases aprendidas
    plot_bases(W, n_show=25, run_id=run_id)

    # 3. Reconstrucciones
    plot_reconstructions(W, run_id=run_id, n_show=8, eval_steps=300, restarts=3)

    # 4. Comparación GD / Momentum / Nesterov
    compare_methods(k=100, steps=150, **PARAMS)

    # 5. Rank sweep RMSE vs k
    rank_sweep(k_values=[25,50,100,150,200], steps=150, **PARAMS)