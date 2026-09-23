"""
Deepfake Detector using Convolutional Neural Networks
=====================================================
Pipeline:
  1. Train DeepFakeNet on synthetic data (fast, controlled, reproducible)
  2. Evaluate on synthetic test set  → baseline metrics
  3. Evaluate on Celeb-DF v2 sample  → real-world generalisation metrics
  4. Save all plots and a side-by-side metrics summary

Celeb-DF v2 path (update if needed):
  CELEBDF_DIR = r"D:\Som\College\Sem-6\Research Methodology\Datasets\Celebdf"
"""

import os
import time
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG — update both paths to match your machine
# ──────────────────────────────────────────────
CELEBDF_DIR = r"D:\Som\College\Sem-6\Research Methodology\Datasets\Celebdf"
OUT_DIR     = r"D:\Som\College\Sem-6\Research Methodology\results"

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


# ══════════════════════════════════════════════
# PART 1 — SYNTHETIC DATASET  (vectorised)
# ══════════════════════════════════════════════
class SyntheticFaceDataset(Dataset):
    """
    Generates synthetic real/fake face images entirely in NumPy (no pixel loops).
    Real  -> smooth skin-tone gradient + subtle noise.
    Fake  -> same base + three GAN-like artefacts:
               checkerboard grid  (upsampling fingerprint)
               horizontal seam    (blending boundary)
               colour patch       (local illumination mismatch)
    """

    def __init__(self, n_samples=800, img_size=128, split="train", transform=None):
        self.n_samples = n_samples
        self.img_size  = img_size
        self.transform = transform
        self.images, self.labels = self._generate(n_samples, img_size, split)

    def _make_real_batch(self, size, rng, n):
        cx = rng.uniform(0.3, 0.7, n) * size
        cy = rng.uniform(0.3, 0.7, n) * size
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
        d = np.sqrt(
            (ys[None] - cy[:, None, None]) ** 2 +
            (xs[None] - cx[:, None, None]) ** 2
        ) / (size * 0.5)
        R = rng.uniform(0.6, 0.9, n)[:, None, None].astype(np.float32) - d * 0.3
        G = rng.uniform(0.4, 0.7, n)[:, None, None].astype(np.float32) - d * 0.2
        B = rng.uniform(0.3, 0.5, n)[:, None, None].astype(np.float32) - d * 0.2
        arr = np.stack([R, G, B], axis=-1)
        arr += rng.normal(0.0, 0.02, arr.shape).astype(np.float32)
        return np.clip(arr, 0.0, 1.0)

    def _add_fake_artefacts(self, arr, size, rng):
        n = arr.shape[0]
        # 1. Checkerboard
        grid, tile = 8, 16
        rows = np.arange(size); cols = np.arange(size)
        mask = ((rows[:, None] % tile) < grid) & ((cols[None, :] % tile) < grid)
        offsets = rng.uniform(-0.05, 0.05, (n, 3)).astype(np.float32)
        arr += mask[None, :, :, None] * offsets[:, None, None, :]
        # 2. Horizontal blending seam
        seams    = rng.randint(size // 3, 2 * size // 3, n)
        row_grid = np.arange(size)[None, :]
        seam_mask = (row_grid >= (seams[:, None] - 2)) & \
                    (row_grid <  (seams[:, None] + 2))
        arr[:, :, :, 0] += seam_mask[:, :, None] * 0.08
        arr[:, :, :, 2] -= seam_mask[:, :, None] * 0.06
        # 3. Colour-inconsistency patch
        px      = rng.randint(10, size - 30, n)
        py      = rng.randint(10, size - 30, n)
        scale   = rng.uniform(1.05, 1.2, n).astype(np.float32)
        r_idx   = np.arange(20)[None, :, None] + py[:, None, None]
        c_idx   = np.arange(20)[None, None, :] + px[:, None, None]
        img_idx = np.arange(n)[:, None, None]
        arr[img_idx, r_idx, c_idx] *= scale[:, None, None, None]
        return np.clip(arr, 0.0, 1.0)

    def _generate(self, n, size, split):
        rng    = np.random.RandomState(SEED + (0 if split == "train" else 99))
        n_real = n // 2 + n % 2
        n_fake = n // 2
        print(f"[INFO] Generating synthetic {split} set: {n_real} real + {n_fake} fake ...",
              flush=True)
        real_arr = self._make_real_batch(size, rng, n_real)
        fake_arr = self._make_real_batch(size, rng, n_fake)
        fake_arr = self._add_fake_artefacts(fake_arr, size, rng)
        images, labels, ri, fi = [], [], 0, 0
        for idx in range(n):
            if idx % 2 == 0:
                arr = real_arr[ri]; ri += 1; label = 0
            else:
                arr = fake_arr[fi]; fi += 1; label = 1
            images.append(Image.fromarray((arr * 255).astype(np.uint8)))
            labels.append(label)
        return images, labels

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ══════════════════════════════════════════════
# PART 2 — CELEB-DF v2 DATASET LOADER
# ══════════════════════════════════════════════
class CelebDFDataset(Dataset):
    """
    Loads Celeb-DF v2 images from flat folder structure:
        <root>/Real/  — real face images
        <root>/Fake/  — deepfake face images

    Automatically splits into val (25%) and test (75%) using SEED.
    Label: 0 = Real,  1 = Fake
    """

    def __init__(self, root, split="test", val_ratio=0.25, transform=None):
        self.transform = transform
        real_dir = os.path.join(root, "Real")
        fake_dir = os.path.join(root, "Fake")

        real_paths = sorted([
            os.path.join(real_dir, f) for f in os.listdir(real_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        fake_paths = sorted([
            os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        rng = random.Random(SEED)
        rng.shuffle(real_paths)
        rng.shuffle(fake_paths)

        split_idx = int(len(real_paths) * val_ratio)

        if split == "val":
            r = real_paths[:split_idx]
            f = fake_paths[:split_idx]
        else:
            r = real_paths[split_idx:]
            f = fake_paths[split_idx:]

        self.paths  = r + f
        self.labels = [0] * len(r) + [1] * len(f)
        print(f"[INFO] Celeb-DF v2 {split}: {len(r)} real + {len(f)} fake "
              f"= {len(self.paths)} images", flush=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ══════════════════════════════════════════════
# PART 3 — CNN ARCHITECTURE
# ══════════════════════════════════════════════
class DeepFakeNet(nn.Module):
    """
    Four-block CNN for binary deepfake detection.
    Block 1 (32 ch)  - low-level edges and colour artefacts
    Block 2 (64 ch)  - texture and grid patterns
    Block 3 (128 ch) - blending seams
    Block 4 (256 ch) - semantic inconsistencies
    FC head          - 4096 -> 512 -> 128 -> 2
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 128),          nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════
# PART 4 — TRAINING AND EVALUATION
# ══════════════════════════════════════════════
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


# ══════════════════════════════════════════════
# PART 5 — VISUALISATION
# ══════════════════════════════════════════════
def save_training_curves(history, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DeepFakeNet - Training Curves (Synthetic Data)",
                 fontsize=14, fontweight="bold")
    axes[0].plot(epochs, history["train_loss"], "b-o", ms=4, label="Train")
    axes[0].plot(epochs, history["val_loss"],   "r-s", ms=4, label="Val")
    axes[0].set_title("Cross-Entropy Loss"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["train_acc"], "b-o", ms=4, label="Train")
    axes[1].plot(epochs, history["val_acc"],   "r-s", ms=4, label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy"); axes[1].set_ylim(0, 1.05)
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[SAVED] {out_path}")


def save_confusion_matrix(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
                linewidths=0.5, linecolor="white")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[SAVED] {out_path}")


def save_roc_curve(y_true, y_probs, auc, title, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#1f77b4")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[SAVED] {out_path}")


def save_comparison_bar(synth_m, celeb_m, out_path):
    """Side-by-side bar chart: synthetic vs Celeb-DF v2 metrics."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    s_vals  = [synth_m["accuracy"], synth_m["precision"], synth_m["recall"],
               synth_m["f1"],       synth_m["auc"]]
    c_vals  = [celeb_m["accuracy"], celeb_m["precision"], celeb_m["recall"],
               celeb_m["f1"],       celeb_m["auc"]]
    x = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w/2, s_vals, w, label="Synthetic Test Set",
                color="#2196F3", alpha=0.85)
    b2 = ax.bar(x + w/2, c_vals, w, label="Celeb-DF v2 Sample",
                color="#FF5722", alpha=0.85)
    ax.set_title("Synthetic vs Celeb-DF v2 - Performance Comparison",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[SAVED] {out_path}")


def compute_metrics(y_true, y_pred, y_probs):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_probs)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return dict(accuracy=float(acc), precision=float(prec), recall=float(rec),
                f1=float(f1), auc=float(auc), specificity=float(spec),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def print_metrics(m, title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(f"  Accuracy    : {m['accuracy']:.4f}")
    print(f"  Precision   : {m['precision']:.4f}")
    print(f"  Recall      : {m['recall']:.4f}")
    print(f"  F1-Score    : {m['f1']:.4f}")
    print(f"  Specificity : {m['specificity']:.4f}")
    print(f"  AUC-ROC     : {m['auc']:.4f}")
    print(f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
    print(f"{'='*55}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    IMG_SIZE   = 128
    BATCH_SIZE = 32
    EPOCHS     = 10
    LR         = 1e-3
    N_TRAIN    = 800
    N_VAL      = 200
    N_TEST     = 200

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Transforms ──────────────────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # ── Synthetic datasets ───────────────────────────────────────────────
    t0 = time.time()
    train_ds   = SyntheticFaceDataset(N_TRAIN, IMG_SIZE, split="train", transform=train_tf)
    val_ds     = SyntheticFaceDataset(N_VAL,   IMG_SIZE, split="val",   transform=val_tf)
    synth_test = SyntheticFaceDataset(N_TEST,  IMG_SIZE, split="test",  transform=val_tf)
    print(f"[INFO] Synthetic data ready in {time.time()-t0:.1f}s")

    train_dl      = DataLoader(train_ds,   BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl        = DataLoader(val_ds,     BATCH_SIZE, shuffle=False, num_workers=0)
    synth_test_dl = DataLoader(synth_test, BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Celeb-DF v2 datasets ─────────────────────────────────────────────
    print("[INFO] Loading Celeb-DF v2 ...")
    celeb_test_ds = CelebDFDataset(CELEBDF_DIR, split="test", transform=val_tf)
    celeb_test_dl = DataLoader(celeb_test_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────
    model = DeepFakeNet(num_classes=2).to(DEVICE)
    print(f"[INFO] DeepFakeNet parameters: {model.count_params():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # ── Training loop ────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state = 0.0, None

    print("\n" + "="*65)
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} "
          f"{'Val Loss':>9} {'Val Acc':>9}")
    print("="*65)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc          = train_epoch(model, train_dl, optimizer, criterion)
        va_loss, va_acc, _, _, _ = eval_epoch(model, val_dl, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"{epoch:>6} {tr_loss:>11.4f} {tr_acc:>10.4f} "
              f"{va_loss:>9.4f} {va_acc:>9.4f}")

    print("="*65)
    print(f"[INFO] Best validation accuracy: {best_val_acc:.4f}")

    # ── Load best checkpoint ─────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.to(DEVICE)

    # ── Evaluate: Synthetic test set ─────────────────────────────────────
    _, _, yp_s, yt_s, ypr_s = eval_epoch(model, synth_test_dl, criterion)
    synth_m = compute_metrics(yt_s, yp_s, ypr_s)
    print_metrics(synth_m, "SYNTHETIC TEST SET")
    print(classification_report(yt_s, yp_s, target_names=["Real", "Fake"]))

    # ── Evaluate: Celeb-DF v2 test set ───────────────────────────────────
    _, _, yp_c, yt_c, ypr_c = eval_epoch(model, celeb_test_dl, criterion)
    celeb_m = compute_metrics(yt_c, yp_c, ypr_c)
    print_metrics(celeb_m, "CELEB-DF v2 REAL-WORLD TEST SET")
    print(classification_report(yt_c, yp_c, target_names=["Real", "Fake"]))

    # ── Save plots ───────────────────────────────────────────────────────
    save_training_curves(history,
        os.path.join(OUT_DIR, "training_curves.png"))

    save_confusion_matrix(yt_s, yp_s,
        "Confusion Matrix - Synthetic Test Set",
        os.path.join(OUT_DIR, "confusion_matrix_synthetic.png"))

    save_confusion_matrix(yt_c, yp_c,
        "Confusion Matrix - Celeb-DF v2",
        os.path.join(OUT_DIR, "confusion_matrix_celebdf.png"))

    save_roc_curve(yt_s, ypr_s, synth_m["auc"],
        "ROC Curve - Synthetic Test Set",
        os.path.join(OUT_DIR, "roc_curve_synthetic.png"))

    save_roc_curve(yt_c, ypr_c, celeb_m["auc"],
        "ROC Curve - Celeb-DF v2",
        os.path.join(OUT_DIR, "roc_curve_celebdf.png"))

    save_comparison_bar(synth_m, celeb_m,
        os.path.join(OUT_DIR, "comparison_bar.png"))

    # ── Save all metrics ─────────────────────────────────────────────────
    all_metrics = {
        "synthetic": {**synth_m,
                      "best_val_acc": best_val_acc,
                      "params": model.count_params(),
                      "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
                      "train_loss": history["train_loss"],
                      "val_loss":   history["val_loss"],
                      "train_acc":  history["train_acc"],
                      "val_acc":    history["val_acc"]},
        "celebdf": celeb_m,
    }
    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[SAVED] {metrics_path}")
    print(f"[DONE]  All results saved to {OUT_DIR}")

    return all_metrics


if __name__ == "__main__":
    main()
