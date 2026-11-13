# ============================================================
# TWO-DATASET PSORIASIS CLASSIFICATION PIPELINE
# ResNet-50 + Multi-stage CBAM + Focal Loss
# Conservative image enhancement + centre crop + progressive unfreezing
# Uses stratified TRAIN / VALIDATION / TEST split (test used only once)
# ============================================================

import os, re, random, hashlib
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CONFIGURATION
# =========================
DATASET_1_ROOT = r"C:\Users\E028.21\Downloads\archive (3)"
DATASET_2_ROOT = r"C:\Users\E028.21\Downloads\archive (4)\SkinDisease\SkinDisease"

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 20
SEED = 42
NUM_WORKERS = 0

LR_HEAD = 3e-4
LR_FINE = 1e-4
WEIGHT_DECAY = 1e-4
MIN_SAMPLES_PER_CLASS = 8

MODEL_OUTPUT = "best_psoriasis_resnet50_cbam_two_datasets.pth"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Keep only labels that are explicitly identifiable in a folder name or filename.
# Do not infer a subtype from a generic 'psoriasis' folder.
CLASS_PATTERNS = {
    "erythrodermic": [r"erythroderm"],
    "pustular": [r"pustular"],
    "guttate": [r"guttate"],
    "plaque": [r"plaque"],
    "inverse": [r"inverse"],
}

# =========================
# REPRODUCIBILITY / DEVICE
# =========================
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# DATA DISCOVERY + LABELING
# =========================
def identify_subtype(path):
    """Return a subtype only when its name is explicit in the path."""
    text = str(path).lower().replace("_", " ").replace("-", " ")
    matches = []
    for label, patterns in CLASS_PATTERNS.items():
        if any(re.search(pattern, text) for pattern in patterns):
            matches.append(label)
    # Ambiguous path containing more than one subtype is not labelled.
    return matches[0] if len(matches) == 1 else None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_dataset_images(root, dataset_name):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    records = []
    skipped_unlabelled = 0
    bad_images = 0
    for p in root.rglob("*"):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        label = identify_subtype(p)
        if label is None:
            skipped_unlabelled += 1
            continue
        try:
            with Image.open(p) as im:
                im.verify()
        except (UnidentifiedImageError, OSError, ValueError):
            bad_images += 1
            continue
        records.append({"path": str(p), "label": label, "source": dataset_name})

    print(f"\n{dataset_name}: labelled images={len(records)}, "
          f"unlabelled/ambiguous skipped={skipped_unlabelled}, corrupt={bad_images}")
    print("Class counts:", dict(Counter(r["label"] for r in records)))
    return records


def remove_exact_duplicates(records):
    """Remove byte-identical images across both datasets."""
    seen = set()
    unique = []
    duplicates = 0
    for rec in records:
        digest = sha256_file(rec["path"])
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        unique.append(rec)
    print(f"Exact duplicate images removed: {duplicates}")
    return unique


def make_splits(records):
    """70% train, 15% validation, 15% test; stratified by subtype."""
    labels = [r["label"] for r in records]
    indices = np.arange(len(records))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=SEED, stratify=labels
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=SEED, stratify=temp_labels
    )
    return ([records[i] for i in train_idx],
            [records[i] for i in val_idx],
            [records[i] for i in test_idx])

# =========================
# IMAGE ENHANCEMENT
# =========================
def enhance_skin_image(img):
    """Mild white balance, edge-preserving denoising, CLAHE on luminance."""
    image = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

    img_f = image.astype(np.float32)
    channel_means = img_f.mean(axis=(0, 1))
    target_mean = float(channel_means.mean())
    gains = target_mean / np.maximum(channel_means, 1e-6)
    gains = np.clip(gains, 0.95, 1.05)
    balanced = np.clip(img_f * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)

    denoised = cv2.bilateralFilter(balanced, d=5, sigmaColor=30, sigmaSpace=30)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(
        cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR
    )
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))


def center_crop_lesion(img):
    """Existing centre crop; this is not an automatic lesion segmentation model."""
    w, h = img.size
    crop = int(0.8 * min(w, h))
    left, top = (w - crop) // 2, (h - crop) // 2
    return img.crop((left, top, left + crop, top + crop))

# =========================
# PYTORCH DATASET
# =========================
class PsoriasisDataset(Dataset):
    def __init__(self, records, class_to_idx, train=False):
        self.records = records
        self.class_to_idx = class_to_idx
        aug = ([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.05, 0.05, 0.03),
        ] if train else [transforms.Resize((IMG_SIZE, IMG_SIZE))])
        self.tf = transforms.Compose(aug + [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        with Image.open(rec["path"]) as im:
            img = im.convert("RGB")
        img = center_crop_lesion(img)
        img = enhance_skin_image(img)
        x = self.tf(img)
        y = self.class_to_idx[rec["label"]]
        return x, torch.tensor(y, dtype=torch.long)

# =========================
# CLASS BALANCING
# =========================
def make_class_weights(records, classes):
    counts = Counter(r["label"] for r in records)
    weights = torch.tensor(
        [1.0 / max(counts[c], 1) for c in classes], dtype=torch.float32
    )
    weights = weights / weights.sum() * len(classes)
    return weights, counts


def make_weighted_sampler(records, classes):
    _, counts = make_class_weights(records, classes)
    sample_weights = [1.0 / counts[r["label"]] for r in records]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# =========================
# CBAM + RESNET50 MODEL
# =========================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3), nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.channel_attention(x)
        spatial = torch.cat([x.mean(1, keepdim=True),
                             x.max(1, keepdim=True)[0]], dim=1)
        return x * self.spatial_attention(spatial)


class ResNet50_CBAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1,
                                  backbone.relu, backbone.maxpool)
        self.l1, self.l2 = backbone.layer1, backbone.layer2
        self.l3, self.l4 = backbone.layer3, backbone.layer4
        self.att3, self.att4 = CBAM(1024), CBAM(2048)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.att3(self.l3(x))
        x = self.att4(self.l4(x))
        return self.fc(self.pool(x).flatten(1))

# =========================
# FOCAL LOSS
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=1.5):
        super().__init__()
        self.register_buffer("alpha", alpha.float())
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = self.alpha[targets]
        return (alpha_t * (1.0 - pt).pow(self.gamma) * ce).mean()

# =========================
# TRAINING / EVALUATION
# =========================
def set_trainable(model, prefixes):
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in prefixes)


def create_optimizer(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)


def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn=None):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    y_true, y_pred = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if loss_fn is not None:
            loss_sum += loss_fn(logits, y).item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    return loss_sum / max(total, 1), correct / max(total, 1), y_true, y_pred

# =========================
# MAIN
# =========================
def main():
    records_1 = collect_dataset_images(DATASET_1_ROOT, "Dataset 1")
    records_2 = collect_dataset_images(DATASET_2_ROOT, "Dataset 2")
    records = remove_exact_duplicates(records_1 + records_2)

    counts = Counter(r["label"] for r in records)
    classes = sorted(c for c, n in counts.items() if n >= MIN_SAMPLES_PER_CLASS)
    records = [r for r in records if r["label"] in classes]

    print("\nCombined class counts:", dict(Counter(r["label"] for r in records)))
    print("Classes used:", classes)
    if len(classes) < 2:
        raise RuntimeError(
            "Fewer than two usable subtype classes were found. The screenshots only show "
            "the train/test folders, not their class folders. Open each train folder and "
            "verify that subtype names (plaque, guttate, pustular, erythrodermic, inverse) "
            "appear in the folder path or image filename. A generic/mixed psoriasis folder "
            "cannot safely be assigned to a subtype."
        )

    # Every class needs enough examples for a stratified 70/15/15 split.
    too_small = {c: counts[c] for c in classes if counts[c] < 7}
    if too_small:
        raise RuntimeError(f"Insufficient examples for stratified splitting: {too_small}")

    train_records, val_records, test_records = make_splits(records)
    print(f"\nSplit sizes: train={len(train_records)}, "
          f"validation={len(val_records)}, test={len(test_records)}")
    print("Train counts:", dict(Counter(r["label"] for r in train_records)))
    print("Validation counts:", dict(Counter(r["label"] for r in val_records)))
    print("Test counts:", dict(Counter(r["label"] for r in test_records)))

    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_ds = PsoriasisDataset(train_records, class_to_idx, train=True)
    val_ds = PsoriasisDataset(val_records, class_to_idx, train=False)
    test_ds = PsoriasisDataset(test_records, class_to_idx, train=False)

    sampler = make_weighted_sampler(train_records, classes)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)

    model = ResNet50_CBAM(len(classes)).to(device)
    alpha, _ = make_class_weights(train_records, classes)
    loss_fn = FocalLoss(alpha).to(device)

    best_val_acc = -1.0
    best_state = None

    # Phase 1: train classification head only (epochs 1-3)
    set_trainable(model, ["fc"])
    optimizer = create_optimizer(model, LR_HEAD)
    phases = [(1, min(3, EPOCHS), "HEAD")]
    if EPOCHS >= 4:
        phases.append((4, min(10, EPOCHS), "L4"))
    if EPOCHS >= 11:
        phases.append((11, EPOCHS, "L3+L4"))

    for start_epoch, end_epoch, phase in phases:
        if phase == "L4":
            set_trainable(model, ["l4", "att4", "fc"])
            optimizer = create_optimizer(model, LR_FINE)
        elif phase == "L3+L4":
            set_trainable(model, ["l3", "l4", "att3", "att4", "fc"])
            # Recreate optimizer so newly unfrozen layer3/attention parameters are included.
            optimizer = create_optimizer(model, LR_FINE * 0.5)

        for epoch in range(start_epoch, end_epoch + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn)
            print(f"Epoch {epoch:02d}/{EPOCHS} [{phase}] "
                  f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% "
                  f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")

    # Restore best validation checkpoint, then evaluate the held-out test set once.
    model.load_state_dict(best_state)
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, loss_fn)
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "img_size": IMG_SIZE,
        "test_accuracy": test_acc,
        "validation_accuracy": best_val_acc,
    }, MODEL_OUTPUT)

    print("\n================ FINAL HELD-OUT TEST RESULT ================")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Test accuracy: {test_acc*100:.2f}%")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification report:\n", classification_report(
        y_true, y_pred, labels=list(range(len(classes))),
        target_names=classes, digits=4, zero_division=0
    ))
    print("Saved model:", MODEL_OUTPUT)


if __name__ == "__main__":
    main()
