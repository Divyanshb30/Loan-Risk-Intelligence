import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch
import joblib
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import load_config, get_project_root

logger = get_logger(__name__)


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props  = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f}GB")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available -- training on CPU")
    return device


# ── Loss ──────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy negatives, focuses on hard misclassified samples.
    alpha: class balance weight (0.25 standard from original paper)
    gamma: focusing parameter — higher = more focus on hard examples
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs   = torch.sigmoid(logits)
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_w = alpha_t * (1 - p_t) ** self.gamma
        return (focal_w * bce_loss).mean()


# ── Dataset ───────────────────────────────────────────────────────────────────

class LoanDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Architecture ──────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Skip connection: output = activation(x + F(x))
    Prevents gradient vanishing in deep networks.
    If block learns nothing, output = x (identity) — safe to add more layers.
    """
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.GELU()
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.activation(x + self.block(x)))


class LoanRiskNN(nn.Module):
    """
    Residual tabular NN: input(63) -> 256 -> ResBlock -> 128 -> ResBlock -> 64 -> 1
    63 = 62 features + 1 XGBoost logit (stacking feature)
    """
    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()

        self.input_norm = nn.BatchNorm1d(input_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res_block1 = ResidualBlock(256, dropout)

        self.compress1  = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res_block2 = ResidualBlock(128, dropout)

        self.compress2  = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.output = nn.Linear(64, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_proj(x)
        x = self.res_block1(x)
        x = self.compress1(x)
        x = self.res_block2(x)
        x = self.compress2(x)
        return self.output(x).squeeze(1)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_epoch(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            probs   = torch.sigmoid(model(X_batch)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)
    return roc_auc_score(y_true, y_prob), y_prob, y_true


# ── Train ─────────────────────────────────────────────────────────────────────

def train_nn(X_train, y_train, X_test, y_test, config, device):

    batch_size = config["nn"]["batch_size"]
    epochs     = config["nn"]["epochs"]
    lr         = config["nn"]["learning_rate"]
    dropout    = config["nn"]["dropout"]
    patience   = config["nn"]["patience"]

    train_ds = LoanDataset(X_train, y_train)
    test_ds  = LoanDataset(X_test,  y_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True
    )
    # 20% subsample of train for fast per-epoch train AUC
    train_eval_ds = torch.utils.data.Subset(
        train_ds,
        np.random.choice(len(train_ds), int(len(train_ds) * 0.2), replace=False)
    )
    train_loader_eval = DataLoader(
        train_eval_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True
    )

    model     = LoanRiskNN(X_train.shape[1], dropout=dropout).to(device)
    if config["nn"].get("use_focal", False):
        criterion = FocalLoss(
            alpha=config["nn"].get("focal_alpha", 0.25),
            gamma=config["nn"].get("focal_gamma", 2.0)
        )
        logger.info("Loss: FocalLoss")
    else:
        pos_rate = y_train.mean()
        raw_weight = (1 - pos_rate) / pos_rate          # ~11.8 at 7.8% positive
        cap = config["nn"].get("pos_weight_cap", 3.0)
        pos_weight_val = min(raw_weight, cap)            # capped at 3.0
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_val], device=device)
        )
        logger.info(f"Loss: BCE | pos_weight={pos_weight_val:.2f} (raw={raw_weight:.1f}, cap={cap})")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=config["nn"].get("weight_decay", 1e-4)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=config["nn"].get("scheduler_patience", 6),
        min_lr=1e-6
    )

    mlflow.set_experiment("LoanRiskIQ_NN")
    output_dir = Path(get_project_root()) / config["paths"]["outputs"]
    output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="nn_stacked_focal"):
        mlflow.log_params({
            "batch_size": batch_size, "epochs": epochs,
            "learning_rate": lr, "dropout": dropout,
            "architecture": "residual_256-128-64",
            "loss": "FocalLoss", "optimizer": "AdamW",
            "stacking": "xgb_logit"
        })

        best_auc    = 0.0
        best_epoch  = 0
        patience_ct = 0

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0

            for X_b, y_b in train_loader:
                X_b = X_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            val_auc,   y_prob, y_true = evaluate_epoch(model, test_loader,       device)
            train_auc, _,      _      = evaluate_epoch(model, train_loader_eval,  device)
            val_auc_pr = average_precision_score(y_true, y_prob)
            val_brier  = brier_score_loss(y_true, y_prob)

            scheduler.step(val_auc)
            current_lr = optimizer.param_groups[0]['lr']

            logger.info(
                f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                f"Train AUC: {train_auc:.4f} | Test AUC: {val_auc:.4f} | "
                f"AUC-PR: {val_auc_pr:.4f} | Brier: {val_brier:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            mlflow.log_metrics({
                "train_loss": avg_loss, "train_auc_roc": train_auc,
                "val_auc_roc": val_auc, "val_auc_pr": val_auc_pr,
                "val_brier": val_brier, "learning_rate": current_lr
            }, step=epoch)

            if val_auc > best_auc:
                best_auc, best_epoch, patience_ct = val_auc, epoch, 0
                torch.save(model.state_dict(), output_dir / "nn_best.pt")
                logger.info(f"  -> New best AUC: {best_auc:.4f} (saved)")
            else:
                patience_ct += 1
                if patience_ct >= patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(best: epoch {best_epoch}, AUC {best_auc:.4f})"
                    )
                    break

        model.load_state_dict(
            torch.load(output_dir / "nn_best.pt",
                       map_location=device, weights_only=True)
        )
        final_auc, y_prob_final, y_true_final = evaluate_epoch(
            model, test_loader, device
        )
        final_auc_pr = average_precision_score(y_true_final, y_prob_final)
        final_brier  = brier_score_loss(y_true_final, y_prob_final)

        mlflow.log_metrics({
            "final_auc_roc": final_auc, "final_auc_pr": final_auc_pr,
            "final_brier": final_brier, "best_epoch": best_epoch
        })

        pd.DataFrame({"y_prob_nn": y_prob_final}).to_parquet(
            output_dir / "nn_predictions.parquet"
        )
        mlflow.pytorch.log_model(model, name="nn_model")
        logger.info(f"NN training complete. Best AUC-ROC: {best_auc:.4f}")

    return model, y_prob_final


# ── Entry Point ───────────────────────────────────────────────────────────────

def run_nn_training(config_path: str = "configs/config.yaml"):
    config     = load_config(config_path)
    device     = get_device()
    output_dir = Path(get_project_root()) / config["paths"]["outputs"]

    logger.info("Loading train/test splits...")
    X_train = pd.read_parquet(output_dir / "X_train.parquet")
    X_test  = pd.read_parquet(output_dir / "X_test.parquet")
    y_train = pd.read_parquet(output_dir / "y_train.parquet").values.ravel()
    y_test  = pd.read_parquet(output_dir / "y_test.parquet").values.ravel()

    # Load OOF predictions + valid mask — filter out first-chunk NaNs
    xgb_oof    = np.load(output_dir / "xgb_oof_predictions.npy")
    valid_mask = np.load(output_dir / "xgb_oof_valid_mask.npy")
    xgb_test   = pd.read_parquet(
        output_dir / "xgb_predictions.parquet"
    )["y_prob_xgb"].values

    # Apply mask — removes 360K NaN rows from train set
    X_train_valid = X_train.values[valid_mask]
    y_train_valid = y_train[valid_mask]
    xgb_oof_valid = xgb_oof[valid_mask]

    logger.info(f"Train size after mask: {X_train_valid.shape[0]:,}")

    eps = 1e-7
    def safe_logit(p):
        return np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))

    xgb_oof_logit  = safe_logit(xgb_oof_valid).reshape(-1, 1)
    xgb_test_logit = safe_logit(xgb_test).reshape(-1, 1)

    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_valid)
    X_test_scaled  = scaler.transform(X_test.values)
    joblib.dump(scaler, output_dir / "scaler.joblib")

    X_train_final = np.hstack([X_train_scaled, xgb_oof_logit])
    X_test_final  = np.hstack([X_test_scaled,  xgb_test_logit])

    logger.info(f"Final input dim: {X_train_final.shape[1]} "
                f"(62 features + 1 XGB logit)")

    # NaN check before training
    assert not np.isnan(X_train_final).any(), "NaN in X_train_final"
    assert not np.isnan(X_test_final).any(),  "NaN in X_test_final"
    logger.info("NaN check passed — clean inputs confirmed")

    return train_nn(X_train_final, y_train_valid,
                    X_test_final,  y_test,
                    config, device)


if __name__ == "__main__":
    run_nn_training()
