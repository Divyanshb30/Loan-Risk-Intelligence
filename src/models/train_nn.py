import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch
import yaml
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import load_config, get_project_root

logger = get_logger(__name__)


# ── Device Setup ──────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Detect and return best available device.
    RTX 3060 with CUDA 12 will return cuda:0.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f}GB")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available -- training on CPU (will be slow)")
    return device


# ── Dataset ───────────────────────────────────────────────────────────────────

class LoanDataset(Dataset):
    """
    PyTorch Dataset wrapping feature matrix and target vector.
    
    Why a custom Dataset instead of using tensors directly:
    DataLoader needs a Dataset object to handle batching, shuffling,
    and parallel data loading via num_workers. This is the standard
    PyTorch pattern for tabular data.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Architecture ──────────────────────────────────────────────────────────────

class LoanRiskNN(nn.Module):
    """
    Deep tabular neural network for binary default prediction.
    
    Architecture decisions:
    
    1. Input BatchNorm instead of StandardScaler inside model:
       BatchNorm on input normalizes each feature to mean=0, std=1
       per batch. This is more robust than a fixed scaler fitted on 
       training data because it adapts to batch statistics. Helps 
       with the macro features that have different scales.
    
    2. Four hidden layers [512, 256, 128, 64]:
       - Wide first layer (512) to learn many feature combinations
       - Progressive narrowing forces compression and generalization
       - 62 input features -> 512 is a reasonable expansion ratio
       - 64 output layer keeps the final representation compact
       
    3. BatchNorm after every Linear layer:
       Stabilizes training, allows higher learning rates,
       acts as a regularizer. Standard for deep tabular nets.
    
    4. GELU activation instead of ReLU:
       GELU is smoother than ReLU (no hard zero cutoff).
       Empirically performs better on tabular data.
       Used by BERT, GPT — not just for NLP, good generally.
    
    5. Dropout after every activation:
       Randomly zeros 30% of neurons during training.
       Forces the network to learn redundant representations.
       Prevents co-adaptation of neurons.
    
    6. Single sigmoid output:
       Binary classification -> sigmoid maps to [0,1] probability.
       We use BCEWithLogitsLoss which applies sigmoid internally
       for numerical stability (avoids log(0) issues).
       So the model outputs raw logit, loss applies sigmoid.
    """
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()

        self.input_norm = nn.BatchNorm1d(input_dim)

        self.block1 = self._block(input_dim, 512, dropout)
        self.block2 = self._block(512, 256, dropout)
        self.block3 = self._block(256, 128, dropout)
        self.block4 = self._block(128, 64, dropout)

        # Output: single logit (BCEWithLogitsLoss handles sigmoid)
        self.output = nn.Linear(64, 1)

        self._init_weights()

    def _block(self, in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def _init_weights(self):
        """
        Kaiming (He) initialization for all Linear layers.
        Designed specifically for layers followed by ReLU/GELU.
        Prevents vanishing/exploding gradients at initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.output(x).squeeze(1)  # shape: (batch_size,)


# ── Training Loop ─────────────────────────────────────────────────────────────

def compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    """
    BCEWithLogitsLoss pos_weight: same concept as XGBoost's scale_pos_weight.
    Tells the loss function to weight positive (bad loan) examples more heavily.
    At 92/8 split: pos_weight = 92/8 = 11.5
    Each bad loan contributes 11.5x more to the loss than a good loan.
    """
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    pw = neg / pos
    logger.info(f"pos_weight: {pw:.4f}")
    return torch.tensor([pw], dtype=torch.float32)


def evaluate_epoch(model: nn.Module, loader: DataLoader,
                   device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one evaluation pass — no gradient computation."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    y_prob  = np.array(all_probs)
    y_true  = np.array(all_labels)
    auc_roc = roc_auc_score(y_true, y_prob)
    return auc_roc, y_prob, y_true


def train_nn(X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray, y_test: np.ndarray,
             config: dict, device: torch.device) -> LoanRiskNN:

    # ── Hyperparameters ───────────────────────────────────────────────────────
    batch_size  = config["nn"]["batch_size"]
    epochs      = config["nn"]["epochs"]
    lr          = config["nn"]["learning_rate"]
    dropout     = config["nn"]["dropout"]
    patience    = config["nn"]["patience"]

    # ── Data Loaders ──────────────────────────────────────────────────────────
    train_ds = LoanDataset(X_train, y_train)
    test_ds  = LoanDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True    # pins CPU memory for faster GPU transfer
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ── Model, Loss, Optimizer ────────────────────────────────────────────────
    input_dim  = X_train.shape[1]
    model      = LoanRiskNN(input_dim, dropout=dropout).to(device)

    pos_weight = compute_pos_weight(y_train).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=config["nn"].get("weight_decay", 1e-4)
    )

    # ── Learning Rate Scheduler ───────────────────────────────────────────────
    # ReduceLROnPlateau: halves LR when validation AUC stops improving
    # Patience=3 means: wait 3 epochs of no improvement before reducing
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=3, min_lr=1e-6
    )

    # ── Training ──────────────────────────────────────────────────────────────
    mlflow.set_experiment("LoanRiskIQ_NN")

    with mlflow.start_run(run_name="nn_baseline"):
        mlflow.log_params({
            "batch_size": batch_size, "epochs": epochs,
            "learning_rate": lr, "dropout": dropout,
            "architecture": "512-256-128-64",
            "activation": "GELU", "optimizer": "AdamW"
        })

        best_auc    = 0.0
        best_epoch  = 0
        patience_ct = 0
        output_dir  = Path(get_project_root()) / config["paths"]["outputs"]
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            # ── Train pass ────────────────────────────────────────────────────
            model.train()
            total_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
                loss.backward()

                # Gradient clipping — prevents exploding gradients
                # Clips gradient norm to max 1.0 if it exceeds that
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # ── Eval pass ─────────────────────────────────────────────────────
            val_auc, y_prob, y_true = evaluate_epoch(model, test_loader, device)
            val_auc_pr  = average_precision_score(y_true, y_prob)
            val_brier   = brier_score_loss(y_true, y_prob)

            scheduler.step(val_auc)

            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                f"AUC-ROC: {val_auc:.4f} | AUC-PR: {val_auc_pr:.4f} | "
                f"Brier: {val_brier:.4f} | LR: {current_lr:.2e}"
            )

            mlflow.log_metrics({
                "train_loss": avg_loss, "val_auc_roc": val_auc,
                "val_auc_pr": val_auc_pr, "val_brier": val_brier,
                "learning_rate": current_lr
            }, step=epoch)

            # ── Early stopping + checkpoint ───────────────────────────────────
            if val_auc > best_auc:
                best_auc   = val_auc
                best_epoch = epoch
                patience_ct = 0
                torch.save(model.state_dict(),
                           output_dir / "nn_best.pt")
                logger.info(f"  -> New best AUC: {best_auc:.4f} (saved checkpoint)")
            else:
                patience_ct += 1
                if patience_ct >= patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(best epoch: {best_epoch}, best AUC: {best_auc:.4f})"
                    )
                    break

        # ── Load best checkpoint for final eval ───────────────────────────────
        model.load_state_dict(
            torch.load(output_dir / "nn_best.pt", map_location=device)
        )
        final_auc, y_prob_final, y_true_final = evaluate_epoch(
            model, test_loader, device
        )
        final_auc_pr = average_precision_score(y_true_final, y_prob_final)
        final_brier  = brier_score_loss(y_true_final, y_prob_final)

        mlflow.log_metrics({
            "final_auc_roc": final_auc,
            "final_auc_pr":  final_auc_pr,
            "final_brier":   final_brier,
            "best_epoch":    best_epoch
        })

        # Save predictions for ensemble
        pd.DataFrame({"y_prob_nn": y_prob_final}).to_parquet(
            output_dir / "nn_predictions.parquet"
        )

        mlflow.pytorch.log_model(model, "nn_model")
        logger.info(f"NN training complete. Best AUC-ROC: {best_auc:.4f}")

    return model, y_prob_final


def run_nn_training(config_path: str = "configs/config.yaml"):
    """Entry point — loads splits from XGBoost run, scales, trains NN."""
    from src.features.build_features import build_features

    config     = load_config(config_path)
    device     = get_device()
    output_dir = Path(get_project_root()) / config["paths"]["outputs"]

    logger.info("Loading train/test splits from XGBoost run...")
    X_train = pd.read_parquet(output_dir / "X_train.parquet").values
    X_test  = pd.read_parquet(output_dir / "X_test.parquet").values
    y_train = pd.read_parquet(output_dir / "y_train.parquet").values.ravel()
    y_test  = pd.read_parquet(output_dir / "y_test.parquet").values.ravel()

    # ── Scale features ────────────────────────────────────────────────────────
    # NNs require standardized inputs — gradient descent converges much
    # faster when all features are on the same scale (~mean 0, std 1).
    # XGBoost doesn't need this (tree splits are scale-invariant).
    # Fit ONLY on train data — never fit on test (data leakage).
    logger.info("Scaling features...")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Save scaler for inference
    import joblib
    joblib.dump(scaler, output_dir / "scaler.joblib")
    logger.info("Scaler saved")

    model, y_prob = train_nn(
        X_train, y_train, X_test, y_test, config, device
    )

    return model, y_prob


if __name__ == "__main__":
    run_nn_training()
