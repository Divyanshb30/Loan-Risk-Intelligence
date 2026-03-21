import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import yaml
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve
from src.utils.logger import get_logger

logger = get_logger(__name__)


from src.utils.config import load_config, get_project_root


def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    XGBoost's built-in class imbalance handler.
    Formula: count(negative class) / count(positive class)
    
    Why: tells XGBoost to penalize missing a bad loan (minority class)
    more heavily than missing a good loan.
    At 92/8 split: scale_pos_weight = 92/8 = 11.5
    This means each bad loan counts as 11.5 good loans in the loss function.
    """
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    spw = neg / pos
    logger.info(f"Class distribution — Negative: {neg:,} | Positive: {pos:,}")
    logger.info(f"scale_pos_weight set to: {spw:.4f}")
    return spw


def time_aware_split(X: pd.DataFrame, y: pd.Series,
                     df: pd.DataFrame, 
                     test_size: float = 0.2,
                     split_mode: str = "year_stratified"
                     ) -> tuple:
    """
    Two split modes:

    'temporal' (old): last 20% of data by time → pure out-of-time test
                      Problem: test is entirely post-2016, NN never sees
                      this distribution during training

    'year_stratified' (new): 20% from each year → representative test
                              NN sees both regimes during training
                              More realistic evaluation of loan risk model
    
    We document this choice explicitly — it's a modeling decision, not a flaw.
    """
    if split_mode == "temporal":
        split_idx  = int(len(X) * (1 - test_size))
        X_train    = X.iloc[:split_idx]
        X_test     = X.iloc[split_idx:]
        y_train    = y.iloc[:split_idx]
        y_test     = y.iloc[split_idx:]

    elif split_mode == "year_stratified":
        from sklearn.model_selection import train_test_split

        # Stratify by year to ensure each year is represented in both sets
        # This ensures the NN sees post-2016 loans during training
        if 'issue_year' in df.columns:
            strat_col = df['issue_year'].astype(str)
        else:
            strat_col = None

        idx = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=42,
            stratify=strat_col
        )
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    logger.info(f"Split mode: {split_mode}")
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    logger.info(
        f"Train default rate: {y_train.mean():.3f} | "
        f"Test default rate:  {y_test.mean():.3f}"
    )
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test: pd.DataFrame, 
                   y_test: pd.Series, split_name: str = "test") -> dict:
    """
    Comprehensive evaluation — not just AUC.
    
    Why multiple metrics:
    - AUC-ROC: overall discrimination ability (threshold-independent)
    - AUC-PR: better for imbalanced data — focuses on minority class performance
    - Brier Score: calibration quality — is the 80% probability actually right 80% of the time?
    - Classification report: precision/recall at default threshold (0.5)
    
    A model with high AUC but poor Brier score is discriminating well
    but its probabilities are not trustworthy — dangerous for a risk system
    that uses scores to make business decisions.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    metrics = {
        f"{split_name}_auc_roc":    roc_auc_score(y_test, y_prob),
        f"{split_name}_auc_pr":     average_precision_score(y_test, y_prob),
        f"{split_name}_brier":      brier_score_loss(y_test, y_prob),
    }
    
    logger.info(f"\n{'='*40}")
    logger.info(f"EVALUATION — {split_name.upper()}")
    logger.info(f"AUC-ROC:     {metrics[f'{split_name}_auc_roc']:.4f}")
    logger.info(f"AUC-PR:      {metrics[f'{split_name}_auc_pr']:.4f}")
    logger.info(f"Brier Score: {metrics[f'{split_name}_brier']:.4f}")
    logger.info(f"\nClassification Report:\n"
                f"{classification_report(y_test, y_pred, digits=4)}")
    
    return metrics, y_prob


def train_with_cv(X_train: pd.DataFrame, y_train: pd.Series,
                  params: dict, n_splits: int = 5) -> dict:
    """
    TimeSeriesSplit cross-validation tracking AUC-ROC, AUC-PR, and Brier Score.
    All three are needed:
    - AUC-ROC: overall discrimination
    - AUC-PR: minority class performance (critical for 92/8 imbalance)
    - Brier: calibration stability across time folds
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_auc_roc = []
    fold_auc_pr  = []
    fold_brier   = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val   = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val   = y_train.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params, eval_metric='auc',
                                   early_stopping_rounds=50,
                                   verbosity=0)
        
        model.fit(X_fold_train, y_fold_train,
                  eval_set=[(X_fold_val, y_fold_val)],
                  verbose=False)
        
        y_prob = model.predict_proba(X_fold_val)[:, 1]
        
        auc_roc = roc_auc_score(y_fold_val, y_prob)
        auc_pr  = average_precision_score(y_fold_val, y_prob)
        brier   = brier_score_loss(y_fold_val, y_prob)
        
        fold_auc_roc.append(auc_roc)
        fold_auc_pr.append(auc_pr)
        fold_brier.append(brier)
        
        logger.info(
            f"Fold {fold} — "
            f"AUC-ROC: {auc_roc:.4f} | "
            f"AUC-PR: {auc_pr:.4f} | "
            f"Brier: {brier:.4f}"
        )
    
    results = {
        "cv_auc_roc_mean": np.mean(fold_auc_roc),
        "cv_auc_roc_std":  np.std(fold_auc_roc),
        "cv_auc_pr_mean":  np.mean(fold_auc_pr),
        "cv_auc_pr_std":   np.std(fold_auc_pr),
        "cv_brier_mean":   np.mean(fold_brier),
        "cv_brier_std":    np.std(fold_brier),
        "cv_scores":       fold_auc_roc      # kept for backward compatibility
    }
    
    logger.info(
        f"\nCV Summary — "
        f"AUC-ROC: {results['cv_auc_roc_mean']:.4f} ± {results['cv_auc_roc_std']:.4f} | "
        f"AUC-PR: {results['cv_auc_pr_mean']:.4f} ± {results['cv_auc_pr_std']:.4f} | "
        f"Brier: {results['cv_brier_mean']:.4f} ± {results['cv_brier_std']:.4f}"
    )
    
    return results


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series,
                  config: dict) -> xgb.XGBClassifier:
    """
    Full training run with MLflow tracking.
    Every hyperparameter, metric, and artifact is logged.
    """
    
    params = {
        "n_estimators":     config["xgboost"]["n_estimators"],
        "max_depth":        config["xgboost"]["max_depth"],
        "learning_rate":    config["xgboost"]["learning_rate"],
        "subsample":        config["xgboost"].get("subsample", 0.8),
        "colsample_bytree": config["xgboost"].get("colsample_bytree", 0.8),
        "min_child_weight": config["xgboost"].get("min_child_weight", 5),
        "gamma":            config["xgboost"].get("gamma", 0.1),
        "reg_alpha":        config["xgboost"].get("reg_alpha", 0.1),
        "reg_lambda":       config["xgboost"].get("reg_lambda", 1.0),
        "scale_pos_weight": compute_scale_pos_weight(y_train),
        "random_state":     config["model"]["random_state"],
        "n_jobs":           -1,
        "tree_method":      "hist",   # fastest for large datasets
        "device":           "cuda",
    }
    
    mlflow.set_experiment("LoanRiskIQ_XGBoost")
    
    with mlflow.start_run(run_name="xgboost_baseline"):
        
        # Log all hyperparameters
        mlflow.log_params(params)
        
        # Cross-validation on training data
        logger.info("Running TimeSeriesSplit cross-validation...")
        cv_results = train_with_cv(X_train, y_train, params,
                                   n_splits=config["model"]["cv_folds"])
        mlflow.log_metrics({
            "cv_auc_roc_mean": cv_results["cv_auc_roc_mean"],
            "cv_auc_roc_std":  cv_results["cv_auc_roc_std"],
            "cv_auc_pr_mean":  cv_results["cv_auc_pr_mean"],
            "cv_auc_pr_std":   cv_results["cv_auc_pr_std"],
            "cv_brier_mean":   cv_results["cv_brier_mean"],
            "cv_brier_std":    cv_results["cv_brier_std"],
        })

        # Final model — train on full training set
        logger.info("Training final model on full training set...")
        model = xgb.XGBClassifier(**params, eval_metric='auc',
                                   early_stopping_rounds=50,
                                   verbosity=0)
        
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=100)
        
        # Evaluate on test set
        test_metrics, y_prob = evaluate_model(model, X_test, y_test, "test")
        mlflow.log_metrics(test_metrics)
        
        # Save feature importance
        importance_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        output_dir = Path(get_project_root()) / config["paths"]["outputs"]
        output_dir.mkdir(parents=True, exist_ok=True)
        importance_path = output_dir / "xgb_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path))
        
        logger.info(f"\nTop 10 features:\n{importance_df.head(10).to_string()}")
        
        # Save model
        model_path = output_dir / "xgb_model.json"
        model.save_model(str(model_path))
        mlflow.xgboost.log_model(model, "xgboost_model")
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
    
    return model, y_prob


def run_xgboost_training(config_path: str = "configs/config.yaml"):
    """Entry point — loads data, builds features, trains model."""
    from src.features.build_features import build_features

    config = load_config(config_path)

    logger.info("Loading processed data...")
    df = pd.read_parquet(config["paths"]["processed_data"])

    X, y = build_features(df)

    X_train, X_test, y_train, y_test = time_aware_split(
        X, y, df,
        test_size=config["model"]["test_size"],
        split_mode=config["model"]["split_mode"]    # ← wire config through
    )


    # ── FIX 1: define output_dir BEFORE it's used ──────────────────
    output_dir = Path(get_project_root()) / config["paths"]["outputs"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── FIX 2: build params BEFORE generate_oof_predictions ────────
    params = {
        "n_estimators":     config["xgboost"]["n_estimators"],
        "max_depth":        config["xgboost"]["max_depth"],
        "learning_rate":    config["xgboost"]["learning_rate"],
        "subsample":        config["xgboost"].get("subsample", 0.8),
        "colsample_bytree": config["xgboost"].get("colsample_bytree", 0.8),
        "min_child_weight": config["xgboost"].get("min_child_weight", 5),
        "gamma":            config["xgboost"].get("gamma", 0.1),
        "reg_alpha":        config["xgboost"].get("reg_alpha", 0.1),
        "reg_lambda":       config["xgboost"].get("reg_lambda", 1.0),
        "scale_pos_weight": compute_scale_pos_weight(y_train),
        "random_state":     config["model"]["random_state"],
        "n_jobs": -1,
        "tree_method": "hist",
        "device": "cuda",
    }

    logger.info("Generating OOF predictions for NN stacking...")
    generate_oof_predictions(X_train, y_train, params, config, output_dir)

    model, y_prob = train_xgboost(X_train, y_train, X_test, y_test, config)

    # ── FIX 3: output_dir already defined above, remove duplicate ──
    X_train.to_parquet(output_dir / "X_train.parquet")
    X_test.to_parquet(output_dir  / "X_test.parquet")
    y_train.to_frame().to_parquet(output_dir / "y_train.parquet")
    y_test.to_frame().to_parquet(output_dir  / "y_test.parquet")

    pd.DataFrame({"y_prob_xgb": y_prob}).to_parquet(
        output_dir / "xgb_predictions.parquet"
    )

    logger.info("All outputs saved. XGBoost training complete.")
    return model

def generate_oof_predictions(X_train: pd.DataFrame, y_train: pd.Series,
                              params: dict, config: dict,
                              output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns OOF predictions AND the boolean mask of which indices
    received predictions. First TimeSeriesSplit chunk is excluded
    because it's never in a validation fold.
    """
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=config["model"]["cv_folds"])
    oof_preds = np.full(len(X_train), np.nan)  # nan instead of 0 — nan is honest

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        logger.info(f"OOF fold {fold}/{config['model']['cv_folds']}...")

        fold_model = xgb.XGBClassifier(
            **params,
            verbosity=0
        )
        fold_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof_preds[val_idx] = fold_model.predict_proba(
            X_train.iloc[val_idx]
        )[:, 1]

        oof_auc = roc_auc_score(y_train.iloc[val_idx], oof_preds[val_idx])
        logger.info(f"  Fold {fold} OOF AUC: {oof_auc:.4f}")

    # Mask of samples that actually got predictions
    valid_mask = ~np.isnan(oof_preds)
    valid_preds = oof_preds[valid_mask]
    valid_labels = y_train.values[valid_mask]

    n_total   = len(X_train)
    n_valid   = valid_mask.sum()
    n_excluded = n_total - n_valid
    logger.info(f"Samples with OOF predictions: {n_valid:,} / {n_total:,} "
                f"({n_excluded:,} excluded — first TimeSeriesSplit chunk)")
    logger.info(f"Valid OOF AUC: {roc_auc_score(valid_labels, valid_preds):.4f}")

    # Save both
    np.save(output_dir / "xgb_oof_predictions.npy", oof_preds)   # full array with nans
    np.save(output_dir / "xgb_oof_valid_mask.npy",  valid_mask)  # which rows are usable

    return oof_preds, valid_mask

if __name__ == "__main__":
    run_xgboost_training()
