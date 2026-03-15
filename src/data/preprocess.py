import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.utils.logger import get_logger
logger = get_logger(__name__)

from src.utils.config import load_config

def load_raw_data(path: str) -> pd.DataFrame:
    path = str(PROJECT_ROOT / path)
    parquet_path = path.replace(".dta", ".parquet")
    
    if os.path.exists(parquet_path):
        logger.info("Loading from parquet cache...")
        return pd.read_parquet(parquet_path)
    
    logger.info("Loading from .dta (first time - will be slow)...")
    df = pd.read_stata(path)
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved parquet cache to {parquet_path}")
    return df


def remove_leakage(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    leakage_cols = config["features"]["drop_leakage"]
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped {len(cols_to_drop)} leakage columns: {cols_to_drop}")
    return df


def remove_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ['url', 'id', 'member_id', 'emp_title', 'title',
               'desc', 'zip_code', 'issue_d', 'earliest_cr_line',
               'last_credit_pull_d']
    
    zero_var_cols = ['pymnt_plan', 'pymnt_plan_', 'policy_code']
    all_missing = [c for c in df.columns if df[c].isna().sum() == len(df)]
    joint_cols = [c for c in df.columns if 'joint' in c.lower()]
    sec_app_cols = [c for c in df.columns if c.startswith('sec_app')]
    political_cols = ['rep', 'dem', 'red', 'christianity', 'religiousity']
    duplicate_targets = ['rankloan_status_', 'loan_status_5', 'loan_status_']
    high_missing = [
        'mths_since_last_record', 'mths_since_last_major_derog',
        'mths_since_recent_bc_dlq', 'mths_since_recent_revol_delinq',
        'lognum_accts_ever_120_pd', 'logpercent_bc_gt_75',
        'verification_status_joint'
    ]
    
    all_drop = list(set(
        id_cols + zero_var_cols + all_missing +
        joint_cols + sec_app_cols + political_cols +
        duplicate_targets + high_missing
    ))
    
    cols_to_drop = [c for c in all_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped {len(cols_to_drop)} useless/redundant columns")
    
    if all_missing:
        logger.warning(f"Found {len(all_missing)} fully null columns: {all_missing}")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # Domain-specific: missing = never happened
    never_happened = ['mths_since_last_delinq', 'mths_since_recent_bc',
                      'mths_since_recent_inq']
    for col in never_happened:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(999)
            logger.info(f"{col}: filled {n_missing:,} nulls with 999 (never delinquent)")

    # Domain-specific: missing = zero of this thing exists
    zero_impute = ['il_util', 'open_acc_6m', 'inq_last_12m',
                   'total_bal_il', 'total_cu_tl', 'inq_fi',
                   'open_act_il', 'mort_acc', 'pub_rec_bankruptcies',
                   'tax_liens', 'acc_now_delinq', 'delinq_amnt',
                   'num_tl_120dpd_2m', 'num_tl_30dpd']
    for col in zero_impute:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                df[col] = df[col].fillna(0)
                logger.info(f"{col}: filled {n_missing:,} nulls with 0")

    # Remaining numeric: median impute
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"{col}: filled {n_missing:,} nulls with median ({median_val:.4f})")

    # Categorical: Unknown
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = df[col].fillna('Unknown')
            logger.info(f"{col}: filled {n_missing:,} nulls with 'Unknown'")

    remaining = df.isna().sum().sum()
    if remaining > 0:
        logger.warning(f"{remaining} missing values still remain after imputation")
    else:
        logger.info("All missing values resolved")

    return df


def filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    initial_len = len(df)
    
    if 'loan_status' in df.columns:
        exclude_statuses = ['Current', 'In Grace Period']
        df = df[~df['loan_status'].isin(exclude_statuses)]
        logger.info(f"Excluded ambiguous loan statuses: {exclude_statuses}")

    df = df[df['badloan'].notna()]
    
    removed = initial_len - len(df)
    logger.info(f"Filtered {removed:,} ambiguous rows — {len(df):,} remain")
    return df


def run_preprocessing(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    
    # Create logs directory if it doesn't exist
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)

    logger.info("="*50)
    logger.info("PREPROCESSING PIPELINE STARTED")
    logger.info("="*50)

    df = load_raw_data(config["paths"]["raw_data"])
    logger.info(f"Loaded raw data: {df.shape}")

    df = remove_leakage(df, config)
    df = remove_useless_columns(df)
    df = filter_valid_rows(df)
    df = handle_missing_values(df)

    logger.info("="*50)
    logger.info(f"PREPROCESSING COMPLETE — Final shape: {df.shape}")
    logger.info(f"Target distribution: {df['badloan'].value_counts(normalize=True).round(3).to_dict()}")
    logger.info("="*50)

    output_path = PROJECT_ROOT / config["paths"]["processed_data"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")

    return df


if __name__ == "__main__":
    df = run_preprocessing()


