import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Feature lists ─────────────────────────────────────────────────────────────

LOAN_FEATURES = [
    'logfunded_amnt',
    'int_rate',
    'installment',
    'inq_last_6mths',
    'open_acc',
    'logtotal_acc',
    'pub_rec',
    'des_length',
    'logpct_tl_nvr_dlq',
    'logtot_hi_cred_lim',
    'revol_util',
    'revol_bal',
    'delinq_2yrs',
    'mths_since_last_delinq',
    'mths_since_recent_inq',
    'mths_since_recent_bc',
    'acc_open_past_24mths',
    'mort_acc',
    'pub_rec_bankruptcies',
    'tax_liens',
    'acc_now_delinq',
    'num_tl_90g_dpd_24m',
    'pct_tl_nvr_dlq',
    'tot_cur_bal',
    'avg_cur_bal',
    'bc_util',
    'num_actv_bc_tl',
    'num_actv_rev_tl',
]

BORROWER_FEATURES = [
    'logannual_inc',
    'logdti',
    'emp_length_',
    'home_ownership_',
    'verification_status_',
]

MACRO_FEATURES = [
    'logearnings',
    'logunemployment',
    'loglabor_force',
    'logempl_birth',
    'lognew_bus',
    'gdpcontrib',
    'loginternetuser',
    'inf',
    'inf_6m',
    'muni_points',
    'muni_6m',
    'FEDFUNDS',
    'FED_lag6',
    'riskprem',
    'CPIUS',
    'cpi',
]

CATEGORICAL_FEATURES = [
    'grade_',
    'term_',
    'purpose_',
    'application_type_',
    'initial_list_status_',
]

TIME_FEATURES = [
    'issue_year',
#     'quarter',
#     'month',
]

TARGET = 'badloan'

# All features that need encoding — CATEGORICAL + the three from BORROWER
ALL_ENCODE_FEATURES = CATEGORICAL_FEATURES + ['emp_length_', 'home_ownership_', 'verification_status_']


def validate_feature_presence(df: pd.DataFrame, features: list, group_name: str) -> list:
    present = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        logger.warning(f"{group_name}: {len(missing)} features missing: {missing}")
    logger.info(f"{group_name}: {len(present)}/{len(features)} features present")
    return present


def encode_categoricals(df: pd.DataFrame, cat_features: list) -> pd.DataFrame:
    """
    Converts all categorical/object columns to numeric integers.
    
    Three dtype cases handled:
    1. 'category' dtype (Stata-encoded) → .cat.codes gives clean 0,1,2,3...
       Note: .cat.codes returns -1 for NaN — we fill those with 0
    2. 'object' dtype (raw strings) → LabelEncoder maps to 0,1,2,3...
    3. Already numeric → no action needed
    
    XGBoost requires int, float, or bool — never category or object.
    """
    for col in cat_features:
        if col not in df.columns:
            continue

        if str(df[col].dtype) == 'category':
            codes = df[col].cat.codes
            # cat.codes returns -1 for NaN — replace with 0 (unknown category)
            codes = codes.replace(-1, 0)
            df[col] = codes.astype(int)
            logger.info(f"{col}: category -> int codes ({df[col].nunique()} unique values)")

        elif df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna('Unknown').astype(str))
            logger.info(f"{col}: object -> label encoded int ({df[col].nunique()} unique values)")

        else:
            logger.info(f"{col}: already numeric ({df[col].dtype}) -- skipped")

    return df


def engineer_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Five interaction features encoding combined risk mechanisms.
    Each has an explicit business justification.
    """
    if 'logdti' in df.columns and 'logunemployment' in df.columns:
        df['dti_x_unemployment'] = df['logdti'] * df['logunemployment']
        logger.info("Created: dti_x_unemployment")

    if 'int_rate' in df.columns and 'inf' in df.columns:
        df['real_interest_rate'] = df['int_rate'] - df['inf']
        logger.info("Created: real_interest_rate")

    if 'logfunded_amnt' in df.columns and 'logannual_inc' in df.columns:
        df['loan_to_income'] = df['logfunded_amnt'] / (df['logannual_inc'] + 1e-8)
        logger.info("Created: loan_to_income")

    if 'revol_util' in df.columns and 'delinq_2yrs' in df.columns:
        df['util_x_delinq'] = df['revol_util'] * (df['delinq_2yrs'] + 1)
        logger.info("Created: util_x_delinq")

    if 'FED_lag6' in df.columns and 'term_' in df.columns:
        df['fed_x_term'] = df['FED_lag6'] * df['term_']
        logger.info("Created: fed_x_term")

    return df


def enforce_numeric_dtypes(X: pd.DataFrame) -> pd.DataFrame:
    """
    Final safety check before any model sees the data.
    
    If anything non-numeric survives here it means upstream
    encoding missed something — we fix it and log a warning
    so you know to fix it properly upstream.
    Raises a hard error if it still can't be converted.
    """
    for col in X.columns:
        if str(X[col].dtype) == 'category':
            logger.warning(f"LATE FIX: {col} still category — converting to int codes")
            X[col] = X[col].cat.codes.replace(-1, 0).astype(int)

        elif X[col].dtype == 'object':
            logger.warning(f"LATE FIX: {col} still object — label encoding")
            X[col] = LabelEncoder().fit_transform(X[col].fillna('Unknown').astype(str))

    # Final hard check — nothing non-numeric should survive
    non_numeric = X.select_dtypes(exclude=[np.number, 'bool']).columns.tolist()
    if non_numeric:
        logger.error(f"Non-numeric columns still present after all encoding: {non_numeric}")
        raise ValueError(f"Feature matrix has non-numeric columns: {non_numeric}")

    logger.info(f"All {X.shape[1]} features verified numeric — safe for XGBoost/PyTorch")
    return X


def select_final_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Assembles final feature matrix X and target vector y.
    Validates each feature group, deduplicates, enforces dtypes.
    """
    loan_present        = validate_feature_presence(df, LOAN_FEATURES, "LOAN")
    borrower_present    = validate_feature_presence(df, BORROWER_FEATURES, "BORROWER")
    macro_present       = validate_feature_presence(df, MACRO_FEATURES, "MACRO")
    cat_present         = validate_feature_presence(df, CATEGORICAL_FEATURES, "CATEGORICAL")
    time_present        = validate_feature_presence(df, TIME_FEATURES, "TIME")

    interaction_features = [
        'dti_x_unemployment', 'real_interest_rate',
        'loan_to_income', 'util_x_delinq', 'fed_x_term'
    ]
    interaction_present = validate_feature_presence(df, interaction_features, "INTERACTION")

    all_features = (
        loan_present + borrower_present + macro_present +
        cat_present + time_present + interaction_present
    )

    # Deduplicate while preserving order
    seen = set()
    final_features = [f for f in all_features if not (f in seen or seen.add(f))]

    logger.info(f"Final feature count: {len(final_features)}")

    X = df[final_features].copy()
    y = df[TARGET].copy()

    return X, y


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Master function — runs full feature engineering pipeline in order."""

    logger.info("=" * 50)
    logger.info("FEATURE ENGINEERING STARTED")
    logger.info("=" * 50)

    logger.info("Step 1: Encoding categorical + borrower categorical features")
    df = encode_categoricals(df, ALL_ENCODE_FEATURES)
    # ↑ KEY FIX: includes emp_length_, home_ownership_, verification_status_
    # These live in BORROWER_FEATURES but are categorical — must be encoded here

    logger.info("Step 2: Engineering interaction features")
    df = engineer_interaction_features(df)

    logger.info("Step 3: Selecting final feature matrix")
    X, y = select_final_features(df)

    logger.info("Step 4: Enforcing numeric dtypes — final safety check")
    X = enforce_numeric_dtypes(X)

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts(normalize=True).round(3).to_dict()}")
    logger.info("FEATURE ENGINEERING COMPLETE")

    return X, y
