import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Feature lists ────────────────────────────────────────────────────────────


LOAN_FEATURES = [
    'logfunded_amnt',       # loan size (log-transformed)
    'int_rate',             # interest rate - encodes LC's own risk assessment
    'installment',          # monthly payment amount
    'inq_last_6mths',       # credit inquiries last 6 months
    'open_acc',             # number of open credit lines
    'logtotal_acc',         # total credit lines ever (log)
    'pub_rec',              # public derogatory records
    'des_length',           # loan description length - behavioral signal
    'logpct_tl_nvr_dlq',   # % trades never delinquent (log)
    'logtot_hi_cred_lim',  # total credit limit (log)
    'revol_util',           # revolving credit utilization
    'revol_bal',            # revolving credit balance
    'delinq_2yrs',          # delinquencies in last 2 years
    'mths_since_last_delinq',   # time since last delinquency (999 = never)
    'mths_since_recent_inq',    # time since last credit inquiry
    'mths_since_recent_bc',     # time since last bankcard opened
    'acc_open_past_24mths',     # accounts opened in last 24 months
    'mort_acc',             # number of mortgage accounts
    'pub_rec_bankruptcies', # bankruptcies specifically
    'tax_liens',            # government tax liens
    'acc_now_delinq',       # accounts currently delinquent
    'num_tl_90g_dpd_24m',  # accounts 90+ days late in last 24 months
    'pct_tl_nvr_dlq',      # raw version alongside log version
    'tot_cur_bal',          # total current balance all accounts
    'avg_cur_bal',          # average current balance
    'bc_util',              # bankcard utilization
    'num_actv_bc_tl',       # active bankcard accounts
    'num_actv_rev_tl',      # active revolving accounts
]

BORROWER_FEATURES = [
    'logannual_inc',        # annual income (log)
    'logdti',               # debt-to-income ratio (log)
    'emp_length_',          # employment length (encoded)
    'home_ownership_',      # rent/own/mortgage (encoded)
    'verification_status_', # income verification status (encoded)
]

MACRO_FEATURES = [
    # State-level economic signals
    'logearnings',          # state average earnings (log)
    'logunemployment',      # state unemployment rate (log)
    'loglabor_force',       # state labor force size (log)
    'logempl_birth',        # new employer births (log) - economic dynamism
    'lognew_bus',           # new businesses (log)
    'gdpcontrib',           # GDP contribution/growth
    'loginternetuser',      # internet penetration - economic development proxy
    # National macro signals
    'inf',                  # current inflation
    'inf_6m',               # 6-month lagged inflation - delayed repayment impact
    'muni_points',          # municipal bond spread - forward-looking stress signal
    'muni_6m',              # 6-month lagged muni spread
    'FEDFUNDS',             # federal funds rate
    'FED_lag6',             # 6-month lagged fed funds rate
    'riskprem',             # risk premium
    'CPIUS',                # national CPI
    'cpi',                  # regional CPI
]

CATEGORICAL_FEATURES = [
    'grade_',               # LC risk grade A-G — single strongest predictor
    'term_',                # 36 vs 60 months
    'purpose_',             # loan purpose
    'application_type_',    # individual vs joint
    'initial_list_status_', # whole vs fractional listing
]

TIME_FEATURES = [
    'issue_year',           # handles 2016 regime shift
    'quarter',              # seasonality
    'month',                # monthly patterns
]

TARGET = 'badloan'


def validate_feature_presence(df: pd.DataFrame, features: list, group_name: str) -> list:
    """Check which features actually exist — warn about missing ones."""
    present = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    
    if missing:
        logger.warning(f"{group_name}: {len(missing)} features missing from data: {missing}")
    
    logger.info(f"{group_name}: {len(present)}/{len(features)} features present")
    return present


def encode_categoricals(df: pd.DataFrame, cat_features: list) -> pd.DataFrame:
    """
    The dataset already has pre-encoded categorical columns (grade_, term_, etc.)
    These are Stata-encoded integers. We verify they're numeric and handle
    any that slipped through as strings.
    """
    for col in cat_features:
        if col not in df.columns:
            continue
        
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            logger.info(f"Label encoding {col} — was {df[col].dtype}")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            logger.info(f"{col} already numeric ({df[col].dtype}) — no encoding needed")
    
    return df


def engineer_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features that capture combined effects.
    """
    
    # DTI × Unemployment: borrower stress amplified by economic stress
    # A high DTI borrower in a high unemployment state is doubly at risk
    if 'logdti' in df.columns and 'logunemployment' in df.columns:
        df['dti_x_unemployment'] = df['logdti'] * df['logunemployment']
        logger.info("Created: dti_x_unemployment (borrower × macro stress)")
    
    # Interest rate × Inflation: real cost of borrowing
    # High nominal rate during high inflation = borrower paying real premium
    if 'int_rate' in df.columns and 'inf' in df.columns:
        df['real_interest_rate'] = df['int_rate'] - df['inf']
        logger.info("Created: real_interest_rate (int_rate - inflation)")
    
    # Loan amount relative to income: affordability ratio
    # Captures whether the loan size is proportional to earning capacity
    if 'logfunded_amnt' in df.columns and 'logannual_inc' in df.columns:
        df['loan_to_income'] = df['logfunded_amnt'] / (df['logannual_inc'] + 1e-8)
        logger.info("Created: loan_to_income (affordability ratio)")
    
    # Credit utilization × delinquency history: behavioral risk compound
    if 'revol_util' in df.columns and 'delinq_2yrs' in df.columns:
        df['util_x_delinq'] = df['revol_util'] * (df['delinq_2yrs'] + 1)
        logger.info("Created: util_x_delinq (utilization × delinquency)")

    # Fed rate lag × loan term: rate environment when loan matures
    # 60-month loans originated during low rates may face high-rate environment at maturity
    if 'FED_lag6' in df.columns and 'term_' in df.columns:
        df['fed_x_term'] = df['FED_lag6'] * df['term_']
        logger.info("Created: fed_x_term (rate environment × loan duration)")
    
    return df


def select_final_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Assemble the final feature matrix X and target vector y.
    Returns X (features only) and y (target only) — clean separation.
    """
    # Validate each group
    loan_present = validate_feature_presence(df, LOAN_FEATURES, "LOAN")
    borrower_present = validate_feature_presence(df, BORROWER_FEATURES, "BORROWER")
    macro_present = validate_feature_presence(df, MACRO_FEATURES, "MACRO")
    cat_present = validate_feature_presence(df, CATEGORICAL_FEATURES, "CATEGORICAL")
    time_present = validate_feature_presence(df, TIME_FEATURES, "TIME")
    
    # Interaction features (these were just created)
    interaction_features = [
        'dti_x_unemployment', 'real_interest_rate', 
        'loan_to_income', 'util_x_delinq', 'fed_x_term'
    ]
    interaction_present = validate_feature_presence(df, interaction_features, "INTERACTION")
    
    all_features = (
        loan_present + borrower_present + macro_present + 
        cat_present + time_present + interaction_present
    )
    
    # Remove duplicates while preserving order
    seen = set()
    final_features = [f for f in all_features 
                      if not (f in seen or seen.add(f))]
    
    logger.info(f"Final feature count: {len(final_features)}")
    
    X = df[final_features].copy()
    y = df[TARGET].copy()
    
    return X, y


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Master function — runs full feature engineering pipeline."""
    
    logger.info("="*50)
    logger.info("FEATURE ENGINEERING STARTED")
    logger.info("="*50)
    
    logger.info("Step 1: Encoding categorical features")
    df = encode_categoricals(df, CATEGORICAL_FEATURES)
    
    logger.info("Step 2: Engineering interaction features")
    df = engineer_interaction_features(df)
    
    logger.info("Step 3: Selecting final feature matrix")
    X, y = select_final_features(df)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts(normalize=True).round(3).to_dict()}")
    logger.info("FEATURE ENGINEERING COMPLETE")
    
    return X, y
