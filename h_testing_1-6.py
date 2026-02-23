#!/usr/bin/env python3
"""
AML Hypothesis Testing & Validation
Develop detection methods and validate against actual labeled suspicious customers
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import sys
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Load transaction data with labels"""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    print(f"\nReading: {filepath}")
    print("This may take a minute for large files...")
    
    df = pd.read_csv(filepath, low_memory=False)
    
    print(f"\n✓ Loaded {len(df):,} transactions")
    print(f"  Unique customers: {df['customer_id'].nunique():,}")
    
    # Check for label column
    if 'label' not in df.columns:
        print("\nERROR: 'label' column not found!")
        print("   This script requires ground truth labels to validate hypotheses.")
        print("   Available columns:", df.columns.tolist()[:20], "...")
        return None
    
    # Check label distribution
    label_counts = df['label'].value_counts(dropna=False)
    print(f"\n  Label distribution:")
    print(label_counts)
    
    labeled_customers = df[df['label'].notna()]['customer_id'].nunique()
    suspicious_customers = df[df['label'] == 1]['customer_id'].nunique()
    normal_customers = df[df['label'] == 0]['customer_id'].nunique()
    
    print(f"\n  Customer-level labels:")
    print(f"    Labeled customers: {labeled_customers:,}")
    print(f"    Suspicious (label=1): {suspicious_customers:,}")
    print(f"    Normal (label=0): {normal_customers:,}")
    print(f"    Unlabeled: {df['customer_id'].nunique() - labeled_customers:,}")
    
    return df

def create_unified_columns(df):
    """Create unified amount and datetime columns if they don't exist"""
    print("\n" + "=" * 80)
    print("CREATING UNIFIED COLUMNS")
    print("=" * 80)
    
    # Check if unified columns already exist
    if 'amount' not in df.columns:
        print("\nCreating unified 'amount' column...")
        
        # List of possible amount columns (check for both old and new naming conventions)
        amount_cols_old = ['amount_cad.y', 'amount_cad.x', 'amount_cad.x.x', 
                           'amount_cad.y.y', 'amount_cad.x.x.x', 'amount_cad.y.y.y', 'amount_cad']
        
        amount_cols_new = ['card_amount', 'abm_amount', 'cheque_amount', 
                           'eft_amount', 'emt_amount', 'westernunion_amount', 'wire_amount']
        
        # Try new naming first, then fall back to old
        amount_cols = amount_cols_new + amount_cols_old
        
        df['amount'] = None
        for col in amount_cols:
            if col in df.columns:
                df['amount'] = df['amount'].combine_first(df[col])
        
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        print(f"  ✓ Created 'amount': {df['amount'].notna().sum():,} non-null values")
    else:
        print(f"\n✓ 'amount' column already exists: {df['amount'].notna().sum():,} non-null values")
    
    if 'datetime' not in df.columns and 'transaction_datetime' not in df.columns:
        print("\nCreating unified 'datetime' column...")
        
        datetime_cols_old = ['transaction_datetime.y', 'transaction_datetime.x', 'transaction_datetime.x.x',
                             'transaction_datetime.y.y', 'transaction_datetime.x.x.x', 
                             'transaction_datetime.y.y.y', 'transaction_datetime']
        
        datetime_cols_new = ['card_datetime', 'abm_datetime', 'cheque_datetime',
                             'eft_datetime', 'emt_datetime', 'westernunion_datetime', 'wire_datetime']
        
        # Try new naming first, then fall back to old
        datetime_cols = datetime_cols_new + datetime_cols_old
        
        df['datetime'] = None
        for col in datetime_cols:
            if col in df.columns:
                df['datetime'] = df['datetime'].combine_first(df[col])
        
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        print(f"  ✓ Created 'datetime': {df['datetime'].notna().sum():,} non-null values")
    elif 'transaction_datetime' in df.columns and 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['transaction_datetime'], errors='coerce')
        print(f"\n✓ Created 'datetime' from 'transaction_datetime': {df['datetime'].notna().sum():,} non-null values")
    else:
        print(f"\n✓ 'datetime' column already exists: {df['datetime'].notna().sum():,} non-null values")
    
    # Create amount_abs
    if 'amount_abs' not in df.columns:
        df['amount_abs'] = df['amount'].abs()
        print(f"✓ Created 'amount_abs': {df['amount_abs'].notna().sum():,} non-null values")
    
    print("\n✓ Unified columns ready")
    
    return df

def aggregate_customer_data(df):
    """Aggregate transaction data to customer level"""
    print("\n" + "=" * 80)
    print("AGGREGATING TO CUSTOMER LEVEL")
    print("=" * 80)
    
    print("\nCalculating customer-level features...")
    
    # Filter to valid transactions
    df_valid = df[df['amount'].notna() & df['datetime'].notna()].copy()
    
    # Group by customer
    customer_agg = df_valid.groupby('customer_id').agg({
        'amount': ['count', 'sum', 'mean', 'median', 'std', 'min', 'max'],
        'amount_abs': ['mean', 'median'],
        'label': 'first',  # Get the label (same for all transactions of a customer)
    }).reset_index()
    
    # Flatten column names
    customer_agg.columns = ['customer_id', 'txn_count', 'total_amount', 'avg_amount', 
                           'median_amount', 'std_amount', 'min_amount', 'max_amount',
                           'avg_amount_abs', 'median_amount_abs', 'label']
    
    print(f"✓ Aggregated {len(customer_agg):,} customers")
    
    return customer_agg, df_valid

def hypothesis_1_high_transaction_volume(customer_agg):
    """
    HYPOTHESIS 1: Suspicious customers have unusually high transaction volumes
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 1: HIGH TRANSACTION VOLUME")
    print("=" * 80)
    
    print("\nTheory: Money launderers need to move large volumes of money")
    print("Prediction: Suspicious customers will have more transactions than normal")
    
    # Calculate statistics
    suspicious = customer_agg[customer_agg['label'] == 1]
    normal = customer_agg[customer_agg['label'] == 0]
    
    print(f"\nTransaction count statistics:")
    print(f"  Suspicious customers (n={len(suspicious)}):")
    print(f"    Mean: {suspicious['txn_count'].mean():.1f} transactions")
    print(f"    Median: {suspicious['txn_count'].median():.1f} transactions")
    
    print(f"  Normal customers (n={len(normal)}):")
    print(f"    Mean: {normal['txn_count'].mean():.1f} transactions")
    print(f"    Median: {normal['txn_count'].median():.1f} transactions")
    
    # Statistical test
    if len(suspicious) > 0 and len(normal) > 0:
        t_stat, p_value = stats.mannwhitneyu(suspicious['txn_count'], normal['txn_count'], alternative='two-sided')
        print(f"\n  Mann-Whitney U test: p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  ✓ SIGNIFICANT DIFFERENCE (p < 0.05)")
            
            if suspicious['txn_count'].median() > normal['txn_count'].median():
                print(f"  → Suspicious customers have MORE transactions")
                conclusion = "SUPPORTED"
            else:
                print(f"  → Suspicious customers have FEWER transactions")
                conclusion = "OPPOSITE"
        else:
            print(f"  ✗ No significant difference (p >= 0.05)")
            conclusion = "NOT SUPPORTED"
    else:
        conclusion = "INSUFFICIENT DATA"
    
    # Set threshold and calculate detection rate
    threshold = normal['txn_count'].quantile(0.90)  # Top 10%
    customer_agg['flag_high_volume'] = customer_agg['txn_count'] > threshold
    
    detected = customer_agg[customer_agg['flag_high_volume'] == True]
    true_positives = detected[detected['label'] == 1]
    
    if len(suspicious) > 0:
        recall = len(true_positives) / len(suspicious) * 100
        print(f"\n  Detection with threshold (>{threshold:.0f} transactions):")
        print(f"    Flagged customers: {len(detected):,}")
        print(f"    Caught suspicious: {len(true_positives)}/{len(suspicious)} ({recall:.1f}%)")
    
    return {
        'hypothesis': 'High Transaction Volume',
        'conclusion': conclusion,
        'suspicious_median': suspicious['txn_count'].median() if len(suspicious) > 0 else None,
        'normal_median': normal['txn_count'].median() if len(normal) > 0 else None,
        'p_value': p_value if len(suspicious) > 0 and len(normal) > 0 else None,
        'recall': recall if len(suspicious) > 0 else 0
    }

def hypothesis_2_large_transactions(customer_agg):
    """
    HYPOTHESIS 2: Suspicious customers make larger-than-average transactions
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 2: LARGE TRANSACTION AMOUNTS")
    print("=" * 80)
    
    print("\nTheory: Money laundering involves moving large sums")
    print("Prediction: Suspicious customers have higher average transaction amounts")
    
    suspicious = customer_agg[customer_agg['label'] == 1]
    normal = customer_agg[customer_agg['label'] == 0]
    
    print(f"\nAverage transaction amount:")
    print(f"  Suspicious customers:")
    print(f"    Mean: ${suspicious['avg_amount'].mean():,.2f}")
    print(f"    Median: ${suspicious['median_amount'].median():,.2f}")
    
    print(f"  Normal customers:")
    print(f"    Mean: ${normal['avg_amount'].mean():,.2f}")
    print(f"    Median: ${normal['median_amount'].median():,.2f}")
    
    # Statistical test
    if len(suspicious) > 0 and len(normal) > 0:
        t_stat, p_value = stats.mannwhitneyu(suspicious['avg_amount'], normal['avg_amount'], alternative='two-sided')
        print(f"\n  Mann-Whitney U test: p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  ✓ SIGNIFICANT DIFFERENCE (p < 0.05)")
            if suspicious['avg_amount'].median() > normal['avg_amount'].median():
                print(f"  → Suspicious customers have LARGER average amounts")
                conclusion = "SUPPORTED"
            else:
                print(f"  → Suspicious customers have SMALLER average amounts")
                conclusion = "OPPOSITE"
        else:
            print(f"  ✗ No significant difference")
            conclusion = "NOT SUPPORTED"
    else:
        conclusion = "INSUFFICIENT DATA"
    
    # Detection
    threshold = normal['avg_amount'].quantile(0.90)
    customer_agg['flag_large_amounts'] = customer_agg['avg_amount'] > threshold
    
    detected = customer_agg[customer_agg['flag_large_amounts'] == True]
    true_positives = detected[detected['label'] == 1]
    
    if len(suspicious) > 0:
        recall = len(true_positives) / len(suspicious) * 100
        print(f"\n  Detection with threshold (>${threshold:,.2f} avg):")
        print(f"    Flagged: {len(detected):,}")
        print(f"    Caught: {len(true_positives)}/{len(suspicious)} ({recall:.1f}%)")
    
    return {
        'hypothesis': 'Large Transaction Amounts',
        'conclusion': conclusion,
        'suspicious_median': suspicious['avg_amount'].median() if len(suspicious) > 0 else None,
        'normal_median': normal['avg_amount'].median() if len(normal) > 0 else None,
        'p_value': p_value if len(suspicious) > 0 and len(normal) > 0 else None,
        'recall': recall if len(suspicious) > 0 else 0
    }

def hypothesis_3_total_value(customer_agg):
    """
    HYPOTHESIS 3: Suspicious customers move large TOTAL amounts
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 3: HIGH TOTAL TRANSACTION VALUE")
    print("=" * 80)
    
    print("\nTheory: Money launderers move large total sums over time")
    print("Prediction: Suspicious customers have higher total transaction values")
    
    suspicious = customer_agg[customer_agg['label'] == 1]
    normal = customer_agg[customer_agg['label'] == 0]
    
    print(f"\nTotal transaction value:")
    print(f"  Suspicious customers:")
    print(f"    Mean: ${suspicious['total_amount'].mean():,.2f}")
    print(f"    Median: ${suspicious['total_amount'].median():,.2f}")
    
    print(f"  Normal customers:")
    print(f"    Mean: ${normal['total_amount'].mean():,.2f}")
    print(f"    Median: ${normal['total_amount'].median():,.2f}")
    
    if len(suspicious) > 0 and len(normal) > 0:
        t_stat, p_value = stats.mannwhitneyu(suspicious['total_amount'], normal['total_amount'], alternative='two-sided')
        print(f"\n  Mann-Whitney U test: p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  ✓ SIGNIFICANT DIFFERENCE")
            if suspicious['total_amount'].median() > normal['total_amount'].median():
                conclusion = "SUPPORTED"
                print(f"  → Suspicious customers move MORE total value")
            else:
                conclusion = "OPPOSITE"
                print(f"  → Suspicious customers move LESS total value")
        else:
            conclusion = "NOT SUPPORTED"
            print(f"  ✗ No significant difference")
    else:
        conclusion = "INSUFFICIENT DATA"
    
    threshold = normal['total_amount'].quantile(0.90)
    customer_agg['flag_high_total'] = customer_agg['total_amount'] > threshold
    
    detected = customer_agg[customer_agg['flag_high_total'] == True]
    true_positives = detected[detected['label'] == 1]
    
    if len(suspicious) > 0:
        recall = len(true_positives) / len(suspicious) * 100
        print(f"\n  Detection with threshold (>${threshold:,.2f} total):")
        print(f"    Flagged: {len(detected):,}")
        print(f"    Caught: {len(true_positives)}/{len(suspicious)} ({recall:.1f}%)")
    
    return {
        'hypothesis': 'High Total Value',
        'conclusion': conclusion,
        'suspicious_median': suspicious['total_amount'].median() if len(suspicious) > 0 else None,
        'normal_median': normal['total_amount'].median() if len(normal) > 0 else None,
        'p_value': p_value if len(suspicious) > 0 and len(normal) > 0 else None,
        'recall': recall if len(suspicious) > 0 else 0
    }

def hypothesis_4_round_numbers(df_valid, customer_agg):
    """
    HYPOTHESIS 4: Suspicious customers use more round numbers
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 4: ROUND NUMBER TRANSACTIONS")
    print("=" * 80)
    
    print("\nTheory: Criminals use round numbers (testing, structuring)")
    print("Prediction: Suspicious customers have more round-number transactions")
    
    # Calculate round number percentage per customer
    df_valid['is_round'] = (df_valid['amount_abs'] % 100 == 0) & (df_valid['amount_abs'] >= 100)
    
    round_pct = df_valid.groupby('customer_id').agg({
        'is_round': 'mean',
        'label': 'first'
    }).reset_index()
    round_pct.columns = ['customer_id', 'round_pct', 'label']
    round_pct['round_pct'] = round_pct['round_pct'] * 100
    
    suspicious = round_pct[round_pct['label'] == 1]
    normal = round_pct[round_pct['label'] == 0]
    
    print(f"\n% of transactions that are round numbers:")
    print(f"  Suspicious customers:")
    print(f"    Mean: {suspicious['round_pct'].mean():.1f}%")
    print(f"    Median: {suspicious['round_pct'].median():.1f}%")
    
    print(f"  Normal customers:")
    print(f"    Mean: {normal['round_pct'].mean():.1f}%")
    print(f"    Median: {normal['round_pct'].median():.1f}%")
    
    if len(suspicious) > 0 and len(normal) > 0:
        t_stat, p_value = stats.mannwhitneyu(suspicious['round_pct'], normal['round_pct'], alternative='two-sided')
        print(f"\n  Mann-Whitney U test: p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  ✓ SIGNIFICANT DIFFERENCE")
            if suspicious['round_pct'].median() > normal['round_pct'].median():
                conclusion = "SUPPORTED"
                print(f"  → Suspicious customers use MORE round numbers")
            else:
                conclusion = "OPPOSITE"
                print(f"  → Suspicious customers use FEWER round numbers")
        else:
            conclusion = "NOT SUPPORTED"
            print(f"  ✗ No significant difference")
    else:
        conclusion = "INSUFFICIENT DATA"
    
    # Merge with customer_agg
    customer_agg = customer_agg.merge(round_pct[['customer_id', 'round_pct']], on='customer_id', how='left')
    
    threshold = normal['round_pct'].quantile(0.90)
    customer_agg['flag_round_numbers'] = customer_agg['round_pct'] > threshold
    
    detected = customer_agg[customer_agg['flag_round_numbers'] == True]
    true_positives = detected[detected['label'] == 1]
    
    if len(suspicious) > 0:
        recall = len(true_positives) / len(suspicious) * 100
        print(f"\n  Detection with threshold (>{threshold:.1f}% round):")
        print(f"    Flagged: {len(detected):,}")
        print(f"    Caught: {len(true_positives)}/{len(suspicious)} ({recall:.1f}%)")
    
    return customer_agg, {
        'hypothesis': 'Round Number Abuse',
        'conclusion': conclusion,
        'suspicious_median': suspicious['round_pct'].median() if len(suspicious) > 0 else None,
        'normal_median': normal['round_pct'].median() if len(normal) > 0 else None,
        'p_value': p_value if len(suspicious) > 0 and len(normal) > 0 else None,
        'recall': recall if len(suspicious) > 0 else 0
    }

def hypothesis_5_structuring(df_valid, customer_agg):
    """
    HYPOTHESIS 5: Suspicious customers structure (transactions near $10K threshold)
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 5: STRUCTURING (NEAR $10K THRESHOLD)")
    print("=" * 80)
    
    print("\nTheory: Criminals avoid $10K reporting by staying just below")
    print("Prediction: Suspicious customers have more $9K-$10K transactions")
    
    # Calculate structuring percentage
    df_valid['is_structuring'] = (df_valid['amount_abs'] >= 9000) & (df_valid['amount_abs'] < 10000)
    
    struct_pct = df_valid.groupby('customer_id').agg({
        'is_structuring': 'mean',
        'label': 'first'
    }).reset_index()
    struct_pct.columns = ['customer_id', 'structuring_pct', 'label']
    struct_pct['structuring_pct'] = struct_pct['structuring_pct'] * 100
    
    suspicious = struct_pct[struct_pct['label'] == 1]
    normal = struct_pct[struct_pct['label'] == 0]
    
    print(f"\n% of transactions in $9K-$10K range:")
    print(f"  Suspicious customers:")
    print(f"    Mean: {suspicious['structuring_pct'].mean():.2f}%")
    print(f"    Median: {suspicious['structuring_pct'].median():.2f}%")
    print(f"    Any structuring: {(suspicious['structuring_pct'] > 0).sum()}/{len(suspicious)}")
    
    print(f"  Normal customers:")
    print(f"    Mean: {normal['structuring_pct'].mean():.2f}%")
    print(f"    Median: {normal['structuring_pct'].median():.2f}%")
    print(f"    Any structuring: {(normal['structuring_pct'] > 0).sum()}/{len(normal)}")
    
    if len(suspicious) > 0 and len(normal) > 0:
        # Use simple comparison since structuring_pct is often 0
        susp_with_struct = (suspicious['structuring_pct'] > 0).sum() / len(suspicious)
        norm_with_struct = (normal['structuring_pct'] > 0).sum() / len(normal)
        
        print(f"\n  Comparison:")
        print(f"    Suspicious with ANY structuring: {susp_with_struct*100:.1f}%")
        print(f"    Normal with ANY structuring: {norm_with_struct*100:.1f}%")
        
        if susp_with_struct > norm_with_struct * 1.5:
            conclusion = "SUPPORTED"
            print(f"  ✓ Suspicious customers are {susp_with_struct/norm_with_struct:.1f}x more likely to structure")
        elif norm_with_struct > susp_with_struct * 1.5:
            conclusion = "OPPOSITE"
            print(f"  → Normal customers structure more (unexpected!)")
        else:
            conclusion = "NOT SUPPORTED"
            print(f"  ✗ Similar rates")
    else:
        conclusion = "INSUFFICIENT DATA"
    
    customer_agg = customer_agg.merge(struct_pct[['customer_id', 'structuring_pct']], on='customer_id', how='left')
    customer_agg['flag_structuring'] = customer_agg['structuring_pct'] > 0
    
    detected = customer_agg[customer_agg['flag_structuring'] == True]
    true_positives = detected[detected['label'] == 1]
    
    if len(suspicious) > 0:
        recall = len(true_positives) / len(suspicious) * 100
        print(f"\n  Detection (ANY structuring transactions):")
        print(f"    Flagged: {len(detected):,}")
        print(f"    Caught: {len(true_positives)}/{len(suspicious)} ({recall:.1f}%)")
    
    return customer_agg, {
        'hypothesis': 'Structuring ($9K-$10K)',
        'conclusion': conclusion,
        'suspicious_pct_with_pattern': susp_with_struct * 100 if len(suspicious) > 0 else None,
        'normal_pct_with_pattern': norm_with_struct * 100 if len(normal) > 0 else None,
        'p_value': None,
        'recall': recall if len(suspicious) > 0 else 0
    }

def hypothesis_6_transaction_variance(customer_agg):
    """
    HYPOTHESIS 6: Suspicious customers have erratic transaction patterns (high variance)
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 6: ERRATIC TRANSACTION PATTERNS (HIGH VARIANCE)")
    print("=" * 80)
    
    print("\nTheory: Criminals have inconsistent transaction patterns")
    print("Prediction: Suspicious customers have higher standard deviation in amounts")
    
    suspicious = customer_agg[customer_agg['label'] == 1]
    normal = customer_agg[customer_agg['label'] == 0]
    
    # Calculate coefficient of variation (std/mean)
    customer_agg['coef_variation'] = customer_agg['std_amount'] / customer_agg['avg_amount']
    
    suspicious_cv = customer_agg[customer_agg['label'] == 1]
    normal_cv = customer_agg[customer_agg['label'] == 0]
    
    print(f"\nCoefficient of Variation (std/mean):")
    print(f"  Suspicious customers:")
    print(f"    Mean CV: {suspicious_cv['coef_variation'].mean():.2f}")
    print(f"    Median CV: {suspicious_cv['coef_variation'].median():.2f}")
    
    print(f"  Normal customers:")
    print(f"    Mean CV: {normal_cv['coef_variation'].mean():.2f}")
    print(f"    Median CV: {normal_cv['coef_variation'].median():.2f}")
    
    if len(suspicious_cv) > 0 and len(normal_cv) > 0:
        # Remove infinite/NaN values
        susp_clean = suspicious_cv['coef_variation'].replace([np.inf, -np.inf], np.nan).dropna()
        norm_clean = normal_cv['coef_variation'].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(susp_clean) > 0 and len(norm_clean) > 0:
            t_stat, p_value = stats.mannwhitneyu(susp_clean, norm_clean, alternative='two-sided')
            print(f"\n  Mann-Whitney U test: p-value = {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"  ✓ SIGNIFICANT DIFFERENCE")
                if susp_clean.median() > norm_clean.median():
                    conclusion = "SUPPORTED"
                    print(f"  → Suspicious customers are MORE erratic")
                else:
                    conclusion = "OPPOSITE"
                    print(f"  → Suspicious customers are LESS erratic")
            else:
                conclusion = "NOT SUPPORTED"
                print(f"  ✗ No significant difference")
        else:
            conclusion = "INSUFFICIENT DATA"
            p_value = None
    else:
        conclusion = "INSUFFICIENT DATA"
        p_value = None
    
    threshold = normal_cv['coef_variation'].quantile(0.90)
    customer_agg['flag_high_variance'] = customer_agg['coef_variation'] > threshold
    
    detected = customer_agg[customer_agg['flag_high_variance'] == True]
    true_positives = detected[detected['label'] == 1]
    
    if len(suspicious) > 0:
        recall = len(true_positives) / len(suspicious) * 100
        print(f"\n  Detection with threshold (CV > {threshold:.2f}):")
        print(f"    Flagged: {len(detected):,}")
        print(f"    Caught: {len(true_positives)}/{len(suspicious)} ({recall:.1f}%)")
    
    return customer_agg, {
        'hypothesis': 'Erratic Patterns (High Variance)',
        'conclusion': conclusion,
        'suspicious_median': susp_clean.median() if len(suspicious_cv) > 0 else None,
        'normal_median': norm_clean.median() if len(normal_cv) > 0 else None,
        'p_value': p_value,
        'recall': recall if len(suspicious) > 0 else 0
    }

def combined_detection(customer_agg):
    """
    Test combined detection: flag customers who meet multiple criteria
    """
    print("\n" + "=" * 80)
    print("COMBINED DETECTION: MULTIPLE FLAGS")
    print("=" * 80)
    
    print("\nTheory: Real money launderers exhibit MULTIPLE red flags")
    print("Method: Flag customers with 2+ suspicious indicators")
    
    # Count flags per customer
    flag_cols = ['flag_high_volume', 'flag_large_amounts', 'flag_high_total', 
                 'flag_round_numbers', 'flag_structuring', 'flag_high_variance']
    
    customer_agg['total_flags'] = customer_agg[flag_cols].sum(axis=1)
    
    # Distribution of flags
    print(f"\nFlag distribution:")
    print(customer_agg['total_flags'].value_counts().sort_index())
    
    suspicious = customer_agg[customer_agg['label'] == 1]
    normal = customer_agg[customer_agg['label'] == 0]
    
    print(f"\nAverage flags per customer:")
    print(f"  Suspicious: {suspicious['total_flags'].mean():.2f} flags")
    print(f"  Normal: {normal['total_flags'].mean():.2f} flags")
    
    # Test different thresholds
    print(f"\nDetection performance at different thresholds:")
    
    results = []
    for threshold in [1, 2, 3, 4]:
        flagged = customer_agg[customer_agg['total_flags'] >= threshold]
        tp = flagged[flagged['label'] == 1]
        fp = flagged[flagged['label'] == 0]
        
        if len(suspicious) > 0:
            recall = len(tp) / len(suspicious) * 100
        else:
            recall = 0
        
        if len(flagged) > 0:
            precision = len(tp) / len(flagged) * 100
        else:
            precision = 0
        
        print(f"\n  Threshold: {threshold}+ flags")
        print(f"    Flagged: {len(flagged):,} customers")
        print(f"    True Positives: {len(tp)} (caught {recall:.1f}% of suspicious)")
        print(f"    False Positives: {len(fp)}")
        print(f"    Precision: {precision:.1f}%")
        print(f"    Recall: {recall:.1f}%")
        
        results.append({
            'threshold': threshold,
            'flagged': len(flagged),
            'true_positives': len(tp),
            'false_positives': len(fp),
            'precision': precision,
            'recall': recall
        })
    
    return customer_agg, results

def generate_summary(results_list, combined_results):
    """Generate final summary report"""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY: HYPOTHESIS VALIDATION")
    print("=" * 80)
    
    print("\nHypothesis Testing Results:")
    print("-" * 80)
    
    for i, result in enumerate(results_list, 1):
        print(f"\n{i}. {result['hypothesis']}")
        print(f"   Conclusion: {result['conclusion']}")
        if result.get('p_value'):
            print(f"   Statistical significance: p = {result['p_value']:.4f}")
        if result.get('recall'):
            print(f"   Detection recall: {result['recall']:.1f}%")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED DETECTION STRATEGY")
    print("=" * 80)
    
    # Find best combined threshold
    best = max(combined_results, key=lambda x: x['recall'] if x['precision'] > 5 else 0)
    
    print(f"\nBest approach: Flag customers with {best['threshold']}+ red flags")
    print(f"  This catches {best['recall']:.1f}% of suspicious customers")
    print(f"  With {best['precision']:.1f}% precision")
    print(f"  Total flagged: {best['flagged']} customers to investigate")
    
    print("\n" + "=" * 80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python aml_hypothesis_validation.py <data_file.csv>")
        print("\nExample:")
        print("  python aml_hypothesis_validation.py joined_data_output.csv")
        print("\nThis script requires:")
        print("  - A 'label' column (1=suspicious, 0=normal)")
        print("  - Transaction data with customer_id, amounts, dates")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Load data
    df = load_data(filepath)
    if df is None:
        sys.exit(1)
    
    # Create unified columns
    df = create_unified_columns(df)
    
    # Aggregate to customer level
    customer_agg, df_valid = aggregate_customer_data(df)
    
    # Test hypotheses
    results = []
    
    result1 = hypothesis_1_high_transaction_volume(customer_agg)
    results.append(result1)
    
    result2 = hypothesis_2_large_transactions(customer_agg)
    results.append(result2)
    
    result3 = hypothesis_3_total_value(customer_agg)
    results.append(result3)
    
    customer_agg, result4 = hypothesis_4_round_numbers(df_valid, customer_agg)
    results.append(result4)
    
    customer_agg, result5 = hypothesis_5_structuring(df_valid, customer_agg)
    results.append(result5)
    
    customer_agg, result6 = hypothesis_6_transaction_variance(customer_agg)
    results.append(result6)
    
    # Combined detection
    customer_agg, combined_results = combined_detection(customer_agg)
    
    # Generate summary
    generate_summary(results, combined_results)
    
    # Save results
    output_file = 'customer_risk_scores.csv'
    customer_agg.to_csv(output_file, index=False)
    print(f"\n✓ Saved customer risk scores to: {output_file}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
