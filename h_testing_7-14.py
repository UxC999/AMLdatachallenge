#!/usr/bin/env python3
"""
Enhanced AML Hypothesis Testing - Part 2
Advanced detection methods using merchant categories, geography, timing, and more
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
import sys
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Load transaction data"""
    print("=" * 80)
    print("ENHANCED AML HYPOTHESIS TESTING - PART 2")
    print("=" * 80)
    
    print(f"\nReading: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    
    print(f"\n✓ Loaded {len(df):,} transactions")
    print(f"  Unique customers: {df['customer_id'].nunique():,}")
    
    # Check for label column
    if 'label' not in df.columns:
        print("\nERROR: 'label' column not found!")
        return None
    
    labeled_customers = df[df['label'].notna()]['customer_id'].nunique()
    suspicious_customers = df[df['label'] == 1]['customer_id'].nunique()
    
    print(f"  Labeled customers: {labeled_customers:,}")
    print(f"  Suspicious (label=1): {suspicious_customers}")
    
    return df

def create_unified_columns(df):
    """Create unified columns"""
    print("\n" + "=" * 80)
    print("PREPARING DATA")
    print("=" * 80)
    
    # Amount columns
    amount_cols = ['card_amount', 'abm_amount', 'cheque_amount', 
                   'eft_amount', 'emt_amount', 'westernunion_amount', 'wire_amount']
    
    if 'amount' not in df.columns:
        df['amount'] = None
        for col in amount_cols:
            if col in df.columns:
                df['amount'] = df['amount'].combine_first(df[col])
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Datetime columns
    datetime_cols = ['card_datetime', 'abm_datetime', 'cheque_datetime',
                     'eft_datetime', 'emt_datetime', 'westernunion_datetime', 'wire_datetime']
    
    if 'datetime' not in df.columns:
        df['datetime'] = None
        for col in datetime_cols:
            if col in df.columns:
                df['datetime'] = df['datetime'].combine_first(df[col])
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    df['amount_abs'] = df['amount'].abs()
    
    print(f"✓ Amount: {df['amount'].notna().sum():,} non-null")
    print(f"✓ Datetime: {df['datetime'].notna().sum():,} non-null")
    
    return df

def hypothesis_7_merchant_category_risk(df):
    """
    HYPOTHESIS 7: Suspicious customers use high-risk merchant categories
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 7: HIGH-RISK MERCHANT CATEGORIES")
    print("=" * 80)
    
    print("\nTheory: Money launderers use specific merchant types")
    print("Prediction: Suspicious customers use gambling, money transfer, crypto, etc.")
    
    if 'card_merchant_category' not in df.columns:
        print("\ncard_merchant_category column not found - SKIPPING")
        return None
    
    # Filter to card transactions only
    card_df = df[df['card_merchant_category'].notna()].copy()
    
    if len(card_df) == 0:
        print("\nNo merchant category data - SKIPPING")
        return None
    
    print(f"\nAnalyzing {len(card_df):,} card transactions")
    
    # High-risk merchant category codes (common AML red flags)
    # These are illustrative - adjust based on actual category codes in your data
    HIGH_RISK_CATEGORIES = {
        '7995',  # Gambling
        '6051',  # Crypto/Digital Currencies  
        '4829',  # Money Transfer
        '6211',  # Securities Brokers
        '7800',  # Government Lotteries
        '7801',  # Internet Gambling
        '7802',  # Horse/Dog Racing
    }
    
    # Calculate % of high-risk transactions per customer
    card_df['is_high_risk_merchant'] = card_df['card_merchant_category'].astype(str).isin(HIGH_RISK_CATEGORIES)
    
    merchant_risk = card_df.groupby('customer_id').agg({
        'is_high_risk_merchant': 'mean',
        'label': 'first'
    }).reset_index()
    merchant_risk.columns = ['customer_id', 'high_risk_merchant_pct', 'label']
    merchant_risk['high_risk_merchant_pct'] *= 100
    
    suspicious = merchant_risk[merchant_risk['label'] == 1]
    normal = merchant_risk[merchant_risk['label'] == 0]
    
    print(f"\n% of transactions in high-risk merchant categories:")
    print(f"  Suspicious customers (n={len(suspicious)}):")
    print(f"    Mean: {suspicious['high_risk_merchant_pct'].mean():.2f}%")
    print(f"    With ANY high-risk: {(suspicious['high_risk_merchant_pct'] > 0).sum()}/{len(suspicious)}")
    
    print(f"  Normal customers (n={len(normal)}):")
    print(f"    Mean: {normal['high_risk_merchant_pct'].mean():.2f}%")
    print(f"    With ANY high-risk: {(normal['high_risk_merchant_pct'] > 0).sum()}/{len(normal)}")
    
    # Top merchant categories for suspicious vs normal
    print(f"\nTop merchant categories for SUSPICIOUS customers:")
    susp_txns = card_df[card_df['label'] == 1]
    if len(susp_txns) > 0:
        print(susp_txns['card_merchant_category'].value_counts().head(10))
    
    print(f"\nTop merchant categories for NORMAL customers:")
    norm_txns = card_df[card_df['label'] == 0]
    if len(norm_txns) > 0:
        print(norm_txns['card_merchant_category'].value_counts().head(10))
    
    # Statistical comparison
    if len(suspicious) > 0 and len(normal) > 0:
        susp_with_risk = (suspicious['high_risk_merchant_pct'] > 0).sum() / len(suspicious)
        norm_with_risk = (normal['high_risk_merchant_pct'] > 0).sum() / len(normal)
        
        if susp_with_risk > norm_with_risk * 1.5:
            conclusion = "SUPPORTED"
            print(f"\n✓ Suspicious customers {susp_with_risk/norm_with_risk:.1f}x more likely to use high-risk merchants")
        else:
            conclusion = "NOT SUPPORTED"
            print(f"\n✗ No significant difference in high-risk merchant usage")
    else:
        conclusion = "INSUFFICIENT DATA"
    
    return {
        'hypothesis': 'High-Risk Merchant Categories',
        'conclusion': conclusion,
        'data': merchant_risk
    }

def hypothesis_8_ecommerce_patterns(df):
    """
    HYPOTHESIS 8: Suspicious customers use more e-commerce (card-not-present fraud)
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 8: E-COMMERCE VS IN-PERSON TRANSACTIONS")
    print("=" * 80)
    
    print("\nTheory: Card-not-present fraud is harder to trace")
    print("Prediction: Suspicious customers use more online/e-commerce transactions")
    
    if 'card_ecommerce_indicator' not in df.columns:
        print("\ncard_ecommerce_indicator column not found - SKIPPING")
        return None
    
    card_df = df[df['card_ecommerce_indicator'].notna()].copy()
    
    if len(card_df) == 0:
        print("\nNo e-commerce indicator data - SKIPPING")
        return None
    
    print(f"\nAnalyzing {len(card_df):,} card transactions")
    
    # Calculate % e-commerce per customer
    card_df['is_ecommerce'] = card_df['card_ecommerce_indicator'].astype(str).str.upper().isin(['Y', 'YES', '1', 'TRUE'])
    
    ecom_pct = card_df.groupby('customer_id').agg({
        'is_ecommerce': 'mean',
        'label': 'first'
    }).reset_index()
    ecom_pct.columns = ['customer_id', 'ecommerce_pct', 'label']
    ecom_pct['ecommerce_pct'] *= 100
    
    suspicious = ecom_pct[ecom_pct['label'] == 1]
    normal = ecom_pct[ecom_pct['label'] == 0]
    
    print(f"\n% of transactions that are e-commerce:")
    print(f"  Suspicious customers:")
    print(f"    Mean: {suspicious['ecommerce_pct'].mean():.1f}%")
    print(f"    Median: {suspicious['ecommerce_pct'].median():.1f}%")
    
    print(f"  Normal customers:")
    print(f"    Mean: {normal['ecommerce_pct'].mean():.1f}%")
    print(f"    Median: {normal['ecommerce_pct'].median():.1f}%")
    
    if len(suspicious) > 0 and len(normal) > 0:
        t_stat, p_value = stats.mannwhitneyu(suspicious['ecommerce_pct'], normal['ecommerce_pct'], alternative='two-sided')
        print(f"\n  Mann-Whitney U test: p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            if suspicious['ecommerce_pct'].median() > normal['ecommerce_pct'].median():
                conclusion = "SUPPORTED"
                print(f"  ✓ Suspicious customers use MORE e-commerce")
            else:
                conclusion = "OPPOSITE"
                print(f"  → Suspicious customers use LESS e-commerce")
        else:
            conclusion = "NOT SUPPORTED"
            print(f"  ✗ No significant difference")
    else:
        conclusion = "INSUFFICIENT DATA"
        p_value = None
    
    return {
        'hypothesis': 'E-commerce Preference',
        'conclusion': conclusion,
        'p_value': p_value,
        'data': ecom_pct
    }

def hypothesis_9_cash_preference(df):
    """
    HYPOTHESIS 9: Suspicious customers withdraw more cash (layering technique)
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 9: CASH WITHDRAWAL PATTERNS")
    print("=" * 80)
    
    print("\nTheory: Money launderers convert to cash to break the paper trail")
    print("Prediction: Suspicious customers have higher % of cash withdrawals")
    
    if 'abm_cash_indicator' not in df.columns:
        print("\nabm_cash_indicator column not found - SKIPPING")
        return None
    
    abm_df = df[df['abm_cash_indicator'].notna()].copy()
    
    if len(abm_df) == 0:
        print("\nNo ATM cash indicator data - SKIPPING")
        return None
    
    print(f"\nAnalyzing {len(abm_df):,} ATM transactions")
    
    # Calculate % cash withdrawals
    abm_df['is_cash'] = abm_df['abm_cash_indicator'].astype(str).str.upper().isin(['Y', 'YES', '1', 'TRUE'])
    
    cash_pct = abm_df.groupby('customer_id').agg({
        'is_cash': 'mean',
        'label': 'first'
    }).reset_index()
    cash_pct.columns = ['customer_id', 'cash_pct', 'label']
    cash_pct['cash_pct'] *= 100
    
    suspicious = cash_pct[cash_pct['label'] == 1]
    normal = cash_pct[cash_pct['label'] == 0]
    
    print(f"\n% of ATM transactions that are cash withdrawals:")
    print(f"  Suspicious customers (n={len(suspicious)}):")
    if len(suspicious) > 0:
        print(f"    Mean: {suspicious['cash_pct'].mean():.1f}%")
        print(f"    Median: {suspicious['cash_pct'].median():.1f}%")
    
    print(f"  Normal customers (n={len(normal)}):")
    if len(normal) > 0:
        print(f"    Mean: {normal['cash_pct'].mean():.1f}%")
        print(f"    Median: {normal['cash_pct'].median():.1f}%")
    
    if len(suspicious) > 0 and len(normal) > 0:
        t_stat, p_value = stats.mannwhitneyu(suspicious['cash_pct'], normal['cash_pct'], alternative='two-sided')
        print(f"\n  Mann-Whitney U test: p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            if suspicious['cash_pct'].median() > normal['cash_pct'].median():
                conclusion = "SUPPORTED"
                print(f"  ✓ Suspicious customers withdraw MORE cash")
            else:
                conclusion = "OPPOSITE"
        else:
            conclusion = "NOT SUPPORTED"
            print(f"  ✗ No significant difference")
    else:
        conclusion = "INSUFFICIENT DATA"
        p_value = None
    
    return {
        'hypothesis': 'Cash Withdrawal Preference',
        'conclusion': conclusion,
        'p_value': p_value,
        'data': cash_pct
    }

def hypothesis_10_geographic_risk(df):
    """
    HYPOTHESIS 10: Suspicious customers transact in high-risk countries
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 10: GEOGRAPHIC RISK (HIGH-RISK COUNTRIES)")
    print("=" * 80)
    
    print("\nTheory: Money launderers use high-risk jurisdictions")
    print("Prediction: Suspicious customers transact more in sanctioned/high-risk countries")
    
    # FATF high-risk countries (as of 2024/2025)
    HIGH_RISK_COUNTRIES = {
        'AF', 'MM', 'KP', 'IR', 'SY',  # FATF blacklist
        'AL', 'BB', 'BF', 'KH', 'KY', 'HT', 'JM', 'JO', 'ML', 'MA',  # FATF greylist (partial)
        'MZ', 'NI', 'PA', 'PH', 'SN', 'SS', 'TZ', 'TR', 'UG', 'AE', 'YE', 'VE'
    }
    
    # Check card transactions (merchant country)
    if 'card_country' in df.columns:
        card_df = df[df['card_country'].notna()].copy()
        
        if len(card_df) > 0:
            print(f"\nAnalyzing {len(card_df):,} card transactions")
            
            card_df['is_high_risk_country'] = card_df['card_country'].astype(str).str.upper().isin(HIGH_RISK_COUNTRIES)
            
            geo_risk = card_df.groupby('customer_id').agg({
                'is_high_risk_country': 'mean',
                'label': 'first'
            }).reset_index()
            geo_risk.columns = ['customer_id', 'high_risk_country_pct', 'label']
            geo_risk['high_risk_country_pct'] *= 100
            
            suspicious = geo_risk[geo_risk['label'] == 1]
            normal = geo_risk[geo_risk['label'] == 0]
            
            print(f"\n% of transactions in high-risk countries:")
            print(f"  Suspicious customers (n={len(suspicious)}):")
            if len(suspicious) > 0:
                print(f"    Mean: {suspicious['high_risk_country_pct'].mean():.2f}%")
                print(f"    With ANY high-risk: {(suspicious['high_risk_country_pct'] > 0).sum()}/{len(suspicious)}")
            
            print(f"  Normal customers (n={len(normal)}):")
            if len(normal) > 0:
                print(f"    Mean: {normal['high_risk_country_pct'].mean():.2f}%")
                print(f"    With ANY high-risk: {(normal['high_risk_country_pct'] > 0).sum()}/{len(normal)}")
            
            # Top countries
            print(f"\nTop countries for suspicious customers:")
            susp_txns = card_df[card_df['label'] == 1]
            if len(susp_txns) > 0:
                print(susp_txns['card_country'].value_counts().head(10))
            
            if len(suspicious) > 0 and len(normal) > 0:
                susp_with_risk = (suspicious['high_risk_country_pct'] > 0).sum() / len(suspicious)
                norm_with_risk = (normal['high_risk_country_pct'] > 0).sum() / len(normal)
                
                if susp_with_risk > norm_with_risk * 1.5 and susp_with_risk > 0.1:
                    conclusion = "SUPPORTED"
                    print(f"\n✓ Suspicious customers {susp_with_risk/max(norm_with_risk, 0.001):.1f}x more likely to use high-risk countries")
                else:
                    conclusion = "NOT SUPPORTED"
                    print(f"\n✗ No significant difference in high-risk country usage")
                
                return {
                    'hypothesis': 'Geographic Risk',
                    'conclusion': conclusion,
                    'data': geo_risk
                }
    
    print("\nInsufficient geographic data - SKIPPING")
    return None

def hypothesis_11_timing_patterns(df):
    """
    HYPOTHESIS 11: Suspicious customers have unusual timing patterns (weekends, late-night)
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 11: UNUSUAL TIMING PATTERNS")
    print("=" * 80)
    
    print("\nTheory: Money launderers operate at unusual times to avoid detection")
    print("Prediction: Suspicious customers transact more on weekends/late nights")
    
    df_valid = df[df['datetime'].notna()].copy()
    
    if len(df_valid) == 0:
        print("\nNo datetime data - SKIPPING")
        return None
    
    # Extract timing features
    df_valid['hour'] = df_valid['datetime'].dt.hour
    df_valid['day_of_week'] = df_valid['datetime'].dt.dayofweek
    df_valid['is_weekend'] = df_valid['day_of_week'].isin([5, 6])
    df_valid['is_late_night'] = df_valid['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5])
    
    timing = df_valid.groupby('customer_id').agg({
        'is_weekend': 'mean',
        'is_late_night': 'mean',
        'label': 'first'
    }).reset_index()
    timing.columns = ['customer_id', 'weekend_pct', 'late_night_pct', 'label']
    timing['weekend_pct'] *= 100
    timing['late_night_pct'] *= 100
    
    suspicious = timing[timing['label'] == 1]
    normal = timing[timing['label'] == 0]
    
    print(f"\n% of transactions on WEEKENDS:")
    print(f"  Suspicious customers (n={len(suspicious)}):")
    print(f"    Mean: {suspicious['weekend_pct'].mean():.1f}%")
    print(f"  Normal customers (n={len(normal)}):")
    print(f"    Mean: {normal['weekend_pct'].mean():.1f}%")
    
    print(f"\n% of transactions LATE NIGHT (10pm-6am):")
    print(f"  Suspicious customers:")
    print(f"    Mean: {suspicious['late_night_pct'].mean():.1f}%")
    print(f"  Normal customers:")
    print(f"    Mean: {normal['late_night_pct'].mean():.1f}%")
    
    # Test weekend pattern
    if len(suspicious) > 0 and len(normal) > 0:
        t_stat_wknd, p_value_wknd = stats.mannwhitneyu(suspicious['weekend_pct'], normal['weekend_pct'], alternative='two-sided')
        t_stat_night, p_value_night = stats.mannwhitneyu(suspicious['late_night_pct'], normal['late_night_pct'], alternative='two-sided')
        
        print(f"\n  Weekend pattern: p-value = {p_value_wknd:.4f}")
        print(f"  Late-night pattern: p-value = {p_value_night:.4f}")
        
        weekend_sig = p_value_wknd < 0.05 and suspicious['weekend_pct'].median() > normal['weekend_pct'].median()
        night_sig = p_value_night < 0.05 and suspicious['late_night_pct'].median() > normal['late_night_pct'].median()
        
        if weekend_sig or night_sig:
            conclusion = "SUPPORTED"
            if weekend_sig:
                print(f"  ✓ Suspicious customers transact MORE on weekends")
            if night_sig:
                print(f"  ✓ Suspicious customers transact MORE late at night")
        else:
            conclusion = "NOT SUPPORTED"
            print(f"  ✗ No significant timing differences")
    else:
        conclusion = "INSUFFICIENT DATA"
        p_value_wknd = None
    
    return {
        'hypothesis': 'Unusual Timing Patterns',
        'conclusion': conclusion,
        'p_value': p_value_wknd,
        'data': timing
    }

def hypothesis_12_velocity_spikes(df):
    """
    HYPOTHESIS 12: Suspicious customers have sudden velocity spikes
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 12: TRANSACTION VELOCITY SPIKES")
    print("=" * 80)
    
    print("\nTheory: Money laundering involves burst activity (sudden spikes)")
    print("Prediction: Suspicious customers have more variable weekly transaction counts")
    
    df_valid = df[df['datetime'].notna()].copy()
    
    if len(df_valid) == 0:
        print("\nNo datetime data - SKIPPING")
        return None
    
    # Group by customer and week
    df_valid['week'] = df_valid['datetime'].dt.to_period('W')
    
    weekly_counts = df_valid.groupby(['customer_id', 'week']).size().reset_index(name='weekly_txn_count')
    
    # Calculate coefficient of variation in weekly counts
    velocity_cv = weekly_counts.groupby('customer_id')['weekly_txn_count'].agg(['mean', 'std']).reset_index()
    velocity_cv['cv'] = velocity_cv['std'] / velocity_cv['mean']
    
    # Merge with labels
    labels = df_valid.groupby('customer_id')['label'].first().reset_index()
    velocity_cv = velocity_cv.merge(labels, on='customer_id', how='left')
    
    suspicious = velocity_cv[velocity_cv['label'] == 1]
    normal = velocity_cv[velocity_cv['label'] == 0]
    
    print(f"\nWeekly transaction count coefficient of variation:")
    print(f"  Suspicious customers (n={len(suspicious)}):")
    if len(suspicious) > 0:
        print(f"    Mean CV: {suspicious['cv'].mean():.2f}")
        print(f"    Median CV: {suspicious['cv'].median():.2f}")
    
    print(f"  Normal customers (n={len(normal)}):")
    if len(normal) > 0:
        print(f"    Mean CV: {normal['cv'].mean():.2f}")
        print(f"    Median CV: {normal['cv'].median():.2f}")
    
    if len(suspicious) > 0 and len(normal) > 0:
        # Remove infinite values
        susp_clean = suspicious['cv'].replace([np.inf, -np.inf], np.nan).dropna()
        norm_clean = normal['cv'].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(susp_clean) > 0 and len(norm_clean) > 0:
            t_stat, p_value = stats.mannwhitneyu(susp_clean, norm_clean, alternative='two-sided')
            print(f"\n  Mann-Whitney U test: p-value = {p_value:.4f}")
            
            if p_value < 0.05:
                if susp_clean.median() > norm_clean.median():
                    conclusion = "SUPPORTED"
                    print(f"  ✓ Suspicious customers have MORE erratic weekly patterns")
                else:
                    conclusion = "OPPOSITE"
            else:
                conclusion = "NOT SUPPORTED"
                print(f"  ✗ No significant difference")
        else:
            conclusion = "INSUFFICIENT DATA"
            p_value = None
    else:
        conclusion = "INSUFFICIENT DATA"
        p_value = None
    
    return {
        'hypothesis': 'Velocity Spikes',
        'conclusion': conclusion,
        'p_value': p_value,
        'data': velocity_cv
    }

def hypothesis_13_income_mismatch(df):
    """
    HYPOTHESIS 13: Transaction volume doesn't match reported income
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 13: INCOME MISMATCH")
    print("=" * 80)
    
    print("\nTheory: Money launderers' transaction volumes exceed legitimate income")
    print("Prediction: Suspicious customers have high transaction-to-income ratio")
    
    if 'customer_income' not in df.columns:
        print("\ncustomer_income column not found - SKIPPING")
        return None
    
    # Calculate total transaction value per customer
    df_valid = df[df['amount'].notna()].copy()
    
    customer_totals = df_valid.groupby('customer_id').agg({
        'amount': 'sum',
        'customer_income': 'first',
        'label': 'first'
    }).reset_index()
    customer_totals.columns = ['customer_id', 'total_txn_value', 'income', 'label']
    
    # Filter to customers with income data
    customer_totals = customer_totals[customer_totals['income'].notna() & (customer_totals['income'] > 0)]
    
    if len(customer_totals) == 0:
        print("\nNo income data available - SKIPPING")
        return None
    
    # Calculate ratio
    customer_totals['txn_to_income_ratio'] = customer_totals['total_txn_value'] / customer_totals['income']
    
    suspicious = customer_totals[customer_totals['label'] == 1]
    normal = customer_totals[customer_totals['label'] == 0]
    
    print(f"\nTransaction-to-Income Ratio:")
    print(f"  Suspicious customers (n={len(suspicious)}):")
    if len(suspicious) > 0:
        print(f"    Mean: {suspicious['txn_to_income_ratio'].mean():.2f}")
        print(f"    Median: {suspicious['txn_to_income_ratio'].median():.2f}")
    
    print(f"  Normal customers (n={len(normal)}):")
    if len(normal) > 0:
        print(f"    Mean: {normal['txn_to_income_ratio'].mean():.2f}")
        print(f"    Median: {normal['txn_to_income_ratio'].median():.2f}")
    
    if len(suspicious) > 0 and len(normal) > 0:
        # Remove outliers
        susp_clean = suspicious['txn_to_income_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
        norm_clean = normal['txn_to_income_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(susp_clean) > 0 and len(norm_clean) > 0:
            t_stat, p_value = stats.mannwhitneyu(susp_clean, norm_clean, alternative='two-sided')
            print(f"\n  Mann-Whitney U test: p-value = {p_value:.4f}")
            
            if p_value < 0.05:
                if susp_clean.median() > norm_clean.median():
                    conclusion = "SUPPORTED"
                    print(f"  ✓ Suspicious customers have HIGHER txn-to-income ratio")
                else:
                    conclusion = "OPPOSITE"
            else:
                conclusion = "NOT SUPPORTED"
                print(f"  ✗ No significant difference")
        else:
            conclusion = "INSUFFICIENT DATA"
            p_value = None
    else:
        conclusion = "INSUFFICIENT DATA"
        p_value = None
    
    return {
        'hypothesis': 'Income Mismatch',
        'conclusion': conclusion,
        'p_value': p_value,
        'data': customer_totals
    }

def hypothesis_14_transaction_diversity(df):
    """
    HYPOTHESIS 14: Suspicious customers use more diverse transaction types (layering)
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 14: TRANSACTION TYPE DIVERSITY")
    print("=" * 80)
    
    print("\nTheory: Money laundering involves layering through multiple transaction types")
    print("Prediction: Suspicious customers use more different transaction types")
    
    # Count transaction types used per customer
    type_flags = ['from_card', 'from_abm', 'from_cheque', 'from_eft', 'from_emt', 'from_westernunion', 'from_wire']
    
    available_flags = [f for f in type_flags if f in df.columns]
    
    if len(available_flags) == 0:
        print("\nNo transaction type flags found - SKIPPING")
        return None
    
    print(f"\nAnalyzing usage of {len(available_flags)} transaction types")
    
    # For each customer, count how many different transaction types they use
    # Method: Count how many flags are True for each customer
    def count_transaction_types(group):
        count = 0
        for flag in available_flags:
            if group[flag].any():  # If customer has ANY transaction of this type
                count += 1
        return count
    
    customer_diversity = df.groupby('customer_id').apply(
        lambda g: pd.Series({
            'num_types_used': count_transaction_types(g),
            'label': g['label'].iloc[0] if 'label' in g.columns else None
        })
    ).reset_index()
    
    suspicious = customer_diversity[customer_diversity['label'] == 1]
    normal = customer_diversity[customer_diversity['label'] == 0]
    
    print(f"\nNumber of different transaction types used:")
    print(f"  Suspicious customers (n={len(suspicious)}):")
    if len(suspicious) > 0:
        print(f"    Mean: {suspicious['num_types_used'].mean():.2f}")
        print(f"    Median: {suspicious['num_types_used'].median():.1f}")
        print(f"    Distribution:")
        print(suspicious['num_types_used'].value_counts().sort_index())
    
    print(f"  Normal customers (n={len(normal)}):")
    if len(normal) > 0:
        print(f"    Mean: {normal['num_types_used'].mean():.2f}")
        print(f"    Median: {normal['num_types_used'].median():.1f}")
    
    if len(suspicious) > 0 and len(normal) > 0:
        t_stat, p_value = stats.mannwhitneyu(suspicious['num_types_used'], normal['num_types_used'], alternative='two-sided')
        print(f"\n  Mann-Whitney U test: p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            if suspicious['num_types_used'].median() > normal['num_types_used'].median():
                conclusion = "SUPPORTED"
                print(f"  ✓ Suspicious customers use MORE transaction types (layering)")
            else:
                conclusion = "OPPOSITE"
        else:
            conclusion = "NOT SUPPORTED"
            print(f"  ✗ No significant difference")
    else:
        conclusion = "INSUFFICIENT DATA"
        p_value = None
    
    return {
        'hypothesis': 'Transaction Type Diversity',
        'conclusion': conclusion,
        'p_value': p_value,
        'data': customer_diversity
    }

def generate_enhanced_summary(results):
    """Generate summary of enhanced hypotheses"""
    print("\n" + "=" * 80)
    print("ENHANCED HYPOTHESIS TESTING SUMMARY")
    print("=" * 80)
    
    print("\nNew Hypotheses Tested (Part 2):")
    print("-" * 80)
    
    for i, result in enumerate(results, 7):
        if result is not None:
            print(f"\n{i}. {result['hypothesis']}")
            print(f"   Conclusion: {result['conclusion']}")
            if result.get('p_value'):
                print(f"   Statistical significance: p = {result['p_value']:.4f}")
    
    # Count conclusions
    supported = sum(1 for r in results if r and r['conclusion'] == 'SUPPORTED')
    not_supported = sum(1 for r in results if r and r['conclusion'] == 'NOT SUPPORTED')
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {supported} SUPPORTED, {not_supported} NOT SUPPORTED")
    print("=" * 80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python aml_enhanced_hypotheses.py <data_file.csv>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Load data
    df = load_data(filepath)
    if df is None:
        sys.exit(1)
    
    # Prepare data
    df = create_unified_columns(df)
    
    # Test enhanced hypotheses
    results = []
    
    result7 = hypothesis_7_merchant_category_risk(df)
    results.append(result7)
    
    result8 = hypothesis_8_ecommerce_patterns(df)
    results.append(result8)
    
    result9 = hypothesis_9_cash_preference(df)
    results.append(result9)
    
    result10 = hypothesis_10_geographic_risk(df)
    results.append(result10)
    
    result11 = hypothesis_11_timing_patterns(df)
    results.append(result11)
    
    result12 = hypothesis_12_velocity_spikes(df)
    results.append(result12)
    
    result13 = hypothesis_13_income_mismatch(df)
    results.append(result13)
    
    result14 = hypothesis_14_transaction_diversity(df)
    results.append(result14)
    
    # Generate summary
    generate_enhanced_summary(results)
    
    print("\n" + "=" * 80)
    print("ENHANCED ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nCombine these results with the original 6 hypotheses")
    print("to build a comprehensive detection model!")

if __name__ == "__main__":
    main()
