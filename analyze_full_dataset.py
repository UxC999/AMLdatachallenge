#!/usr/bin/env python3
"""
Full Dataset Analysis Script
===========================================================
This script analyzes the complete AML dataset with renamed columns to understand:
1. Overall size and structure
2. Transaction type distribution
3. Label distribution
4. Data completeness
5. Whether rows can have multiple transaction types

"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys

print("="*80)
print("AML DATASET FULL ANALYSIS (RENAMED COLUMNS)")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# PART 1: LOAD DATA & BASIC STATS

if len(sys.argv) < 2:
    print("ERROR: Please provide the input file as a command line argument")
    sys.exit(1)

DATA_FILE = sys.argv[1]
print(f"Input file: {DATA_FILE}")

try:
    # Load with low_memory=False to avoid dtype warnings
    print(f"Loading {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    print(f"✓ Data loaded successfully!")
except FileNotFoundError:
    print(f"ERROR: File not found: {DATA_FILE}")
    print("Please update the DATA_FILE variable with the correct path.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR loading file: {e}")
    sys.exit(1)

print(f"\nDataset Shape:")
print(f"  Rows: {len(df):,}")
print(f"  Columns: {len(df.columns):,}")
print(f"  Unique customers: {df['customer_id'].nunique():,}")
print(f"  Unique transactions: {df['transaction_id'].nunique():,}")

print(f"\nColumn names (first 20):")
for i, col in enumerate(df.columns[:20], 1):
    print(f"  {i:2d}. {col}")
if len(df.columns) > 20:
    print(f"  ... and {len(df.columns) - 20} more")

# PART 2: TRANSACTION TYPE DISTRIBUTION

print("\n" + "="*80)
print("PART 2: TRANSACTION TYPE DISTRIBUTION")
print("-"*80)

transaction_types = [
    'from_abm',
    'from_card', 
    'from_cheque',
    'from_eft',
    'from_emt',
    'from_westernunion',
    'from_wire'
]

print("\nTransaction counts by type:")
type_counts = {}
for txn_type in transaction_types:
    if txn_type in df.columns:
        # Handle both boolean and float types
        if df[txn_type].dtype == bool:
            count = df[txn_type].sum()
        else:
            count = (df[txn_type] == True).sum()
        type_counts[txn_type] = count
        pct = (count / len(df) * 100) if len(df) > 0 else 0
        print(f"  {txn_type:20s}: {count:,} ({pct:.2f}%)")
    else:
        print(f"  {txn_type:20s}: Column not found")

# Check if any row has multiple transaction types
print("\n" + "-"*80)
print("Checking for rows with MULTIPLE transaction types...")

# Create boolean mask for each type
type_masks = {}
for txn_type in transaction_types:
    if txn_type in df.columns:
        type_masks[txn_type] = df[txn_type] == True

# Count how many types per row
if type_masks:
    df['num_txn_types'] = sum(mask.astype(int) for mask in type_masks.values())
    
    print(f"\nRows by number of transaction types:")
    type_distribution = df['num_txn_types'].value_counts().sort_index()
    for num_types, count in type_distribution.items():
        pct = (count / len(df) * 100)
        print(f"  {num_types} type(s): {count:,} rows ({pct:.2f}%)")
    
    # Show examples of multi-type rows if they exist
    multi_type_rows = df[df['num_txn_types'] > 1]
    if len(multi_type_rows) > 0:
        print(f"\nWARNING: Found {len(multi_type_rows):,} rows with MULTIPLE transaction types!")
        print("\nSample multi-type row:")
        sample = multi_type_rows.iloc[0]
        print(f"  Transaction ID: {sample['transaction_id']}")
        print(f"  Customer ID: {sample['customer_id']}")
        print(f"  Active transaction types:")
        for txn_type in transaction_types:
            if txn_type in df.columns and sample[txn_type] == True:
                print(f"    {txn_type}")
    else:
        print(f"\nNice: Each row has exactly ONE transaction type")

# PART 3: LABEL DISTRIBUTION

print("\n" + "="*80)
print("PART 3: LABEL DISTRIBUTION")
print("-"*80)

if 'label' in df.columns:
    print("\nLabel column analysis:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Labeled rows: {df['label'].notna().sum():,}")
    print(f"  Unlabeled rows (NA): {df['label'].isna().sum():,}")
    print(f"  Labeled percentage: {df['label'].notna().sum()/len(df)*100:.2f}%")
    
    if df['label'].notna().any():
        print(f"\nLabel value distribution:")
        label_counts = df['label'].value_counts(dropna=False).sort_index()
        for label, count in label_counts.items():
            if pd.notna(label):
                label_name = "Suspicious" if label == 1 else "Not Suspicious" if label == 0 else "Unknown"
                pct = (count / len(df) * 100)
                pct_of_labeled = (count / df['label'].notna().sum() * 100) if df['label'].notna().sum() > 0 else 0
                print(f"    {label} ({label_name}): {count:,} ({pct:.3f}% of all, {pct_of_labeled:.1f}% of labeled)")
        
        # Customer-level labels
        print(f"\nCustomer-level label analysis:")
        customer_labels = df[df['label'].notna()].groupby('customer_id')['label'].first().reset_index()
        
        print(f"  Total unique customers: {df['customer_id'].nunique():,}")
        print(f"  Customers with labels: {len(customer_labels):,}")
        
        if len(customer_labels) > 0:
            suspicious_customers = customer_labels[customer_labels['label'] == 1]
            not_suspicious_customers = customer_labels[customer_labels['label'] == 0]
            print(f"  Suspicious customers (label=1): {len(suspicious_customers):,}")
            print(f"  Not suspicious customers (label=0): {len(not_suspicious_customers):,}")
            
            # Check if suspicious customers have multiple transactions
            if len(suspicious_customers) > 0:
                susp_customer_ids = suspicious_customers['customer_id'].tolist()
                susp_txns = df[df['customer_id'].isin(susp_customer_ids)]
                avg_txns_per_susp = len(susp_txns) / len(suspicious_customers)
                print(f"  Avg transactions per suspicious customer: {avg_txns_per_susp:.1f}")
else:
    print("ERROR: No 'label' column found in dataset")

# PART 4: DATA COMPLETENESS BY TRANSACTION TYPE

print("\n" + "="*80)
print("PART 4: DATA COMPLETENESS BY TRANSACTION TYPE")
print("-"*80)

# For each transaction type, check which amount/datetime fields have data
transaction_fields = {
    'ABM': ['abm_amount', 'abm_debit_credit', 'abm_datetime', 'abm_cash_indicator',
            'abm_country', 'abm_province', 'abm_city'],
    'Card': ['card_amount', 'card_debit_credit', 'card_datetime', 
             'card_merchant_category', 'card_ecommerce_indicator',
             'card_country', 'card_province', 'card_city'],
    'Cheque': ['cheque_amount', 'cheque_debit_credit', 'cheque_datetime'],
    'EFT': ['eft_amount', 'eft_debit_credit', 'eft_datetime'],
    'EMT': ['emt_amount', 'emt_debit_credit', 'emt_datetime'],
    'WesternUnion': ['westernunion_amount', 'westernunion_debit_credit', 'westernunion_datetime'],
    'Wire': ['wire_amount', 'wire_debit_credit', 'wire_datetime']
}

print("\nNon-null counts for transaction fields:")
for txn_name, fields in transaction_fields.items():
    has_data = False
    for field in fields:
        if field in df.columns and df[field].notna().sum() > 0:
            has_data = True
            break
    
    if has_data:
        print(f"\n{txn_name}:")
        for field in fields:
            if field in df.columns:
                non_null = df[field].notna().sum()
                pct = (non_null / len(df) * 100) if len(df) > 0 else 0
                if non_null > 0:  # Only show fields with data
                    print(f"  {field:30s}: {non_null:,} ({pct:.1f}%)")

# PART 5: CUSTOMER KYC DATA COMPLETENESS

print("\n" + "="*80)
print("PART 5: CUSTOMER KYC DATA COMPLETENESS")
print("-"*80)

kyc_fields = {
    'Individual Customer': [
        'customer_gender', 'customer_marital_status', 'customer_birth_date', 
        'customer_occupation_code', 'customer_occupation_title', 'customer_income',
        'customer_country', 'customer_province', 'customer_city', 'customer_onboard_date'
    ],
    'Business Customer': [
        'business_industry_code', 'business_industry', 'business_employee_count', 
        'business_sales', 'business_established_date',
        'business_country', 'business_province', 'business_city', 'business_onboard_date'
    ]
}

for kyc_type, fields in kyc_fields.items():
    has_data = any(df[field].notna().sum() > 0 for field in fields if field in df.columns)
    if has_data:
        print(f"\n{kyc_type}:")
        for field in fields:
            if field in df.columns:
                non_null = df[field].notna().sum()
                pct = (non_null / len(df) * 100) if len(df) > 0 else 0
                if non_null > 0:  # Only show fields with data
                    print(f"  {field:30s}: {non_null:,} ({pct:.1f}%)")

# PART 6: AMOUNT STATISTICS BY TRANSACTION TYPE

print("\n" + "="*80)
print("PART 6: AMOUNT STATISTICS BY TRANSACTION TYPE")
print("-"*80)

amount_fields = {
    'ABM': 'abm_amount',
    'Card': 'card_amount',
    'Cheque': 'cheque_amount',
    'EFT': 'eft_amount',
    'EMT': 'emt_amount',
    'WesternUnion': 'westernunion_amount',
    'Wire': 'wire_amount'
}

for txn_name, amount_col in amount_fields.items():
    if amount_col in df.columns:
        amounts = df[amount_col].dropna()
        if len(amounts) > 0:
            print(f"\n{txn_name}:")
            print(f"  Count: {len(amounts):,}")
            print(f"  Total: ${amounts.sum():,.2f}")
            print(f"  Mean: ${amounts.mean():,.2f}")
            print(f"  Median: ${amounts.median():,.2f}")
            print(f"  Min: ${amounts.min():,.2f}")
            print(f"  Max: ${amounts.max():,.2f}")

# PART 7: SAMPLE ROWS FROM EACH TRANSACTION TYPE

print("\n" + "="*80)
print("PART 7: SAMPLE ROWS BY TRANSACTION TYPE")
print("-"*80)

for txn_type in transaction_types:
    if txn_type in df.columns:
        sample = df[df[txn_type] == True].head(1)
        if len(sample) > 0:
            print(f"\nSample row for {txn_type}:")
            print(f"  Transaction ID: {sample['transaction_id'].iloc[0]}")
            print(f"  Customer ID: {sample['customer_id'].iloc[0]}")
            
            # Find the corresponding amount field
            txn_prefix = txn_type.replace('from_', '')
            amount_col = f'{txn_prefix}_amount'
            datetime_col = f'{txn_prefix}_datetime'
            debit_credit_col = f'{txn_prefix}_debit_credit'
            
            if amount_col in df.columns and pd.notna(sample[amount_col].iloc[0]):
                print(f"  Amount: ${sample[amount_col].iloc[0]:,.2f}")
            if debit_credit_col in df.columns and pd.notna(sample[debit_credit_col].iloc[0]):
                print(f"  Type: {sample[debit_credit_col].iloc[0]}")
            if datetime_col in df.columns and pd.notna(sample[datetime_col].iloc[0]):
                print(f"  DateTime: {sample[datetime_col].iloc[0]}")

# PART 8: SUMMARY & RECOMMENDATIONS

print("\n" + "="*80)
print("PART 8: SUMMARY & RECOMMENDATIONS")
print("="*80)

print("\nKEY FINDINGS:")
print(f"1. Total transactions: {len(df):,}")
print(f"2. Unique customers: {df['customer_id'].nunique():,}")

if type_counts:
    dominant_type = max(type_counts, key=type_counts.get)
    print(f"3. Most common transaction type: {dominant_type} ({type_counts[dominant_type]:,} transactions)")

if 'label' in df.columns and df['label'].notna().any():
    labeled_count = df['label'].notna().sum()
    print(f"4. Labeled data: {labeled_count:,} transactions ({labeled_count/len(df)*100:.2f}%)")
    if (df['label'] == 1).any():
        susp_count = (df['label'] == 1).sum()
        print(f"   - Suspicious: {susp_count:,}")
        print(f"   - Not suspicious: {(df['label'] == 0).sum():,}")

print("\n" + "-"*80)
print("DATA STRUCTURE INSIGHTS:")
print("-"*80)

if 'num_txn_types' in df.columns:
    if (df['num_txn_types'] > 1).any():
        multi_count = (df['num_txn_types'] > 1).sum()
        print(f"\nMultiple transaction types per row detected!")
        print(f"   {multi_count:,} rows have more than one transaction type")
    else:
        print(f"\n✓ Clean structure: Each row = exactly one transaction type")

print("\n" + "="*80)
print(f"Analysis complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
