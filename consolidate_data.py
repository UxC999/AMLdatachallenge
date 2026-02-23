#!/usr/bin/env python3
"""
AML Data Consolidation Script
==============================
This script consolidates the master_data.csv nto a clean, structured format:
1. Quality checks on raw data
2. Handles rows with no transaction type
3. Stacks all transaction types into unified transaction table
4. Creates customer master table

"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys

print("="*80)
print("AML DATA CONSOLIDATION")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# CONFIGURATION

# Get filename from command line
if len(sys.argv) < 2:
    print("ERROR: Please provide the input file as a command line argument")
    print("\nUsage:")
    print("  python consolidate_data.py <input_file>")
    print("\nExample:")
    print("  python consolidate_data.py master_data.csv")
    sys.exit(1)

INPUT_FILE = sys.argv[1]
OUTPUT_TRANSACTIONS = "transactions_consolidated.csv"
OUTPUT_CUSTOMERS = "customers_master.csv"
OUTPUT_SUMMARY = "consolidation_summary.txt"

print(f"Input file: {INPUT_FILE}")
print(f"Output files:")
print(f"  - {OUTPUT_TRANSACTIONS}")
print(f"  - {OUTPUT_CUSTOMERS}")
print(f"  - {OUTPUT_SUMMARY}")
print()

# STEP 1: LOAD DATA

print("="*80)
print("STEP 1: LOADING DATA")
print("-"*80)

try:
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"Loaded {len(df):,} rows, {len(df.columns):,} columns")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# STEP 2: QUALITY CHECKS

print("\n" + "="*80)
print("STEP 2: DATA QUALITY CHECKS")
print("-"*80)

# Check for null critical fields
print("\nChecking critical fields...")
critical_fields = ['transaction_id', 'customer_id']
for field in critical_fields:
    null_count = df[field].isnull().sum()
    if null_count > 0:
        print(f"  {field}: {null_count:,} null values")
    else:
        print(f"  {field}: No null values")

# Check for duplicate transaction IDs
dup_txn = df['transaction_id'].duplicated().sum()
if dup_txn > 0:
    print(f"  Duplicate transaction IDs: {dup_txn:,}")
else:
    print(f"  No duplicate transaction IDs")

# Check for exact duplicate rows
dup_rows = df.duplicated().sum()
if dup_rows > 0:
    print(f"  Exact duplicate rows: {dup_rows:,}")
    print(f"     Removing duplicates...")
    df = df.drop_duplicates()
    print(f"     Removed. New row count: {len(df):,}")
else:
    print(f"  No exact duplicate rows")

# STEP 3: HANDLE ROWS WITH NO TRANSACTION TYPE

print("\n" + "="*80)
print("STEP 3: HANDLING ROWS WITH NO TRANSACTION TYPE")
print("-"*80)

transaction_type_cols = ['from_abm', 'from_card', 'from_cheque', 'from_eft', 
                         'from_emt', 'from_westernunion', 'from_wire']

# Identify rows with no transaction type
df['num_txn_types'] = 0
for col in transaction_type_cols:
    if col in df.columns:
        df['num_txn_types'] += (df[col] == True).astype(int)

no_type = df[df['num_txn_types'] == 0]
print(f"\nRows with no transaction type: {len(no_type):,} ({len(no_type)/len(df)*100:.2f}%)")

if len(no_type) > 0:
    print("\nInvestigating these rows...")
    
    # Check if they have any amount data
    amount_cols = ['abm_amount', 'card_amount', 'cheque_amount', 'eft_amount',
                   'emt_amount', 'westernunion_amount', 'wire_amount']
    
    has_any_amount = False
    for col in amount_cols:
        if col in df.columns:
            if no_type[col].notna().any():
                has_any_amount = True
                print(f"  Some rows have data in: {col}")
                break
    
    if not has_any_amount:
        print(f"  These rows have NO transaction amount data")
        print(f"  → Decision: DROPPING these {len(no_type):,} rows")
        df = df[df['num_txn_types'] > 0].copy()
        print(f"  New row count: {len(df):,}")
    else:
        print(f"  Some rows have amount data but no type flag")
        print(f"  → Keeping them for manual review")

# Check for multiple transaction types per row
multi_type = df[df['num_txn_types'] > 1]
if len(multi_type) > 0:
    print(f"\nWARNING: {len(multi_type):,} rows have MULTIPLE transaction types")
    print(f"   This is unusual. Keeping first type found.")
else:
    print(f"\nGood: Each row has exactly ONE transaction type")

# STEP 4: STACK TRANSACTION TYPES INTO UNIFIED FORMAT

print("\n" + "="*80)
print("STEP 4: CONSOLIDATING TRANSACTION DATA")
print("-"*80)
print("\nThis will take several minutes...")

# Define transaction type mapping
transaction_configs = {
    'ABM': {
        'flag': 'from_abm',
        'amount': 'abm_amount',
        'debit_credit': 'abm_debit_credit',
        'datetime': 'abm_datetime',
        'country': 'abm_country',
        'province': 'abm_province',
        'city': 'abm_city',
        'extra_fields': ['abm_cash_indicator']
    },
    'CARD': {
        'flag': 'from_card',
        'amount': 'card_amount',
        'debit_credit': 'card_debit_credit',
        'datetime': 'card_datetime',
        'country': 'card_country',
        'province': 'card_province',
        'city': 'card_city',
        'extra_fields': ['card_merchant_category', 'card_ecommerce_indicator']
    },
    'CHEQUE': {
        'flag': 'from_cheque',
        'amount': 'cheque_amount',
        'debit_credit': 'cheque_debit_credit',
        'datetime': 'cheque_datetime',
        'country': None,
        'province': None,
        'city': None,
        'extra_fields': []
    },
    'EFT': {
        'flag': 'from_eft',
        'amount': 'eft_amount',
        'debit_credit': 'eft_debit_credit',
        'datetime': 'eft_datetime',
        'country': None,
        'province': None,
        'city': None,
        'extra_fields': []
    },
    'EMT': {
        'flag': 'from_emt',
        'amount': 'emt_amount',
        'debit_credit': 'emt_debit_credit',
        'datetime': 'emt_datetime',
        'country': None,
        'province': None,
        'city': None,
        'extra_fields': []
    },
    'WESTERNUNION': {
        'flag': 'from_westernunion',
        'amount': 'westernunion_amount',
        'debit_credit': 'westernunion_debit_credit',
        'datetime': 'westernunion_datetime',
        'country': None,
        'province': None,
        'city': None,
        'extra_fields': []
    },
    'WIRE': {
        'flag': 'from_wire',
        'amount': 'wire_amount',
        'debit_credit': 'wire_debit_credit',
        'datetime': 'wire_datetime',
        'country': None,
        'province': None,
        'city': None,
        'extra_fields': []
    }
}

# Create consolidated transactions list
consolidated_rows = []

print("Processing transaction types:")
for txn_type, config in transaction_configs.items():
    # Get rows for this transaction type
    if config['flag'] in df.columns:
        mask = df[config['flag']] == True
        txn_df = df[mask].copy()
        
        if len(txn_df) > 0:
            print(f"  {txn_type:15s}: {len(txn_df):,} transactions")
            
            # Create consolidated record
            consolidated = pd.DataFrame()
            consolidated['transaction_id'] = txn_df['transaction_id']
            consolidated['customer_id'] = txn_df['customer_id']
            consolidated['transaction_type'] = txn_type
            
            # Core transaction fields
            consolidated['amount'] = txn_df[config['amount']] if config['amount'] in df.columns else np.nan
            consolidated['debit_credit'] = txn_df[config['debit_credit']] if config['debit_credit'] in df.columns else None
            consolidated['transaction_datetime'] = txn_df[config['datetime']] if config['datetime'] in df.columns else None
            
            # Location fields
            consolidated['txn_country'] = txn_df[config['country']] if config['country'] and config['country'] in df.columns else None
            consolidated['txn_province'] = txn_df[config['province']] if config['province'] and config['province'] in df.columns else None
            consolidated['txn_city'] = txn_df[config['city']] if config['city'] and config['city'] in df.columns else None
            
            # Extra fields (type-specific)
            for extra_field in config['extra_fields']:
                if extra_field in df.columns:
                    # Use original column name for type-specific fields
                    consolidated[extra_field] = txn_df[extra_field]
            
            # Add label if present
            if 'label' in df.columns:
                consolidated['label'] = txn_df['label']
            
            consolidated_rows.append(consolidated)

# Combine all transaction types
print("\nCombining all transaction types...")
transactions = pd.concat(consolidated_rows, ignore_index=True)
print(f"Created consolidated transaction table: {len(transactions):,} rows")

# STEP 5: CREATE CUSTOMER MASTER TABLE

print("\n" + "="*80)
print("STEP 5: CREATING CUSTOMER MASTER TABLE")
print("-"*80)

# Define customer fields
individual_fields = {
    'customer_gender': 'first',
    'customer_marital_status': 'first',
    'customer_birth_date': 'first',
    'customer_occupation_code': 'first',
    'customer_occupation_title': 'first',
    'customer_income': 'first',
    'customer_country': 'first',
    'customer_province': 'first',
    'customer_city': 'first',
    'customer_onboard_date': 'first'
}

business_fields = {
    'business_industry_code': 'first',
    'business_industry': 'first',
    'business_employee_count': 'first',
    'business_sales': 'first',
    'business_established_date': 'first',
    'business_country': 'first',
    'business_province': 'first',
    'business_city': 'first',
    'business_onboard_date': 'first'
}

# Combine all customer fields
customer_agg = {}
for field, agg_func in {**individual_fields, **business_fields}.items():
    if field in df.columns:
        customer_agg[field] = agg_func

print(f"Aggregating {len(customer_agg)} customer fields...")
customers = df.groupby('customer_id').agg(customer_agg).reset_index()

# Add customer-level flags
if 'from_kyc_individual' in df.columns:
    customers['has_individual_kyc'] = df.groupby('customer_id')['from_kyc_individual'].any().reset_index(drop=True)
if 'from_kyc_smallbusiness' in df.columns:
    customers['has_business_kyc'] = df.groupby('customer_id')['from_kyc_smallbusiness'].any().reset_index(drop=True)

print(f"Created customer master table: {len(customers):,} customers")

# STEP 6: DATA TYPE CONVERSIONS

print("\n" + "="*80)
print("STEP 6: CONVERTING DATA TYPES")
print("-"*80)

# Convert datetime fields
datetime_fields = ['transaction_datetime']
for field in datetime_fields:
    if field in transactions.columns:
        print(f"Converting {field} to datetime...")
        transactions[field] = pd.to_datetime(transactions[field], errors='coerce')

customer_date_fields = ['customer_birth_date', 'customer_onboard_date', 
                        'business_established_date', 'business_onboard_date']
for field in customer_date_fields:
    if field in customers.columns:
        print(f"Converting {field} to datetime...")
        customers[field] = pd.to_datetime(customers[field], errors='coerce')

# Convert amount to float (should already be, but ensure)
if 'amount' in transactions.columns:
    transactions['amount'] = transactions['amount'].astype(float)

# Convert categorical fields
categorical_fields = ['transaction_type', 'debit_credit']
for field in categorical_fields:
    if field in transactions.columns:
        print(f"Converting {field} to category...")
        transactions[field] = transactions[field].astype('category')

customer_categorical = ['customer_gender', 'customer_marital_status', 'customer_occupation_code']
for field in customer_categorical:
    if field in customers.columns:
        print(f"Converting {field} to category...")
        customers[field] = customers[field].astype('category')

print("Data type conversions complete")

# STEP 7: SAVE CONSOLIDATED DATA

print("\n" + "="*80)
print("STEP 7: SAVING CONSOLIDATED DATA")
print("-"*80)

# Save transactions
print(f"\nSaving transactions to {OUTPUT_TRANSACTIONS}...")
print("This may take several minutes...")
transactions.to_csv(OUTPUT_TRANSACTIONS, index=False)
print(f"Saved {len(transactions):,} transactions")

# Save customers
print(f"\nSaving customers to {OUTPUT_CUSTOMERS}...")
customers.to_csv(OUTPUT_CUSTOMERS, index=False)
print(f"Saved {len(customers):,} customers")

# STEP 8: GENERATE SUMMARY REPORT

print("\n" + "="*80)
print("STEP 8: GENERATING SUMMARY REPORT")
print("-"*80)

summary = []
summary.append("="*80)
summary.append("AML DATA CONSOLIDATION SUMMARY")
summary.append("="*80)
summary.append(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
summary.append(f"\nInput File: {INPUT_FILE}")
summary.append(f"  Original rows: {len(df):,}")
summary.append(f"  Original columns: {len(df.columns):,}")

summary.append(f"\n{'-'*80}")
summary.append("OUTPUT FILES:")
summary.append(f"{'-'*80}")
summary.append(f"\n1. {OUTPUT_TRANSACTIONS}")
summary.append(f"   Rows: {len(transactions):,}")
summary.append(f"   Columns: {len(transactions.columns):,}")
summary.append(f"   Transaction types: {transactions['transaction_type'].nunique()}")
summary.append(f"   Date range: {transactions['transaction_datetime'].min()} to {transactions['transaction_datetime'].max()}")

summary.append(f"\n2. {OUTPUT_CUSTOMERS}")
summary.append(f"   Rows: {len(customers):,}")
summary.append(f"   Columns: {len(customers.columns):,}")

summary.append(f"\n{'-'*80}")
summary.append("TRANSACTION TYPE DISTRIBUTION:")
summary.append(f"{'-'*80}")
for txn_type, count in transactions['transaction_type'].value_counts().items():
    pct = count / len(transactions) * 100
    summary.append(f"  {txn_type:15s}: {count:,} ({pct:.2f}%)")

if 'label' in transactions.columns:
    summary.append(f"\n{'-'*80}")
    summary.append("LABEL DISTRIBUTION:")
    summary.append(f"{'-'*80}")
    summary.append(f"  Labeled: {transactions['label'].notna().sum():,}")
    summary.append(f"  Unlabeled: {transactions['label'].isna().sum():,}")
    if transactions['label'].notna().any():
        summary.append(f"  Suspicious (1): {(transactions['label'] == 1).sum():,}")
        summary.append(f"  Not Suspicious (0): {(transactions['label'] == 0).sum():,}")

summary.append(f"\n{'='*80}")

# Print summary
summary_text = "\n".join(summary)
print("\n" + summary_text)

# Save summary
with open(OUTPUT_SUMMARY, 'w') as f:
    f.write(summary_text)
print(f"\nSummary saved to {OUTPUT_SUMMARY}")

# COMPLETION

print("\n" + "="*80)
print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nOutput files created:")
print(f"  {OUTPUT_TRANSACTIONS}")
print(f"  {OUTPUT_CUSTOMERS}")
print(f"  {OUTPUT_SUMMARY}")
print("="*80)
