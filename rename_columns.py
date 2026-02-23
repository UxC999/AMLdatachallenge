#!/usr/bin/env python3
"""
Column Renaming Script for AML Dataset
=======================================
This script renames the .x, .y, .x.x column suffixes to clear, descriptive names
based on the actual data source (transaction type or KYC source).

"""

import pandas as pd
import sys

print("="*80)
print("AML DATASET - COLUMN RENAMING")
print("="*80)

INPUT_FILE = "joined_data_output.csv"
OUTPUT_FILE = "master_data.csv"

rename_map = {
    # CORE IDENTIFIERS (no change needed)
    'transaction_id': 'transaction_id',
    'customer_id': 'customer_id',
    
    # ABM TRANSACTION DATA (.x suffix)
    'amount_cad.x': 'abm_amount',
    'debit_credit.x': 'abm_debit_credit',
    'transaction_datetime.x': 'abm_datetime',
    'cash_indicator': 'abm_cash_indicator',  # Only for ABM
    'country.x': 'abm_country',
    'province.x': 'abm_province',
    'city.x': 'abm_city',
    'from_abm': 'from_abm',
    
    # CARD TRANSACTION DATA (.y suffix on transaction fields)
    'amount_cad.y': 'card_amount',
    'debit_credit.y': 'card_debit_credit',
    'transaction_datetime.y': 'card_datetime',
    'merchant_category': 'card_merchant_category',
    'ecommerce_ind': 'card_ecommerce_indicator',
    'country.y.y': 'card_country',
    'province.y.y': 'card_province',
    'city.y.y': 'card_city',
    'from_card': 'from_card',
    
    # CHEQUE TRANSACTION DATA (.x.x suffix on transaction fields)
    'amount_cad.x.x': 'cheque_amount',
    'debit_credit.x.x': 'cheque_debit_credit',
    'transaction_datetime.x.x': 'cheque_datetime',
    'from_cheque': 'from_cheque',
   
    # EFT TRANSACTION DATA (.y.y suffix on transaction fields)
    'amount_cad.y.y': 'eft_amount',
    'debit_credit.y.y': 'eft_debit_credit',
    'transaction_datetime.y.y': 'eft_datetime',
    'from_eft': 'from_eft',
    
    # EMT TRANSACTION DATA (.x.x.x suffix)
    'amount_cad.x.x.x': 'emt_amount',
    'debit_credit.x.x.x': 'emt_debit_credit',
    'transaction_datetime.x.x.x': 'emt_datetime',
    'from_emt': 'from_emt',
    
    # WESTERN UNION TRANSACTION DATA (.y.y.y suffix)
    'amount_cad.y.y.y': 'westernunion_amount',
    'debit_credit.y.y.y': 'westernunion_debit_credit',
    'transaction_datetime.y.y.y': 'westernunion_datetime',
    'from_westernunion': 'from_westernunion',
    
    # WIRE TRANSACTION DATA (no suffix on transaction fields)
    'amount_cad': 'wire_amount',
    'debit_credit': 'wire_debit_credit',
    'transaction_datetime': 'wire_datetime',
    'from_wire': 'from_wire',
    
    # INDIVIDUAL CUSTOMER KYC DATA (.y suffix on location)
    'country.y': 'customer_country',
    'province.y': 'customer_province',
    'city.y': 'customer_city',
    'gender': 'customer_gender',
    'marital_status': 'customer_marital_status',
    'occupation_code': 'customer_occupation_code',
    'occupation_title': 'customer_occupation_title',
    'income': 'customer_income',
    'birth_date': 'customer_birth_date',
    'onboard_date.x': 'customer_onboard_date',
    'from_kyc_individual': 'from_kyc_individual',
    'from_kyc_occupation_codes': 'from_kyc_occupation_codes',
    
    # BUSINESS CUSTOMER KYC DATA (.x.x suffix on location)
    'country.x.x': 'business_country',
    'province.x.x': 'business_province',
    'city.x.x': 'business_city',
    'industry_code': 'business_industry_code',
    'industry': 'business_industry',
    'employee_count': 'business_employee_count',
    'sales': 'business_sales',
    'established_date': 'business_established_date',
    'onboard_date.y': 'business_onboard_date',
    'from_kyc_smallbusiness': 'from_kyc_smallbusiness',
    'from_kyc_industry_codes': 'from_kyc_industry_codes',
    
    # LABELS
    'label': 'label',
    'from_labels': 'from_labels',
}

# LOAD AND RENAME

print(f"\nLoading data from: {INPUT_FILE}")

try:
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"Loaded {len(df):,} rows, {len(df.columns):,} columns")
except FileNotFoundError:
    print(f"ERROR: File not found: {INPUT_FILE}")
    print("Please update INPUT_FILE path in the script.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

print("\n" + "-"*80)
print("Renaming columns...")

# Check which columns exist in the dataframe
existing_renames = {old: new for old, new in rename_map.items() if old in df.columns}
missing_columns = [old for old in rename_map.keys() if old not in df.columns]

print(f"\nFound {len(existing_renames)} columns to rename")
if missing_columns:
    print(f"{len(missing_columns)} columns from map not found in data")
    print(f"   First few: {missing_columns[:5]}")

# Rename columns
df_renamed = df.rename(columns=existing_renames)

print("\n" + "-"*80)
print("BEFORE vs AFTER column names (sample):")
print("-"*80)

sample_comparisons = [
    ('amount_cad.x', 'abm_amount'),
    ('amount_cad.y', 'card_amount'),
    ('country.x', 'abm_country'),
    ('country.y', 'customer_country'),
    ('country.x.x', 'business_country'),
]

for old, new in sample_comparisons:
    if old in df.columns:
        status = "✓" if old in existing_renames else "✗"
        print(f"  {status} {old:30s} → {new}")

# Save renamed file

print("\n" + "-"*80)
print(f"Saving renamed data to: {OUTPUT_FILE}")

try:
    df_renamed.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved!")
    print(f"  Rows: {len(df_renamed):,}")
    print(f"  Columns: {len(df_renamed.columns):,}")
except Exception as e:
    print(f"ERROR saving file: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nOriginal file: {INPUT_FILE}")
print(f"New file: {OUTPUT_FILE}")
print(f"Columns renamed: {len(existing_renames)}")

print("\nNew column categories:")
print("- Transaction types: abm_*, card_*, cheque_*, eft_*, emt_*, westernunion_*, wire_*")
print("- Individual KYC: customer_*")
print("- Business KYC: business_*")
print("- Indicators: from_*")

# Generate a column mapping reference file

print("\nAlso generating a column mapping reference file...")
mapping_df = pd.DataFrame([
    {'Original': old, 'Renamed': new, 'Category': new.split('_')[0]}
    for old, new in existing_renames.items()
])
mapping_df.to_csv('column_mapping_reference.csv', index=False)
print("Saved to: column_mapping_reference.csv")

