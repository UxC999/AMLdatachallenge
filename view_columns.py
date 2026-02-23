#!/usr/bin/env python3
"""
Column Viewer - See all columns in your CSV and search for specific ones
"""

import pandas as pd
import sys

def view_columns(filepath, search_term=None):
    """
    View all columns in a CSV file and optionally search for specific ones.
    """
    
    print("=" * 80)
    print("CSV COLUMN VIEWER")
    print("=" * 80)
    
    print(f"\nFile: {filepath}")
    print("\nLoading column names (reading first row only)...")
    
    # Read just the first row to get column names
    df = pd.read_csv(filepath, nrows=1)
    
    columns = df.columns.tolist()
    
    print(f"\n✓ Found {len(columns)} columns\n")
    
    # Print all columns
    print("=" * 80)
    print("ALL COLUMNS:")
    print("=" * 80)
    
    for i, col in enumerate(columns, 1):
        print(f"{i:3d}. {col}")
    
    # Search for specific terms if provided
    if search_term:
        print("\n" + "=" * 80)
        print(f"SEARCH RESULTS for '{search_term}':")
        print("=" * 80)
        
        search_lower = search_term.lower()
        matches = [col for col in columns if search_lower in col.lower()]
        
        if matches:
            print(f"\nFound {len(matches)} matching columns:")
            for col in matches:
                print(f"  ✓ {col}")
        else:
            print(f"\n✗ No columns found containing '{search_term}'")
    
    # Auto-search for common indicator/label columns
    print("\n" + "=" * 80)
    print("COMMON INDICATOR/LABEL COLUMNS:")
    print("=" * 80)
    
    keywords = ['label', 'indicator', 'flag', 'from_labels', 'risk', 'suspicious', 'alert']
    
    found_any = False
    for keyword in keywords:
        matches = [col for col in columns if keyword in col.lower()]
        if matches:
            found_any = True
            print(f"\nColumns containing '{keyword}':")
            for col in matches:
                print(f"  ✓ {col}")
    
    if not found_any:
        print("\n✗ No indicator/label columns found")
        print("\nCommon column names to look for:")
        print("  - 'label' or 'labels'")
        print("  - 'from_labels'")
        print("  - 'indicator'")
        print("  - 'flag' or 'flagged'")
        print("  - 'risk' or 'risk_score'")
        print("  - 'suspicious'")
        print("  - 'alert'")
    
    # Show sample of data for label columns if they exist
    label_cols = [col for col in columns if 'label' in col.lower()]
    
    if label_cols:
        print("\n" + "=" * 80)
        print("SAMPLE VALUES FROM LABEL COLUMNS:")
        print("=" * 80)
        
        # Read a few more rows to show examples
        df_sample = pd.read_csv(filepath, nrows=100)
        
        for col in label_cols:
            print(f"\n{col}:")
            print(f"  Non-null values: {df_sample[col].notna().sum()} / 100")
            print(f"  Unique values: {df_sample[col].nunique()}")
            
            if df_sample[col].notna().sum() > 0:
                print(f"  Sample values: {df_sample[col].dropna().unique()[:10].tolist()}")
            else:
                print(f"  (All null in first 100 rows)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_columns.py <csv_file> [search_term]")
        print("\nExamples:")
        print("  python view_columns.py cleaned_data.csv")
        print("  python view_columns.py cleaned_data.csv label")
        print("  python view_columns.py cleaned_data.csv indicator")
        sys.exit(1)
    
    filepath = sys.argv[1]
    search_term = sys.argv[2] if len(sys.argv) > 2 else None
    
    view_columns(filepath, search_term)
