#!/usr/bin/env python3
"""
Occupation & Business-Based AML Hypotheses
Based on manual review findings: students, unemployed, shell companies
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Load transaction data"""
    print("=" * 80)
    print("OCCUPATION & BUSINESS-BASED AML DETECTION")
    print("=" * 80)
    
    print(f"\nReading: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    
    print(f"\n✓ Loaded {len(df):,} transactions")
    print(f"  Unique customers: {df['customer_id'].nunique():,}")
    
    if 'label' not in df.columns:
        print("\nERROR: 'label' column not found!")
        return None
    
    labeled_customers = df[df['label'].notna()]['customer_id'].nunique()
    suspicious_customers = df[df['label'] == 1]['customer_id'].nunique()
    
    print(f"  Labeled customers: {labeled_customers:,}")
    print(f"  Suspicious (label=1): {suspicious_customers}")
    
    return df

def prepare_data(df):
    """Prepare unified columns and customer-level data"""
    print("\n" + "=" * 80)
    print("PREPARING DATA")
    print("=" * 80)
    
    # Create unified amount
    amount_cols = ['card_amount', 'abm_amount', 'cheque_amount', 
                   'eft_amount', 'emt_amount', 'westernunion_amount', 'wire_amount']
    
    df['amount'] = pd.Series(dtype='float64')
    for col in amount_cols:
        if col in df.columns:
            df['amount'] = df['amount'].combine_first(df[col])
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['amount_abs'] = df['amount'].abs()
    
    print(f"✓ Amount: {df['amount'].notna().sum():,} non-null")
    
    # Aggregate to customer level
    customer_agg = df.groupby('customer_id').agg({
        'amount_abs': ['sum', 'mean', 'count'],
        'label': 'first',
        'customer_occupation_code': 'first',
        'customer_occupation_title': 'first',
        'customer_income': 'first',
        'business_sales': 'first',
        'business_employee_count': 'first',
        'from_kyc_smallbusiness': 'any',
    }).reset_index()
    
    # Flatten column names
    customer_agg.columns = ['customer_id', 'total_amount', 'avg_amount', 'txn_count',
                            'label', 'occupation_code', 'occupation_title', 'income',
                            'business_sales', 'business_employees', 'has_business']
    
    print(f"✓ Aggregated {len(customer_agg):,} customers")
    
    return df, customer_agg

def hypothesis_15_student_money_mules(customer_agg):
    """
    HYPOTHESIS 15: Students with unusually high transaction volumes (money mules)
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 15: STUDENT MONEY MULES")
    print("=" * 80)
    
    print("\nTheory: Money mules are often students receiving/spending far beyond their means")
    print("Prediction: Students with >$50K total transactions are suspicious")
    
    # Identify students
    if 'occupation_code' not in customer_agg.columns and 'occupation_title' not in customer_agg.columns:
        print("\nNo occupation data - SKIPPING")
        return None
    
    # Look for students in occupation code or title
    student_mask = pd.Series(False, index=customer_agg.index)
    
    if 'occupation_code' in customer_agg.columns:
        student_mask |= customer_agg['occupation_code'].astype(str).str.upper().str.contains('STUDENT|STUD', na=False)
    
    if 'occupation_title' in customer_agg.columns:
        student_mask |= customer_agg['occupation_title'].astype(str).str.upper().str.contains('STUDENT|STUD', na=False)
    
    students = customer_agg[student_mask].copy()
    
    if len(students) == 0:
        print("\nNo students found in data - SKIPPING")
        return None
    
    print(f"\nFound {len(students)} students")
    
    # Check transaction volumes
    print(f"\nStudent transaction statistics:")
    print(f"  Mean total: ${students['total_amount'].mean():,.2f}")
    print(f"  Median total: ${students['total_amount'].median():,.2f}")
    print(f"  Max total: ${students['total_amount'].max():,.2f}")
    
    # Check how many students have labels
    labeled_students = students[students['label'].notna()]
    suspicious_students = students[students['label'] == 1]
    
    print(f"\nLabeled students: {len(labeled_students)}")
    if len(suspicious_students) > 0:
        print(f"  Suspicious students: {len(suspicious_students)}")
        print(f"\nSuspicious student details:")
        for idx, row in suspicious_students.iterrows():
            print(f"  {row['customer_id']}: ${row['total_amount']:,.0f} total, {row['txn_count']:.0f} transactions")
    
    # Flag students with >$50K
    THRESHOLD = 50000
    high_volume_students = students[students['total_amount'] > THRESHOLD]
    
    print(f"\nStudents with >${THRESHOLD:,} total transactions: {len(high_volume_students)}")
    
    # Statistical comparison: suspicious students vs normal students
    normal_students = students[students['label'] == 0]
    
    if len(suspicious_students) > 0 and len(normal_students) > 0:
        print(f"\nStatistical comparison (suspicious students vs normal students):")
        print(f"  Suspicious student median: ${suspicious_students['total_amount'].median():,.2f}")
        print(f"  Normal student median: ${normal_students['total_amount'].median():,.2f}")
        
        # Mann-Whitney U test
        t_stat, p_value = stats.mannwhitneyu(
            suspicious_students['total_amount'], 
            normal_students['total_amount'], 
            alternative='two-sided'
        )
        
        print(f"  Mann-Whitney U test: p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  STATISTICALLY SIGNIFICANT (p < 0.05)")
            statistical_conclusion = 'STATISTICALLY SUPPORTED'
        else:
            print(f"  Not statistically significant (p >= 0.05)")
            statistical_conclusion = 'NOT STATISTICALLY SIGNIFICANT'
    else:
        p_value = None
        statistical_conclusion = 'INSUFFICIENT DATA'
    
    # Detection rate
    if len(suspicious_students) > 0:
        caught = high_volume_students[high_volume_students['label'] == 1]
        recall = len(caught) / len(suspicious_students) * 100
        
        print(f"\nDetection performance:")
        print(f"  Flagged: {len(high_volume_students)} students")
        print(f"  Caught: {len(caught)}/{len(suspicious_students)} suspicious students ({recall:.1f}%)")
        
        # Add flag to customer_agg
        customer_agg['flag_student_mule'] = customer_agg['customer_id'].isin(high_volume_students['customer_id'])
        
        return {
            'hypothesis': 'Student Money Mules',
            'conclusion': statistical_conclusion,
            'p_value': p_value,
            'recall': recall,
            'flagged': len(high_volume_students),
            'caught': len(caught)
        }
    else:
        print("\n⚠️ No suspicious students to validate against")
        return None

def hypothesis_16_unemployed_with_income(customer_agg):
    """
    HYPOTHESIS 16: Unemployed customers reporting income (fraud indicator)
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 16: UNEMPLOYED WITH REPORTED INCOME")
    print("=" * 80)
    
    print("\nTheory: Unemployed people shouldn't have employment income")
    print("Prediction: Unemployed with income >$0 are fraudulent/suspicious")
    
    if 'occupation_code' not in customer_agg.columns and 'occupation_title' not in customer_agg.columns:
        print("\nNo occupation data - SKIPPING")
        return None
    
    if 'income' not in customer_agg.columns:
        print("\nNo income data - SKIPPING")
        return None
    
    # Identify unemployed
    unemployed_mask = pd.Series(False, index=customer_agg.index)
    
    if 'occupation_code' in customer_agg.columns:
        unemployed_mask |= customer_agg['occupation_code'].astype(str).str.upper().str.contains('UNEMPLOY|UNEMP|NONE', na=False)
    
    if 'occupation_title' in customer_agg.columns:
        unemployed_mask |= customer_agg['occupation_title'].astype(str).str.upper().str.contains('UNEMPLOY|UNEMP|NONE', na=False)
    
    unemployed = customer_agg[unemployed_mask].copy()
    
    if len(unemployed) == 0:
        print("\nNo unemployed customers found - SKIPPING")
        return None
    
    print(f"\nFound {len(unemployed)} unemployed customers")
    
    # Check who has income
    unemployed_with_income = unemployed[(unemployed['income'].notna()) & (unemployed['income'] > 0)]
    
    print(f"Unemployed with income >$0: {len(unemployed_with_income)}")
    
    if len(unemployed_with_income) > 0:
        print(f"\nIncome statistics for unemployed:")
        print(f"  Mean: ${unemployed_with_income['income'].mean():,.2f}")
        print(f"  Median: ${unemployed_with_income['income'].median():,.2f}")
        print(f"  Max: ${unemployed_with_income['income'].max():,.2f}")
        
        # Statistical comparison: unemployed with income vs employed with income
        employed = customer_agg[~unemployed_mask & customer_agg['income'].notna() & (customer_agg['income'] > 0)]
        
        if len(unemployed_with_income) > 0 and len(employed) > 0:
            print(f"\nStatistical comparison (unemployed with income vs employed with income):")
            print(f"  Unemployed median income: ${unemployed_with_income['income'].median():,.2f}")
            print(f"  Employed median income: ${employed['income'].median():,.2f}")
            
            # Mann-Whitney U test
            t_stat, p_value = stats.mannwhitneyu(
                unemployed_with_income['income'], 
                employed['income'], 
                alternative='two-sided'
            )
            
            print(f"  Mann-Whitney U test: p-value = {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"  STATISTICALLY SIGNIFICANT (p < 0.05)")
                statistical_conclusion = 'STATISTICALLY SUPPORTED'
            else:
                print(f"  Not statistically significant (p >= 0.05)")
                statistical_conclusion = 'NOT STATISTICALLY SIGNIFICANT'
        else:
            p_value = None
            statistical_conclusion = 'INSUFFICIENT DATA'
        
        # Check labels
        suspicious_unemployed = unemployed_with_income[unemployed_with_income['label'] == 1]
        
        if len(suspicious_unemployed) > 0:
            print(f"\nFound {len(suspicious_unemployed)} suspicious unemployed with income!")
            for idx, row in suspicious_unemployed.iterrows():
                print(f"  {row['customer_id']}: Income=${row['income']:,.0f}, Transactions=${row['total_amount']:,.0f}")
            
            # Add flag
            customer_agg['flag_unemployed_income'] = customer_agg['customer_id'].isin(unemployed_with_income['customer_id'])
            
            # Calculate recall
            all_suspicious = customer_agg[customer_agg['label'] == 1]
            recall = len(suspicious_unemployed) / len(all_suspicious) * 100
            
            return {
                'hypothesis': 'Unemployed with Income',
                'conclusion': statistical_conclusion,
                'p_value': p_value,
                'recall': recall,
                'flagged': len(unemployed_with_income),
                'caught': len(suspicious_unemployed)
            }
    
    print("\n✗ No suspicious unemployed with income found")
    return None

def hypothesis_17_shell_companies(customer_agg):
    """
    HYPOTHESIS 17: Shell companies with suspicious sales-to-employee ratios
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 17: SHELL COMPANY INDICATORS")
    print("=" * 80)
    
    print("\nTheory: Legitimate businesses have reasonable sales per employee")
    print("Prediction: Sales-to-employee ratio >$500K indicates shell company")
    
    if 'business_sales' not in customer_agg.columns or 'business_employees' not in customer_agg.columns:
        print("\nNo business data - SKIPPING")
        return None
    
    # Filter to businesses only
    businesses = customer_agg[customer_agg['has_business'] == True].copy()
    
    if len(businesses) == 0:
        print("\nNo business customers - SKIPPING")
        return None
    
    print(f"\nFound {len(businesses)} business customers")
    
    # Calculate sales per employee
    businesses = businesses[
        (businesses['business_sales'].notna()) & 
        (businesses['business_employees'].notna()) &
        (businesses['business_employees'] > 0)
    ].copy()
    
    businesses['sales_per_employee'] = businesses['business_sales'] / businesses['business_employees']
    
    print(f"Businesses with valid sales/employee data: {len(businesses)}")
    
    if len(businesses) > 0:
        print(f"\nSales per employee statistics:")
        print(f"  Mean: ${businesses['sales_per_employee'].mean():,.0f}")
        print(f"  Median: ${businesses['sales_per_employee'].median():,.0f}")
        print(f"  Max: ${businesses['sales_per_employee'].max():,.0f}")
        
        # Flag high ratio
        THRESHOLD = 500000
        shell_companies = businesses[businesses['sales_per_employee'] > THRESHOLD]
        
        print(f"\nBusinesses with >${THRESHOLD:,} sales/employee: {len(shell_companies)}")
        
        if len(shell_companies) > 0:
            # Check labels
            suspicious_shells = shell_companies[shell_companies['label'] == 1]
            
            print(f"\nSuspicious shell companies:")
            for idx, row in shell_companies.iterrows():
                label_text = "SUSPICIOUS" if row['label'] == 1 else "Normal"
                print(f"  {row['customer_id']} ({label_text}): ${row['business_sales']:,.0f} sales / {row['business_employees']:.0f} employees = ${row['sales_per_employee']:,.0f} per employee")
            
            if len(suspicious_shells) > 0:
                # Add flag
                customer_agg['flag_shell_company'] = customer_agg['customer_id'].isin(shell_companies['customer_id'])
                
                all_suspicious = customer_agg[customer_agg['label'] == 1]
                recall = len(suspicious_shells) / len(all_suspicious) * 100
                
                return {
                    'hypothesis': 'Shell Company (High Sales/Employee)',
                    'conclusion': 'SUPPORTED',
                    'recall': recall,
                    'flagged': len(shell_companies),
                    'caught': len(suspicious_shells)
                }
    
    print("\n✗ No suspicious shell companies detected")
    return None

def hypothesis_18_zero_business_transactions(df, customer_agg):
    """
    HYPOTHESIS 18: Businesses with zero business transactions but high personal spending
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 18: BUSINESSES WITH ZERO BUSINESS ACTIVITY")
    print("=" * 80)
    
    print("\nTheory: Real businesses have business transactions")
    print("Prediction: Businesses with no business txns but high personal txns are shells")
    
    # Check which transaction types are business vs personal
    # Typically: CARD, ABM, EMT = personal; WIRE, CHEQUE, EFT = business
    
    if 'from_kyc_smallbusiness' not in df.columns:
        print("\nNo business flag - SKIPPING")
        return None
    
    businesses = customer_agg[customer_agg['has_business'] == True].copy()
    
    if len(businesses) == 0:
        print("\nNo business customers - SKIPPING")
        return None
    
    print(f"\nAnalyzing {len(businesses)} business customers...")
    
    # For each business customer, count business vs personal transactions
    business_txn_types = ['from_wire', 'from_cheque', 'from_eft']
    personal_txn_types = ['from_card', 'from_abm', 'from_emt']
    
    results = []
    for customer_id in businesses['customer_id']:
        cust_df = df[df['customer_id'] == customer_id]
        
        # Count business transactions
        business_txns = 0
        for txn_type in business_txn_types:
            if txn_type in cust_df.columns:
                business_txns += (cust_df[txn_type] == True).sum()
        
        # Count personal transactions
        personal_txns = 0
        for txn_type in personal_txn_types:
            if txn_type in cust_df.columns:
                personal_txns += (cust_df[txn_type] == True).sum()
        
        # Get total personal spending
        personal_amount = cust_df[
            (cust_df['from_card'] == True) | 
            (cust_df['from_abm'] == True) | 
            (cust_df['from_emt'] == True)
        ]['amount_abs'].sum() if 'amount_abs' in cust_df.columns else 0
        
        results.append({
            'customer_id': customer_id,
            'business_txns': business_txns,
            'personal_txns': personal_txns,
            'personal_spending': personal_amount
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.merge(businesses[['customer_id', 'label']], on='customer_id', how='left')
    
    # Flag: No business txns but >$100K personal spending
    THRESHOLD = 100000
    suspicious_pattern = results_df[
        (results_df['business_txns'] == 0) & 
        (results_df['personal_spending'] > THRESHOLD)
    ]
    
    print(f"\nBusinesses with ZERO business transactions: {(results_df['business_txns'] == 0).sum()}")
    print(f"Businesses with >${THRESHOLD:,} personal spending: {(results_df['personal_spending'] > THRESHOLD).sum()}")
    print(f"Businesses with BOTH (flagged): {len(suspicious_pattern)}")
    
    if len(suspicious_pattern) > 0:
        print(f"\nFlagged businesses:")
        for idx, row in suspicious_pattern.iterrows():
            label_text = "SUSPICIOUS" if row['label'] == 1 else "Normal"
            print(f"  {row['customer_id']} ({label_text}): {row['business_txns']} business txns, ${row['personal_spending']:,.0f} personal spending")
        
        # Check labels
        caught = suspicious_pattern[suspicious_pattern['label'] == 1]
        
        if len(caught) > 0:
            customer_agg['flag_zero_business_txns'] = customer_agg['customer_id'].isin(suspicious_pattern['customer_id'])
            
            all_suspicious = customer_agg[customer_agg['label'] == 1]
            recall = len(caught) / len(all_suspicious) * 100
            
            return {
                'hypothesis': 'Zero Business Transactions',
                'conclusion': 'SUPPORTED',
                'recall': recall,
                'flagged': len(suspicious_pattern),
                'caught': len(caught)
            }
    
    print("\n✗ No suspicious pattern detected")
    return None

def hypothesis_19_round_large_amounts(df, customer_agg):
    """
    HYPOTHESIS 19: Exact round large amounts ($10K, $100K, etc.)
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 19: EXACT ROUND LARGE AMOUNTS")
    print("=" * 80)
    
    print("\nTheory: Criminals use exact round amounts for easy tracking")
    print("Prediction: Frequent use of $10K, $20K, $50K, $100K indicates laundering")
    
    # Define round amounts
    ROUND_AMOUNTS = [10000, 20000, 50000, 100000, 200000]
    
    df['amount_abs'] = df['amount'].abs() if 'amount_abs' not in df.columns else df['amount_abs']
    
    # Check for exact round amounts
    df['is_exact_round_large'] = df['amount_abs'].isin(ROUND_AMOUNTS)
    
    total_round = df['is_exact_round_large'].sum()
    print(f"\nTransactions with exact round large amounts: {total_round:,}")
    
    if total_round > 0:
        print(f"\nBreakdown by amount:")
        for amt in ROUND_AMOUNTS:
            count = (df['amount_abs'] == amt).sum()
            if count > 0:
                print(f"  ${amt:,}: {count} transactions")
        
        # Customer level
        round_by_customer = df.groupby('customer_id')['is_exact_round_large'].agg(['sum', 'mean']).reset_index()
        round_by_customer.columns = ['customer_id', 'round_large_count', 'round_large_pct']
        round_by_customer['round_large_pct'] *= 100
        
        # Merge with labels
        round_by_customer = round_by_customer.merge(
            customer_agg[['customer_id', 'label', 'txn_count']], 
            on='customer_id', 
            how='left'
        )
        
        # Flag customers with >3 exact round large amounts OR >20% of their transactions
        flagged = round_by_customer[
            (round_by_customer['round_large_count'] >= 3) | 
            (round_by_customer['round_large_pct'] > 20)
        ]
        
        print(f"\nCustomers flagged: {len(flagged)}")
        
        if len(flagged) > 0:
            suspicious_flagged = flagged[flagged['label'] == 1]
            
            print(f"\nFlagged customer details:")
            for idx, row in flagged.head(20).iterrows():
                label_text = "SUSPICIOUS" if row['label'] == 1 else "Normal"
                print(f"  {row['customer_id']} ({label_text}): {row['round_large_count']:.0f} round large amounts ({row['round_large_pct']:.1f}% of {row['txn_count']:.0f} txns)")
            
            if len(suspicious_flagged) > 0:
                customer_agg['flag_round_large'] = customer_agg['customer_id'].isin(flagged['customer_id'])
                
                all_suspicious = customer_agg[customer_agg['label'] == 1]
                recall = len(suspicious_flagged) / len(all_suspicious) * 100
                
                return {
                    'hypothesis': 'Exact Round Large Amounts',
                    'conclusion': 'SUPPORTED',
                    'recall': recall,
                    'flagged': len(flagged),
                    'caught': len(suspicious_flagged)
                }
    
    print("\n✗ No suspicious pattern detected")
    return None

def combined_detection(customer_agg):
    """
    Combine all flags for final detection
    """
    print("\n" + "=" * 80)
    print("COMBINED DETECTION: ALL FLAGS")
    print("=" * 80)
    
    print("\nCombining all occupation & business-based flags...")
    
    # Count flags
    flag_cols = [col for col in customer_agg.columns if col.startswith('flag_')]
    
    if len(flag_cols) == 0:
        print("\n⚠️ No flags created")
        return
    
    print(f"Available flags: {len(flag_cols)}")
    for flag in flag_cols:
        count = customer_agg[flag].sum() if flag in customer_agg.columns else 0
        print(f"  {flag}: {count} customers")
    
    customer_agg['total_new_flags'] = customer_agg[flag_cols].sum(axis=1)
    
    # Detection performance
    all_suspicious = customer_agg[customer_agg['label'] == 1]
    
    print(f"\nDetection performance:")
    for threshold in [1, 2, 3]:
        flagged = customer_agg[customer_agg['total_new_flags'] >= threshold]
        caught = flagged[flagged['label'] == 1]
        
        if len(all_suspicious) > 0:
            recall = len(caught) / len(all_suspicious) * 100
            precision = len(caught) / len(flagged) * 100 if len(flagged) > 0 else 0
            
            print(f"\n  {threshold}+ flags:")
            print(f"    Flagged: {len(flagged)} customers")
            print(f"    Caught: {len(caught)}/{len(all_suspicious)} suspicious ({recall:.1f}% recall)")
            print(f"    Precision: {precision:.1f}%")
    
    # Show which suspicious customers were caught
    print(f"\n" + "=" * 80)
    print("SUSPICIOUS CUSTOMERS CAUGHT")
    print("=" * 80)
    
    for idx, row in all_suspicious.iterrows():
        flags_triggered = []
        for flag in flag_cols:
            if flag in row and row[flag]:
                flags_triggered.append(flag.replace('flag_', ''))
        
        if len(flags_triggered) > 0:
            print(f"\n✓ {row['customer_id']}: {len(flags_triggered)} flags")
            for flag in flags_triggered:
                print(f"    - {flag}")
        else:
            print(f"\n✗ {row['customer_id']}: NO FLAGS (missed)")

def main():
    if len(sys.argv) < 2:
        print("Usage: python occupation_business_hypotheses.py <data_file.csv>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Load data
    df = load_data(filepath)
    if df is None:
        sys.exit(1)
    
    # Prepare data
    df, customer_agg = prepare_data(df)
    
    # Test hypotheses
    results = []
    
    result15 = hypothesis_15_student_money_mules(customer_agg)
    if result15:
        results.append(result15)
    
    result16 = hypothesis_16_unemployed_with_income(customer_agg)
    if result16:
        results.append(result16)
    
    result17 = hypothesis_17_shell_companies(customer_agg)
    if result17:
        results.append(result17)
    
    result18 = hypothesis_18_zero_business_transactions(df, customer_agg)
    if result18:
        results.append(result18)
    
    result19 = hypothesis_19_round_large_amounts(df, customer_agg)
    if result19:
        results.append(result19)
    
    # Combined detection
    combined_detection(customer_agg)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: NEW HYPOTHESES")
    print("=" * 80)
    
    for i, result in enumerate(results, 15):
        print(f"\n{i}. {result['hypothesis']}")
        print(f"   Conclusion: {result['conclusion']}")
        if result.get('p_value') is not None:
            print(f"   Statistical significance: p = {result['p_value']:.4f}")
        print(f"   Recall: {result['recall']:.1f}% ({result['caught']}/{10} suspicious)")
        print(f"   Flagged: {result['flagged']} customers total")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
