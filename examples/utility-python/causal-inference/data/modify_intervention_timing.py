#!/usr/bin/env python3
"""
Script to modify the intervention timing in the longitudinal dataset.
Changes LicensingWeek from 6,7 to 5 and updates Post variable accordingly.
"""

import pandas as pd
import numpy as np

def modify_intervention_timing(input_file, output_file):
    """
    Modify the intervention timing to happen around week 5.
    
    Parameters:
    - input_file: path to the original CSV file
    - output_file: path to save the modified CSV file
    """
    
    # Read the data
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Display current intervention structure
    print("\nCurrent intervention structure:")
    licensed_summary = df[df['Licensed'] == 1].groupby(['PersonID', 'LicensingWeek'])['Week'].agg(['min', 'max']).reset_index()
    print(licensed_summary)
    
    # Check current Post variable timing
    post_summary = df[df['Post'] == 1].groupby(['PersonID'])['Week'].agg(['min', 'max']).reset_index()
    print(f"\nCurrent Post=1 timing:")
    print(post_summary)
    
    # Modify intervention timing
    print(f"\nModifying intervention timing...")
    
    # For people who are Licensed (treatment group), change LicensingWeek to 5
    # Currently Person 6 has LicensingWeek=6, Person 7 has LicensingWeek=7
    
    # Update LicensingWeek to 5 for all licensed individuals
    df.loc[df['Licensed'] == 1, 'LicensingWeek'] = 5
    
    # Update Post variable: Post=1 when Week >= 5 for licensed individuals
    # Reset Post to 0 first for licensed individuals
    df.loc[df['Licensed'] == 1, 'Post'] = 0
    
    # Set Post=1 when Week >= 5 for licensed individuals
    licensed_mask = df['Licensed'] == 1
    post_intervention_mask = df['Week'] >= 5
    df.loc[licensed_mask & post_intervention_mask, 'Post'] = 1
    
    # Verify the changes
    print(f"\nNew intervention structure:")
    licensed_summary_new = df[df['Licensed'] == 1].groupby(['PersonID', 'LicensingWeek'])['Week'].agg(['min', 'max']).reset_index()
    print(licensed_summary_new)
    
    post_summary_new = df[df['Post'] == 1].groupby(['PersonID'])['Week'].agg(['min', 'max']).reset_index()
    print(f"\nNew Post=1 timing:")
    print(post_summary_new)
    
    # Save the modified data
    print(f"\nSaving modified data to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print(f"✓ Successfully modified intervention timing!")
    print(f"  • Intervention now happens at week 5 for all licensed individuals")
    print(f"  • Post-intervention period starts from week 5 onwards")
    print(f"  • Pre-intervention period: weeks 0-4")
    print(f"  • Post-intervention period: weeks 5-15")

if __name__ == "__main__":
    input_file = "longitudinal_data.csv"
    output_file = "longitudinal_data.csv"  # Overwrite the original file
    
    modify_intervention_timing(input_file, output_file)
