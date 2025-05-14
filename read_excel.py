#!/usr/bin/env python
# Simple script to read the Excel file

import pandas as pd
import os

excel_path = "Spine-Mets-CT-SEG_Clinical.xlsx"

if not os.path.exists(excel_path):
    print(f"File {excel_path} not found!")
else:
    print(f"Found file: {excel_path}")
    try:
        df = pd.read_excel(excel_path)
        print(f"Successfully loaded Excel file with {len(df)} rows")
        print("\nColumns:")
        for col in df.columns:
            print(f"- {col}")
        
        # Print first 5 rows
        print("\nFirst 5 rows:")
        print(df.head())
    except Exception as e:
        print(f"Error reading Excel file: {e}")
