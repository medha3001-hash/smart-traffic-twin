import pandas as pd

# Load the CSV file into Python
# Think of 'df' as a table — like an Excel sheet in Python
# 'df' stands for DataFrame — the standard name data scientists use
df = pd.read_csv('data/traffic.csv')

# ---- STEP 1: See the size of our dataset ----
print("=== DATASET SIZE ===")
print(f"Rows: {df.shape[0]}")      # How many rows (readings)
print(f"Columns: {df.shape[1]}")   # How many columns

# ---- STEP 2: See the column names ----
print("\n=== COLUMN NAMES ===")
print(df.columns.tolist())

# ---- STEP 3: See the first 5 rows ----
print("\n=== FIRST 5 ROWS ===")
print(df.head())

# ---- STEP 4: See basic statistics ----
# This tells us min, max, average vehicle counts etc.
print("\n=== BASIC STATISTICS ===")
print(df.describe())

# ---- STEP 5: Check for missing values ----
# Missing values can break our ML model later, so we check now
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# ---- STEP 6: See data types of each column ----
print("\n=== DATA TYPES ===")
print(df.dtypes)