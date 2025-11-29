import pandas as pd

df = pd.read_csv("training_data.csv")

print("\n=== CSV COLUMN NAMES ===")
print(df.columns.tolist())

print("\n=== FIRST 5 ROWS ===")
print(df.head())
