import pandas as pd

# Read CSV
df = pd.read_csv("data.csv")

# Drop rows with any NaN or missing values
df_cleaned = df.dropna()

# Save cleaned data
df_cleaned.to_csv("new_removed.csv", index=False)

print(f"Removed incomplete rows. Cleaned data saved as new.csv")
print(f"Original rows: {len(df)} → Cleaned rows: {len(df_cleaned)}")
