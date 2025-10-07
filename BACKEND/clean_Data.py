import pandas as pd

# Read CSV
df = pd.read_csv("data.csv")

# Compute medians (ignoring NaN) for each column
medians = df.median(numeric_only=True)

# Replace NaN with column median
df_filled = df.fillna(medians)

# Save cleaned data
df_filled.to_csv("new.csv", index=False)

print("Cleaned data saved as new.csv")
