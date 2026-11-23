import pandas as pd
from pathlib import Path
df = pd.read_csv("data/raw/Kolkata.csv")
print("Original dataset shape:", df.shape)
df = df.drop_duplicates()
df = df.dropna(subset=["Price", "Area"])
df = df[(df["Price"] > 0) & (df["Area"] > 0)]
output_path = Path("data/processed")
output_path.mkdir(parents=True, exist_ok=True)
df.to_csv(
    output_path / "Kolkata_cleaned.csv",
    index=False
)
print("\nCleaning completed!")
print("Cleaned dataset shape:", df.shape)
print("Remaining duplicates:", df.duplicated().sum())
print("Saved: data/processed/Kolkata_cleaned.csv")