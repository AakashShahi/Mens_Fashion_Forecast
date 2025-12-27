
import pandas as pd
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FILE_PATH = os.path.join(DATA_DIR, "sales_cohort_kathmandu_male_17_25.csv")

# Valid Criteria
VALID_GENDERS = ["male"]
VALID_AGES = ["17-25", "15-19", "20-24", "17-24", "18-25"]
VALID_CITY = "kathmandu"

def clean_data():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    print(f"Reading {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH)
    original_count = len(df)

    # Normalize
    df["Gender"] = df["Gender"].str.strip().str.lower()
    df["AgeGroup"] = df["AgeGroup"].str.strip()
    df["City"] = df["City"].str.strip().str.lower()

    # Filter
    mask = (
        (df["Gender"].isin(VALID_GENDERS)) &
        (df["AgeGroup"].isin(VALID_AGES)) &
        (df["City"] == VALID_CITY)
    )
    
    clean_df = df[mask]
    
    # Restore casing for aesthetics if needed, but standardizing is fine
    clean_df["Gender"] = "Male"
    clean_df["City"] = "Kathmandu"
    
    cleaned_count = len(clean_df)
    dropped_count = original_count - cleaned_count
    
    print(f"Rows Before: {original_count}")
    print(f"Rows After:  {cleaned_count}")
    print(f"Dropped:     {dropped_count}")
    
    if dropped_count > 0:
        clean_df.to_csv(FILE_PATH, index=False)
        print("✅ File updated with cleaned data.")
    else:
        print("✨ Data was already clean.")

if __name__ == "__main__":
    clean_data()
