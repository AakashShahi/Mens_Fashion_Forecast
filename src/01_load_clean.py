import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_SALES = os.path.join(DATA_DIR, "sales_data_1yr_nepal.csv")
RAW_SOCIAL = os.path.join(DATA_DIR, "social_media_1yr_nepal.csv")
RAW_TRENDS = os.path.join(DATA_DIR, "google_trends_1yr_nepal.csv")

COHORT_SALES = os.path.join(DATA_DIR, "sales_cohort_kathmandu_male_17_25.csv")
COHORT_SOCIAL = os.path.join(DATA_DIR, "social_cohort_kathmandu_male_17_25.csv")
COHORT_TRENDS = os.path.join(DATA_DIR, "google_trends_1yr_nepal_filtered.csv")

FEATURE_DIR = os.path.join(DATA_DIR, "features")


# --------------------------------------------------------------
# 🔍 Detect File Type Based on Columns
# --------------------------------------------------------------
def detect_file_type(df):
    cols = set(df.columns)

    if {"Date", "ItemID", "Category", "UnitsSold"}.issubset(cols):
        return "sales_raw"

    if {"PostDate", "PostID", "Caption", "Hashtags"}.issubset(cols):
        return "social_raw"

    if {"Date", "Region", "Keyword", "InterestScore"}.issubset(cols):
        return "trends_raw"

    if {"Date", "UnitsSold"}.issubset(cols) and "InterestScore" in cols:
        return "feature_category"

    return "unknown"


# --------------------------------------------------------------
# 🔧 Cohort Filter (Kathmandu, Male, 17-25)
# --------------------------------------------------------------
def apply_cohort_filter_sales(df):
    # User requested specifically 17-25 Male
    # We include 15-19 and 20-24 for backward compatibility if data comes in split format
    ages = ["15-19", "20-24", "17-25", "17-24", "18-25"] 
    
    # Normalize inputs
    df["City"] = df["City"].str.strip().str.title()
    df["Gender"] = df["Gender"].str.strip().str.title()
    df["AgeGroup"] = df["AgeGroup"].str.strip()
    
    return df[
        (df["City"] == "Kathmandu") &
        (df["Gender"] == "Male") &
        (df["AgeGroup"].isin(ages))
    ]


def apply_cohort_filter_social(df):
    ages = ["15-19", "20-24", "17-25", "17-24", "18-25"]
    return df[
        (df["City"] == "Kathmandu") &
        (df["Gender"] == "Male") &
        (df["AgeGroup"].isin(ages))
    ]


def apply_filter_trends(df):
    return df[df["Region"].str.contains("Nepal", na=False)]


# --------------------------------------------------------------
# 🚀 Intelligent CSV Processor
# --------------------------------------------------------------
def process_uploaded_csv(path):
    print(f"\n📌 Processing uploaded CSV: {path}\n")

    df = pd.read_csv(path)
    file_type = detect_file_type(df)

    print(f"🔍 Detected type: {file_type}")

    # ------------------------
    # RAW SALES (Full Year)
    # ------------------------
    if file_type == "sales_raw":
        df_c = apply_cohort_filter_sales(df)
        df_c.to_csv(COHORT_SALES, index=False)
        print(f"✅ Saved cohort sales → {COHORT_SALES}")
        return "sales_raw"

    # ------------------------
    # RAW SOCIAL
    # ------------------------
    if file_type == "social_raw":
        df_c = apply_cohort_filter_social(df)
        df_c.to_csv(COHORT_SOCIAL, index=False)
        print(f"✅ Saved cohort social → {COHORT_SOCIAL}")
        return "social_raw"

    # ------------------------
    # RAW GOOGLE TRENDS
    # ------------------------
    if file_type == "trends_raw":
        df_c = apply_filter_trends(df)
        df_c.to_csv(COHORT_TRENDS, index=False)
        print(f"✅ Saved filtered trends → {COHORT_TRENDS}")
        return "trends_raw"

    # ------------------------
    # CATEGORY FEATURE FILE
    # ------------------------
    if file_type == "feature_category":
        # Detect category name from filename
        filename = os.path.basename(path)
        name = filename.replace("feat_", "").replace(".csv", "")
        name = name.replace("_", " ").title()

        out_path = os.path.join(FEATURE_DIR, f"feat_{name.lower().replace(' ', '_')}.csv")
        df.to_csv(out_path, index=False)

        print(f"📦 Updated feature file for category: {name}")
        print(f"➡ Saved to {out_path}")
        return "feature_category"

    # ------------------------
    # Unknown file
    # ------------------------
    print("⚠ Unknown CSV format — skipping.")
    return "unknown"



# --------------------------------------------------------------
# 📊 Data Profiling & Dictionary Generation (Phase 2)
# --------------------------------------------------------------
def generate_data_dictionary(df, name):
    """
    Creates a markdown dictionary of the dataset's columns and types.
    """
    dict_path = os.path.join(DATA_DIR, f"dictionary_{name}.md")
    
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write(f"# Data Dictionary: {name}\n\n")
        f.write(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}\n\n")
        f.write("| Column | Type | Missing (%) | Description/Example |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isnull().mean() * 100
            example = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A"
            # Simple heuristic descriptions
            desc = "Unique Identifier" if "ID" in col else "Metric/Value"
            if "Date" in col: desc = "Time of record"
            if "Category" in col: desc = "Product Category"
            if "City" in col: desc = "Location"
            
            f.write(f"| **{col}** | `{dtype}` | {missing:.1f}% | {desc} (e.g., '{example}') |\n")
    
    print(f"📘 Generated Dictionary: {dict_path}")


def simple_profile_report(df, name):
    """
    Generates a simple text-based profile of the data distribution.
    """
    report_path = os.path.join(DATA_DIR, f"profile_{name}.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"DATA PROFILE REPORT: {name}\n")
        f.write("=" * 40 + "\n\n")
        
        # Numerical Summary
        nums = df.select_dtypes(include=['number'])
        if not nums.empty:
            f.write("NUMERICAL SUMMARY:\n")
            f.write(nums.describe().to_string())
            f.write("\n\n")
        
        # Categorical Summary
        cats = df.select_dtypes(include=['object', 'category'])
        if not cats.empty:
            f.write("CATEGORICAL SUMMARY (Top 3):\n")
            for col in cats.columns:
                top = df[col].value_counts().head(3).to_dict()
                f.write(f"- {col}: {top}\n")
            f.write("\n")
            
    print(f"📊 Generated Profile: {report_path}")



# --------------------------------------------------------------
# 🧹 Data Cleaning & Preparation (Phase 3)
# --------------------------------------------------------------
def impute_missing_values(df):
    """
    Handles missing values:
    - Numerical: Fill with Median
    - Categorical: Fill with Mode
    - Drops columns with > 50% missing
    """
    print("   ... Imputing missing values")
    # Drop columns with too many missing values
    threshold = 0.5 * len(df)
    df = df.dropna(thresh=threshold, axis=1)
    
    # Numerical Imputation
    nums = df.select_dtypes(include=[np.number]).columns
    for col in nums:
        if df[col].isnull().sum() > 0:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            
    # Categorical Imputation
    cats = df.select_dtypes(include=['object', 'category']).columns
    for col in cats:
        if df[col].isnull().sum() > 0:
            mod = df[col].mode()[0]
            df[col] = df[col].fillna(mod)
            
    return df

def treat_outliers(df, columns=None):
    """
    Caps outliers using IQR method for specified numerical columns.
    """
    print("   ... Treating outliers (IQR capping)")
    if not columns:
        columns = df.select_dtypes(include=[np.number]).columns

    for col in columns:
        # Skip ID columns or binary
        if "ID" in col or df[col].nunique() < 5:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # Cap
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
        
    return df

def clean_data_pipeline(df, name):
    print(f"🧹 Starting Data Cleaning for {name}...")
    df = impute_missing_values(df)
    df = treat_outliers(df)
    return df

# --------------------------------------------------------------
# 🏁 MAIN function (used by pipeline)
# --------------------------------------------------------------
def main(uploaded_files=None):
    print("===== 01_load_clean.py START (Phase 2: Profiling & Phase 3: Cleaning) =====")

    # If called by Streamlit with uploaded files
    if uploaded_files:
        results = {}
        for f in uploaded_files:
            file_type = process_uploaded_csv(f)
            results[f] = file_type
        return results

    # ------------------------------------------------------------------
    # DYNAMIC DATA LOGIC:
    # If a cohort file already exists (likely uploaded by user via App),
    # use it as the source of truth instead of overwriting from Raw.
    # ------------------------------------------------------------------
    if os.path.exists(COHORT_SALES):
        print(f"🔄 DETECTED EXISTING COHORT DATA: {COHORT_SALES}")
        print("   -> Skipping Raw Data Overwrite to preserve User Uploads.")
        print("   -> Proceeding to Feature Engineering (Next Step)...")
        # We can still run cleaning on the existing file to be safe
        print("   -> Re-verifying/Cleaning existing cohort data...")
        
        df = pd.read_csv(COHORT_SALES)
        df = clean_data_pipeline(df, "cohort_sales (existing)")
        df.to_csv(COHORT_SALES, index=False)
        generate_data_dictionary(df, "cohort_sales")
        
    else:
        # Fallback: Create from Raw if no cohort file exists
        print("⚠️ No cohort file found. Generating from RAW defaults...")
        
        # 1. Sales
        if os.path.exists(RAW_SALES):
            raw_sales = pd.read_csv(RAW_SALES)
            generate_data_dictionary(raw_sales, "raw_sales")
            simple_profile_report(raw_sales, "raw_sales")
            
            cohort_sales = apply_cohort_filter_sales(raw_sales)
            cohort_sales = clean_data_pipeline(cohort_sales, "cohort_sales")
            
            cohort_sales.to_csv(COHORT_SALES, index=False)
            generate_data_dictionary(cohort_sales, "cohort_sales")
        else:
            print(f"❌ Critical Error: Raw Sales missing at {RAW_SALES}")

    # 2. Social (Always re-process raw for now unless we add upload for social)
    if os.path.exists(RAW_SOCIAL):
        raw_social = pd.read_csv(RAW_SOCIAL)
        generate_data_dictionary(raw_social, "raw_social")
        cohort_social = apply_cohort_filter_social(raw_social)
        cohort_social = impute_missing_values(cohort_social) 
        cohort_social.to_csv(COHORT_SOCIAL, index=False)

    # 3. Trends (Always re-process raw for now unless we add upload for trends)
    if os.path.exists(RAW_TRENDS):
        raw_trends = pd.read_csv(RAW_TRENDS)
        cohort_trends = apply_filter_trends(raw_trends)
        cohort_trends = clean_data_pipeline(cohort_trends, "cohort_trends")
        cohort_trends.to_csv(COHORT_TRENDS, index=False)
        generate_data_dictionary(cohort_trends, "cohort_trends")

    print("✅ Completed Data Phase")
    print("===== 01_load_clean.py END =====")



if __name__ == "__main__":
    main()
