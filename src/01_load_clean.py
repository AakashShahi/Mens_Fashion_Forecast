# src/01_load_clean.py
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data") if __file__ else "data"
SALES = os.path.join(DATA_DIR, "sales_data_1yr_nepal.csv")
SOCIAL = os.path.join(DATA_DIR, "social_media_1yr_nepal.csv")
TRENDS = os.path.join(DATA_DIR, "google_trends_1yr_nepal.csv")

OUT_SALES = os.path.join(DATA_DIR, "sales_cohort_kathmandu_male_17_25.csv")
OUT_SOCIAL = os.path.join(DATA_DIR, "social_cohort_kathmandu_male_17_25.csv")
OUT_TRENDS = os.path.join(DATA_DIR, "google_trends_1yr_nepal_filtered.csv")

def load_safe(path, date_cols=None):
    if date_cols:
        try:
            return pd.read_csv(path, parse_dates=date_cols)
        except Exception:
            df = pd.read_csv(path)
            for c in date_cols:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
            return df
    else:
        return pd.read_csv(path)

def main():
    print("Loading CSVs from:", DATA_DIR)
    sales = load_safe(SALES, date_cols=["Date"])
    social = load_safe(SOCIAL, date_cols=["PostDate"])
    trends = load_safe(TRENDS, date_cols=["Date"])

    print("Sales columns:", list(sales.columns))
    print("Social columns:", list(social.columns))
    print("Trends columns:", list(trends.columns))
    print("----")

    # Cohort: Kathmandu, Male, ages 15-19 & 20-24 (approx 17-25)
    cohort_ages = ["15-19", "20-24"]
    city_name = "Kathmandu"
    gender_value = "Male"

    sales_c = sales.copy()
    if "City" in sales_c.columns:
        sales_c = sales_c[sales_c["City"] == city_name]
    else:
        print("WARNING: 'City' missing in sales. No city filter applied.")

    if "AgeGroup" in sales_c.columns:
        sales_c = sales_c[sales_c["AgeGroup"].isin(cohort_ages)]
    else:
        print("WARNING: 'AgeGroup' missing in sales. No age filter applied.")

    if "Gender" in sales_c.columns:
        sales_c = sales_c[sales_c["Gender"] == gender_value]
    else:
        print("NOTICE: 'Gender' missing in sales. Proceeding with City+AgeGroup only.")

    social_c = social.copy()
    if "City" in social_c.columns:
        social_c = social_c[social_c["City"] == city_name]
    if "AgeGroup" in social_c.columns:
        social_c = social_c[social_c["AgeGroup"].isin(cohort_ages)]
    if "Gender" in social_c.columns:
        social_c = social_c[social_c["Gender"] == gender_value]

    trends_c = trends.copy()
    if "Region" in trends_c.columns:
        trends_c = trends_c[trends_c["Region"].str.contains("Nepal", na=False)]

    os.makedirs(DATA_DIR, exist_ok=True)
    sales_c.to_csv(OUT_SALES, index=False)
    social_c.to_csv(OUT_SOCIAL, index=False)
    trends_c.to_csv(OUT_TRENDS, index=False)

    print("Saved cohort CSVs:")
    print(" -", OUT_SALES)
    print(" -", OUT_SOCIAL)
    print(" -", OUT_TRENDS)

if __name__ == "__main__":
    main()
