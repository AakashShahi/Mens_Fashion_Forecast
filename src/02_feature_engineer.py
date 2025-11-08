# src/02_feature_engineer.py
import pandas as pd
import os

ROOT = os.path.dirname(os.path.dirname(__file__)) if __file__ else "."
DATA_DIR = os.path.join(ROOT, "data")
FEATURE_DIR = os.path.join(DATA_DIR, "features")
os.makedirs(FEATURE_DIR, exist_ok=True)

SALES_C = os.path.join(DATA_DIR, "sales_cohort_kathmandu_male_17_25.csv")
SOCIAL_C = os.path.join(DATA_DIR, "social_cohort_kathmandu_male_17_25.csv")
TRENDS_C = os.path.join(DATA_DIR, "google_trends_1yr_nepal_filtered.csv")

def prepare_feat(cat):
    sales = pd.read_csv(SALES_C, parse_dates=["Date"])
    social = pd.read_csv(SOCIAL_C, parse_dates=["PostDate"])
    trends = pd.read_csv(TRENDS_C, parse_dates=["Date"])
    # Daily sales
    s = sales[sales["Category"]==cat].groupby("Date")["UnitsSold"].sum().reset_index().set_index("Date").asfreq("D").fillna(0).reset_index()
    # Social signals
    social["Hashtags_lower"] = social["Hashtags"].fillna("").str.lower()
    social["Caption_lower"] = social["Caption"].fillna("").str.lower()
    key = cat.lower()
    social["is_related"] = social["Hashtags_lower"].str.contains(key, na=False) | social["Caption_lower"].str.contains(key, na=False)
    soc = social.groupby("PostDate").agg(hashtag_count=("is_related","sum"), avg_likes=("Likes","mean")).reset_index().rename(columns={"PostDate":"Date"})
    soc["Date"] = pd.to_datetime(soc["Date"])
    soc = soc.set_index("Date").asfreq("D").fillna(0).reset_index()
    # Trends
    tr = trends[trends["Keyword"].str.lower()==key].groupby("Date")["InterestScore"].mean().reset_index().set_index("Date").asfreq("D").fillna(method="ffill").reset_index()
    # Merge
    df = s.merge(soc, on="Date", how="left").merge(tr, on="Date", how="left")
    df["hashtag_count"] = df.get("hashtag_count",0).fillna(0)
    df["avg_likes"] = df.get("avg_likes",0).fillna(0)
    df["InterestScore"] = df.get("InterestScore",0).fillna(method="ffill").fillna(0)
    df["weekday"] = df["Date"].dt.weekday
    df["month"] = df["Date"].dt.month
    for l in [1,7,14,30]:
        df[f"lag_{l}"] = df["UnitsSold"].shift(l).fillna(0)
    df["rm_7"] = df["UnitsSold"].rolling(7,min_periods=1).mean().shift(1).fillna(0)
    df["rm_30"] = df["UnitsSold"].rolling(30,min_periods=1).mean().shift(1).fillna(0)
    df = df.fillna(0)
    fn = os.path.join(FEATURE_DIR, f"feat_{cat.lower().replace(' ','_')}.csv")
    df.to_csv(fn, index=False)
    print("Saved", fn)
    return fn

def prepare_all():
    sales = pd.read_csv(SALES_C)
    cats = sorted(sales["Category"].unique())
    print("Found categories:", cats)
    for c in cats:
        prepare_feat(c)

if __name__ == "__main__":
    prepare_all()
