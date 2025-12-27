# src/02_feature_engineer.py
import pandas as pd
import os
import re

ROOT = os.path.dirname(os.path.dirname(__file__)) if __file__ else "."
DATA_DIR = os.path.join(ROOT, "data")
FEATURE_DIR = os.path.join(DATA_DIR, "features")
os.makedirs(FEATURE_DIR, exist_ok=True)

SALES_C = os.path.join(DATA_DIR, "sales_cohort_kathmandu_male_17_25.csv")
SOCIAL_C = os.path.join(DATA_DIR, "social_cohort_kathmandu_male_17_25.csv")
TRENDS_C = os.path.join(DATA_DIR, "google_trends_1yr_nepal_filtered.csv")

# -------------------------------------------------------
# 🔍 Auto category keyword mapping (for hashtag relevance)
# -------------------------------------------------------
CATEGORY_KEYWORDS = {
    "hoodies": ["hoodie", "hoodies", "streetwear", "oversized", "kfashion"],
    "cargo pants": ["cargo", "cargopants", "utility", "baggy"],
    "crop tops": ["crop", "croptop", "kstyle", "kfashion"],
    "jackets": ["jacket", "outerwear", "layering"],
    "pants": ["pants", "trousers"],
    "t-shirts": ["tshirt", "tshirts", "t-shirt", "tee"],
}

def hashtags_match(text, keywords):
    if pd.isna(text):
        return False
    text = text.lower()
    return any(k in text for k in keywords)


# -------------------------------------------------------
# 🔧 Main Feature Builder for Each Category
# -------------------------------------------------------
def prepare_feat(cat):
    print(f"⚙️ Building features for: {cat}")

    # Load pre-filtered cohort data
    sales = pd.read_csv(SALES_C, parse_dates=["Date"])
    social = pd.read_csv(SOCIAL_C, parse_dates=["PostDate"])
    trends = pd.read_csv(TRENDS_C, parse_dates=["Date"])

    # Normalize category key
    cat_key = cat.lower()
    kw_list = CATEGORY_KEYWORDS.get(cat_key, [cat_key])

    # ----------------------------------------
    # 1️⃣ DAILY SALES
    # ----------------------------------------
    s = (
        sales[sales["Category"] == cat]
        .groupby("Date")["UnitsSold"]
        .sum()
        .reset_index()
        .set_index("Date")
        .asfreq("D", fill_value=0)
        .reset_index()
    )

    # ----------------------------------------
    # 2️⃣ SOCIAL SIGNALS
    # ----------------------------------------
    social["hashtags_clean"] = social["Hashtags"].fillna("").apply(lambda x: re.sub(r"[^a-zA-Z# ]", "", x.lower()))
    social["caption_clean"] = social["Caption"].fillna("").str.lower()

    social["is_related"] = social.apply(
        lambda r: hashtags_match(r["hashtags_clean"], kw_list)
                  or hashtags_match(r["caption_clean"], kw_list),
        axis=1
    )

    soc = (
        social.groupby("PostDate")
        .agg(
            hashtag_count=("is_related", "sum"),
            avg_likes=("Likes", "mean")
        )
        .reset_index()
        .rename(columns={"PostDate": "Date"})
    )

    soc["Date"] = pd.to_datetime(soc["Date"])
    soc = soc.set_index("Date").asfreq("D", fill_value=0).reset_index()

    # ----------------------------------------
    # 3️⃣ GOOGLE TRENDS (FLEXIBLE MATCHING)
    # ----------------------------------------
    trends["Keyword"] = trends["Keyword"].astype(str).str.lower()

    tr = trends[trends["Keyword"].str.contains(cat_key.replace(" ", ""), na=False)]

    if tr.empty:
        print(f"⚠️ No exact Trends match for {cat}, using fallback keyword scan")
        tr = trends[trends["Keyword"].apply(lambda k: any(w in k for w in kw_list))]

    tr = (
        tr.groupby("Date")["InterestScore"]
        .mean()
        .reset_index()
        .set_index("Date")
        .asfreq("D")
        .fillna(method="ffill")
        .fillna(0)
        .reset_index()
    )

    # ----------------------------------------
    # 4️⃣ MERGE ALL DATA SOURCES
    # ----------------------------------------
    df = s.merge(soc, on="Date", how="left").merge(tr, on="Date", how="left")

    # Cleaning
    df["hashtag_count"] = df["hashtag_count"].fillna(0)
    df["avg_likes"] = df["avg_likes"].fillna(0)
    df["UnitsSold"] = df["UnitsSold"].fillna(0)
    df["InterestScore"] = df["InterestScore"].fillna(method="ffill").fillna(0)

    # ----------------------------------------
    # 5️⃣ ADD TIME FEATURES & LAGS
    # ----------------------------------------
    df["weekday"] = df["Date"].dt.weekday
    df["month"] = df["Date"].dt.month

    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = df["UnitsSold"].shift(lag).fillna(0)

    df["rm_7"] = df["UnitsSold"].rolling(7, min_periods=1).mean().shift(1).fillna(0)
    df["rm_30"] = df["UnitsSold"].rolling(30, min_periods=1).mean().shift(1).fillna(0)

    df = df.fillna(0)

    # ----------------------------------------
    # 6️⃣ SAVE TO FILE
    # ----------------------------------------
    out_path = os.path.join(FEATURE_DIR, f"feat_{cat.lower().replace(' ', '_')}.csv")
    df.to_csv(out_path, index=False)

    print(f"✅ Saved feature file → {out_path}\n")
    return out_path


# -------------------------------------------------------
# 🏁 Build All Categories
# -------------------------------------------------------
def prepare_all():
    sales = pd.read_csv(SALES_C)
    categories = sorted(sales["Category"].dropna().unique())

    print("📦 Categories detected:", categories)

    for cat in categories:
        prepare_feat(cat)


if __name__ == "__main__":
    prepare_all()
