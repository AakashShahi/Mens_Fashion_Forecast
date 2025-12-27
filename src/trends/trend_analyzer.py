import pandas as pd
import numpy as np
import os
import re


class TrendAnalyzer:
    def __init__(
        self,
        google_path="data/google_trends_1yr_nepal.csv",
        social_path="data/social_media_1yr_nepal.csv",
        sales_path="data/sales_data_1yr_nepal.csv",
        psychology_path="data/features/psychology_scores.csv",
        output_path="data/features/trend_scores_hybrid.csv"
    ):
        self.google_path = google_path
        self.social_path = social_path
        self.sales_path = sales_path
        self.psychology_path = psychology_path
        self.output_path = output_path

        self.categories = [
            "Hoodies", "Cargo Pants", "Crop Tops",
            "Jackets", "Pants", "T-Shirts"
        ]

        # More robust keyword detection
        self.keyword_map = {
            "Hoodies": ["hoodie", "hoodies", "oversized", "streetwear"],
            "Cargo Pants": ["cargo", "cargopants", "baggy", "utility"],
            "Crop Tops": ["crop top", "croptop", "kfashion", "kstyle", "kpop"],
            "Jackets": ["jacket", "outerwear"],
            "Pants": ["pants", "trousers", "minimalist", "cleanfit"],
            "T-Shirts": ["tshirt", "t-shirt", "tee", "graphictee"]
        }

        # Map for Google Trends (case-insensitive)
        self.google_map = {
            "Hoodies": "Hoodies",
            "Cargo Pants": "Cargo Pants",
            "Crop Tops": "Crop Tops",
            "Jackets": "Jackets",
            "Pants": None,
            "T-Shirts": "T-Shirts"
        }


    # ----------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------
    def load_data(self):
        google = pd.read_csv(self.google_path)
        social = pd.read_csv(self.social_path)
        sales = pd.read_csv(self.sales_path)
        psychology = pd.read_csv(self.psychology_path)

        google["Date"] = pd.to_datetime(google["Date"])
        social["PostDate"] = pd.to_datetime(social["PostDate"])
        sales["Date"] = pd.to_datetime(sales["Date"])

        return google, social, sales, psychology


    # ----------------------------------------------------
    # NLP MATCHING
    # ----------------------------------------------------
    def keyword_match(self, text, keywords):
        if pd.isna(text):
            return False
        txt = str(text).lower()

        return any(
            re.search(rf"\b{re.escape(k)}\b", txt)
            for k in keywords
        )


    # ----------------------------------------------------
    # SOCIAL SCORE
    # ----------------------------------------------------
    def compute_social_score(self, social_df):
        rows = []

        for cat in self.categories:
            kw = self.keyword_map[cat]

            mask = (
                social_df["Hashtags"].apply(lambda x: self.keyword_match(x, kw)) |
                social_df["Caption"].apply(lambda x: self.keyword_match(x, kw))
            )

            posts = social_df[mask]

            if posts.empty:
                rows.append([cat, 0, 0, 0])
                continue

            freq = len(posts)
            engagement = posts["Likes"].mean()

            # Influencer boost
            influencer_rate = (posts["Likes"] > 120).mean()
            influencer_boost = influencer_rate * 50

            social_score = freq * 0.5 + engagement * 0.4 + influencer_boost * 0.1

            rows.append([cat, freq, engagement, social_score])

        return pd.DataFrame(
            rows,
            columns=["Category", "SocialFreq", "Engagement", "SocialScore"]
        )


    # ----------------------------------------------------
    # GOOGLE SCORE
    # ----------------------------------------------------
    def compute_google_score(self, google_df):
        rows = []

        for cat in self.categories:
            key = self.google_map.get(cat, None)

            if key is None:
                rows.append([cat, 0, 0])
                continue

            cat_data = google_df[
                google_df["Keyword"].str.lower().str.contains(key.lower())
            ]

            if cat_data.empty:
                avg_interest = 0
                growth = 0
            else:
                avg_interest = cat_data["InterestScore"].mean()
                growth = cat_data["InterestScore"].iloc[-1] - cat_data["InterestScore"].iloc[0]

            google_score = avg_interest * 0.7 + growth * 0.3

            rows.append([cat, avg_interest, google_score])

        return pd.DataFrame(
            rows,
            columns=["Category", "GoogleInterest", "GoogleScore"]
        )


    # ----------------------------------------------------
    # SALES MOMENTUM (Improved)
    # ----------------------------------------------------
    def compute_sales_momentum(self, sales_df):
        rows = []

        for cat in self.categories:
            df = sales_df[sales_df["Category"] == cat].copy()

            if len(df) < 14:
                rows.append([cat, 0, 0])
                continue

            df = df.sort_values("Date")

            last7 = df["UnitsSold"].iloc[-7:].mean()
            prev7 = df["UnitsSold"].iloc[-14:-7].mean()

            if prev7 == 0:
                momentum = 1 if last7 > 0 else 0
            else:
                momentum = (last7 - prev7) / prev7

            avg_sales = last7
            sales_score = avg_sales * 0.7 + momentum * 0.3

            rows.append([cat, avg_sales, sales_score])

        return pd.DataFrame(
            rows,
            columns=["Category", "AvgSales", "SalesScore"]
        )


    # ----------------------------------------------------
    # COMBINE ALL SCORES
    # ----------------------------------------------------
    def combine_scores(self, social, google, sales, psychology):

        df = (
            social.merge(google, on="Category")
                  .merge(sales, on="Category")
                  .merge(psychology[["Category", "PsychologyScore"]], on="Category")
        )

        df["TrendScore"] = (
            df["SocialScore"] * 0.35 +
            df["GoogleScore"] * 0.25 +
            df["PsychologyScore"] * 0.20 +
            df["SalesScore"] * 0.20
        )

        return df.sort_values("TrendScore", ascending=False)


    # ----------------------------------------------------
    # RUN PIPELINE
    # ----------------------------------------------------
    def run(self):
        google, social, sales, psychology = self.load_data()

        s = self.compute_social_score(social)
        g = self.compute_google_score(google)
        m = self.compute_sales_momentum(sales)

        final = self.combine_scores(s, g, m, psychology)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        final.to_csv(self.output_path, index=False)

        print(f"✔ Hybrid Trend Scores saved to {self.output_path}")
        return final


if __name__ == "__main__":
    TrendAnalyzer().run()
