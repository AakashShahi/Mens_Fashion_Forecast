import pandas as pd
import numpy as np
import os


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

        # Final fashion categories
        self.categories = [
            "Hoodies",
            "Cargo Pants",
            "Crop Tops",
            "Jackets",
            "Pants",
            "T-Shirts"
        ]

        # Keyword dictionary for hashtag / caption matching (based on your data)
        self.keyword_map = {
            "Hoodies": [
                "hoodie", "hoodies", "oversized", "streetwear",
                "seoulvibes", "nepalstreetstyle", "minimalist"
            ],
            "Cargo Pants": [
                "cargo", "baggy", "utility", "workwear", "streetwear", "wideleg"
            ],
            "Crop Tops": [
                "crop", "kfashion", "kstyle", "kpop", "seoulvibes"
            ],
            "Jackets": [
                "jacket", "jackets", "outerwear", "streetwear", "layering"
            ],
            "Pants": [
                "pants", "trousers", "minimalist", "neutral", "cleanfit",
                "essentialwear"
            ],
            "T-Shirts": [
                "tshirt", "tshirts", "tee", "tees", "graphictees", "streetwear"
            ]
        }

        # Map from category to Google Trends keyword
        # Pants has no direct Google Trends keyword → None (handled safely)
        self.google_map = {
            "Hoodies": "Hoodies",
            "Cargo Pants": "Cargo Pants",
            "Crop Tops": "Crop Tops",
            "Jackets": "Jackets",
            "Pants": None,              # <- no GoogleTrend keyword, will get 0
            "T-Shirts": "T-Shirts",
            "Graphic Tees": "T-Shirts"  # extra mapping if ever needed
        }

    # ------------------------ LOAD DATA ------------------------

    def load_data(self):
        google = pd.read_csv(self.google_path)
        social = pd.read_csv(self.social_path)
        sales = pd.read_csv(self.sales_path)
        psychology = pd.read_csv(self.psychology_path)

        # Parse dates where relevant
        google["Date"] = pd.to_datetime(google["Date"])
        social["PostDate"] = pd.to_datetime(social["PostDate"])
        sales["Date"] = pd.to_datetime(sales["Date"])

        return google, social, sales, psychology

    # ------------------------ HELPERS ------------------------

    def hashtag_match(self, text, keywords):
        if pd.isna(text):
            return False
        text = str(text).lower()
        return any(k in text for k in keywords)

    # ------------------------ SOCIAL SCORE -------------------

    def compute_social_score(self, social_df):
        rows = []

        for cat in self.categories:
            keywords = self.keyword_map[cat]

            mask = (
                social_df["Hashtags"].apply(lambda x: self.hashtag_match(x, keywords)) |
                social_df["Caption"].apply(lambda x: self.hashtag_match(x, keywords))
            )

            cat_posts = social_df[mask]

            if len(cat_posts) == 0:
                freq = 0
                engagement = 0
            else:
                freq = len(cat_posts)
                engagement = cat_posts["Likes"].mean()

            social_score = freq * 0.6 + engagement * 0.4
            rows.append([cat, freq, engagement, social_score])

        return pd.DataFrame(
            rows,
            columns=["Category", "SocialFreq", "Engagement", "SocialScore"]
        )

    # ------------------------ GOOGLE SCORE -------------------

    def compute_google_score(self, google_df):
        rows = []

        for cat in self.categories:
            google_key = self.google_map.get(cat, None)

            if google_key is None:
                # No GoogleTrends keyword → neutral 0
                rows.append([cat, 0, 0])
                continue

            cat_data = google_df[google_df["Keyword"] == google_key]

            if len(cat_data) == 0:
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

    # ------------------------ SALES MOMENTUM -----------------

    def compute_sales_momentum(self, sales_df):
        rows = []

        for cat in self.categories:
            cat_sales = sales_df[sales_df["Category"] == cat].copy()

            if len(cat_sales) < 2:
                rows.append([cat, 0, 0])
                continue

            cat_sales = cat_sales.sort_values("Date")
            cat_sales["UnitsSold"] = cat_sales["UnitsSold"].astype(float)

            initial = cat_sales["UnitsSold"].iloc[0]
            final = cat_sales["UnitsSold"].iloc[-1]

            if initial <= 0:
                momentum = final
            else:
                momentum = (final - initial) / initial

            avg_sales = cat_sales["UnitsSold"].mean()
            sales_score = avg_sales * 0.6 + momentum * 0.4

            rows.append([cat, avg_sales, sales_score])

        return pd.DataFrame(
            rows,
            columns=["Category", "AvgSales", "SalesScore"]
        )

    # ------------------------ COMBINE ALL --------------------

    def combine_scores(self, social, google, sales, psychology):
        df = (
            social
            .merge(google, on="Category")
            .merge(sales, on="Category")
            .merge(psychology[["Category", "PsychologyScore"]], on="Category")
        )

        # Hybrid TrendScore: Social + Google + Sales + Psychology
        df["TrendScore"] = (
            df["SocialScore"] * 0.40 +
            df["GoogleScore"] * 0.25 +
            df["SalesScore"] * 0.20 +
            df["PsychologyScore"] * 0.15
        )

        return df.sort_values("TrendScore", ascending=False)

    # ------------------------ RUN ---------------------------

    def run(self):
        google, social, sales, psychology = self.load_data()

        social_s = self.compute_social_score(social)
        google_s = self.compute_google_score(google)
        sales_s = self.compute_sales_momentum(sales)

        final = self.combine_scores(social_s, google_s, sales_s, psychology)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        final.to_csv(self.output_path, index=False)

        print(f"✔ Hybrid Trend Scores saved to {self.output_path}")
        return final


if __name__ == "__main__":
    analyzer = TrendAnalyzer()
    analyzer.run()
