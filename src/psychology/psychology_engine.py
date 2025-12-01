import pandas as pd
import numpy as np
import os
import re


class PsychologyEngine:
    def __init__(
        self,
        social_path="data/social_cohort_kathmandu_male_17_25.csv",
        sales_path="data/sales_cohort_kathmandu_male_17_25.csv",
        trend_path="data/features/trend_scores.csv",
        output_path="data/features/psychology_scores.csv"
    ):
        self.social_path = social_path
        self.sales_path = sales_path
        self.trend_path = trend_path
        self.output_path = output_path

        # Fashion categories in your forecast models
        self.categories = [
            "Hoodies",
            "Cargo Pants",
            "Crop Tops",
            "Jackets",
            "Pants",
            "T-Shirts"
        ]

        # PERSONA SET from your dataset
        self.personas = [
            "K-Fashion Enthusiast",
            "American Streetwear",
            "Bohemian/Indie",
            "Classic Minimalist"
        ]

        # NLP keyword dictionary for hashtag → category detection
        self.keyword_map = {
            "Hoodies": [
                "hoodie", "hoodies", "oversized", "streetwear", "layering"
            ],
            "Cargo Pants": [
                "cargo", "baggy", "utility", "workwear", "wideleg"
            ],
            "Crop Tops": [
                "crop", "kfashion", "kstyle", "kpop", "korean"
            ],
            "Jackets": [
                "jacket", "jackets", "outerwear", "layering", "streetwear"
            ],
            "Pants": [
                "pants", "trousers", "minimalist", "neutral", "cleanfit", "essentialwear"
            ],
            "T-Shirts": [
                "tshirt", "tshirts", "tee", "tees", "streetwear"
            ]
        }

        # Persona → Category Identity Affinity
        self.identity_map = {
            "K-Fashion Enthusiast": {
                "Hoodies": 0.9, "Cargo Pants": 0.9, "Crop Tops": 0.85,
                "Jackets": 0.7, "Pants": 0.6, "T-Shirts": 0.7
            },
            "American Streetwear": {
                "Hoodies": 0.95, "Cargo Pants": 0.75, "Crop Tops": 0.4,
                "Jackets": 0.85, "Pants": 0.7, "T-Shirts": 0.9
            },
            "Bohemian/Indie": {
                "Hoodies": 0.4, "Cargo Pants": 0.55, "Crop Tops": 0.95,
                "Jackets": 0.6, "Pants": 0.7, "T-Shirts": 0.5
            },
            "Classic Minimalist": {
                "Hoodies": 0.55, "Cargo Pants": 0.5, "Crop Tops": 0.4,
                "Jackets": 0.9, "Pants": 0.95, "T-Shirts": 0.85
            }
        }

        # Psychology metric weights (theory-driven)
        self.weights = {
            "aspirational": 0.35,
            "conformity": 0.25,
            "identity": 0.20,
            "cultural": 0.15,
            "availability": 0.05
        }


    # -------------------------- UTILITIES --------------------------

    def load_data(self):
        social = pd.read_csv(self.social_path)
        sales = pd.read_csv(self.sales_path)
        trends = pd.read_csv(self.trend_path)

        return social, sales, trends

    def match_category(self, text, keywords):
        if pd.isna(text):
            return False
        text = text.lower()
        return any(k in text for k in keywords)


    # ----------------------- PSYCHOLOGY METRICS ---------------------

    def compute_conformity(self, social_df):
        """
        Measures how much adoption is concentrated around a single persona group.
        High concentration → high conformity.
        """
        rows = []

        for cat in self.categories:
            keywords = self.keyword_map[cat]

            mask = (
                social_df["Caption"].apply(lambda x: self.match_category(x, keywords)) |
                social_df["Hashtags"].apply(lambda x: self.match_category(x, keywords))
            )

            cat_posts = social_df[mask]

            if len(cat_posts) == 0:
                conformity = 0
            else:
                freq = cat_posts["StylePersona"].value_counts()
                conformity = freq.max() / freq.sum()

            rows.append([cat, conformity])

        return pd.DataFrame(rows, columns=["Category", "Conformity"])


    def compute_aspirational(self, social_df):
        """
        Measures aspirational influence via:
        - K-Fashion influence (Korean aesthetic)
        - High engagement (likes)
        """
        rows = []

        for cat in self.categories:
            keywords = self.keyword_map[cat]

            mask = (
                social_df["Caption"].apply(lambda x: self.match_category(x, keywords)) |
                social_df["Hashtags"].apply(lambda x: self.match_category(x, keywords))
            )

            cat_posts = social_df[mask]

            if len(cat_posts) == 0:
                score = 0
            else:
                k_influence = (cat_posts["StylePersona"] == "K-Fashion Enthusiast").mean()
                like_influence = cat_posts["Likes"].mean() / 50  
                score = 0.6 * k_influence + 0.4 * like_influence

            rows.append([cat, score])

        return pd.DataFrame(rows, columns=["Category", "Aspirational"])


    def compute_identity(self, social_df):
        """
        Identity alignment: persona affinity weighted by real persona usage.
        """
        rows = []

        for cat in self.categories:
            keywords = self.keyword_map[cat]

            mask = (
                social_df["Caption"].apply(lambda x: self.match_category(x, keywords)) |
                social_df["Hashtags"].apply(lambda x: self.match_category(x, keywords))
            )

            cat_posts = social_df[mask]

            if len(cat_posts) == 0:
                score = 0
            else:
                scores = [
                    self.identity_map.get(p, {}).get(cat, 0)
                    for p in cat_posts["StylePersona"]
                ]
                score = np.mean(scores)

            rows.append([cat, score])

        return pd.DataFrame(rows, columns=["Category", "Identity"])


    def compute_cultural(self, trends_df):
        """
        Cultural resonance = normalized trendscore.
        """
        df = trends_df.copy()
        df["Cultural"] = df["TrendScore"] / df["TrendScore"].max()
        return df[["Category", "Cultural"]]


    def compute_availability(self, social_df):
        """
        Availability heuristic = how visible the category is on social feeds.
        """
        rows = []

        for cat in self.categories:
            keywords = self.keyword_map[cat]

            mask = (
                social_df["Caption"].apply(lambda x: self.match_category(x, keywords)) |
                social_df["Hashtags"].apply(lambda x: self.match_category(x, keywords))
            )

            score = len(social_df[mask]) / len(social_df)

            rows.append([cat, score])

        return pd.DataFrame(rows, columns=["Category", "Availability"])


    # -------------------- COMBINE SCORES -----------------------------

    def combine(self, conformity, aspirational, identity, cultural, availability):

        df = (
            conformity.merge(aspirational, on="Category")
            .merge(identity, on="Category")
            .merge(cultural, on="Category")
            .merge(availability, on="Category")
        )

        df["PsychologyScore"] = (
            df["Aspirational"] * self.weights["aspirational"] +
            df["Conformity"] * self.weights["conformity"] +
            df["Identity"] * self.weights["identity"] +
            df["Cultural"] * self.weights["cultural"] +
            df["Availability"] * self.weights["availability"]
        )

        return df.sort_values("PsychologyScore", ascending=False)


    # -------------------------- RUN ENGINE ---------------------------

    def run(self):
        social, sales, trends = self.load_data()

        conformity = self.compute_conformity(social)
        aspirational = self.compute_aspirational(social)
        identity = self.compute_identity(social)
        cultural = self.compute_cultural(trends)
        availability = self.compute_availability(social)

        final = self.combine(conformity, aspirational, identity, cultural, availability)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        final.to_csv(self.output_path, index=False)

        print(f"✔ Psychology scores saved to {self.output_path}")
        return final


if __name__ == "__main__":
    engine = PsychologyEngine()
    engine.run()
