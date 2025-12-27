# src/psychology_engine.py  (UPGRADED V2)
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

        self.categories = [
            "Hoodies", "Cargo Pants", "Crop Tops",
            "Jackets", "Pants", "T-Shirts"
        ]

        # Robust keyword map (regex safe)
        self.keyword_map = {
            "Hoodies": ["hoodie", "hoodies", "oversized", "streetwear"],
            "Cargo Pants": ["cargo", "cargopants", "baggy", "utility"],
            "Crop Tops": ["crop top", "croptop", "kfashion", "kstyle", "kpop"],
            "Jackets": ["jacket", "outerwear"],
            "Pants": ["pants", "trousers", "cleanfit", "minimalist"],
            "T-Shirts": ["tshirt", "t-shirt", "tee"]
        }

        # Persona affinity scores
        self.identity_map = {
            "K-Fashion Enthusiast": {
                "Hoodies": 0.9, "Cargo Pants": 0.9, "Crop Tops": 1.0,
                "Jackets": 0.7, "Pants": 0.5, "T-Shirts": 0.6
            },
            "American Streetwear": {
                "Hoodies": 1.0, "Cargo Pants": 0.7, "Crop Tops": 0.4,
                "Jackets": 0.8, "Pants": 0.6, "T-Shirts": 0.85
            },
            "Bohemian/Indie": {
                "Hoodies": 0.4, "Cargo Pants": 0.55, "Crop Tops": 0.95,
                "Jackets": 0.6, "Pants": 0.7, "T-Shirts": 0.50
            },
            "Classic Minimalist": {
                "Hoodies": 0.55, "Cargo Pants": 0.45, "Crop Tops": 0.4,
                "Jackets": 0.95, "Pants": 1.0, "T-Shirts": 0.85
            }
        }

        # Psychology weights
        self.weights = {
            "aspirational": 0.35,
            "conformity": 0.25,
            "identity": 0.20,
            "cultural": 0.15,
            "availability": 0.05
        }


    # ---------------------------------------------------------
    # LOADING
    # ---------------------------------------------------------
    def load_data(self):
        social = pd.read_csv(self.social_path)
        sales = pd.read_csv(self.sales_path)
        trends = pd.read_csv(self.trend_path)

        # Persona fallback
        if "StylePersona" not in social.columns:
            social["StylePersona"] = social["Hashtags"].fillna("").apply(self.infer_persona_from_hashtags)

        return social, sales, trends


    # ---------------------------------------------------------
    # NLP Helper: strong regex match
    # ---------------------------------------------------------
    def match_category(self, text, keywords):
        if pd.isna(text):
            return False
        text = text.lower()

        return any(
            re.search(rf"\b{re.escape(k)}\b", text)
            for k in keywords
        )


    # ---------------------------------------------------------
    # Persona fallback (from hashtags)
    # ---------------------------------------------------------
    def infer_persona_from_hashtags(self, hashtags):
        if pd.isna(hashtags):
            return "Classic Minimalist"

        h = hashtags.lower()

        if any(k in h for k in ["kpop", "kstyle", "kfashion", "seoul"]):
            return "K-Fashion Enthusiast"

        if any(k in h for k in ["streetwear", "hoodie", "oversized"]):
            return "American Streetwear"

        if any(k in h for k in ["indie", "boho", "vintage"]):
            return "Bohemian/Indie"

        return "Classic Minimalist"


    # ---------------------------------------------------------
    # Conformity Score
    # ---------------------------------------------------------
    def compute_conformity(self, social_df):
        rows = []

        for cat in self.categories:
            keywords = self.keyword_map[cat]

            mask = (
                social_df["Caption"].apply(lambda x: self.match_category(x, keywords)) |
                social_df["Hashtags"].apply(lambda x: self.match_category(x, keywords))
            )

            posts = social_df[mask]

            if len(posts) == 0:
                score = 0
            else:
                freq = posts["StylePersona"].value_counts()
                score = freq.max() / freq.sum()

            rows.append([cat, score])

        return pd.DataFrame(rows, columns=["Category", "Conformity"])


    # ---------------------------------------------------------
    # Aspirational Score (improved)
    # ---------------------------------------------------------
    def compute_aspirational(self, social_df):
        rows = []

        for cat in self.categories:
            kw = self.keyword_map[cat]

            mask = (
                social_df["Caption"].apply(lambda x: self.match_category(x, kw)) |
                social_df["Hashtags"].apply(lambda x: self.match_category(x, kw))
            )

            posts = social_df[mask]

            if len(posts) == 0:
                score = 0
            else:
                k_influence = (posts["StylePersona"] == "K-Fashion Enthusiast").mean()

                influencers = (posts["Likes"] > 120).mean()  
                like_norm = (posts["Likes"].mean() / 60)

                score = 0.50 * k_influence + 0.30 * like_norm + 0.20 * influencers

            rows.append([cat, score])

        return pd.DataFrame(rows, columns=["Category", "Aspirational"])


    # ---------------------------------------------------------
    # Identity Score
    # ---------------------------------------------------------
    def compute_identity(self, social_df):
        rows = []

        for cat in self.categories:
            kw = self.keyword_map[cat]

            mask = (
                social_df["Caption"].apply(lambda x: self.match_category(x, kw)) |
                social_df["Hashtags"].apply(lambda x: self.match_category(x, kw))
            )

            posts = social_df[mask]

            if len(posts) == 0:
                score = 0
            else:
                persona_scores = [
                    self.identity_map.get(p, {}).get(cat, 0)
                    for p in posts["StylePersona"]
                ]
                score = np.mean(persona_scores)

            rows.append([cat, score])

        return pd.DataFrame(rows, columns=["Category", "Identity"])


    # ---------------------------------------------------------
    # Cultural Score (Improved: log scaled)
    # ---------------------------------------------------------
    def compute_cultural(self, trends_df):
        df = trends_df.copy()
        df["Cultural"] = np.log1p(df["TrendScore"]) / np.log1p(df["TrendScore"].max())
        return df[["Category", "Cultural"]]


    # ---------------------------------------------------------
    # Availability Score (category visibility)
    # ---------------------------------------------------------
    def compute_availability(self, social_df):
        rows = []

        total_posts = len(social_df)

        for cat in self.categories:
            kw = self.keyword_map[cat]
            mask = (
                social_df["Caption"].apply(lambda x: self.match_category(x, kw)) |
                social_df["Hashtags"].apply(lambda x: self.match_category(x, kw))
            )
            count = social_df[mask].shape[0]

            score = count / (total_posts / len(self.categories))

            rows.append([cat, min(score, 1)])

        return pd.DataFrame(rows, columns=["Category", "Availability"])


    # ---------------------------------------------------------
    # Combine ALL scores
    # ---------------------------------------------------------
    def combine(self, conformity, aspirational, identity, cultural, availability):

        df = (conformity
              .merge(aspirational, on="Category")
              .merge(identity, on="Category")
              .merge(cultural, on="Category")
              .merge(availability, on="Category"))

        df["PsychologyScore"] = (
            df["Aspirational"] * self.weights["aspirational"] +
            df["Conformity"] * self.weights["conformity"] +
            df["Identity"] * self.weights["identity"] +
            df["Cultural"] * self.weights["cultural"] +
            df["Availability"] * self.weights["availability"]
        )

        return df.sort_values("PsychologyScore", ascending=False)


    # ---------------------------------------------------------
    # RUN ENGINE
    # ---------------------------------------------------------
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
    PsychologyEngine().run()
