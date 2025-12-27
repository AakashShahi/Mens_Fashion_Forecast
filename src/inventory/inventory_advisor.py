# src/inventory/inventory_advisor.py
import pandas as pd
import numpy as np
import os


class InventoryAdvisor:
    def __init__(
        self,
        sales_cohort_path="data/sales_cohort_kathmandu_male_17_25.csv",
        trend_path="data/features/trend_scores_hybrid.csv",
        psychology_path="data/features/psychology_scores.csv",
        pred_dir="models/predictions",
        output_path="data/features/inventory_recommendations.csv"
    ):

        self.sales_cohort_path = sales_cohort_path
        self.trend_path = trend_path
        self.psychology_path = psychology_path
        self.pred_dir = pred_dir
        self.output_path = output_path

        self.categories = [
            "Hoodies",
            "Cargo Pants",
            "Crop Tops",
            "Jackets",
            "Pants",
            "T-Shirts"
        ]

        # Price tier multiplier
        self.price_effect = {
            "Low": 1.25,
            "Mid": 1.00,
            "High": 0.75
        }


    # ----------------------------------------------------
    # SAFE LOADERS
    # ----------------------------------------------------
    def load_sales_data(self):
        return pd.read_csv(self.sales_cohort_path, parse_dates=["Date"])

    def load_trend_scores(self):
        return pd.read_csv(self.trend_path)

    def load_psych_scores(self):
        return pd.read_csv(self.psychology_path)


    # ----------------------------------------------------
    # HYBRID FORECASTING (0.6 XGB + 0.4 Prophet)
    # ----------------------------------------------------
    def load_forecasts(self, cat):

        base = cat.lower().replace(" ", "_")

        p_file = os.path.join(self.pred_dir, f"prophet_forecast_{base}.csv")
        x_file = os.path.join(self.pred_dir, f"xgb_preds_{base}.csv")

        if not (os.path.exists(p_file) and os.path.exists(x_file)):
            raise FileNotFoundError(f"Missing prediction files for category: {cat}")

        prophet = pd.read_csv(p_file)
        xgb = pd.read_csv(x_file)

        prophet_next7 = prophet["yhat"].tail(7).mean()
        xgb_next = xgb["y_pred"].tail(30).mean()

        hybrid = 0.6 * xgb_next + 0.4 * prophet_next7
        return float(hybrid)


    # ----------------------------------------------------
    # SALES VELOCITY (Simpler + More Stable)
    # ----------------------------------------------------
    def compute_sales_velocity(self, sales_df, cat):

        df = sales_df[sales_df["Category"] == cat].copy()
        df = df.sort_values("Date")

        if len(df) < 14:
            return 0

        last_7 = df["UnitsSold"].tail(7).mean()
        prev_7 = df["UnitsSold"].iloc[-14:-7].mean()

        if prev_7 == 0:
            return 1 if last_7 > 0 else 0

        velocity = (last_7 - prev_7) / prev_7
        return float(max(0, min(1, velocity)))


    # ----------------------------------------------------
    # PRICE TIER DETECTOR
    # ----------------------------------------------------
    def get_price_tier(self, sales_df, cat):
        df = sales_df[sales_df["Category"] == cat]
        if "PriceTier" not in df.columns or df.empty:
            return "Mid"
        return df["PriceTier"].mode().iloc[0]


    # ----------------------------------------------------
    # PERSONA ENGINE (Improved)
    # ----------------------------------------------------
    def find_persona_target(self, psychology_score, trend_score):
        if psychology_score > 0.65:
            return "K-Fashion Enthusiast"
        if trend_score > 2000:
            return "American Streetwear"
        if psychology_score < 0.25:
            return "Classic Minimalist"
        return "Bohemian/Indie"


    # ----------------------------------------------------
    # RISK LEVEL ENGINE
    # ----------------------------------------------------
    def compute_risk(self, forecast, trend, psychology):

        if forecast > 5 and trend > 2000:
            return "🔥 High Demand / Stockout Risk"
        if forecast < 1 and psychology < 0.25:
            return "🐢 Overstock Risk"
        if trend < 500:
            return "Stable / Low Risk"
        return "Moderate Risk"


    # ----------------------------------------------------
    # ORDER QUANTITY ENGINE
    # ----------------------------------------------------
    def recommend_qty(self, forecast, trend, psychology, sales_vel, price_mult):

        base_demand = forecast * (1 + 0.6 * sales_vel)

        trend_influence = trend / 2500
        psych_influence = psychology

        qty = base_demand * (0.55 + 0.30 * trend_influence + 0.15 * psych_influence)
        qty *= price_mult

        return int(max(1, round(qty)))


    # ----------------------------------------------------
    # MAIN EXECUTION
    # ----------------------------------------------------
    def run(self):

        sales = self.load_sales_data()
        trends = self.load_trend_scores()
        psych = self.load_psych_scores()

        rows = []

        for cat in self.categories:
            print(f"Processing: {cat}")

            forecast = self.load_forecasts(cat)
            trend = float(trends[trends["Category"] == cat]["TrendScore"].iloc[0])
            psychology = float(psych[psych["Category"] == cat]["PsychologyScore"].iloc[0])
            sales_vel = self.compute_sales_velocity(sales, cat)
            tier = self.get_price_tier(sales, cat)
            price_mult = self.price_effect[tier]
            persona = self.find_persona_target(psychology, trend)
            risk = self.compute_risk(forecast, trend, psychology)
            qty = self.recommend_qty(forecast, trend, psychology, sales_vel, price_mult)

            text = (
                f"📌 **{cat}** — Hybrid Forecast={forecast:.2f}, Trend={trend:.0f}, Psychology={psychology:.2f}, "
                f"Velocity={sales_vel:.2f}, Price Tier={tier}. "
                f"Recommended **{qty} units**. Risk: **{risk}**. Persona Focus: **{persona}**."
            )

            rows.append([
                cat, forecast, trend, psychology, sales_vel,
                tier, price_mult, risk, qty, persona, text
            ])

        df = pd.DataFrame(rows, columns=[
            "Category", "HybridForecast", "TrendScore", "PsychologyScore",
            "SalesVelocity", "PriceTier", "PriceTierEffect", "RiskLevel",
            "RecommendedOrderQty", "PersonaTarget", "RecommendationText"
        ])

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)

        print(f"✔ Inventory Advisor Output Saved → {self.output_path}")
        return df



if __name__ == "__main__":
    InventoryAdvisor().run()
