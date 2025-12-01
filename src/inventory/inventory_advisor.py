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

        # Price tier effects
        self.price_effect = {
            "Low": 1.25,
            "Mid": 1.00,
            "High": 0.70
        }

        # Persona blend multiplier (identity + cultural + aspiration)
        self.persona_keywords = {
            "K-Fashion Enthusiast": ["kstyle", "kfashion", "kpop", "seoulvibes"],
            "American Streetwear": ["streetwear", "hoodie", "oversized"],
            "Classic Minimalist": ["cleanfit", "neutral", "minimalist"],
            "Bohemian/Indie": ["indievibes", "boho", "vintage"]
        }


    # ----------------------- LOAD FILES -----------------------

    def load_sales_data(self):
        df = pd.read_csv(self.sales_cohort_path, parse_dates=["Date"])
        return df

    def load_trend_scores(self):
        return pd.read_csv(self.trend_path)

    def load_psych_scores(self):
        return pd.read_csv(self.psychology_path)

    def load_forecasts(self, cat):
        """Load hybrid forecasts from Prophet + XGB."""
        p_file = os.path.join(self.pred_dir, f"prophet_forecast_{cat.lower().replace(' ','_')}.csv")
        x_file = os.path.join(self.pred_dir, f"xgb_preds_{cat.lower().replace(' ','_')}.csv")

        prophet = pd.read_csv(p_file)
        xgb = pd.read_csv(x_file)

        prophet_next7 = prophet["yhat"].iloc[-7:].mean()
        xgb_next = xgb["y_pred"].mean()

        hybrid = 0.6 * xgb_next + 0.4 * prophet_next7
        return hybrid


    # ----------------------- SALES VELOCITY --------------------

    def compute_sales_velocity(self, sales_df, cat):

        cat_sales = sales_df[sales_df["Category"] == cat].copy()

        if len(cat_sales) < 30:
            return 0

        cat_sales = cat_sales.sort_values("Date").reset_index(drop=True)

        recent = cat_sales["UnitsSold"].iloc[-1]
        past = cat_sales["UnitsSold"].iloc[-30]

        if past == 0:
            return 1 if recent > 0 else 0

        velocity = (recent - past) / past
        return max(0, min(1, velocity))


    # ----------------------- PRICE TIER -----------------------

    def get_price_tier(self, sales_df, cat):
        tier_mode = sales_df[sales_df["Category"] == cat]["PriceTier"].mode()
        if len(tier_mode) == 0:
            return "Mid"
        return tier_mode[0]


    # ----------------------- PERSONA TARGET -----------------------

    def find_persona_target(self, psychology_score, trend_score):
        """Blend psychological affinity with trending persona groups."""
        
        if psychology_score > 0.6:
            return "K-Fashion Enthusiast"
        if trend_score > 2000:
            return "American Streetwear"
        if psychology_score < 0.3:
            return "Classic Minimalist"
        return "Bohemian/Indie"


    # ----------------------- RISK LEVEL -----------------------

    def compute_risk(self, forecast, trend, psychology):
        if forecast > 3 and trend > 1500:
            return "High Demand / Risk of Stockout"
        if forecast < 1 and psychology < 0.2:
            return "Overstock Risk"
        if trend < 500:
            return "Stable / Low Risk"
        return "Moderate Risk"


    # ----------------------- ORDER QUANTITY --------------------

    def recommend_qty(self, forecast, trend, psychology, sales_vel, price_mult):

        base = forecast * (1 + sales_vel)

        trend_factor = trend / 2000
        psych_factor = psychology

        qty = base * (0.5 + 0.3 * trend_factor + 0.2 * psych_factor)

        qty *= price_mult
        return max(1, round(qty))


    # ----------------------- MAIN RUN --------------------------

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
                f"For {cat}: Forecast={forecast:.2f}, Trend={trend:.2f}, Psy={psychology:.2f}, "
                f"SalesVel={sales_vel:.2f}, Tier={tier}. "
                f"Recommended order quantity: {qty}. Risk: {risk}. Persona Target: {persona}."
            )

            rows.append([
                cat, forecast, trend, psychology, sales_vel, tier, price_mult,
                risk, qty, persona, text
            ])

        df = pd.DataFrame(rows, columns=[
            "Category", "HybridForecast", "TrendScore", "PsychologyScore",
            "SalesVelocity", "PriceTier", "PriceTierEffect", "RiskLevel",
            "RecommendedOrderQty", "PersonaTarget", "RecommendationText"
        ])

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)

        print(f"✔ Inventory Brain output saved to {self.output_path}")
        return df



if __name__ == "__main__":
    advisor = InventoryAdvisor()
    advisor.run()
