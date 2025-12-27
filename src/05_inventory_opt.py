# src/05_inventory_opt.py (FULL RETAIL AI INVENTORY BRAIN)
import joblib, os, pandas as pd, numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
FEATURE_DIR = os.path.join(DATA_DIR, "features")
MODEL_DIR = os.path.join(ROOT, "models")
PRED_DIR = os.path.join(MODEL_DIR, "predictions")

PRICE_TIER_MULT = {
    "Low": 1.25,
    "Mid": 1.10,
    "High": 0.85
}

PERSONA_PRIORITY = [
    "K-Fashion Enthusiast",
    "American Streetwear",
    "Bohemian/Indie",
    "Classic Minimalist"
]


# -----------------------------------------------------
# Hybrid Forecast = 0.7*Prophet + 0.3*XGB
# -----------------------------------------------------
def hybrid_forecast(cat, horizon=30):
    cat_key = cat.lower().replace(" ", "_")

    # Prophet forecast
    prophet_file = os.path.join(PRED_DIR, f"prophet_forecast_{cat_key}.csv")
    if not os.path.exists(prophet_file):
        return None
    prophet = pd.read_csv(prophet_file)
    prophet_pred = prophet["yhat"].tail(horizon).mean()

    # XGB forecast
    xgb_file = os.path.join(PRED_DIR, f"xgb_preds_{cat_key}.csv")
    if not os.path.exists(xgb_file):
        xgb_pred = prophet_pred
    else:
        xdf = pd.read_csv(xgb_file)
        xgb_pred = xdf["y_pred"].tail(30).mean()

    hybrid = (0.7 * prophet_pred) + (0.3 * xgb_pred)
    return hybrid


# -----------------------------------------------------
# Price tier detection using sales cohort
# -----------------------------------------------------
def detect_price_tier(cat):
    sales_path = os.path.join(DATA_DIR, "sales_cohort_kathmandu_male_17_25.csv")
    df = pd.read_csv(sales_path)
    df = df[df["Category"] == cat]

    if "PriceTier" not in df.columns or df.empty:
        return "Low"

    return df["PriceTier"].mode().iloc[0]


# -----------------------------------------------------
# Sales Velocity (last 14 days)
# -----------------------------------------------------
def compute_sales_velocity(cat):
    cat_key = cat.lower().replace(" ", "_")
    feat_file = os.path.join(FEATURE_DIR, f"feat_{cat_key}.csv")

    df = pd.read_csv(feat_file, parse_dates=["Date"])
    df = df.sort_values("Date")

    if len(df) < 15:
        return 0

    recent = df.tail(14)
    return recent["UnitsSold"].mean()


# -----------------------------------------------------
# Trend Score + Psychology Score Integration
# -----------------------------------------------------
def load_scores(cat):
    trend_path = os.path.join(FEATURE_DIR, "trend_scores_hybrid.csv")
    psych_path = os.path.join(FEATURE_DIR, "psychology_scores.csv")
    
    t, p = 0, 0
    
    if os.path.exists(trend_path):
        trend = pd.read_csv(trend_path)
        match = trend[trend["Category"] == cat]
        if not match.empty:
            t = match["TrendScore"].iloc[0]
            
    if os.path.exists(psych_path):
        psych = pd.read_csv(psych_path)
        match = psych[psych["Category"] == cat]
        if not match.empty:
            p = match["PsychologyScore"].iloc[0]

    return t, p


# -----------------------------------------------------
# Persona Target Selector
# -----------------------------------------------------
def persona_target(cat):
    social = pd.read_csv(os.path.join(DATA_DIR, "social_cohort_kathmandu_male_17_25.csv"))
    df = social[social["Caption"].str.contains(cat.split()[0], case=False, na=False)]

    if df.empty or "StylePersona" not in df.columns:
        return PERSONA_PRIORITY[0]

    return df["StylePersona"].mode().iloc[0]


# -----------------------------------------------------
# MAIN INVENTORY OPTIMIZATION FUNCTION
# -----------------------------------------------------
def compute_reorder(cat, horizon=30, lead_time=14, service_z=1.65):
    cat_key = cat.lower().replace(" ", "_")

    # ---- Hybrid forecast ----
    hybrid = hybrid_forecast(cat, horizon=horizon)
    
    # ---- Sales velocity ----
    sales_vel = compute_sales_velocity(cat)

    # ---- Trend + Psychology ----
    trend_score, psych_score = load_scores(cat)

    # ---- Price Tier ----
    price_tier = detect_price_tier(cat)
    price_mult = PRICE_TIER_MULT.get(price_tier, 1.0)

    if hybrid is None:
        # Fallback to Sales Velocity if forecast failed
        hybrid = sales_vel * horizon # simple projection
        risk = "Low Data / Naive Forecast"
    else:
        # Standard Risk calc
        if hybrid > 8:
            risk = "High Demand / Risk of Stockout"
        elif hybrid > 4:
            risk = "Moderate Risk"
        else:
            risk = "Low Risk"

    # ---- Final Order Quantity ----
    base_order = max(1, round(hybrid * price_mult))
    adjusted = base_order + round(psych_score * 5)

    persona = persona_target(cat)

    result = {
        "Category": cat,
        "HybridForecast": round(hybrid, 3),
        "TrendScore": trend_score,
        "PsychologyScore": psych_score,
        "SalesVelocity": round(sales_vel, 3),
        "PriceTier": price_tier,
        "PriceTierEffect": price_mult,
        "RiskLevel": risk,
        "RecommendedOrderQty": int(adjusted),
        "PersonaTarget": persona,
        "RecommendationText": (
            f"For {cat}: Forecast={round(hybrid,2)}, Trend={trend_score}, Psy={round(psych_score,2)}, "
            f"SalesVel={round(sales_vel,2)}, Tier={price_tier}. "
            f"Recommended order quantity: {adjusted}. Risk: {risk}. Persona Target: {persona}."
        )
    }

    return result


# -----------------------------------------------------
# Save CSV for all categories
# -----------------------------------------------------
def run_full_inventory_export():
    files = [f for f in os.listdir(FEATURE_DIR) if f.startswith("feat_")]
    cats = [f[len("feat_"):-4].replace("_", " ").title() for f in files]

    out = []
    for c in cats:
        try:
            out.append(compute_reorder(c))
        except Exception as e:
            print("Error:", c, e)

    df = pd.DataFrame(out)
    df.to_csv(os.path.join(FEATURE_DIR, "inventory_recommendations.csv"), index=False)
    print("📦 Saved inventory_recommendations.csv")
    return df


if __name__ == "__main__":
    run_full_inventory_export()
