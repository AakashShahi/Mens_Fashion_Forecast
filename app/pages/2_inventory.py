import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="📦 Inventory Advisor", layout="wide")

st.title("📦 Inventory Advisor")
st.caption("Hybrid Engine: Forecast + TrendScore + Psychology + Sales Velocity + Price Tier")

DATA_PATH = os.path.join("data", "features", "inventory_recommendations.csv")

if not os.path.exists(DATA_PATH):
    st.error("❌ inventory_recommendations.csv missing. Run Inventory Advisor pipeline.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# 1️⃣ Model consistency check
missing_models = df[df["HybridForecast"] <= 0]["Category"].tolist()
if missing_models:
    st.warning(f"⚠ Hybrid forecast incomplete for: {', '.join(missing_models)}. Consider retraining.")


# 2️⃣ Inventory Health Score (robust to emoji / text variants)
def compute_health(row):
    label = str(row["RiskLevel"])

    # interpret the text rather than exact equality
    if "High Demand" in label:
        risk_score = 90
    elif "Overstock" in label:
        risk_score = 30
    elif "Stable" in label or "Low Risk" in label:
        risk_score = 80
    else:
        # "Moderate", "Medium", unknown → mid score
        risk_score = 60

    score = (
        risk_score * 0.4 +
        float(row["HybridForecast"]) * 8 +
        float(row["PsychologyScore"]) * 20 +
        float(row["PriceTierEffect"]) * 10
    )
    return min(100, round(score))


df["HealthScore"] = df.apply(compute_health, axis=1)


# 3️⃣ Risk colors (also robust to wording / emojis)
def map_risk_color(label: str) -> str:
    label = str(label)
    if "High Demand" in label:
        return "#ff4d4d"   # red
    if "Overstock" in label:
        return "#ff66cc"   # pink
    if "Stable" in label or "Low Risk" in label:
        return "#3fbf3f"   # green
    return "#ffaa00"       # amber for "Moderate"/other


df["RiskColor"] = df["RiskLevel"].apply(map_risk_color)


# ==========================================================
# FULL TABLE
# ==========================================================
st.subheader("📊 Full Inventory Recommendations")
st.dataframe(df.style.highlight_max("RecommendedOrderQty", color="#7ed957"))


# ==========================================================
# QTY CHART
# ==========================================================
st.subheader("📦 Suggested Order Quantity")
fig_qty = px.bar(
    df,
    x="Category",
    y="RecommendedOrderQty",
    color="RecommendedOrderQty",
    color_continuous_scale="Blues",
    title="Recommended Order Quantity by Category"
)
st.plotly_chart(fig_qty, use_container_width=True)


# ==========================================================
# RISK CHART
# ==========================================================
st.subheader("⚠ Risk Levels")

fig_risk = px.bar(
    df,
    x="Category",
    y="RecommendedOrderQty",
    color="RiskLevel",
    color_discrete_map={r: map_risk_color(r) for r in df["RiskLevel"].unique()},
    title="Risk Level Overview"
)
st.plotly_chart(fig_risk, use_container_width=True)


# ==========================================================
# EXPANDERS (per-category insight)
# ==========================================================
st.subheader("📦 Detailed Insights")
for _, row in df.iterrows():
    with st.expander(f"📌 {row['Category']} — {row['RiskLevel']}"):
        st.markdown(f"""
        <div style="padding:15px; border-radius:10px; background:#15151c; border:1px solid #333;">
            <h4 style="color:#7ed957;">Recommended Order: {row['RecommendedOrderQty']}</h4>
            <p><b>Hybrid Forecast:</b> {row['HybridForecast']:.2f}</p>
            <p><b>TrendScore:</b> {row['TrendScore']:.2f}</p>
            <p><b>PsychologyScore:</b> {row['PsychologyScore']:.2f}</p>
            <p><b>Sales Velocity:</b> {row['SalesVelocity']:.2f}</p>
            <p><b>Price Tier:</b> {row['PriceTier']} ({row['PriceTierEffect']})</p>
            <p><b>Persona Target:</b> {row['PersonaTarget']}</p>
            <p style="color:#ccc">{row['RecommendationText']}</p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================================
# HEALTH SCORE CHART
# ==========================================================
st.subheader("❤️ Inventory Health Score (0–100)")
fig_health = px.bar(
    df,
    x="Category",
    y="HealthScore",
    color="HealthScore",
    color_continuous_scale="Magma",
    title="Inventory Health Score"
)
st.plotly_chart(fig_health, use_container_width=True)


# ==========================================================
# PERSONA PIE CHART
# ==========================================================
st.subheader("🎭 Persona Strategy Breakdown")

persona_df = df.groupby("PersonaTarget")["RecommendedOrderQty"].sum().reset_index()
fig_persona = px.pie(
    persona_df,
    values="RecommendedOrderQty",
    names="PersonaTarget",
    color_discrete_sequence=px.colors.qualitative.Set2,
    title="Ordering Volume by Persona Target"
)
st.plotly_chart(fig_persona, use_container_width=True)
