import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="📦 Inventory Advisor", layout="wide")

st.title("📦 Inventory Advisor — Recommended Order Quantity")
st.write("Smart ordering engine using Forecast + Trends + Psychology + Sales Velocity + Price Tier.")

DATA_PATH = os.path.join("data", "features", "inventory_recommendations.csv")

if not os.path.exists(DATA_PATH):
    st.error("inventory_recommendations.csv not found. Run Inventory Advisor first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

st.subheader("📊 Inventory Recommendations")
st.dataframe(df)

# Charts
st.subheader("📦 Suggested Order Quantity by Category")
st.bar_chart(df.set_index("Category")["RecommendedOrderQty"])

st.subheader("⚠ Risk Level Indicator")
risk_colors = {
    "High Demand / Risk of Stockout": "#ff4d4d",
    "Moderate Risk": "#ffaa00",
    "Low Risk": "#3fbf3f"
}

df["RiskColor"] = df["RiskLevel"].map(risk_colors)

st.write("### 🛑 Risk Visualization")
for _, row in df.iterrows():
    st.markdown(
        f"""
        <div style='padding:12px; margin:8px 0; border-radius:8px; background-color:{row["RiskColor"]}'>
            <b>{row["Category"]}</b><br>
            Risk: {row["RiskLevel"]}<br>
            Recommended Qty: {row["RecommendedOrderQty"]}
        </div>
        """,
        unsafe_allow_html=True
    )
