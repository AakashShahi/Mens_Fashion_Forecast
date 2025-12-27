import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import importlib.util

# ============================================
# Dynamic Import of Insights
# ============================================
try:
    insights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "insights.py"))
    spec_ins = importlib.util.spec_from_file_location("insights", insights_path)
    insights = importlib.util.module_from_spec(spec_ins)
    spec_ins.loader.exec_module(insights)
except Exception as e:
    # Fallback if import fails
    insights = None

st.set_page_config(page_title="📊 Trend Analyzer", layout="wide")

st.title("📊 Hybrid Trend Analyzer Dashboard")
st.caption("Social Media 📣 + Google Interest 🔍 + Sales Momentum 💰 + Psychology 🧠 = Final TrendScore")

DATA_PATH = os.path.join("data", "features", "trend_scores_hybrid.csv")

# ==========================================================
# LOAD DATA
# ==========================================================
if not os.path.exists(DATA_PATH):
    st.error("❌ trend_scores_hybrid.csv missing. Run Trend Analyzer pipeline first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Ensure numeric columns
num_cols = ["SocialScore", "GoogleScore", "SalesScore", "PsychologyScore", "TrendScore"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==========================================================
# SUMMARY VIEW
# ==========================================================
st.subheader("📈 Trend Score Table")
st.dataframe(df.style.highlight_max(axis=0, color="#2ecc71"))


# ==========================================================
# TOP CATEGORY BADGE
# ==========================================================
top_cat = df.loc[df["TrendScore"].idxmax()]

trend_text = f"Trend Score is {top_cat['TrendScore']:.2f}"
if insights:
    trend_desc = insights.describe_metric("Trend Score", top_cat['TrendScore'])
else:
    trend_desc = f"{top_cat['TrendScore']:.2f}"

st.markdown(f"""
<div style="padding:15px;border-radius:12px;background:#14141a;border:1px solid #333;margin-bottom:10px">
<h3>🏆 Current Top Trend: <span style="color:#7ed957">{top_cat['Category']}</span></h3>
<p>
<strong>{trend_desc}</strong> <br>
<strong>Why?</strong> Social={top_cat['SocialScore']:.1f}, Google={top_cat['GoogleScore']:.1f}, Sales={top_cat['SalesScore']:.1f}, Psychology={top_cat['PsychologyScore']:.2f}
</p>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# TREND SCORE BAR CHART
# ==========================================================
st.subheader("🔥 Final Trend Score Comparison")
fig = px.bar(
    df, x="Category", y="TrendScore",
    color="TrendScore", color_continuous_scale="Inferno",
    title="Combined TrendScore for All Fashion Categories"
)
st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# CONTRIBUTION BREAKDOWN
# ==========================================================
st.subheader("📊 Contribution Breakdown Per Category")

selected_cat = st.selectbox("Choose Category", df["Category"])

row = df[df["Category"] == selected_cat].iloc[0]

pie_df = pd.DataFrame({
    "Factor": ["Social", "Google", "Sales", "Psychology"],
    "Score": [
        row["SocialScore"],
        row["GoogleScore"],
        row["SalesScore"],
        row["PsychologyScore"]
    ]
})

fig2 = px.pie(
    pie_df, values="Score", names="Factor",
    color="Factor",
    title=f"Contribution Breakdown for {selected_cat}",
    color_discrete_map={
        "Social": "#ff7675",
        "Google": "#74b9ff",
        "Sales": "#55efc4",
        "Psychology": "#a29bfe"
    }
)
st.plotly_chart(fig2, use_container_width=True)


# ==========================================================
# MULTI-CHART COMPARISON GRID
# ==========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📣 Social Influence Score")
    st.bar_chart(df.set_index("Category")["SocialScore"])

with col2:
    st.subheader("🔍 Google Search Score")
    st.bar_chart(df.set_index("Category")["GoogleScore"])

col3, col4 = st.columns(2)

with col3:
    st.subheader("💰 Sales Momentum Score")
    st.bar_chart(df.set_index("Category")["SalesScore"])

with col4:
    st.subheader("🧠 Psychology Influence Score")
    st.bar_chart(df.set_index("Category")["PsychologyScore"])


# ==========================================================
# TREND HEALTH NARRATIVE
# ==========================================================
def trend_health_dynamic(row):
    """Generates natural language health check."""
    # Use insights module if available for flexible descriptions
    if insights:
        # We can implement a specific health check function in insights, 
        # but for now we reuse the logic here wrapped in descriptive text
        pass

    ts = row["TrendScore"]
    psych = row["PsychologyScore"]
    social = row["SocialScore"]

    if ts > 2500:
        return "🔥 **Explosive Trend:** High adoption driven by massive social buzz. Ensure stock levels are maximized."
    if ts > 1500:
        return "📈 **Strong Trend:** Demand is growing steadily. Good for core inventory."
    if ts > 800:
        return "⚖️ **Moderate Trend:** Stable performance. Monitor for sudden shifts."
    if psych > 0.6:
        return "✨ **Niche & Psychological:** Strong cult following despite lower volume. High margin potential."
    return "📉 **Weak Trend:** Momentum is fading. Avoid overstocking."


st.subheader("🧠 Trend Health Insights (Prescriptive)")

for _, r in df.iterrows():
    st.markdown(f"""
    - **{r['Category']}**: {trend_health_dynamic(r)}
    """)

