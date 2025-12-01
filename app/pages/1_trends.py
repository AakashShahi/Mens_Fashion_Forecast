import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="📊 Trend Analyzer", layout="wide")

st.title("📊 Hybrid Trend Analyzer")
st.write("This page shows trend scores combining Social Media + Google Interest + Sales Momentum + Psychology Signals.")

DATA_PATH = os.path.join("data", "features", "trend_scores_hybrid.csv")

if not os.path.exists(DATA_PATH):
    st.error("trend_scores_hybrid.csv not found. Run Trend Analyzer first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Display Data
st.subheader("📈 Trend Score Table")
st.dataframe(df)

# Bar chart
st.subheader("🔥 Trend Score Comparison")
st.bar_chart(df.set_index("Category")["TrendScore"])

# Social Score
st.subheader("📣 Social Media Influence (Frequency × Engagement)")
st.bar_chart(df.set_index("Category")["SocialScore"])

# Google Trend
st.subheader("🔍 Google Search Interest Score")
st.bar_chart(df.set_index("Category")["GoogleScore"])

# Sales Momentum
st.subheader("💰 Sales Momentum Score")
st.bar_chart(df.set_index("Category")["SalesScore"])
