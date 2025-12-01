import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="🧠 Psychology Engine", layout="wide")

st.title("🧠 Fashion Psychology Engine")
st.write("Identity, Aspirational Fit, Conformity, Cultural Signals — mapped into a unified PsychologyScore.")

DATA_PATH = os.path.join("data", "features", "psychology_scores.csv")

if not os.path.exists(DATA_PATH):
    st.error("psychology_scores.csv not found. Run Psychology Engine first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Show data
st.subheader("🧠 Psychology Metrics Table")
st.dataframe(df)

# Psychology Score chart
st.subheader("✨ Psychology Score by Category")
st.bar_chart(df.set_index("Category")["PsychologyScore"])

# Social identity breakdown
st.subheader("👤 Identity Components")
cols = ["Conformity", "Aspirational", "Identity", "Cultural"]

st.line_chart(df.set_index("Category")[cols])
