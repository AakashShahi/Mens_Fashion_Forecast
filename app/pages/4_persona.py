import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="👤 Persona Dashboard", layout="wide")

st.title("👤 Persona Dashboard — Style Personas of Kathmandu Youth")
st.write("Understanding buyer psychology and fashion identities (male 17–25, Kathmandu).")

SOCIAL_PATH = os.path.join("data", "social_cohort_kathmandu_male_17_25.csv")
SALES_PATH = os.path.join("data", "sales_cohort_kathmandu_male_17_25.csv")

# Load social data
if not os.path.exists(SOCIAL_PATH):
    st.error("social_cohort CSV missing.")
    st.stop()

social = pd.read_csv(SOCIAL_PATH)

# Load sales data
if os.path.exists(SALES_PATH):
    sales = pd.read_csv(SALES_PATH)
else:
    sales = None
    st.warning("⚠ sales_cohort CSV missing — Persona × Category heatmap may be limited.")


# ---------------- Persona Distribution ----------------
st.subheader("📊 Persona Distribution (Social Media Activity)")

persona_counts = social["StylePersona"].value_counts()

st.bar_chart(persona_counts)

st.write("**Most active personas in Kathmandu’s fashion scene:**")
top_personas = persona_counts.head(3).index.tolist()
st.success(", ".join(top_personas))


# ---------------- Persona × Category Heatmap ----------------
st.subheader("🎨 Persona × Category Popularity Heatmap")

if sales is not None and "Category" in sales.columns:
    pivot = sales.pivot_table(
        index="StylePersona",
        columns="Category",
        values="UnitsSold",
        aggfunc="sum",
        fill_value=0
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, cmap="Blues", fmt='g', ax=ax)
    st.pyplot(fig)

else:
    st.warning("Sales data missing, cannot build Persona × Category heatmap.")


# ---------------- Persona Insights ----------------
st.subheader("💡 Persona Insights")

def insight(persona):
    if persona == "K-Fashion Enthusiast":
        return "Strong influence from Korean streetwear, oversized fits, neutral tones."
    if persona == "American Streetwear":
        return "Prefers hoodies, varsity jackets, sneaker culture."
    if persona == "Bohemian/Indie":
        return "Vintage patterns, earthy tones, handcrafted aesthetics."
    if persona == "Classic Minimalist":
        return "Neutral colors, clean silhouettes, capsule wardrobe."
    return "General urban fashion identity."

for p, count in persona_counts.items():
    st.markdown(f"""
    <div style="padding:12px; margin:6px 0; border-radius:10px; background:#1b1b1e;">
        <b>{p}</b> — {count} social posts<br>
        <i>{insight(p)}</i>
    </div>
    """, unsafe_allow_html=True)
