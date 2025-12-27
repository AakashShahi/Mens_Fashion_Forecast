import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="🧠 Psychology Engine", layout="wide")

# ==========================================================
# HEADER
# ==========================================================
st.title("🧠 Fashion Psychology Engine Dashboard")
st.caption("Conformity • Aspirational Influence • Identity Fit • Cultural Resonance • Availability Bias")

DATA_PATH = os.path.join("data", "features", "psychology_scores.csv")

# ==========================================================
# LOAD DATA
# ==========================================================
if not os.path.exists(DATA_PATH):
    st.error("❌ psychology_scores.csv missing. Run Psychology Engine pipeline.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Ensure numeric fields
for col in ["Conformity", "Aspirational", "Identity", "Cultural", "Availability", "PsychologyScore"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==========================================================
# MAIN TABLE
# ==========================================================
st.subheader("🧠 Full Psychology Metrics Table")
st.dataframe(df.style.highlight_max("PsychologyScore", color="#7ed957"))

# ==========================================================
# MAIN SCORE VISUALIZATION
# ==========================================================
st.subheader("✨ Psychology Score by Category")

fig_main = px.bar(
    df, x="Category", y="PsychologyScore",
    color="PsychologyScore",
    color_continuous_scale="Tealgrn",
    title="Final Psychology Influence Score"
)
st.plotly_chart(fig_main, use_container_width=True)


# ==========================================================
# CONTRIBUTION BREAKDOWN
# ==========================================================
st.subheader("📊 Component Contribution Breakdown")

selected_cat = st.selectbox("Select a Category", df["Category"])

row = df[df["Category"] == selected_cat].iloc[0]

pie_df = pd.DataFrame({
    "Component": ["Aspirational", "Conformity", "Identity", "Cultural", "Availability"],
    "Score": [
        row["Aspirational"],
        row["Conformity"],
        row["Identity"],
        row["Cultural"],
        row["Availability"]
    ]
})

fig_pie = px.pie(
    pie_df, values="Score", names="Component",
    title=f"Psychology Breakdown for {selected_cat}",
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig_pie, use_container_width=True)


# ==========================================================
# MULTI-METRIC COMPARISON GRID
# ==========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📣 Aspirational Influence")
    st.bar_chart(df.set_index("Category")["Aspirational"])

with col2:
    st.subheader("🧭 Cultural Resonance")
    st.bar_chart(df.set_index("Category")["Cultural"])

col3, col4 = st.columns(2)

with col3:
    st.subheader("👤 Identity Alignment")
    st.bar_chart(df.set_index("Category")["Identity"])

with col4:
    st.subheader("👥 Conformity Strength")
    st.bar_chart(df.set_index("Category")["Conformity"])


# ==========================================================
# INTERPRETIVE INSIGHTS
# ==========================================================
st.subheader("🧠 Interpretive Insights Per Category")

def psychology_narrative(r):
    score = r["PsychologyScore"]
    asp = r["Aspirational"]
    con = r["Conformity"]
    ident = r["Identity"]
    cult = r["Cultural"]

    if score > 0.6:
        return "🔥 Strong psychological pull — identity-driven + aspirational trend."
    if asp > 0.5:
        return "✨ High aspirational interest — K-fashion or influencer-driven."
    if con > 0.5:
        return "👥 Strong conformity — consumers are copying a dominant persona."
    if ident > 0.6:
        return "👤 Identity alignment is strong — fits a clear persona segment."
    if cult > 0.5:
        return "📣 Strong cultural signal — trending in Nepal’s social sphere."
    return "⚪ Mild psychological influence — not a strong behavioral driver."

for _, r in df.iterrows():
    st.markdown(f"""
    **{r['Category']}** → {psychology_narrative(r)}
    """)


# ==========================================================
# HEATMAP STYLE PERSONA FIT (Synthetic)
# ==========================================================
st.subheader("🎭 Optional: Persona Fit Heatmap")

persona_map = {
    "Hoodies": ["K-Fashion", "Streetwear"],
    "Cargo Pants": ["K-Fashion", "Streetwear"],
    "Crop Tops": ["Indie", "K-Fashion"],
    "Jackets": ["Minimalist", "Streetwear"],
    "Pants": ["Minimalist"],
    "T-Shirts": ["Streetwear", "Minimalist"]
}

heat_rows = []
for _, r in df.iterrows():
    cat = r["Category"]
    psych = r["PsychologyScore"]
    personas = persona_map.get(cat, [])
    for p in personas:
        heat_rows.append([cat, p, psych])

heat_df = pd.DataFrame(heat_rows, columns=["Category", "Persona", "Score"])

if not heat_df.empty:
    fig_heat = px.density_heatmap(
        heat_df,
        x="Persona", y="Category", z="Score",
        color_continuous_scale="Viridis",
        title="Persona Fit Heatmap (Influence Strength)"
    )
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("No persona mapping available for heatmap.")

