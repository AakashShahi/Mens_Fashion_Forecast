import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="🏷 Hashtag Heatmap", layout="wide")

# ================================================================
# HEADER
# ================================================================
st.title("🏷 Social Media Hashtag Heatmap")
st.caption("Trend signals from Kathmandu male (17–25) fashion cohort")

DATA_PATH = os.path.join("data", "social_cohort_kathmandu_male_17_25.csv")

if not os.path.exists(DATA_PATH):
    st.error("❌ Cohort CSV missing.")
    st.stop()

df = pd.read_csv(DATA_PATH)
df["Hashtags"] = df["Hashtags"].fillna("")

# Extract hashtags (cleaned)
tags = (
    df["Hashtags"]
    .str.replace(",", " ", regex=False)
    .str.split()
    .explode()
    .str.lower()
    .str.strip()
)
tags = tags[tags.str.startswith("#")]

# ================================================================
# 1) TOP TRENDING HASHTAGS
# ================================================================
st.subheader("🔥 Top 20 Trending Hashtags")

top20 = tags.value_counts().head(20)
fig_top = px.bar(
    top20,
    x=top20.index,
    y=top20.values,
    text=top20.values,
    color=top20.values,
    color_continuous_scale="Oranges",
    title="Trending Hashtags",
)
fig_top.update_layout(xaxis_title="Hashtag", yaxis_title="Frequency", height=450)
st.plotly_chart(fig_top, use_container_width=True)


# ================================================================
# 2) WORDCLOUD
# ================================================================
st.subheader("☁️ Global Hashtag WordCloud")

wc = WordCloud(
    background_color="black",
    width=1400,
    height=700,
    colormap="viridis",
    min_font_size=10
)

wc_img = wc.generate(" ".join(tags.dropna().tolist()))

fig_wc, ax_wc = plt.subplots(figsize=(14, 6))
ax_wc.imshow(wc_img, interpolation="bilinear")
ax_wc.axis("off")
st.pyplot(fig_wc)


# ================================================================
# 3) CATEGORY × HASHTAG HEATMAP (FIXED & IMPROVED)
# ================================================================
st.subheader("🎨 Category × Hashtag Frequency Heatmap")

# You can extend this list anytime
category_keywords = {
    "Hoodies": ["hoodie", "hoodies", "oversized", "streetwear"],
    "Cargo Pants": ["cargo", "baggy", "utility", "wideleg"],
    "Crop Tops": ["crop", "kfashion", "kstyle"],
    "Jackets": ["jacket", "outerwear", "layering"],
    "Pants": ["pants", "trousers", "minimalist"],
    "T-Shirts": ["tshirt", "tee", "graphictees"]
}

heatmap_data = {}
for cat, kws in category_keywords.items():
    heatmap_data[cat] = sum(tags.str.contains("|".join(kws), regex=True))

heat_df = pd.DataFrame.from_dict(heatmap_data, orient="index", columns=["Hashtag Frequency"])

fig_heat, ax_heat = plt.subplots(figsize=(8, 4))
sns.heatmap(heat_df, annot=True, cmap="Purples", fmt='g', ax=ax_heat)
st.pyplot(fig_heat)


# ================================================================
# 4) HASHTAG CO-OCCURRENCE NETWORK (MICRO-TREND FINDER)
# ================================================================
st.subheader("🔗 Hashtag Co-Occurrence Map (Micro-Trends)")

# Build co-occurrence matrix
pair_counts = {}

for text in df["Hashtags"]:
    items = (
        str(text)
        .replace(",", " ")
        .split()
    )
    items = [h.lower() for h in items if h.startswith("#")]
    
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            pair = tuple(sorted([items[i], items[j]]))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

pair_df = pd.DataFrame(
    [(a, b, c) for (a, b), c in pair_counts.items() if c > 3],  # threshold
    columns=["Tag1", "Tag2", "Count"]
)

if len(pair_df) == 0:
    st.info("Not enough co-occurring hashtags to build a network.")
else:
    fig_network = px.scatter(
        pair_df,
        x="Tag1",
        y="Tag2",
        size="Count",
        color="Count",
        color_continuous_scale="Tealgrn",
        title="Co-Occurring Hashtags",
    )
    st.plotly_chart(fig_network, use_container_width=True)


# ================================================================
# 5) FAST MICRO-TREND DETECTOR
# ================================================================
st.subheader("📈 Micro-Trend Detector (Sudden Spike Finder)")

rolling = tags.value_counts().rolling(window=3).mean().sort_values(ascending=False)
micro = rolling.head(10)

fig_micro = px.bar(
    micro,
    x=micro.index,
    y=micro.values,
    text=micro.values,
    title="Fastest Emerging Micro-Trends",
    color=micro.values,
    color_continuous_scale="Bluered"
)

st.plotly_chart(fig_micro, use_container_width=True)
