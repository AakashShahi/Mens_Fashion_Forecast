import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

st.set_page_config(page_title="🏷 Hashtag Heatmap", layout="wide")

st.title("🏷 Social Media Hashtag Heatmap")
st.write("Analyzing social media trends from Kathmandu male (17–25) fashion cohort.")

DATA_PATH = os.path.join("data", "social_cohort_kathmandu_male_17_25.csv")

if not os.path.exists(DATA_PATH):
    st.error("Cohort file missing.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Prepare hashtag list
df["Hashtags"] = df["Hashtags"].fillna("")
tags = df["Hashtags"].str.split(",| ").explode().str.lower().str.strip()
tags = tags[tags.str.startswith("#")]

# ---------------- Top Hashtags -----------------
st.subheader("🔥 Top 20 Trending Hashtags")

top20 = tags.value_counts().head(20)
st.bar_chart(top20)

# ---------------- Hashtag Wordcloud -----------------
st.subheader("☁️ Hashtag WordCloud")

wc = WordCloud(background_color="black", width=1200, height=600, colormap="cool")
wc_img = wc.generate(" ".join(tags.dropna().tolist()))

fig, ax = plt.subplots(figsize=(14, 7))
ax.imshow(wc_img, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# ---------------- Category Hashtag Heatmap -----------------
st.subheader("🎨 Category × Hashtag Presence (Heatmap)")

categories = ["hoodie", "jacket", "pants", "cargo", "tshirt", "crop", "streetwear", "kstyle", "kfashion"]

heat_data = {cat: [] for cat in categories}

for cat in categories:
    count = tags[tags.str.contains(cat, regex=False)].count()
    heat_data[cat].append(count)

heat_df = pd.DataFrame(heat_data, index=["Frequency"])

fig2, ax2 = plt.subplots(figsize=(10, 3))
sns.heatmap(heat_df, annot=True, cmap="Purples", ax=ax2)
st.pyplot(fig2)
