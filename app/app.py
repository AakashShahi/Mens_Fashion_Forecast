# app/app.py
import streamlit as st
import pandas as pd, os, joblib, io
import matplotlib.pyplot as plt
import importlib.util
from gtts import gTTS
import base64

# ===== Load compute_reorder dynamically =====
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "05_inventory_opt.py"))
spec = importlib.util.spec_from_file_location("inventory_opt", src_path)
inventory_opt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory_opt)
compute_reorder = inventory_opt.compute_reorder

# ===== Page config =====
st.set_page_config(page_title="Kathmandu Youth Fashion Forecaster", layout="wide", page_icon="👕")

# ===== Custom CSS =====
st.markdown("""
<style>
    body {background-color: #0e0e10; color: #f0f0f5;}
    .stApp {background-color: #0e0e10;}
    h1, h2, h3, h4 {color: #f8f8ff;}
    .stButton>button {
        background-color: #2f2f35;
        color: white;
        border-radius: 10px;
        border: 1px solid #444;
        padding: 0.6em 1.2em;
    }
    .stButton>button:hover {background-color: #3e3e45; border-color: #777;}
    .uploadedFile {background-color: #1c1c20;}
    .suggest-card {
        background-color: #1c1c22;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        border: 1px solid #333;
        box-shadow: 0 0 8px rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ===== Header =====
st.title("👕 Kathmandu Youth Fashion Forecaster")
st.caption("Predicting urban male fashion trends (17–25, Kathmandu)")

FEATURE_DIR = os.path.join("data", "features")
MODEL_DIR = "models"

# ===== Category selection =====
cats = sorted([f[len("feat_"):-4].replace("_", " ").title()
               for f in os.listdir(FEATURE_DIR) if f.startswith("feat_")])
sel = st.selectbox("👔 Select Product Category", cats)
horizon = st.slider("📅 Forecast Horizon (days)", 30, 180, 90, 30)

# ===== CSV upload section =====
st.markdown("### 📤 Upload Your Updated Feature Data (Optional)")
st.write("Upload a CSV file containing your recent sales and trend data. It should look like this:")

# Example CSV
sample = pd.DataFrame({
    "Date": pd.date_range("2025-08-01", periods=6, freq="D"),
    "UnitsSold": [42, 47, 45, 53, 50, 52],
    "InterestScore": [0.63, 0.68, 0.70, 0.75, 0.74, 0.72]
})
st.dataframe(sample)

# Downloadable sample template
csv_buffer = io.StringIO()
sample.to_csv(csv_buffer, index=False)
st.download_button("⬇️ Download CSV Template", csv_buffer.getvalue(),
                   "fashion_forecast_template.csv", "text/csv")

# File uploader
uploaded_file = st.file_uploader("Upload new feature CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    st.success("✅ New CSV loaded successfully!")
    st.dataframe(df.head())
else:
    feat_fn = os.path.join(FEATURE_DIR, f"feat_{sel.lower().replace(' ', '_')}.csv")
    df = pd.read_csv(feat_fn, parse_dates=["Date"])
    st.info(f"Using default feature file: {feat_fn}")

# ===== Historical Chart =====
st.subheader("📈 Historical Trend (7-day Avg Units Sold)")
st.line_chart(df.set_index("Date")["UnitsSold"].rolling(7).mean().tail(180))

# ===== Load Prophet model =====
model_path = os.path.join(MODEL_DIR, f"prophet_{sel.lower().replace(' ', '_')}.pkl")
if not os.path.exists(model_path):
    st.error("❌ Model not found. Run src/03_train_models.py first.")
    st.stop()

model = joblib.load(model_path)

# ===== Forecast =====
future = model.make_future_dataframe(periods=horizon)
if "InterestScore" in df.columns:
    last_trend = df["InterestScore"].iloc[-1]
    future = future.merge(df[["Date", "InterestScore"]].rename(columns={"Date": "ds"}), on="ds", how="left")
    future["InterestScore"] = future["InterestScore"].fillna(last_trend)

forecast = model.predict(future)

# ===== Forecast Plots =====
st.subheader("🔮 Prophet Forecast")
fig = model.plot(forecast)
st.pyplot(fig)

st.subheader("🧩 Forecast Components")
st.pyplot(model.plot_components(forecast))

# ===== Smart Fashion Suggestion =====
recent_mean = forecast.tail(30)["yhat"].mean()
prev_mean = forecast.iloc[-60:-30]["yhat"].mean()
change_ratio = recent_mean / prev_mean if prev_mean != 0 else 1

if change_ratio > 1.2:
    suggestion = "📈 Strong upward trend! Increase stock for popular designs and prepare for higher demand. Launch bold promotions for Hoodies and streetwear."
    tone = "positive"
elif change_ratio < 0.85:
    suggestion = "📉 Demand dip expected. Consider reducing orders, offering discounts, or shifting focus to new collections."
    tone = "negative"
else:
    suggestion = "⚖️ Stable forecast. Maintain current stock levels and prepare upcoming seasonal designs."
    tone = "neutral"

# Display suggestion card
bg_color = {"positive": "#16351c", "negative": "#3a1f1f", "neutral": "#1f1f2e"}[tone]
st.markdown(f"""
<div class="suggest-card" style="background-color:{bg_color}">
<h3>🧠 Fashion Insight</h3>
<p style="font-size:1.1em;">{suggestion}</p>
</div>
""", unsafe_allow_html=True)

# ===== Voice Feature =====
if st.button("🔊 Hear Suggestion"):
    tts = gTTS(text=suggestion, lang='en')
    tts.save("suggestion.mp3")
    with open("suggestion.mp3", "rb") as audio_file:
        audio_bytes = audio_file.read()
        b64 = base64.b64encode(audio_bytes).decode()
        md = f"""
        <audio controls autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
        st.markdown(md, unsafe_allow_html=True)

# ===== Inventory Recommendation =====
try:
    rec = compute_reorder(sel, horizon=horizon)
    st.subheader("📦 Inventory Recommendation")
    st.json(rec)
except Exception as e:
    st.error(f"Inventory recommendation failed: {e}")

# ===== Hashtag Trends =====
st.subheader("🏷️ Trending Hashtags (Social Media Cohort)")
soc_fn = os.path.join("data", "social_cohort_kathmandu_male_17_25.csv")
if os.path.exists(soc_fn):
    soc = pd.read_csv(soc_fn)
    if not soc.empty and "Hashtags" in soc.columns:
        tags = soc["Hashtags"].dropna().str.split().explode().str.strip("#").str.lower()
        top = tags.value_counts().head(15)
        st.bar_chart(top)
    else:
        st.write("No hashtag data available.")
else:
    st.write("Social cohort file not found.")
