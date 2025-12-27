import streamlit as st
import pandas as pd
import os, joblib, io, glob
import matplotlib.pyplot as plt
import importlib.util
from gtts import gTTS
import base64
import sys
import subprocess

# ============================================
# Load compute_reorder dynamically
# ============================================
try:
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "05_inventory_opt.py"))
    spec = importlib.util.spec_from_file_location("inventory_opt", src_path)
    inventory_opt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inventory_opt)
    compute_reorder = inventory_opt.compute_reorder

    # Import Insights Module
    insights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "insights.py"))
    spec_ins = importlib.util.spec_from_file_location("insights", insights_path)
    insights = importlib.util.module_from_spec(spec_ins)
    spec_ins.loader.exec_module(insights)

except Exception as e:
    st.error(f"Error loading modules: {e}")
    compute_reorder = None
    insights = None

# ============================================
# Streamlit Config
# ============================================
st.set_page_config(page_title="Kathmandu Fashion Forecaster", layout="wide", page_icon="👕")

# ============================================
# Custom CSS
# ============================================
st.markdown("""
<style>
    body {background-color: #0e0e10; color: #f0f0f5;}
    .stApp {background-color: #0e0e10;}
    h1, h2, h3, h4 {color: #f8f8ff;}
    .stButton>button {
        background-color: #2f2f35; color: white; border-radius: 10px;
        border: 1px solid #444; padding: 0.6em 1.2em;
    }
    .stButton>button:hover {background-color: #3e3e45; border-color: #777;}
    .suggest-card {
        background-color: #1c1c22; border-radius: 12px; padding: 20px;
        margin-top: 15px; border: 1px solid #333; box-shadow: 0 0 8px rgba(255,255,255,0.05);
    }
    .metric-card {
        background:#1d1d26; padding:20px; margin:10px; border-radius:12px; border:1px solid #444; 
    }
    .big-number { font-size: 2.5em; font-weight: bold; color: #00ff88; }
    .explanation { font-size: 1.0em; color: #b0b0bb; margin-top: 5px; }
    .risk-high { border-left: 5px solid #ff4444; }
    .risk-med { border-left: 5px solid #ffaa00; }
    .risk-low { border-left: 5px solid #00ff88; }
</style>
""", unsafe_allow_html=True)

# ============================================
# Header
# ============================================
st.title("👕 Kathmandu Youth Fashion Forecaster")
st.caption("AI-Powered Trend Prediction & Inventory Optimization")

FEATURE_DIR = os.path.join("data", "features")
DATA_DIR = "data"
MODEL_DIR = "models"
INSIGHTS_REPORT = "INSIGHTS_REPORT.md"

# ============================================
# Tabs Structure
# ============================================
tab_dash, tab_acc, tab_cluster, tab_exec, tab_prof = st.tabs(["📊 Smart Dashboard", "🎯 Model Accuracy", "🧩 Smart Segmentation", "📝 Executive Summary", "📋 Data Profile"])

# ============================================
# TAB 1: DASHBOARD
# ============================================
with tab_dash:
    # ------------------------------------------------
    # Sidebar / Controls
    # ------------------------------------------------
    with st.expander("⚙️ Forecast Controls", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            cats = []
            if os.path.exists(FEATURE_DIR):
                cats = sorted([f[len("feat_"):-4].replace("_", " ").title() for f in os.listdir(FEATURE_DIR) if f.startswith("feat_")])
            
            if cats:
                sel = st.selectbox("Select Category", cats)
            else:
                st.warning("No categories found. Run pipeline.")
                sel = None
                
            horizon = st.slider("Forecast Horizon (Days)", 30, 180, 90, 30)
            
        with col2:
            st.info("ℹ️ **To update data:** Go to the 'Update Data' page in the sidebar.")

    if sel:
        # Load Data
        feat_file = os.path.join(FEATURE_DIR, f"feat_{sel.lower().replace(' ', '_')}.csv")
        if os.path.exists(feat_file):
            df = pd.read_csv(feat_file, parse_dates=["Date"])
            
            # Show Data Coverage Insight
            if insights:
                coverage_info = insights.describe_forecast_horizon(df)
                st.info(f"📅 **Data Context:** {coverage_info}")

            # ------------------------------------------------
            # AI Recommendation Engine
            # ------------------------------------------------
            if compute_reorder:
                try:
                    rec = compute_reorder(sel, horizon=horizon)
                    
                    risk_class = "risk-low"
                    if "High" in rec['RiskLevel']: risk_class = "risk-high"
                    elif "Moderate" in rec['RiskLevel']: risk_class = "risk-med"
                    
                    # Generate Prescriptive Advice
                    advice = ""
                    if insights:
                        advice = insights.prescriptive_action(rec['RiskLevel'], rec['TrendScore'])
                    else:
                        advice = rec['RecommendationText']

                    # Layout
                    c1, c2 = st.columns([1, 1.5])
                    
                    with c1:
                        # 1. Action Card
                        st.markdown(f"""
                        <div class="metric-card {risk_class}">
                            <h3>📢 Action Recommendation</h3>
                            <span class="big-number">Order {rec['RecommendedOrderQty']} Units</span>
                            <p class="explanation">{advice}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 2. Context Card
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>🎯 Why this number?</h3>
                            <ul>
                                <li><b>Trend Score:</b> {rec['TrendScore']}/10 (Social Buzz + Google Search)</li>
                                <li><b>Psychology:</b> {rec['PsychologyScore']:.2f} (Fits '{rec['PersonaTarget']}' tribe)</li>
                                <li><b>Risk:</b> {rec['RiskLevel']}</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

                        if st.button("🔊 Play Audio Suggestion"):
                            text_to_speak = f"For {sel}, we suggest ordering {rec['RecommendedOrderQty']} units. {advice.replace('*', '')}"
                            tts = gTTS(text=text_to_speak, lang='en')
                            tts.save("suggestion.mp3")
                            st.audio("suggestion.mp3")

                    with c2:
                        st.subheader(f"📈 Forecast: {sel}")
                        
                        # Prophet Forecast Visualization
                        model_path = os.path.join(MODEL_DIR, f"prophet_{sel.lower().replace(' ', '_')}.pkl")
                        if os.path.exists(model_path):
                            model = joblib.load(model_path)
                            
                            # DYNAMIC HORIZON:
                            # Start future dataframe from the last date in the actual data
                            last_date = df['Date'].max()
                            future = model.make_future_dataframe(periods=horizon)
                            
                            if "InterestScore" in df.columns:
                                lts = df["InterestScore"].iloc[-1]
                                merged = df.rename(columns={"Date": "ds"})[["ds", "InterestScore"]]
                                future = future.merge(merged, on="ds", how="left")
                                future["InterestScore"] = future["InterestScore"].fillna(lts)
                            
                            forecast = model.predict(future)
                            
                            # Plotly for better interaction
                            fig = model.plot(forecast)
                            st.pyplot(fig)
                            
                            # Insights on Trend Direction
                            if insights:
                                recent_trend = insights.trend_direction_text(forecast['yhat'])
                                st.caption(f"🤖 **Model Insight:** {recent_trend}")
                        else:
                            st.line_chart(df.set_index("Date")["UnitsSold"].rolling(7).mean().tail(180))

                except Exception as e:
                    st.error(f"Error calculating inventory: {e}")

# ============================================
# TAB 2: MODEL ACCURACY
# ============================================
with tab_acc:
    st.header("🎯 Predictive Model Performance (Benchmarking)")
    st.markdown("Comparison of **XGBoost**, **Prophet**, and **Random Forest** (New).")
    
    metrics_path = os.path.join(MODEL_DIR, "metrics_summary.csv")
    if os.path.exists(metrics_path):
        met = pd.read_csv(metrics_path)
        
        # Best Model Per Category
        st.subheader("🏆 Champion Models")
        best_models = met.loc[met.groupby("Category")["RMSE"].idxmin()]
        st.dataframe(best_models[["Category", "Model", "RMSE", "MAE"]].style.highlight_min(axis=0, subset=["RMSE"], color="#16351c"))

        st.markdown("---")
        st.subheader("📊 Full Comparison Breakdown")
        col_view, col_def = st.columns([2, 1])
        
        with col_view:
            st.dataframe(met)
        
        with col_def:
            st.info("""
            **Thesis Note:**
            We test multiple algorithms (Prophet, XGBoost, Random Forest) and select the one with the lowest RMSE for each category to ensure robust forecasting.
            """)
    else:
        st.warning("⚠️ No accuracy metrics found. Please run the pipeline to generate them.")

# ============================================
# TAB 3: SMART SEGMENTATION (CLUSTERING)
# ============================================
with tab_cluster:
    st.header("🧩 Inventory Segmentation (K-Means)")
    st.caption("Using Unsupervised Learning to classify products into strategic groups.")
    
    clus_path = os.path.join(FEATURE_DIR, "product_clusters.csv")
    if os.path.exists(clus_path):
        clus_df = pd.read_csv(clus_path)
        
        c1, c2 = st.columns([2, 1])
        
        with c1:
            # Scatter Plot: Volatility vs Sales
            fig, ax = plt.subplots()
            scatter = ax.scatter(clus_df["AvgSales"], clus_df["Volatility"], c=clus_df["ClusterID"], cmap="viridis", s=100)
            ax.set_xlabel("Average Daily Sales (Volume)")
            ax.set_ylabel("Volatility (Risk)")
            ax.set_title("Product Strategy Matrix")
            
            # Annotate
            for i, txt in enumerate(clus_df["Category"]):
                ax.annotate(txt, (clus_df["AvgSales"][i], clus_df["Volatility"][i]))
                
            st.pyplot(fig)
            
        with c2:
            st.dataframe(clus_df[["Category", "ClusterLabel", "TrendScore"]])
            
    else:
        st.warning("⚠️ No clusters found. Run pipeline.")

# ============================================
# TAB 4: EXECUTIVE SUMMARY
# ============================================
with tab_exec:
    st.header("📝 AI Generated Insights Report")
    
    if os.path.exists(INSIGHTS_REPORT):
        with open(INSIGHTS_REPORT, "r", encoding="utf-8") as f:
            report_content = f.read()
        st.markdown(report_content)
    else:
        st.warning("Report not found. Please run the pipeline first.")
        
    st.markdown("---")
    st.subheader("Trending Hashtags Cloud")
    soc_path = os.path.join("data", "social_cohort_kathmandu_male_17_25.csv")
    if os.path.exists(soc_path):
        soc = pd.read_csv(soc_path)
        tags = soc["Hashtags"].fillna("").str.replace(",", " ").str.split().explode().str.lower().str.strip("#")
        top_tags = tags.value_counts().head(20)
        st.bar_chart(top_tags)

# ============================================
# TAB 5: DATA PROFILE (Data Dictionary)
# ============================================
with tab_prof:
    st.header("📋 Data Understanding (Phase 2 & 3)")
    
    # List all dictionary files
    dict_files = glob.glob(os.path.join(DATA_DIR, "dictionary_*.md"))
    
    if dict_files:
        selected_dict = st.selectbox("Select Dataset to View", [os.path.basename(f) for f in dict_files])
        full_p = os.path.join(DATA_DIR, selected_dict)
        with open(full_p, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("No data dictionaries found. Run the pipeline to generate them.")
        
    st.markdown("---")
    st.subheader("Raw Data Files Preview")
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    sel_csv = st.selectbox("Preview CSV", [os.path.basename(f) for f in csv_files])
    if sel_csv:
        st.dataframe(pd.read_csv(os.path.join(DATA_DIR, sel_csv)).head(50))
