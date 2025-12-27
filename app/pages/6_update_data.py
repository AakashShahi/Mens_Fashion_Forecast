import streamlit as st
import pandas as pd
import os
import sys
import subprocess
import joblib

# Set Page Config
st.set_page_config(page_title="Update Data | Fashion Forecaster", page_icon="⚡", layout="wide")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
FEATURE_DIR = os.path.join(DATA_DIR, "features")

# Custom CSS
st.markdown("""
<style>
    body {background-color: #0e0e10; color: #f0f0f5;}
    .stApp {background-color: #0e0e10;}
    .stButton>button {
        background-color: #00ff88; color: black; border-radius: 10px;
        border: none; padding: 0.6em 1.2em; font-weight: bold;
    }
    .stButton>button:hover {background-color: #00cc66;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Helper Functions
# ----------------------------
def run_full_pipeline():
    """Runs the master pipeline.py script."""
    pipeline_script = os.path.join(BASE_DIR, "src", "pipeline.py")
    
    # FORCE UTF-8 to prevent Windows UnicodeEncodeError with emojis
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    result = subprocess.run(
        [sys.executable, pipeline_script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=BASE_DIR,
        env=env
    )
    
    return {"pipeline": result}

# ----------------------------
# Main Page
# ----------------------------
st.title("⚡ Dynamic Forecasting Engine")
st.markdown("### Upload New Sales Data & Retrain Models")
st.caption("Upload your latest sales CSV to update the forecast models and inventory recommendations instantly.")

col1, col2 = st.columns([1, 1])

with col1:
    st.info("""
    **Required CSV Format:**
    - **Demographics:** Male, Age 17-25, Kathmandu
    - **Columns:** Date, Category, UnitsSold, City, AgeGroup, Gender
    """)
    
    # SAMPLE DATA UI
    with st.expander("ℹ️ See Sample CSV Format"):
        sample_data = pd.DataFrame({
            "Date": ["2025-09-01", "2025-09-02"],
            "Category": ["Hoodies", "Hoodies"],
            "UnitsSold": [5, 12],
            "City": ["Kathmandu", "Kathmandu"],
            "AgeGroup": ["17-25", "17-25"],
            "Gender": ["Male", "Male"]
        })
        st.table(sample_data)

with col2:
    uploaded_file = st.file_uploader("Upload Updated CSV (Appends/Updates Data)", type=["csv"])

# ----------------------------
# UPLOAD LOGIC
# ----------------------------
if uploaded_file:
    target_path = os.path.join(DATA_DIR, "sales_cohort_kathmandu_male_17_25.csv")
    
    try:
        if os.path.exists(target_path):
            existing_df = pd.read_csv(target_path)
        else:
            existing_df = pd.DataFrame()
        
        new_df = pd.read_csv(uploaded_file)
        
        # Ensure columns match
        common_cols = list(set(existing_df.columns) & set(new_df.columns))
        if len(common_cols) < 3:
            st.error("❌ The uploaded CSV columns don't match the existing dataset format.")
        else:
            # ------------------------------------------------
            # DEMOGRAPHIC FILTER (Thesis Requirement)
            # ------------------------------------------------
            valid_ages = ["17-25", "15-19", "20-24", "17-24", "18-25"]
            
            # Normalize for filtering
            new_df["Gender_Clean"] = new_df["Gender"].astype(str).str.strip().str.lower()
            new_df["City_Clean"] = new_df["City"].astype(str).str.strip().str.lower()
            new_df["AgeGroup_Clean"] = new_df["AgeGroup"].astype(str).str.strip()

            # Debugging: Show what we are seeing before filtering
            st.write("### 🔍 Debugging Upload Data")
            st.write("Unique Genders found:", new_df["Gender"].unique())
            st.write("Unique AgeGroups found:", new_df["AgeGroup"].unique())
            st.write("Unique Cities found:", new_df["City"].unique())

            mask = (
                (new_df["Gender_Clean"] == "male") & 
                (new_df["City_Clean"] == "kathmandu") &
                (new_df["AgeGroup_Clean"].isin(valid_ages))
            )
            
            filtered_df = new_df[mask].drop(columns=["Gender_Clean", "City_Clean", "AgeGroup_Clean"])
            dropped_rows = len(new_df) - len(filtered_df)
            
            if len(filtered_df) == 0:
                st.error(f"❌ All {len(new_df)} rows were dropped!")
                st.error(f"**Expected:** Male, 17-25, Kathmandu")
                st.error(f"**Your Data contains:** {new_df['Gender'].unique()}, {new_df['AgeGroup'].unique()}, {new_df['City'].unique()}")
            else:
                if dropped_rows > 0:
                    st.warning(f"⚠️ Filtered out {dropped_rows} rows. Keeping {len(filtered_df)} valid rows.")
                
                # SAVING
                merged_df = pd.concat([existing_df, filtered_df], ignore_index=True)
                
                # Deduplicate
                merged_df["Date"] = pd.to_datetime(merged_df["Date"])
                merged_df = merged_df.drop_duplicates(subset=["Date", "Category", "City", "AgeGroup", "Gender"], keep="last")
                merged_df = merged_df.sort_values("Date")
                
                merged_df.to_csv(target_path, index=False)
                st.success(f"✅ Data Merged & Saved! Total Rows: {len(merged_df)}")
                st.info(f"💾 Saved to: {target_path}")

    except Exception as e:
        st.error(f"Error processing CSV: {e}")

# ----------------------------
# RETRAIN LOGIC
# ----------------------------
st.markdown("---")
if st.button("🚀 Process & Retrain Model"):
    st.info("Triggering Dynamic Pipeline...")
    st.markdown("**Status:** Cleaning Data → Feature Engineering → Training Models...")
    
    # Clear cache to force reload of new data
    st.cache_data.clear()
    
    # Run pipeline
    logs = run_full_pipeline()
    
    result = logs["pipeline"]
    if result.returncode == 0:
        st.success("✅ Pipeline Completed Successfully!")
        st.balloons()
        
        # Show Results Preview
        st.subheader("📋 Updated Inventory Recommendations")
        rec_path = os.path.join(FEATURE_DIR, "inventory_recommendations.csv")
        if os.path.exists(rec_path):
            rec_df = pd.read_csv(rec_path)
            st.dataframe(rec_df[["Category", "RecommendedOrderQty", "RiskLevel", "RecommendationText"]])

    else:
        st.error("❌ Pipeline Failed! See details below.")
    
    with st.expander("Show Pipeline Details (Terminal Output)"):
        st.text("STDOUT:")
        st.code(result.stdout)
        
        if result.stderr:
            st.text("STDERR:")
            st.code(result.stderr)
