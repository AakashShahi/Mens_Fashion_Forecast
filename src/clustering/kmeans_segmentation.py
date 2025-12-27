# src/clustering/kmeans_segmentation.py
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
FEATURE_DIR = os.path.join(ROOT, "data", "features")
OUTPUT_PATH = os.path.join(FEATURE_DIR, "product_clusters.csv")

# -------------------------------------------------------------------------
# CLUSTERING ENGINE
# -------------------------------------------------------------------------
class SegmentationEngine:
    def __init__(self):
        self.features = []
        
    def load_features(self):
        """Aggregates features from all category files."""
        if not os.path.exists(FEATURE_DIR):
            return

        feat_files = [f for f in os.listdir(FEATURE_DIR) if f.startswith("feat_") and f.endswith(".csv")]
        
        all_data = []
        
        for f in feat_files:
            cat_name = f[len("feat_"):-4].replace("_", " ").title()
            path = os.path.join(FEATURE_DIR, f)
            df = pd.read_csv(path)
            
            # Feature Extraction per Category
            avg_sales = df["UnitsSold"].mean()
            volatility = df["UnitsSold"].std()
            trend_score = df["InterestScore"].iloc[-1] if "InterestScore" in df.columns else 0
            
            all_data.append({
                "Category": cat_name,
                "AvgSales": avg_sales,
                "Volatility": volatility,
                "TrendScore": trend_score
            })
            
        return pd.DataFrame(all_data)

    def run_clustering(self, df):
        """Runs K-Means to identify 4 strategic clusters."""
        if df.empty or len(df) < 4:
            print("⚠️ Not enough categories for clustering. Need at least 4.")
            return df
        
        # Prepare Data
        X = df[["AvgSales", "Volatility", "TrendScore"]].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-Means (k=4 for Cash Cow, Star, Question Mark, Dog)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df["ClusterID"] = kmeans.fit_predict(X_scaled)
        
        # Labeling Logic (Simplified Heuristic)
        # We classify clusters based on their centroids relative to the mean
        
        def label_cluster(row):
            # Higher Sales + Low Volatility = Cash Cow
            # High Sales + High Trend = Star
            # Low Sales + High Trend = Opportunity
            # Low Sales + Low Trend = Risk
            # (Note: This is a simplification; optimal labeling requires analyzing centroids)
            return f"Cluster {row['ClusterID']}"

        df["ClusterLabel"] = df.apply(label_cluster, axis=1)
        
        # Better Labeling based on rank
        # Rank clusters by AvgSales
        cluster_rank = df.groupby("ClusterID")["AvgSales"].mean().sort_values(ascending=False).index
        rank_map = {
            cluster_rank[0]: "⭐ Star (High Volume)",
            cluster_rank[1]: "💰 Cash Cow (Steady)",
            cluster_rank[2]: "❓ Opportunity (Volatile)",
            cluster_rank[3]: "⚠️ Risk (Low Performance)"
        }
        
        if len(cluster_rank) < 4:
             # Fallback if fewer clusters emerged
             pass 
        else:
            df["ClusterLabel"] = df["ClusterID"].map(rank_map)

        return df

    def execute(self):
        print("🧩 Starting K-Means Inventory Segmentation...")
        df = self.load_features()
        if df is None or df.empty:
            print("❌ No features found to cluster.")
            return

        final_df = self.run_clustering(df)
        final_df.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ Segmentation saved to: {OUTPUT_PATH}")
        print(final_df[["Category", "ClusterLabel"]])

if __name__ == "__main__":
    SegmentationEngine().execute()
