# src/04_evaluate.py  (THESIS UPGRADE: TIME SERIES CV)
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

ROOT = os.path.dirname(os.path.dirname(__file__)) if __file__ else "."
DATA_DIR = os.path.join(ROOT, "data")
FEATURE_DIR = os.path.join(DATA_DIR, "features")
MODEL_DIR = os.path.join(ROOT, "models")
PRED_DIR = os.path.join(MODEL_DIR, "predictions")

OUT = os.path.join(MODEL_DIR, "metrics_summary.csv")


# ---------------------------------------------------------
# Safe MAPE — zero-resistant
# ---------------------------------------------------------
def safe_mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0: return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100)


# ---------------------------------------------------------
# Safe RMSE
# ---------------------------------------------------------
def safe_rmse(y_true, y_pred):
    if len(y_true) == 0: return None
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ---------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------
def evaluate():
    if not os.path.exists(PRED_DIR):
        print("Predictions dir missing")
        return pd.DataFrame()

    results = []

    # Helper to process a prediction file
    def process_pred(model_name, cat, df, y_col="UnitsSold", pred_col="y_pred"):
        y_true = df[y_col].values
        y_pred = df[pred_col].values
        
        # Calculate scores
        rmse = safe_rmse(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = safe_mape(y_true, y_pred)
        
        results.append({
            "Model": model_name,
            "Category": cat,
            "RMSE": round(rmse, 3),
            "MAE": round(mae, 3),
            "MAPE": round(mape, 3) if mape else None
        })

    for fn in sorted(os.listdir(PRED_DIR)):
        fp = os.path.join(PRED_DIR, fn)
        
        # 1. XGBoost
        if fn.startswith("xgb_preds_") and fn.endswith(".csv"):
            cat = fn[len("xgb_preds_"):-4].replace("_", " ").title()
            df = pd.read_csv(fp)
            process_pred("XGBoost", cat, df)

        # 2. Random Forest (New)
        if fn.startswith("rf_preds_") and fn.endswith(".csv"):
            cat = fn[len("rf_preds_"):-4].replace("_", " ").title()
            df = pd.read_csv(fp)
            process_pred("Random Forest", cat, df)
            
        # 3. Prophet
        if fn.startswith("prophet_forecast_") and fn.endswith(".csv"):
            cat = fn[len("prophet_forecast_"):-4].replace("_", " ").title()
            pred = pd.read_csv(fp, parse_dates=["ds"])
            
            # We need ground truth from feature file
            feat_path = os.path.join(FEATURE_DIR, f"feat_{cat.lower().replace(' ','_')}.csv")
            if os.path.exists(feat_path):
                feat = pd.read_csv(feat_path, parse_dates=["Date"])
                feat = feat.rename(columns={"Date": "ds", "UnitsSold": "y"})
                
                merged = pd.merge(feat, pred, on="ds", how="inner")
                if len(merged) > 10:
                    # Evaluate on last 30 days logic (similar to CV fold)
                    eval_df = merged.tail(30)
                    process_pred("Prophet", cat, eval_df, y_col="y", pred_col="yhat")

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df.to_csv(OUT, index=False)
        print("\n📊 Saved evaluation metrics to:", OUT)
        print(res_df.sort_values("RMSE").head(10))
    
    return res_df

if __name__ == "__main__":
    evaluate()
