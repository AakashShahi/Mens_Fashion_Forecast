# src/04_evaluate.py  (compatibility-fixed)
import pandas as pd, os, numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
PRED_DIR = os.path.join(MODEL_DIR, "predictions")
OUT = os.path.join(MODEL_DIR, "metrics_summary.csv")

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

def safe_rmse(y_true, y_pred):
    # convert to numpy arrays and compute RMSE robustly
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

metrics = []
if not os.path.exists(PRED_DIR):
    raise FileNotFoundError("Predictions directory not found: " + PRED_DIR)

for fn in sorted(os.listdir(PRED_DIR)):
    fp = os.path.join(PRED_DIR, fn)
    # Prophet forecasts (ds, yhat)
    if fn.startswith("prophet_forecast_") and fn.endswith(".csv"):
        cat = fn[len("prophet_forecast_"):-4].replace("_"," ").title()
        df = pd.read_csv(fp, parse_dates=["ds"])
        feat_fn = os.path.join("data","features", f"feat_{cat.lower().replace(' ','_')}.csv")
        if os.path.exists(feat_fn):
            feat = pd.read_csv(feat_fn, parse_dates=["Date"])
            act = feat.rename(columns={"Date":"ds","UnitsSold":"y"})[["ds","y"]]
            merged = pd.merge(act, df, on="ds", how="inner")
            if len(merged) >= 30:
                y_true = merged["y"].values[-30:]
                y_pred = merged["yhat"].values[-30:]
                rmse = safe_rmse(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                mapev = mape(y_true, y_pred)
                metrics.append({"Model":"Prophet","Category":cat,"RMSE":rmse,"MAE":mae,"MAPE":mapev})
    # XGBoost predictions (UnitsSold, y_pred)
    if fn.startswith("xgb_preds_") and fn.endswith(".csv"):
        cat = fn[len("xgb_preds_"):-4].replace("_"," ").title()
        df = pd.read_csv(fp)
        if "UnitsSold" in df.columns and "y_pred" in df.columns:
            y_true = df["UnitsSold"].values
            y_pred = df["y_pred"].values
            rmse = safe_rmse(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            mapev = mape(y_true, y_pred)
            metrics.append({"Model":"XGBoost","Category":cat,"RMSE":rmse,"MAE":mae,"MAPE":mapev})

mdf = pd.DataFrame(metrics)
mdf.to_csv(OUT, index=False)
print("Saved metrics to", OUT)
print(mdf)
