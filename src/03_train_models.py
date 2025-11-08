import pandas as pd
import os, joblib, numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(__file__)) if __file__ else "."
FEATURE_DIR = os.path.join(ROOT, "data", "features")
MODEL_DIR = os.path.join(ROOT, "models")
PRED_DIR = os.path.join(MODEL_DIR, "predictions")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

def safe_rmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def train_prophet(cat):
    fn = os.path.join(FEATURE_DIR, f"feat_{cat.lower().replace(' ','_')}.csv")
    df = pd.read_csv(fn, parse_dates=["Date"])
    p_df = df.rename(columns={"Date":"ds","UnitsSold":"y"})[["ds","y","InterestScore"]]
    if len(p_df) < 40:
        print(f"Not enough rows to train Prophet for {cat}. Need more data.")
        return None
    train = p_df.iloc[:-30]
    val = p_df.iloc[-30:].copy()
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
    m.add_regressor("InterestScore")
    m.fit(train)

    future = m.make_future_dataframe(periods=30)
    future = future.merge(p_df[["ds","InterestScore"]], on="ds", how="left")
    # use ffill for forward filling
    future["InterestScore"] = future["InterestScore"].ffill().fillna(0)

    fc = m.predict(future)
    joblib.dump(m, os.path.join(MODEL_DIR, f"prophet_{cat.lower().replace(' ','_')}.pkl"))
    fc[['ds','yhat','yhat_lower','yhat_upper']].to_csv(os.path.join(PRED_DIR, f"prophet_forecast_{cat.lower().replace(' ','_')}.csv"), index=False)

    # Metrics on last 30 days (ensure alignment)
    val_fc = fc[fc['ds'].isin(val['ds'])]
    if len(val_fc) >= len(val):
        y_true = val['y'].values
        y_pred = val_fc['yhat'].values[:len(y_true)]
        rmse = safe_rmse(y_true, y_pred)
    else:
        rmse = None
    print(f"Prophet {cat} RMSE: {rmse}")
    return rmse

def train_xgb(cat):
    fn = os.path.join(FEATURE_DIR, f"feat_{cat.lower().replace(' ','_')}.csv")
    df = pd.read_csv(fn, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in ("Date","UnitsSold")]
    if len(df) < 40:
        print(f"Not enough rows to train XGBoost for {cat}.")
        return None
    X = df[feat_cols].fillna(0)
    y = df["UnitsSold"].values
    split = len(df)-30
    X_train, y_train = X.iloc[:split], y[:split]
    X_test, y_test = X.iloc[split:], y[split:]
    # Simpler fit call to maximize compatibility
    model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODEL_DIR, f"xgb_{cat.lower().replace(' ','_')}.joblib"))
    preds = model.predict(X_test)
    rmse = safe_rmse(y_test, preds)
    df_test = df.iloc[split:].copy()
    df_test["y_pred"] = preds
    df_test.to_csv(os.path.join(PRED_DIR, f"xgb_preds_{cat.lower().replace(' ','_')}.csv"), index=False)
    print(f"XGB {cat} RMSE: {rmse}")
    return rmse

if __name__ == "__main__":
    # detect categories from feature files
    if not os.path.exists(FEATURE_DIR):
        raise FileNotFoundError("Feature directory not found: " + FEATURE_DIR)
    fns = [f for f in os.listdir(FEATURE_DIR) if f.startswith("feat_") and f.endswith(".csv")]
    cats = [f[len("feat_"):-4].replace("_"," ").title() for f in fns]
    print("Categories detected:", cats)
    for c in cats:
        print("Training", c)
        try:
            train_prophet(c)
        except Exception as e:
            print("Prophet error:", e)
        try:
            train_xgb(c)
        except Exception as e:
            print("XGB error:", e)
    print("Training finished.")