# src/03_train_models.py
import pandas as pd
import numpy as np
import os, joblib, warnings
from prophet import Prophet
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(__file__)) if __file__ else "."
FEATURE_DIR = os.path.join(ROOT, "data", "features")
MODEL_DIR = os.path.join(ROOT, "models")
PRED_DIR = os.path.join(MODEL_DIR, "predictions")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


# -------------------------------------------------------
# Safe RMSE
# -------------------------------------------------------
def safe_rmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    if len(y_true) == 0 or len(y_pred) == 0:
        return None
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# -------------------------------------------------------
# TRAIN PROPHET
# -------------------------------------------------------
def train_prophet(cat):
    fn = os.path.join(FEATURE_DIR, f"feat_{cat.lower().replace(' ','_')}.csv")
    if not os.path.exists(fn):
        print(f"❌ Feature file missing for {cat}")
        return None

    df = pd.read_csv(fn, parse_dates=["Date"])

    # Prophet input data
    p_df = df.rename(columns={"Date":"ds", "UnitsSold":"y"})[
        ["ds", "y", "InterestScore"]
    ].dropna()

    # Minimum requirement
    if len(p_df) < 7:
        print(f"⚠️ Prophet skipped for {cat}: Not enough rows ({len(p_df)}). Using Naive Forecast.")
        # Generate Naive Forecast (Flat line using mean)
        mean_val = p_df["y"].mean() if not p_df.empty else 0
        
        future_dates = pd.date_range(start=p_df["ds"].max() + pd.Timedelta(days=1), periods=30)
        naive_fc = pd.DataFrame({
            "ds": future_dates,
            "yhat": [mean_val] * 30,
            "yhat_lower": [mean_val * 0.8] * 30,
            "yhat_upper": [mean_val * 1.2] * 30
        })
        
        naive_fc.to_csv(
            os.path.join(PRED_DIR, f"prophet_forecast_{cat.lower().replace(' ','_')}.csv"),
            index=False
        )
        return None

    # Stability fixes
    p_df = p_df.sort_values("ds")
    p_df["InterestScore"] = p_df["InterestScore"].fillna(0)

    # ---- Train/Validation split ----
    train = p_df.iloc[:-30]
    val = p_df.iloc[-30:].copy()

    # ---- Prophet model ----
    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.5,
        seasonality_mode='multiplicative'
    )

    m.add_regressor("InterestScore")

    try:
        m.fit(train)
    except Exception as e:
        print(f"❌ Prophet failed for {cat}: {e}")
        return None

    # ---- Future prediction ----
    future = m.make_future_dataframe(periods=30)
    future = future.merge(
        p_df[["ds", "InterestScore"]],
        on="ds",
        how="left"
    )
    future["InterestScore"] = future["InterestScore"].ffill().fillna(0)

    fc = m.predict(future)

    # ---- Save model & predictions ----
    joblib.dump(m, os.path.join(MODEL_DIR, f"prophet_{cat.lower().replace(' ','_')}.pkl"))
    fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
        os.path.join(PRED_DIR, f"prophet_forecast_{cat.lower().replace(' ','_')}.csv"),
        index=False
    )

    # ---- RMSE ----
    val_fc = fc[fc['ds'].isin(val['ds'])]
    if len(val_fc) == len(val):
        rmse = safe_rmse(val["y"].values, val_fc["yhat"].values)
    else:
        rmse = None

    print(f"📈 Prophet trained → {cat} | RMSE = {rmse}")
    return rmse



# -------------------------------------------------------
# TRAIN XGBOOST
# -------------------------------------------------------
def train_xgb(cat):
    fn = os.path.join(FEATURE_DIR, f"feat_{cat.lower().replace(' ','_')}.csv")
    if not os.path.exists(fn):
        print(f"❌ Feature file missing for {cat}")
        return None

    df = pd.read_csv(fn, parse_dates=["Date"])
    df = df.sort_values("Date")

    if len(df) < 14:
        print(f"⚠️ XGB skipped for {cat}: Not enough rows ({len(df)})")
        return None

    # Remove non-numeric columns except Date
    feat_cols = [c for c in df.columns if c not in ("Date", "UnitsSold")]
    X = df[feat_cols].select_dtypes(include=[np.number]).fillna(0)
    y = df["UnitsSold"].values

    split = len(df) - 30
    X_train, y_train = X.iloc[:split], y[:split]
    X_test, y_test = X.iloc[split:], y[split:]

    model = xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )

    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
    except Exception as e:
        print(f"❌ XGB failed for {cat}: {e}")
        return None

    joblib.dump(model, os.path.join(MODEL_DIR, f"xgb_{cat.lower().replace(' ','_')}.joblib"))

    preds = model.predict(X_test)
    rmse = safe_rmse(y_test, preds)

    # Save predictions
    df_test = df.iloc[split:].copy()
    df_test["y_pred"] = preds
    df_test.to_csv(os.path.join(PRED_DIR, f"xgb_preds_{cat.lower().replace(' ','_')}.csv"), index=False)

    print(f"⚡ XGBoost trained → {cat} | RMSE = {rmse}")
    return rmse



# -------------------------------------------------------
# TRAIN RANDOM FOREST (Thesis Addition)
# -------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor

def train_rf(cat):
    fn = os.path.join(FEATURE_DIR, f"feat_{cat.lower().replace(' ','_')}.csv")
    if not os.path.exists(fn):
        return None

    df = pd.read_csv(fn, parse_dates=["Date"])
    df = df.sort_values("Date")

    if len(df) < 14:
        return None

    feat_cols = [c for c in df.columns if c not in ("Date", "UnitsSold", "y_pred")]
    X = df[feat_cols].select_dtypes(include=[np.number]).fillna(0)
    y = df["UnitsSold"].values

    split = len(df) - 30
    X_train, y_train = X.iloc[:split], y[:split]
    X_test, y_test = X.iloc[split:], y[split:]

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, os.path.join(MODEL_DIR, f"rf_{cat.lower().replace(' ','_')}.joblib"))

    preds = model.predict(X_test)
    rmse = safe_rmse(y_test, preds)

    # Save predictions
    df_test = df.iloc[split:].copy()
    df_test["y_pred"] = preds
    df_test.to_csv(os.path.join(PRED_DIR, f"rf_preds_{cat.lower().replace(' ','_')}.csv"), index=False)

    print(f"🌲 Random Forest trained → {cat} | RMSE = {rmse}")
    return rmse


# -------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(FEATURE_DIR):
        raise FileNotFoundError("Feature directory missing: " + FEATURE_DIR)

    fns = [f for f in os.listdir(FEATURE_DIR) if f.startswith("feat_")]
    cats = sorted({f[len("feat_"):-4].replace("_", " ").title() for f in fns})

    print("\n🧵 Categories detected:\n", cats, "\n")

    for cat in cats:
        print(f"==============================")
        print(f"🚀 Training models for: {cat}")
        print(f"==============================")

        train_prophet(cat)
        train_xgb(cat)
        train_rf(cat)

    print("\n🎉 Training finished.\n")
