# src/05_inventory_opt.py
import joblib, os, pandas as pd, numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
FEATURE_DIR = os.path.join("data","features")

def compute_reorder(cat, horizon=30, lead_time=14, service_z=1.65):
    mpath = os.path.join(MODEL_DIR, f"prophet_{cat.lower().replace(' ','_')}.pkl")
    feat_fn = os.path.join(FEATURE_DIR, f"feat_{cat.lower().replace(' ','_')}.csv")
    if not os.path.exists(mpath):
        raise FileNotFoundError("Model not found: "+mpath)
    model = joblib.load(mpath)
    df = pd.read_csv(feat_fn, parse_dates=["Date"]).rename(columns={"Date":"ds","UnitsSold":"y"})
    last_trend = df["InterestScore"].iloc[-1] if "InterestScore" in df.columns else 0
    future = model.make_future_dataframe(periods=horizon)
    future = future.merge(df[["ds","InterestScore"]], on="ds", how="left")
    future["InterestScore"] = future["InterestScore"].fillna(last_trend)
    fc = model.predict(future).tail(horizon)
    mean_daily = fc["yhat"].mean()
    std_daily = fc["yhat"].std()
    demand_during_lead = mean_daily * lead_time
    safety_stock = service_z * std_daily * (lead_time**0.5)
    reorder = int(max(0, round(demand_during_lead + safety_stock)))
    return {
        "category": cat,
        "mean_daily": float(mean_daily),
        "std_daily": float(std_daily),
        "lead_time": int(lead_time),
        "safety_stock": int(round(safety_stock)),
        "reorder_qty": reorder
    }

if __name__=="__main__":
    files = [f for f in os.listdir(FEATURE_DIR) if f.startswith("feat_")]
    cats = [f[len("feat_"):-4].replace("_"," ").title() for f in files]
    for c in cats:
        try:
            print(c, compute_reorder(c))
        except Exception as e:
            print("Error for",c, e)
