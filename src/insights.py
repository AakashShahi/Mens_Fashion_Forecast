import pandas as pd
import numpy as np

def describe_metric(label, value, unit="", context="neutral"):
    """
    Generates a simple natural language description for a metric.
    
    Args:
        label (str): The name of the metric (e.g., "Sales", "Trend Score").
        value (float): The numeric value.
        unit (str): Unit string (e.g., "units", "%").
        context (str): 'positive' (higher is better), 'negative' (lower is better), or 'neutral'.
        
    Returns:
        str: A descriptive sentence.
    """
    try:
        val = float(value)
    except:
        return f"{label} is {value}."

    desc = ""
    
    if label == "Trend Score":
        if val > 2000: desc = "Extremely high"
        elif val > 1000: desc = "Strong"
        elif val > 500: desc = "Moderate"
        else: desc = "Low"
    elif label == "Risk":
        # Usually risk is handled by string, but if numeric:
        if val > 0.7: desc = "High"
        elif val > 0.3: desc = "Medium"
        else: desc = "Low"
    
    result = f"{label} is **{val:,.2f} {unit}**"
    if desc:
        result += f" ({desc})"
        
    return result

def prescriptive_action(risk_level, trend_score):
    """
    Returns a prescriptive action based on risk and trend.
    """
    risk = risk_level.lower()
    
    if "high" in risk:
        if trend_score > 1500:
            return "⚠️ **High Risk but High Reward:** Order cautiously in small batches. Monitor daily."
        else:
            return "⛔ **High Risk & Low Trend:** Do not reorder. Clear existing stock."
            
    if "moderate" in risk:
        if trend_score > 1000:
            return "✅ **Good Bet:** Increase stock levels by 20% to capture demand."
        else:
            return "⚖️ **Stable:** Maintain current inventory levels. No major changes needed."
            
    if "low" in risk:
        if trend_score > 800:
            return "🚀 **Safe & Trending:** Aggressively stock up. This is a winner."
        else:
            return "📉 **Low Risk but Low Demand:** Keep minimum stock. Focus on other items."
            
    return "Analyze further before ordering."

def describe_forecast_horizon(df, date_col="Date"):
    """
    Analyzes the date range of the dataframe and returns a coverage string.
    """
    if df is None or df.empty or date_col not in df.columns:
        return "No data available."
        
    dates = pd.to_datetime(df[date_col])
    start = dates.min().strftime('%Y-%b-%d')
    end = dates.max().strftime('%Y-%b-%d')
    
    duration = (dates.max() - dates.min()).days
    
    return f"Forecasting based on **{duration} days** of data, ranging from **{start}** to **{end}**."

def trend_direction_text(series):
    """
    Analyzes the last 7 days of a series to determine direction.
    """
    if len(series) < 7:
        return "Not enough data for trend."
        
    recent = series.iloc[-7:]
    slope = (recent.iloc[-1] - recent.iloc[0])
    
    if slope > 0:
        return f"📈 Trending **UP** over the last week (+{slope:.0f})."
    elif slope < 0:
        return f"📉 Trending **DOWN** over the last week ({slope:.0f})."
    else:
        return "➡️ Flat trend over the last week."
