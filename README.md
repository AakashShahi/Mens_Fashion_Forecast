🌆 Kathmandu Youth Fashion Forecaster
AI-Powered Trend Prediction, Psychology Modeling & Inventory Optimization for Nepali Fashion Retailers

This project is a research-based academic AI system designed to predict clothing trends among urban Nepali male youth (17–25) using:

✔ Historical sales
✔ Google Trends
✔ Social media signals
✔ Fashion psychology
✔ Social Identity Theory
✔ Machine learning forecasting (Prophet + XGBoost Hybrid)
✔ Inventory optimization algorithms

It includes a full Streamlit dashboard, an ML training pipeline, and supports live data ingestion from social media and web sources.



Project Structure
Mens_Fashion_Forecast/
│
├── data/
│   ├── raw CSV files
│   ├── cohort datasets
│   ├── google trends
│   ├── social media datasets
│   └── features/
│        ├── feat_*.csv
│        ├── trend_scores.csv
│        ├── trend_scores_hybrid.csv
│        ├── psychology_scores.csv
│        └── inventory_recommendations.csv
│
├── models/
│   ├── prophet_*.pkl
│   ├── xgb_*.joblib
│   └── predictions/
│        ├── prophet_forecast_*.csv
│        └── xgb_preds_*.csv
│
├── src/
│   ├── 01_load_clean.py
│   ├── 02_feature_engineer.py
│   ├── 03_train_models.py
│   ├── 04_evaluate.py
│   ├── 05_inventory_opt.py
│   ├── psychology/
│   ├── trends/
│   └── inventory/
│
├── app/
│   ├── app.py
│   └── pages/
│        ├── 1_trends.py
│        ├── 2_inventory.py
│        └── 3_psychology.py
│
├── notebooks/
│── requirements.txt
└── README.md


Key Features
1️⃣ Trend Analyzer (Hybrid ML + Behavioral Signals)
Social Media Trend Score
Google Search Interest Score
Sales Momentum
Weighted TrendScore ranking
Supports Hybrid Forecast:
Prophet Forecast × Social × Google × Psychology × Sales Velocity

2️⃣ Fashion Psychology Engine
Uses:
Conformity bias
Aspirational bias
Identity signals (persona alignment)
Cultural fit
Availability heuristics
Auto-learns weights OR uses fixed scoring

Outputs:
PsychologyScore.csv per clothing category.

3️⃣ Inventory Optimization Brain
Combines:
Hybrid forecast
Trend surge
Sales velocity
Psychology demand
Price tiers (Low / Mid / High)
Risk model
Persona recommendation

Outputs:
RecommendedOrderQty + RiskLevel + PersonaTarget

4️⃣ Streamlit Dashboard
Includes:
Prophet forecast viewer
Trend Analyzer Dashboard
Inventory Recommendation Dashboard
Psychology Engine Dashboard
Hashtag frequency charts
Audio suggestions (text-to-speech)




Venv:
# from project root
python -m venv venv
# PowerShell
.\venv\Scripts\Activate.ps1
# or CMD
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

#Install packages
pip install -r requirements.txt


Quick run (after activating venv):
pip install -r requirements.txt
python src/01_load_clean.py
python src/02_feature_engineer.py
python src/03_train_models.py
python src/04_evaluate.py
python src/05_inventory_opt.py
python src/psychology/psychology_engine.py
python src/trends/trend_analyzer.py
python src/inventory/inventory_advisor.py
streamlit run app/app.py


Academic Relevance
This project is aligned with:
Fashion psychology
Social Identity Theory
Behavioral economics
Predictive modeling
Retail inventory optimization
Machine learning forecasting
Nepal-specific youth culture and social trends