# Kathmandu Youth Fashion Forecaster - Project

Folders:
- data/ : raw CSVs and generated feature CSVs
- src/  : pipeline scripts (cleaning, fe, training, eval, inventory)
- models/: saved trained models and predictions
- app/  : Streamlit app
- notebooks/ : exploratory notebooks
- venv/ : Python virtual environment

Quick run (after activating venv):
pip install -r requirements.txt
python src/01_load_clean.py
python src/02_feature_engineer.py
python src/03_train_models.py
python src/04_evaluate.py
python src/05_inventory_opt.py
streamlit run app/app.py
