# src/pipeline.py
import os
import sys
import subprocess

def run_step(script_name, description):
    print(f"\n{'='*60}")
    print(f"🚀 PART: {description}")
    print(f"   Executing: {script_name}...")
    print(f"{'='*60}\n")
    
    # Using subprocess to run the script in a separate process
    # This ensures clean state for each step, though imports would be faster
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Error in {script_name}. Pipeline stopped.")
        sys.exit(1)
    else:
        print(f"✅ {description} Completed Successfully.")

def main():
    print("\n🌟 STARTING MENS FASHION FORECAST DYNAMIC PIPELINE 🌟\n")
    
    # Define the steps of the pipeline
    steps = [
        ("src/01_load_clean.py", "Phase 1-3: Load, Profile & Clean Data"),
        ("src/02_feature_engineer.py", "Phase 3: Feature Engineering"),
        ("src/03_train_models.py", "Phase 5: Model Training (Prophet + XGBoost + RF)"),
        ("src/04_evaluate.py", "Phase 5: Evaluation"),
        ("src/psychology/psychology_engine.py", "Phase 5: Psychology Engine"),
        ("src/trends/trend_analyzer.py", "Phase 5: Trend Analysis"),
        ("src/clustering/kmeans_segmentation.py", "Phase 5: Inventory Segmentation (K-Means)"),
        ("src/05_inventory_opt.py", "Phase 7: Inventory Optimization"),
        ("src/06_generate_insights.py", "Phase 6: Insight Synthesis & Reporting")
    ]
    
    for script, desc in steps:
        run_step(script, desc)
        
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("Check 'INSIGHTS_REPORT.md' and the Dashboard for results.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
