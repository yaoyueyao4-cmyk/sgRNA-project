# main.py
import os
import traceback
import pandas as pd
import numpy as np
import tensorflow as tf

from config import DATASETS_CONFIG
from pipeline import run_micro_tuning_pipeline

# Ensure reproducibility and suppress TF logs
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    all_results = []

    for config in DATASETS_CONFIG:
        try:
            res = run_micro_tuning_pipeline(config)
            if res:
                all_results.append(res)
        except Exception as e:
            print(f"Failed processing dataset {config['name']}: {e}")
            traceback.print_exc()

    if all_results:
        print(f"\n{'=' * 70}\n[Complete] Saving Final Report...\n{'=' * 70}")
        df_summary = pd.DataFrame(all_results)
        df_summary.to_excel("Final_MicroTuned_Ensemble_Report.xlsx", index=False)
        print(df_summary[['Dataset', 'AUC', 'Accuracy']].to_string(index=False))