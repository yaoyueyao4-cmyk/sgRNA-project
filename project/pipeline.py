# pipeline.py
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef, confusion_matrix)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data_loader import load_data, prepare_all_data
from model import build_micro_tuned_model, CVTuner


def run_micro_tuning_pipeline(config):
    dataset_name = config['name']
    print(f"\n{'=' * 70}\n[Initiating Pipeline] {dataset_name}\n{'=' * 70}")

    # 1. Load and prepare data
    features, df_labels, df_seqs, valid_ids = load_data(
        config['feature_folder'], config['excel_file'],
        config['sequence_file'], config['id_column']
    )
    if not features:
        return None

    X_rna, X_seq, y = prepare_all_data(
        features, df_labels, df_seqs, valid_ids,
        config['id_column'], config['label_column'], config['sequence_column']
    )
    if X_seq is None:
        return None

    # 2. Train-Test Split (Hold out 20% for independent testing)
    X_rna_tr, X_rna_test, X_seq_tr, X_seq_test, y_tr, y_test = train_test_split(
        X_rna, X_seq, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3. Micro-Tuning via 3-Fold CV
    print(">>> Stage 1: Running CV Micro-Tuning...")
    tuner_dir = "tuner_logs"
    project_name = f"micro_opt_{dataset_name}"

    tuner = CVTuner(
        hypermodel=lambda hp: build_micro_tuned_model(hp, X_rna.shape[1], X_seq.shape[1:]),
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=10,
        directory=tuner_dir,
        project_name=project_name,
        overwrite=True
    )

    stop_early = EarlyStopping(monitor='val_auc', patience=5, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0)

    tuner.search(
        X=[X_rna_tr, X_seq_tr], y=y_tr,
        epochs=30, batch_size=64,
        callbacks=[stop_early, reduce_lr]
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f">>> Optimal Hyperparameters Found:")
    print(f"    - Dropout: {best_hps.get('dropout')}")
    print(f"    - L2 Reg: {best_hps.get('l2_reg')}")
    print(f"    - Learning Rate: {best_hps.get('learning_rate')}")

    # 4. Ensemble Training
    print("\n>>> Stage 2: Training Ensemble Models...")
    n_models = 5
    models = []

    X_rna_train, X_rna_val, X_seq_train, X_seq_val, y_train, y_val = train_test_split(
        X_rna_tr, X_seq_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=123
    )

    for i in range(n_models):
        print(f"   -- Training Sub-model {i + 1}/{n_models} --")
        model = tuner.hypermodel.build(best_hps)

        cb = [
            EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
        ]

        model.fit(
            [X_rna_train, X_seq_train], y_train,
            validation_data=([X_rna_val, X_seq_val], y_val),
            epochs=100, batch_size=64,
            callbacks=cb, verbose=0
        )
        models.append(model)

    # 5. Independent Blind Testing
    print("\n>>> Stage 3: Independent Ensemble Evaluation...")
    preds = [model.predict([X_rna_test, X_seq_test], verbose=0) for model in models]

    final_preds_proba = np.mean(preds, axis=0)
    final_preds_bin = (final_preds_proba > 0.5).astype(int).flatten()

    tn, fp, fn, tp = confusion_matrix(y_test, final_preds_bin).ravel()

    metrics = {
        'Dataset': dataset_name,
        'AUC': roc_auc_score(y_test, final_preds_proba),
        'Accuracy': accuracy_score(y_test, final_preds_bin),
        'Sensitivity (Sn)': recall_score(y_test, final_preds_bin, zero_division=0),
        'Specificity (Sp)': tn / (fp + tn) if (fp + tn) > 0 else 0,
        'Precision': precision_score(y_test, final_preds_bin, zero_division=0),
        'F1-Score': f1_score(y_test, final_preds_bin, zero_division=0),
        'MCC': matthews_corrcoef(y_test, final_preds_bin)
    }

    print(f"\n   [Results: {dataset_name}]")
    print(f"   ROC AUC  : {metrics['AUC']:.5f}")
    print(f"   Accuracy : {metrics['Accuracy']:.5f}")

    tf.keras.backend.clear_session()
    return metrics