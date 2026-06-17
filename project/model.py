# model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPool1D, AveragePooling1D,
                                     Concatenate, Dropout, Flatten, Dense,
                                     BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import keras_tuner as kt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def build_micro_tuned_model(hp, rna_fm_dim, onehot_shape):
    """
    Fixed architecture based on best baseline.
    Only tunes regularization and learning rate to prevent overfitting.
    """
    # Hyperparameter search space
    l2_rate = hp.Choice('l2_reg', [5e-4, 1e-3, 2e-3])
    drop_rate = hp.Choice('dropout', [0.4, 0.5, 0.6])
    lr = hp.Choice('learning_rate', [5e-4, 8e-4, 1e-3])

    l2_reg = regularizers.l2(l2_rate)

    # Channel A: Sequence Features (Fixed CNN Architecture)
    onehot_input = Input(shape=onehot_shape, name='onehot_input')
    x_emb = Conv1D(filters=16, kernel_size=1, use_bias=False, name='biochem_embedding')(onehot_input)

    conv1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2_reg)(x_emb)
    conv2 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2_reg)(x_emb)
    conv3 = Conv1D(filters=32, kernel_size=7, padding='same', activation='relu', kernel_regularizer=l2_reg)(x_emb)

    concatenated_conv = Concatenate()([conv1, conv2, conv3])
    max_pool = MaxPool1D(pool_size=2, strides=2, padding='same')(concatenated_conv)
    avg_pool = AveragePooling1D(pool_size=2, strides=2, padding='same')(concatenated_conv)

    pooled = Concatenate()([max_pool, avg_pool])
    cnn_flat = Flatten()(pooled)

    # Channel B: RNA-FM Features
    rna_fm_input = Input(shape=(rna_fm_dim,), name='rna_fm_input')
    rna_feat = Dense(64, activation='relu', kernel_regularizer=l2_reg)(rna_fm_input)
    rna_feat = BatchNormalization()(rna_feat)
    rna_feat = Dropout(drop_rate)(rna_feat)

    # Feature Fusion & Classification Head
    merged = Concatenate()([cnn_flat, rna_feat])

    x = Dense(128, activation='relu', kernel_regularizer=l2_reg)(merged)
    x = BatchNormalization()(x)
    x = Dropout(drop_rate)(x)

    x = Dense(64, activation='relu', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = Dropout(drop_rate)(x)

    output = Dense(1, activation='sigmoid', name="output")(x)

    model = Model(inputs=[rna_fm_input, onehot_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model


class CVTuner(kt.RandomSearch):
    """
    Custom Keras Tuner utilizing 3-fold cross-validation.
    Ensures hyperparameters are robust and prevents validation set memorization.
    """

    def run_trial(self, trial, X, y, **fit_kwargs):
        X_rna, X_seq = X
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        val_aucs = []

        callbacks = fit_kwargs.get('callbacks', [])
        epochs = fit_kwargs.get('epochs', 30)
        batch_size = fit_kwargs.get('batch_size', 64)

        for train_idx, val_idx in cv.split(X_rna, y):
            X_tr_rna, X_val_rna = X_rna[train_idx], X_rna[val_idx]
            X_tr_seq, X_val_seq = X_seq[train_idx], X_seq[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = self.hypermodel.build(trial.hyperparameters)

            model.fit(
                [X_tr_rna, X_tr_seq], y_tr,
                validation_data=([X_val_rna, X_val_seq], y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )

            val_pred = model.predict([X_val_rna, X_val_seq], verbose=0)
            val_aucs.append(roc_auc_score(y_val, val_pred))

            tf.keras.backend.clear_session()

        avg_auc = np.mean(val_aucs)
        return {'val_auc': avg_auc}