# data_loader.py
import os
import glob
import numpy as np
import pandas as pd


def load_features_from_folder(feature_folder, df_labels, id_column='ID'):
    """Load pre-extracted numpy features mapped by sequence IDs."""
    npy_files = glob.glob(os.path.join(feature_folder, "*.npy"))
    features = {}
    valid_ids = []
    target_ids = set(df_labels[id_column].astype(str).values)

    for npy_file in npy_files:
        seq_id = os.path.splitext(os.path.basename(npy_file))[0]
        if seq_id in target_ids:
            try:
                feat = np.load(npy_file)
                features[seq_id] = feat.flatten() if len(feat.shape) > 1 else feat
                valid_ids.append(seq_id)
            except Exception as e:
                # Silently skip corrupted or unreadable files
                pass

    return features, valid_ids


def load_data(feature_folder, excel_file, sequence_file, id_column):
    """Aggregate labels, sequences, and features."""
    df_labels = pd.read_excel(excel_file)
    df_labels[id_column] = df_labels[id_column].astype(str)

    df_sequences = pd.read_csv(sequence_file)
    df_sequences[id_column] = df_sequences[id_column].astype(str)

    features, valid_ids = load_features_from_folder(feature_folder, df_labels, id_column)
    return features, df_labels, df_sequences, valid_ids


def one_hot_encode_sequences(sequences, length):
    """Convert DNA/RNA sequence strings to one-hot encoded numpy arrays."""
    data_n = len(sequences)
    one_hot = np.zeros((data_n, length, 4), dtype=np.int8)
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3, 'u': 3}

    for l, seq in enumerate(sequences):
        seq = str(seq)
        for i, base in enumerate(seq):
            if i >= length:
                break
            if base in mapping:
                one_hot[l, i, mapping[base]] = 1

    return one_hot


def prepare_all_data(features, df_labels, df_sequences, valid_ids, id_column, label_column, sequence_column):
    """Align features and labels, returning final training arrays."""
    df_valid = df_labels[df_labels[id_column].isin(valid_ids)]
    X_rna_fm, sequences, y = [], [], []

    df_labels_idx = df_valid.set_index(id_column)
    df_seqs_idx = df_sequences.set_index(id_column)

    for seq_id in valid_ids:
        try:
            X_rna_fm.append(features[seq_id])
            y.append(df_labels_idx.loc[seq_id, label_column])
            sequences.append(df_seqs_idx.loc[seq_id, sequence_column])
        except KeyError:
            continue

    X_rna_fm = np.array(X_rna_fm)
    y = np.array(y)
    X_onehot = one_hot_encode_sequences(sequences, len(sequences[0])) if sequences else None

    return X_rna_fm, X_onehot, y