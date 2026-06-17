import os
import re
import argparse
import numpy as np
import torch
from Bio import SeqIO
import fm


def load_rna_fm_model(model_path):
    """Loads the RNA-FM model by temporarily patching torch.load to bypass weights_only restrictions."""
    original_load = fm.pretrained.load_model_and_alphabet_local

    def patched_load(model_location, theme="protein"):
        from pathlib import Path
        loc = Path(model_location)
        model_data = torch.load(str(loc), map_location="cpu", weights_only=False)
        model_name = loc.stem

        regression_data = None
        if fm.pretrained._has_regression_weights(model_name):
            reg_loc = str(loc.with_suffix("")) + "-contact-regression.pt"
            regression_data = torch.load(reg_loc, map_location="cpu", weights_only=False)

        return fm.pretrained.load_model_and_alphabet_core(model_name, model_data, regression_data, theme)

    fm.pretrained.load_model_and_alphabet_local = patched_load
    try:
        model, alphabet = fm.pretrained.rna_fm_t12(model_location=model_path)
        return model, alphabet
    finally:
        fm.pretrained.load_model_and_alphabet_local = original_load


def sanitize_filename(filename):
    """Cleans sequence IDs to create safe, cross-platform filenames."""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename).strip('. ')
    return sanitized if sanitized else "unknown_sequence"


def extract_features(fasta_file, model_path, output_dir="features", batch_size=4):
    """Parses FASTA input and extracts sequence embeddings using the RNA-FM language model."""
    os.makedirs(output_dir, exist_ok=True)

    # Parse FASTA and normalize sequences
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        clean_seq = str(record.seq).upper().replace('T', 'U')
        sequences.append((record.id, clean_seq))

    print(f"Loaded {len(sequences)} sequences from {fasta_file}")

    # Initialize model environment
    model, alphabet = load_rna_fm_model(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Running inference on: {device}")

    batch_converter = alphabet.get_batch_converter()

    # Inference loop
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]

        try:
            labels, strs, tokens = batch_converter(batch)
            tokens = tokens.to(device)

            with torch.no_grad():
                results = model(tokens, repr_layers=[12])
                embeddings = results["representations"][12]

            # Average pooling masked by non-padding tokens
            mask = (tokens != alphabet.padding_idx)
            seq_embeddings = (embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)
            seq_embeddings = seq_embeddings.cpu().numpy()

            # Save individual representations
            for j, (seq_id, _) in enumerate(batch):
                out_path = os.path.join(output_dir, f"{sanitize_filename(seq_id)}.npy")
                np.save(out_path, seq_embeddings[j])

        except Exception as e:
            print(f"Skipping batch starting at index {i} due to error: {e}")
            continue

    print(f"Feature extraction completed. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract RNA-FM sequence representations.")

    # Command-line interfaces
    parser.add_argument("--fasta", type=str, required=True,
                        help="Path to the input FASTA file.")
    parser.add_argument("--model", type=str, default="pretrained_models/RNA-FM_pretrained.pth",
                        help="Path to the downloaded RNA-FM pretrained weights.")
    parser.add_argument("--outdir", type=str, default="output_features",
                        help="Output directory to save the extracted .npy files.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for model inference.")

    args = parser.parse_args()

    print(f"--- RNA-FM Feature Extraction ---")
    print(f"Input FASTA : {args.fasta}")
    print(f"Model Weights: {args.model}")
    print(f"Output Dir  : {args.outdir}")
    print(f"---------------------------------")

    extract_features(
        fasta_file=args.fasta,
        model_path=args.model,
        output_dir=args.outdir,
        batch_size=args.batch_size
    )