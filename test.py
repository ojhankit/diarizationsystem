# from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate


# print(DiarizationErrorRate)
# print(JaccardErrorRate)


#!/usr/bin/env python3
"""
Evaluate diarization outputs using pyannote.metrics
Computes DER (Diarization Error Rate) and JER (Jaccard Error Rate).
"""

import os
import csv
import argparse
from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate


def load_rttm(file_path):
    """Load RTTM file into pyannote.core.Annotation."""
    ann = Annotation()
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 8 or parts[0].upper() != "SPEAKER":
                continue
            file_id = parts[1]
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            ann[Segment(start, start + duration)] = speaker
    return ann


def evaluate_diarization(truth_dir, model_dir, output_csv):
    der_metric = DiarizationErrorRate()
    jer_metric = JaccardErrorRate()

    results = []

    print(f"{'File':20s} {'DER':>8s} {'JER':>8s}")

    for file_name in sorted(os.listdir(truth_dir)):
        if not file_name.endswith(".rttm"):
            continue

        file_id = file_name.replace(".rttm", "")
        truth_path = os.path.join(truth_dir, file_name)

        # Prediction must exist
        model_path = os.path.join(model_dir, file_name)
        if not os.path.exists(model_path):
            # Try flexible matching: filename without "_diarization"
            alt_name = file_name.replace("_diarization", "")
            model_path = os.path.join(model_dir, alt_name)
            if not os.path.exists(model_path):
                print(f"⚠ Missing prediction for {file_name}, skipping.")
                continue

        truth_ann = load_rttm(truth_path)
        model_ann = load_rttm(model_path)

        # Compute DER and JER
        der = der_metric(truth_ann, model_ann)
        jer = jer_metric(truth_ann, model_ann)

        results.append([file_id, der, jer])
        print(f"{file_id:20s} {der*100:8.2f} {jer*100:8.2f}")

    # Print averages
    if results:
        avg_der = sum(r[1] for r in results) / len(results)
        avg_jer = sum(r[2] for r in results) / len(results)
        print("\nAVERAGE RESULTS")
        print(f"DER: {avg_der*100:.2f}%")
        print(f"JER: {avg_jer*100:.2f}%")

        # Save to CSV
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "DER", "JER"])
            writer.writerows(results)
            writer.writerow(["average", avg_der, avg_jer])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate diarization output using DER and JER (pyannote.metrics)"
    )
    parser.add_argument("--truth_dir", required=True, help="Directory with ground truth RTTM files")
    parser.add_argument("--model_dir", required=True, help="Directory with predicted RTTM files")
    parser.add_argument("--output_csv", required=True, help="Output CSV file path")
    args = parser.parse_args()

    evaluate_diarization(args.truth_dir, args.model_dir, args.output_csv)
