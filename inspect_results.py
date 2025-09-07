#!/usr/bin/env python3
"""
Inspect saved results_summary.joblib to surface suspicious metrics (very high accuracy,
zero/near-zero training time), and optionally recompute accuracy if y_test/y_pred are present.
Run with the same Python interpreter used for Streamlit / installing packages.
"""
import os
import pprint
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

ROOT = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(ROOT, "saved_models", "results_summary.joblib")


def pretty(v):
    try:
        return float(v)
    except Exception:
        return v


def main():
    print("Inspecting:", SUMMARY_PATH)
    if not os.path.exists(SUMMARY_PATH):
        print("MISSING:", SUMMARY_PATH)
        return
    data = joblib.load(SUMMARY_PATH)
    print("\nTop-level keys:")
    pprint.pprint(list(data.keys()))
    print("\nPer-model sanity checks:")
    any_y = False
    for name, entry in (data.items() if isinstance(data, dict) else []):
        if not isinstance(entry, dict):
            continue
        acc = entry.get("test_accuracy") or entry.get("accuracy") or entry.get("acc") or None
        f1 = entry.get("f1_score") or entry.get("f1") or None
        train_t = entry.get("training_time") or entry.get("train_time") or entry.get("fit_time") or None
        print(f"\nModel: {name}")
        print("  accuracy:", pretty(acc))
        print("  f1_score:", pretty(f1))
        print("  training_time (s):", pretty(train_t))
        if acc is not None:
            try:
                accf = float(acc)
                if accf >= 0.99:
                    print("  ⚠️ HIGH ACCURACY (>=99%) — possible data leakage or evaluation-on-train")
            except Exception:
                pass
        if train_t is None or float(train_t) == 0.0:
            print("  ⚠️ training_time is zero or missing — check trainer timing code")
        # recompute accuracy if possible
        if "y_test" in entry and "y_pred" in entry:
            any_y = True
            yt = np.asarray(entry["y_test"])
            yp = np.asarray(entry["y_pred"])
            try:
                acc_re = accuracy_score(yt, yp)
                print(f"  recomputed accuracy from stored y_test/y_pred: {acc_re:.6f}")
                print("  classification_report:")
                print(classification_report(yt, yp, zero_division=0))
            except Exception as e:
                print("  error recomputing metrics:", e)
    if not any_y:
        print("\nNo y_test/y_pred pairs found in summary — cannot recompute metrics for sanity check.")
    print("\nDone.")


if __name__ == "__main__":
    main()
