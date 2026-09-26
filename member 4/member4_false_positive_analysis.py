"""
Member 4 - False-Positive Screening
Team: Hack-vok
Dataset: Kepler Exoplanet Search Results

This script consumes Member 2's candidate prediction output and creates
a transparent false-positive screening table. It does NOT claim that
screened candidates are confirmed false positives. The result is a
prioritization for human review based on catalog flags and observable
candidate-level indicators.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT = "memeber2_candidate_predictions.csv"

df = pd.read_csv(INPUT)

fp_cols = ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]
df["catalog_fp_flag_count"] = df[fp_cols].fillna(0).sum(axis=1)

df["eb_indicator"] = (
    (df["koi_fpflag_ss"] == 1) |
    (df["koi_prad"] >= 10) |
    (df["koi_depth"] >= 10000)
).astype(int)

df["contamination_indicator"] = (
    (df["koi_fpflag_co"] == 1) |
    (df["koi_fpflag_ec"] == 1)
).astype(int)

df["artifact_indicator"] = (
    (df["koi_fpflag_nt"] == 1) |
    (df["outlier_flag"] == 1) |
    (df["koi_model_snr"] < 10)
).astype(int)

df["fp_screening_score"] = (
    df["catalog_fp_flag_count"] * 5
    + (df["koi_prad"] >= 10).astype(int) * 3
    + ((df["koi_prad"] >= 4) & (df["koi_prad"] < 10)).astype(int) * 2
    + (df["koi_depth"] >= 10000).astype(int) * 3
    + ((df["koi_depth"] >= 3000) & (df["koi_depth"] < 10000)).astype(int) * 2
    + df["outlier_flag"].fillna(0).astype(int)
    + (df["koi_model_snr"] < 10).astype(int)
    + ((df["koi_period"] < 2) & (df["koi_prad"] >= 4)).astype(int)
)

df["fp_risk"] = pd.cut(
    df["fp_screening_score"],
    bins=[-1, 2, 4, np.inf],
    labels=["Low", "Medium", "High"]
)

df.to_csv("member4_false_positive_screening.csv", index=False)
print(df["fp_risk"].value_counts())
print(df.nlargest(20, "fp_screening_score")[
    ["kepoi_name", "koi_prad", "koi_depth", "koi_period",
     "fp_screening_score", "fp_risk"]
])
