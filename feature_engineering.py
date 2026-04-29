"""
Bus Overcrowding Prediction System
Stage 2 — Feature Engineering & Preprocessing
Reva University Mini Project | ISE Dept.

Reads bus_occupancy_data.csv and outputs train/test splits
ready for ML model training.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import os

INPUT_FILE  = "bus_occupancy_data.csv"
OUTPUT_DIR  = "model_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    return df

def engineer_features(df):
    print("\nEngineering features...")

    drop_cols = [
        "trip_id", "date", "route_name", "day_name",
        "passengers_in", "passengers_out",
        "occupancy_count", "occupancy_pct", "bus_capacity"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    le_route = LabelEncoder()
    le_stop  = LabelEncoder()
    le_label = LabelEncoder()

    df["route_id_enc"]  = le_route.fit_transform(df["route_id"])
    df["stop_name_enc"] = le_stop.fit_transform(df["stop_name"])
    df["label_enc"]     = le_label.fit_transform(df["occupancy_label"])

    pickle.dump(le_route, open(f"{OUTPUT_DIR}/le_route.pkl", "wb"))
    pickle.dump(le_stop,  open(f"{OUTPUT_DIR}/le_stop.pkl",  "wb"))
    pickle.dump(le_label, open(f"{OUTPUT_DIR}/le_label.pkl", "wb"))
    pickle.dump(list(le_route.classes_), open(f"{OUTPUT_DIR}/routes.pkl", "wb"))
    pickle.dump(list(le_stop.classes_),  open(f"{OUTPUT_DIR}/stops.pkl",  "wb"))

    print(f"  Routes encoded: {list(le_route.classes_)}")
    print(f"  Labels encoded: {dict(zip(le_label.classes_, le_label.transform(le_label.classes_)))}")

    FEATURES = [
        "route_id_enc", "stop_name_enc", "stop_index", "total_stops",
        "hour", "day_of_week", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "stop_popularity"
    ]
    TARGET = "label_enc"

    X = df[FEATURES]
    y = df[TARGET]

    scaler = MinMaxScaler()
    scale_cols = ["hour", "day_of_week", "stop_index", "total_stops", "stop_popularity"]
    X = X.copy()
    X[scale_cols] = scaler.fit_transform(X[scale_cols])
    pickle.dump(scaler,   open(f"{OUTPUT_DIR}/scaler.pkl",   "wb"))
    pickle.dump(FEATURES, open(f"{OUTPUT_DIR}/features.pkl", "wb"))

    print(f"  Features ready: {FEATURES}")
    return X, y

def split_and_save(X, y):
    print("\nSplitting dataset (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pickle.dump((X_train, X_test, y_train, y_test),
                open(f"{OUTPUT_DIR}/train_test_split.pkl", "wb"))

    print(f"  Training samples : {len(X_train):,}")
    print(f"  Testing samples  : {len(X_test):,}")

    print("\nClass distribution in training set:")
    labels = ["Low", "Medium", "High"]
    for i, label in enumerate(labels):
        count = (y_train == i).sum()
        pct = count / len(y_train) * 100
        print(f"  {label:<8} {count:>6,} ({pct:.1f}%)")

    return X_train, X_test, y_train, y_test

def main():
    print("=" * 50)
    print("  Feature Engineering & Preprocessing")
    print("=" * 50)

    df = load_data(INPUT_FILE)
    X, y = engineer_features(df)
    X_train, X_test, y_train, y_test = split_and_save(X, y)

    print(f"\nAll artifacts saved to: {OUTPUT_DIR}/")
    print("  le_route.pkl, le_stop.pkl, le_label.pkl")
    print("  scaler.pkl, features.pkl, routes.pkl, stops.pkl")
    print("  train_test_split.pkl")
    print("\nNext step: Run model_training.py")

if __name__ == "__main__":
    main()