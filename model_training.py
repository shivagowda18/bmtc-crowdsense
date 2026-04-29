"""
Bus Overcrowding Prediction System
Stage 3 — ML Model Training & Evaluation
Reva University Mini Project | ISE Dept.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, ConfusionMatrixDisplay
)
import os

MODEL_DIR  = "model_artifacts"
LABELS = ["Low", "Medium", "High"]

def load_splits():
    print("Loading train/test splits...")
    X_train, X_test, y_train, y_test = pickle.load(
        open(f"{MODEL_DIR}/train_test_split.pkl", "rb")
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("\nTraining Random Forest classifier...")
    print("  This will take 1-2 minutes, please wait...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("  Training complete!")
    return model

def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model on test set...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Overall Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LABELS))
    return y_pred, acc

def save_confusion_matrix(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(7, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Bus Occupancy Classifier", fontsize=13, pad=14)
    plt.tight_layout()
    path = f"{MODEL_DIR}/confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {path}")

def save_feature_importance(model):
    features = pickle.load(open(f"{MODEL_DIR}/features.pkl", "rb"))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1d9e75" if i == 0 else "#9FE1CB" for i in range(len(features))]
    ax.bar(range(len(features)),
           importances[indices],
           color=colors, edgecolor="none")
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([features[i] for i in indices], rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Importance Score")
    ax.set_title("Feature Importance — Random Forest", fontsize=13, pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = f"{MODEL_DIR}/feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Feature importance saved: {path}")
    print("\nTop 5 most important features:")
    for i in range(5):
        print(f"  {features[indices[i]]:<20} {importances[indices[i]]:.4f}")

def save_model(model):
    path = f"{MODEL_DIR}/bus_model.pkl"
    pickle.dump(model, open(path, "wb"))
    print(f"\nModel saved: {path}")

def main():
    print("=" * 50)
    print("  ML Model Training & Evaluation")
    print("=" * 50)
    X_train, X_test, y_train, y_test = load_splits()
    model = train_model(X_train, y_train)
    y_pred, acc = evaluate_model(model, X_test, y_test)
    save_confusion_matrix(y_test, y_pred)
    save_feature_importance(model)
    save_model(model)
    print("\n" + "=" * 50)
    print(f"  Final Accuracy: {acc * 100:.2f}%")
    print("  All artifacts saved to model_artifacts/")
    print("  Next step: Run app.py")
    print("=" * 50)

if __name__ == "__main__":
    main()