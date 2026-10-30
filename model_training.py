"""
model_training.py
-----------------
Phishing URL Detector - Model Training and Evaluation Script.

Yeh script PhiUSIIL dataset ko load karti hai, duplicate URLs remove karti hai,
13 lexical features extract karti hai, 3 Machine Learning models train karti hai,
held-out test set par unhe evaluate karti hai, aur best model ko 'phishing_model.pkl'
ke roop me save karti hai.

Models:
1. Logistic Regression (Linear Baseline)
2. Decision Tree Classifier (Single Tree)
3. Random Forest Classifier (Ensemble of Trees)
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Custom feature extraction module
from feature_extraction import extract_features_df, FEATURE_NAMES


def load_and_preprocess_data(dataset_path: str):
    """
    Dataset load karta hai, sirf raw 'URL' aur 'label' columns select karta hai,
    missing values aur duplicate URLs ko clean karta hai, aur labels map karta hai.
    """
    print(f"[1/6] Loading dataset from: {dataset_path}...")
    start_time = time.time()
    
    # Sirf raw URL aur label read karenge (Zero data leakage from pre-computed HTML columns)
    df = pd.read_csv(
        dataset_path,
        usecols=["URL", "label"],
        encoding="utf-8-sig"
    )
    raw_count = len(df)
    print(f"      Total raw records: {raw_count:,}")
    
    # 1. Clean missing or invalid rows
    df = df.dropna(subset=["URL", "label"]).copy()
    df["URL"] = df["URL"].astype(str).str.strip()
    df = df[df["URL"] != ""]
    
    # 2. Deduplication check: Remove duplicate URLs to prevent train-test data leakage
    initial_len = len(df)
    df = df.drop_duplicates(subset=["URL"]).copy()
    dropped_dups = initial_len - len(df)
    print(f"      Deduplication: Removed {dropped_dups:,} duplicate URLs. Remaining unique: {len(df):,}")
    
    # 3. Label Mapping:
    # Original UCI Dataset: label 1 = Legitimate, label 0 = Phishing
    # Hamara ML Mapping:   1 = Phishing (Positive / Threat Class), 0 = Legitimate (Safe)
    # Isse predict_proba() ka output directly "Phishing Risk Probability" banta hai.
    print("[2/6] Mapping target labels: 1 -> Phishing, 0 -> Legitimate...")
    df["target"] = df["label"].apply(lambda x: 1 if int(x) == 0 else 0)
    
    counts = df["target"].value_counts()
    print(f"      Legitimate (0): {counts.get(0, 0):,} URLs")
    print(f"      Phishing   (1): {counts.get(1, 0):,} URLs")
    
    return df


def extract_dataset_features(df: pd.DataFrame):
    """
    Har ek URL par hamara custom feature extractor run karke numerical matrix banata hai.
    """
    print(f"[3/6] Extracting {len(FEATURE_NAMES)} lexical features for {len(df):,} unique URLs...")
    start_time = time.time()
    
    X = extract_features_df(df["URL"])
    y = df["target"].values
    
    elapsed = time.time() - start_time
    print(f"      Features extracted in {elapsed:.2f}s ({len(df)/elapsed:.1f} URLs/sec)")
    print(f"      Feature Matrix Shape: X={X.shape}, y={y.shape}")
    
    return X, y


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    3 models ko train karta hai aur unka evaluation metrics compare karta hai.
    """
    print("[4/6] Training & evaluating 3 Machine Learning models...")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=16, n_jobs=-1, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n      --- Training: {name} ---")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        # Training performance (for overfitting audit)
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        # Test performance (held-out test set)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        test_acc = accuracy_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred, zero_division=0)
        rec = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else 0.0
        cm = confusion_matrix(y_test, y_test_pred)
        
        results[name] = {
            "model": model,
            "train_time": train_time,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm
        }
        
        print(f"      Train Time     : {train_time:.2f}s")
        print(f"      Train Accuracy : {train_acc * 100:.2f}%")
        print(f"      Test Accuracy  : {test_acc * 100:.2f}%")
        print(f"      Precision      : {prec * 100:.2f}%")
        print(f"      Recall         : {rec * 100:.2f}%")
        print(f"      F1-Score       : {f1 * 100:.2f}%")
        print(f"      ROC-AUC Score  : {roc_auc:.4f}")
        print(f"      Confusion Matrix:\n        [TN={cm[0,0]:,}, FP={cm[0,1]:,}]\n        [FN={cm[1,0]:,}, TP={cm[1,1]:,}]")
    
    return results


def plot_feature_importance(rf_model, feature_names, output_file="feature_importance.png"):
    """
    Random Forest ki feature importances ka clean visualization graph save karta hai.
    """
    print(f"\n[5/6] Generating Feature Importance visualization -> {output_file}...")
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, hue=sorted_features, palette="viridis", legend=False)
    plt.title("URL Feature Importance (Random Forest)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.ylabel("URL Features", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()
    print(f"      Saved: {output_file}")


def save_best_model(results, output_path="phishing_model.pkl"):
    """
    Best performing model ko metadata aur thresholds ke saath joblib me save karta hai.
    """
    print(f"\n[6/6] Selecting & saving best model to '{output_path}'...")
    
    # Best model select karna based on F1-score
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_info = results[best_name]
    
    model_bundle = {
        "model": best_info["model"],
        "model_name": best_name,
        "feature_names": FEATURE_NAMES,
        "metrics": {
            "train_accuracy": float(best_info["train_accuracy"]),
            "accuracy": float(best_info["test_accuracy"]),
            "precision": float(best_info["precision"]),
            "recall": float(best_info["recall"]),
            "f1": float(best_info["f1"]),
            "roc_auc": float(best_info["roc_auc"])
        },
        "thresholds": {
            "safe_max": 0.35,      # Score < 35% -> Likely Safe
            "suspicious_max": 0.70 # Score 35% - 70% -> Suspicious, > 70% -> Phishing
        }
    }
    
    joblib.dump(model_bundle, output_path)
    print(f"      Selected Model: {best_name} (Test F1: {best_info['f1']*100:.2f}%)")
    print(f"      Model bundle saved successfully at: {output_path}")


def main():
    import sys
    if "--force" not in sys.argv:
        print("=" * 70)
        print("[NOTICE]: 'model_training.py' is the older baseline training script.")
        print("    The shipped production model ('phishing_model.pkl') was produced by")
        print("    'train_experimental_model.py' (path-balanced Random Forest).")
        print("    To run the authentic production training pipeline, execute:")
        print("        python train_experimental_model.py")
        print("    If you specifically want to run this older baseline script, use:")
        print("        python model_training.py --force")
        print("=" * 70)
        return

    print("==========================================================")
    print("        PHISHING URL DETECTOR - BASELINE TRAINING         ")
    print("==========================================================")
    
    dataset_file = os.path.join("data", "PhiUSIIL_Phishing_URL_Dataset.csv")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found at {dataset_file}! Please check Stage 2.")
    
    # 1. Load data and deduplicate
    df = load_and_preprocess_data(dataset_file)
    
    # 2. Extract features
    X, y = extract_dataset_features(df)
    
    # 3. Train-Test Split (80% Train, 20% Test, Stratified, random_state=42)
    print("\n      Performing Stratified Train-Test Split (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Training samples : {len(X_train):,}")
    print(f"      Testing samples  : {len(X_test):,}")
    
    # 4. Train and evaluate all 3 models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # 5. Feature Importance plot (using Random Forest)
    plot_feature_importance(results["Random Forest"]["model"], FEATURE_NAMES)
    
    # 6. Save best model
    save_best_model(results, "phishing_model.pkl")
    
    print("\n==========================================================")
    print("          TRAINING AND EVALUATION COMPLETE!               ")
    print("==========================================================")


if __name__ == "__main__":
    main()
