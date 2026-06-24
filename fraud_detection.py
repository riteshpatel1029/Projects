
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             recall_score, precision_score, f1_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")



def generate_synthetic_data(n_samples: int = 10_000, fraud_ratio: float = 0.02) -> pd.DataFrame:
    """Generate a synthetic financial transaction dataset."""
    np.random.seed(42)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    legit = pd.DataFrame({
        "amount":        np.random.exponential(scale=100, size=n_legit),
        "hour":          np.random.randint(0, 24, size=n_legit),
        "merchant_risk": np.random.uniform(0, 0.3, size=n_legit),
        "velocity":      np.random.poisson(lam=2, size=n_legit),
        "distance_km":   np.random.exponential(scale=10, size=n_legit),
        "is_fraud":      0,
    })
    fraud = pd.DataFrame({
        "amount":        np.random.exponential(scale=800, size=n_fraud),
        "hour":          np.random.choice([0, 1, 2, 3, 23], size=n_fraud),
        "merchant_risk": np.random.uniform(0.6, 1.0, size=n_fraud),
        "velocity":      np.random.poisson(lam=10, size=n_fraud),
        "distance_km":   np.random.exponential(scale=200, size=n_fraud),
        "is_fraud":      1,
    })
    return pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)


# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    features = ["amount", "hour", "merchant_risk", "velocity", "distance_km"]
    X = df[features].values
    y = df["is_fraud"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, features


# ─────────────────────────────────────────────
# 3. SMOTE Oversampling
# ─────────────────────────────────────────────
def apply_smote(X_train, y_train):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[SMOTE] Resampled class distribution: {np.bincount(y_res)}")
    return X_res, y_res


# ─────────────────────────────────────────────
# 4. Isolation Forest (unsupervised anomaly)
# ─────────────────────────────────────────────
def run_isolation_forest(X_train, X_test, y_test, contamination=0.02):
    print("\n── Isolation Forest ──")
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    iso.fit(X_train)
    preds = iso.predict(X_test)
    # IsolationForest: -1 = anomaly (fraud), 1 = normal
    preds_binary = np.where(preds == -1, 1, 0)
    print(classification_report(y_test, preds_binary, target_names=["Legit", "Fraud"]))
    return iso, preds_binary


# ─────────────────────────────────────────────
# 5. Logistic Regression Baseline
# ─────────────────────────────────────────────
def run_logistic_regression(X_train, y_train, X_test, y_test):
    print("\n── Logistic Regression Baseline ──")
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    print(classification_report(y_test, preds, target_names=["Legit", "Fraud"]))
    return lr, preds


# ─────────────────────────────────────────────
# 6. Threshold-Based Alert Logic
# ─────────────────────────────────────────────
def flag_suspicious(df: pd.DataFrame, scaler, model, features: list,
                    threshold: float = 0.5) -> pd.DataFrame:
    """Return rows whose fraud probability exceeds threshold."""
    X = scaler.transform(df[features].values)
    proba = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["fraud_score"] = proba
    df["alert"] = proba >= threshold
    return df[df["alert"]].sort_values("fraud_score", ascending=False)


# ─────────────────────────────────────────────
# 7. Visualisation
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_test, preds, title="Confusion Matrix"):
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"], ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"outputs/{title.replace(' ', '_').lower()}.png", dpi=150)
    plt.close()
    print(f"[Saved] {title}")


def plot_fraud_score_distribution(df_flagged):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df_flagged["fraud_score"], bins=30, color="#E74C3C", edgecolor="white")
    ax.set_title("Fraud Score Distribution (Flagged Transactions)")
    ax.set_xlabel("Fraud Probability")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("outputs/fraud_score_distribution.png", dpi=150)
    plt.close()
    print("[Saved] fraud_score_distribution.png")


# ─────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────
def main():
    import os
    os.makedirs("outputs", exist_ok=True)

    print("=== Fraud Detection System ===\n")
    df = generate_synthetic_data()
    print(f"Dataset shape: {df.shape}  |  Fraud rate: {df['is_fraud'].mean():.2%}")

    X, y, scaler, features = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Baseline LR (no SMOTE)
    lr_base, lr_preds_base = run_logistic_regression(X_train, y_train, X_test, y_test)
    recall_base = recall_score(y_test, lr_preds_base)

    # LR with SMOTE
    X_res, y_res = apply_smote(X_train, y_train)
    lr_smote, lr_preds_smote = run_logistic_regression(X_res, y_res, X_test, y_test)
    recall_smote = recall_score(y_test, lr_preds_smote)
    print(f"\nRecall improvement (SMOTE): {recall_base:.2%} → {recall_smote:.2%}")

    
    iso, iso_preds = run_isolation_forest(X_train, X_test, y_test)

    
    plot_confusion_matrix(y_test, lr_preds_smote, "LR SMOTE Confusion Matrix")
    plot_confusion_matrix(y_test, iso_preds, "Isolation Forest Confusion Matrix")

    
    print("\n── Alert Demo (top suspicious transactions) ──")
    flagged = flag_suspicious(df.iloc[-200:].reset_index(drop=True),
                              scaler, lr_smote, features, threshold=0.5)
    print(flagged[["amount", "merchant_risk", "velocity", "fraud_score"]].head(10).to_string())
    if not flagged.empty:
        plot_fraud_score_distribution(flagged)

    print("\n✅ Done. Check the 'outputs/' folder for plots.")


if __name__ == "__main__":
    main()
