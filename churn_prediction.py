import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


def generate_telecom_data(n=5000, seed=42) -> pd.DataFrame:
    np.random.seed(seed)
    churn_prob = 0.27
    n_churn = int(n * churn_prob)
    n_stay  = n - n_churn

    def make_group(size, churned):
        return pd.DataFrame({
            "tenure":           np.random.randint(1, 72, size) if not churned
                                else np.random.randint(1, 20, size),
            "monthly_charges":  np.random.uniform(20, 120, size) if not churned
                                else np.random.uniform(60, 120, size),
            "total_charges":    np.random.uniform(100, 8000, size),
            "num_services":     np.random.randint(1, 9, size) if not churned
                                else np.random.randint(1, 4, size),
            "contract_type":    np.random.choice(["Month-to-month", "One year", "Two year"],
                                                  size,
                                                  p=[0.5, 0.3, 0.2] if churned else [0.2, 0.4, 0.4]),
            "payment_method":   np.random.choice(
                                    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
                                    size),
            "internet_service": np.random.choice(["DSL", "Fiber optic", "No"], size),
            "tech_support":     np.random.choice(["Yes", "No"], size,
                                                  p=[0.3, 0.7] if churned else [0.6, 0.4]),
            "senior_citizen":   np.random.choice([0, 1], size, p=[0.7, 0.3]),
            "dependents":       np.random.choice(["Yes", "No"], size),
            "churn":            int(churned),
        })

    df = pd.concat([make_group(n_stay, False), make_group(n_churn, True)], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def run_eda(df: pd.DataFrame, save_dir="outputs"):
    import os; os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Customer Churn EDA", fontsize=16, fontweight="bold")


    churn_counts = df["churn"].value_counts()
    axes[0, 0].pie(churn_counts, labels=["Stay", "Churn"], autopct="%1.1f%%",
                   colors=["#2ECC71", "#E74C3C"])
    axes[0, 0].set_title("Churn Distribution")


    df.groupby("churn")["tenure"].hist(ax=axes[0, 1], bins=20, alpha=0.7,
                                        color=["#2ECC71", "#E74C3C"])
    axes[0, 1].set_title("Tenure Distribution by Churn")
    axes[0, 1].set_xlabel("Tenure (months)")


    for label, color in zip([0, 1], ["#2ECC71", "#E74C3C"]):
        axes[0, 2].hist(df[df["churn"] == label]["monthly_charges"],
                        bins=20, alpha=0.7, label=f"Churn={label}", color=color)
    axes[0, 2].set_title("Monthly Charges by Churn")
    axes[0, 2].legend()


    ct = df.groupby(["contract_type", "churn"]).size().unstack()
    ct.plot(kind="bar", ax=axes[1, 0], color=["#2ECC71", "#E74C3C"], edgecolor="white")
    axes[1, 0].set_title("Contract Type vs Churn")
    axes[1, 0].tick_params(axis="x", rotation=15)


    axes[1, 1].boxplot(
        [df[df["churn"] == 0]["num_services"], df[df["churn"] == 1]["num_services"]],
        labels=["Stay", "Churn"])
    axes[1, 1].set_title("Number of Services vs Churn")


    num_cols = ["tenure", "monthly_charges", "total_charges", "num_services", "churn"]
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", ax=axes[1, 2])
    axes[1, 2].set_title("Correlation Heatmap")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/eda_dashboard.png", dpi=150)
    plt.close()
    print("[Saved] eda_dashboard.png")



def preprocess(df: pd.DataFrame):
    df = df.copy()


    cat_cols = ["contract_type", "payment_method", "internet_service",
                "tech_support", "dependents"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])


    df["charges_per_month"] = df["total_charges"] / (df["tenure"] + 1)
    df["service_density"]   = df["num_services"] / (df["tenure"] + 1)

    features = ["tenure", "monthly_charges", "total_charges", "num_services",
                "contract_type", "payment_method", "internet_service",
                "tech_support", "senior_citizen", "charges_per_month", "service_density"]

    X = df[features]
    y = df["churn"]
    return X, y, features



def train_xgboost(X_train, y_train):
    print("\n── XGBoost + GridSearchCV ──")
    param_grid = {
        "n_estimators":    [100, 200],
        "max_depth":       [3, 5],
        "learning_rate":   [0.05, 0.1],
        "subsample":       [0.8, 1.0],
        "colsample_bytree":[0.8, 1.0],
    }
    xgb_clf = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(xgb_clf, param_grid, cv=cv,
                        scoring="accuracy", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")
    return grid.best_estimator_



def evaluate(model, X_test, y_test, features, save_dir="outputs"):
    preds  = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probas)
    print(f"\nTest Accuracy: {acc:.4f}  |  ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, preds, target_names=["Stay", "Churn"]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))


    fpr, tpr, _ = roc_curve(y_test, probas)
    axes[0].plot(fpr, tpr, color="#E74C3C", lw=2, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].legend()


    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Stay", "Churn"],
                yticklabels=["Stay", "Churn"], ax=axes[1])
    axes[1].set_title("Confusion Matrix")


    importances = pd.Series(model.feature_importances_, index=features).sort_values()
    importances.plot(kind="barh", ax=axes[2], color="#3498DB")
    axes[2].set_title("Feature Importance")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/model_evaluation.png", dpi=150)
    plt.close()
    print("[Saved] model_evaluation.png")
    return acc



def main():
    import os
    os.makedirs("outputs", exist_ok=True)

    print("=== Customer Churn Prediction ===\n")
    df = generate_telecom_data()
    print(f"Dataset: {df.shape}  |  Churn rate: {df['churn'].mean():.2%}")

    run_eda(df)

    X, y, features = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = train_xgboost(X_train, y_train)
    acc   = evaluate(model, X_test, y_test, features)

    print(f"\n✅ Final Test Accuracy: {acc:.2%}")
    print("Check 'outputs/' folder for EDA dashboard and model evaluation plots.")


if __name__ == "__main__":
    main()
