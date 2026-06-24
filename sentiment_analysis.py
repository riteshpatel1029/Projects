"""
Sentiment Analysis on Product Reviews
Author: Ritesh Patel
Tech Stack: Python · NLTK · BERT/Transformers · Scikit-learn
"""

import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# ── Optional: use transformers if available, else fallback to TF-IDF+LR ──
try:
    from transformers import (BertTokenizer, BertForSequenceClassification,
                               Trainer, TrainingArguments)
    import torch
    from torch.utils.data import Dataset
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore")

# Download NLTK assets
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


# ─────────────────────────────────────────────
# 1. Data Generation
# ─────────────────────────────────────────────
POSITIVE = [
    "This product is absolutely amazing and exceeded all my expectations!",
    "Fantastic quality, fast delivery and great value for money.",
    "I love this item. Works perfectly and looks beautiful.",
    "Best purchase I have made this year. Highly recommend!",
    "Excellent build quality and easy to use. Will buy again.",
    "Super happy with this. The performance is outstanding.",
    "Great product, arrived on time and packaging was perfect.",
    "Very satisfied. Does exactly what it claims to do.",
    "Top notch quality. Customer service was also brilliant.",
    "Incredible product! My whole family loves it.",
]

NEGATIVE = [
    "Terrible product. Broke after just two days of use.",
    "Worst purchase ever. Complete waste of money.",
    "Very disappointed. Does not match the description at all.",
    "Poor quality and terrible customer service. Avoid!",
    "Returned immediately. Item arrived damaged and dirty.",
    "Awful experience. Package was late and product was broken.",
    "Do not buy this. It stopped working within a week.",
    "Not worth the price at all. Very cheap materials.",
    "Product is fake. Nothing like the photos shown.",
    "Defective item. Company refused to give a refund.",
]

NEUTRAL = [
    "The product is okay. Nothing special but does the job.",
    "Average quality. It works as expected, nothing more.",
    "Decent product for the price. Could be better.",
    "It is fine. Not amazing but not terrible either.",
    "Reasonable product. Delivery was on time.",
    "Works as described. No complaints but no wow factor.",
    "Fairly standard product. Meets basic expectations.",
    "Neither impressed nor disappointed. Just okay.",
    "Adequate for everyday use. Nothing to write home about.",
    "Generic product. Does what it says on the tin.",
]

LABEL_MAP = {"positive": 2, "negative": 0, "neutral": 1}
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def generate_review_dataset(n_per_class=300, seed=42) -> pd.DataFrame:
    np.random.seed(seed)
    rows = []
    for label, samples in [("positive", POSITIVE), ("negative", NEGATIVE), ("neutral", NEUTRAL)]:
        for _ in range(n_per_class):
            base = np.random.choice(samples)
            rows.append({"text": base, "label": LABEL_MAP[label], "label_name": label})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


# ─────────────────────────────────────────────
# 2. Text Preprocessing
# ─────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


# ─────────────────────────────────────────────
# 3a. TF-IDF + Logistic Regression (baseline)
# ─────────────────────────────────────────────
def run_tfidf_lr(X_train, y_train, X_test, y_test):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    print("\n── TF-IDF + Logistic Regression ──")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
        ("lr",    LogisticRegression(max_iter=500, random_state=42, C=1.0)),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["Negative", "Neutral", "Positive"]))
    return pipe, preds, acc


# ─────────────────────────────────────────────
# 3b. BERT Fine-Tuning (when transformers available)
# ─────────────────────────────────────────────
class ReviewDataset(Dataset if BERT_AVAILABLE else object):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def run_bert(texts_train, y_train, texts_test, y_test, epochs=2):
    if not BERT_AVAILABLE:
        print("[BERT] transformers not installed — skipping BERT training.")
        return None, None

    print("\n── Fine-tuning BERT (bert-base-uncased) ──")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model     = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3)

    enc_train = tokenizer(list(texts_train), truncation=True, padding=True, max_length=128)
    enc_test  = tokenizer(list(texts_test),  truncation=True, padding=True, max_length=128)

    train_ds = ReviewDataset(enc_train, list(y_train))
    test_ds  = ReviewDataset(enc_test,  list(y_test))

    args = TrainingArguments(
        output_dir="./bert-sentiment",
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
        save_strategy="no",
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds)
    trainer.train()

    preds_out  = trainer.predict(test_ds)
    preds_bert = np.argmax(preds_out.predictions, axis=1)
    acc_bert   = accuracy_score(y_test, preds_bert)
    print(f"BERT Test Accuracy: {acc_bert:.4f}")
    print(classification_report(y_test, preds_bert,
                                 target_names=["Negative", "Neutral", "Positive"]))
    return model, preds_bert


# ─────────────────────────────────────────────
# 4. Visualisations
# ─────────────────────────────────────────────
def plot_wordclouds(df: pd.DataFrame, save_dir="outputs"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (label_name, label_id) in zip(axes,
            [("positive", 2), ("neutral", 1), ("negative", 0)]):
        subset = df[df["label"] == label_id]["clean_text"].str.cat(sep=" ")
        wc = WordCloud(width=600, height=400,
                       background_color="white",
                       colormap="RdYlGn" if label_name != "negative" else "Reds"
                       ).generate(subset)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{label_name.capitalize()} Reviews", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/wordclouds.png", dpi=150)
    plt.close()
    print("[Saved] wordclouds.png")


def plot_confusion(y_test, preds, title, save_dir="outputs"):
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Neg", "Neu", "Pos"],
                yticklabels=["Neg", "Neu", "Pos"], ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    fname = f"{save_dir}/{title.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[Saved] {fname}")


# ─────────────────────────────────────────────
# 5. Inference Helper
# ─────────────────────────────────────────────
def predict_sentiment(text: str, model) -> str:
    """Predict sentiment for a single review using the TF-IDF pipeline."""
    clean = preprocess_text(text)
    pred  = model.predict([clean])[0]
    return LABEL_NAMES[pred]


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────
def main():
    import os
    os.makedirs("outputs", exist_ok=True)

    print("=== Sentiment Analysis on Product Reviews ===\n")

    df = generate_review_dataset(n_per_class=350)
    print(f"Dataset: {df.shape}  |  Label counts:\n{df['label_name'].value_counts()}\n")

    df["clean_text"] = df["text"].apply(preprocess_text)

    X = df["clean_text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Baseline model
    pipe, preds_lr, acc_lr = run_tfidf_lr(X_train, y_train, X_test, y_test)
    plot_confusion(y_test, preds_lr, "TF-IDF LR Confusion Matrix")

    # WordClouds
    plot_wordclouds(df)

    # BERT (optional)
    bert_model, preds_bert = run_bert(X_train, y_train, X_test, y_test, epochs=2)
    if preds_bert is not None:
        plot_confusion(y_test, preds_bert, "BERT Confusion Matrix")

    # Demo inference
    print("\n── Inference Demo ──")
    samples = [
        "This product is absolutely incredible, I love it!",
        "Waste of money. Broke after one day.",
        "It is okay, does the job I suppose.",
    ]
    for s in samples:
        print(f"  '{s[:50]}...' → {predict_sentiment(s, pipe).upper()}")

    print("\n✅ Done. Check 'outputs/' for plots.")


if __name__ == "__main__":
    main()
