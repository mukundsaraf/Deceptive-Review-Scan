import argparse
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

warnings.filterwarnings("ignore")

# threshold boundaries for labelling predictions
FAKE_THRESHOLD      = 0.65
REAL_THRESHOLD      = 0.35

# parameters for semi-supervised pseudo-labeling
PSEUDO_CONFIDENCE   = 0.80
MAX_PSEUDO_RATIO    = 0.40

TFIDF_MAX_FEATURES  = 15_000
TFIDF_NGRAM_RANGE   = (1, 2)
RANDOM_STATE        = 42

def build_text_column(df: pd.DataFrame) -> pd.Series:
    # join title and main text together for text processing
    title = df["review_title"].fillna("").astype(str)
    text  = df["review_text"].fillna("").astype(str)
    return title + " " + text

def build_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    # pull out length and structural features from text columns
    feat = pd.DataFrame(index=df.index)

    text  = df["review_text"].fillna("").astype(str)
    title = df["review_title"].fillna("").astype(str)

    feat["text_len"]          = text.str.len()
    feat["word_count"]        = text.str.split().str.len().fillna(0)
    feat["title_length"]      = title.str.len()

    feat["caps_ratio"]        = text.apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
    )
    feat["exclamation_count"] = text.str.count(r"!")
    feat["question_count"]    = text.str.count(r"\?")
    feat["avg_word_len"]      = text.apply(
        lambda t: np.mean([len(w) for w in t.split()] or [0])
    )

    # get numeric fields and flag extreme ratings
    feat["review_rating"]     = pd.to_numeric(df["review_rating"], errors="coerce").fillna(3)
    feat["rating_extreme"]    = feat["review_rating"].apply(lambda r: int(r in (1, 5)))
    feat["rating_5"]          = (feat["review_rating"] == 5).astype(int)
    feat["rating_1"]          = (feat["review_rating"] == 1).astype(int)

    # get the helpfulness metrics
    feat["helpful"]           = pd.to_numeric(
        df.get("number_of_helpful", pd.Series(0, index=df.index)),
        errors="coerce"
    ).fillna(0)
    feat["helpfulness_ratio"] = feat["helpful"] / (feat["text_len"] + 1)

    # see if photos are included
    feat["has_photos"]        = pd.to_numeric(
        df.get("has_photos", df.get("number_of_photos", pd.Series(0, index=df.index))),
        errors="coerce"
    ).fillna(0).astype(int)

    feat["number_of_photos"]  = pd.to_numeric(
        df.get("number_of_photos", pd.Series(0, index=df.index)),
        errors="coerce"
    ).fillna(0)

    # look at purchase context indicators
    feat["is_campaign"]       = pd.to_numeric(
        df.get("is_campaign_product", pd.Series(0, index=df.index)),
        errors="coerce"
    ).fillna(0).astype(int)

    feat["verified_purchase"] = pd.to_numeric(
        df.get("verified_purchase", pd.Series(0, index=df.index)),
        errors="coerce"
    ).fillna(0).astype(int)

    # extract temporal features
    feat["review_year"]       = pd.to_numeric(
        df.get("review_year", pd.Series(2019, index=df.index)),
        errors="coerce"
    ).fillna(2019)
    feat["review_month"]      = pd.to_numeric(
        df.get("review_month", pd.Series(6, index=df.index)),
        errors="coerce"
    ).fillna(6)

    # check optional domain columns
    feat["image_count"]       = pd.to_numeric(
        df.get("image_count", pd.Series(0, index=df.index)),
        errors="coerce"
    ).fillna(0)
    feat["has_images"]        = (feat["image_count"] > 0).astype(int)

    feat["style_present"]     = pd.to_numeric(
        df.get("style_present", pd.Series(0, index=df.index)),
        errors="coerce"
    ).fillna(0).astype(int)

    feat["very_short_review"] = (feat["word_count"] < 5).astype(int)

    return feat.astype(float)

def build_label(df: pd.DataFrame) -> pd.Series:
    # combine multiple labels into one single target column
    fake   = df["fake_review_product"].astype(bool)
    rfake  = df["reviewer_classified_fake"].astype(bool)
    removed = pd.to_numeric(
        df.get("review_is_removed_by_amazon", pd.Series(0, index=df.index)),
        errors="coerce"
    ).fillna(0).astype(bool)

    return (fake | rfake | removed).astype(int)

class FakeReviewClassifier:
    def __init__(self):
        self.tfidf_pipe   = None
        self.numeric_pipe = None
        self.scaler       = StandardScaler()

    def _tfidf_pipeline(self):
        # build tfidf feature extractor and text classifier
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features  = TFIDF_MAX_FEATURES,
                ngram_range   = TFIDF_NGRAM_RANGE,
                sublinear_tf  = True,
                strip_accents = "unicode",
                min_df        = 3,
            )),
            ("clf", LogisticRegression(
                C             = 1.0,
                max_iter      = 500,
                solver        = "saga",
                class_weight  = "balanced",
                random_state  = RANDOM_STATE,
            )),
        ])

    def _numeric_lr(self):
        # logistic regression setup for the numeric side
        return LogisticRegression(
            C            = 1.0,
            max_iter     = 500,
            solver       = "saga",
            class_weight = "balanced",
            random_state = RANDOM_STATE,
        )

    def fit(self, df: pd.DataFrame, y: pd.Series):
        # fit both sub-models on texts and scaled numbers
        texts   = build_text_column(df)
        numeric = build_numeric_features(df)
        numeric_scaled = self.scaler.fit_transform(numeric)

        self.tfidf_pipe   = self._tfidf_pipeline()
        self.numeric_pipe = self._numeric_lr()

        self.tfidf_pipe.fit(texts, y)
        self.numeric_pipe.fit(numeric_scaled, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        # compute ensemble prediction probability as weighted average
        texts          = build_text_column(df)
        numeric        = build_numeric_features(df)
        numeric_scaled = self.scaler.transform(numeric)

        p_text    = self.tfidf_pipe.predict_proba(texts)[:, 1]
        p_numeric = self.numeric_pipe.predict_proba(numeric_scaled)[:, 1]

        return 0.60 * p_text + 0.40 * p_numeric

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        # get predictions using a threshold
        return (self.predict_proba(df) >= threshold).astype(int)

def train_supervised(labeled_path: str) -> tuple[FakeReviewClassifier, dict]:
    # load labeled data and split into train and validation sets
    print("\n[Stage C] Loading labeled dataset …")
    df = pd.read_csv(labeled_path)
    print(f"          {len(df):,} rows, columns: {df.columns.tolist()}")

    y = build_label(df)
    print(f"          Label distribution — fake: {y.sum():,}  real: {(y==0).sum():,}")

    X_train, X_val, y_train, y_val = train_test_split(
        df, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )

    print(f"          Training on {len(X_train):,} samples …")
    clf = FakeReviewClassifier()
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_val)
    preds = (proba >= 0.5).astype(int)
    auc   = roc_auc_score(y_val, proba)

    print("\n[Stage C] Validation Results")
    print(f"          ROC-AUC : {auc:.4f}")
    print(classification_report(y_val, preds, target_names=["real", "fake"]))

    metrics = {"auc": auc, "val_size": len(X_val)}
    return clf, metrics

def domain_adapt(
    clf_base: FakeReviewClassifier,
    labeled_path: str,
    unlabeled_path: str,
) -> FakeReviewClassifier:
    # adapt model to unlabeled dataset via pseudo-label generation
    print("\n[Stage D] Loading unlabeled domain data …")
    df_domain = pd.read_csv(unlabeled_path)
    print(f"          {len(df_domain):,} rows")

    proba        = clf_base.predict_proba(df_domain)
    pseudo_label = (proba >= 0.5).astype(int)
    confident    = (proba >= PSEUDO_CONFIDENCE) | (proba <= (1 - PSEUDO_CONFIDENCE))

    df_pseudo        = df_domain[confident].copy()
    df_pseudo["_y"]  = pseudo_label[confident]

    df_labeled = pd.read_csv(labeled_path)
    max_pseudo = int(len(df_labeled) * MAX_PSEUDO_RATIO)
    if len(df_pseudo) > max_pseudo:
        df_pseudo = df_pseudo.sample(max_pseudo, random_state=RANDOM_STATE)

    print(f"          Pseudo-labels accepted: {len(df_pseudo):,}  ")

    y_labeled  = build_label(df_labeled)
    df_labeled = df_labeled.copy()
    df_labeled["_y"] = y_labeled

    shared_cols = [c for c in df_labeled.columns if c in df_pseudo.columns and c != "_y"]
    shared_cols_with_y = shared_cols + ["_y"]

    df_combined = pd.concat(
        [df_labeled[shared_cols_with_y], df_pseudo[shared_cols_with_y]],
        ignore_index=True
    )
    y_combined  = df_combined.pop("_y")

    print(f"          Combined training size: {len(df_combined):,}")

    clf_adapted = FakeReviewClassifier()
    clf_adapted.fit(df_combined, y_combined)
    print("[Stage D] Domain-adapted model trained.")
    return clf_adapted

def infer(clf: FakeReviewClassifier, infer_path: str, output_dir: str):
    # run inference on test data and save soft and hard predictions
    print(f"\n[Stage E] Running inference on: {infer_path}")
    df    = pd.read_csv(infer_path)
    proba = clf.predict_proba(df)

    hard = np.where(
        proba >= FAKE_THRESHOLD, "fake",
        np.where(proba <= REAL_THRESHOLD, "real", "uncertain")
    )

    df_soft       = df.copy()
    df_soft["fake_probability"] = proba
    df_soft["prediction_soft"]  = hard

    df_hard       = df.copy()
    df_hard["fake_label"]       = (proba >= 0.5).astype(int)
    df_hard["confidence"]       = np.where(proba >= 0.5, proba, 1 - proba)

    soft_path = Path(output_dir) / "inference_results_soft.csv"
    hard_path = Path(output_dir) / "inference_results_hard.csv"

    df_soft.to_csv(soft_path, index=False)
    df_hard.to_csv(hard_path, index=False)

def _distribution_report(proba: np.ndarray, label: str):
    # print prediction percentages for different classes
    hard = np.where(
        proba >= FAKE_THRESHOLD, "fake",
        np.where(proba <= REAL_THRESHOLD, "real", "uncertain")
    )
    counts = pd.Series(hard).value_counts()
    total  = len(hard)
    print(f"\n  [{label}] Prediction Distribution  (n={total:,})")
    for lbl in ["fake", "uncertain", "real"]:
        c = counts.get(lbl, 0)
        print(f"  {lbl:<12} {c:>8,}  {c/total*100:>5.1f}%")
    return hard

def evaluate_on_domain(
    clf_base:    FakeReviewClassifier,
    clf_adapted: FakeReviewClassifier,
    domain_path: str,
    output_dir:  str,
):
    # run comparison report across both models on unlabeled domain data
    df = pd.read_csv(domain_path)
    
    p_base = clf_base.predict_proba(df)
    hard_base = _distribution_report(p_base, "Foundational")

    p_adapted = clf_adapted.predict_proba(df)
    hard_adapted = _distribution_report(p_adapted, "Adapted")

    agree     = (hard_base == hard_adapted).sum()
    disagree  = len(hard_base) - agree
    print(f"\n  Agree    : {agree:,}  ({agree/len(hard_base)*100:.1f}%)")

    domain_name = Path(domain_path).stem
    out_path = Path(output_dir) / f"{domain_name}_predictions.csv"

    df_out = df.copy()
    df_out["base_fake_probability"]    = p_base
    df_out["adapted_fake_probability"] = p_adapted
    df_out["base_prediction"]          = hard_base
    df_out["adapted_prediction"]       = hard_adapted
    df_out["models_agree"]             = (hard_base == hard_adapted).astype(int)
    df_out.to_csv(out_path, index=False)

def main():
    parser = argparse.ArgumentParser(description="Fake Review Detection Pipeline")
    parser.add_argument("--labeled",   required=True,  help="Path to cleaned_reviews.csv")
    parser.add_argument("--unlabeled", required=True,  help="Path to Appliances.csv")
    parser.add_argument("--infer",     default=None,   help="Optional: CSV for inference")
    parser.add_argument("--output",    default="output/", help="Output directory")
    parser.add_argument("--sample",    type=int, default=None, help="Max rows to sample")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    clf_base, metrics = train_supervised(args.labeled)
    clf_adapted = domain_adapt(clf_base, args.labeled, args.unlabeled)
    evaluate_on_domain(clf_base, clf_adapted, args.unlabeled, args.output)

    if args.infer:
        infer(clf_adapted, args.infer, args.output)

if __name__ == "__main__":
    main()