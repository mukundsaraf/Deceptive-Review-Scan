import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# setup paths for files
BASE_DIR = r"C:\Users\vansh\OneDrive\Desktop\college\MLPR\Project"
DATA_PATH = os.path.join(BASE_DIR, "outputs", "cleaned_reviews.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "pure_eda_outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "reviewer_classified_fake"

# load the cleaned dataset
df = pd.read_csv(DATA_PATH)

print("Loaded shape:", df.shape)

# fix missing text with empty strings
df["review_title"] = df["review_title"].fillna("")
df["product_title"] = df["product_title"].fillna("")

# plot helpful votes before taking log
plt.figure()
sns.histplot(df["number_of_helpful"], bins=50)
plt.title("Helpful Votes Before Log")
plt.savefig(os.path.join(OUTPUT_DIR, "helpful_before_log.png"))
plt.close()

# take log transform because data is skewed
df["log_helpful_votes"] = np.log1p(df["number_of_helpful"])

# plot after log transform
plt.figure()
sns.histplot(df["log_helpful_votes"], bins=50)
plt.title("Helpful Votes After Log")
plt.savefig(os.path.join(OUTPUT_DIR, "helpful_after_log.png"))
plt.close()

# get numeric columns and make a heatmap
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()

# calculate length of reviews
df["review_length"] = df["review_text"].astype(str).apply(len)

# plot length distribution for fake vs real
plt.figure()
sns.histplot(data=df, x="review_length", hue=TARGET, bins=50)
plt.title("Review Length Distribution (Fake vs Genuine)")
plt.savefig(os.path.join(OUTPUT_DIR, "review_length_distribution.png"))
plt.close()

# save the mean lengths to a csv
length_stats = df.groupby(TARGET)["review_length"].mean()
length_stats.to_csv(os.path.join(OUTPUT_DIR, "review_length_stats.csv"))

# boxplot for helpful votes vs target
plt.figure()
sns.boxplot(x=TARGET, y="log_helpful_votes", data=df)
plt.title("Helpful Votes vs Fake Reviews")
plt.savefig(os.path.join(OUTPUT_DIR, "helpful_vs_fake.png"))
plt.close()

# see fake rates for campaign products
campaign_rate = (
    df.groupby("is_campaign_product")[TARGET]
    .mean()
)

campaign_rate.to_csv(os.path.join(OUTPUT_DIR, "campaign_fake_rate.csv"))

plt.figure()
campaign_rate.plot(kind="bar")
plt.title("Fake Review Rate for Campaign vs Non-Campaign")
plt.ylabel("Fake Review Ratio")
plt.savefig(os.path.join(OUTPUT_DIR, "campaign_fake_rate.png"))
plt.close()

# create and save a basic summary dictionary
summary = {
    "dataset_size": df.shape[0],
    "fake_ratio": df[TARGET].mean(),
    "avg_review_length_fake": length_stats[1],
    "avg_review_length_genuine": length_stats[0],
}

pd.Series(summary).to_csv(os.path.join(OUTPUT_DIR, "dataset_summary.csv"))

print("EDA complete. Outputs saved to:", OUTPUT_DIR)