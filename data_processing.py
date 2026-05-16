import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# path settings
BASE_DIR = r"C:\Users\vansh\OneDrive\Desktop\college\MLPR\Project"
DATA_PATH = os.path.join(BASE_DIR, "public_reviews_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "reviewer_classified_fake"

# load original data
df = pd.read_csv(DATA_PATH, low_memory=False)

print("Original shape:", df.shape)

# see where data is missing
plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Data Pattern")
plt.savefig(os.path.join(OUTPUT_DIR, "missingness_heatmap.png"))
plt.close()

# remove columns that are ids or links or cause leakage
cols_to_drop = [
    "asin", "review_id", "reviewer_id",
    "product_url", "review_url", "reviewer_url", "asin_url",
    "reviewer_labeled_fake", "reviewer_labeled_honest"
]

df.drop(columns=cols_to_drop, inplace=True)

# convert photo urls to a binary feature
df["has_photos"] = df["photo_thumbnail_urls"].notna().astype(int)
df.drop(columns=["photo_thumbnail_urls", "photo_fullsize_urls"], inplace=True)

# create campaign indicator feature
df["is_campaign_product"] = df["fake_review_campaign_start_date"].notna().astype(int)
df.drop(columns=["fake_review_campaign_start_date"], inplace=True)

# convert date and pull out year and month
df["review_date"] = pd.to_datetime(df["review_date"])

df["review_year"] = df["review_date"].dt.year
df["review_month"] = df["review_date"].dt.month

df.drop(columns=["review_date"], inplace=True)

# fill missing numbers using median value
num_cols = ["number_of_helpful", "number_of_photos"]

imputer = SimpleImputer(strategy="median")
df[num_cols] = imputer.fit_transform(df[num_cols])

# remove the extra honest label column
df.drop(columns=["reviewer_classified_honest"], inplace=True)

# check class distribution balance
plt.figure(figsize=(6,4))
sns.countplot(x=df[TARGET])
plt.title("Class Distribution")
plt.savefig(os.path.join(OUTPUT_DIR, "class_balance.png"))
plt.close()

# check helpful votes distribution
plt.figure(figsize=(6,4))
sns.histplot(df["number_of_helpful"], bins=50)
plt.title("Helpful Votes Distribution")
plt.savefig(os.path.join(OUTPUT_DIR, "helpful_votes_distribution.png"))
plt.close()

# check if campaign affects fake reviews
plt.figure(figsize=(6,4))
sns.countplot(x="is_campaign_product", hue=TARGET, data=df)
plt.title("Campaign Effect on Fake Reviews")
plt.savefig(os.path.join(OUTPUT_DIR, "campaign_vs_fake.png"))
plt.close()

# save out the clean dataset
clean_path = os.path.join(OUTPUT_DIR, "cleaned_reviews.csv")
df.to_csv(clean_path, index=False)

print("\nAfter preprocessing:", df.shape)

missing_after = df.isnull().mean() * 100
print("\nRemaining missing values (%):\n", missing_after[missing_after > 0])

print(f"\nClean dataset saved to: {clean_path}")
print("All plots saved inside:", OUTPUT_DIR)