import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. LOAD DATA
# ==========================================
# Load the Titanic dataset from Seaborn[cite: 3]
df = sns.load_dataset('titanic')


# ==========================================
# 2. EXPLORE (EDA & Missing Value Detection)
# ==========================================
# Check dataset structure and missing values using info() and isnull().sum()[cite: 3]
print("--- DataFrame Info ---")
df.info()

print("\n--- Missing Value Counts ---")
print(df.isnull().sum())
print("-" * 50)


# ==========================================
# 3. CLEAN DATA
# ==========================================
# Fill missing values in the 'age' column with the mean age of all passengers[cite: 3]
average = round(df["age"].mean())
df["age"] = df["age"].fillna(average)

# Drop the 'deck' column completely due to an excessive number of missing values[cite: 3]
df = df.drop(columns="deck")


# ==========================================
# 4. VISUALIZE (EDA)
# ==========================================
# Create a bar plot comparing the survival rates of females and males[cite: 3]
plt.figure(figsize=(6, 4))
sns.barplot(x="sex", y="survived", data=df, errorbar=None, palette="Set2")
plt.title("Survival Rate by Gender")
plt.show()

# Create a Correlation Heatmap to see relationships between numerical variables[cite: 3]
corr = df.corr(numeric_only=True)
plt.figure(figsize=(8, 6))
sns.heatmap(corr, square=True, cmap="RdBu", vmin=-1, vmax=1, annot=True, annot_kws={'fontsize': 11, 'fontweight': "bold"})
plt.title("Correlation Heatmap")
plt.show()


# ==========================================
# 5. FEATURE ENGINEERING
# ==========================================
# Convert categorical string columns ('sex' and 'embarked') into numeric (1/0) using One-Hot Encoding[cite: 3]
yeni_df = pd.get_dummies(df, columns=["sex", "embarked"], dtype=int)

print("\n--- New DataFrame After One-Hot Encoding ---")
print(yeni_df.head(10))