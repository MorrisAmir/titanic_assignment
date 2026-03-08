# Titanic Survival Prediction — Feature Engineering Assignment

## Project Overview
This project builds a predictive foundation for Titanic survival classification
by performing data cleaning, feature engineering and feature selection on the
Titanic dataset from Kaggle. The goal is to transform raw, messy passenger data
into a clean, intelligent dataset ready for machine learning.

---

## Project Structure
```
titanic_assignment/
├── data/
│   ├── train.csv               ← original dataset
│   ├── train_cleaned.csv       ← after data cleaning
│   └── train_engineered.csv    ← final dataset after feature engineering
├── notebooks/
│   └── Titanic_Feature_Engineering.ipynb
├── scripts/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── feature_selection.py
├── README.md
└── requirements.txt
```


## How to Run

### 1. Clone the repository and install dependecies

git clone https://github.com/YOURUSERNAME/titanic_assignment.git
cd titanic_assignment
pip install -r requirements.txt


### 2. Download the dataset
- Go to https://www.kaggle.com/c/titanic/data
- Download train.csv and test.csv
- Place them in the data/ folder

### 3. Run the notebook

jupyter notebook notebooks/Titanic_Feature_Engineering.ipynb

### APPROACH
Done in 3 stages:

### a) Data cleaning
To remove inconsistencies and handle missing values

### b) Feature engineering 
Engineering new features from the existing columns

### c) Feature selection 
To identify and keep only the most useful columns

### DATA CLEANING DECISIONS

## Age:
-Missing values in age were filled with medina ge of 28. Median was preferred because mean deals with extremes

## Cabin:
-Deck letter was extracted( getting the first character). The rest was dropped. Passengers wth no cabi were labelled unknown
-Embarked had only 2 missing values. These were filled with S (Southampton) since it was by far the most common port of departure.
-Fare had outliers, they were capped at the 99th percentile to reduce influence without losing passenger rows entirely
-No duplicate values found

### FEATURES ENGINEERED
FamilySize — created by adding SibSp + Parch + 1. Gives the total number of family members on board including the passenger themselves.

IsAlone — equals 1 if FamilySize is 1, otherwise 0. Passengers travelling alone behaved differently during evacuation.

Title — extracted from the Name column using the pattern before the dot (Mr, Mrs, Miss, Master etc.). Rare titles like Rev, Dr and Col were grouped into a single Rare category. This feature captures both gender and social status in one column.

Deck — extracted as the first letter of the Cabin value. Passengers with no cabin record were assigned Unknown. Higher decks had noticeably better survival rates.

AgeGroup — Age was grouped into four categories: Child (0-12), Teen (13-18), Adult (19-60) and Senior (61+). This captures the priority given to children during evacuation better than raw age alone.

FarePerPerson — calculated as Fare divided by FamilySize. A family sharing one ticket should not all be counted as paying full fare. This is a more accurate measure of individual wealth.

Fare_log — log transformation of the Fare column. Fare was heavily skewed with a small number of very high values. The log transform compressed those extremes and produced a more balanced distribution.

### FEATURE SELECTION
A Random Forest model was trained on all features to rank them by importance. Features scoring below 0.01 importance were dropped as they added more noise than signal.

## Features kept and why:

Age (0.1655) — the strongest single predictor. Younger passengers were prioritised during evacuation.
Title_Mr (0.1380) — adult males had the lowest survival rate. This title was the clearest marker of that group.
FarePerPerson (0.1081) — outperformed raw Fare, confirming that the engineered version was a better signal.
Sex_male (0.1024) — gender was one of the most decisive factors in who survived.
Fare_log (0.1020) — the log-transformed fare captured wealth differences more fairly than the raw skewed values.
Fare (0.0939) — kept alongside Fare_log as tree-based models can use both effectively.
Pclass (0.0466) — ticket class determined lifeboat priority and physical proximity to the boat deck.
FamilySize (0.0408) — larger families had more difficulty evacuating together.
Title_Miss (0.0344) and Title_Mrs (0.0286) — female titles confirmed the gender survival pattern.
SibSp (0.0222) — number of siblings and spouses affected evacuation behaviour.
Deck_Unknown (0.0218) — passengers with no cabin record had a survival rate of only 0.30, making this a strong signal.
Embarked_S (0.0165), Parch (0.0120), AgeGroup_Adult (0.0109) — moderate signal, kept above the threshold.

## Features dropped and why:

IsAlone — redundant, already captured by FamilySize.
Individual Deck columns (B, C, D, E, F, G) — too few passengers per deck to form reliable patterns.
AgeGroup_Teen and AgeGroup_Senior — too few passengers in these groups.
Title_Rare and Embarked_Q — very small groups with near-zero importance scores.

### KEY FINDINGS
Age was the most important feature with a score of 0.1655. This independently confirms the historical account that children were given priority during evacuation.

Gender and title together dominated the top four features. Being an adult male was the strongest predictor of death on the Titanic.

The engineered feature FarePerPerson scored higher than raw Fare, demonstrating that feature engineering produced genuinely better signals than what the original dataset provided.

Passengers with unknown deck (missing cabin) had a lower survival rate compared to those recorded with a cabin. This single feature captured class inequality very effectively.

### REQUIREMENTS
See requirements.txt for full list.

 Main libraries used: pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter.