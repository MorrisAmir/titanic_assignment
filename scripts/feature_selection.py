import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def select_features(input_path, output_path):

    # load engineered data
    df = pd.read_csv(input_path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    # separate features from target
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    # train random forest to get importance scores
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # rank features by importance
    importance = pd.Series(rf.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)

    print("\nFeature Importances:")
    print(importance)

    # plot importance
    plt.figure(figsize=(10, 8))
    importance.plot(kind='barh', color='teal')
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance Score')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('../data/feature_importance.png')
    print("Feature importance chart saved.")

    # drop weak features below threshold
    threshold = 0.01
    weak_features = importance[importance < threshold].index.tolist()
    print(f"\nDropping {len(weak_features)} weak features: {weak_features}")

    df_selected = df.drop(columns=weak_features)
    print(f"Final shape: {df_selected.shape}")
    print(f"Selected features: {df_selected.columns.tolist()}")

    # save final dataset
    df_selected.to_csv(output_path, index=False)
    print(f"\nFinal dataset saved to {output_path}")

    return df_selected


if __name__ == "__main__":
    select_features(
        input_path='../data/train_featured.csv',
        output_path='../data/train_engineered.csv'
    )