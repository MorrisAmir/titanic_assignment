import pandas as pd
import numpy as np

def clean_data(input_path, output_path):
    
    # load data
    df = pd.read_csv(input_path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    # fix embarked — fill 2 missing with most common port
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # fix age — fill missing with median
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # extract deck from cabin then drop cabin
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'Unknown')
    df['Deck'] = df['Deck'].replace('T', 'Unknown')
    df = df.drop(columns=['Cabin'])

    # cap fare outliers at 99th percentile
    fare_cap = df['Fare'].quantile(0.99)
    df['Fare'] = df['Fare'].clip(upper=fare_cap)

    # remove duplicates
    df = df.drop_duplicates()

    # save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Remaining missing values: {df.isnull().sum().sum()}")

    return df


if __name__ == "__main__":
    clean_data(
        input_path='../data/train.csv',
        output_path='../data/train_cleaned.csv'
    )