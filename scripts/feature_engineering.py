import pandas as pd
import numpy as np

def engineer_features(input_path, output_path):

    # load cleaned data
    df = pd.read_csv(input_path)
    print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")

    # family size and is alone
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # extract title from name
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
    df['Title'] = df['Title'].replace({
        'Mlle': 'Miss',
        'Ms': 'Miss',
        'Mme': 'Mrs',
        'Lady': 'Rare',
        'Countess': 'Rare',
        'the Countess': 'Rare',
        'Capt': 'Rare',
        'Col': 'Rare',
        'Don': 'Rare',
        'Dr': 'Rare',
        'Major': 'Rare',
        'Rev': 'Rare',
        'Sir': 'Rare',
        'Jonkheer': 'Rare',
        'Dona': 'Rare'
    })

    # age groups
    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[0, 12, 18, 60, 100],
        labels=['Child', 'Teen', 'Adult', 'Senior']
    )

    # fare per person
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # log transform fare
    df['Fare_log'] = np.log1p(df['Fare'])

    # drop columns no longer needed
    df = df.drop(columns=['Name', 'Ticket', 'PassengerId'])

    # encode categorical columns
    df = pd.get_dummies(
        df,
        columns=['Sex', 'Embarked', 'Title', 'Deck', 'AgeGroup'],
        drop_first=True
    )

    # drop stray countess column if it exists
    if 'Title_the Countess' in df.columns:
        df = df.drop(columns=['Title_the Countess'])

    # save
    df.to_csv(output_path, index=False)
    print(f"Engineered data saved to {output_path}")
    print(f"Shape after feature engineering: {df.shape}")

    return df


if __name__ == "__main__":
    engineer_features(
        input_path='../data/train_cleaned.csv',
        output_path='../data/train_featured.csv'
    )