from ucimlrepo import fetch_ucirepo
import pandas as pd


def load_data():
    """Load the Wine Quality dataset."""
    wine_quality = fetch_ucirepo(id=186)
    X = wine_quality.data.features
    y = wine_quality.data.targets
    return X, y


def preprocess_data(X, y):
    """Remove duplicates and prepare the data."""
    combined_df = pd.concat([X, y], axis=1)
    unique_df = combined_df.drop_duplicates()
    X = unique_df.drop(columns=['quality'])
    y = unique_df['quality']
    return X, y
