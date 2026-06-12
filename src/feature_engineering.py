import pandas as pd
from lifetimes.datasets import load_cdnow_summary_data_with_monetary_value

def load_and_validate_data():
    df = load_cdnow_summary_data_with_monetary_value()
    df = df.reset_index()
    print(f"Loaded {len(df)} customers")
    print(f"Columns : {df.columns.tolist()}")
    print(f"Nulls:\n{df.isnull().sum()}")
    return df

def feature_engineering(df):
    features = df.copy()


    # Recency ratio — how recent vs how long they've been a customer
    features['recency_ratio'] = features['recency'] / (features['T'] + 1)

    # Purchase rate — how often they buy per unit time
    features['purchase_rate'] = features['frequency'] / (features['T'] + 1)

    # One-time buyer flag
    features['is_one_time_buyer'] = (features['frequency']== 0).astype(int)

    # High value flag — top 25% by monetary value
    threshold = features['monetary_value'].quantile(0.75)
    features['is_high_value'] = (features['monetary_value'] >= threshold).astype(int)

    # Churn label — inactive customers (frequency=0 OR recency very low relative to T)
    features['churned'] = features['churned'] = (features['frequency'] == 0).astype(int)
    
   # (
    #    (features['frequency'] == 1) | 
    #    (features['monetary_value'] < features['monetary_value'].median())
    #).astype(int)
    
    return features

if __name__ == "__main__":
    df = load_and_validate_data()
    features = feature_engineering(df)
    print(features.head())
    print(f"\nChurn rate: {features['churned'].mean():.2%}")
    print(f"High value customers: {features['is_high_value'].sum()}")


