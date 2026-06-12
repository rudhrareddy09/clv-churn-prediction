import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from feature_engineering import load_and_validate_data, feature_engineering

def train_churn_model(features):
    feature_cols = ['recency', 'monetary_value', 'T']

    X = features[feature_cols]
    y = features['churned']

    X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Feature importance
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    print("\nFeature Importances:")
    print(importance.sort_values(ascending=False))
    
    return model

if __name__ == "__main__":
    df = load_and_validate_data()
    features = feature_engineering(df)
    model = train_churn_model(features)