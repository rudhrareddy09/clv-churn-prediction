import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.datasets import load_cdnow_summary_data_with_monetary_value
from feature_engineering import load_and_validate_data, feature_engineering

def fit_bgnbd_model(df):
    bgf = BetaGeoFitter (penalizer_coef = 0.01)
    bgf.fit(df['frequency'], df['recency'], df['T'])
    print(bgf.summary)
    return bgf

def fit_gamma_gamma_model(df):
    repeat_buyers = df[df['frequency'] > 0]
    ggf = GammaGammaFitter (penalizer_coef = 0.01)
    ggf.fit(repeat_buyers['frequency'], repeat_buyers['monetary_value'])
    print(ggf.summary)
    return ggf

def predict_clv(df, bgf, ggf, months=12):
     repeat_buyers = df[df['frequency'] > 0].copy()
     repeat_buyers['predicted_clv'] = ggf.customer_lifetime_value(
        bgf,
        repeat_buyers['frequency'],
        repeat_buyers['recency'],
        repeat_buyers['T'],
        repeat_buyers['monetary_value'],
        time=months,
        freq='D'
    )
     return repeat_buyers

if __name__ == "__main__":
    df = load_and_validate_data()
    bgf = fit_bgnbd_model(df)
    ggf = fit_gamma_gamma_model(df)
    results = predict_clv(df, bgf, ggf)
    print(results[['customer_id','frequency','monetary_value','predicted_clv']].head(10))
    print(f"\nAvg predicted 12-month CLV: ${results['predicted_clv'].mean():.2f}")
