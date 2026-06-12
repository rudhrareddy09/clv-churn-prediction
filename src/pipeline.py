import pandas as pd
from feature_engineering import load_and_validate_data, feature_engineering
from clv_model import fit_bgnbd_model, fit_gamma_gamma_model, predict_clv

def run_pipeline():
    print("=== CLV Pipeline Starting ===")

    #Step 1: Load and Validate
    df = load_and_validate_data()

    #Step 2: Feature Engineering
    features = feature_engineering(df)

    #Step 3: Fit Models
    bgf = fit_bgnbd_model(df)
    ggf = fit_gamma_gamma_model(df)

    #Step 4: Predict CLV
    results = predict_clv(df, bgf, ggf)

    #Step 5: Segment customers
    results['segment'] = pd.cut(
        results['predicted_clv'],
        bins=[0, 50, 200, 500, float('inf')],
        labels=['Low', 'Medium', 'High', 'VIP']
    )

    #Step 6: Answer the 5 business questions
    print("\n--- Business Question 1: Avg 12 months CLV ---")
    print(f"${results['predicted_clv'].mean():.2f}")

    print("\n--- Business Question 2: Customer Segments ---")
    print(results['segment'].value_counts())
    
    print("\n--- Business Question 3: Top 10 VIP Customers ---")
    print(results.nlargest(10, 'predicted_clv')[['customer_id','predicted_clv','frequency','monetary_value']])
    
    print("\n--- Business Question 4: Churn Risk (zero repeat purchases) ---")
    churn_count = (df['frequency'] == 0).sum()
    print(f"{churn_count} customers ({churn_count/len(df):.1%}) never made a repeat purchase")
    
    print("\n--- Business Question 5: High-frequency high-value customers ---")
    top_segment = results[results['segment'] == 'VIP']
    print(f"{len(top_segment)} VIP customers averaging ${top_segment['predicted_clv'].mean():.2f} CLV")
    
    # Step 7: Write output
    results.to_csv('outputs/clv_predictions.csv', index=False)
    print("\n✓ Output written to outputs/clv_predictions.csv")
    print("=== Pipeline Complete ===")

if __name__ == "__main__":
    run_pipeline()  