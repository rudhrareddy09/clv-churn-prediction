from lifetimes.datasets import load_cdnow_summary_data_with_monetary_value

df = load_cdnow_summary_data_with_monetary_value()
print(df.head())
print(df.shape)
print(df.dtypes)