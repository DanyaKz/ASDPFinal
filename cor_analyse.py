import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


df_currency = pd.read_excel('currency_dataset')
df_products = pd.read_excel('main_dataset')

df_currency['Date'] = pd.to_datetime(df_currency['Date'])
df_products['Date'] = pd.to_datetime(df_products['Date'])


df_currency['Price'] = pd.to_numeric(df_currency['Price'], errors='coerce')
df_products['Avg price per unit'] = pd.to_numeric(df_products['Avg price per unit'], errors='coerce')


df = pd.merge(df_products, df_currency, on='Date', how='left')

df.dropna(subset=['Price', 'Avg price per unit', 'Country', 'Currency'], inplace=True)

df_cur = df[(df['Full trade name'] == 
    'prod_name') & (df['Currency'] == 'USD')].copy()


df_cur['Price'] = pd.to_numeric(df_cur['Price'], errors='coerce')
df_cur['Avg price per unit'] = pd.to_numeric(df_cur['Avg price per unit'], errors='coerce')


df_cur.dropna(subset=['Price', 'Avg price per unit'], inplace=True)


X = df_cur[['Price']]  
y = df_cur['Avg price per unit']  


X_sm = sm.add_constant(X)


model_usd = sm.OLS(y, X_sm)
results_usd = model_usd.fit()
print(results_usd.summary())


corr_usd = df_cur['Price'].corr(df_cur['Avg price per unit'])
print(f'Коэффициент корреляции между курсом RUB и средней ценой товаров Kazakhstan: {corr_usd}')
