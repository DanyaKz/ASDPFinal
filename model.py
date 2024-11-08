import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from hurst import compute_Hc
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import r2_score

df = pd.read_excel("dataset", parse_dates=['Date'], index_col='Date')
df = df.sort_values(by = "Date" , ascending=True)

df = df[df['Full trade name'] == 'product_name']

#- рассчитать программно среднее значение ряда, медиану и дисперсию;
# mean_close = df['Close'].mean()
# # print(f"Среднее значение: {mean_close}")

# median_close = df['Close'].median()
# # print(f"Медиана: {median_close}")

# variance_close = df['Close'].var()
# print(f"Дисперсия: {variance_close}")

#- рассчитать программно значение показателя Херста на основе R/S-анализа или DFA. 
# Визуализировать и вывести промежуточные и конечные результаты. 
# Проанализировать полученное значение показателя Херста

# time_series = df['Close']

# Рассчет показателя Херста на основе R/S-анализа
def calculate_rs(data):
    n = len(data)
    rs = np.zeros(n)
    
    for i in range(1, n):
        segment = data[:i+1]
        mean = np.mean(segment)
        centered = segment - mean
        cumsum = np.cumsum(centered)
        range_ = np.max(cumsum) - np.min(cumsum)
        std_dev = np.std(segment)
        rs[i] = range_ / std_dev if std_dev != 0 else 0
        
    return rs

def calculate_hurst(rs):
    log_rs = np.log(rs + 1e-10)
    log_n = np.log(np.arange(1, len(rs) + 1))
    hurst = np.polyfit(log_n, log_rs, 1)[0]
    
    return hurst

rs = calculate_rs(df['Avg_price_per_unit'])
hurst = calculate_hurst(rs)

print("Показатель Херста на основе R/S-анализа:", hurst)

# Прогнозирование временного ряда

def simple_exponential_smoothing(series, alpha):
    result = [series[0]] # pervoe znach ryada bez izmeneniya
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

def holt_linear_trend(series, alpha, beta):
    level = [series[0]]
    trend = [series[1] - series[0]]
    
    for n in range(1, len(series)):
        level.append(alpha * series[n] + (1 - alpha) * (level[n-1] + trend[n-1]))
        trend.append(beta * (level[n] - level[n-1]) + (1 - beta) * trend[n-1])
    
    return [level[i] + trend[i] * (i+1) for i in range(len(series))]


def holt_winters_additive(series, alpha, beta, gamma, seasonality_period):
    series_length = len(series)
    level, trend, seasonals = [series[0]], [series[1] - series[0]], [(series[i] - series[0]) for i in range(seasonality_period)]
    
    forecast = [level[0] + trend[0] + seasonals[i % seasonality_period] for i in range(series_length)]
    for n in range(1, series_length):
        if n < seasonality_period:
            level.append(alpha * (series[n] - seasonals[n]) + (1 - alpha) * (level[n-1] + trend[n-1]))
            trend.append(beta * (level[n] - level[n-1]) + (1 - beta) * trend[n-1])
            seasonals[n] = gamma * (series[n] - level[n]) + (1 - gamma) * seasonals[n]
            forecast[n] = level[n] + trend[n] + seasonals[n % seasonality_period]
        else:
            m = n % seasonality_period
            level.append(alpha * (series[n] - seasonals[m]) + (1 - alpha) * (level[n-1] + trend[n-1]))
            trend.append(beta * (level[n] - level[n-1]) + (1 - beta) * trend[n-1])
            seasonals[m] = gamma * (series[n] - level[n]) + (1 - gamma) * seasonals[m]
            forecast[n] = level[n] + trend[n] + seasonals[m]
    return forecast


def triple_exponential_smoothing(series, alpha, beta, gamma, seasonality_period):
    season = [series[i] for i in range(seasonality_period)]
    season_avg = sum(season) / float(seasonality_period)
    season_adj = [season[i] - season_avg for i in range(seasonality_period)]
    
    smooth = [series[0]]  
    trend = [series[1] - series[0]] 
    season_comp = season_adj * (len(series) // seasonality_period + 1)  
    forecasts = [smooth[0] + trend[0] + season_comp[0]]  
    
    for i in range(1, len(series)):
        if i < seasonality_period:
            smooth_val = alpha * (series[i] - season_adj[i]) + (1 - alpha) * (smooth[i - 1] + trend[i - 1])
        else:
            smooth_val = alpha * (series[i] - season_comp[i - seasonality_period]) + (1 - alpha) * (smooth[i - 1] + trend[i - 1])
        
        trend_val = beta * (smooth_val - smooth[i - 1]) + (1 - beta) * trend[i - 1]
        season_comp_val = gamma * (series[i] - smooth_val) + (1 - gamma) * season_comp[i - seasonality_period]
        
        smooth.append(smooth_val)
        trend.append(trend_val)
        season_comp[i % seasonality_period] = season_comp_val  # Correctly update season_comp
        
        forecast = smooth[i] + trend[i] + season_comp[i % seasonality_period]
        forecasts.append(forecast)
    
    return forecasts


# Параметры моделей
alpha_brown = 0.2
alpha_holt = 0.2
beta_holt = 0.1
alpha_holt_winters = 0.2
beta_holt_winters = 0.1
gamma_holt_winters = 0.1
alpha_triple = 0.2
beta_triple = 0.1
gamma_triple = 0.1
seasonality_period = 4

plt.figure(figsize=(12, 16))


from sklearn.metrics import r2_score



# Рассчет показателя Херста на основе R/S-анализа
# plt.subplot(2, 2, 1)
# plt.plot(rs, label='R/S статистика')
# plt.title("R/S анализ")
# plt.xlabel("Window Size")
# plt.ylabel("R/S")
# plt.legend()


# Прогнозы моделей
# plt.subplot(2, 2, 1)
# forecast_brown_custom = simple_exponential_smoothing(df['Close'].values, alpha_brown)[-50:]
# plt.plot(df.index, df['Close'], label='Actual')
# plt.plot(df[-50:].index, forecast_brown_custom, label='Brown', linestyle='--')
# plt.title("Прогноз модели Брауна")
# plt.xlabel("Дата")
# plt.ylabel("Цена закрытия")
# plt.legend()

# plt.subplot(2, 2, 2)
# forecast_holt_custom = holt_linear_trend(df['Close'].values, alpha_holt, beta_holt)[-50:]
# plt.plot(df.index, df['Close'], label='Actual')
# plt.plot(df[-50:].index, forecast_holt_custom, label='Holt', linestyle='--')
# plt.title("Прогноз модели Хольта")
# plt.xlabel("Дата")
# plt.ylabel("Цена закрытия")
# plt.legend()

# plt.subplot(2, 2, 3)
# forecast_holt_winters_custom = holt_winters_additive(df['Close'].values, alpha_holt_winters, beta_holt_winters, gamma_holt_winters, seasonality_period)[-50:]
# plt.plot(df.index, df['Close'], label='Actual')
# plt.plot(df[-50:].index, forecast_holt_winters_custom, label='Holt-Winters', linestyle='--')
# plt.title("Прогноз модели Хольта-Винтерса")
# plt.xlabel("Дата")
# plt.ylabel("Цена закрытия")
# plt.legend()

# plt.subplot(2, 2, 4)
forecast_triple_custom = triple_exponential_smoothing(df['Avg_price_per_unit'].values, alpha_triple, beta_triple, gamma_triple, seasonality_period)[-30:]
plt.plot(df.index, df['Avg_price_per_unit'], label='Actual')
plt.plot(df[-30:].index, forecast_triple_custom, label='Triple', linestyle='--')
plt.title("Прогноз модели Тригга-Лича")
plt.xlabel("Дата")
plt.ylabel("Цена закрытия")
plt.legend()


# Для прогноза модели Брауна
# r2_brown = r2_score(df['Close'].values[-50:], forecast_brown_custom)
# print(f"R^2 Брауна: {r2_brown}")

# # Для прогноза модели Хольта
# r2_holt = r2_score(df['Close'].values[-50:], forecast_holt_custom)
# print(f"R^2 Хольта: {r2_holt}")

# # Для прогноза модели Хольта-Винтерса
# r2_holt_winters = r2_score(df['Close'].values[-50:], forecast_holt_winters_custom)
# print(f"R^2 Хольта-Винтерса: {r2_holt_winters}")

# Для прогноза модели Тригга-Лича
r2_triple = r2_score(df['Avg_price_per_unit'].values[-30:], forecast_triple_custom)
print(f"R^2 Тригга-Лича: {r2_triple}")



plt.tight_layout()
plt.show()
