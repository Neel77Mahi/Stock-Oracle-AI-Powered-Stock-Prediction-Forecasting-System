import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import HistGradientBoostingRegressor
import streamlit as st

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Cache feature engineering to avoid recomputation
@st.cache_data(show_spinner=False)
def prepare_features(data, lookback):
    df = data.copy().sort_values('date')
    # lagged close values
    for lag in range(1, lookback + 1):
        df[f'lag_{lag}'] = df['close'].shift(lag)
    # rolling volatility and momentum
    df['rolling_std'] = df['close'].rolling(window=lookback).std().fillna(0)
    df['momentum'] = df['close'] - df['close'].shift(lookback)
    # keep volume and sentiment
    features = [f'lag_{lag}' for lag in range(1, lookback + 1)] + ['volume', 'sentiment_value', 'rolling_std', 'momentum']
    df.dropna(inplace=True)
    return df, features

# Cache model training to reuse across runs
@st.cache_resource(show_spinner=False)
def train_ml_model(X, y, max_iter=100):
    # Use fast histogram-based gradient boosting
    model = HistGradientBoostingRegressor(max_iter=max_iter)
    model.fit(X, y)
    return model

@st.cache_data(show_spinner=False)
def holt_winters_forecast(price_series, days):
    """
    Fast Holt's linear trend exponential smoothing.
    """
    model = ExponentialSmoothing(
        price_series,
        trend='add',
        seasonal=None,
        initialization_method='estimated'
    ).fit(optimized=True)
    return np.array(model.forecast(days))


def ml_forecast(data, days, lookback=30, max_iter=50):
    """
    Forecast via gradient boosting on lagged features.
    """
    if len(data) < lookback + 1:
        return None

    df_feat, features = prepare_features(data, lookback)
    if df_feat.empty:
        return None

    X = df_feat[features].values
    y = df_feat['close'].values
    model = train_ml_model(X, y, max_iter)

    # iterative forecasting
    last_window = data.sort_values('date').tail(lookback)
    last_values = last_window['close'].tolist()
    last_vol = last_window['volume'].iloc[-1]
    last_sent = last_window['sentiment_value'].iloc[-1]

    forecast = []
    for i in range(days):
        row = last_values[-lookback:] + [last_vol, last_sent,
                                          np.std(last_values),
                                          last_values[-1] - last_values[0]]
        pred = model.predict(np.array(row).reshape(1, -1))[0]
        forecast.append(pred)
        last_values.append(pred)
    return np.array(forecast)


def forecast_stock_prices(data, days=30):
    """
    Fast ensemble forecast: Holt–Winters + ML.
    """
    df = data.copy().sort_values('date')
    price_series = np.nan_to_num(df['close'].values, nan=np.nanmean(df['close'].values))
    last_price = price_series[-1]

    # Generate HW and ML forecasts in parallel
    hw = holt_winters_forecast(price_series, days)
    ml = ml_forecast(df, days)

    # Combine with dynamic weight: more weight to HW early, ML later
    weights = np.linspace(0.7, 0.3, days)
    if ml is not None:
        forecast_values = hw * weights + ml * (1 - weights)
    else:
        forecast_values = hw

    # Vectorized noise injection
    hist_vol = np.std(np.diff(price_series))
    noise_scale = hist_vol * (1 + np.arange(1, days + 1) / days)
    forecast_values = np.maximum(0.01, forecast_values + np.random.randn(days) * noise_scale)

    # Build DataFrame
    last_date = df['date'].max()
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'close': forecast_values,
        'predicted_price': forecast_values,
        'forecast': True
    })
    hist_df = df[['date', 'close']].copy()
    hist_df['predicted_price'] = hist_df['close']
    hist_df['forecast'] = False
    return pd.concat([hist_df, forecast_df], ignore_index=True)


def get_recommendation(price_change):
    """
    BUY/HOLD/SELL recommendation based on % change.
    """
    if price_change > 5:
        return "BUY", min(100, 50 + price_change)
    if price_change < -5:
        return "SELL", min(100, 50 + abs(price_change))
    return "HOLD", max(0, 50 - abs(price_change) * 5)
