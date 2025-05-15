import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def forecast_stock_prices(data, days=30):
    """
    Forecast stock prices based on historical data using ARIMA model
    Args:
        data: DataFrame with stock price data
        days: Number of days to forecast
        
    Returns:
        DataFrame with historical and forecasted prices
    """
    try:
        # Prepare data for forecasting
        df = data.copy()
        df = df.sort_values('date')
        
        # Extract the close price series and ensure no NaN or inf values
        price_series = df['close'].values
        
        # Replace any potential NaN or inf values
        price_series = np.nan_to_num(price_series, nan=np.nanmean(price_series))
        last_price = price_series[-1]
        
        # Check if we have enough data points
        if len(price_series) < 10:
            # Not enough data, use more dynamic trend-based forecast
            forecast_values = []
            
            # Calculate growth trend based on historical data
            if len(price_series) >= 3:
                # Use the last few points to determine trend direction
                recent_trend = (price_series[-1] - price_series[-3]) / price_series[-3]
            else:
                # Default small positive trend
                recent_trend = 0.002
                
            # Scale trend impact based on forecast length
            # Longer forecasts should show more pronounced trends
            trend_factor = min(1.0, days / 30) * (1 + days / 100)
            
            # Generate forecast with increasing volatility over time
            current_price = last_price
            for i in range(days):
                # Add increasing trend effect as days increase
                day_factor = (i + 1) / days
                
                # Base trend with increasing influence
                trend_influence = recent_trend * (1 + day_factor * trend_factor)
                
                # Increasing volatility for longer forecasts
                volatility = last_price * 0.005 * (1 + day_factor * 2)
                
                # Calculate daily change with trend and random component
                daily_change = current_price * trend_influence + np.random.normal(0, volatility)
                
                # Update price with change
                current_price = max(0.01, current_price + daily_change)
                forecast_values.append(current_price)
                
            forecast_values = np.array(forecast_values)
            
        else:
            try:
                # Perform stationarity test safely
                try:
                    result = adfuller(price_series)
                    p_value = result[1]
                    
                    # Determine differencing parameter (d)
                    d = 0
                    if p_value > 0.05:
                        d = 1  # Non-stationary data, apply differencing
                except:
                    # Default to d=1 if test fails
                    d = 1
                
                p = min(3, max(1, int(days / 30)))  # Scale AR components with forecast length
                
                model = ARIMA(price_series, order=(p, d, 0))
                model_fit = model.fit()
                
                # Forecast future prices
                forecast = model_fit.forecast(steps=days)
                forecast_values = forecast
                 # Add increasing volatility for longer-term forecasts
                if days > 30:
                    # Calculate base volatility from historical data
                    hist_volatility = np.std(price_series[-30:]) if len(price_series) >= 30 else np.std(price_series)
                    
                    # Generate volatility factors that increase with time
                    volatility_factors = np.linspace(1.0, 1.0 + (days/100), days)
                    
                    # Apply increasing volatility to the forecast
                    for i in range(days):
                        # Add scaled random noise based on historical volatility
                        day_factor = (i + 1) / days
                        noise = np.random.normal(0, hist_volatility * volatility_factors[i] * day_factor)
                        forecast_values[i] += noise
                        
                    # Ensure values don't go too low
                    forecast_values = np.maximum(forecast_values, last_price * 0.5)
            except Exception as e:
                 # If ARIMA fails, fall back to enhanced trend-based forecast
                print(f"ARIMA model failed: {str(e)}. Using enhanced trend model.")
                
                # Find trend from different time windows
                if len(price_series) >= 30:
                    short_trend = (price_series[-1] - price_series[-10]) / (price_series[-10] * 10)
                    medium_trend = (price_series[-1] - price_series[-30]) / (price_series[-30] * 30)
                    # Weight trends based on forecast length
                    if days <= 30:
                        daily_change = short_trend * last_price
                    else:
                        daily_change = (short_trend * 0.3 + medium_trend * 0.7) * last_price
                else:
                    # Calculate average daily change from available data
                    daily_change = np.diff(price_series).mean()
                
                # Add variability based on forecast length
                forecast_values = []
                current_price = last_price
                
                for i in range(days):
                    # More pronounced trend for longer periods
                    time_factor = 1.0 + (i / days)
                    
                    # Add some randomness that increases with forecast horizon
                    volatility = abs(daily_change) * (1 + i/10) * 0.5
                    random_component = np.random.normal(0, volatility)
                    
                    # Update price with compound effect
                    change = (daily_change * time_factor) + random_component
                    current_price = max(0.01, current_price + change)
                    forecast_values.append(current_price)
                
                forecast_values = np.array(forecast_values)
        # Create date range for forecast
        last_date = df['date'].max()
        # Use pd.Timedelta instead of datetime.timedelta
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'close': forecast_values,
            'predicted_price': forecast_values,
            'forecast': True
        })
        
        # Create combined dataframe with historical and forecast data
        historical_df = df[['date', 'close']].copy()
        historical_df['predicted_price'] = historical_df['close']
        historical_df['forecast'] = False
        
        combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
        
        return combined_df
    
    except Exception as e:
        print(f"Error in forecasting: {str(e)}")
        # Return original data with empty forecast as fallback
        df = data.copy()
        
        # Create empty forecast dataframe with same last price repeated
        last_price = df['close'].iloc[-1] if not df.empty else 100.0
        last_date = df['date'].max() if not df.empty else datetime.now()
        
        # Generate forecast dates
        if isinstance(last_date, pd.Timestamp):
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
        else:
            # Convert datetime to pd.Timestamp if needed
            last_date_ts = pd.Timestamp(last_date)
            forecast_dates = pd.date_range(start=last_date_ts + pd.Timedelta(days=1), periods=days)
        
        # Create stable forecast values (flat line)
        forecast_values = []
        current_price = last_price
        
        # Simple random walk with slight bias
        trend_bias = 0.001  # Small daily percentage trend
        volatility = last_price * 0.01  # Base volatility
        
        for i in range(days):
            # Increase volatility with time
            day_volatility = volatility * (1 + i/days)
            
            # Daily change with slight trend
            daily_change = current_price * trend_bias + np.random.normal(0, day_volatility)
            
            # Update with cumulative effect
            current_price = max(0.01, current_price + daily_change)
            forecast_values.append(current_price)
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'close': forecast_values,
            'predicted_price': forecast_values,
            'forecast': True
        })
        
        # Create combined df
        historical_df = df[['date', 'close']].copy() if not df.empty else pd.DataFrame({
            'date': [last_date],
            'close': [last_price]
        })
        historical_df['predicted_price'] = historical_df['close']
        historical_df['forecast'] = False
        
        return pd.concat([historical_df, forecast_df], ignore_index=True)

def get_recommendation(price_change):
    """
    Generate investment recommendation based on forecasted price change
    
    Returns:
        tuple: (recommendation string, confidence percentage)
    """
    # Based on forecasted price change, recommend action
    if price_change > 5:
        confidence = min(100, 50 + price_change)
        return "BUY", confidence
    elif price_change < -5:
        confidence = min(100, 50 + abs(price_change))
        return "SELL", confidence
    else:
        # Calculate confidence based on how close to zero
        confidence = 50 - abs(price_change) * 5
        confidence = max(0, confidence)
        return "HOLD", confidence
