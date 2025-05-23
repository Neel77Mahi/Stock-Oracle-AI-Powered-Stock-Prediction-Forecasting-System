import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_and_process_data(file_path):
    """
    Load stock data from CSV and process it for analysis
    """
    # Load data
    data = pd.read_csv(file_path)
    
    # Convert date column to datetime
    data['date'] = pd.to_datetime(data['date'])
    
    # Sort by date
    data = data.sort_values(['ticker', 'date'])
    
    # Extract numeric price from text
    data['price'] = data['original'].str.extract(r'higher at (\d+\.\d+)').astype(float)
    
    # Extract price from alternative format if the above fails
    price_mask = data['price'].isna()
    if price_mask.any():
        # Try alternative extraction patterns
        alt_prices = data.loc[price_mask, 'original'].str.extract(r'dropping to (\d+\.\d+)').astype(float)
        data.loc[price_mask, 'price'] = alt_prices
    
    # Extract volume from text
    data['volume'] = data['original'].str.extract(r'volume (?:of|surging at) (\d+)').astype(float)
    
    # Create additional columns for analysis
    data['close'] = data['price']  # Rename price to close for clarity
    
    # Handle missing price values by forward-filling and then backward-filling
    # This ensures there are no NaN values which would cause forecasting errors
    for ticker in data['ticker'].unique():
        ticker_mask = data['ticker'] == ticker
        data.loc[ticker_mask, 'close'] = data.loc[ticker_mask, 'close'].fillna(method='ffill').fillna(method='bfill')
    
    # Calculate daily returns
    data['daily_return'] = data.groupby('ticker')['close'].pct_change() * 100
    
    # Replace any NaN in daily_return with 0 (happens on first day of data)
    data['daily_return'] = data['daily_return'].fillna(0)
    
    # Convert sentiment labels to binary (bullish=1, bearish=-1, neutral=0)
    sentiment_map = {
        'bullish': 1,
        'bearish': -1,
        'neutral': 0
    }
    data['sentiment_value'] = data['senti_label'].map(sentiment_map)
    
    # Final check to remove any remaining NaN values
    data = data.fillna({
        'volume': data['volume'].median(),
        'sentiment_value': 0
    })
    
    return data

def get_unique_tickers(data):
    """
    Get list of unique tickers in the dataset
    """
    return sorted(data['ticker'].unique())

def get_ticker_data(data, ticker):
    """
    Filter data for a specific ticker
    """
    return data[data['ticker'] == ticker].copy()

def generate_extended_dates(last_date, days=30):
    """
    Generate date range for forecasting
    """
    # Convert to pandas Timestamp if it's not already
    if not isinstance(last_date, pd.Timestamp):
        last_date = pd.Timestamp(last_date)
    
    # Use the proper method to add days to a timestamp
    next_day = last_date + pd.Timedelta(days=1)
    date_range = pd.date_range(start=next_day, periods=days)
    return date_range
