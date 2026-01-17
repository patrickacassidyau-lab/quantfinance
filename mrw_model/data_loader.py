#!/usr/bin/env python3
"""
Data loader module for fetching and processing financial market data.
"""

import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional


def fetch_market_data(
    tickers: List[str],
    start_date: str = "2010-01-01",
    end_date: str = "2024-01-01"
) -> Dict[str, pd.DataFrame]:
    """
    Fetch market data for multiple tickers using yfinance.
    
    Args:
        tickers: List of ticker symbols to fetch
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        
    Returns:
        Dictionary mapping ticker symbols to DataFrames containing market data
    """
    data = {}
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        ticker_data = yf.download(ticker, start=start_date, end=end_date)
        data[ticker] = ticker_data
    return data


def calculate_returns(data: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """
    Calculate returns from price data.
    
    Args:
        data: DataFrame containing price data
        price_col: Column name for price data (e.g., "Close", "Adj Close")
        
    Returns:
        Series containing percentage returns
    """
    # Handle both single-level and multi-level column indexes
    if isinstance(data.columns, pd.MultiIndex):
        # For multi-level columns (new yfinance format)
        price_series = data[(price_col, data.columns.get_level_values(1)[0])]
    else:
        # For single-level columns (old format or other sources)
        if price_col in data.columns:
            price_series = data[price_col]
        else:
            # Try to find a close price column
            close_cols = [col for col in data.columns if 'close' in str(col).lower()]
            if close_cols:
                price_series = data[close_cols[0]]
            else:
                raise ValueError(f"Could not find price column '{price_col}' in data")
    
    returns = price_series.pct_change().dropna()
    return returns


def calculate_rolling_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
    """
    Calculate rolling volatility.
    
    Args:
        returns: Series of returns
        window: Rolling window size
        
    Returns:
        Series containing rolling volatility
    """
    rolling_vol = returns.rolling(window=window).std()
    return rolling_vol


def get_sample_data() -> Dict[str, Dict[str, pd.Series]]:
    """
    Get sample data for the assets mentioned in the ticket.
    
    Returns:
        Dictionary containing data for each asset
    """
    tickers = ["^GSPC", "EURUSD=X", "BTC-USD", "^VIX"]
    
    # Fetch data
    market_data = fetch_market_data(tickers)
    
    # Process data
    result = {}
    for ticker in tickers:
        returns = calculate_returns(market_data[ticker])
        rolling_vol = calculate_rolling_volatility(returns)
        
        result[ticker] = {
            'returns': returns,
            'rolling_vol': rolling_vol,
            'raw_data': market_data[ticker]
        }
    
    return result