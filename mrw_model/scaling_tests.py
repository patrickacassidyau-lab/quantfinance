#!/usr/bin/env python3
"""
Scaling behavior and statistical tests module.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, jarque_bera, normaltest
from typing import Dict, Tuple, List, Optional


def test_fat_tails(returns: pd.Series) -> Dict:
    """
    Test for fat tails using kurtosis and normality tests.
    
    Args:
        returns: Series of returns
        
    Returns:
        Dictionary containing test results
    """
    # Calculate kurtosis
    k = kurtosis(returns)
    excess_kurtosis = k - 3  # Normal distribution has kurtosis of 3
    
    # Perform normality tests
    jb_stat, jb_pvalue = jarque_bera(returns)
    dn_stat, dn_pvalue = normaltest(returns)
    
    return {
        'kurtosis': k,
        'excess_kurtosis': excess_kurtosis,
        'jarque_bera': {
            'statistic': jb_stat,
            'p_value': jb_pvalue
        },
        'd_agostino': {
            'statistic': dn_stat,
            'p_value': dn_pvalue
        },
        'is_normal': jb_pvalue > 0.05 and dn_pvalue > 0.05
    }


def analyze_volatility_clustering(
    returns: pd.Series,
    window: int = 30
) -> Dict:
    """
    Analyze volatility clustering in returns.
    
    Args:
        returns: Series of returns
        window: Rolling window size for volatility calculation
        
    Returns:
        Dictionary containing volatility clustering analysis
    """
    # Calculate rolling volatility
    rolling_vol = returns.rolling(window=window).std()
    
    # Calculate autocorrelation of squared returns (proxy for volatility)
    squared_returns = returns ** 2
    autocorr_lag1 = squared_returns.autocorr(lag=1)
    autocorr_lag5 = squared_returns.autocorr(lag=5)
    autocorr_lag10 = squared_returns.autocorr(lag=10)
    
    # Calculate rolling volatility statistics
    vol_mean = rolling_vol.mean()
    vol_std = rolling_vol.std()
    vol_cv = vol_std / vol_mean  # Coefficient of variation
    
    return {
        'rolling_volatility': rolling_vol,
        'autocorrelation': {
            'lag_1': autocorr_lag1,
            'lag_5': autocorr_lag5,
            'lag_10': autocorr_lag10
        },
        'volatility_stats': {
            'mean': vol_mean,
            'std': vol_std,
            'coefficient_of_variation': vol_cv
        },
        'has_clustering': abs(autocorr_lag1) > 0.1  # Simple heuristic
    }


def analyze_scaling_behavior(
    returns: pd.Series,
    scales: Optional[List[int]] = None
) -> Dict:
    """
    Analyze scaling behavior of returns.
    
    Args:
        returns: Series of returns
        scales: List of time scales to analyze
        
    Returns:
        Dictionary containing scaling analysis results
    """
    if scales is None:
        scales = [1, 5, 20, 60, 120]
    
    # Calculate scaling variance
    vars = []
    for s in scales:
        agg = returns.rolling(s).sum().dropna()
        vars.append(np.var(agg))
    
    # Calculate log-log relationship
    log_scales = np.log(scales)
    log_vars = np.log(vars)
    
    # Fit linear regression to log-log data
    A = np.vstack([log_scales, np.ones(len(log_scales))]).T
    slope, intercept = np.linalg.lstsq(A, log_vars, rcond=None)[0]
    
    # Calculate R-squared
    log_vars_pred = slope * log_scales + intercept
    ss_res = np.sum((log_vars - log_vars_pred) ** 2)
    ss_tot = np.sum((log_vars - np.mean(log_vars)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    return {
        'scales': scales,
        'variances': vars,
        'log_scales': log_scales,
        'log_vars': log_vars,
        'regression': {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared
        },
        'is_linear': r_squared > 0.8  # Simple heuristic for linearity
    }


def compare_scaling_behaviors(
    empirical_returns: pd.Series,
    simulated_returns: pd.Series,
    scales: Optional[List[int]] = None
) -> Dict:
    """
    Compare scaling behavior between empirical and simulated returns.
    
    Args:
        empirical_returns: Series of empirical returns
        simulated_returns: Series of simulated returns
        scales: List of time scales to analyze
        
    Returns:
        Dictionary containing comparison results
    """
    if scales is None:
        scales = [1, 5, 20, 60, 120]
    
    # Analyze both series
    empirical_results = analyze_scaling_behavior(empirical_returns, scales)
    simulated_results = analyze_scaling_behavior(simulated_returns, scales)
    
    # Compare regression parameters
    slope_diff = abs(empirical_results['regression']['slope'] - simulated_results['regression']['slope'])
    intercept_diff = abs(empirical_results['regression']['intercept'] - simulated_results['regression']['intercept'])
    
    return {
        'empirical': empirical_results,
        'simulated': simulated_results,
        'comparison': {
            'slope_difference': slope_diff,
            'intercept_difference': intercept_diff,
            'similar_scaling': slope_diff < 0.5  # Simple heuristic
        }
    }