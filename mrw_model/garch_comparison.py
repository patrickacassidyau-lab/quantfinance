#!/usr/bin/env python3
"""
GARCH model comparison module.
"""

import pandas as pd
import numpy as np
from arch import arch_model
from typing import Tuple, Dict, Optional, List


def fit_garch_model(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    vol: str = 'Garch'
) -> Tuple:
    """
    Fit GARCH model to returns data.
    
    Args:
        returns: Series of returns
        p: Order of ARCH term
        q: Order of GARCH term
        vol: Volatility model specification
        
    Returns:
        Tuple containing (model_result, conditional_volatility)
    """
    # Convert returns to percentage for arch_model
    returns_pct = returns * 100
    
    # Fit GARCH model
    garch = arch_model(returns_pct, vol=vol, p=p, q=q)
    res = garch.fit(disp='off')
    
    # Extract conditional volatility
    garch_vol = res.conditional_volatility
    
    return res, garch_vol


def compare_volatility_models(
    mrw_volatility: np.ndarray,
    garch_volatility: pd.Series,
    n_points: int = 500
) -> Dict:
    """
    Compare MRW and GARCH volatility models.
    
    Args:
        mrw_volatility: MRW volatility array
        garch_volatility: GARCH conditional volatility series
        n_points: Number of points to compare
        
    Returns:
        Dictionary containing comparison results
    """
    # Ensure we don't exceed array bounds
    n_points = min(n_points, len(mrw_volatility), len(garch_volatility))
    
    # Normalize GARCH volatility to match MRW scale
    garch_vol_normalized = garch_volatility.iloc[:n_points] / 100
    mrw_vol_normalized = mrw_volatility[:n_points]
    
    # Calculate statistics
    mrw_mean = np.mean(mrw_vol_normalized)
    garch_mean = np.mean(garch_vol_normalized)
    mrw_std = np.std(mrw_vol_normalized)
    garch_std = np.std(garch_vol_normalized)
    
    # Calculate correlation
    correlation = np.corrcoef(mrw_vol_normalized, garch_vol_normalized)[0, 1]
    
    return {
        'mrw_volatility': mrw_vol_normalized,
        'garch_volatility': garch_vol_normalized,
        'mrw_mean': mrw_mean,
        'garch_mean': garch_mean,
        'mrw_std': mrw_std,
        'garch_std': garch_std,
        'correlation': correlation,
        'n_points': n_points
    }


def analyze_model_fit(
    returns: pd.Series,
    model_result
) -> Dict:
    """
    Analyze GARCH model fit statistics.
    
    Args:
        returns: Original returns series
        model_result: GARCH model result object
        
    Returns:
        Dictionary containing fit statistics
    """
    # Get model parameters
    params = model_result.params
    
    # Calculate AIC and BIC
    aic = model_result.aic
    bic = model_result.bic
    
    # Get log likelihood
    log_likelihood = model_result.loglikelihood
    
    return {
        'parameters': dict(params),
        'aic': aic,
        'bic': bic,
        'log_likelihood': log_likelihood,
        'residuals_std': np.std(model_result.resid)
    }