#!/usr/bin/env python3
"""
Multifractal Random Walk (MRW) simulation module.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List


def simulate_mrw_volatility(
    N: int = 2000,
    lambda2: float = 0.02,
    T: float = 1000.0,
    dt: float = 1.0,
    sigma0: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate MRW volatility process.
    
    Args:
        N: Number of time steps
        lambda2: Variance parameter for the log-normal distribution
        T: Time horizon parameter
        dt: Time increment
        sigma0: Base volatility level
        
    Returns:
        Tuple containing (omega, sigma, returns_mrw)
    """
    # Create covariance matrix
    cov = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            cov[i, j] = lambda2 * np.log(T / (abs(i - j) + dt))
    
    # Generate log-normal volatility process
    omega = np.random.multivariate_normal(mean=np.zeros(N), cov=cov)
    sigma = sigma0 * np.exp(omega)
    
    # Generate returns
    eps = np.random.normal(0, 1, N)
    returns_mrw = sigma * eps
    
    return omega, sigma, returns_mrw


def simulate_mrw_process(
    N: int = 2000,
    lambda2: float = 0.02,
    T: float = 1000.0,
    dt: float = 1.0,
    sigma0: float = 0.01
) -> pd.DataFrame:
    """
    Simulate complete MRW process and return as DataFrame.
    
    Args:
        N: Number of time steps
        lambda2: Variance parameter
        T: Time horizon parameter
        dt: Time increment
        sigma0: Base volatility level
        
    Returns:
        DataFrame containing the simulated MRW process
    """
    omega, sigma, returns_mrw = simulate_mrw_volatility(N, lambda2, T, dt, sigma0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'omega': omega,
        'sigma': sigma,
        'returns': returns_mrw
    })
    
    return df


def calculate_scaling_variance(
    returns: pd.Series,
    scales: Optional[List[int]] = None
) -> Tuple[List[int], List[float]]:
    """
    Calculate variance scaling across different time scales.
    
    Args:
        returns: Series of returns
        scales: List of time scales to analyze
        
    Returns:
        Tuple containing (scales, variances)
    """
    if scales is None:
        scales = [1, 5, 20, 60, 120]
    
    vars = []
    for s in scales:
        agg = returns.rolling(s).sum().dropna()
        vars.append(np.var(agg))
    
    return scales, vars