#!/usr/bin/env python3
"""
Main analysis script demonstrating MRW model implementation and empirical testing.
This script reproduces the analysis described in the ticket.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from typing import List

# Import local modules
from data_loader import get_sample_data
from mrw_simulation import simulate_mrw_process, calculate_scaling_variance
from garch_comparison import fit_garch_model, compare_volatility_models
from scaling_tests import (
    test_fat_tails,
    analyze_volatility_clustering,
    analyze_scaling_behavior
)


def create_plots_directory():
    """Create plots directory if it doesn't exist."""
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir


def plot_rolling_volatility(rolling_vol: pd.Series, ticker: str, plots_dir: str):
    """Plot rolling volatility for a given asset."""
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_vol)
    plt.title(f"Rolling 30-day Volatility - {ticker}")
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(plots_dir, f"rolling_vol_{ticker.replace('=', '').replace('^', '').replace('-', '_')}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved rolling volatility plot: {plot_path}")


def plot_scaling_behavior(scales: List, vars: List, ticker: str, plots_dir: str):
    """Plot variance scaling behavior."""
    plt.figure(figsize=(12, 6))
    plt.loglog(scales, vars, 'o-')
    plt.title(f"Variance Scaling Across Time Scales - {ticker}")
    plt.xlabel("Time Scale (days)")
    plt.ylabel("Variance")
    plt.grid(True, alpha=0.3, which="both")
    
    # Save plot
    plot_path = os.path.join(plots_dir, f"scaling_{ticker.replace('=', '').replace('^', '').replace('-', '_')}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved scaling behavior plot: {plot_path}")


def plot_volatility_comparison(mrw_vol: np.ndarray, garch_vol: pd.Series, ticker: str, plots_dir: str):
    """Plot comparison between MRW and GARCH volatility."""
    n_points = min(500, len(mrw_vol), len(garch_vol))
    
    plt.figure(figsize=(12, 6))
    plt.plot(mrw_vol[:n_points], label="MRW volatility", alpha=0.7)
    plt.plot(garch_vol.iloc[:n_points] / 100, label="GARCH volatility", alpha=0.7)
    plt.title(f"Volatility Model Comparison - {ticker}")
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(plots_dir, f"vol_comparison_{ticker.replace('=', '').replace('^', '').replace('-', '_')}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved volatility comparison plot: {plot_path}")


def main():
    """Main analysis function."""
    print("Starting MRW empirical analysis...")
    
    # Create plots directory
    plots_dir = create_plots_directory()
    
    # Step 1: Get market data
    print("\n=== Step 1: Fetching Market Data ===")
    data = get_sample_data()
    
    # Step 2: Analyze each asset
    for ticker, asset_data in data.items():
        print(f"\n=== Analyzing {ticker} ===")
        
        returns = asset_data['returns']
        rolling_vol = asset_data['rolling_vol']
        
        # Step 2: Show volatility clustering
        print(f"\n--- Volatility Clustering Analysis ---")
        clustering_results = analyze_volatility_clustering(returns)
        print(f"Volatility clustering detected: {clustering_results['has_clustering']}")
        print(f"Autocorrelation (lag 1): {clustering_results['autocorrelation']['lag_1']:.4f}")
        print(f"Autocorrelation (lag 5): {clustering_results['autocorrelation']['lag_5']:.4f}")
        
        # Plot rolling volatility
        plot_rolling_volatility(rolling_vol, ticker, plots_dir)
        
        # Step 3: Fat tails test
        print(f"\n--- Fat Tails Analysis ---")
        fat_tails_results = test_fat_tails(returns)
        print(f"Kurtosis: {fat_tails_results['kurtosis']:.4f}")
        print(f"Excess kurtosis: {fat_tails_results['excess_kurtosis']:.4f}")
        print(f"Normal distribution kurtosis: 3.0")
        print(f"Fat tails detected: {fat_tails_results['excess_kurtosis'] > 0}")
        
        # Step 4: Scaling behavior
        print(f"\n--- Scaling Behavior Analysis ---")
        scales = [1, 5, 20, 60, 120]
        scaling_results = analyze_scaling_behavior(returns, scales)
        
        print(f"Scaling regression slope: {scaling_results['regression']['slope']:.4f}")
        print(f"Scaling regression R²: {scaling_results['regression']['r_squared']:.4f}")
        print(f"Linear scaling detected: {scaling_results['is_linear']}")
        
        # Plot scaling behavior
        plot_scaling_behavior(scales, scaling_results['variances'], ticker, plots_dir)
        
        # Step 5: Simulate MRW and compare with GARCH
        print(f"\n--- MRW Simulation and GARCH Comparison ---")
        
        # Simulate MRW
        mrw_data = simulate_mrw_process(N=len(returns))
        mrw_volatility = mrw_data['sigma'].values
        
        # Fit GARCH model
        try:
            garch_result, garch_volatility = fit_garch_model(returns)
            print(f"GARCH model fitted successfully")
            print(f"GARCH AIC: {garch_result.aic:.2f}")
            print(f"GARCH BIC: {garch_result.bic:.2f}")
            
            # Compare models
            comparison_results = compare_volatility_models(mrw_volatility, garch_volatility)
            print(f"MRW volatility mean: {comparison_results['mrw_mean']:.6f}")
            print(f"GARCH volatility mean: {comparison_results['garch_mean']:.6f}")
            print(f"Correlation between MRW and GARCH: {comparison_results['correlation']:.4f}")
            
            # Plot comparison
            plot_volatility_comparison(mrw_volatility, garch_volatility, ticker, plots_dir)
            
        except Exception as e:
            print(f"GARCH model fitting failed: {e}")
        
        print(f"\n--- Summary for {ticker} ---")
        print(f"✓ Volatility clustering: {clustering_results['has_clustering']}")
        print(f"✓ Fat tails (excess kurtosis): {fat_tails_results['excess_kurtosis']:.4f}")
        print(f"✓ Scaling behavior R²: {scaling_results['regression']['r_squared']:.4f}")
        print(f"Analysis complete for {ticker}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run main analysis
    main()
    
    print("\n=== All analyses complete! ===")
    print("Plots saved in the 'plots' directory")
    print("You can now use these results for your paper and CV")