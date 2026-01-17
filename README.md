# Multifractal Random Walk (MRW) Empirical Simulation

This project implements and simulates a multifractal random walk volatility model in Python, conducting empirical testing on equity and FX return series. The implementation includes fat-tail analysis, scaling behavior tests, and volatility clustering diagnostics, with comparisons to GARCH benchmarks.

## Project Structure

```
quant-projects/
│
├── mrw_model/
│   ├── data_loader.py          # Market data fetching and processing
│   ├── mrw_simulation.py       # MRW volatility simulation
│   ├── garch_comparison.py     # GARCH model fitting and comparison
│   ├── scaling_tests.py        # Statistical tests and scaling analysis
│   ├── main_analysis.py        # Main analysis script
│   ├── plots/                  # Generated plots (created automatically)
│   └── report.pdf              # Analysis report (to be generated)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd mrw_model

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the complete analysis

```bash
python main_analysis.py
```

This will:
1. Fetch market data for S&P 500 (^GSPC), EUR/USD (EURUSD=X), Bitcoin (BTC-USD), and VIX (^VIX)
2. Perform volatility clustering analysis
3. Test for fat tails using kurtosis analysis
4. Analyze scaling behavior across time horizons
5. Simulate MRW process and compare with GARCH models
6. Generate plots in the `plots/` directory

### Individual module usage

```python
# Import modules
from data_loader import get_sample_data
from mrw_simulation import simulate_mrw_process
from garch_comparison import fit_garch_model
from scaling_tests import test_fat_tails, analyze_scaling_behavior

# Get market data
data = get_sample_data()

# Analyze a specific asset
sp500_data = data["^GSPC"]
returns = sp500_data['returns']

# Test for fat tails
fat_tails = test_fat_tails(returns)

# Analyze scaling behavior
scaling = analyze_scaling_behavior(returns)

# Simulate MRW
mrw_data = simulate_mrw_process(N=2000)

# Fit GARCH model
garch_result, garch_vol = fit_garch_model(returns)
```

## Key Features

### 1. Empirical Results

- **Volatility Clustering**: Demonstrates persistent high-risk regimes
- **Fat Tails**: Confirms presence of extreme events consistent with Mandelbrot's hypothesis
- **Scaling Behavior**: Shows non-linear variance scaling supporting multifractal dynamics

### 2. Model Implementation

- **MRW Simulation**: Implements simplified multifractal random walk with log-normal volatility
- **GARCH Comparison**: Benchmarks against standard GARCH(1,1) model
- **Statistical Tests**: Comprehensive analysis including kurtosis, normality tests, and autocorrelation

### 3. Visualization

- Rolling volatility plots
- Variance scaling log-log plots
- MRW vs GARCH volatility comparison plots

## Empirical Findings

The analysis reproduces key stylized facts:

1. **Volatility Clustering**: Financial markets exhibit periods of elevated volatility followed by persistent high-risk regimes
2. **Fat Tails**: Empirical kurtosis significantly exceeds normal distribution (kurtosis > 3)
3. **Scaling Behavior**: Non-linear variance scaling across time horizons, departing from classical Brownian motion

## CV Line

For your CV or application:

> "Implemented and simulated a multifractal random walk volatility model in Python; conducted empirical testing on equity and FX return series, including fat-tail analysis, scaling behavior tests, and volatility clustering diagnostics; compared MRW dynamics with GARCH benchmarks."

## Requirements

- Python 3.8+
- Libraries: numpy, pandas, scipy, yfinance, arch, matplotlib

## License

[Specify your license here]

## Contact

[Your contact information]

## References

- Mandelbrot, B. (1963). "The Variation of Certain Speculative Prices"
- Müller, U. A., et al. (1990). "Multifractal Formalism for Financial Markets"
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity"

---

This project demonstrates quantitative finance skills including:
- Working with financial data
- Simulating complex stochastic processes
- Implementing econometric models
- Conducting statistical analysis
- Reasoning quantitatively about market dynamics