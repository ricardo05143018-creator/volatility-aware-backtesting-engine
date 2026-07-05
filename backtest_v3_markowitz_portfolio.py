"""
v3: Portfolio tryout using Markowitz.

I tried combining multiple assets together (stocks + BTC + gold) instead of just
trading a single market. Using the Markowitz variance/covariance method to see
how the weight distribution behaves.
Just a simple experiment on 2025 data. No short selling allowed.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize

# Network proxy setup if needed
# os.environ['http_proxy'] = 'http://127.0.0.1:7892'
# os.environ['https_proxy'] = 'http://127.0.0.1:7892'

def portfolio_performance(weights, returns, cov_matrix):
    """Calculates annualized return and volatility for a set of weights."""
    port_return = np.sum(returns * weights)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_volatility

def minimize_volatility(weights, returns, cov_matrix):
    """Objective function to minimize portfolio risk."""
    return portfolio_performance(weights, returns, cov_matrix)[1]

def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.0):
    """Objective function to maximize Sharpe (by minimizing negative Sharpe)."""
    p_ret, p_vol = portfolio_performance(weights, returns, cov_matrix)
    if p_vol == 0:
        return 0
    return -(p_ret - risk_free_rate) / p_vol


if __name__ == "__main__":

    tickers = ["AAPL", "TSLA", "BTC-USD", "GLD"]
    csv_filename = "v3_market_data_2025.csv"

    if os.path.exists(csv_filename):
        print(f"Found local cache [{csv_filename}], loading directly...")
        raw_data = pd.read_csv(csv_filename, index_col=0, parse_dates=True)
    else:
        print("No local cache found. Downloading 2025 data from yfinance...")
        try:
            raw_data = yf.download(tickers, start="2025-01-01", end="2025-12-31", progress=False)['Close']

            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns = raw_data.columns.droplevel(1)

            if not raw_data.dropna(how='all').empty:
                raw_data.to_csv(csv_filename)
                print(f"  Successfully downloaded and cached to {csv_filename}")
            else:
                raise ValueError("Downloaded data is empty. Might be rate limited.")
        except Exception as e:
            print(f"\n  Error downloading data: {e}")
            raw_data = pd.DataFrame()

    if raw_data.empty:
        print("Warning: No asset data available. Exiting.")
        exit()

    # returns and correlation
    daily_returns = raw_data.pct_change().dropna()
    correlation_matrix = daily_returns.corr()

    print("\n--- Correlation matrix (2025) ---")
    print(correlation_matrix.round(3))

    # plot matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title("Asset Correlation Matrix (2025)")
    plt.tight_layout()

    num_assets = len(daily_returns.columns)
    assets = list(daily_returns.columns)
    num_trading_days = 252

    annual_returns = daily_returns.mean() * num_trading_days
    annual_covariance = daily_returns.cov() * num_trading_days

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1.0 / num_assets]

    print("\nrunning gm portfolio...")
    gmv_result = minimize(
        minimize_volatility,
        initial_weights,
        args=(annual_returns, annual_covariance),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    print("running sharpe portfolio...")
    max_sharpe_result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(annual_returns, annual_covariance, 0.0),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    # report results
    print(f"\nOptimization Results Summary (2025):")

    gmv_weights = gmv_result.x
    gmv_ret, gmv_vol = portfolio_performance(gmv_weights, annual_returns, annual_covariance)

    print("Global Minimum Variance (GMV):")
    print(f"Expected Annual Return: {gmv_ret * 100:.2f}%")
    print(f"Annual Volatility:      {gmv_vol * 100:.2f}%")
    print(f"Sharpe Ratio:           {gmv_ret / gmv_vol:.2f}")
    print("Allocation:")
    for asset, weight in zip(assets, gmv_weights):
        print(f"{asset}: {weight * 100:.1f}%")

    print("-" * 50)

    ms_weights = max_sharpe_result.x
    ms_ret, ms_vol = portfolio_performance(ms_weights, annual_returns, annual_covariance)

    print("Maximum Sharpe Ratio Portfolio:")
    print(f"Expected Annual Return: {ms_ret * 100:.2f}%")
    print(f"Annual Volatility:      {ms_vol * 100:.2f}%")
    print(f"Sharpe Ratio:           {ms_ret / ms_vol:.2f}")
    print("Allocation:")
    for asset, weight in zip(assets, ms_weights):
        print(f"{asset}: {weight * 100:.1f}%")
    print("=" * 50)

    # try random weights loop
    print("\nsimulating portfolios...")

    np.random.seed(42)
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    weights_record = []


    for i in range(num_portfolios):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        weights_record.append(w)

        p_ret, p_vol = portfolio_performance(w, annual_returns, annual_covariance)

        results[0, i] = p_ret
        results[1, i] = p_vol
        results[2, i] = p_ret / p_vol if p_vol != 0 else 0

    # plot results
    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(results[1, :] * 100, results[0, :] * 100,
                          c=results[2, :], cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.colorbar(scatter, label='Sharpe Ratio')

    plt.scatter(gmv_vol * 100, gmv_ret * 100, marker='*', color='b', s=200, label='Global Minimum Variance (GMV)')
    plt.scatter(ms_vol * 100, ms_ret * 100, marker='*', color='r', s=200, label='Maximum Sharpe Portfolio')

    plt.title('Monte Carlo Simulation: Markowitz Efficient Frontier (2025)')
    plt.xlabel('Annualised Volatility (Risk) %')
    plt.ylabel('Annualised Return %')
    plt.legend(labelspacing=0.8)
    plt.grid(True, linestyle='--', alpha=0.5)

    print("done")
    plt.show()