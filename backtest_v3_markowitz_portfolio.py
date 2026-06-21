"""
v3.0 - Modern Portfolio Theory (MPT) Optimization
Calculates asset matrix dependencies (AAPL, TSLA, BTC-USD, GLD).
Employs the SLSQP solver to isolate the Global Minimum Variance (GMV) and
Maximum Sharpe allocations, mapping the portfolio topology via Monte Carlo.
Date: June 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize

os.environ['http_proxy'] = 'http://127.0.0.1:7892'
os.environ['https_proxy'] = 'http://127.0.0.1:7892'


def portfolio_performance(weights, returns, cov_matrix):
    port_return = np.sum(returns * weights)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_volatility


def minimize_volatility(weights, returns, cov_matrix):
    return portfolio_performance(weights, returns, cov_matrix)[1]


def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.0):
    p_ret, p_vol = portfolio_performance(weights, returns, cov_matrix)
    if p_vol == 0:
        return 0
    return -(p_ret - risk_free_rate) / p_vol


if __name__ == "__main__":
    tickers = ["AAPL", "TSLA", "BTC-USD", "GLD"]
    csv_filename = "v3_market_data_2025.csv"

    # --- Data Retrieval & Caching Interception ---
    if os.path.exists(csv_filename):
        print(f"Local cache hit [{csv_filename}]. Loading data array directly.")
        raw_data = pd.read_csv(csv_filename, index_col=0, parse_dates=True)
    else:
        print("Cache empty. Launching yfinance asset download pipeline...")
        try:
            raw_data = yf.download(tickers, start="2025-01-01", end="2025-12-31", progress=False)['Close']
            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns = raw_data.columns.droplevel(1)
            if not raw_data.dropna(how='all').empty:
                raw_data.to_csv(csv_filename)
            else:
                raise ValueError("Downloaded dataset contains null vectors. Check API limits.")
        except Exception as e:
            print(f"Data stream exception: {e}")
            raw_data = pd.DataFrame()

    if raw_data.empty:
        exit()

    daily_returns = raw_data.pct_change().dropna()
    correlation_matrix = daily_returns.corr()

    print("\n=== 2025 Empirical Correlation Matrix ===")
    print(correlation_matrix.round(3))

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title("Asset Correlation Matrix (2025)")
    plt.tight_layout()

    num_assets = len(daily_returns.columns)
    assets = list(daily_returns.columns)
    num_trading_days = 252

    # Annualized first moment (expected returns vector) and second moment (covariance matrix)
    annual_returns = daily_returns.mean() * num_trading_days
    annual_covariance = daily_returns.cov() * num_trading_days

    # --- Convex Optimization Constraints: Fully invested long-only regime ---
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1.0 / num_assets]

    # Convex solver execution
    gmv_result = minimize(
        minimize_volatility,
        initial_weights,
        args=(annual_returns, annual_covariance),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    max_sharpe_result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(annual_returns, annual_covariance, 0.0),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    print("\n==================================================")
    print("         Markowitz Frontier Optimization Audit (2025)")
    print("==================================================")

    gmv_weights = gmv_result.x
    gmv_ret, gmv_vol = portfolio_performance(gmv_weights, annual_returns, annual_covariance)
    print("[1] Global Minimum Variance Portfolio (GMV):")
    print(f"  -> Ret: {gmv_ret * 100:.2f}% | Vol: {gmv_vol * 100:.2f}% | Sharpe: {gmv_ret / gmv_vol:.2f}")
    for asset, weight in zip(assets, gmv_weights):
        print(f"     * {asset}: {weight * 100:.1f}%")

    print("-" * 50)

    ms_weights = max_sharpe_result.x
    ms_ret, ms_vol = portfolio_performance(ms_weights, annual_returns, annual_covariance)
    print("[2] Tangency Maximum Sharpe Portfolio:")
    print(f"  -> Ret: {ms_ret * 100:.2f}% | Vol: {ms_vol * 100:.2f}% | Sharpe: {ms_ret / ms_vol:.2f}")
    for asset, weight in zip(assets, ms_weights):
        print(f"     * {asset}: {weight * 100:.1f}%")
    print("==================================================")

    # --- Monte Carlo Simulation: Mapping 10,000 random vectors ---
    print("\nRunning 10,000-portfolio simulation...")
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    np.random.seed(42)

    for i in range(num_portfolios):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        p_ret, p_vol = portfolio_performance(w, annual_returns, annual_covariance)
        results[0, i] = p_ret
        results[1, i] = p_vol
        results[2, i] = p_ret / p_vol if p_vol != 0 else 0

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(results[1, :] * 100, results[0, :] * 100, c=results[2, :], cmap='viridis', s=10, alpha=0.3)
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.scatter(gmv_vol * 100, gmv_ret * 100, marker='*', color='b', s=200, label='GMV')
    plt.scatter(ms_vol * 100, ms_ret * 100, marker='*', color='r', s=200, label='Max Sharpe')
    plt.title('Monte Carlo Simulation: Efficient Frontier')
    plt.xlabel('Annualised Volatility %')
    plt.ylabel('Annualised Return %')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    print("Optimization and frontier mapping complete. Close the plot window to exit.")
    plt.show()