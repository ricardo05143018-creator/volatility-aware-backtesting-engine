"""
v3_1: markowitz out-of-sample forward test on unseen market window
Date: June 2026
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
            else:
                raise ValueError("Downloaded data is empty. Might be rate limited.")
        except Exception as e:
            print(f"Error downloading data: {e}")
            raw_data = pd.DataFrame()

    if raw_data.empty:
        exit()

    raw_data = raw_data.loc[:, tickers].dropna()
    daily_returns = raw_data.pct_change(fill_method=None).dropna()
    correlation_matrix = daily_returns.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title("Asset Correlation Matrix (2025)")
    plt.tight_layout()

    num_assets = len(daily_returns.columns)
    assets = list(daily_returns.columns)
    num_trading_days = 252

    # Scale daily returns and covariance to annual assuming i.i.d. observations.
    annual_returns = daily_returns.mean() * num_trading_days
    annual_covariance = daily_returns.cov() * num_trading_days

    # Long-only portfolio constraints (no short selling allowed, weights sum to 1).
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1.0 / num_assets]

    max_sharpe_result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(annual_returns, annual_covariance, 0.0),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    if not max_sharpe_result.success:
        raise RuntimeError(
            f"Maximum-Sharpe optimization failed: {max_sharpe_result.message}"
        )
    ms_weights = max_sharpe_result.x

    print("\n2025 maximum-Sharpe weights:")
    print(pd.Series(ms_weights, index=assets).to_string())

    gmv_result = minimize(
        minimize_volatility,
        initial_weights,
        args=(annual_returns, annual_covariance),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    if not gmv_result.success:
        raise RuntimeError(
            f"GMV optimization failed: {gmv_result.message}"
        )
    gmv_weights = gmv_result.x

    np.random.seed(42)
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        p_ret, p_vol = portfolio_performance(w, annual_returns, annual_covariance)
        results[0, i] = p_ret
        results[1, i] = p_vol
        results[2, i] = p_ret / p_vol if p_vol != 0 else 0

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(results[1, :] * 100, results[0, :] * 100, c=results[2, :], cmap='viridis', marker='o', s=10,
                          alpha=0.3)
    plt.colorbar(scatter, label='Sharpe Ratio')

    gmv_ret_p, gmv_vol_p = portfolio_performance(gmv_weights, annual_returns, annual_covariance)
    ms_ret_p, ms_vol_p = portfolio_performance(ms_weights, annual_returns, annual_covariance)
    plt.scatter(gmv_vol_p * 100, gmv_ret_p * 100, marker='*', color='b', s=200, label='Global Minimum Variance (GMV)')
    plt.scatter(ms_vol_p * 100, ms_ret_p * 100, marker='*', color='r', s=200, label='Maximum Sharpe Portfolio')

    plt.title("Random Long-Only Portfolios and Optimized Allocations (2025)")
    plt.xlabel('Annualised Volatility (Risk) %')
    plt.ylabel('Annualised Return %')
    plt.legend(labelspacing=0.8)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Forward test the 2025 optimized weights on a fixed early-2026 window.
    print("\nExecuting forward test on unseen 2026 window...")
    try:
        oos_raw = yf.download(tickers, start="2026-01-01", end="2026-06-15", progress=False)['Close']
        if isinstance(oos_raw.columns, pd.MultiIndex):
            oos_raw.columns = oos_raw.columns.droplevel(1)

        oos_raw = oos_raw.loc[:, assets].dropna()
        oos_returns = oos_raw.pct_change(fill_method=None).dropna()

        oos_ms_daily = oos_returns.dot(ms_weights)
        oos_ms_cum = (1 + oos_ms_daily).cumprod() - 1
        oos_ms_sharpe = (oos_ms_daily.mean() / oos_ms_daily.std()) * np.sqrt(num_trading_days)

        eq_weights = np.array([1.0 / num_assets] * num_assets)
        oos_eq_daily = oos_returns.dot(eq_weights)
        oos_eq_cum = (1 + oos_eq_daily).cumprod() - 1
        oos_eq_sharpe = (oos_eq_daily.mean() / oos_eq_daily.std()) * np.sqrt(num_trading_days)

        print("\n--- 2026 Out-of-Sample Forward Test ---")
        print("Max Sharpe Allocation:")
        print(f"  OOS Cumulative Return: {oos_ms_cum.iloc[-1] * 100:.2f}%")
        print(f"  OOS Sharpe Ratio:      {oos_ms_sharpe:.2f}")

        print(f"Naïve Equal Weight (1/N) Benchmark Baseline:")
        print(f"  OOS Cumulative Return: {oos_eq_cum.iloc[-1] * 100:.2f}%")
        print(f"  OOS Sharpe Ratio:      {oos_eq_sharpe:.2f}")

        print("\nScope:")
        print("  This is one fixed holdout comparison.")
        print("  It does not establish persistent alpha or isolate the cause of the result.")

    except Exception as e:
        raise RuntimeError("Out-of-sample analysis failed") from e