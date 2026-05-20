"""
Robust CVaR Portfolio Optimization and Alpha Sensitivity Testing (V4)
Author: Zhixun (Ricardo) Zheng
Date: August 2026

What this does:
- Loads the cached 2025 asset daily price data generated in previous experiments
- Fits a heavy-tailed Student's t-distribution to map true empirical tail risk
- Simulates 20,000 multi-asset correlated joint scenarios using Cholesky Decomposition
- Runs convex optimization to minimize CVaR across multiple alpha thresholds (90%, 95%, 99%)
- Evaluates out-of-sample tail protection limits against unseen 2026 market data

Note:
This is the fourth module in a staged backtesting system:
V1 = MA crossover baseline strategy
V2 = ATR risk-adjusted trend strategy
V3 = Markowitz portfolio optimization
V4 = Robust CVaR optimization (this file)
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import scipy.stats as stats

# Standard proxy configuration for home network to avoid timeout errors
os.environ['http_proxy'] = 'http://127.0.0.1:7892'
os.environ['https_proxy'] = 'http://127.0.0.1:7892'


def portfolio_scenario_losses(weights, scenario_returns):
    """Calculates portfolio losses for a given set of weights across all simulated scenarios."""
    return -np.dot(scenario_returns, weights)


def cvar_objective(weights, scenario_returns, alpha):
    """
    Objective function: Computes the Conditional Value at Risk (CVaR) at a variable alpha level.
    """
    losses = portfolio_scenario_losses(weights, scenario_returns)
    losses_sorted = np.sort(losses)
    cutoff_idx = int(len(losses_sorted) * alpha)
    tail_losses = losses_sorted[cutoff_idx:]
    return np.mean(tail_losses)


def calculate_max_drawdown(daily_returns):
    """Computes the maximum peak-to-trough drawdown of a daily return series."""
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    return drawdown.min()


def calculate_worst_daily_loss(daily_returns):
    """Isolates the single worst day of negative performance."""
    return daily_returns.min()


# ===============================================================
if __name__ == "__main__":

    tickers = ["AAPL", "TSLA", "BTC-USD", "GLD"]
    csv_filename = "v3_market_data_2025.csv"
    num_trading_days = 252

    # ---- Step 1: Load 2025 Historical Data and Fit Student's t ----
    if os.path.exists(csv_filename):
        print(f"Found local cache [{csv_filename}], loading directly...")
        raw_data = pd.read_csv(csv_filename, index_col=0, parse_dates=True)
    else:
        print(f"CRITICAL ERROR: [{csv_filename}] not found. Run Phase 3 script first to generate cache.")
        exit()

    daily_returns = raw_data.pct_change().dropna()
    assets = list(daily_returns.columns)
    num_assets = len(assets)

    print("\nFitting Student's t-distributions to historical returns...")
    t_params = {}
    for asset in assets:
        df_fit, loc_fit, scale_fit = stats.t.fit(daily_returns[asset].values)
        t_params[asset] = (df_fit, loc_fit, scale_fit)
        print(f"  * {asset} Fit Result: df = {df_fit:.2f} | scale = {scale_fit:.4f}")

    # ---- Step 2: Correlation Mapping & Scenario Generation ----
    print("\nGenerating 20,000 multi-asset correlated return scenarios...")
    num_scenarios = 20000

    corr_matrix = daily_returns.corr().values
    L = np.linalg.cholesky(corr_matrix)

    # set fixed random seed for reproducible results
    np.random.seed(42)

    random_normal = np.random.normal(0, 1, size=(num_assets, num_scenarios))
    correlated_normal = np.dot(L, random_normal).T

    simulated_returns = np.zeros((num_scenarios, num_assets))
    for idx, asset in enumerate(assets):
        df_fit, loc_fit, scale_fit = t_params[asset]
        uniform_probs = stats.norm.cdf(correlated_normal[:, idx])
        simulated_returns[:, idx] = stats.t.ppf(uniform_probs, df=df_fit, loc=loc_fit, scale=scale_fit)

    # ---- Step 3: Multi-Alpha CVaR Optimization Loop ----
    print("\nOptimizing portfolio allocations across variable alpha thresholds...")
    alpha_levels = [0.90, 0.95, 0.99]
    optimized_weights_by_alpha = {}

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1.0 / num_assets]

    for alpha in alpha_levels:
        cvar_result = minimize(
            cvar_objective,
            initial_weights,
            args=(simulated_returns, alpha),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        optimized_weights_by_alpha[alpha] = cvar_result.x

    # Print Phase 4 Multi-Alpha Allocation Report
    print("\n" + "=" * 50)
    print("      Phase 4: Multi-Alpha Robust Allocation Report")
    print("=" * 50)
    for alpha in alpha_levels:
        print(f"Configured Tail Confidence Level Alpha = {alpha * 100:.0f}%:")
        for asset, weight in zip(assets, optimized_weights_by_alpha[alpha]):
            print(f"  -> {asset}: {weight * 100:.2f}%")
        print("-" * 50)
    print("=" * 50)

    # ---- Step 4: Out-of-Sample Forward Test (2026 Data) ----
    print("\n" + "=" * 50)
    print("      Out-of-Sample Testing (Jan-June 2026)")
    print("=" * 50)

    try:
        print("Downloading unseen 2026 data for OOS validation...")
        oos_raw = yf.download(tickers, start="2026-01-01", end="2026-06-15", progress=False)['Close']
        if isinstance(oos_raw.columns, pd.MultiIndex):
            oos_raw.columns = oos_raw.columns.droplevel(1)
        oos_returns = oos_raw.pct_change().dropna()

        # baseline weights from standard max sharpe optimization
        phase3_max_sharpe_weights = np.array([0.129, 0.000, 0.871, 0.000])

        # 1. Evaluate Robust CVaR Strategy across alpha configurations
        print("\n[1] Robust CVaR Strategy Performance:")
        for alpha in alpha_levels:
            w_robust = optimized_weights_by_alpha[alpha]
            oos_cvar_daily = oos_returns.dot(w_robust)
            cvar_sharpe = (oos_cvar_daily.mean() / oos_cvar_daily.std()) * np.sqrt(num_trading_days)
            print(f"  * Tail Protection Alpha = {alpha * 100:.0f}% Portfolio:")
            print(f"    -> OOS Cumulative Return: {(1 + oos_cvar_daily).cumprod().iloc[-1] * 100 - 100:.2f}%")
            print(f"    -> OOS Annualized Sharpe Ratio: {cvar_sharpe:.2f}")
            print(f"    -> OOS Max Drawdown: {calculate_max_drawdown(oos_cvar_daily) * 100:.2f}%")
            print(f"    -> OOS Worst Daily Loss: {calculate_worst_daily_loss(oos_cvar_daily) * 100:.2f}%")

        # 2. Evaluate Phase 3 Traditional Portfolio Benchmarks
        oos_mpt_daily = oos_returns.dot(phase3_max_sharpe_weights)
        mpt_sharpe = (oos_mpt_daily.mean() / oos_mpt_daily.std()) * np.sqrt(num_trading_days)
        print(f"\n[2] Phase 3 Traditional Max Sharpe Portfolio (Fitted on 2025):")
        print(f"  -> OOS Cumulative Return: {(1 + oos_mpt_daily).cumprod().iloc[-1] * 100 - 100:.2f}%")
        print(f"  -> OOS Annualized Sharpe Ratio: {mpt_sharpe:.2f}")
        print(f"  -> OOS Max Drawdown: {calculate_max_drawdown(oos_mpt_daily) * 100:.2f}%")
        print(f"  -> OOS Worst Daily Loss: {calculate_worst_daily_loss(oos_mpt_daily) * 100:.2f}%")

        # 3. Evaluate Naive Baseline
        oos_eq_daily = oos_returns.dot(np.array([1.0 / num_assets] * num_assets))
        eq_sharpe = (oos_eq_daily.mean() / oos_eq_daily.std()) * np.sqrt(num_trading_days)
        print(f"\n[3] Naive Equal Weight (1/N) Benchmark:")
        print(f"  -> OOS Cumulative Return: {(1 + oos_eq_daily).cumprod().iloc[-1] * 100 - 100:.2f}%")
        print(f"  -> OOS Annualized Sharpe Ratio: {eq_sharpe:.2f}")

        print("\n" + "-" * 50)
        print("--- forward test summary ---")
        print("Robust CVaR control limits tail exposures compared to standard MPT allocation.")
        print("Tail protection parameters alter allocation efficiency depending on alpha specification.")
        print("-" * 50)

    except Exception as e:
        print(f"\nForward test execution error: {e}")