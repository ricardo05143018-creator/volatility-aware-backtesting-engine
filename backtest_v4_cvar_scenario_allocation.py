"""
v4: simulation-based CVaR allocation with Student-t marginals
Date: August 2026
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import scipy.stats as stats


def portfolio_scenario_losses(weights, scenario_returns):
    """Scenario losses; positive values represent losses."""
    return -np.dot(scenario_returns, weights)


def cvar_objective(weights, scenario_returns, alpha):
    """Mean loss in the worst 1-alpha fraction of scenarios."""
    losses = portfolio_scenario_losses(weights, scenario_returns)
    losses_sorted = np.sort(losses)
    cutoff_idx = int(len(losses_sorted) * alpha)
    tail_losses = losses_sorted[cutoff_idx:]
    return np.mean(tail_losses)


def calculate_max_drawdown(daily_returns):
    cumulative = np.r_[1.0, (1 + daily_returns).cumprod().to_numpy()]
    running_max = np.maximum.accumulate(cumulative)
    return np.min(cumulative / running_max - 1.0)


def calculate_worst_daily_loss(daily_returns):
    return daily_returns.min()


if __name__ == "__main__":
    tickers = ["AAPL", "TSLA", "BTC-USD", "GLD"]
    csv_filename = "v3_market_data_2025.csv"
    num_trading_days = 252

    if os.path.exists(csv_filename):
        print(f"Found local cache [{csv_filename}], loading directly...")
        raw_data = pd.read_csv(csv_filename, index_col=0, parse_dates=True)
    else:
        print(f"Error: {csv_filename} missing. Run v3 script first to dump cache.")
        exit()

    raw_data = raw_data.loc[:, tickers].dropna()

    # Keep only dates shared by all four assets; BTC weekend rows are excluded.
    daily_returns = raw_data.pct_change(fill_method=None).dropna()
    assets = list(daily_returns.columns)
    num_assets = len(assets)

    print("\nfitting student-t to 2025 returns...")
    t_params = {}
    for asset in assets:
        values = daily_returns[asset].values
        df_fit, loc_fit, scale_fit = stats.t.fit(values)

        # If the fitted distribution has infinite variance, refit with df fixed at 3.
        if df_fit <= 2.0:
            print(f"  {asset}: fitted df={df_fit:.2f}; refitting with df fixed at 3.0")
            df_fit, loc_fit, scale_fit = stats.t.fit(values, fdf=3.0)

        t_params[asset] = (df_fit, loc_fit, scale_fit)

    print("\nsimulating 20k joint scenarios...")
    num_scenarios = 20000

    corr_matrix = daily_returns.corr().values
    L = np.linalg.cholesky(corr_matrix)

    np.random.seed(42)

    random_normal = np.random.normal(0, 1, size=(num_assets, num_scenarios))
    correlated_normal = np.dot(L, random_normal).T

    simulated_returns = np.zeros((num_scenarios, num_assets))
    for idx, asset in enumerate(assets):
        df_fit, loc_fit, scale_fit = t_params[asset]
        uniform_probs = np.clip(
            stats.norm.cdf(correlated_normal[:, idx]),
            1e-10,
            1 - 1e-10,
        )
        simulated_returns[:, idx] = stats.t.ppf(uniform_probs, df=df_fit, loc=loc_fit, scale=scale_fit)

    print("\nrunning cvar optimization...")
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

        if not cvar_result.success:
            raise RuntimeError(f"CVaR optimization failed for alpha={alpha}: {cvar_result.message}")

        print(f"  [Info] Alpha={alpha} optimization func value: {cvar_result.fun:.6f}")
        optimized_weights_by_alpha[alpha] = cvar_result.x

    print("\n" + "=" * 50)
    print("         Optimization results (CVaR)")
    print("=" * 50)
    for alpha in alpha_levels:
        print(f"Alpha = {alpha * 100:.0f}% allocation:")
        for asset, weight in zip(assets, optimized_weights_by_alpha[alpha]):
            print(f"  -> {asset}: {weight * 100:.2f}%")
        print("-" * 50)
    print("=" * 50)

    print("\n" + "=" * 50)
    print("      testing on 2026 data...")
    print("=" * 50)

    try:
        oos_raw = yf.download(tickers, start="2026-01-01", end="2026-06-15", progress=False)['Close']
        if isinstance(oos_raw.columns, pd.MultiIndex):
            oos_raw.columns = oos_raw.columns.droplevel(1)

        oos_raw = oos_raw.loc[:, assets].dropna()
        oos_returns = oos_raw.pct_change(fill_method=None).dropna()

        # Use the V3 weights rounded to four decimals and align them by ticker.
        v3_weights = pd.Series({
            "AAPL": 0.0616,
            "TSLA": 0.0429,
            "BTC-USD": 0.0000,
            "GLD": 0.8955,
        })
        v3_sharpe_weights = v3_weights.reindex(assets).to_numpy()

        print("\n[1] CVaR portfolios:")
        for alpha in alpha_levels:
            cvar_weights = optimized_weights_by_alpha[alpha]
            oos_cvar_daily = oos_returns.dot(cvar_weights)
            cvar_sharpe = (oos_cvar_daily.mean() / oos_cvar_daily.std()) * np.sqrt(num_trading_days)
            print(f"  * Alpha = {alpha * 100:.0f}% Portfolio:")
            print(f"    -> OOS Return:    {(1 + oos_cvar_daily).cumprod().iloc[-1] * 100 - 100:.2f}%")
            print(f"    -> OOS Sharpe:    {cvar_sharpe:.2f}")
            print(f"    -> OOS Max DD:    {calculate_max_drawdown(oos_cvar_daily) * 100:.2f}%")
            print(f"    -> OOS Worst Day: {calculate_worst_daily_loss(oos_cvar_daily) * 100:.2f}%")

        print("\n[2] V3 Max Sharpe benchmark (2025 weights):")
        oos_mpt_daily = oos_returns.dot(v3_sharpe_weights)
        mpt_sharpe = (oos_mpt_daily.mean() / oos_mpt_daily.std()) * np.sqrt(num_trading_days)
        print(f"  -> OOS Return:    {(1 + oos_mpt_daily).cumprod().iloc[-1] * 100 - 100:.2f}%")
        print(f"  -> OOS Sharpe:    {mpt_sharpe:.2f}")
        print(f"  -> OOS Max DD:    {calculate_max_drawdown(oos_mpt_daily) * 100:.2f}%")
        print(f"  -> OOS Worst Day: {calculate_worst_daily_loss(oos_mpt_daily) * 100:.2f}%")

        print("\n[3] Equal Weight (1/N) benchmark:")
        oos_eq_daily = oos_returns.dot(np.array([1.0 / num_assets] * num_assets))
        eq_sharpe = (oos_eq_daily.mean() / oos_eq_daily.std()) * np.sqrt(num_trading_days)
        print(f"  -> OOS Return:    {(1 + oos_eq_daily).cumprod().iloc[-1] * 100 - 100:.2f}%")
        print(f"  -> OOS Sharpe:    {eq_sharpe:.2f}")
        print(f"  -> OOS Max DD:    {calculate_max_drawdown(oos_eq_daily) * 100:.2f}%")
        print(f"  -> OOS Worst Day: {calculate_worst_daily_loss(oos_eq_daily) * 100:.2f}%")

        print("\n" + "-" * 50)
        print("--- Summary ---")
        print("This comparison uses one fixed 2026 holdout window.")
        print("It does not identify the contribution of any single modelling choice.")
        print("-" * 50)

    except Exception as e:
        raise RuntimeError("Out-of-sample analysis failed") from e