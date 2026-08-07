# From Single-Asset Timing to Scenario-Based Portfolio Allocation

This repository traces a project that moved from a single-asset
moving-average backtest to a small scenario-based allocation experiment.

V2 replaced V1's fixed percentage stop because the same 5% distance meant
different things for assets with very different recent volatility. V3 moved
from single-asset timing to a four-asset portfolio because changing the stop
still left the comparison tied to separate realized market paths. V3.1 tested the resulting maximum-Sharpe weights on an early-2026 holdout. V4 then changed the distributional assumptions and risk objective.

By V4, the question had changed: how much did the answer move when I changed
the asset universe, data window, distributional assumptions, and risk
objective?

## How the Project Changed

| Version | Main change | Why it changed |
| --- | --- | --- |
| V1 | Moving-average crossover with a fixed 5% stop and a 0.1% commission | I first needed to make sure a signal was available before I credited it with a return. |
| V2 | Replaced the common percentage stop with a 14-day ATR distance fixed at entry | A percentage threshold does not represent the same amount of recent price movement across assets |
| V3 | Replaced single-asset timing with long-only Markowitz allocation across AAPL, TSLA, BTC-USD, and GLD | The single-asset tests did not answer how the assets should be combined in one portfolio |
| V3.1 | Froze the 2025 maximum-Sharpe weights and applied them to one early-2026 holdout | The in-sample optimum needed a forward check |
| V4 | Fitted Student-t marginals, generated 20,000 Gaussian-copula scenarios, and minimized empirical CVaR at three confidence levels | Variance and Sharpe do not directly target the worst simulated losses |

## V1 and V2: A Rule That Did Not Transfer Cleanly

V1 selects MA(8, 60) on AAPL and then applies the same pair and fixed 5% stop
to TSLA and BTC-USD. The crossover signal is evaluated using the previous row,
so a signal calculated at the close of day `t` is not credited with the return
already realized during day `t`.

V2 keeps the same event loop but sets the stop once, at entry:

```text
stop = entry price - 1.5 × ATR(14)
```

It is an ATR-scaled entry stop, not a trailing stop.

| Asset | V1 return | V1 Sharpe | V2 return | V2 Sharpe |
| --- | ---: | ---: | ---: | ---: |
| AAPL | 31.6% | 2.16 | 31.6% | 2.16 |
| TSLA | -5.4% | -0.05 | -6.5% | -0.09 |
| BTC-USD | 8.9% | 0.58 | 13.7% | 0.84 |

The ATR version returned more for BTC-USD, slightly less for TSLA, and the same for AAPL under the selected settings. 
That was not a general improvement. The comparison held the signal and price path fixed within each asset, 
but three realized paths were not enough to show that ATR stops would transfer better elsewhere.

## V3 and V3.1: An Optimizer That Looked Better in Sample

V3 uses dates shared by all four assets, computes simple daily returns, and
annualizes the sample mean and covariance matrix with 252 trading days. It
solves two long-only, fully invested problems with SciPy's SLSQP:

- global minimum variance;
- maximum Sharpe ratio with a zero risk-free rate.

I plotted 10,000 normalized random portfolios to see where the optimized points sat. 
They are not a proof that the solver found a global
optimum.

| Portfolio | AAPL | TSLA | BTC-USD | GLD | Annualized mean return | Volatility | Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Global minimum variance | 22.9% | 0.0% | 8.7% | 68.5% | 38.98% | 16.73% | 2.33 |
| Maximum Sharpe | 6.2% | 4.3% | 0.0% | 89.6% | 48.71% | 18.36% | 2.65 |

The concentration in GLD was one reason not to treat the optimizer's output as
a recommendation. V3.1 freezes the full-precision maximum-Sharpe weights and
uses them on the fixed early-2026 window configured in the code:

| Portfolio | OOS cumulative return | OOS Sharpe |
| --- | ---: | ---: |
| 2025 maximum-Sharpe weights | -2.12% | 0.01 |
| Equal weight (1/N) | -7.30% | -0.56 |

The optimized portfolio lost less than equal weight in this window, but its
in-sample Sharpe ratio did not survive. One holdout cannot show that the
weights have persistent predictive value.

## V4: Changing the Risk Objective

V4 fits a separate Student-t distribution to each asset's 2025 returns. It
uses the 2025 Pearson correlation matrix in a Gaussian copula, transforms
20,000 correlated normal draws through the fitted marginal distributions, and
minimizes the mean loss in the worst `1 - alpha` fraction of scenarios.

| CVaR level | AAPL | TSLA | BTC-USD | GLD |
| --- | ---: | ---: | ---: | ---: |
| 90% | 15.29% | 0.00% | 9.21% | 75.50% |
| 95% | 11.80% | 0.43% | 12.86% | 74.91% |
| 99% | 5.36% | 2.40% | 19.81% | 72.42% |

The same 2026 window gives:

| Portfolio | OOS return | Sharpe | Max drawdown | Worst day |
| --- | ---: | ---: | ---: | ---: |
| CVaR 90% | -3.30% | -0.13 | -19.41% | -7.73% |
| CVaR 95% | -4.68% | -0.24 | -20.70% | -7.69% |
| CVaR 99% | -7.30% | -0.43 | -22.92% | -7.44% |
| V3 maximum Sharpe | -2.12% | 0.01 | -21.66% | -9.03% |
| Equal weight | -7.30% | -0.56 | -15.13% | -4.79% |

Which portfolio looked best depended on the metric. V3 has the least negative return and the
highest Sharpe ratio. Equal weight has the shallowest drawdown and mildest
worst day. The 90% and 95% CVaR portfolios improve drawdown and worst-day loss
relative to V3, but give up cumulative return.

The ordering also does not support a simple story that a higher confidence
level was safer in the realized holdout. These are descriptive results from
one path, not estimates of how the portfolios will rank in future periods.

## Details That Affect the Interpretation

- V1 and V2 check the current close against the stop but record the exit at
  the theoretical stop price. They do not model intraday paths, bid-ask
  spreads, slippage, or gaps through the stop.
- AAPL is used both to select the V1/V2 parameters and to report their AAPL
  performance, so that result is in sample.
- V1 and V2 annualize Sharpe ratios with 252 days for equities and 365 days for
  BTC-USD. V3 and V4 use 252 days after aligning all four assets to shared
  dates.
- V3 and V4 keep only dates available for AAPL, TSLA, BTC-USD, and GLD
  together. BTC weekend observations are removed rather than filled with zero
  equity returns.
- V3.1 and V4 apply fixed weights to each day's returns. This is equivalent to
  costless daily rebalancing, not a buy-and-hold portfolio.
- Student-t marginals allow heavier univariate tails, but a Gaussian copula
  does not add non-zero asymptotic tail dependence.
- V4 changes the marginal distributions, scenario construction, and objective
  at the same time. The comparison cannot attribute its result to any one of
  those changes.
- V3.1 uses the optimizer's full-precision weights. The V3 benchmark inside V4
  uses the same weights rounded to four decimal places.
- The cached 2025 prices reproduce the V3 estimates. The 2026 holdout is
  downloaded at run time and is not cached, so later data revisions may change
  the exact OOS output.

## Repository Guide

- `backtest.py` — V1 moving-average engine and fixed 5% stop
- `backtest_v2_atr_overfit_experiment.py` — V2 ATR-scaled stop fixed at entry
- `backtest_v3_markowitz_portfolio.py` — V3 correlation matrix, GMV,
  maximum-Sharpe allocation, and random-portfolio plot
- `backtest_v3_1_oos_forward_test.py` — V3.1 fixed early-2026 holdout
- `backtest_v4_cvar_scenario_allocation.py` — V4 scenario generation, CVaR
  allocations, and OOS comparison
- `v3_market_data_2025.csv` — cached 2025 Close series used by V3 and V4
- `From_Single_Asset_Timing_to_Scenario_Based_Allocation.pdf` — full
  technical report

## Running the Code

Install the required packages:

```bash
pip install numpy pandas scipy matplotlib seaborn yfinance
```

Then run the versions in order:

```bash
python backtest.py
python backtest_v2_atr_overfit_experiment.py
python backtest_v3_markowitz_portfolio.py
python backtest_v3_1_oos_forward_test.py
python backtest_v4_cvar_scenario_allocation.py
```

V4 reads `v3_market_data_2025.csv` rather than downloading the estimation
sample itself. If the file is missing, run V3 or V3.1 first. V1, V2, and the
holdout sections of V3.1 and V4 require a working connection to Yahoo Finance.

The longer write-up is available in
[`From_Single_Asset_Timing_to_Scenario_Based_Allocation.pdf`](./From_Single_Asset_Timing_to_Scenario_Based_Allocation.pdf)
