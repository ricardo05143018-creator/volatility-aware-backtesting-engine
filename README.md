# Volatility-Aware Backtesting Engine
An independent project exploring trend-following strategies, look-ahead bias, and the limitations of single-asset risk management.

## Overview
This repository documents my process of building a custom backtesting engine from scratch in Python. I started this because I noticed many online trading tutorials use vectorised pandas operations that accidentally introduce **look-ahead bias**. My goal was to build a strict day-by-day simulation loop and test how different risk management rules behave across assets with different volatility profiles (e.g., AAPL, TSLA, and BTC-USD).

## Research Log

### Phase 1: The Fixed Stop-Loss Flaw (v1)
* **File:** `backtest.py`
* **What I did:** Tested a Dual Moving Average (DMA) crossover strategy with a hard 5% fixed stop-loss.
* **What went wrong:** The fixed stop-loss worked perfectly for stable stocks like AAPL and TSLA, but caused a severe "whipsaw" effect on Bitcoin, forcefully exiting positions during normal daily fluctuations and actively destroying performance compared to buy-and-hold.

### Phase 2: ATR & The Overfitting Trap (v2)
* **File:** `backtest_v2_atr_overfit_experiment.py`
* **What I did:** Replaced the fixed stop-loss with an Average True Range (ATR) dynamic threshold to account for time-varying volatility and overnight gaps. I ran a Grid Search to find the optimal ATR multiplier.
* **The Trap:** The grid search perfectly curve-fitted the ATR multiplier (1.5x) to AAPL's low-volatility 2025 data. When I tested this "optimal" parameter on TSLA, the strategy performed even worse than the fixed 5% stop-loss.
* **Lesson Learned:** Parameter optimisation on historical data is highly prone to overfitting. A risk management rule calibrated for one volatility regime can fail badly in another. 

### Phase 3: Modern Portfolio Theory & The Efficient Frontier (v3)
* **File:** `backtest_v3_markowitz_portfolio.py`
* **What I did:** Instead of optimizing stop-loss parameters for individual assets (which led to curve-fitting), I shifted to Asset Allocation. I introduced Gold (GLD) as an uncorrelated asset and built a covariance matrix from 2025 daily price data to lower portfolio variance. I then used scipy.optimize to calculate the Global Minimum Variance (GMV) and Max Sharpe portfolios.
* **The Verification (Brute Force):** Because the underlying optimization algorithm (SLSQP) is currently a mathematical "black box" to me, I wrote a Monte Carlo simulation. I generated 10,000 random portfolio weights to draw the Markowitz Efficient Frontier from scratch. The optimizer's theoretical results aligned perfectly with the visual edges of my scatter plot.
* **Optimization Results (2025):**
  * Global Minimum Variance (GMV): Allocated 72.9% to GLD, 16.0% to AAPL, 11.1% to BTC, and 0.0% to TSLA (Expected Return: 23.46%, Volatility: 15.50%).
  * Maximum Sharpe Ratio: Allocated 87.1% to GLD, 12.9% to AAPL, and 0.0% to the rest (Expected Return: 28.14%, Volatility: 16.34%).
* **The Trap:** Adding an uncorrelated asset like GLD fundamentally lowers portfolio variance. However, Markowitz optimization is highly sensitive to historical returns. The Max Sharpe algorithm acted as an "error maximizer," heavily over-allocating to GLD simply because it had a strong bull run in 2025. This proves that feeding historical returns directly into an optimizer is just another form of look-ahead bias.

---
*Author: Zhixun (Ricardo) Zheng | June 2026*
