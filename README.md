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

### Phase 3: Markowitz Portfolio Optimization (Work in Progress)
* **Next Step:** Optimising parameters asset-by-asset feels like chasing a ghost. Instead of looking for a perfect single-asset stop-loss, my next learning objective is to step back and apply modern portfolio theory (MPT). I am currently rebuilding the engine to use covariance matrices to allocate weights across multiple assets simultaneously.

---
*Author: Zhixun (Ricardo) Zheng | May 2026*
