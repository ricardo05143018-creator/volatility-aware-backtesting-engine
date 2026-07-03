# Volatility-Aware Backtesting Engine

An independent quantitative finance project documenting the evolution of a custom Python backtesting engine. Rather than searching for a profitable strategy, this project investigates **why seemingly reasonable quantitative methods fail when exposed to increasingly realistic market conditions.**

**Tech Stack:** Python 3.11+, NumPy, SciPy, Pandas, Matplotlib, Git.

---

## Research Evolution

* **Phase 1: Single-Asset Framework**
  Event-Driven Backtesting -> Fixed Stop-Loss -> Whipsaw Effects on Volatile Assets
* **Phase 2: Adaptive Risk Mitigation**
  ATR Dynamic Stop-Loss -> Grid Search Optimisation -> Curve Fitting & Regime Trap
* **Phase 3: Multi-Asset Mathematical Expansion**
  Markowitz MPT -> Monte Carlo Verification (10k) -> Out-of-Sample Forward Failure

---

## Phase 1 — Eliminating Look-Ahead Bias

* **Implementation:** I built a strict event-driven engine using explicit daily iteration loops. Trading decisions use only T-1 close data to completely block data leakage.
* **The Whipsaw Problem:** I tested a baseline DMA strategy with a fixed 5% stop-loss. While it removed look-ahead bias, applying a static risk rule caused severe whipsawing on high-volatility assets like TSLA and BTC-USD, proving that fixed thresholds do not survive across different volatility regimes.

## Phase 2 – ATR & The Overfitting Trap

To address the whipsaw effect, I introduced an Average True Range (ATR) dynamic stop-loss band and ran a multi-dimensional Grid Search to find the optimal multipliers.

The result was a brutal lesson in historical curve-fitting. While the "optimal" parameter looked great in-sample, it failed completely when tested out-of-sample across un-correlated assets. The optimizer simply memorized a specific market regime rather than discovering a robust trading rule.

## Phase 3 – Portfolio Allocation & Forward Failure

Realising single-asset timing was too fragile, I shifted to risk diversification using Modern Portfolio Theory (Markowitz), adding Gold (GLD) to isolate equity noise. Because SciPy's SLSQP solver felt like a black box at my current level of study, I verified the optimizer by brute-forcing a Monte Carlo simulation of 10,000 randomised portfolios, switching to a Dirichlet distribution after noticing the standard method clustered all weights right in the centre.

**The Out-of-Sample Reality Check:**
I locked the optimal weights fitted on 2025 data and forward-tested them on unseen 2026 data. The performance deterioration was substantial. It proved that historical covariance scaling is highly fragile under macro-economic shifts—especially after I caught a data gap where a standard `dropna()` was silently stripping Bitcoin's weekend volatility from the matrix. Re-aligning the timeline with `.ffill()` and 365-day scaling actually made the forward failure look worse. Mathematical optimality in-sample guarantees nothing about forward predictability–it often just acts as an error maximizer that amplifies historical anomalies.

---

## Repository Structure

## Repository Structure

* `backtest.py` — Phase 1: Event-driven engine & fixed stop-loss
* `backtest_v2_atr_overfit_experiment.py` — Phase 2: ATR grid search & robustness testing
* `backtest_v3_markowitz_portfolio.py` — Phase 3: MPT optimization & Monte Carlo simulation
* `backtest_v3_markowitz_portfolio_Pro.py` — Phase 3 Pro: 2026 out-of-sample forward validation
* `From_Single_Asset_Timing_to_Portfolio_Allocation.pdf` — Complete research report

---

**Dev Note:** *The current repository relies heavily on historical backtests across 2025/2026 data regimes. While the event-driven daily iteration mechanism completely blocks look-ahead data leakage, the sample covariance inputs remain highly sensitive to local stationary assumptions. Next step is to replace raw sample matrices with shrinkage estimators to fix the optimizer's stability issues.*

**Author:** Zhixun (Ricardo) Zheng
**Date:** July 2026
**Status:** Completed
