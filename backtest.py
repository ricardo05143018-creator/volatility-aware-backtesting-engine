"""
Simple moving-average crossover backtest (V1 Baseline).

This version uses:
- Golden Cross / Death Cross for entries and exits
- Fixed 5% stop loss from the entry price

I wrote this first to establish a basic trend-following baseline
before trying more advanced volatility-based stops.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf

# Network proxy setup if needed
# os.environ['http_proxy'] = 'http://127.0.0.1:7892'
# os.environ['https_proxy'] = 'http://127.0.0.1:7892'


def get_annual_trading_days(ticker):
    """
    252 days for stocks, 365 for crypto.
    Need this so the crypto Sharpe ratio doesn't get messed up.
    """
    if "-USD" in ticker or "-USDT" in ticker:
        return 365
    return 252


def run_backtest(ticker, short_window, long_window, stop_loss_pct=0.05, show_plot=True):
    """
    Runs the actual bar-by-bar loop for the MA crossover + fixed stop loss.
    Returns a dictionary of metrics like Sharpe, Max DD, etc.
    """

    # --- load data ---
    raw_df = yf.download(ticker, start="2025-01-01", end="2025-12-31", progress=False)

    # idk why yfinance returns multiindex sometimes but whatever
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.droplevel(1)

    if raw_df.empty:
        print(f"  Warning: no data returned for {ticker}, skipping.")
        return None

    close = raw_df['Close']

    # --- calculate short and long MAs ---
    raw_df['MA_Short'] = close.rolling(window=short_window).mean()
    raw_df['MA_Long']  = close.rolling(window=long_window).mean()

    df = raw_df.dropna().copy()

    if len(df) < 2:
        print(f"  Not enough rows for {ticker} with windows {short_window}/{long_window}")
        return None

    # check if short MA is above long MA
    df['Signal'] = df['MA_Short'] > df['MA_Long']

    # --- run loop day by day ---
    # tried vectorized first, got weird results, switched to loop

    closes  = df['Close'].values
    signals = df['Signal'].values

    commission = 0.001   # 0.1% per trade for fees and slippage
    # print(f"debug: {ticker}  short={short_window}  long={long_window}")

    strategy_returns = [0.0]
    positions        = [0]
    trade_log        = []

    position    = 0      # 0 = cash, 1 = long
    entry_price = 0.0
    entry_day   = 0

    for i in range(1, len(df)):
        daily_mkt_return = (closes[i] - closes[i - 1]) / closes[i - 1]
        strat_ret = 0.0

        if position == 1:

            drawdown_from_entry = (closes[i] - entry_price) / entry_price

            if drawdown_from_entry <= -stop_loss_pct:
                # assume stop executes at exactly stop price -- probably not realistic
                # in real life you'd get slippage but close enough for now
                stop_price = entry_price * (1 - stop_loss_pct)
                strat_ret  = (stop_price - closes[i - 1]) / closes[i - 1] - commission
                position   = 0
                trade_log.append({
                    'entry_price' : round(entry_price, 2),
                    'exit_price'  : round(stop_price, 2),
                    'return_pct'  : round(strat_ret * 100, 3),
                    'holding_days': i - entry_day,
                    'exit_reason' : 'stop_loss'
                })

            elif not signals[i - 1]:
                # death cross exit
                strat_ret = -commission
                position  = 0
                trade_log.append({
                    'entry_price' : round(entry_price, 2),
                    'exit_price'  : round(closes[i - 1], 2),
                    'return_pct'  : round(strat_ret * 100, 3),
                    'holding_days': i - entry_day,
                    'exit_reason' : 'death_cross'
                })

            else:
                # still in the trade
                # TODO: try trailing stop or ATR later, 5% is too rigid for crypto
                strat_ret = daily_mkt_return

        else:
            # golden cross entry signal
            if i >= 2 and signals[i - 1] and not signals[i - 2]:
                position    = 1
                entry_price = closes[i - 1]
                entry_day   = i
                strat_ret   = daily_mkt_return - commission

        strategy_returns.append(strat_ret)
        positions.append(position)


    # --- handle open position at the end of the year ---
    if position == 1:
        final_return_pct = (closes[-1] - entry_price) / entry_price * 100
        trade_log.append({
            'entry_price' : round(entry_price, 2),
            'exit_price'  : round(closes[-1], 2),
            'return_pct'  : round(final_return_pct, 3),
            'holding_days': len(df) - entry_day,
            'exit_reason' : 'year_end'
        })


    df['Strategy_Return'] = strategy_returns
    df['Position']        = positions

    # --- calculate final metrics ---
    df['Market_Return'] = df['Close'].pct_change().fillna(0)
    df['Market_Cum']    = (1 + df['Market_Return']).cumprod()
    df['Strategy_Cum']  = (1 + df['Strategy_Return'].fillna(0)).cumprod()

    ann_days = get_annual_trading_days(ticker)
    mean_ret = df['Strategy_Return'].mean()
    std_ret  = df['Strategy_Return'].std()

    # annualize by multiplying by sqrt of days
    if std_ret == 0 or pd.isna(std_ret):
        sharpe = 0.0
    else:
        sharpe = (mean_ret / std_ret) * (ann_days ** 0.5)

    peak         = df['Strategy_Cum'].cummax()
    max_drawdown = (df['Strategy_Cum'] / peak - 1.0).min()
    total_return = df['Strategy_Cum'].iloc[-1] - 1.0

    num_trades = len(trade_log)
    if num_trades > 0:
        wins            = [t for t in trade_log if t['return_pct'] > 0]
        win_rate        = len(wins) / num_trades
        avg_holding     = sum(t['holding_days'] for t in trade_log) / num_trades
        stop_loss_exits = sum(1 for t in trade_log if t['exit_reason'] == 'stop_loss')
    else:
        win_rate = avg_holding = stop_loss_exits = 0

    # --- plotting code ---
    if show_plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1]})

        axes[0].plot(df.index, df['Market_Cum'],
                     label='Buy & Hold', color='gray', linestyle='--')
        axes[0].plot(df.index, df['Strategy_Cum'],
                     label=f'MA Strategy', color='steelblue')

        dd_series = df['Strategy_Cum'] / df['Strategy_Cum'].cummax() - 1.0
        dd_end    = dd_series.idxmin()
        dd_start  = df['Strategy_Cum'].loc[:dd_end].idxmax()
        axes[0].axvspan(dd_start, dd_end, color='red', alpha=0.15, label='Max Drawdown Zone')

        axes[0].set_title(
            f"{ticker}  |  Return: {total_return * 100:.1f}%  |  "
            f"Sharpe: {sharpe:.2f}  |  Trades: {num_trades}",
            fontsize=9
        )
        axes[0].legend(fontsize=8)
        axes[0].set_ylabel("Cumulative Return")

        axes[1].fill_between(df.index, df['Position'], step='post',
                             alpha=0.3, color='steelblue')
        axes[1].set_ylabel("Position")
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(["Cash", "Long"])

        plt.tight_layout()
        plt.show()

    return {
        "Sharpe"          : round(sharpe, 2),
        "Max_Drawdown"    : round(max_drawdown, 4),
        "Total_Return"    : round(total_return, 4),
        "Num_Trades"      : num_trades,
        "Win_Rate"        : round(win_rate, 3),
        "Avg_Holding_Days": round(avg_holding, 1),
        "Stop_Loss_Exits" : stop_loss_exits,
        "Trade_Log"       : trade_log
    }


# =============================================================
if __name__ == "__main__":

    # ---- Step 1: run grid search on AAPL ----
    short_options = [3, 5, 8, 10]
    long_options  = [20, 30, 40, 50, 60]
    results       = []

    print("Running parameter grid search on AAPL (2025)...\n")

    for s in short_options:
        for l in long_options:
            if s >= l:
                continue
            r = run_backtest("AAPL", s, l, stop_loss_pct=0.05, show_plot=False)
            if r:
                results.append({
                    "Short"       : s,
                    "Long"        : l,
                    "Sharpe"      : r["Sharpe"],
                    "Total_Return": r["Total_Return"],
                    "Max_Drawdown": r["Max_Drawdown"],
                    "Num_Trades"  : r["Num_Trades"],
                    "Win_Rate"    : r["Win_Rate"]
                })

    results_df = pd.DataFrame(results)

    print("Top 5 parameters sorted by Sharpe:\n")
    print(results_df.sort_values("Sharpe", ascending=False).head(5).to_string(index=False))

    best_row   = results_df.loc[results_df["Sharpe"].idxmax()]
    best_short = int(best_row["Short"])
    best_long  = int(best_row["Long"])

    print(f"\nBest settings found: MA({best_short}, {best_long})")
    print(f"  Sharpe: {best_row['Sharpe']}  "
          f"Return: {best_row['Total_Return'] * 100:.2f}%  "
          f"MDD: {best_row['Max_Drawdown'] * 100:.2f}%\n")

    # ---- Step 2: cross-market test ----
    print(f"Testing these settings across other markets...\n")

    tickers     = ["AAPL", "TSLA", "BTC-USD"]
    all_results = []

    for t in tickers:
        print(f"--- {t} ---")
        report = run_backtest(t, best_short, best_long, stop_loss_pct=0.05, show_plot=True)
        if report:
            print(f"  Sharpe:          {report['Sharpe']}")
            print(f"  Total Return:    {report['Total_Return'] * 100:.2f}%")
            print(f"  Max Drawdown:    {report['Max_Drawdown'] * 100:.2f}%")
            print(f"  Trades:          {report['Num_Trades']}")
            print()
            all_results.append({"Ticker": t, **{k: v for k, v in report.items()
                                                 if k != "Trade_Log"}})

    print("=== Final Results Summary ===")
    summary_df = pd.DataFrame(all_results).drop(columns=["Trade_Log"], errors="ignore")
    print(summary_df.to_string(index=False))