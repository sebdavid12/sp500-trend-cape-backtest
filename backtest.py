"""
S&P 500 Trend-Following Backtest and CAPE Valuation Study (1881-2023)
=====================================================================

Two questions, answered with ~140 years of monthly S&P 500 data
(Robert Shiller's dataset, mirrored at github.com/datasets/s-and-p-500):

  Part 1. Does a simple trend rule (own stocks when the index is above its
          N-month moving average, otherwise own bonds) beat buy-and-hold
          after transaction costs, when the rule is chosen on one period and
          tested on a later period it never saw?

  Part 2. Does starting valuation (CAPE, the price / 10-year average real
          earnings ratio) predict the next 10 years of real stock returns,
          and does that relationship hold up out of sample?

Run:   python backtest.py            (downloads data if missing)
Needs: pandas, numpy, matplotlib     (pip install -r requirements.txt)
Output: results/ (metrics tables as CSV + charts as PNG)
"""

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # write charts to files; no display needed
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
DATA_PATH = Path("data/sp500_monthly.csv")
OUT = Path("results")

START, END = "1881-01-01", "2023-06-01"  # Shiller columns are complete in this window
IN_SAMPLE_END = "1949-12-01"             # choose the rule using data up to here...
OUT_SAMPLE_START = "1950-01-01"          # ...then judge it only on data from here on
LOOKBACKS = range(6, 13)                 # candidate moving-average lengths (months)
COST_PER_SWITCH = 0.001                  # 0.10% each time we move between stocks and bonds


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load Shiller's monthly data and build the return series we need."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date")
    df = df.loc[START:END].copy()

    # Monthly total return on stocks: price change plus 1/12 of the annual dividend.
    df["stock_ret"] = (df["SP500"] + df["Dividend"] / 12) / df["SP500"].shift(1) - 1

    # Monthly return on the "safe" asset: 1/12 of the 10-year Treasury yield.
    # Simplification: this counts bond income only and ignores bond price changes.
    df["bond_ret"] = df["Long Interest Rate"] / 100 / 12

    # Real (inflation-adjusted) total return index, used in Part 2.
    real_ret = (df["Real Price"] + df["Real Dividend"] / 12) / df["Real Price"].shift(1) - 1
    df["real_tr_index"] = (1 + real_ret.fillna(0)).cumprod()

    return df.dropna(subset=["stock_ret"])


# ----------------------------------------------------------------------------
# Part 1: trend-following backtest
# ----------------------------------------------------------------------------
def trend_strategy(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Hold stocks next month only if this month's close is above its moving average."""
    sma = df["SP500"].rolling(lookback).mean()
    signal = (df["SP500"] > sma).astype(float)

    # shift(1): the decision made at the end of month t applies to month t+1.
    # Without this shift the backtest would "see the future" (look-ahead bias).
    position = signal.shift(1)

    gross = position * df["stock_ret"] + (1 - position) * df["bond_ret"]
    switches = position.diff().abs()                 # 1 in months where we traded
    net = gross - switches * COST_PER_SWITCH

    out = pd.DataFrame({"ret": net, "position": position, "switch": switches})
    return out.dropna()


def metrics(ret: pd.Series, bond_ret: pd.Series, position=None, switches=None) -> dict:
    """Standard performance statistics for a series of monthly returns."""
    years = len(ret) / 12
    wealth = (1 + ret).cumprod()
    excess = ret - bond_ret.loc[ret.index]           # return above the safe asset
    drawdown = wealth / wealth.cummax() - 1

    m = {
        "CAGR": wealth.iloc[-1] ** (1 / years) - 1,
        "Volatility": ret.std() * np.sqrt(12),
        "Sharpe": excess.mean() / excess.std() * np.sqrt(12),
        "Max drawdown": drawdown.min(),
        "Worst month": ret.min(),
    }
    if position is not None:
        m["Time in stocks"] = position.mean()
        m["Trades per year"] = switches.sum() / years
    return m


def run_part1(df: pd.DataFrame) -> pd.DataFrame:
    ins = df.loc[:IN_SAMPLE_END]
    oos = df.loc[OUT_SAMPLE_START:]

    # Step 1: pick the lookback with the best in-sample Sharpe ratio.
    search = []
    for n in LOOKBACKS:
        s = trend_strategy(df, n).loc[ins.index.min():IN_SAMPLE_END]
        search.append({"Lookback (months)": n, **metrics(s["ret"], df["bond_ret"])})
    search = pd.DataFrame(search).set_index("Lookback (months)")
    search.to_csv(OUT / "part1_insample_lookback_search.csv", float_format="%.4f")
    best = int(search["Sharpe"].idxmax())

    # Step 2: judge that single choice on the later period only.
    rows = {}
    for label, period in [("1881-1949 (in-sample)", ins), ("1950-2023 (out-of-sample)", oos)]:
        s = trend_strategy(df, best).loc[period.index.min():period.index.max()]
        bh = period["stock_ret"].loc[s.index]
        rows[(label, "Buy & hold")] = metrics(bh, df["bond_ret"])
        rows[(label, f"Trend ({best}-mo)")] = metrics(
            s["ret"], df["bond_ret"], s["position"], s["switch"])
    table = pd.DataFrame(rows).T
    table.to_csv(OUT / "part1_results.csv", float_format="%.4f")

    # Step 3: is the out-of-sample difference statistically meaningful?
    s = trend_strategy(df, best).loc[OUT_SAMPLE_START:]
    diff = s["ret"] - df.loc[s.index, "stock_ret"]
    t_stat = diff.mean() / (diff.std() / np.sqrt(len(diff)))

    # Step 4: how did each approach do in the worst market crashes?
    crashes = {"1929-32 Crash": ("1929-09-01", "1932-06-01"),
               "1973-74 Bear": ("1973-01-01", "1974-09-01"),
               "2000-02 Dot-com": ("2000-08-01", "2002-09-01"),
               "2007-09 GFC": ("2007-10-01", "2009-02-01")}
    full = trend_strategy(df, best)
    crash_rows = []
    for name, (a, b) in crashes.items():
        crash_rows.append({
            "Episode": name,
            "Buy & hold": (1 + df.loc[a:b, "stock_ret"]).prod() - 1,
            f"Trend ({best}-mo)": (1 + full.loc[a:b, "ret"]).prod() - 1})
    crash_tbl = pd.DataFrame(crash_rows).set_index("Episode")
    crash_tbl.to_csv(OUT / "part1_crash_episodes.csv", float_format="%.4f")

    # Charts: growth of $1 (log scale) and drawdowns, out-of-sample period.
    wealth_bh = (1 + df.loc[s.index, "stock_ret"]).cumprod()
    wealth_tr = (1 + s["ret"]).cumprod()
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(wealth_bh, label="Buy & hold S&P 500", color="#7a7a7a")
    ax[0].plot(wealth_tr, label=f"Trend rule ({best}-month MA)", color="#8c1515")
    ax[0].set_yscale("log"); ax[0].set_ylabel("Growth of $1 (log)")
    ax[0].set_title("Out-of-sample, 1950-2023 (after 0.10% cost per trade)")
    ax[0].legend()
    ax[1].fill_between(wealth_bh.index, wealth_bh / wealth_bh.cummax() - 1, color="#7a7a7a", alpha=.5, label="Buy & hold")
    ax[1].fill_between(wealth_tr.index, wealth_tr / wealth_tr.cummax() - 1, color="#8c1515", alpha=.5, label="Trend rule")
    ax[1].set_ylabel("Drawdown"); ax[1].legend(loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "part1_equity_and_drawdown.png", dpi=150); plt.close(fig)

    return table, crash_tbl, best, t_stat


# ----------------------------------------------------------------------------
# Part 2: does CAPE predict 10-year returns?
# ----------------------------------------------------------------------------
def ols(x: np.ndarray, y: np.ndarray):
    """Ordinary least squares y = a + b*x. Returns (a, b, r_squared)."""
    X = np.column_stack([np.ones_like(x), x])
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - (a + b * x)
    r2 = 1 - resid.var() / y.var()
    return a, b, r2


def run_part2(df: pd.DataFrame) -> dict:
    d = df[df["PE10"] > 0].copy()  # CAPE needs 10 years of earnings history

    # Target: annualized real total return over the NEXT 120 months.
    d["fwd10"] = (d["real_tr_index"].shift(-120) / d["real_tr_index"]) ** (1 / 10) - 1
    d["earn_yield"] = 1 / d["PE10"]  # CAPE earnings yield; linear in expected return
    full = d.dropna(subset=["fwd10"])

    # Full-sample fit (descriptive). Note: overlapping 10-year windows inflate
    # apparent precision, so R-squared here is a description, not a test.
    a, b, r2 = ols(full["earn_yield"].values, full["fwd10"].values)

    # Out-of-sample test: at each month from 1950 on, fit the model using ONLY
    # observations whose 10-year outcome was already known at that date, then
    # forecast. Compare against the naive forecast (historical average return).
    preds, naive, actual = [], [], []
    for date in full.loc[OUT_SAMPLE_START:].index:
        known = full.loc[: date - pd.DateOffset(months=120)]
        if len(known) < 240:
            continue
        aa, bb, _ = ols(known["earn_yield"].values, known["fwd10"].values)
        preds.append(aa + bb * full.loc[date, "earn_yield"])
        naive.append(known["fwd10"].mean())
        actual.append(full.loc[date, "fwd10"])
    preds, naive, actual = map(np.array, (preds, naive, actual))
    oos_r2 = 1 - np.sum((actual - preds) ** 2) / np.sum((actual - naive) ** 2)

    # Forecast from the latest CAPE in the data, using the full-sample fit.
    last_date = d.index.max()
    last_cape = d.loc[last_date, "PE10"]
    implied = a + b / last_cape

    # Chart: CAPE vs. subsequent 10-year real return.
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(full["PE10"], full["fwd10"], s=4, alpha=.35, color="#8c1515")
    xs = np.linspace(full["PE10"].min(), full["PE10"].max(), 200)
    ax.plot(xs, a + b / xs, color="black", lw=1.5, label=f"Fit: return = {a:.3f} + {b:.2f} x (1/CAPE)")
    ax.axvline(last_cape, ls="--", color="#7a7a7a", label=f"CAPE {last_cape:.1f} ({last_date:%b %Y})")
    ax.set_xlabel("CAPE (Shiller P/E)"); ax.set_ylabel("Next 10-yr real return, annualized")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Starting valuation vs. subsequent 10-year returns, 1881-2013 starts")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "part2_cape_vs_returns.png", dpi=150); plt.close(fig)

    res = {"Intercept": a, "Slope on 1/CAPE": b, "Full-sample R2": r2,
           "Out-of-sample R2 vs. historical mean (1950+)": oos_r2,
           "Out-of-sample forecasts": len(preds),
           "Latest CAPE": last_cape, "Latest CAPE date": f"{last_date:%Y-%m}",
           "Implied 10-yr real return": implied}
    pd.Series(res).to_csv(OUT / "part2_results.csv", header=["value"])
    return res


# ----------------------------------------------------------------------------
def main():
    OUT.mkdir(exist_ok=True)
    df = load_data()
    print(f"Data: {df.index.min():%Y-%m} to {df.index.max():%Y-%m} ({len(df)} months)\n")

    table, crashes, best, t_stat = run_part1(df)
    pct = lambda x: f"{x:.1%}"
    fmt = {"CAGR": pct, "Volatility": pct, "Sharpe": "{:.2f}".format, "Max drawdown": pct,
           "Worst month": pct, "Time in stocks": pct, "Trades per year": "{:.1f}".format}
    print(f"PART 1 - Lookback chosen in-sample: {best} months")
    print(table.to_string(formatters=fmt, na_rep="-"))
    print(f"\nOut-of-sample t-stat on monthly return difference (trend - buy&hold): {t_stat:.2f}")
    print("\nCrash episodes (cumulative return):")
    print(crashes.to_string(float_format=lambda x: f"{x:.1%}"))

    r = run_part2(df)
    print("\nPART 2 - CAPE vs. next-10-year real return")
    for k, v in r.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nCharts and tables written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
