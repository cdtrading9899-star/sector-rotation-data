"""
NSE Sector Rotation — EOD Data Fetcher & RRG Calculator
=========================================================

WHAT THIS DOES
---------------
1. Pulls EOD (end-of-day) OHLC history for NIFTY 50 (benchmark) and the major
   NSE sectoral indices from Yahoo Finance (via the `yfinance` library).
   Yahoo Finance aggregates exchange-sourced data and is the most widely used
   free, programmatic EOD source for NSE indices among retail/systematic
   traders — it is NOT the exchange's own feed. If you need exchange-of-record
   data, use NSE's own bhavcopy / index-history downloads (see NOTES below).
2. Computes JdK-style RS-Ratio / RS-Momentum (the standard Relative Rotation
   Graph, "RRG", methodology) for each sector versus NIFTY 50.
3. Computes multi-timeframe returns (1D/1W/1M/3M/6M/1Y) per sector.
4. Writes a single JSON file that the companion React dashboard
   (sector_rotation_dashboard.jsx) reads directly — paste its contents into
   the dashboard's "Load data" panel, or host the JSON somewhere with CORS
   enabled (e.g. a GitHub Gist / raw.githubusercontent.com) and paste that
   URL into the dashboard instead.

WHY THE DASHBOARD CAN'T "AUTO-PULL" ON ITS OWN
------------------------------------------------
The React dashboard runs in a sandboxed browser environment with no general
internet access — it cannot call the NSE or Yahoo Finance APIs directly, and
even in a normal browser both would block the request (no CORS headers, and
NSE actively blocks non-browser traffic). The realistic automated pipeline is:

    this script (run on YOUR machine / a cron job / GitHub Action)
        --> writes sector_data.json
        --> dashboard reads it (paste, upload, or fetch-by-URL if you host it)

Suggested automation: schedule this script via cron/Task Scheduler for ~15-30
min after NSE close (market closes 15:30 IST, indices settle shortly after),
commit sector_data.json to a public GitHub repo, and point the dashboard's
"Load from URL" field at the raw.githubusercontent.com URL. The dashboard
will then reflect the latest push whenever you reopen it.

INSTALL
-------
    pip install yfinance pandas numpy --break-system-packages

USAGE
-----
    python fetch_sector_data.py

Produces ./sector_data.json in the current directory.

NOTES ON DATA SOURCES / VERIFICATION
-------------------------------------
- Yahoo tickers for NSE sectoral indices occasionally change or get delisted
  as NSE renames/restructures indices. The TICKERS dict below reflects
  commonly-used symbols as of early 2026 — verify each resolves before relying
  on it (the script prints a warning and skips any ticker that fails to
  download rather than silently producing bad data).
- For exchange-of-record EOD data instead of Yahoo's feed, NSE publishes
  official daily bhavcopy and index-history files at nseindia.com (requires
  session/cookie handling and is rate-limited — the `nsepython` or `jugaad-data`
  libraries wrap this). Swap out `download_yahoo()` for an NSE-based fetcher
  if you need that instead; the rest of the pipeline (RRG math, JSON schema)
  is unchanged.
"""

import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit(
        "Missing dependency. Run: pip install yfinance pandas numpy --break-system-packages"
    ) from e

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BENCHMARK_TICKER = "^NSEI"
BENCHMARK_NAME = "NIFTY 50"

# Yahoo Finance tickers for NSE sectoral / thematic indices.
# VERIFY these against live Yahoo data before trusting them — symbols drift.
TICKERS = {
    "NIFTY BANK":        "^NSEBANK",
    "NIFTY IT":          "^CNXIT",
    "NIFTY AUTO":        "^CNXAUTO",
    "NIFTY PHARMA":      "^CNXPHARMA",
    "NIFTY FMCG":        "^CNXFMCG",
    "NIFTY METAL":       "^CNXMETAL",
    "NIFTY REALTY":      "^CNXREALTY",
    "NIFTY MEDIA":       "^CNXMEDIA",
    "NIFTY ENERGY":      "^CNXENERGY",
    "NIFTY PSU BANK":    "^CNXPSUBANK",
    "NIFTY PVT BANK":    "NIFTY_PVT_BANK.NS",
    "NIFTY FIN SERVICE": "NIFTY_FIN_SERVICE.NS",
    "NIFTY INFRA":       "^CNXINFRA",
    "NIFTY COMMODITIES": "^CNXCMDT",
}

LOOKBACK_PERIOD = "2y"        # history window to download
RRG_ZSCORE_WINDOW = 63        # ~3 trading months, standard RRG smoothing window
RRG_MOMENTUM_WINDOW = 63
TAIL_WEEKS = 10                # number of weekly points to show in the RRG trail
OUTPUT_PATH = "sector_data.json"

# ---------------------------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------------------------

def download_yahoo(ticker: str) -> pd.Series | None:
    """Download EOD close series for one ticker. Returns None on failure."""
    try:
        df = yf.download(ticker, period=LOOKBACK_PERIOD, interval="1d",
                          auto_adjust=True, progress=False)
        if df.empty:
            print(f"  [skip] {ticker}: no data returned")
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):  # yfinance sometimes returns MultiIndex cols
            close = close.iloc[:, 0]
        close.name = ticker
        return close
    except Exception as exc:
        print(f"  [skip] {ticker}: {exc}")
        return None


# ---------------------------------------------------------------------------
# RRG MATH  (JdK RS-Ratio / RS-Momentum, standard normalization)
# ---------------------------------------------------------------------------

def compute_rrg(sector_close: pd.Series, bench_close: pd.Series) -> pd.DataFrame:
    """
    RS-Ratio and RS-Momentum, both centered at 100:
      RS            = sector / benchmark
      RS-Ratio      = 100 + rolling z-score of RS
      RS-Momentum   = 100 + rolling z-score of the day-over-day change in RS-Ratio
    This is a widely-used simplified form of the JdK RRG methodology
    (exact proprietary StockCharts constants differ slightly but the quadrant
    behavior — leading / weakening / lagging / improving — is equivalent).
    """
    df = pd.DataFrame({"sector": sector_close, "bench": bench_close}).dropna()
    rs = df["sector"] / df["bench"] * 100

    rs_mean = rs.rolling(RRG_ZSCORE_WINDOW).mean()
    rs_std = rs.rolling(RRG_ZSCORE_WINDOW).std()
    rs_ratio = 100 + (rs - rs_mean) / rs_std

    rs_ratio_diff = rs_ratio.diff()
    mom_mean = rs_ratio_diff.rolling(RRG_MOMENTUM_WINDOW).mean()
    mom_std = rs_ratio_diff.rolling(RRG_MOMENTUM_WINDOW).std()
    rs_momentum = 100 + (rs_ratio_diff - mom_mean) / mom_std

    out = pd.DataFrame({"rs_ratio": rs_ratio, "rs_momentum": rs_momentum})
    return out.dropna()


def quadrant(rs_ratio: float, rs_momentum: float) -> str:
    if rs_ratio >= 100 and rs_momentum >= 100:
        return "Leading"
    if rs_ratio >= 100 and rs_momentum < 100:
        return "Weakening"
    if rs_ratio < 100 and rs_momentum < 100:
        return "Lagging"
    return "Improving"


def pct_return(close: pd.Series, days: int) -> float | None:
    if len(close) <= days:
        return None
    return float((close.iloc[-1] / close.iloc[-1 - days] - 1) * 100)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print(f"Downloading benchmark: {BENCHMARK_NAME} ({BENCHMARK_TICKER})")
    bench = download_yahoo(BENCHMARK_TICKER)
    if bench is None:
        raise SystemExit("Could not download benchmark data — aborting.")

    sectors_out = []

    for name, ticker in TICKERS.items():
        print(f"Downloading: {name} ({ticker})")
        close = download_yahoo(ticker)
        if close is None:
            continue

        rrg = compute_rrg(close, bench)
        if rrg.empty:
            print(f"  [skip] {name}: not enough history for RRG window")
            continue

        # weekly-resampled tail for a smoother RRG trail (standard practice)
        weekly = rrg.resample("W").last().dropna().tail(TAIL_WEEKS)
        tail = [
            {"date": idx.strftime("%Y-%m-%d"),
             "rs_ratio": round(row.rs_ratio, 2),
             "rs_momentum": round(row.rs_momentum, 2)}
            for idx, row in weekly.iterrows()
        ]

        current = rrg.iloc[-1]
        ltp = float(close.iloc[-1])

        sectors_out.append({
            "name": name,
            "ticker": ticker,
            "ltp": round(ltp, 2),
            "returns": {
                "1D": round(pct_return(close, 1), 2) if pct_return(close, 1) is not None else None,
                "1W": round(pct_return(close, 5), 2) if pct_return(close, 5) is not None else None,
                "1M": round(pct_return(close, 21), 2) if pct_return(close, 21) is not None else None,
                "3M": round(pct_return(close, 63), 2) if pct_return(close, 63) is not None else None,
                "6M": round(pct_return(close, 126), 2) if pct_return(close, 126) is not None else None,
                "1Y": round(pct_return(close, 252), 2) if pct_return(close, 252) is not None else None,
            },
            "current": {
                "rs_ratio": round(float(current.rs_ratio), 2),
                "rs_momentum": round(float(current.rs_momentum), 2),
                "quadrant": quadrant(current.rs_ratio, current.rs_momentum),
            },
            "tail": tail,
        })

    payload = {
        "asof": bench.index[-1].strftime("%Y-%m-%d"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark": BENCHMARK_NAME,
        "benchmark_ticker": BENCHMARK_TICKER,
        "rrg_params": {
            "zscore_window": RRG_ZSCORE_WINDOW,
            "momentum_window": RRG_MOMENTUM_WINDOW,
            "tail_weeks": TAIL_WEEKS,
        },
        "sectors": sectors_out,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote {len(sectors_out)} sectors to {OUTPUT_PATH} (as of {payload['asof']})")


if __name__ == "__main__":
    main()
