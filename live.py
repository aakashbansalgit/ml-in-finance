"""
A live out-of-sample record for the 21-day VIX forecasts.

    python live.py

Once a month every model is refitted on all the data available that day and
its forecast of the VIX 21 trading days ahead is added to
results/live_forecasts.csv. Later runs fill in what the VIX actually did
once the 21 days have passed. A forecast is never changed after it is
written, and the git history dates each one, so the record cannot be fitted
after the fact the way a backtest can.
"""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from vix_forecast import build_features, make_models, target

ROOT = Path(__file__).parent
LEDGER = ROOT / "results" / "live_forecasts.csv"
README = ROOT / "README.md"
START, END = "<!-- live:start -->", "<!-- live:end -->"
H = 21
MODELS = ["Random walk", "HAR", "HAR, all features", "Random forest", "Neural network"]
NY = ZoneInfo("America/New_York")


def latest_prices():
    """Daily closes to the last completed session. Today's row is dropped
    before the close, because Yahoo reports the intraday price there."""
    import yfinance as yf
    px = yf.download(["^VIX", "^VIX3M", "^GSPC"], start="2006-01-01",
                     auto_adjust=True, progress=False)["Close"]
    px = px.rename(columns={"^VIX": "vix", "^VIX3M": "vix3m", "^GSPC": "spx"}).dropna()
    now = datetime.now(NY)
    if now.hour < 17 and px.index[-1].date() == now.date():
        px = px.iloc[:-1]
    return px


def fill_outcomes(ledger, px):
    dates = px.index
    for i, r in ledger[ledger["vix_after"].isna()].iterrows():
        k = dates.searchsorted(pd.Timestamp(r["made_on"])) + H
        if k < len(dates):
            ledger.loc[i, "target_date"] = dates[k].strftime("%Y-%m-%d")
            ledger.loc[i, "vix_after"] = round(float(px["vix"].iloc[k]), 2)
    return ledger


def forecast(px):
    feats = build_features(px)
    asof = feats.index[-1]
    train = feats.assign(y=target(px, H)).dropna()
    row = {"made_on": asof.strftime("%Y-%m-%d"), "vix": round(float(px["vix"].iloc[-1]), 2)}
    for name, fit in make_models().items():
        change = fit(train, train["y"].values)(feats.loc[[asof]])[0]
        row[name] = round(float(px["vix"].iloc[-1] * np.exp(change)), 2)
    row["target_date"], row["vix_after"] = None, np.nan
    return row


def update_readme(ledger):
    short = {"Random walk": "Random walk", "HAR": "HAR", "HAR, all features": "HAR, all",
             "Random forest": "Forest", "Neural network": "Network"}
    lines = ["| Made on | VIX | " + " | ".join(short[m] for m in MODELS) + " | VIX 21 days later |",
             "|---" * (len(MODELS) + 3) + "|"]
    for _, r in ledger.iloc[::-1].iterrows():
        after = f"{r.vix_after:.2f} ({r.target_date})" if pd.notna(r.vix_after) else "pending"
        lines.append(f"| {r.made_on} | {r.vix:.2f} | "
                     + " | ".join(f"{r[m]:.2f}" for m in MODELS) + f" | {after} |")
    done = ledger.dropna(subset=["vix_after"])
    lines.append("")
    if len(done):
        err = {m: np.sqrt(np.mean(np.log(done["vix_after"] / done[m]) ** 2)) for m in MODELS}
        lines.append(f"Error so far on {len(done)} scored forecast{'s' if len(done) > 1 else ''}, "
                     "RMSE of the log VIX: "
                     + ", ".join(f"{short[m]} {e:.3f}" for m, e in err.items()) + ".")
    else:
        lines.append("No forecast has reached its 21 days yet.")
    text = README.read_text(encoding="utf-8")
    before, rest = text.split(START)
    README.write_text(before + START + "\n" + "\n".join(lines) + "\n" + END + rest.split(END)[1],
                      encoding="utf-8")


def main():
    px = latest_prices()
    ledger = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame(
        columns=["made_on", "vix", *MODELS, "target_date", "vix_after"])
    ledger["target_date"] = ledger["target_date"].astype(object)
    ledger = fill_outcomes(ledger, px)

    month = px.index[-1].strftime("%Y-%m")
    if not ledger["made_on"].astype(str).str.startswith(month).any():
        new = forecast(px)
        new = pd.DataFrame([new])
        ledger = new if ledger.empty else pd.concat([ledger, new], ignore_index=True)
        print(new.to_string(index=False))
    ledger.to_csv(LEDGER, index=False)
    update_readme(ledger)


if __name__ == "__main__":
    main()
