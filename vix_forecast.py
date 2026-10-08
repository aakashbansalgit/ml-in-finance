"""
Forecasting the VIX one day and one month ahead.

Compares a random walk, HAR regressions, a random forest and a small neural
network out of sample. Everything is refitted at the start of each test year
on data up to that point, and training rows whose target falls inside the test
year are dropped so that overlapping horizons cannot leak.

    python vix_forecast.py

Writes results/oos_metrics.csv, results/oos_by_year.csv and figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "prices.csv"
HORIZONS = (1, 21)
FIRST_TEST_YEAR = 2012
SEED = 0

HAR = ["log_vix", "log_vix_5d", "log_vix_22d"]
ALL = HAR + ["slope", "ret_1d", "ret_5d", "ret_22d", "log_rv_5d", "log_rv_22d", "vrp"]


# ---- data and features -----------------------------------------------------

def load_prices():
    if DATA.exists():
        return pd.read_csv(DATA, index_col=0, parse_dates=True)
    import yfinance as yf
    px = yf.download(["^VIX", "^VIX3M", "^GSPC"], start="2006-01-01",
                     end="2025-10-01", auto_adjust=True, progress=False)["Close"]
    px = px.rename(columns={"^VIX": "vix", "^VIX3M": "vix3m", "^GSPC": "spx"})
    px = px.dropna()
    DATA.parent.mkdir(exist_ok=True)
    px.to_csv(DATA)
    return px


def build_features(px):
    """Every feature on day t uses closes up to and including day t."""
    f = pd.DataFrame(index=px.index)
    lv = np.log(px["vix"])
    r = np.log(px["spx"]).diff()

    f["log_vix"] = lv
    f["log_vix_5d"] = lv.rolling(5).mean()
    f["log_vix_22d"] = lv.rolling(22).mean()
    f["slope"] = np.log(px["vix3m"] / px["vix"])     # positive in contango
    f["ret_1d"] = r
    f["ret_5d"] = r.rolling(5).sum()
    f["ret_22d"] = r.rolling(22).sum()
    f["log_rv_5d"] = np.log(r.rolling(5).std() * np.sqrt(252) * 100)
    f["log_rv_22d"] = np.log(r.rolling(22).std() * np.sqrt(252) * 100)
    f["vrp"] = lv - f["log_rv_22d"]                   # implied over realised
    return f


def target(px, h):
    """Change in log VIX from day t to day t + h."""
    lv = np.log(px["vix"])
    return lv.shift(-h) - lv


# ---- models ----------------------------------------------------------------

class MLP(torch.nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, 32), torch.nn.Tanh(),
            torch.nn.Linear(32, 16), torch.nn.Tanh(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def fit_mlp(X, y, epochs=400, patience=30):
    """Train on the first 80% of the window and stop early on the last 20%,
    so the validation set is always later in time than the training set."""
    torch.manual_seed(SEED)
    split = int(len(X) * 0.8)
    scaler = StandardScaler().fit(X[:split])
    y_scale = y[:split].std()

    def t(a):
        return torch.tensor(a, dtype=torch.float32)

    Xtr, Xva = t(scaler.transform(X[:split])), t(scaler.transform(X[split:]))
    ytr, yva = t(y[:split] / y_scale)[:, None], t(y[split:] / y_scale)[:, None]

    model = MLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    best, best_state, wait = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xva), yva).item()
        if val < best - 1e-6:
            best, wait = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()

    def predict(Xnew):
        with torch.no_grad():
            return model(t(scaler.transform(Xnew))).numpy().ravel() * y_scale
    return predict


def make_models():
    def linear(cols):
        def fit(Xdf, y):
            m = LinearRegression().fit(Xdf[cols].values, y)
            return lambda Xnew: m.predict(Xnew[cols].values)
        return fit

    def forest(Xdf, y):
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=50,
                                  max_features=0.5, n_jobs=-1,
                                  random_state=SEED).fit(Xdf[ALL].values, y)
        return lambda Xnew: m.predict(Xnew[ALL].values)

    def mlp(Xdf, y):
        p = fit_mlp(Xdf[ALL].values, y)
        return lambda Xnew: p(Xnew[ALL].values)

    return {
        "Random walk": lambda Xdf, y: (lambda Xnew: np.zeros(len(Xnew))),
        "HAR": linear(HAR),
        "HAR, all features": linear(ALL),
        "Random forest": forest,
        "Neural network": mlp,
    }


# ---- walk-forward evaluation -----------------------------------------------

def walk_forward(features, y, h):
    data = features.assign(y=y).dropna()
    preds = []
    for year in range(FIRST_TEST_YEAR, data.index[-1].year + 1):
        start = pd.Timestamp(f"{year}-01-01")
        test = data[(data.index >= start) & (data.index.year == year)]
        if test.empty:
            continue
        # Drop training rows whose h-day target ends on or after the test start.
        train = data[data.index < start].iloc[:-h]
        out = pd.DataFrame({"actual": test["y"]}, index=test.index)
        for name, fit in make_models().items():
            out[name] = fit(train, train["y"].values)(test)
        preds.append(out)
    return pd.concat(preds)


def diebold_mariano(e_model, e_bench, h):
    """DM test on squared errors with Newey-West variance (lag h - 1).
    Negative statistic means the model has the smaller loss."""
    d = e_model ** 2 - e_bench ** 2
    n = len(d)
    d_c = d - d.mean()
    var = np.sum(d_c ** 2) / n
    for lag in range(1, h):
        w = 1 - lag / h
        var += 2 * w * np.sum(d_c[lag:] * d_c[:-lag]) / n
    stat = d.mean() / np.sqrt(var / n)
    return stat, 2 * stats.norm.sf(abs(stat))


def score(pred, h):
    actual = pred["actual"].values
    sse_rw = np.sum(actual ** 2)
    e_har = actual - pred["HAR"].values
    rows = {}
    for name in pred.columns.drop("actual"):
        e = actual - pred[name].values
        row = {
            "rmse": np.sqrt(np.mean(e ** 2)),
            "r2_vs_random_walk": 1 - np.sum(e ** 2) / sse_rw,
            "direction_hit_rate": np.mean(np.sign(pred[name]) == np.sign(actual))
            if name != "Random walk" else np.nan,
        }
        if name not in ("HAR", "Random walk"):
            row["dm_vs_har"], row["dm_p"] = diebold_mariano(e, e_har, h)
        rows[name] = row
    return pd.DataFrame(rows).T


def main():
    px = load_prices()
    feats = build_features(px)
    out = ROOT / "results"
    figs = ROOT / "figures"
    out.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)

    tables, by_year = [], []
    for h in HORIZONS:
        pred = walk_forward(feats, target(px, h), h)
        table = score(pred, h)
        table.insert(0, "horizon_days", h)
        tables.append(table)

        yr = pred.groupby(pred.index.year).apply(
            lambda g: pd.Series({c: 1 - np.sum((g["actual"] - g[c]) ** 2)
                                 / np.sum(g["actual"] ** 2)
                                 for c in g.columns.drop(["actual", "Random walk"])}))
        yr.insert(0, "horizon_days", h)
        by_year.append(yr)

        if h == 21:
            fig, ax = plt.subplots(figsize=(10, 4))
            lv = np.log(px["vix"]).reindex(pred.index)
            np.exp(lv + pred["actual"]).plot(ax=ax, color="black", lw=0.8,
                                             label="VIX 21 days later")
            np.exp(lv + pred["HAR"]).plot(ax=ax, lw=0.8, label="HAR forecast")
            ax.set_yscale("log")
            ax.set_xlabel("")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(figs / "har_21d.png", dpi=120)

    metrics = pd.concat(tables)
    metrics.to_csv(out / "oos_metrics.csv", float_format="%.4f")
    pd.concat(by_year).to_csv(out / "oos_by_year.csv", float_format="%.4f")

    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(metrics)


if __name__ == "__main__":
    main()
