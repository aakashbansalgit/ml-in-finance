# ml-in-finance

Can machine learning forecast implied volatility better than a simple regression? This tests a random forest and a neural network against HAR regressions and a random walk, forecasting the VIX 1 and 21 trading days ahead, out of sample from 2012 to September 2025.

## Where this came from

It grew out of a six-person group project for Machine Learning in Finance (IE 517) at the University of Illinois. That project predicted the implied volatility of individual SPX options from features that included the option's own bid, ask and mid price, the strike, the expiry and the greeks. Implied volatility is a deterministic inversion of the price and those same contract terms, so the network was relearning the Black-Scholes formula rather than predicting anything, and a random train/test split let neighbouring quotes from the same day sit on both sides. I rebuilt the question as a genuine forecast, using only information available on the day the forecast is made.

## Setup

Daily closes of the VIX, the 3-month VIX and the S&P 500 from July 2006, from Yahoo Finance. The target is the change in log VIX over the next h days. Features on day t use closes up to day t:

- log VIX and its 5-day and 22-day averages, the three HAR terms
- the slope from the VIX to the 3-month VIX
- S&P 500 returns over 1, 5 and 22 days
- realised volatility over 5 and 22 days, and the gap between implied and realised

Every model is refitted at the start of each test year on all earlier data. Training rows whose target ends inside the test year are dropped, so the 21-day overlap cannot leak. The neural network, two tanh layers of 32 and 16 units, stops early on the latest 20% of its training window rather than a random slice. The random forest has 300 trees with at least 50 observations per leaf.

## Results

Out of sample, 2012 to September 2025. R-squared is measured against the random walk, which forecasts no change. The Diebold-Mariano statistic compares squared errors with HAR, with Newey-West errors for the overlapping horizon, and a negative value favours the model over HAR.

| Model | Horizon | RMSE | R² vs random walk | Direction right | DM vs HAR (p) |
|---|---|---|---|---|---|
| HAR | 1 day | 0.0774 | 1.6% | 53.3% | |
| HAR, all features | 1 day | 0.0774 | 1.6% | 53.4% | 0.11 (0.92) |
| Random forest | 1 day | 0.0773 | 2.0% | 54.7% | -0.78 (0.43) |
| Neural network | 1 day | 0.0771 | 2.5% | 54.7% | -1.64 (0.10) |
| HAR | 21 days | 0.2362 | 13.3% | 58.3% | |
| HAR, all features | 21 days | 0.2371 | 12.6% | 59.7% | 0.98 (0.33) |
| Random forest | 21 days | 0.2452 | 6.5% | 60.4% | 3.03 (0.003) |
| Neural network | 21 days | 0.2414 | 9.4% | 59.9% | 1.86 (0.06) |

A day ahead, the VIX is close to a random walk, and nothing explains more than 2.5% of the next day's move. The neural network has the lowest error, but the difference from HAR is not significant at 5%.

A month ahead is where the forecastable part shows up, and almost all of it is mean reversion. Three lagged averages of the VIX explain 13% of the 21-day change, and adding returns, realised volatility, the term structure and the variance premium makes the linear model slightly worse. The random forest is significantly worse than HAR (p = 0.003). The two machine learning models call the direction a little more often, but they miss by more when they are wrong.

Year by year in `results/oos_by_year.csv`, the forest has the best single years, 2021 and 2024, and the worst, 2012 and 2017, which is the pattern you would expect from a model fitting noise in calm periods. `figures/har_21d.png` plots the HAR forecast against the VIX 21 days later.

## Running it

```
pip install numpy pandas scipy scikit-learn torch matplotlib yfinance
python vix_forecast.py
```

The first run downloads prices into `data/prices.csv`. The full walk-forward takes under a minute on a laptop CPU.
