"""Leakage checks for the walk-forward study.

Run from the repo root: python -m pytest tests  (or python tests/test_no_leakage.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vix_forecast as vf


def test_features_use_only_past_prices():
    px = vf.load_prices()
    t = px.index[2000]
    shocked = px.copy()
    shocked.loc[shocked.index > t] *= 1.5
    a = vf.build_features(px).loc[:t]
    b = vf.build_features(shocked).loc[:t]
    pd.testing.assert_frame_equal(a, b)


def test_training_targets_end_before_each_test_year(monkeypatch):
    px = vf.load_prices()
    feats = vf.build_features(px)
    h = 21
    seen = []

    def recorder(train, y):
        seen.append(train.index.max())
        return lambda test: np.zeros(len(test))

    monkeypatch.setattr(vf, "make_models", lambda: {"probe": recorder})
    pred = vf.walk_forward(feats, vf.target(px, h), h)

    dates = feats.index
    for last_train, year in zip(seen, sorted(set(pred.index.year))):
        # the target of the last training row is h trading days later
        target_date = dates[dates.get_loc(last_train) + h]
        assert target_date < pd.Timestamp(f"{year}-01-01"), (last_train, target_date, year)


if __name__ == "__main__":
    test_features_use_only_past_prices()

    class _MP:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    test_training_targets_end_before_each_test_year(_MP())
    print("no leakage")
