"""Five baselines for monthly PV forecasts."""

import warnings

from joblib import Parallel, delayed
import numpy as np
from scipy.optimize import least_squares
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX


def bass(history):
    first = np.flatnonzero(history > 0)
    start = max(0, int(first[0]) - 12)
    values = history[start:]
    total = max(values.sum(), 1.0)
    time = np.arange(len(values) + 1) / 12
    lower = [np.log10(1.001), -8, -6]
    upper = [2, 0, np.log10(5)]
    starts = [[np.log10(k), np.log10(p), np.log10(q)] for k in (1.3, 5, 20)
              for p, q in ((0.001, 0.2), (0.02, 0.7))]

    def cumulative(t, parameters):
        capacity = total * 10**parameters[0]
        innovation, imitation = 10**parameters[1], 10**parameters[2]
        decay = np.exp(-(innovation + imitation) * t)
        return capacity * -np.expm1(-(innovation + imitation) * t) / (1 + imitation / innovation * decay)

    fits = [
        least_squares(lambda parameters: (np.diff(cumulative(time, parameters)) - values) / max(values.mean(), 1),
                      initial, bounds=(lower, upper), max_nfev=200, ftol=1e-6, xtol=1e-6, gtol=1e-6, x_scale="jac")
        for initial in starts
    ]
    fits = [f for f in fits if np.isfinite(f.cost)]
    fitted = min([f for f in fits if f.success] or fits, key=lambda f: f.cost)
    future = np.arange(len(values), len(values) + 13) / 12
    return np.maximum(0, np.diff(cumulative(future, fitted.x))), bool(fitted.success)


def local_fit(method, history):
    if history.sum() == 0:
        return np.zeros(12), {"zero_history": True}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if method == "Bass":
                prediction, converged = bass(history)
            else:
                fitted = SARIMAX(np.log1p(history[-60:]), order=(1, 0, 0),
                                 seasonal_order=(0, 1, 0, 12), trend="n",
                                 enforce_stationarity=True).fit(disp=False, maxiter=60)
                prediction = np.expm1(np.clip(fitted.forecast(12), -20, 15))
                converged = bool(fitted.mle_retvals.get("converged", False))
        prediction = np.maximum(0, np.asarray(prediction))
        if not np.isfinite(prediction).all():
            raise ArithmeticError("Non-finite forecast.")
        return prediction, {"converged": converged, "warnings": list(dict.fromkeys(str(w.message) for w in caught))[:2]}
    except Exception as error:
        return np.repeat(history[-12:].mean(), 12), {"fallback": "Mean12", "error": str(error)}


def features(panel, origin, horizon, county_indicators):
    counts = panel["y"]
    last12 = counts[origin-11:origin+1].sum(axis=0)
    previous12 = counts[origin-23:origin-11].sum(axis=0)
    n = counts.shape[1]
    columns = [np.log1p(counts[origin-lag]) for lag in range(12)]
    columns += [np.log1p(counts[origin-span+1:origin+1].mean(axis=0)) for span in (3, 6, 12, 24)]
    columns += [np.log((last12 + 12) / (previous12 + 12)), np.log1p(counts[:origin+1].sum(axis=0))]
    utility_total = np.zeros(n)
    for company in np.unique(panel["utilities"]):
        ids = panel["utilities"] == company
        utility_total[ids] = last12[ids].sum()
    columns += [np.log1p(utility_total), np.full(n, horizon),
                np.full(n, np.sin(2*np.pi*((origin+horizon) % 12)/12)),
                np.full(n, np.cos(2*np.pi*((origin+horizon) % 12)/12)),
                np.full(n, (origin+horizon)/12)]
    columns += [np.log1p(panel["weights"] @ counts[origin-lag]) for lag in range(3)]
    columns += [np.log1p(panel["weights"] @ last12)]
    return np.column_stack((np.column_stack(columns), county_indicators))


def county_boosting(panel):
    counts, dates = panel["y"], panel["dates"]
    cutoff = len(counts) - 1
    indicators = (panel["county"][:, None] == np.unique(panel["county"])[None, :]).astype(float)
    origins = [i for i, date in enumerate(dates) if date.endswith("-12") and date >= "2010-12" and i+12 <= cutoff]
    train = np.vstack([features(panel, origin, h, indicators) for origin in origins for h in range(1, 13)])
    target = np.concatenate([counts[origin+h] for origin in origins for h in range(1, 13)])
    future = np.vstack([features(panel, cutoff, h, indicators) for h in range(1, 13)])
    if len(target) > 60000:
        sample = np.sort(np.random.default_rng(917).choice(len(target), 60000, replace=False))
        train, target = train[sample], target[sample]
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    future = scaler.transform(future)
    fitted = HistGradientBoostingRegressor(
        loss="poisson", max_iter=120, learning_rate=0.07, max_leaf_nodes=15,
        min_samples_leaf=25, l2_regularization=1, early_stopping=False, random_state=17,
    ).fit(train, target)
    return np.maximum(0, fitted.predict(future)).reshape(12, counts.shape[1])


def predict(panel, jobs=4):
    counts = panel["y"]
    predictions = {
        "SeasonalNaive": counts[-12:].copy(),
        "Mean3": np.repeat(counts[-3:].mean(axis=0)[None, :], 12, axis=0),
    }
    diagnostics = {}
    for method in ("Bass", "SARIMA"):
        fitted = Parallel(n_jobs=jobs)(delayed(local_fit)(method, history) for history in counts.T)
        predictions[method] = np.column_stack([pair[0] for pair in fitted])
        diagnostics[method] = {
            "nonconverged": sum(pair[1].get("converged") is False for pair in fitted),
            "fallbacks": sum("fallback" in pair[1] for pair in fitted),
        }
    predictions["HistGBCounty"] = county_boosting(panel)
    return predictions, diagnostics
