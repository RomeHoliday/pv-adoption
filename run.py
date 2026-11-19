"""Fit and evaluate monthly PV forecasts."""

import argparse
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

import bandit
import baselines
import model

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "results"
LABELS = {
    "SeasonalNaive": "Same month last year",
    "Mean3": "Mean of the last three months",
    "Bass": "Basic Bass diffusion",
    "SARIMA": "Fixed-order seasonal ARIMA",
    "HistGBCounty": "Poisson boosting with same-county history",
    "Own": "Markov using the local area only",
    "Markov": "Markov using the local area and neighbors",
    "Distance": "Distance-weighted Markov",
}


def load(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def metrics(actual, prediction):
    error = prediction - actual
    total = actual.sum()
    return {
        "wape_pct": 100 * np.abs(error).sum() / total,
        "mae": np.abs(error).mean(),
        "rmse": np.sqrt(np.square(error).mean()),
        "bias_pct": 100 * error.sum() / total,
    }


def refit(panel, reference, jobs):
    forecasts = {name: np.empty((36, 1015)) for name in reference["methods"]}
    diagnostics = {}
    for block, year in enumerate((2023, 2024, 2025)):
        rows = slice(block*12, (block+1)*12)
        cutoff = panel["dates"].tolist().index(f"{year-1}-12") + 1
        visible = {**panel, "y": panel["y"][:cutoff], "dates": panel["dates"][:cutoff]}
        points, diagnostics[str(year)] = baselines.predict(visible, jobs=jobs)
        for name, forecast in points.items():
            forecasts[name][rows] = forecast
        for utility in ("PGE", "SDGE"):
            case = load(DATA / "cases" / f"{utility}_{year}.npz")
            graph = load_npz(DATA / "cases" / f"{utility}_graph.npz").tocsr()
            graph.sort_indices()
            ids = panel["utilities"] == utility
            np.testing.assert_array_equal(case["regions"], panel["regions"][ids])
            assert str(case["origin"]) == f"{year-1}-12" and max(case["dates"]) <= str(case["origin"])
            own = model.fit(case, graph, neighbors=False)
            edges = model.fit(case, graph)
            gamma = model.fit_distance(case, graph, edges[:, 0])
            for name, coefficients, distance in (("Own", own, None), ("Markov", edges, None), ("Distance", edges, gamma)):
                forecasts[name][rows, ids] = model.predict(case, graph, coefficients, distance=distance)
            diagnostics[str(year)][utility] = {"distance_coefficient": gamma}
            print(f"Fitted {utility} for {year}.", flush=True)
    return np.stack([forecasts[name] for name in reference["methods"]]), diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit", action="store_true", help="Refit the models from the prepared training cases.")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel jobs for Bass and SARIMA.")
    args = parser.parse_args()
    panel, reference = load(DATA / "panel.npz"), load(DATA / "reference.npz")
    np.testing.assert_array_equal(panel["regions"], reference["regions"])
    np.testing.assert_array_equal(panel["utilities"], reference["utilities"])
    start = panel["dates"].tolist().index("2023-01")
    actual = panel["y"][start:start+36]
    assert actual.shape == (36, 1015) and actual.sum() == 297252
    forecasts, diagnostics = reference["forecast"].copy(), {}
    if args.refit:
        forecasts, diagnostics = refit(panel, reference, args.jobs)
        np.testing.assert_allclose(forecasts, reference["forecast"], rtol=1e-9, atol=1e-7)
    else:
        for i, year in enumerate((2023, 2024, 2025)):
            end = panel["dates"].tolist().index(f"{year-1}-12") + 1
            np.testing.assert_array_equal(forecasts[0, i*12:(i+1)*12], panel["y"][end-12:end])
            np.testing.assert_array_equal(forecasts[1, i*12:(i+1)*12],
                                          np.repeat(panel["y"][end-3:end].mean(axis=0)[None], 12, axis=0))
    markov = forecasts[reference["methods"].tolist().index("Markov")]
    policies, actions = bandit.predict(panel, markov, pd.read_csv(DATA / "search.csv"))
    np.testing.assert_array_equal(actions, reference["actions"])
    np.testing.assert_array_equal(policies, reference["bandit"])
    rows = [{"method": LABELS[name], **metrics(actual, point)} for name, point in zip(reference["methods"], forecasts)]
    for method, predictions in zip(("Markov + Bandit", "Markov + Search + Bandit"), policies):
        runs = [metrics(actual, point) for point in predictions]
        rows.append({"method": method, **{key: np.mean([row[key] for row in runs]) for key in runs[0]}})
    table = pd.DataFrame(rows)
    yearly = []
    for i, year in enumerate((2023, 2024, 2025)):
        part = slice(i*12, (i+1)*12)
        yearly.append({"year": year, "actual": actual[part].sum(), "forecast": markov[part].sum(),
                       **metrics(actual[part], markov[part])})
    region_actual = actual.sum(axis=0)
    region_error = np.abs(markov - actual).sum(axis=0)
    regional = pd.DataFrame({"region": panel["regions"], "actual": region_actual,
                             "absolute_error": region_error,
                             "wape_pct": np.divide(100*region_error, region_actual,
                                                   out=np.full(1015, np.nan), where=region_actual > 0)})
    OUTPUT.mkdir(exist_ok=True)
    table.to_csv(OUTPUT / "scores.csv", index=False)
    pd.DataFrame(yearly).to_csv(OUTPUT / "yearly.csv", index=False)
    regional.to_csv(OUTPUT / "regions.csv", index=False)
    np.savez_compressed(OUTPUT / "predictions.npz", forecast=forecasts, bandit=policies, actions=actions)
    (OUTPUT / "verification.json").write_text(json.dumps({
        "mode": "refit" if args.refit else "reference replay", "prediction_rows": actual.size,
        "forecast_reference_matched": True, "bandit_runs_reproduced": 20, "fits": diagnostics,
    }, indent=2) + "\n")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved tables to {OUTPUT}")


if __name__ == "__main__":
    main()
