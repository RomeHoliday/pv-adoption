"""Annual multiplier choices with delayed, selected-action feedback."""

import numpy as np

MULTIPLIERS = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
SEEDS = (17, 29, 43, 59, 71, 83, 97, 109, 127, 139)
TIE_ORDER = (2, 1, 3, 0, 4)


def search_states(search):
    states = []
    for year in (2022, 2023, 2024):
        end = search.month.tolist().index(f"{year}-12") + 1
        values = search[["PGE", "SDGE"]].to_numpy()
        ratio = values[end-3:end].mean(axis=0) / values[end-15:end-3].mean(axis=0)
        states.append(np.where(ratio > 1.1, 2, np.where(ratio < 0.9, 0, 1)))
    return np.array(states)


def replay(panel, baseline, states, seed, use_search):
    companies = ("PGE", "SDGE")
    groups = [np.flatnonzero(panel["utilities"] == company) for company in companies]
    historic = (panel["dates"] >= "2018-01") & (panel["dates"] <= "2019-12")
    scales = [panel["y"][historic][:, group].sum(axis=1).mean() for group in groups]
    start = panel["dates"].tolist().index("2023-01")
    actual = panel["y"][start:start+36]
    scores = np.zeros((2, 3 if use_search else 1, 5))
    actions = np.empty((36, 2), dtype=int)
    prediction = np.empty_like(baseline)
    rng = np.random.default_rng(seed)
    for year in range(3):
        if year:
            for month in range((year-1)*12, year*12):
                for company, group in enumerate(groups):
                    state = int(states[year-1, company]) if use_search else 0
                    action = actions[month, company]
                    before = np.abs(baseline[month, group] - actual[month, group]).sum()
                    after = np.abs(baseline[month, group] * MULTIPLIERS[action] - actual[month, group]).sum()
                    reward = (before - after) / scales[company]
                    old = scores[company, state, action]
                    scores[company, state, action] = 0.9 * old + 0.1 * reward
        for month in range(year*12, (year+1)*12):
            for company, group in enumerate(groups):
                state = int(states[year, company]) if use_search else 0
                trial = rng.random()
                random_action = int(rng.integers(5))
                best = max(TIE_ORDER, key=lambda action: scores[company, state, action])
                action = random_action if trial < 0.1 else best
                actions[month, company] = action
                prediction[month, group] = baseline[month, group] * MULTIPLIERS[action]
    return prediction, actions


def predict(panel, baseline, search):
    states = search_states(search)
    predictions, decisions = [], []
    for use_search in (False, True):
        runs = [replay(panel, baseline, states, seed, use_search) for seed in SEEDS]
        predictions.append(np.stack([run[0] for run in runs]))
        decisions.append(np.stack([run[1] for run in runs]))
    return np.stack(predictions), np.stack(decisions)
