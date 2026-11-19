"""Poisson Markov forecasts with own-area and neighbor residuals."""

import numpy as np
from scipy.sparse import csr_matrix

SEEDS = (17, 29, 43)
PATHS = 4096
RIDGE = 60.0


def project_l1(values):
    """Project each coefficient row onto the unit L1 ball."""
    magnitude = np.abs(values)
    ordered = np.sort(magnitude, axis=-1)[..., ::-1]
    shifted = (np.cumsum(ordered, axis=-1) - 1) / np.arange(1, values.shape[-1] + 1)
    count = (ordered > shifted).sum(axis=-1)
    threshold = np.take_along_axis(shifted, np.maximum(count - 1, 0)[..., None], axis=-1)[..., 0]
    projected = np.sign(values) * np.maximum(0, magnitude - np.maximum(threshold, 0)[..., None])
    return np.where((magnitude.sum(axis=-1) <= 1)[..., None], values, projected)


def solve(quadratic, linear, denominator):
    """Fit the ridge objective under a unit absolute row-gain constraint."""
    coefficients = project_l1(np.linalg.solve(quadratic, linear[..., None])[..., 0])
    lipschitz = np.linalg.eigvalsh(quadratic)[..., -1]
    accelerated = coefficients.copy()
    momentum = 1.0
    for _ in range(5000):
        gradient = np.einsum("nij,nj->ni", quadratic, accelerated, optimize=False) - linear
        proposal = project_l1(accelerated - gradient / lipschitz[:, None])
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        accelerated = proposal + (momentum - 1) / next_momentum * (proposal - coefficients)
        coefficients, momentum = proposal, next_momentum
        gradient = np.einsum("nij,nj->ni", quadratic, coefficients, optimize=False) - linear
        residual = (coefficients - project_l1(coefficients - gradient / lipschitz[:, None]))
        residual *= lipschitz[:, None] / denominator[:, None]
        if np.abs(residual).max() <= 1e-8:
            break
    if np.abs(residual).max() > 1e-6:
        raise ArithmeticError("Coefficient optimization did not converge.")
    return coefficients


def fit(case, graph, neighbors=True):
    scale = case["scale"]
    residual = case["r"] / scale
    target = (case["y"] - case["b"]) / scale
    weights = case["weight"]
    features = residual[..., None]
    penalty = np.ones((len(scale), 1))
    if neighbors:
        width = max(int(np.diff(graph.indptr).max(initial=0)), 1)
        indices = np.zeros((len(scale), width), dtype=int)
        square_root_weights = np.zeros((len(scale), width))
        for i in range(len(scale)):
            start, stop = graph.indptr[i:i+2]
            indices[i, :stop-start] = graph.indices[start:stop]
            square_root_weights[i, :stop-start] = np.sqrt(graph.data[start:stop])
        neighbor_weights = square_root_weights**2
        features = np.concatenate((features, residual[..., indices] * (neighbor_weights > 0)), axis=-1)
        neighbor_penalty = np.divide(1, neighbor_weights, out=np.ones_like(neighbor_weights),
                                     where=neighbor_weights > 0)
        penalty = np.concatenate((penalty, neighbor_penalty), axis=-1)
    quadratic = np.einsum("tni,tn,tnj->nij", features, weights, features, optimize=False)
    diagonal = np.arange(features.shape[-1])
    quadratic[:, diagonal, diagonal] += RIDGE * penalty
    linear = np.einsum("tni,tn->ni", features, weights * target, optimize=False)
    denominator = (weights * target**2).sum(axis=0) + RIDGE
    return solve(quadratic, linear, denominator)


def fit_distance(case, graph, own):
    peer = np.asarray((graph @ case["r"].T).T)
    target = case["y"] - case["b"] - own * case["r"]
    denominator = np.sum(case["weight"] * peer**2)
    numerator = np.sum(case["weight"] * peer * target)
    nonempty = np.diff(graph.indptr) > 0
    bound = np.min(1 - np.abs(own[nonempty])) if nonempty.any() else 0.0
    return float(np.clip(numerator / denominator, -bound, bound)) if denominator else 0.0


def simulate(case, graph, coefficients, paths, seed, distance=None):
    graph = graph.copy()
    graph.sort_indices()
    n = len(case["scale"])
    if distance is None:
        edge_values = []
        for i in range(n):
            degree = graph.indptr[i+1] - graph.indptr[i]
            edge_values.extend(coefficients[i, 1:1+degree])
        if coefficients.shape[1] == 1:
            effective = csr_matrix((n, n))
        else:
            effective = csr_matrix((edge_values, graph.indices.copy(), graph.indptr.copy()), shape=graph.shape)
    rng = np.random.default_rng(seed)
    scale = case["scale"]
    state = np.broadcast_to(case["initial_r"] / scale, (paths, n)).copy()
    draws = np.empty((paths, 12, n), dtype=np.int32)
    for horizon in range(12):
        if distance is None:
            peer = 0.0 if coefficients.shape[1] == 1 else (effective @ state.T).T
            rate = case["background"][horizon] + scale * (coefficients[:, 0] * state + peer)
        else:
            rate = case["background"][horizon] + scale * (coefficients[:, 0] * state + 0.0)
            if distance:
                rate = rate + distance * (graph @ (scale * state).T).T
        values = rng.poisson(np.maximum(0, rate))
        draws[:, horizon] = values
        state = (values - case["background"][horizon]) / scale
    return draws


def predict(case, graph, coefficients, distance=None):
    """Take each month's median over three batches of 4,096 simulated paths."""
    draws = np.empty((len(SEEDS) * PATHS, 12, len(case["scale"])), dtype=np.int32)
    for batch, seed in enumerate(SEEDS):
        draws[batch * PATHS:(batch + 1) * PATHS] = simulate(
            case, graph, coefficients, PATHS, seed, distance=distance
        )
    return np.stack([np.median(draws[:, month], axis=0) for month in range(12)])
