"""Shared encoding for every generator: scenario vectors <-> normalised [-1, 1] model space,
and condition specs -> a (value, mask) conditioning vector.

Conditions can be given partially: any condition left out is masked, so e.g.
{"difficulty": 0.8} asks for a hard scenario and lets the model choose everything else.
"""
import numpy as np

from scenarios.spec import BOUNDS, K_OBS, MAP_SIZE, VECTOR_DIM, Scenario, condition_vector

_OBS_RANGES = [BOUNDS["obs_pos"], BOUNDS["obs_pos"], (-BOUNDS["obs_speed"][1], BOUNDS["obs_speed"][1]),
               (-BOUNDS["obs_speed"][1], BOUNDS["obs_speed"][1]), BOUNDS["radius"], BOUNDS["t_appear"], (0.0, 1.0)]
_W = BOUNDS["wind"][1]
RANGES = np.array(
    [(0.0, MAP_SIZE)] * 4 + [BOUNDS["speed"]]
    + _OBS_RANGES * K_OBS
    + [(-_W, _W), (-_W, _W), BOUNDS["gust_amp"], BOUNDS["gust_t"], BOUNDS["sense_range"], BOUNDS["noise_std"]]
)
assert len(RANGES) == VECTOR_DIM
LO, HI = RANGES[:, 0], RANGES[:, 1]
OBS0 = 5                                   # index of the first obstacle field
ACTIVE_IDX = np.array([OBS0 + 7 * k + 6 for k in range(K_OBS)])

COND_KEYS = ("speed", "n_obstacles", "dynamic_frac", "wind", "wind_rel_cos", "wind_rel_sin", "sense_range",
             "difficulty")
COND_RANGES = np.array([BOUNDS["speed"], (0, K_OBS), (0, 1), BOUNDS["wind"], (-1, 1), (-1, 1),
                        BOUNDS["sense_range"], (0, 1)], dtype=float)
COND_DIM = 2 * len(COND_KEYS)              # values + mask


def canonical(vec: np.ndarray) -> np.ndarray:
    """Active obstacles first, ordered along the direction of travel (removes slot-order ambiguity)."""
    v = np.array(vec, dtype=float)
    obs = v[OBS0:OBS0 + 7 * K_OBS].reshape(K_OBS, 7)
    d = v[2:4] - v[0:2]
    along = (obs[:, :2] - v[0:2]) @ d
    key = np.where(obs[:, 6] > 0.5, along, np.inf)
    v[OBS0:OBS0 + 7 * K_OBS] = obs[np.argsort(key, kind="stable")].ravel()
    return v


def encode(vecs: np.ndarray) -> np.ndarray:
    """Raw scenario vectors (n, 46) -> model space; inactive slots set to 0 (their data is meaningless)."""
    vecs = np.stack([canonical(v) for v in np.atleast_2d(vecs)])
    z = 2 * (vecs - LO) / (HI - LO) - 1
    for k in range(K_OBS):
        off = vecs[:, ACTIVE_IDX[k]] < 0.5
        z[np.ix_(off, np.arange(OBS0 + 7 * k, OBS0 + 7 * k + 6))] = 0.0
    return z.astype(np.float32)


def decode(z: np.ndarray, clip: bool = False) -> np.ndarray:
    """Model space -> raw scenario vectors. clip=True clamps every field into its physical range."""
    z = np.clip(z, -1, 1) if clip else np.asarray(z, dtype=float)
    vecs = LO + (z + 1) / 2 * (HI - LO)
    vecs[:, ACTIVE_IDX] = (vecs[:, ACTIVE_IDX] > 0.5).astype(float)
    return vecs


def scenario_conditions(vecs: np.ndarray, difficulty: np.ndarray) -> np.ndarray:
    """Full condition values (n, 8) describing given scenarios and their measured failure score."""
    rows = [condition_vector(Scenario.from_vector(v)) for v in vecs]
    return np.column_stack([np.array(rows), difficulty])


def cond_tensor_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(n, 8) values + (n, 8) mask -> (n, 16) conditioning input in [-1, 1]."""
    lo, hi = COND_RANGES[:, 0], COND_RANGES[:, 1]
    v = 2 * (values - lo) / (hi - lo) - 1
    return np.concatenate([np.where(mask > 0, v, 0.0), mask], axis=1).astype(np.float32)


def cond_from_spec(spec: dict | None, n: int) -> np.ndarray:
    """{'difficulty': 0.8, 'n_obstacles': 4, ...} -> (n, 16). Unknown keys raise."""
    values = np.zeros((n, len(COND_KEYS)))
    mask = np.zeros((n, len(COND_KEYS)))
    for k, val in (spec or {}).items():
        i = COND_KEYS.index(k)
        values[:, i] = val
        mask[:, i] = 1.0
    return cond_tensor_values(values, mask)


def random_cond_mask(rng: np.random.Generator, n: int, p_drop_each=0.3, p_drop_all=0.15) -> np.ndarray:
    """Training-time mask: drop each condition with p_drop_each, all of them with p_drop_all."""
    mask = (rng.random((n, len(COND_KEYS))) > p_drop_each).astype(float)
    mask[rng.random(n) < p_drop_all] = 0.0
    return mask
