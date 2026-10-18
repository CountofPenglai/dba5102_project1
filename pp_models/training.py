"""Run-scoped H0 fitting shared by all report branches; no persistent/global cache."""
import hashlib
import copy

import pandas as pd
from pp_data.splitting import RollingSplitter
from .h0 import probability_frame
from .h0_time_decay import TimeDecayH0

MIN_TRAINING_ORDERS = 100


def _pool_fingerprint(pool):
    digest = hashlib.sha256()
    for table in (pool.X, pool.y, pool.metadata):
        digest.update(str(table.dtypes).encode())
        digest.update(pd.util.hash_pandas_object(table, index=True).to_numpy().tobytes())
    return digest.hexdigest()


class TrainingSession:
    """Fit each (weekly origin, half-life) once on strictly available history.

    A session belongs to one unchanged pool and protocol. Callers may use returned
    models for prediction/state export only; never refit or mutate a cached model.
    """
    def __init__(self, pool, protocol=None):
        self.pool = pool
        self.splitter = RollingSplitter(pool, protocol)
        self.protocol = self.splitter.protocol
        self._original_protocol = copy.deepcopy(self.protocol)
        self._fingerprint = _pool_fingerprint(pool)
        self._models = {}
        self.fit_counts = {}
        self.cache_hits = 0

    def require(self, pool, protocol):
        if (pool is not self.pool or protocol != self._original_protocol
                or self.protocol != self._original_protocol
                or _pool_fingerprint(pool) != self._fingerprint):
            raise ValueError('Training session cannot be reused with changed data or protocol')
        return self

    def fit(self, window, half_life_days=None):
        self.require(self.pool, self.protocol)
        h = self.splitter.history(window.origin)
        if not h.equals(window.history_ids):
            raise ValueError('Training window does not match causal history')
        if len(h) < MIN_TRAINING_ORDERS:
            raise ValueError('Insufficient H0 historical support')
        if half_life_days not in (None, 7, 14):
            raise ValueError('Unsupported report half-life')
        key = (window.origin, half_life_days)
        if key in self._models:
            self.cache_hits += 1
            return self._models[key]
        model = TimeDecayH0(half_life_days).fit(probability_frame(self.pool.X.loc[h]),
            self.pool.y.loc[h], self.pool.metadata.loc[h, 'created_at_utc'], window.origin)
        self._models[key] = model
        name = 'h0' if half_life_days is None else f'h0_decay{half_life_days}'
        self.fit_counts[name] = self.fit_counts.get(name, 0)+1
        return model


def training_session(pool, protocol, existing=None):
    """Standalone studies get a fresh session; the unified runner passes one in."""
    return TrainingSession(pool, protocol) if existing is None else existing.require(pool, protocol)
