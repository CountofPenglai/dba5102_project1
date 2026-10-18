"""Forward-prediction history for causal, settled-error blend selection."""
import numpy as np
import pandas as pd


class ForwardPredictionHistory:
    def __init__(self, pool):
        self.pool = pool
        self._frames = []
        self._last_origin = None

    def append(self, window, h0, base):
        """Store forecasts only; labels are joined later when they are available."""
        q = window.query_ids
        created = self.pool.metadata.loc[q, 'created_at_utc']
        if (self._last_origin is not None and window.origin <= self._last_origin
                or np.shape(h0) != (len(q),) or np.shape(base) != (len(q),)
                or not np.isfinite(h0).all() or not np.isfinite(base).all()
                or not created.ge(window.origin).all() or not created.lt(window.stop).all()):
            raise ValueError('Invalid or nonchronological forward predictions')
        self._frames.append(pd.DataFrame(dict(h0=h0, base=base,
            side=self.pool.X.loc[q, 'option_type'], created=created, forecast_origin=window.origin), index=q))
        self._last_origin = window.origin

    def settled(self, origin, *, history_days=30):
        """Release past labels only when closed < origin; retain original row order."""
        origin = pd.Timestamp(origin)
        previous = pd.concat(self._frames) if self._frames else pd.DataFrame({
            **{k: pd.Series(dtype=float) for k in ('h0', 'base', 'side')},
            **{k: pd.Series(dtype='datetime64[ns, UTC]') for k in ('created', 'forecast_origin')}})
        closed = pd.to_datetime(self.pool.metadata.closed_at_utc.reindex(previous.index), utc=True)
        eligible = (previous.created.ge(origin-pd.Timedelta(days=history_days))
            & previous.created.lt(origin) & previous.forecast_origin.lt(origin) & closed.lt(origin))
        past = previous.loc[eligible].copy()
        past['closed'] = closed.reindex(past.index)
        past['actual'] = self.pool.y.reindex(past.index)
        return past
