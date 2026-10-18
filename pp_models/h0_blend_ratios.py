"""Fixed directional convex mixtures and settled-history grid selection."""
import numpy as np
import pandas as pd
from .h0_state_fallback import history_weight

GRID = (0., .25, .5, .75, 1.)
# Each pair stores H0 weights in (Call, Put) order; Base receives 1-alpha.
FIXED = {}
for call in GRID:
    for put in GRID:
        name = ('official_raw' if call == put == 0 else 'h0' if call == put == 1
                else 'half_blend' if call == put == .5 else f'blend_c{round(call*100):03d}_p{round(put*100):03d}')
        FIXED[name] = (call, put)
DYNAMIC = {'dynamic14': 14, 'dynamic30': 30}
MODELS = tuple(FIXED)+tuple(DYNAMIC)
CANDIDATES = tuple(m for m in MODELS if m not in ('official_raw', 'h0'))


def grid_weight(past, origin, history_days):
    """Only prior settled OOS errors; regularization and support match the prior study."""
    if history_days not in (14, 30): raise ValueError('Unsupported fixed history window')
    origin = pd.Timestamp(origin)
    # Validate the supplied eligible 30-day cache before shortening its window.
    history_weight(past, origin)
    recent = past.loc[past.created.ge(origin-pd.Timedelta(days=history_days))]
    par = history_weight(recent, origin)
    par.update(history_days=history_days, continuous_alpha=par['alpha'])
    if par['supported']:
        d = recent.h0.to_numpy()-recent.base.to_numpy()
        errors = recent.base.to_numpy()[:, None]+d[:, None]*np.asarray(GRID)-recent.actual.to_numpy()[:, None]
        mse = np.mean(errors**2, axis=0)
        # Match the continuous history-weight objective on the fixed candidate grid.
        penalized = mse+par['penalty']/len(recent)*(np.asarray(GRID)-.5)**2
        # GRID is ascending, so an exact tie selects the smaller H0 weight.
        par.update(alpha=float(GRID[int(np.argmin(penalized))]), grid_mse=mse.tolist(),
                   grid_penalized_mse=penalized.tolist())
    return par
