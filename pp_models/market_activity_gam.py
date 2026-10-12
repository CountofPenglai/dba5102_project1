"""Sequential price-dynamics / platform-activity GAM feature ablations."""
from pygam import LinearGAM, l, s
from .directional_gam import (CenteredDirectionSpline, DiagnosticGAM, build_directional_model,
                              features_for as directional_features, variant_config as directional_config)
from .market_activity_features import PRICE_FEATURES, ACTIVITY_FEATURES

VARIANTS = ('original', 'semivol', 'price_dynamics', 'platform_activity')


def features_for(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown market/activity variant')
    if variant in ('original', 'semivol'):
        return directional_features(variant)
    return directional_features('semivol') + PRICE_FEATURES + (ACTIVITY_FEATURES if variant == 'platform_activity' else [])


def variant_config(base, variant, lam):
    cfg = directional_config(base, 'original' if variant == 'original' else 'semivol', lam)
    cfg.update(gam_variant=variant, features=features_for(variant), activity_metrics=True)
    return cfg


def build_activity_model(name, resolved):
    variant, p = resolved['gam_variant'], resolved['models']['gam']
    features = resolved['features']
    if name != 'gam' or resolved['selected_models'] != ['gam'] or features != features_for(variant):
        raise ValueError('Invalid activity GAM model/features')
    if variant in ('original', 'semivol'):
        return build_directional_model(name, resolved)
    if (set(p) != {'lam', 'n_splines', 'max_iter', 'tol'} or isinstance(p['lam'], bool)
            or p['lam'] not in (10, 100, 1000) or p['n_splines'] != 4 or p['max_iter'] != 100 or p['tol'] != 1e-4):
        raise ValueError('Invalid activity GAM parameters')
    terms = l(0, lam=p['lam'])
    for i, feature in enumerate(features[1:], 1):
        terms += s(i, n_splines=4, spline_order=3, lam=p['lam'], penalties='derivative')
        if feature != 'market_price_usdc_per_btc':
            terms += CenteredDirectionSpline(i, by=0, n_splines=4, spline_order=3,
                lam=p['lam'], penalties='derivative')
    return DiagnosticGAM(LinearGAM(terms, max_iter=p['max_iter'], tol=p['tol']), features)
