"""Register fixed exploratory comparisons before calculating uncertainty."""
import pandas as pd


def build_comparison_registry(macro, branch1, branch2, final, ablation_pairs):
    scores = macro.set_index('model')
    if not scores.index.is_unique:
        raise ValueError('Comparison scores must have unique model names')
    registered = {}

    def positive(candidate, reference):
        return (scores.loc[candidate, 'macro_rmse_per_btc'] < scores.loc[reference, 'macro_rmse_per_btc']
                or scores.loc[candidate, 'macro_mae_per_btc'] < scores.loc[reference, 'macro_mae_per_btc'])

    def add(candidate, reference, role, included, status):
        if candidate == reference:
            return
        key = (candidate, reference)
        if key not in registered:
            registered[key] = {'candidate': candidate, 'reference': reference,
                               'included': bool(included), 'status': status, 'roles': role,
                               'rmse_delta': float(scores.loc[candidate, 'macro_rmse_per_btc']
                                                   - scores.loc[reference, 'macro_rmse_per_btc']),
                               'mae_delta': float(scores.loc[candidate, 'macro_mae_per_btc']
                                                  - scores.loc[reference, 'macro_mae_per_btc'])}
        else:
            row = registered[key]
            row['roles'] += ';' + role
            if included and not row['included']:
                row.update(included=True, status=status)

    for reference in ('official_raw', 'grouped_mean'):
        add('h0', reference, 'core_reference', True, 'reference')
    retained_first = []
    for row in branch1.itertuples():
        candidate = row.model
        references = ['h0'] if candidate == 'g0' else ['g0', 'h0']
        retained = bool(row.replacement_accuracy_pass)
        if retained:
            retained_first.append(candidate)
        local = any(positive(candidate, reference) for reference in references)
        status = 'accuracy_candidate' if retained else 'local_evidence' if local else 'no_positive_increment'
        for reference in references:
            add(candidate, reference, 'branch1', retained or local, status)
    for row in branch2.itertuples():
        candidate = row.model
        matched = {'g2a': 'g1a', 'g2b': 'g1b'}[candidate]
        retained = bool(row.accuracy_vs_h0_pass and row.paired_increment_pass)
        local = positive(candidate, matched) or positive(candidate, 'h0')
        status = 'accuracy_candidate' if retained else 'local_evidence' if local else 'no_positive_increment'
        for reference in dict.fromkeys([matched, 'h0'] + retained_first):
            add(candidate, reference, 'branch2', retained or local, status)
    final_positive = positive('final_combination', 'h0')
    status = 'adopted_candidate' if final.final_adopted else (
        'local_evidence' if final_positive else 'not_adopted_no_increment')
    for reference in dict.fromkeys([final.raw_model, 'h0', 'official_raw', 'grouped_mean']):
        add('final_combination', reference, 'branch3', final.final_adopted or final_positive, status)
    for row in ablation_pairs.itertuples():
        if row.layer == 'shrinkage':
            continue
        local = positive(row.candidate, row.reference)
        add(row.candidate, row.reference, 'branch4_' + row.layer, local,
            'local_evidence' if local else 'no_positive_increment')
    return pd.DataFrame(registered.values())
