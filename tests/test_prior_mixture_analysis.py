"""The two data draws are the independent data units, not the six fits."""
import pytest
from fm_stress.analyze_prior_mixture import summarize_cells, paired_contrasts


def rows():
    # Baseline error differs by dataset. Paired effects must remove this shift.
    records = []
    for draw, offset in [(10, 100.), (20, 1000.)]:
        for seed in (1, 2, 3):
            for alpha in (0., .5, 1.):
                for head, effect in [('mlp', 0.), ('s4', -2. + alpha), ('whitened_mlp', 1. - 4. * alpha)]:
                    records.append(dict(data_seed=draw, seed=seed, alpha=alpha, head=head,
                                        fair_crps=offset + seed + effect + alpha * 10))
    return records


def test_draw_means_do_not_pool_optimization_and_data_uncertainty():
    cell = next(x for x in summarize_cells(rows(), ['fair_crps']) if x['head'] == 'mlp' and x['alpha'] == 0)
    assert cell['runs'] == 6 and cell['data_draws'] == 2
    assert cell['metrics']['fair_crps']['mean'] == 552
    assert cell['metrics']['fair_crps']['draw_means'] == [102, 1002]
    assert [x['metrics']['fair_crps']['seed_sd'] for x in cell['by_draw']] == [1, 1]
    assert 'ci_low' not in cell['metrics']['fair_crps']


def test_paired_head_interaction_and_source_effect():
    contrasts = paired_contrasts(rows(), ['fair_crps'])
    head = next(c for c in contrasts if c['kind'] == 'head_difference' and c['first'] == 's4' and c['alpha'] == .5)
    assert head['equal_draw_mean_difference'] == -1.5
    interaction = next(c for c in contrasts if c['kind'] == 'head_by_source_interaction_vs_white' and c['first'] == 's4' and c['alpha'] == .5)
    assert interaction['equal_draw_mean_difference'] == .5
    source = next(c for c in contrasts if c['kind'] == 'source_difference_vs_white' and c['first'] == 'whitened_mlp' and c['alpha'] == 1.)
    assert source['equal_draw_mean_difference'] == 6.


def test_contrasts_reject_missing_pairs_and_duplicate_runs():
    with pytest.raises(ValueError, match='Unpaired'):
        paired_contrasts(rows()[1:], ['fair_crps'])
    with pytest.raises(ValueError, match='Duplicate'):
        paired_contrasts(rows() + rows()[:1], ['fair_crps'])


def test_gp_factorial_does_not_require_extra_heads_on_white():
    records = rows()
    gp_raw = [r for r in records if r['alpha'] == 1 and r['head'] == 'mlp']
    for r in gp_raw:
        records.append({**r, 'head':'balanced_mlp', 'fair_crps':r['fair_crps']+2})
        records.append({**r, 'head':'whitened_raw_mlp', 'fair_crps':r['fair_crps']+4})
    contrasts = paired_contrasts(records, ['fair_crps'])
    interaction = next(c for c in contrasts if c['kind']=='gp_coordinate_by_loss_interaction')
    # Full whitened at alpha1 is raw−3; the two single controls are raw+2/raw+4.
    assert interaction['equal_draw_mean_difference'] == -9
    assert len([c for c in contrasts if c['kind']=='gp_coordinate_loss_control']) == 4
