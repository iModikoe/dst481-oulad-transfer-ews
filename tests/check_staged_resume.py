"""Exercise staged orchestration with real cached inputs and dummy fits.

Outputs are temporary machinery checks, never study findings.
"""
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_all


def dummy_fit(name, X, y, groups, num_cols, cat_cols, cfg, class_weight=None):
    pipe = run_all.models.make_pipeline('majority', num_cols, cat_cols, 11)
    pipe.fit(X, y)
    return pipe, {}


with tempfile.TemporaryDirectory(prefix='ews_staged_check_') as tmp:
    dest = Path(tmp) / 'outputs'
    shutil.copytree(ROOT / 'outputs' / 'tables', dest / 'tables')
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    cfg['paths'] = {'oulad': str(ROOT / 'data' / 'oulad'),
                    'uci': str(ROOT / 'data' / 'uci' / 'data.csv'),
                    'outputs': str(dest)}
    config = Path(tmp) / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    argv = ['run_all.py', '--config', str(config), '--tiers', '2,3,4',
            '--quick', '--skip-shap', '--skip-sensitivity']
    with patch.object(sys, 'argv', argv), \
         patch.object(run_all.models, 'fit_with_search', side_effect=dummy_fit), \
         patch.object(run_all, 'make_figures'):
        assert run_all.main() == 0
    tables = dest / 'tables'
    pairs = pd.read_csv(tables / 'tier3_per_pair.csv')
    base = pd.read_csv(tables / 'tier1_internal.csv').set_index('model_key').roc_auc
    assert len(pairs) == 39
    np.testing.assert_allclose(pairs.delta_roc_auc_vs_tier1,
                               pairs.model_key.map(base) - pairs.roc_auc)
    fair = pd.read_csv(tables / 'fairness_eod.csv')
    assert (fair.tier == 1).sum() == 6
    conf = pd.read_csv(tables / 'confirmatory_family.csv')
    assert len(conf) == 5
    assert (conf.loc[conf.hypothesis == 'H1', 'n_pairs'] == 7).all()
    for attr in ['gender', 'deprivation']:
        source = fair[(fair.tier == 1) & (fair.model_key == 'xgb') & (fair.attribute == attr)]
        target = fair[(fair.tier == 3) & (fair.model_key == 'xgb') & (fair.attribute == attr)]
        row = conf[conf.comparison.str.contains(attr)].iloc[0]
        assert 'XGBoost' in row.comparison
        assert row.n_pairs == len(target)
        np.testing.assert_allclose(row.estimate, target.eod.mean() - source.eod.iloc[0])
    assert len(pd.read_csv(tables / 'fairness_mitigation.csv')) > 0
    assert (pd.read_csv(tables / 'calibration_by_tier.csv').tier == 1).sum() == 5
    assert (pd.read_csv(tables / 'predictions.csv').tier == 1).sum() == 82659
    print('PASS: 39 degradation values, H3 family selection/counts, mitigation, and Tier 1 retention.')
    print(conf[['comparison', 'n_pairs']].to_string(index=False))
