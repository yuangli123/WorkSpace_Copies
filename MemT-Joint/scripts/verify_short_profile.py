"""Gate training on two measured RL updates with explicit missing-U records."""
import argparse
from pathlib import Path

from experiments.io import read_json, write_json
from methods.joint.short_episode import validate_short_weights


def verify(run, output):
    run = Path(run)
    state = read_json(run/'train/TRAINING_COMPLETE.json')
    manifest = read_json(run/'train/manifest.json')
    if state['step'] != 2 or state.get('fixture_only') is not False or manifest['config']['backend'] != 'verl':
        raise ValueError('Need two real distributed updates, not fixtures/capacity samples')
    files = sorted((run/'train/attempts').glob('*/step-*/metrics.json'))
    values = [read_json(p) for p in files]
    if sorted(m['iteration'] for m in values) != [0,1]:
        raise ValueError('Missing or duplicate measured updates')
    for m in values:
        if not 0 < m['wall_seconds'] <= 1800:
            raise ValueError('Ordinary training computation exceeded 30 minutes')
        if not m['nonzero_advantages'] or m['fixture_only']:
            raise ValueError('No real learning signal')
        if not (2 <= m['short_episode']['blocks'] <= 8 and m['short_episode']['initial_state'] == 'empty'):
            raise ValueError('Wrong empty-state episode protocol')
        validate_short_weights(m, manifest['config']['joint']['branch_weights'],
                               m['short_episode'].get('skipped_branches', {}))
        if m['answer_policy_version'] != m['iteration']:
            raise ValueError('Answer must use this update behavior policy')
    if read_json(run/'train/LATEST.json')['step'] != 2:
        raise ValueError('Missing second-step checkpoint with warm optimizer state')
    receipt = dict(passed=True, run=str(run), formal_updates=0, calibration_updates=2,
                   scope='training_compute_checkpoint_separate',
                   full_step_including_checkpoint_passed=all(m['ordinary_step_seconds_including_checkpoint'] <= 1800 for m in values),
                   ordinary_step_limit_seconds=1800, updates=values,
                   config=manifest['config'], code=manifest['code'], models=manifest['models'], data=manifest['data'],
                   common_protocol=manifest['common_protocol'])
    write_json(output, receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = verify(args.run, args.output)
    print('COMPUTE_PROFILE_PASSED', [u['wall_seconds'] for u in result['updates']],
          'FULL_WITH_CHECKPOINT', [u['ordinary_step_seconds_including_checkpoint'] for u in result['updates']])
