"""Resume a sealed own step-1 checkpoint in a new, explicitly linked 50-step run."""
import copy
import json
import os
import shutil
import sys
from pathlib import Path

from experiments.io import read_json, write_json, sha256
from experiments.joint import corpus, identity
from experiments.joint_config import load, variant, validate
from experiments.window import load_bundle

root = Path('/public/home/wwxia/yuang.li').resolve()
assert root.stat().st_uid == os.getuid()
config_file, source, output = [Path(x).resolve() for x in sys.argv[1:4]]
assert all(root in p.parents for p in (config_file, source, output))
assert source.stat().st_uid == os.getuid() and not output.exists()
cfg = variant(load(config_file), 'M4')
validate(cfg, training=True)
assert cfg['retrieval']['steps'] == 50 and cfg['window']['step_seconds'] == 0
cfg['runtime']['stop_file'] = str(output / 'STOP')
data = root / 'workspace/Mem-T/artifacts/retrieval-trial-20261026/data/full'
contract = identity(cfg, data / 'train')
contract['validation'] = {name: sha256(data / 'valid' / name) for name in ('episodes.jsonl', 'questions.jsonl', 'partition.json')}
episodes, _ = corpus(data / 'train')
_, _, contract['window_artifacts'] = load_bundle(cfg['window']['bundle'], episodes, data / 'train', contract['models']['base'])
old = read_json(source / 'manifest.json')
for key in ('models', 'data', 'validation', 'window_artifacts', 'environment', 'python', 'common_protocol'):
    assert old[key] == contract[key], key
configs = [copy.deepcopy(x['config']) for x in (old, contract)]
for item in configs:
    item['retrieval'].pop('steps')
    for key in ('stop_file', 'save_every', 'validate_every'):
        item['runtime'].pop(key, None)
    item['window'].pop('step_seconds')
    item['window'].pop('baseline_source', None)
assert configs[0] == configs[1], 'Training conditions changed beyond the requested launch controls'
changed = {k for k in old['code'].keys() | contract['code'].keys() if old['code'].get(k) != contract['code'].get(k)}
assert changed == {'experiments/joint_config.py'}, changed
latest = read_json(source / 'LATEST.json')
assert latest['step'] == 1
baseline = read_json(source / 'BASELINE.json')
assert baseline['step'] == 0
output.mkdir()
linked_bytes = 0

def copy_tree(src, dst, checkpoint=False):
    global linked_bytes
    assert root in src.resolve().parents and output in dst.resolve().parents
    assert not src.is_symlink() and not dst.exists()
    for directory, dirs, files in os.walk(src, followlinks=False):
        rel = Path(directory).relative_to(src)
        target = dst / rel
        target.mkdir(parents=True, exist_ok=True)
        for name in dirs + files:
            file = Path(directory) / name
            assert not file.is_symlink() and file.stat().st_uid == os.getuid(), str(file)
        for name in files:
            file = Path(directory) / name
            if checkpoint and file.suffix == '.pt':
                assert file.stat().st_dev == output.stat().st_dev
                os.link(file, target / name)  # Immutable sealed checkpoint shards; new saves use fresh directories.
                linked_bytes += file.stat().st_size
            else:
                shutil.copy2(file, target / name)

for selection in (baseline, latest):
    src = (source / selection['path']).resolve()
    assert source in src.parents
    for name, expected in read_json(src / 'RECEIPT.json').items():
        file = (src / name).resolve()
        assert src in file.parents and sha256(file) == expected, str(file)
    copy_tree(src, output / selection['path'], checkpoint=True)
state = read_json(output / latest['path'] / 'state.json')
assert state['step'] == 1 and state['world_size'] == 2
metrics = source / state['last_metrics']
assert source in metrics.resolve().parents
copy_tree(metrics.parent, output / metrics.parent.relative_to(source))
validation = (source / baseline['validation']).resolve()
assert source in validation.parents and sha256(validation / 'summary.json') == baseline['summary_sha256']
assert sha256(validation / 'answers.jsonl') == read_json(validation / 'COMPLETE.json')['answers_sha256']
copy_tree(validation, output / baseline['validation'])
write_json(output / baseline['validation'] / 'INHERITED_FROM.json', {
    'source_run': str(source), 'source_manifest_sha256': sha256(source / 'manifest.json'),
    'note': 'Inherited theta0 validation; not a newly evaluated result.'})
for name in ('LATEST.json', 'BASELINE.json', 'BEST.json'):
    shutil.copy2(source / name, output / name)
write_json(output / 'manifest.json', contract)
write_json(output / 'SOURCE_RUN.json', {'source': str(source), 'source_manifest_sha256': sha256(source / 'manifest.json'),
    'inherited_step': 1, 'target_cumulative_step': 50, 'new_updates_to_run': 49, 'linked_checkpoint_bytes': linked_bytes,
    'checkpoint_receipt_sha256': sha256(source / latest['path'] / 'RECEIPT.json'),
    'code_changes': sorted(changed), 'fork_script_sha256': sha256(Path(__file__)),
    'config_file_sha256': sha256(config_file), 'reason': 'User requested continuation to50; pilot timer disabled; no performance changes.'})
print('SEALED_CHECKPOINT_FORK_VERIFIED', 1, 50, 'LINKED_BYTES', linked_bytes, flush=True)
