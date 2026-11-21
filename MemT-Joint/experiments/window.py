"""Historical window helpers; memt-common-v1 rejects this path."""
import signal
from contextlib import contextmanager
from pathlib import Path

from experiments.io import read_json, read_jsonl, sha256
from memory.branch import digest
from memory.protocol import flatten
from methods.joint.rollout import blocks


def parse_response(text):
    import json
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    value, end = json.JSONDecoder().raw_decode(text)
    # Some offline reviews append an unmatched closer. Never discard another value or prose.
    if set(text[end:].strip()) - set('}] \t\r\n'):
        raise ValueError('Substantive trailing content in offline JSON response')
    return value


def inside(root, name):
    path = (root / name).resolve()
    if root not in path.parents:
        raise ValueError('Window artifact escapes bundle')
    return path


def reuse_baseline(source, contract, destination):
    """Reuse theta0 only after a training-credit-only repair, never a new inference protocol."""
    import copy
    import shutil
    source, destination = Path(source).resolve(), Path(destination)
    old = read_json(source / 'manifest.json')
    for key in ('common_protocol', 'models', 'data', 'validation', 'window_artifacts', 'environment', 'python'):
        if old[key] != contract[key]:
            raise ValueError('Baseline identity differs: ' + key)
    configs = [copy.deepcopy(x['config']) for x in (old, contract)]
    for cfg in configs:
        cfg['runtime'].pop('stop_file', None)
        cfg['window'].pop('baseline_source', None)
    if configs[0] != configs[1] or contract['common_protocol']['name'] != 'memt-window-v2':
        raise ValueError('Baseline configuration changed')
    # These reviewed changes affect training credit and baseline import only.
    allowed = {'methods/joint/credit.py', 'experiments/joint.py', 'experiments/window.py'}
    changed = {k for k in old['code'].keys() | contract['code'].keys() if old['code'].get(k) != contract['code'].get(k)}
    if not changed <= allowed:
        raise ValueError('Baseline inference source changed: ' + repr(changed - allowed))
    selection = read_json(source / 'BASELINE.json')
    validation = inside(source, selection['validation'])
    summary_file = validation / 'summary.json'
    complete = read_json(validation / 'COMPLETE.json')
    summary = read_json(summary_file)
    if (selection['step'] != 0 or summary['policy_version'] != 0 or summary['fixture_only']
        or summary['count'] != 81 or sha256(summary_file) != selection['summary_sha256']
        or sha256(validation / 'answers.jsonl') != complete['answers_sha256']
        or sha256(source / 'manifest.json') != read_json(validation / 'manifest.json')['training_manifest_sha256']
        or summary['protocol_contract'] != contract['common_protocol']
        or len(read_jsonl(validation / 'answers.jsonl')) != 81):
        raise ValueError('Incomplete or unbound theta0 baseline')
    for item in validation.iterdir():
        if item.name == 'manifest.json':
            continue  # The destination already binds its new training manifest/checkpoint.
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    from experiments.io import write_json
    write_json(destination / 'REUSED_BASELINE.json', {'source': str(validation),
        'source_manifest_sha256': sha256(source / 'manifest.json'), 'summary_sha256': sha256(summary_file),
        'reviewed_training_only_code_changes': sorted(changed), 'evaluation_seconds_this_run': 0})
    return summary


def load_bundle(location, episodes, data, model_hashes=None):
    file = Path(location).resolve()
    root, bundle = file.parent, read_json(file)
    if bundle.get('split') != 'train' or bundle.get('version') != 'memt-window-v2':
        raise ValueError('Only train window bundles are eligible')
    if bundle['episodes_sha256'] != sha256(Path(data) / 'episodes.jsonl'):
        raise ValueError('Window source identity changed')
    if model_hashes is not None and bundle['frozen_prefix_model'] != model_hashes:
        raise ValueError('Frozen prefix model differs from training base')
    audit = read_json(root / 'ASSISTANT_AUDIT.json')
    if audit.get('bundle_sha256') != sha256(file) or audit.get('source_checked') is not True:
        raise ValueError('Independent QA results still require source audit')
    probes_file = inside(root, bundle['probes'])
    if sha256(probes_file) != bundle['probes_sha256']:
        raise ValueError('Reviewed probes changed')
    probes = read_jsonl(probes_file)
    allowed = set(audit['accepted_qa_ids'])
    selected = [q for q in probes if q['qa_id'] in allowed]
    if len(selected) != len(allowed) or {q['qa_id'] for q in selected} != allowed:
        raise ValueError('Audit references missing or duplicate questions')
    files = {'bundle.json': sha256(file), 'ASSISTANT_AUDIT.json': sha256(root / 'ASSISTANT_AUDIT.json'),
             bundle['probes']: sha256(probes_file)}
    episode_map = {e['sample_id']: e for e in episodes}
    for q in selected:
        episode = episode_map[q['sample_id']]
        endpoint = q['eval_session']
        if q.get('reviewed') is not True or not 0 <= endpoint < len(episode['sessions']):
            raise ValueError('Missing reviewed QA or invalid cutoff')
        prefix = dict(episode, sessions=episode['sessions'][:endpoint+1])
        if q['source_cutoff']['prefix_sha256'] != digest(prefix):
            raise ValueError('QA source prefix changed')
        visible = {t['turn_id'] for s in prefix['sessions'] for t in s['turns']}
        if not q.get('evidence') or not set(q['evidence']) <= visible:
            raise ValueError('QA missing evidence or requires future source')
        review = q['review']
        for key in ('blind', 'audit'):
            name = review[key + '_file']
            f = inside(root, name)
            if sha256(f) != review[key + '_sha256']:
                raise ValueError('QA review receipt changed')
            files[name] = sha256(f)
        blind_messages = read_json(inside(root, review['blind_file']))['messages']
        # Blind request contains the prefix/questions only, never candidate gold/evidence.
        import json
        blind_payload = json.loads(blind_messages[-1]['content'])
        if set(blind_payload) != {'prefix', 'questions', 'cutoff_time'} or blind_payload['prefix'] != prefix:
            raise ValueError('Reviewer saw candidate labels or a different prefix')
        if review['verdict'].get('accept') is not True or review['blind_answer'].get('answerable') is not True:
            raise ValueError('Rejected QA in training bundle')
        def response(value):
            result = parse_response(value['response'])
            return result if isinstance(result, list) else [result]
        index = review['index']
        blind_receipt = read_json(inside(root, review['blind_file']))
        audit_receipt = read_json(inside(root, review['audit_file']))
        audited = json.loads(audit_receipt['messages'][-1]['content'])
        candidate = {'question': q['question'], 'answer': q['answer'], 'evidence': q['evidence'],
                     'blind': review['blind_answer']}
        ai = review.get('audit_index', index)
        if (response(blind_receipt)[index] != review['blind_answer'] or response(audit_receipt)[ai] != review['verdict']
            or blind_payload['questions'][index] != q['question'] or audited['prefix'] != prefix
            or audited['candidates'][ai] != candidate):
            raise ValueError('QA differs from the independently reviewed request/response')
    for w in bundle['windows']:
        episode = episode_map[w['sample_id']]
        sequence = list(blocks(episode, 4))
        start, end = w['start'], w['end']
        if not 0 <= start < end <= len(sequence) or end - start != 8:
            raise ValueError('Window must contain eight complete blocks')
        if sequence[end-1].position[0] != w['eval_session'] or (end < len(sequence) and sequence[end].position[0] == w['eval_session']):
            raise ValueError('QA endpoint must be session end')
        snapshot = inside(root, w['snapshot'])
        if sha256(snapshot) != w['snapshot_sha256']:
            raise ValueError('Prefix snapshot changed')
        value = read_json(snapshot)
        expected = {t['turn_id']: '%s at %s: %s' % (t['speaker'], b.session['time'], t['text'])
                    for b in sequence[:start] for t in b.turns}
        if value['sample_id'] != episode['sample_id'] or value['sources'] != expected:
            raise ValueError('Snapshot is not the exact pre-window prefix')
        if value['position'] != (list(sequence[start-1].position) if start else None):
            raise ValueError('Snapshot cursor is not immediately before the window')
        for values in value['collections'].values():
            for _, meta in values.values():
                if not set(flatten(meta.get('source_turn_ids', []))) <= expected.keys():
                    raise ValueError('Memory metadata references a future turn')
        eligible = [q for q in selected if q['sample_id'] == w['sample_id'] and q['eval_session'] == w['eval_session']]
        if len(eligible) < 4:
            raise ValueError('Insufficient independently reviewed QA for a window')
        recent = {t['turn_id'] for b in sequence[start:end] for t in b.turns}
        if (sum(bool(set(q['evidence']) & recent) for q in eligible) < 2 or
            sum(not set(q['evidence']) & recent for q in eligible) < 2):
            raise ValueError('Need two current-window QA and two prefix-retention QA before initialization')
        files[w['snapshot']] = sha256(snapshot)
        w['snapshot_path'] = str(snapshot)
    return bundle['windows'], selected, files


@contextmanager
def step_deadline(seconds):
    """An over-budget step fails without relabelling a partial update as complete."""
    if not seconds:
        yield
        return
    if not hasattr(signal, 'setitimer'):
        raise RuntimeError('Real timed pilot requires Linux setitimer')
    def expired(*_):
        raise TimeoutError('Ordinary training step exceeded %ss' % seconds)
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
