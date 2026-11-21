from __future__ import annotations

import copy
import json
import platform
import random
import time
import uuid
import os
import sys
from importlib.metadata import version, PackageNotFoundError
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path

from experiments.io import read_json, read_jsonl, write_json, write_jsonl, sha256, checked_output
from experiments.joint_config import validate, contract as common_contract
from memory.branch import State, digest
from memory.data import validate_corpus
from models.joint import Runtime, FixturePolicy
from methods.joint.rollout import collect, blocks, formation, evolution, new_state
from methods.joint.retrieval import sample
from methods.joint.credit import score_round, score_record


@contextmanager
def no_phase():
    yield


def serializable(value):
    if is_dataclass(value):
        return serializable(asdict(value))
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def corpus(path, probes=None):
    path = Path(path)
    episodes, questions = read_jsonl(path / 'episodes.jsonl'), read_jsonl(path / 'questions.jsonl')
    if probes:
        for probe in read_jsonl(probes):
            if probe.get('reviewed') is not True or not isinstance(probe.get('eval_session'), int) or isinstance(probe.get('eval_session'), bool):
                raise ValueError('Online probes require reviewed=true and explicit zero-based eval_session')
            questions.append(probe)
    episode_map = validate_corpus(episodes, questions)
    sibling_sets = {}
    for split in ('train', 'valid', 'test'):
        file = path.parent / split / 'episodes.jsonl'
        if file.exists():
            sibling_sets[split] = {e['sample_id'] for e in read_jsonl(file)}
    for left, a in sibling_sets.items():
        for right, b in sibling_sets.items():
            if left < right and a & b:
                raise ValueError('Conversation leakage across partitions')
    for q in questions:
        if q.get('eval_session') is not None:
            sessions = episode_map[q['sample_id']]['sessions']
            if not 0 <= q['eval_session'] < len(sessions):
                raise ValueError('Probe time outside episode')
            visible = {t['turn_id'] for s in sessions[:q['eval_session'] + 1] for t in s['turns']}
            references = set(q.get('evidence', []))
            for group in q.get('evidence_groups', []):
                if not isinstance(group, list) or not group or not all(isinstance(s, str) for s in group):
                    raise ValueError('Malformed alternative evidence group')
                references.update(group)
            if not references <= visible:
                raise ValueError('Probe requires future sources')
    return episodes, questions


def identity(config, data, probes=None):
    root = Path(__file__).resolve().parents[1]
    files = [p for package in ('models', 'memory', 'methods', 'training', 'experiments', 'evaluation')
             for p in (root / package).rglob('*.py')]
    files += [root / 'cli.py', root / 'UPSTREAM.lock.json', root / 'SOURCE_ORIGIN.json', root / 'MEMT_EXECUTION_REFERENCE.json']
    vendor = root / 'third_party' / 'Tree-GRPO'
    files += list(vendor.rglob('*.py')) + list(vendor.rglob('*.yaml'))
    files += [vendor / 'verl' / 'version' / 'version']
    sources = {str(p.relative_to(root)).replace('\\', '/'): sha256(p) for p in files}
    inputs = {name: sha256(Path(data) / name) for name in ('episodes.jsonl', 'questions.jsonl', 'partition.json')}
    if probes:
        inputs['probes'] = sha256(probes)
    if config.get('short_episode') and read_json(Path(data) / 'partition.json')['split'] == 'train':
        audit = Path(data) / 'DATA_AUDIT.json'
        if not audit.is_file():
            raise ValueError('Short-episode training requires a bound DATA_AUDIT.json')
        checked = read_json(audit)
        if checked.get('version') != 'memt-short-empty-v1' or checked.get('review') != 'assistant_source_review':
            raise ValueError('Unreviewed short-episode data')
        if any(checked['files'].get(name) != value for name, value in inputs.items()):
            raise ValueError('Short-episode data changed after review')
        inputs['short_episode_audit'] = sha256(audit)
    models = {}
    if config['backend'] != 'fixture':
        for name, location in (('base', config['models']['base']), ('embedding', config['memory']['embedding_model'])):
            location = Path(location)
            if not location.is_dir():
                raise ValueError('Missing local model: ' + str(location))
            # Full shard hashes bind resume to weights, not just a mutable directory name.
            models[name] = {str(p.relative_to(location)).replace('\\', '/'): sha256(p) for p in location.rglob('*')
                            if p.is_file() and p.suffix in ('.json', '.safetensors', '.bin', '.model', '.txt', '.tiktoken')}
    packages = {}
    for name in ('torch', 'transformers', 'vllm', 'ray', 'numpy', 'sentence-transformers', 'deepspeed', 'accelerate'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {'config': config, 'common_protocol': common_contract(config), 'data': inputs, 'code': sources, 'models': models, 'python': platform.python_version(),
            'environment': {'executable': sys.executable, 'conda_prefix': os.getenv('CONDA_PREFIX'), 'packages': packages}}


def make_runtime(config, training):
    if config.get('protocol', {}).get('name') == 'memt-execution-v2':
        os.environ['VLLM_USE_V1'] = '0'
        if config['backend'] != 'fixture':
            for package, expected in [('vllm', '0.8.5.post1'), ('transformers', '4.51.3')]:
                if version(package) != expected:
                    raise ValueError('Common generation stack version mismatch: ' + package)
    if config['backend'] == 'fixture':
        policy = FixturePolicy()
        return Runtime(policy, config), policy, no_phase, None
    from models.hf import HFPolicy
    components = config['experiment'].get('components')
    if config['experiment'].get('variant') == 'M5' or components:
        if config['backend'] != 'hf':
            raise ValueError('Separate/composite roles use the HF offload backend, not multiple Ray pools')
        from training.separate import create
        policy, updater = create(config, training, **(components or {}))
    elif config['backend'] == 'verl':
        from training.verl_backend import VerlBackend
        policy = VerlBackend(config['experiment'].get('policy_model') or config['models']['base'], config)
        updater = policy if training else None
    else:
        policy = HFPolicy(config['models']['base'], config)
        from training.native import NativeUpdater
        updater = NativeUpdater(policy, config) if training else None
    if config.get('protocol', {}).get('name') == 'memt-execution-v2' and config['backend'] == 'hf':
        from models.vllm import attach_generation
        if hasattr(policy, 'policies'):
            policy.common_config = config
        else:
            attach_generation(policy, config)
    # The current actor is also the retrieval answerer.  There is no frozen
    # base-model inference path in the end-to-end protocol.
    return Runtime(policy, config), policy, no_phase, updater


def metrics(round_data, runtime, rows):
    finals = [r for p in round_data['global'] for r in p['evaluations']]
    leaves = [r['trees'][0][-1] for r in finals]
    return {'fixture_only': runtime.config['backend'] == 'fixture', 'train_rows': len(rows),
            'rows_by_role': {role: sum(r['role'] == role for r in rows) for role in ('B', 'U', 'R')},
            'nonzero_advantages': sum(abs(r['advantage']) > 1e-8 for r in rows),
            'nonzero_advantages_by_role': {role: sum(r['role'] == role and abs(r['advantage']) > 1e-8 for r in rows) for role in ('B', 'U', 'R')},
            'branch_weights': {role: {branch: sum(r['weight'] for r in rows if r['role'] == role and r['branch'] == branch)
                for branch in ('global', 'local', 'mot')} for role in ('B', 'U', 'R')},
            'role_weights': {role: sum(r['weight'] for r in rows if r['role'] == role) for role in ('B', 'U', 'R')},
            'active_weight_mass': sum(r['weight'] for r in rows),
            'skipped_branches': round_data.get('short_episode', {}).get('skipped_branches', {}),
            'global_qa_f1': sum(n.performance for n in leaves) / max(1, len(leaves)),
            'policy_prompt_tokens': sum(len(r['prompt_ids']) for r in runtime.calls),
            'policy_generated_tokens': sum(len(r['response_ids']) for r in runtime.calls),
            'generation_seconds_by_role': {role: sum(r['generation_seconds_share'] for r in runtime.calls if r['role'] == role) for role in ('B', 'U', 'R')},
            'generation_requests_by_role': {role: sum(r['role'] == role for r in runtime.calls) for role in ('B', 'U', 'R')},
            'continuation_seconds': sum(r['generation_seconds_share'] for r in runtime.calls if '/continuation-' in r['event']),
            'training_generated_tokens': sum(len(r['response_ids']) for r in runtime.calls if r.get('training_eligible')),
            'greedy_evaluator_generated_tokens': sum(len(r['response_ids']) for r in runtime.calls if r['evaluator_only']),
            'continuation_calls': sum('/continuation-' in r['event'] for r in runtime.calls),
            'answer_calls': sum('/fallback' in r['event'] for r in runtime.calls),
            'answer_prompt_tokens': sum(len(r['prompt_ids']) for r in runtime.calls if '/fallback' in r['event']),
            'answer_generated_tokens': sum(len(r['response_ids']) for r in runtime.calls if '/fallback' in r['event']),
            'r_local_groups': len(round_data['local_R']), 'eligible_questions': round_data['eligible_questions'],
            'skipped_local_anchors': round_data['skipped_local_anchors']}


def checkpoint(updater, output, state):
    path = Path(output) / 'checkpoints' / ('step-%06d-%s' % (state['step'], uuid.uuid4().hex[:8]))
    updater.save(path, state)
    hashes = {str(p.relative_to(path)).replace('\\', '/'): sha256(p) for p in path.rglob('*') if p.is_file()}
    write_json(path / 'RECEIPT.json', hashes)
    write_json(Path(output) / 'LATEST.json', {'path': str(path.relative_to(output)), 'step': state['step']})
    if updater.config.get('runtime', {}).get('checkpoint_retention') == 'latest_best':
        from training.checkpoints import CheckpointManager
        manager = CheckpointManager(output, updater, updater.config)
        for older in (Path(output) / 'checkpoints').iterdir():
            if older != path and (older / 'COMPLETE').is_file():
                manager.retire(older, 'new complete Joint checkpoint validated; keep latest and selected best')
    return path


def checkpoint_path(output, selection='latest'):
    pointer = 'BEST.json' if selection == 'best' and (Path(output) / 'BEST.json').exists() else 'LATEST.json'
    latest = read_json(Path(output) / pointer)
    path = (Path(output) / latest['path']).resolve()
    if Path(output).resolve() not in path.parents:
        raise ValueError('Checkpoint escapes run directory')
    return path


def restore(updater, output, selection='latest'):
    path = checkpoint_path(output, selection)
    expected = read_json(path / 'RECEIPT.json')
    for name, value in expected.items():
        file = (path / name).resolve()
        if path not in file.parents or sha256(file) != value:
            raise ValueError('Incomplete or modified checkpoint')
    return updater.restore(path)


def run(config, data, output, probes=None, resume=False, smoke=False, valid_data=None):
    validate(config, training=not smoke)
    output = checked_output(output, resume)
    config = copy.deepcopy(config)
    config['runtime']['stop_file'] = str((output / 'STOP').resolve())
    from experiments.runtime import ResourceMonitor, stop_requests
    telemetry = output / 'telemetry' / uuid.uuid4().hex
    started, status = time.perf_counter(), 'failed'
    try:
        with ResourceMonitor(telemetry, config), stop_requests(config) as stopped:
            result = _run(config, data, output, probes, resume, smoke, stopped, valid_data)
        status = 'stopped' if (output / 'STOP').exists() else 'returned'
        return result
    finally:
        allocation = os.getenv('SLURM_GPUS_ON_NODE', '')
        elapsed = time.perf_counter() - started
        write_json(telemetry / 'attempt.json', {'status': status, 'wall_seconds_including_initialization_save_validation': elapsed,
                   'allocated_gpu_hours': elapsed * int(allocation) / 3600 if allocation.isdigit() else None,
                   'slurm_job_id': os.getenv('SLURM_JOB_ID'), 'argv': sys.argv})


def _run(config, data, output, probes=None, resume=False, smoke=False, stopped=lambda: False, valid_data=None):
    validate(config, training=not smoke)
    if smoke and config['backend'] != 'fixture':
        raise ValueError('smoke is the fixture flow; use train with a real model for RL')
    output = Path(output)
    contract = identity(config, data, probes)
    if valid_data:
        if read_json(Path(valid_data) / 'partition.json')['split'] != 'valid':
            raise ValueError('Checkpoint selection requires valid, never test or train')
        _, valid_questions = corpus(valid_data)
        contract['validation'] = {name: sha256(Path(valid_data) / name) for name in ('episodes.jsonl', 'questions.jsonl', 'partition.json')}
    window_specs = None
    if config.get('window'):
        from experiments.window import load_bundle
        window_episodes, _ = corpus(data)
        window_specs, window_questions, window_identity = load_bundle(
            config['window']['bundle'], window_episodes, data, contract['models'].get('base'))
        contract['window_artifacts'] = window_identity
    manifest = output / 'manifest.json'
    if manifest.exists():
        if read_json(manifest) != contract:
            raise ValueError('Config, code, model, or input identity changed; use a new run')
    else:
        write_json(manifest, contract)
    with (output / 'commands.jsonl').open('a', encoding='utf-8') as command_log:
        command_log.write(json.dumps({'argv': sys.argv, 'cwd': str(Path.cwd()), 'time_unix': time.time()}, ensure_ascii=False) + '\n')
    partition = read_json(Path(data) / 'partition.json')
    if partition['split'] != 'train':
        raise ValueError('Training may only use the train partition')
    episodes, questions = corpus(data, probes)
    if window_specs:
        questions = window_questions
        validate_corpus(episodes, questions)
    if valid_data:
        valid_episodes, _ = corpus(valid_data)
        if {e.get('source_sample_id', e['sample_id']) for e in episodes} & {e.get('source_sample_id', e['sample_id']) for e in valid_episodes}:
            raise ValueError('Training and validation conversations overlap')
    runtime, _, _, updater = make_runtime(config, training=not smoke)
    runtime.protocol_id = digest({'protocol': common_contract(config), 'models': contract['models']}) if common_contract(config) else None
    state = restore(updater, output) if resume and updater else {'step': 0}
    runtime.policy_version = state['step']
    if updater and not resume:
        checkpoint(updater, output, state)
    attempt = output / 'attempts' / uuid.uuid4().hex
    steps = 1 if smoke else config['retrieval']['steps']
    info = read_json(output / state['last_metrics']) if state.get('last_metrics') else {}

    def validate_checkpoint():
        latest = read_json(output / 'LATEST.json')
        key = Path(latest['path']).name
        receipt = output / 'validation' / (key + '.json')
        if receipt.exists():
            return
        destination = checked_output(output / 'validation' / (key + '-' + uuid.uuid4().hex[:8]))
        runtime.calls = []
        write_json(destination / 'manifest.json', {'training_manifest_sha256': sha256(manifest), 'checkpoint': latest})
        # Every validation checkpoint is a fresh end-to-end replay from an
        # empty state; no prior baseline or frozen answer is reused.
        summary = evaluate_runtime(runtime, valid_data, destination)
        selection = dict(latest, f1=summary['f1'], validation=str(destination.relative_to(output)),
                         summary_sha256=sha256(destination / 'summary.json'))
        best = output / 'BEST.json'
        if not best.exists() or selection['f1'] > read_json(best)['f1']:
            if config['runtime'].get('checkpoint_retention') == 'latest_best':
                from training.checkpoints import CheckpointManager
                selected = CheckpointManager(output, updater, config).select(output / latest['path'], summary)
                selected_path = Path(selected['path'])
                write_json(selected_path / 'RECEIPT.json', {
                    str(p.relative_to(selected_path)).replace('\\', '/'): sha256(p)
                    for p in selected_path.rglob('*') if p.is_file() and p.name != 'RECEIPT.json'})
                write_json(best, dict(selection, path=selected['path']))
            else:
                write_json(best, selection)
        write_json(receipt, selection)
        if state['step'] == 0:
            write_json(output / 'BASELINE.json', selection)

    if resume and updater and valid_data and state['step'] and (
            state['step'] % config['runtime']['validate_every'] == 0 or state['step'] == steps):
        validate_checkpoint()
    if updater and valid_data and state['step'] == 0 and common_contract(config):
        validate_checkpoint()
    for iteration in range(state['step'], steps):
        if stopped():
            if updater:
                checkpoint(updater, output, state)
            write_json(output / 'STOPPED.json', state)
            return state
        from experiments.window import step_deadline
        try:
            with step_deadline(config.get('window', {}).get('step_seconds', 0)):
                started = time.perf_counter()
                runtime.calls = []
                runtime.progress = lambda value: write_json(output / 'PROGRESS.json', dict(value, step=iteration+1, elapsed=time.perf_counter()-started))
                runtime.policy.progress = runtime.progress
                sync_before = getattr(runtime.policy, 'sync_seconds', 0.)
                sessions_before = getattr(runtime.policy, 'generation_sessions', 0)
                episode = episodes[iteration % len(episodes)]
                qs = [q for q in questions if q['sample_id'] == episode['sample_id']]
                horizon = next((s['sessions'] for s in config['joint'].get('curriculum', []) if iteration < s['until']), None)
                runtime.progress({'phase': 'weight_sync'})
                with runtime.generation_session():
                    if window_specs:
                        from methods.joint.window import collect_window
                        spec = window_specs[iteration % len(window_specs)]
                        episode = next(e for e in episodes if e['sample_id'] == spec['sample_id'])
                        initial_state = State.from_dict(read_json(spec['snapshot_path']), runtime.ranker)
                        round_data = collect_window(episode, questions, initial_state, spec, runtime, iteration)
                    elif config.get('short_episode'):
                        from methods.joint.short_episode import collect_short, validate_short_weights
                        round_data = collect_short(episode, qs, runtime, iteration)
                    else:
                        round_data = collect(episode, qs, runtime, iteration, horizon)
                    sampled = time.perf_counter()
                answered = time.perf_counter()
                rows, groups = score_round(round_data, runtime)
                info = metrics(round_data, runtime, rows)
                if window_specs and (set(info['rows_by_role']) != {'B', 'U', 'R'} or
                    any(abs(info['role_weights'][role] - 1/3) > 1e-6 for role in ('B', 'U', 'R'))):
                    raise ValueError('Missing B/U/R branch samples or role weight mass')
                if config.get('short_episode') and not window_specs:
                    validate_short_weights(info, config['joint']['branch_weights'], info['skipped_branches'])
                if updater and iteration == 0 and not info['nonzero_advantages']:
                    raise ValueError('Initial on-policy advantages are all zero; inspect formats/QA coverage before spending a full run')
                scored = time.perf_counter()
                if updater:
                    runtime.progress({'phase': 'old_ref_backward_adam'})
                    info.update(updater.update(rows))
                    runtime.policy_version = iteration + 1
                info.update(answer_policy_version=iteration, window=bool(window_specs), iteration=iteration, wall_seconds=time.perf_counter() - started,
                            sampling_seconds=sampled-started, answer_seconds=answered-sampled,
                            scoring_seconds=scored-answered, optimization_seconds=time.perf_counter()-scored)
                if config.get('short_episode'):
                    info['short_episode'] = round_data['short_episode']
                target = attempt / ('step-%06d' % (iteration + 1))
                write_json(target / 'round.json.gz', serializable(round_data))
                write_jsonl(target / 'rows.jsonl.gz', rows)
                write_json(target / 'answers.json', {'interface': 'finish(answer)', 'fallback_calls': [
                    {'event': r['event'], 'text': r['text']} for r in runtime.calls if '/fallback' in r['event']
                ]})
                write_json(target / 'metrics.json', info)
                state = {'step': iteration + 1, 'last_metrics': str((target / 'metrics.json').relative_to(output))}
                due_validation = valid_data and (state['step'] % config['runtime']['validate_every'] == 0 or state['step'] == steps)
                if updater and (state['step'] % config['runtime']['save_every'] == 0 or state['step'] == steps or stopped() or due_validation):
                    runtime.progress({'phase': 'checkpoint'})
                    checkpoint(updater, output, state)
                info['ordinary_step_seconds_including_checkpoint'] = time.perf_counter() - started
                info['generation_sessions_total'] = getattr(runtime.policy, 'generation_sessions', None)
                info['weight_sync_seconds_total'] = getattr(runtime.policy, 'sync_seconds', None)
                info['weight_sync_seconds'] = getattr(runtime.policy, 'sync_seconds', 0.) - sync_before
                info['generation_sessions_this_step'] = getattr(runtime.policy, 'generation_sessions', 0) - sessions_before
                info['generation_release_seconds'] = getattr(runtime.policy, 'last_release_seconds', None)
                write_json(target / 'metrics.json', info)
        except Exception as error:
            write_json(output / 'FAILED_STEP.json', {'step': iteration+1,
                'completed_checkpoint_step': read_json(output / 'LATEST.json')['step'] if (output / 'LATEST.json').exists() else 0,
                'optimizer_update_returned': runtime.policy_version == iteration + 1,
                'error': repr(error), 'elapsed': time.perf_counter()-started if 'started' in locals() else None})
            write_jsonl(attempt / ('failed-step-%06d-calls.jsonl.gz' % (iteration+1)), runtime.calls)
            raise
        runtime.progress = None
        runtime.policy.progress = None
        validation_started = time.perf_counter()
        if updater and due_validation and not stopped():
            validate_checkpoint()
        info['validation_seconds_separate'] = time.perf_counter() - validation_started if due_validation else 0.
        write_json(target / 'metrics.json', info)
        write_json(target / 'total_timing.json', {'wall_seconds_including_save_validation': time.perf_counter()-started})
        print(json.dumps(info, ensure_ascii=False), flush=True)
        if stopped():
            write_json(output / 'STOPPED.json', state)
            return info
    if smoke:
        write_json(output / 'SMOKE_COMPLETE.json', dict(state, fixture_only=True, optimizer_updates=0))
    else:
        write_json(output / 'TRAINING_COMPLETE.json', dict(state, fixture_only=False,
                   model_selection='valid_f1' if valid_data else 'final_only_no_validation'))
    return info if state['step'] else {}


def evaluate(config, data, output, model=None, mode=None, construction=None, retrieval=None):
    config = copy.deepcopy(config)
    if mode:
        config['experiment']['mode'] = mode
    if model and config['backend'] == 'verl':
        if config['experiment'].get('variant') == 'M0':
            raise ValueError('M0 must use the untouched base model')
        config['experiment']['policy_model'] = str(Path(model).resolve())
    if construction or retrieval:
        config['experiment']['components'] = {'construction': construction, 'retrieval': retrieval}
    validate(config)
    output = checked_output(output)
    contract = identity(config, data)
    extra_paths = {'policy_export': model, 'construction': construction, 'retrieval': retrieval}
    contract['evaluated_weights'] = {name: {str(p.relative_to(path)): sha256(p) for p in Path(path).rglob('*.safetensors')}
                                    for name, path in extra_paths.items() if path}
    write_json(output / 'manifest.json', contract)
    episodes, questions = corpus(data)
    runtime, _, _, _ = make_runtime(config, training=False)
    runtime.protocol_id = digest({'protocol': common_contract(config), 'models': contract['models']}) if common_contract(config) else None
    if model and config['backend'] != 'verl':
        if config['backend'] != 'hf':
            raise ValueError('Evaluate exported HF weights using backend=hf')
        from transformers import AutoModelForCausalLM
        roles_file = Path(model) / 'roles.json'
        if roles_file.exists():
            if not hasattr(runtime.policy, 'policies'):
                raise ValueError('A role-bundle export requires --variant M5')
            for role, folder in read_json(roles_file).items():
                policy = runtime.policy.policies[role]
                loaded = AutoModelForCausalLM.from_pretrained(Path(model) / folder, torch_dtype=policy.model.dtype)
                policy.model.load_state_dict(loaded.state_dict())
                del loaded
        else:
            loaded = AutoModelForCausalLM.from_pretrained(model, torch_dtype=runtime.policy.model.dtype)
            runtime.policy.model.load_state_dict(loaded.state_dict())
            del loaded
    return evaluate_runtime(runtime, data, output)


def evaluate_runtime(runtime, data, output):
    started = time.perf_counter()
    with runtime.generation_session():
        result = _evaluate_runtime(runtime, data, output)
    result['evaluation_seconds_separate'] = time.perf_counter()-started
    write_json(Path(output) / 'summary.json', result)
    return result


def _evaluate_runtime(runtime, data, output):
    config = runtime.config
    output = Path(output)
    episodes, questions = corpus(data)
    results = []
    for episode in episodes:
        state = new_state(episode['sample_id'], runtime)
        snapshots = {}
        all_blocks = list(blocks(episode, config['memory']['summary_context_turns']))
        for bi, block in enumerate(all_blocks):
            started_block = time.perf_counter()
            calls_before = len(runtime.calls)
            stats_before = dict(getattr(runtime.ranker, 'chroma_stats', {}))
            state.observe(block.session, block.turns, block.position, runtime.count)
            state, candidates, _, _ = formation(state, block, runtime, 'eval/%s/%d/B' % (episode['sample_id'], bi), greedy=True)
            state, _, _, _ = evolution(state, candidates, block, runtime, 'eval/%s/%d/U' % (episode['sample_id'], bi), greedy=True)
            if bi + 1 == len(all_blocks) or all_blocks[bi + 1].position[0] != block.position[0]:
                snapshots[block.position[0]] = state.to_dict()
            progress = {'phase': 'memory_construction', 'episode': episode['sample_id'], 'completed_blocks': bi + 1,
                        'total_blocks': len(all_blocks),
                        'policy_calls': len(runtime.calls), 'block_seconds': time.perf_counter()-started_block,
                        'generation_seconds': sum(row.get('generation_seconds_share', 0.) for row in runtime.calls[calls_before:]),
                        'chroma': {key: value - stats_before.get(key, 0) for key, value in
                                   getattr(runtime.ranker, 'chroma_stats', {}).items()}}
            write_json(output / 'PROGRESS.json', progress)
            print(json.dumps(progress), flush=True)
        safe_name = digest(episode['sample_id'])[:24] + '.json'
        write_json(output / 'snapshots' / safe_name, state.to_dict())
        write_json(output / 'storage' / safe_name, state.statistics(runtime.count))
        # Construction and retrieval use the same Chroma+BGE-M3 backend in
        # real runs; fixture/lexical remains an explicit test-only path.
        old_depth = config['retrieval']['max_depth']
        config['retrieval']['max_depth'] = config['evaluation']['max_steps']
        episode_questions = [q for q in questions if q['sample_id'] == episode['sample_id']]
        for qi, q in enumerate(episode_questions):
            when = q.get('eval_session')
            visible = state if when is None else State.from_dict(snapshots[when], runtime.ranker)
            results.append(sample(visible, q, runtime, 'eval/%s/%d/R' % (episode['sample_id'], qi), forest=False, greedy=True))
            write_json(output / 'PROGRESS.json', {'phase': 'retrieval', 'episode': episode['sample_id'], 'completed_questions': qi + 1})
        config['retrieval']['max_depth'] = old_depth
    write_json(output / 'PROGRESS.json', {'phase': 'retrieval_answer', 'retrieval_questions': len(results),
                                        'unique_answer_requests': 0})
    from methods.retrieval.tree import answer_bleu1
    rows = []
    for record in results:
        score_record(record, runtime)
        q, node = record['question'], record['trees'][0][-1]
        prediction = node.answer
        rows.append({'qa_id': q['qa_id'], 'sample_id': q['sample_id'], 'question': q['question'], 'answer': q['answer'], 'category': q.get('category'),
                     'protocol_id': getattr(runtime, 'protocol_id', None),
                     'eval_session': q.get('eval_session'), 'probe_type': q.get('probe_type'),
                     'response': prediction, 'f1': node.answer_performance, 'bleu1': answer_bleu1(prediction, q['answer']),
                     'strict_f1': node.performance, 'strict_bleu1': answer_bleu1(prediction, q['answer']) if not node.fallback else 0.,
                     'fallback_answer': bool(node.fallback and node.answer),
                     'evidence_recall': node.answer_evidence, 'submitted_memory_refs': node.submitted,
                     'answer_memory_refs': node.answer_refs,
                     'answer_source_ids': sorted({s for ref in node.answer_refs for s in node.observed[ref]['sources']}),
                     'submitted_source_ids': sorted({s for ref in node.submitted for s in node.observed[ref]['sources']}),
                     'finished': bool(node.valid and node.stop_reason == 'finish'), 'retrieval_tokens': node.cost_tokens})
    summary = {'count': len(rows), 'f1': sum(r['f1'] for r in rows) / max(1, len(rows)),
               'bleu1': sum(r['bleu1'] for r in rows) / max(1, len(rows)), 'fixture_only': config['backend'] == 'fixture',
               'protocol': runtime.mode, 'accuracy': None, 'note': 'LongMemEval official Accuracy requires the official judge; F1 is not Accuracy.'}
    summary.update(strict_f1=sum(r['strict_f1'] for r in rows)/max(1,len(rows)),
                   strict_bleu1=sum(r['strict_bleu1'] for r in rows)/max(1,len(rows)),
                   finish_rate=sum(r['finished'] for r in rows)/max(1,len(rows)),
                   fallback_rate=sum(r['fallback_answer'] for r in rows)/max(1,len(rows)),
                   common_protocol=config.get('protocol', {}).get('name', 'legacy-joint'))
    summary['policy_version'] = runtime.policy_version
    summary['by_category'] = {
        category: {'count': len(group), 'f1': sum(r['f1'] for r in group) / len(group),
                   'bleu1': sum(r['bleu1'] for r in group) / len(group)}
        for category in sorted({str(r.get('category', '')) for r in rows})
        for group in [[r for r in rows if str(r.get('category', '')) == category]]}
    summary['answer_weights'] = config.get('protocol', {}).get('answer_weights', 'current_policy')
    summary['protocol_contract'] = common_contract(config)
    write_jsonl(output / 'answers.jsonl', rows)
    write_json(output / 'trajectories.json.gz', serializable(results))
    write_json(output / 'summary.json', summary)
    write_json(output / 'COMPLETE.json', {'answers_sha256': sha256(output / 'answers.jsonl'), 'fixture_only': summary['fixture_only']})
    write_json(output / 'PROGRESS.json', {'phase': 'complete', 'completed_questions': len(rows)})
    return summary
