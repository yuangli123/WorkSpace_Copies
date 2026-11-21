from __future__ import annotations

import copy
import random
import time
from pathlib import Path

from methods.construction.rollout import construct_episode
from methods.construction.credit import hindsight_scores, select_operations
from memory.data import validate_corpus
from experiments.io import checked_output, manifest, read_json, read_jsonl, iter_jsonl, sha256, write_json, write_jsonl
from memory.journal import JournalDB, open_db
from methods.retrieval.tree import RetrievalEnvironment, answer_f1, answer_bleu1, forest_record, sample_forests, sample_tree, training_rows

__all__ = ['MemoryRun', 'build', 'collect_forests', 'derive_sft', 'evaluate', 'seed_all', 'train_retrieval',
           'train_managed', 'answer_f1', 'answer_bleu1', 'sample_tree']


def seed_all(seed):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build(config, data_dir, output, policy, resume=False):
    from .progress import save_chunk, restore_chunk, read_chunk
    output = checked_output(output, resume=resume)
    data_dir = Path(data_dir)
    episodes = read_jsonl(data_dir / "episodes.jsonl")
    questions = read_jsonl(data_dir / "questions.jsonl")
    validate_corpus(episodes, questions)
    partition = read_json(data_dir / "partition.json")
    contract = manifest(config, [data_dir / "episodes.jsonl", data_dir / "questions.jsonl"])
    if (output / 'manifest.json').exists():
        previous = read_json(output / 'manifest.json')
        if any(previous[k] != contract[k] for k in ('config', 'inputs', 'code_sha256')):
            raise ValueError('Construction resume contract changed')
    else:
        write_json(output / "manifest.json", contract)
    database = open_db(config, output / (f'db-resume-{time.time_ns()}' if resume else 'db'))
    index, all_operations, augmented_questions = {}, [], []
    for replica in range(config["construction"]["rollouts"]):
        for source_episode in episodes:
            episode = copy.deepcopy(source_episode)
            source_id = episode["sample_id"]
            sid = f"{source_id}__r{replica}"
            episode["sample_id"] = sid
            progress_folder = output / 'progress' / sid
            snapshot_name = f"snapshots/{sid}.json"
            completed = progress_folder / 'COMPLETE.json'
            if resume and completed.exists():
                receipt = read_json(completed)
                if receipt['snapshot_sha256'] != sha256(output / snapshot_name):
                    raise ValueError('Completed construction snapshot changed')
                state = read_chunk(progress_folder)['state']
                if state['cursor'] != receipt['chunks']:
                    raise ValueError('Completed construction chunk count changed')
                operations = state['operations']
            else:
                journal, progress = JournalDB(database), None
                if resume and (progress_folder / 'CURRENT.json').exists():
                    progress, journal = restore_chunk(progress_folder, database)
                collections = [f'{sid}_{kind}' for kind in ('turns', 'facts', 'experiences', 'personas', 'summary')]
                def checkpoint(state):
                    save_chunk(progress_folder, state, journal, collections)
                    write_json(output / 'BUILD_PROGRESS.json', {'completed_episodes': len(index), 'active_episode': sid,
                               'chunk': state['cursor'], 'eligible_operations': eligible_count(all_operations),
                               'target_eligible': config['construction'].get('minimum_valid_operations', 0)})
                operations, collections = construct_episode(episode, policy, journal, config, progress=progress,
                                                             checkpoint_fn=checkpoint)
                journal.save(output / snapshot_name, collections)
                chunks = sum((len(s['turns']) + config['memory']['summary_context_turns'] - 1) //
                             config['memory']['summary_context_turns'] for s in episode['sessions'])
                write_json(completed, {'snapshot_sha256': sha256(output / snapshot_name), 'chunks': chunks})
            index[sid] = {"source_sample_id": source_id, "snapshot": snapshot_name,
                          "snapshot_sha256": sha256(output / snapshot_name)}
            all_operations.extend(operations)
            for qa in questions:
                if qa["sample_id"] == source_id:
                    augmented_questions.append(dict(qa, sample_id=sid, qa_id=f'{qa["qa_id"]}__r{replica}'))
        target = config['construction'].get('minimum_valid_operations', 0)
        write_json(output / 'BUILD_PROGRESS.json', {'completed_episodes': len(index), 'completed_rollouts': replica + 1,
                   'eligible_operations': eligible_count(all_operations), 'target_eligible': target})
        if target and eligible_count(all_operations) >= target:
            break
    if len({op['op_id'] for op in all_operations}) != len(all_operations):
        raise ValueError('Duplicate construction operation identity')
    if config['construction'].get('minimum_valid_operations', 0) > eligible_count(all_operations):
        raise ValueError('Independent rollouts exhausted before collecting the requested eligible actions')
    write_json(output / "index.json", index)
    operations_name = 'operations.jsonl.gz' if config.get('runtime', {}).get('compress_traces') else 'operations.jsonl'
    write_jsonl(output / operations_name, all_operations)
    write_jsonl(output / "questions.jsonl", augmented_questions)
    write_json(output / "partition.json", partition)
    write_json(output / "COMPLETE.json", {"episodes": len(index), "operations": len(all_operations),
                                          "questions": len(augmented_questions),
                                          "eligible_operations": eligible_count(all_operations)})


def eligible_count(operations):
    invalid = {op.get('completion_id') for op in operations if not op['valid'] or op.get('completion_valid') is False}
    invalid.discard(None)
    return sum(op['valid'] and op.get('completion_id') not in invalid for op in operations)


class MemoryRun:
    def __init__(self, path, config):
        self.path = Path(path)
        self.index = read_json(self.path / "index.json")
        self.partition = read_json(self.path / "partition.json")
        if not (self.path / "COMPLETE.json").exists():
            raise ValueError("Incomplete memory build")
        self.questions = read_jsonl(self.path / "questions.jsonl")
        operations = self.path / 'operations.jsonl.gz'
        self.operations_file = operations if operations.exists() else self.path / 'operations.jsonl'
        self.operations = read_jsonl(self.operations_file)
        self.database = open_db(config, self.path / "db")
        self.loaded = {}

    def db(self, sample_id):
        if sample_id not in self.loaded:
            entry = self.index[sample_id]
            snapshot = self.path / entry["snapshot"]
            if sha256(snapshot) != entry["snapshot_sha256"]:
                raise ValueError("Memory snapshot changed since construction")
            self.loaded[sample_id] = JournalDB.restore(snapshot, self.database)
        return self.loaded[sample_id]

    def environment(self, qa, config, log_path):
        return RetrievalEnvironment(self.db(qa["sample_id"]), qa["sample_id"], config, log_path)


from evaluation.retrieval import evaluate


def collect_forests(config, memory, policy, output, resume=False):
    output = checked_output(output, resume=resume)
    contract = manifest(config, [memory.path / 'index.json', memory.path / 'questions.jsonl'])
    if (output / 'manifest.json').exists():
        previous = read_json(output / 'manifest.json')
        if any(previous[k] != contract[k] for k in ('config', 'inputs', 'code_sha256')):
            raise ValueError('Forest resume contract changed')
    else:
        write_json(output / 'manifest.json', contract)
    suffix = '.json.gz' if config.get('runtime', {}).get('compress_forests') else '.json'
    width = config['construction'].get('forest_batch_size', config['retrieval']['batch_size'])
    def forest_path(index):
        return output / 'forests' / (f'{index:06d}' + suffix)
    for offset in range(0, len(memory.questions), width):
        batch = memory.questions[offset:offset + width]
        seeds = [config["seed"] + offset + i for i in range(len(batch))]
        pending = [i for i in range(len(batch)) if not forest_path(offset+i).exists()]
        envs = [memory.environment(batch[i], config, output / 'tool_logs') for i in pending]
        sampled = sample_forests(policy, envs, [batch[i] for i in pending], config, [seeds[i] for i in pending]) if pending else []
        for index, forest in zip(pending, sampled):
            write_json(forest_path(offset+index), forest_record(batch[index], forest, 'final_policy', seeds[index]))
        for index, qa in enumerate(batch):
            record = read_json(forest_path(offset+index))
            if record['qa_id'] != qa['qa_id'] or record['sample_id'] != qa['sample_id'] or record['seed'] != seeds[index]:
                raise ValueError('Saved forest does not match ordered questions and sampling seed')
        write_json(output / 'PROGRESS.json', {'completed': offset + len(batch), 'target': len(memory.questions)})
    combined = 'forests.jsonl.gz' if suffix.endswith('.gz') else 'forests.jsonl'
    write_jsonl(output / combined, (read_json(forest_path(i)) for i in range(len(memory.questions))))
    write_json(output / "manifest.json", manifest(config, [memory.path / "index.json", memory.path / "questions.jsonl"]))


def derive_sft(config, memory, forests_file, output):
    if memory.partition["split"] != "train":
        raise ValueError("Construction training may only use the train partition")
    output = checked_output(output)
    expected = {q['qa_id'] for q in memory.questions}
    def records():
        seen = set()
        for record in iter_jsonl(forests_file):
            if record.get('step') != 'final_policy':
                raise ValueError('Use forests collected from one final policy')
            if record['qa_id'] in seen:
                raise ValueError('Duplicate final-policy forest')
            seen.add(record['qa_id'])
            yield record
        if seen != expected:
            raise ValueError('Final forests must cover the selected training QA set exactly')
    forest_manifest = read_json(Path(forests_file).parent / "manifest.json")
    input_hashes = set(forest_manifest["inputs"].values())
    if not {sha256(memory.path / "index.json"), sha256(memory.path / "questions.jsonl")} <= input_hashes:
        raise ValueError("Final forests were collected from a different memory run")
    scored = hindsight_scores(memory.operations, records(), memory.questions, config)
    selected, audit = select_operations(scored, config["construction"]["keep_fraction"], config["construction"]["maximum_operations"])
    if not selected:
        raise ValueError("No eligible construction operations; inspect validity and attribution")
    suffix = '.jsonl.gz' if config.get('runtime', {}).get('compress_sft_records') else '.jsonl'
    write_jsonl(output / ('scored' + suffix), scored)
    write_jsonl(output / ('selected' + suffix), selected)
    write_json(output / "selection_audit.json", {"categories": audit, "eligible": sum(a["eligible"] for a in audit.values()),
                                                "selected": len(selected),
                                                "all_candidates_tied": all(a["distinct_scores"] == 1 for a in audit.values())})
    write_json(output / "manifest.json", manifest(config, [memory.operations_file, forests_file]))


def train_retrieval(config, memory, policy, updater, output, valid_memory=None, resume=None):
    if memory.partition["split"] != "train":
        raise ValueError("Retrieval training may only use the train partition")
    output = checked_output(output, resume=resume is not None)
    config_file = output / "manifest.json"
    managed = config.get('runtime', {}).get('managed_checkpoints', False)
    inputs = [memory.path / 'index.json', memory.path / 'questions.jsonl']
    if managed and valid_memory:
        inputs.extend([valid_memory.path / 'index.json', valid_memory.path / 'questions.jsonl'])
    current_manifest = manifest(config, inputs)
    state = {"step": 0, "question_cursor": 0, "order": list(range(len(memory.questions)))}
    if not memory.questions:
        raise ValueError("No training questions")
    random.Random(config["seed"]).shuffle(state["order"])
    if resume:
        old_manifest = read_json(config_file)
        for key in ("config", "inputs", "code_sha256"):
            if old_manifest[key] != current_manifest[key]:
                raise ValueError(f"Resume contract mismatch: {key}")
        state = updater.restore(resume)
    else:
        write_json(config_file, current_manifest)
    if managed:
        return train_managed(config, memory, policy, updater, output, valid_memory, state, resume)
    batch_size = config["retrieval"]["batch_size"]
    for step in range(state["step"], config["retrieval"]["steps"]):
        started = time.perf_counter()
        seed_all(config["seed"] + step)
        batch = [memory.questions[state["order"][(state["question_cursor"] + i) % len(state["order"])]] for i in range(batch_size)]
        rows, records = [], []
        seeds = [config["seed"] + step * batch_size + position for position in range(batch_size)]
        environments = [memory.environment(qa, config, output / "tool_logs") for qa in batch]
        forests = sample_forests(policy, environments, batch, config, seeds)
        for qa, forest, seed in zip(batch, forests, seeds):
            rows.extend(training_rows(forest, batch_size))
            records.append(forest_record(qa, forest, step + 1, seed))
        metrics = updater.update(rows)
        flat = [n for record in records for tree in record["trees"] for n in tree[1:]]
        metrics.update(step=step + 1, nodes=len(flat), valid_rate=sum(n["valid"] for n in flat) / len(flat),
                       mean_reward=sum(n["reward"] for n in flat) / len(flat),
                       advantage_min=min(n["advantage"] for n in flat), advantage_max=max(n["advantage"] for n in flat),
                       mean_evidence=sum(n["evidence"] for n in flat) / len(flat),
                       generated_tokens=sum(len(n["response_ids"]) for n in flat))
        leaves = [n for n in flat if not n["children"]]
        metrics.update(leaf_performance=sum(n["performance"] for n in leaves) / len(leaves),
                       finish_rate=sum(n["valid"] and n["terminal"] for n in leaves) / len(leaves),
                       wall_seconds=time.perf_counter() - started)
        write_jsonl(output / "rollouts" / f"step-{step + 1:06d}.jsonl", records)
        write_json(output / "metrics" / f"step-{step + 1:06d}.json", metrics)
        print(metrics, flush=True)
        state.update(step=step + 1, question_cursor=state["question_cursor"] + batch_size)
        if (step + 1) % config["retrieval"]["save_every"] == 0 or step + 1 == config["retrieval"]["steps"]:
            checkpoint = output / "checkpoints" / f"step-{step + 1:06d}"
            updater.save(checkpoint, state)
            if valid_memory:
                evaluate(config, valid_memory, policy, output / "validation" / f"step-{step + 1:06d}.jsonl")
    updater.export(output / "final") if hasattr(updater, "export") else policy.save_model(output / "final")


def train_managed(config, memory, policy, updater, output, valid_memory, state, resume):
    from training.checkpoints import CheckpointManager
    from experiments.runtime import ResourceMonitor, resource_snapshot, stop_requests, verify_live_restore
    from experiments.io import get_budget
    manager = CheckpointManager(output, updater, config)
    manager.event('resume' if resume else 'start', checkpoint=str(resume) if resume else None,
                  step=state['step'], cursor=state['question_cursor'], resources=resource_snapshot())
    restore_step = config['runtime']['restore_probe_step']
    if resume and state['step'] == restore_step and not (output / 'RESTORE_VERIFIED.json').exists():
        verify_live_restore(manager, resume, state)
    if not resume:
        manager.save(state)
    batch_size = config['retrieval']['batch_size']
    validate_every = config['retrieval']['validate_every']
    def validate(step):
        target = output / 'validation' / f'step-{step:06d}.jsonl.gz'
        summary_file = output / 'validation' / f'step-{step:06d}.metrics.json'
        metrics = read_json(summary_file) if summary_file.exists() else evaluate(config, valid_memory, policy, target)
        manager.select(output / 'checkpoints' / f'step-{step:06d}', metrics)
    if resume and state['step'] and state['step'] % validate_every == 0:
        validate(state['step'])
    with stop_requests(config) as stop_requested, ResourceMonitor(output, config) as monitor:
        for step in range(state['step'], config['retrieval']['steps']):
            if stop_requested():
                checkpoint = manager.save(state)
                manager.event('stopped_at_checkpoint', checkpoint=str(checkpoint), step=state['step'])
                write_json(output / 'STOPPED.json', dict(state, checkpoint=str(checkpoint)))
                return
            started = time.perf_counter()
            seed_all(config['seed'] + step)
            batch = [memory.questions[state['order'][(state['question_cursor'] + i) % len(state['order'])]]
                     for i in range(batch_size)]
            seeds = [config['seed'] + step * batch_size + i for i in range(batch_size)]
            environments = [memory.environment(qa, config, output / 'tool_logs') for qa in batch]
            forests = sample_forests(policy, environments, batch, config, seeds)
            rows = [row for forest in forests for row in training_rows(forest, batch_size)]
            metrics = updater.update(rows)
            flat = [node for forest in forests for tree in forest for node in tree[1:]]
            leaves = [node for node in flat if not node.children]
            metrics.update(step=step + 1, global_query_batch=len(batch), nodes=len(flat),
                           valid_rate=sum(node.valid for node in flat) / len(flat),
                           mean_reward=sum(node.reward for node in flat) / len(flat),
                           mean_evidence=sum(node.evidence for node in flat) / len(flat),
                           advantage_min=min(node.advantage for node in flat), advantage_max=max(node.advantage for node in flat),
                           generated_tokens=sum(len(node.response_ids) for node in flat),
                           leaf_performance=sum(node.performance for node in leaves) / len(leaves),
                           finish_rate=sum(node.valid and node.terminal for node in leaves) / len(leaves),
                           prompt_tokens=sum(len(node.prompt_ids) for node in flat),
                           max_prompt_tokens=max(len(node.prompt_ids) for node in flat),
                           wall_seconds=time.perf_counter() - started, resources=resource_snapshot(),
                           resource_peaks=dict(monitor.peaks))
            write_jsonl(output / 'rollouts' / f'step-{step + 1:06d}.jsonl.gz',
                        (forest_record(qa, forest, step + 1, seed) for qa, forest, seed in zip(batch, forests, seeds)))
            budget = get_budget()
            if budget:
                metrics['storage_observed_bytes'] = budget.check(0, 'step finished')
            write_json(output / 'metrics' / f'step-{step + 1:06d}.json', metrics)
            print(metrics, flush=True)
            state.update(step=step + 1, question_cursor=state['question_cursor'] + batch_size)
            scheduled_restore = state['step'] == restore_step and not (output / 'RESTORE_VERIFIED.json').exists()
            if state['step'] % config['retrieval']['save_every'] == 0 or state['step'] == config['retrieval']['steps'] or stop_requested():
                checkpoint = manager.save(state)
            if stop_requested():
                write_json(output / 'STOPPED.json', dict(state, checkpoint=str(checkpoint)))
                manager.event('stopped_at_checkpoint', checkpoint=str(checkpoint), step=state['step'])
                return
            if scheduled_restore:
                write_json(output / 'RESTORE_REQUIRED.json', {'checkpoint': str(checkpoint), 'step': state['step']})
                manager.event('scheduled_restart_for_restore_verification', checkpoint=str(checkpoint), step=state['step'])
                return
            if state['step'] % validate_every == 0 or state['step'] == config['retrieval']['steps']:
                validate(state['step'])
            if stop_requested():
                write_json(output / 'STOPPED.json', dict(state, checkpoint=str(checkpoint)))
                manager.event('stopped_at_checkpoint', checkpoint=str(checkpoint), step=state['step'])
                return
    selected = read_json(output / 'BEST.json')
    from training.checkpoints import verify_checkpoint
    verify_checkpoint(selected['path'], optimizer=False)
    manager.event('export_selected_model', **selected)
    final = output / 'final'
    marker = final / 'COMPLETE.json'
    if marker.exists():
        if read_json(marker)['checkpoint'] != selected['path']:
            raise ValueError('Existing export is from a different model selection')
    else:
        if final.exists() and any(final.iterdir()):
            manager.event('discard_incomplete_export', path=str(final),
                          files=[{'path': file.name, 'bytes': file.stat().st_size} for file in final.iterdir()])
            for file in final.iterdir():
                if file.is_symlink() or not file.is_file():
                    raise ValueError('Unexpected path in incomplete HF export')
            for file in final.glob('*.safetensors'):
                file.unlink()
            archived = output / 'exports_interrupted' / str(time.time_ns())
            archived.parent.mkdir(parents=True, exist_ok=True)
            final.rename(archived)
        updater.export(final, checkpoint=selected['path'])
    write_json(output / 'TRAINING_COMPLETE.json', dict(state, selected=selected))
