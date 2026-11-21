import copy
import json

import pytest

from experiments.io import load_config, read_json, write_json, write_jsonl
from experiments.workflow import build, MemoryRun
from memory.formation import CreateRawTool
from memory.journal import JournalDB, ListDB
from memory.protocol import Completion
from methods.construction.rollout import construct_episode
from methods.construction.credit import select_operations


@pytest.fixture
def config():
    cfg = load_config('configs/memt4b.yaml')
    cfg['memory'].update(backend='lexical_test', summary_context_turns=1)
    cfg['construction'].update(protocol='paper-v2', rollouts=1)
    cfg['retrieval'].update(expansion_rounds=1, evidence_alpha=1)
    return cfg


class RawPolicy:
    def __init__(self, interrupt=False):
        self.interrupt = interrupt
        self.calls = 0
        self.seed = None
    def set_seed(self, seed):
        self.seed = seed
    def complete(self, messages, **kwargs):
        self.calls += 1
        if self.interrupt and self.calls == 2:
            raise RuntimeError('interrupted between sealed chunks')
        user = messages[-1]['content']
        turn = 'D1:2' if '[D1:2]' in user else 'D1:1'
        text = '<tool_call>' + json.dumps({'name': 'create_raw', 'arguments': {'turn_ids': [turn]}}) + '</tool_call>'
        return Completion(text, [1], [2])


def episode():
    return {'sample_id': 's', 'sessions': [{'session_id': 'a', 'time': '2023-01-01', 'turns': [
        {'turn_id': 'D1:1', 'speaker': 'Alice', 'text': 'Paris'},
        {'turn_id': 'D1:2', 'speaker': 'Alice', 'text': 'Rome'}]}]}


def test_raw_id_boundary_and_verbatim_store(config):
    assert CreateRawTool.valid_ids(['D1:1'], ['D1:1'])
    for invalid in ([], ['D1:1', 'D1:1'], ['D9:1'], 'D1:1', [7]):
        assert not CreateRawTool.valid_ids(invalid, ['D1:1'])
    db = JournalDB(ListDB())
    operations, _ = construct_episode(episode(), RawPolicy(), db, config)
    rows = db.get('s_turns')
    assert len(rows['ids']) == len(operations) == 2
    assert rows['documents'] == ['Alice at 2023-01-01: Paris', 'Alice at 2023-01-01: Rome']
    assert all(op['protocol_version'] == 'paper-v2' and op['sampling_seed'] is not None for op in operations)
    assert rows['metadatas'][0]['operation_ids'] == [operations[0]['op_id']]


def test_invalid_call_excludes_entire_completion():
    a = dict(op_id='a', valid=True, score=1, phase='formation', category='create_raw', completion_id='same')
    b = dict(a, op_id='b', valid=False)
    c = dict(a, op_id='c', completion_id='other')
    selected, _ = select_operations([a, b, c])
    assert [op['op_id'] for op in selected] == ['c']


def test_chunk_resume_has_identical_operations_and_sources(config, tmp_path):
    data = tmp_path / 'data'
    write_jsonl(data / 'episodes.jsonl', [episode()])
    write_jsonl(data / 'questions.jsonl', [{'qa_id': 'q', 'sample_id': 's', 'question': 'Where?',
                                         'answer': 'Rome', 'category': 1, 'evidence': ['D1:2']}])
    write_json(data / 'partition.json', {'split': 'train'})
    with pytest.raises(RuntimeError, match='interrupted'):
        build(copy.deepcopy(config), data, tmp_path / 'resumed', RawPolicy(interrupt=True))
    build(copy.deepcopy(config), data, tmp_path / 'resumed', RawPolicy(), resume=True)
    build(copy.deepcopy(config), data, tmp_path / 'continuous', RawPolicy())
    left, right = MemoryRun(tmp_path / 'resumed', config), MemoryRun(tmp_path / 'continuous', config)
    assert left.operations == right.operations
    assert left.db('s__r0').get('s__r0_turns') == right.db('s__r0').get('s__r0_turns')
    changed = copy.deepcopy(config)
    changed['seed'] += 1
    with pytest.raises(ValueError, match='contract changed'):
        build(changed, data, tmp_path / 'resumed', RawPolicy(), resume=True)


def test_config_scientific_notation_and_includes(tmp_path):
    (tmp_path / 'base.yaml').write_text('construction:\n  learning_rate: 5e-06\n', encoding='utf-8')
    (tmp_path / 'run.yaml').write_text('extends: [base.yaml]\nconstruction:\n  steps: 4\n', encoding='utf-8')
    result = load_config(tmp_path / 'run.yaml')
    assert result['construction']['learning_rate'] == 5e-6
    assert isinstance(result['construction']['learning_rate'], float)
    (tmp_path / 'base.yaml').write_text('extends: [run.yaml]\n', encoding='utf-8')
    with pytest.raises(ValueError, match='cycle'):
        load_config(tmp_path / 'run.yaml')


def test_root_layout_entry_and_source_manifest():
    import subprocess
    import sys
    from pathlib import Path
    from experiments.io import manifest
    root = Path(__file__).resolve().parents[1]
    assert not (root / 'memt').exists() and not (root / 'memt_training').exists()
    assert b'\r' not in (root / 'scripts/slurm_train.sh').read_bytes()
    result = subprocess.run([sys.executable, 'cli.py', '--help'], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0 and 'run-construction' in result.stdout
    code = manifest({})['code_sha256']
    assert all(name in code for name in ('cli.py','models/hf.py','memory/formation_tools.py','methods/retrieval/tree.py'))
    assert all(not name.startswith(('memt/', 'memt_training/')) for name in code)


def test_oversized_target_is_never_silently_truncated():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from training.sft import sft_example
    raw = Tokenizer(models.WordLevel({'[UNK]': 0, '[EOS]': 1, 'a': 2, 'user': 3, 'assistant': 4}, unk_token='[UNK]'))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', eos_token='[EOS]')
    tokenizer.chat_template = '{% for m in messages %}{{ m["role"] + " " + m["content"] }}{% endfor %}assistant '
    text = ' '.join(['a'] * 50)
    op = {'messages': [{'role': 'user', 'content': 'a'}], 'response': text, 'target_span': [0, len(text)]}
    with pytest.raises(ValueError, match='partial-action'):
        sft_example(tokenizer, op, 16)
    short = dict(op, response='a a', target_span=[2, 3])
    row = sft_example(tokenizer, short, 16)
    assert sum(v != -100 for v in row['labels']) == 2  # selected action and final EOS


def test_sealed_progress_rejects_corruption(config, tmp_path):
    from experiments.progress import save_chunk, restore_chunk
    db = JournalDB(ListDB())
    db.create_collection('x')
    save_chunk(tmp_path, {'cursor': 1}, db, ['x'])
    pointer = read_json(tmp_path / 'CURRENT.json')
    (tmp_path / pointer['file']).write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='seal'):
        restore_chunk(tmp_path, ListDB())


def test_cpu_adam_cache_normalization_preserves_state_and_scheduler_hook():
    from training.cpu_adam import normalize_bias_cache
    class Kernel:
        def __init__(self):
            self.calls = []
        def destroy_adam(self, identity):
            self.calls.append(('destroy', identity))
        def create_adam(self, *args):
            self.calls.append(('create', args))
    class Optimizer:
        def __init__(self):
            self.ds_opt_adam, self.opt_id, self.adam_w_mode = Kernel(), 7, True
            self.state = {'step': 2, 'moment': [1, 2]}
            self.param_groups = [{'params': [object()], 'lr': .01, 'betas': (.9, .999), 'eps': 1e-8, 'weight_decay': 0.}]
        def step(self, closure=None):
            self.state['step'] += 1
            return closure() if closure else None
    Optimizer.step._wrapped_by_lr_sched = True
    optimizer = Optimizer()
    state, groups = optimizer.state, optimizer.param_groups
    assert normalize_bias_cache(optimizer) and normalize_bias_cache(optimizer)
    assert optimizer.step._wrapped_by_lr_sched
    assert optimizer.step(closure=lambda: 'result') == 'result'
    optimizer.param_groups[0]['lr'] = .005
    optimizer.step()
    assert optimizer.state is state and optimizer.param_groups is groups
    assert optimizer.state == {'step': 4, 'moment': [1, 2]}
    assert len(optimizer.ds_opt_adam.calls) == 4
    assert optimizer.ds_opt_adam.calls[-1][1][1] == .005
    assert not normalize_bias_cache(None) and not normalize_bias_cache(object())
