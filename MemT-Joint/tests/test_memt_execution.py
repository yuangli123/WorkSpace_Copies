"""Differential checks against the user's original Mem-T checkout."""
import ast
import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from experiments.joint_config import load, contract
from memory.branch import State, Ranker, builder, builder_prompt, updater, updater_prompt
from memory.protocol import Completion
from methods.joint.prompts import retrieval_prompt, retriever
from methods.joint.retrieval import Node, step, sample
from models.joint import Runtime, FixturePolicy

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / 'Mem-T'


def original(relative):
    module = ModuleType('memt_reference')
    source = (REFERENCE / relative).read_text(encoding='utf-8')
    source = source.replace('from loguru import logger', 'from memory.logging import logger')
    exec(compile(source, str(REFERENCE / relative), 'exec'), module.__dict__)
    return module


def action(name, **args):
    return '<tool_call>' + json.dumps({'name': name, 'arguments': args}) + '</tool_call>'


@pytest.fixture
def environment():
    config = load(ROOT / 'configs/fixture.json')
    runtime = Runtime(FixturePolicy(), config)
    state = State('episode', Ranker(fixture=True))
    turns = [{'turn_id': 't1', 'speaker': 'User', 'text': 'reading The Lean Startup'},
             {'turn_id': 't2', 'speaker': 'User', 'text': 'residence=Beijing'}]
    state.observe({'session_id': 'one', 'time': '20 January, 2023'}, turns, (0, 0), runtime.count)
    text = action('create_raw', turn_ids=['t2', 't1']) + action('update_summary', content='reading and residence')
    state, _, _ = builder(state, turns, Completion(text, [1], [2]), 'B')
    return state, runtime


def test_reference_hashes_and_tool_implementation():
    reference = json.loads((ROOT / 'MEMT_EXECUTION_REFERENCE.json').read_text())
    for name, expected in reference['sha256'].items():
        assert hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest() == expected
    # The only adaptation in this vendor file is the project logger import.
    def definitions(path):
        source = path.read_text(encoding='utf-8').replace(
            'logger.error(f"Error retrieving latest summary: {e}")\n                return "", []',
            'logger.error(f"Error retrieving latest summary: {e}")\n                raise')
        tree = ast.parse(source)
        return [ast.dump(n, include_attributes=False) for n in tree.body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    for name in ['memory/retrieval.py', 'memory/formation.py', 'memory/formation_tools.py', 'memory/update.py']:
        assert definitions(ROOT / name) == definitions(REFERENCE / name)


def test_summary_database_failure_propagates(environment):
    state, runtime = environment
    class FailedDB:
        def get(self, *args, **kwargs):
            raise RuntimeError('database unavailable')
    state.journal = FailedDB()
    with pytest.raises(RuntimeError, match='database unavailable'):
        step(state, Node(1, 0, 1, []), action('search_summary', query=''), runtime)


@pytest.mark.parametrize('category', [1, 2, 3, 4])
def test_prompt_and_schema_equal_original(category):
    reader = retriever(None, 'source_backed', 5, 6)
    source = original('memory/retrieval.py').MemoryRetriever(None, None, reader.config)
    question = {'question': 'When did Jon lose his job?', 'category': category, 'answer': 'GOLD'}
    assert retrieval_prompt(question, 'source_backed', 5, 6) == source.construct_prompt(question['question'], str(category))
    assert reader.get_tool_schemas() == source.get_tool_schemas()
    assert 'GOLD' not in json.dumps(retrieval_prompt(question, 'source_backed', 5, 6))


@pytest.mark.parametrize('name,args', [('search_turns', {'query': 'reading', 'top_k': 50}),
                                     ('search_summary', {'query': ''}),
                                     ('search_summary', {'query': 'reading'}),
                                     ('search_facts', {'query': 'absent'})])
def test_observation_and_dedup_equal_original(environment, name, args):
    state, runtime = environment
    reader = retriever(state.journal, 'source_backed', 5, 6)
    source = original('memory/retrieval.py').MemoryRetriever(state.journal, None, reader.config)
    seen, node = set(), Node(1, 0, 1, [])
    for _ in range(2):
        expected, _ = source.tool_map[name](**args, sample_id=state.sample_id, seen_turns=seen)
        step(state, node, action(name, **args), runtime)
        assert node.observation == (expected or 'No matching memories.')
        assert set(node.seen_turns) == seen
    if name == 'search_turns':
        assert 'No new relevant turns' in node.observation
        assert set(node.displayed) == {'t1', 't2'}


def test_fallback_prompt_equal_original_without_role_tags(environment):
    state, runtime = environment
    class Capture:
        def get_completion(self, text):
            self.text = text
            return 'expected'
    capture = Capture()
    question = {'question': 'what book?', 'category': 4}
    original('memory/retrieval.py').AnswerWithMemoriesTool(capture, 'locomo', '4')(question['question'], 'visible evidence')
    runtime.fallback_answer(question, 'visible evidence', 'q')
    assert runtime.calls[-1]['messages'] == [{'role': 'user', 'content': capture.text}]
    assert runtime.calls[-1]['role'] == 'R'
    assert '[ROLE:' not in json.dumps(runtime.calls[-1]['messages'])


@pytest.mark.parametrize('invalid', ['malformed', action('nonexistent_tool')])
def test_invalid_action_fallback_context_matches_memt(environment, invalid):
    from methods.retrieval.tree import RetrievalEnvironment
    state, runtime = environment
    runtime.config['evaluation']['fallback_answer'] = True
    source = RetrievalEnvironment(state.journal, state.sample_id, runtime.config, ROOT/'artifacts/tool-logs')
    _, _, _, observation, _, _ = source.step(invalid, [])
    class Capture:
        def get_completion(self, text):
            self.text = text
            return 'fallback'
    capture = Capture()
    question = {'question': 'What is Gina reading?', 'category': 4, 'answer': 'The Lean Startup'}
    original('memory/retrieval.py').AnswerWithMemoriesTool(capture, 'locomo', '4')(question['question'], observation)
    runtime.policy = SimpleNamespace(set_seed=lambda seed: None,
        complete=lambda messages, stop_at_tool=True, **kwargs: Completion(invalid if stop_at_tool else 'fallback', [1], [2]))
    result = sample(state, question, runtime, 'invalid', forest=False, greedy=True)
    assert runtime.calls[-1]['messages'] == [{'role': 'user', 'content': capture.text}]
    assert result['trees'][0][-1].fallback


def test_real_model_budget_errors_are_not_invalid_actions(environment):
    state, runtime = environment
    runtime.config['retrieval']['max_observation_tokens'] = 1
    with pytest.raises(ValueError, match='observation budget'):
        step(state, Node(1, 0, 1, []), action('search_turns', query='reading'), runtime)
    runtime.config['retrieval']['max_observation_tokens'] = 10000
    runtime.config['retrieval']['token_budget'] = 1
    step(state, Node(1, 0, 1, []), action('search_turns', query='reading'), runtime)


def test_memory_only_hides_raw_sources(environment):
    state, runtime = environment
    runtime.mode = 'memory_only'
    node = Node(1, 0, 1, [])
    step(state, node, action('search_summary', query='reading'), runtime)
    assert 'The Lean Startup' not in node.observation
    assert not node.displayed


def test_unshown_metadata_sources_do_not_count_as_evidence(environment):
    state, runtime = environment
    state.journal.add(state.collection('facts'), ['partial'], ['a fact'],
        [{'memory_content': 'a fact', 'original_turns': [state.sources['t1']], 'source_turn_ids': ['t1', 't2']}])
    node = Node(1, 0, 1, [])
    step(state, node, action('search_facts', query='fact'), runtime)
    assert node.displayed == ['t1']
    assert 'residence=Beijing' not in node.observation


def test_builder_persona_context_matches_memt(environment):
    state, runtime = environment
    state.personas = {'Gina': 'an avid reader'}
    turns = [{'turn_id': 't1'}, {'turn_id': 't2'}]
    formation = original('memory/formation.py').MemoryFormation(None, 'paper-v2')
    expected = formation.construct_prompt('\n'.join('[%s] %s' % (t['turn_id'], state.sources[t['turn_id']]) for t in turns),
        state.summary, 'Name: Gina, Profile: an avid reader')
    assert builder_prompt(state, turns) == expected


def test_multi_turn_messages_and_direct_answer_match_memt(environment):
    from methods.retrieval.tree import Node as MemTNode, next_messages
    state, runtime = environment
    outputs = iter([action('search_turns', query='reading'), action('finish', answer='The Lean Startup')])
    runtime.policy = SimpleNamespace(set_seed=lambda seed: None,
        complete=lambda messages, **kwargs: Completion(next(outputs), [1], [2]))
    question = {'question': 'What is Gina reading?', 'category': 4, 'answer': 'The Lean Startup'}
    result = sample(state, question, runtime, 'dialogue', forest=False, greedy=True)
    root, first, leaf = result['trees'][0]
    parent = MemTNode(1, 0, 1, root.messages, text=first.text, observation=first.observation)
    assert runtime.calls[1]['messages'] == next_messages(parent, question)
    assert leaf.answer == 'The Lean Startup' and not leaf.fallback
    assert len(runtime.calls) == 2


def test_updater_candidate_time_and_original_source_grouping(environment):
    state, runtime = environment
    state.journal.add(state.collection('facts'), ['existing'], ['fact'],
                      [{'memory_content': 'fact', 'turn_time': '19 January, 2023'}])
    item = {'type': 'fact', 'document': 'new fact', 'start_time': '', 'end_time': '',
            'producer': 'B', 'source_turn_ids': ['t1', 't2'], 'turn_time': state.time}
    messages, related = updater_prompt(state, item, 3)
    potential = {k: v for k, v in item.items() if k not in ('producer', 'source_turn_ids')}
    source = original('memory/update.py').MemoryUpdate(None, state.journal)
    assert messages == source.construct_prompt(potential, related, source.get_tool_schemas(state.collection('facts')))
    assert related[0]['turn_time'] == '19 January, 2023'
    # A known database target is valid even when absent from the visible top-k.
    updated, operations = updater(state, item, [], Completion(action('update_item', id='existing', document='changed'), [1], [2]), 'U')
    assert operations[0]['valid']
    meta = updated.journal.get(state.collection('facts'), ['existing'])['metadatas'][0]
    assert len(meta['original_turns']) == 1 and '[t1]' in meta['original_turns'][0] and '[t2]' in meta['original_turns'][0]


def test_new_protocol_rejects_old_runs_and_has_no_role_tags(environment):
    state, runtime = environment
    runtime.policy = SimpleNamespace(set_seed=lambda seed: None,
        complete=lambda messages, **kwargs: Completion('ok', [1], [2]))
    for role in 'BUR':
        runtime.call(role, [{'role': 'system', 'content': 'prompt'}, {'role': 'user', 'content': 'query'}], role)
    assert all('[ROLE:' not in json.dumps(row['messages']) for row in runtime.calls)
    config = load(ROOT / 'configs/common-a100x2-50.json')
    protocol = contract(config)
    assert protocol['memory_construction'] == 'empty_state_chronological'
    assert protocol['role_tags'] is False and protocol['baseline_reuse'] is False
    config['protocol']['name'] = 'memt-common-v1'
    with pytest.raises(ValueError, match='Unknown common'):
        contract(config)
