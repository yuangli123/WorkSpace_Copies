from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import time
from contextlib import nullcontext

from memory.branch import Ranker, digest
from memory.protocol import Completion
from methods.joint.prompts import ANSWER_RULES


class AnswerQueue:
    def __init__(self, policy_version=0, protocol=None):
        self.pending, self.results = {}, {}
        self.policy_version = policy_version
        self.protocol = dict(rules=ANSWER_RULES, temperature=0, top_p=1., top_k=-1,
                             min_p=0., stop_at_tool=False, **(protocol or {}))

    def submit(self, question, pack, mode):
        # Gold answer/evidence/category never enter the answerer's request or cache key.
        payload = {'question': question['question'], 'question_time': question.get('question_time', ''),
                   'evidence': pack, 'mode': mode}
        key = digest([self.policy_version, self.protocol, payload])
        self.pending[key] = payload
        return key

    @staticmethod
    def messages(payload):
        return [{'role': 'system', 'content': '[ROLE:Answer] ' + ANSWER_RULES},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]

    def resolve(self, answerer, phase):
        missing = [(key, value) for key, value in self.pending.items() if key not in self.results]
        if not missing:
            return
        with phase():
            if hasattr(answerer, 'complete_batch'):
                completions = answerer.complete_batch([self.messages(p) for _, p in missing],
                    list(range(len(missing))), temperature=0, stop_at_tool=False)
                if len(completions) != len(missing):
                    raise RuntimeError('Answer batch size mismatch')
                for (key, _), completion in zip(missing, completions):
                    self.results[key] = {'text': completion.text, 'prompt_tokens': len(completion.prompt_ids),
                        'generated_tokens': len(completion.response_ids), 'policy_version': self.policy_version}
                return
            for key, payload in missing:
                messages = self.messages(payload)
                completion = answerer.complete(messages, temperature=0, stop_at_tool=False)
                self.results[key] = {'text': completion.text, 'prompt_tokens': len(completion.prompt_ids),
                                     'generated_tokens': len(completion.response_ids), 'policy_version': self.policy_version}

    def text(self, key):
        if not key:
            return ''
        return self.results[key]['text']


class SharedAnswer:
    """Inference on this step's policy; never creates trainable Answer rows."""
    def __init__(self, policy):
        self.policy = policy

    def phase(self):
        return nullcontext()

    def complete_batch(self, messages, seeds, **kwargs):
        if hasattr(self.policy, 'complete_batch'):
            return self.policy.complete_batch(messages, seeds, **kwargs)
        result = []
        for prompt, seed in zip(messages, seeds):
            self.policy.set_seed(seed)
            result.append(self.policy.complete(prompt, **kwargs))
        return result


class FixturePolicy:
    """Rule fixture for environment checks, deliberately not an RL policy."""
    def __init__(self):
        self.next_seed, self.calls = 0, []
        self.role = 'R'

    def generation_for(self, role):
        self.role = role
        return self

    def set_seed(self, seed):
        self.next_seed = seed

    def complete(self, messages, max_new_tokens=None, temperature=None, stop_at_tool=True):
        self.calls.append(copy.deepcopy(messages))
        rng = random.Random(self.next_seed if temperature != 0 else 0)
        system, user = messages[0]['content'], messages[-1]['content']
        def call(name, arguments):
            return '<tool_call>' + json.dumps({'name': name, 'arguments': arguments}) + '</tool_call>'
        if self.role == 'B':
            items = re.findall(r'\[([^\]\n]+)\][^\n]*?residence=(\w+)', user)
            calls = [call('update_summary', {'content': '; '.join('residence=' + city for _, city in items)})]
            if items and (temperature == 0 or rng.random() < 0.8):
                calls += [call('create_fact', {'fact': 'residence=' + city, 'start_time': '', 'end_time': ''}) for _, city in items]
            if items and rng.random() < 0.6:
                calls.append(call('create_raw', {'turn_ids': [sid for sid, _ in items]}))
            text = '\n'.join(calls)
        elif self.role == 'U':
            match = re.search(r'\[New memory to be added\]\s*(.*?)\s*\[Existing memories', user, re.S)
            candidate = json.loads(match.group(1))
            text = call('add_item', {'document': candidate['document'], 'start_time': '', 'end_time': ''})
        elif not stop_at_tool:
            cities = re.findall(r'residence=(\w+)', user)
            text = (cities[0] if 'previous' in user.lower() else cities[-1]) if cities else 'Information is insufficient.'
        else:
            if '<observation>' in user:
                observation = user.split('<observation>')[1].split('</observation>')[0]
                if observation:
                    cities = re.findall(r'residence=(\w+)', observation)
                    text = call('finish', {'answer': cities[-1] if cities else 'Information is insufficient.'})
                else:
                    text = call('search_summary', {'query': 'residence'})
            else:
                tools = ['search_facts', 'search_summary']
                if 'search_turns' in system:
                    tools.append('search_turns')
                text = call(rng.choice(tools), {'query': 'residence'})
        encode = lambda x: [b + 1 for b in x.encode('utf-8')]
        return Completion(text, encode(json.dumps(messages, ensure_ascii=False)), encode(text))


class Runtime:
    def __init__(self, policy, config, fixed=None, ranker=None, fixed_policy=None):
        self.policy, self.config, self.fixed = policy, config, fixed
        self.ranker = ranker or Ranker(config['memory'].get('embedding_model'), config['backend'] == 'fixture')
        self.mode = config['experiment']['mode']
        # Kept as constructor arguments for fixture/test compatibility.  The
        # production path always generates with the current shared policy.
        self.fixed_policy = None
        # AnswerQueue/SharedAnswer remain import-compatible legacy helpers, but
        # the production runtime has no queued or second Answer phase.
        self.answer_protocol = {'name': config.get('protocol', {}).get('name', 'legacy'),
                                'backend': config['backend'], 'max_new_tokens': config['model']['max_new_tokens']}
        self.calls = []
        self.policy_version = 0
        self.progress = None

    def generation_session(self):
        return self.policy.generation_session() if hasattr(self.policy, 'generation_session') else nullcontext()

    def seed(self, event):
        return int(hashlib.sha256((str(self.config['seed']) + '/' + event).encode()).hexdigest()[:8], 16) % 2147483647

    def random(self, event):
        return random.Random(self.seed(event))

    def count(self, text):
        tokenizer = getattr(self.policy, 'tokenizer', None)
        return len(tokenizer.encode(text, add_special_tokens=False)) if tokenizer else len(text.encode('utf-8'))

    def prompt_fits(self, messages):
        tokenizer = getattr(self.policy, 'tokenizer', None)
        if tokenizer:
            from models.hf import render_prompt
            size = len(tokenizer.encode(render_prompt(tokenizer, messages), add_special_tokens=False))
        else:
            size = self.count(json.dumps(messages, ensure_ascii=False))
        return size <= self.config['model'].get('effective_prompt_tokens', self.config['retrieval']['max_prompt_tokens'])

    def fallback_answer(self, question, memories, event):
        """Use the unmodified Mem-T fallback prompt with the current policy."""
        from memory.retrieval import AnswerWithMemoriesTool
        runtime = self
        class AnswerPolicy:
            def get_completion(self, text):
                completion, _ = runtime.call_batch('R', [[{'role': 'user', 'content': text}]],
                    [event + '/fallback'], greedy=True, stop_at_tool=False)[0]
                return completion.text
        answer, _ = AnswerWithMemoriesTool(AnswerPolicy(), self.config.get('benchmark', 'locomo'),
                                          str(question.get('category', '')))(question['question'], memories)
        return answer

    def call(self, role, messages, event, greedy=False):
        return self.call_batch(role, [messages], [event], greedy)[0]

    def call_batch(self, role, batches, events, greedy=False, stop_at_tool=None):
        started = time.perf_counter()
        if len(batches) != len(events):
            raise ValueError('Batch/event count mismatch')
        if not batches:
            return []
        policy = self.policy.generation_for(role) if hasattr(self.policy, 'generation_for') else (
            self.policy.activate(role) if hasattr(self.policy, 'activate') else self.policy)
        tagged = []
        for messages in batches:
            messages = copy.deepcopy(messages)
            tagged.append(messages)
        seeds = [self.seed(event) for event in events]
        temperature = 0 if greedy else self.config['model']['temperature']
        if stop_at_tool is None:
            stop_at_tool = role == 'R'
        if hasattr(policy, 'complete_batch'):
            completions = policy.complete_batch(tagged, seeds, max_new_tokens=self.config['model']['max_new_tokens'],
                                                temperature=temperature, stop_at_tool=stop_at_tool)
        else:
            completions = []
            for messages, seed in zip(tagged, seeds):
                policy.set_seed(seed)
                completions.append(policy.complete(messages, temperature=temperature,
                    max_new_tokens=self.config['model']['max_new_tokens'], stop_at_tool=stop_at_tool))
        if len(completions) != len(events):
            raise RuntimeError('Generation backend returned an incorrect batch size')
        result = []
        elapsed = time.perf_counter() - started
        for messages, event, seed, completion in zip(tagged, events, seeds, completions):
            row = {'role': role, 'event': event, 'sampling_seed': seed, 'messages': messages,
                   'prompt_ids': completion.prompt_ids, 'response_ids': completion.response_ids,
                   'text': completion.text, 'evaluator_only': greedy}
            row.update(generation_seconds_share=elapsed/len(completions), generation_backend=type(policy).__name__)
            row['policy_version'] = self.policy_version
            self.calls.append(row)
            result.append((completion, row))
        if self.progress:
            self.progress({'phase': 'sampling', 'role': role, 'event': events[-1],
                           'policy_calls': len(self.calls), 'policy_version': self.policy_version,
                           'last_batch_seconds': elapsed})
        return result

