from __future__ import annotations

import copy
import math
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from memory.protocol import flatten, parse_calls, validate_call


def normalize(text):
    text = re.sub(r"[^\w\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", re.sub(r"\b(a|an|the)\b", " ", text)).strip()


def answer_f1(prediction, answer):
    if isinstance(answer, list):
        return max((answer_f1(prediction, a) for a in answer), default=0.0)
    left, right = normalize(prediction).split(), normalize(answer).split()
    if not left or not right:
        return float(left == right)
    common = sum((Counter(left) & Counter(right)).values())
    return 2 * common / (len(left) + len(right))


def answer_bleu1(prediction, answer):
    if isinstance(answer, list):
        return max((answer_bleu1(prediction, a) for a in answer), default=0.0)
    left, right = normalize(prediction).split(), normalize(answer).split()
    if not left or not right:
        return 0.0
    precision = sum((Counter(left) & Counter(right)).values()) / len(left)
    return precision * (1.0 if len(left) >= len(right) else math.exp(1 - len(right) / len(left)))


@dataclass
class Node:
    id: int
    parent: int
    depth: int
    messages: list
    prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    text: str = ""
    valid: bool = True
    terminal: bool = False
    answer: str = ""
    observation: str = ""
    retrieved: list[dict] = field(default_factory=list)
    seen_turns: list[str] = field(default_factory=list)
    children: list[int] = field(default_factory=list)
    evidence: float = 0.0
    performance: float = 0.0
    reward: float = 0.0
    intra: float = 0.0
    inter: float = 0.0
    advantage: float = 0.0


class RetrievalEnvironment:
    def __init__(self, db, sample_id, config, log_path):
        from memory.settings import SystemConfig
        from memory.retrieval import MemoryRetriever
        settings = SystemConfig()
        settings.data_name = config.get("benchmark", "locomo")
        settings.memory.retrieval_topk = config["memory"]["retrieval_topk"]
        settings.memory.max_tool_steps = config["retrieval"]["max_depth"]
        settings.traj_dir = str(log_path)
        self.retriever = MemoryRetriever(db, None, settings)
        self.sample_id = sample_id

    def prompt(self, question, category):
        return self.retriever.construct_prompt(question, str(category))

    def step(self, text, seen_turns):
        calls = parse_calls(text)
        if len(calls) != 1:
            return False, True, "", "Invalid tool-call format.", [], seen_turns
        call = calls[0]
        tool = self.retriever.tool_map.get(call["name"])
        if tool is None or not validate_call(call, tool):
            return False, True, "", "Invalid tool name or arguments.", [], seen_turns
        if call["name"] == "finish":
            answer = call["arguments"]["answer"]
            return bool(answer.strip()), True, answer, "", [], seen_turns
        seen = set(seen_turns)
        # Call directly so database/runtime failures are not mislabelled as bad policy actions.
        observation, metadata = tool(**call["arguments"], sample_id=self.sample_id, seen_turns=seen)
        return True, False, "", observation or "No matching memories.", metadata, sorted(seen)


def next_messages(parent, question):
    messages = copy.deepcopy(parent.messages)
    if parent.id:
        messages.extend([{"role": "assistant", "content": parent.text},
                         {"role": "user", "content": f'The retrieved memories are: <observation>{parent.observation}</observation>\n Verify if you are confident enough to answer the <question>{question["question"]}</question> correctly and sufficiently based on the above memories; if not, continue using the tools to retrieve more information.'}])
    return messages


def append_node(nodes, parent, messages, completion, environment, policy):
    valid, terminal, answer, observation, metadata, seen = environment.step(completion.text, parent.seen_turns)
    if hasattr(policy, "check_observation"):
        policy.check_observation(observation)
    retrieved = {m["memory_ref"]: m for m in parent.retrieved if "memory_ref" in m}
    for meta in metadata:
        if "memory_ref" not in meta:
            raise ValueError("Retrieval result lacks versioned provenance")
        retrieved[meta["memory_ref"]] = meta
    node = Node(len(nodes), parent.id, parent.depth + 1, messages,
                completion.prompt_ids, completion.response_ids, completion.text,
                valid, terminal, answer, observation, list(retrieved.values()), seen)
    nodes.append(node)
    parent.children.append(node.id)
    return node


def sample_tree(policy, environment, question, config, rng):
    nodes = [Node(0, -1, 0, environment.prompt(question["question"], question["category"]))]
    depth = config["retrieval"]["max_depth"]

    def extend(pivot):
        parent = nodes[pivot]
        while parent.depth < depth and not parent.terminal:
            messages = next_messages(parent, question)
            completion = policy.complete(messages, max_new_tokens=config["model"]["max_new_tokens"])
            parent = append_node(nodes, parent, messages, completion, environment, policy)
    extend(0)
    for _ in range(config["retrieval"]["expansion_rounds"]):
        candidates = [n.id for n in nodes if not n.terminal and n.depth < depth]
        # Match the released Tree-GRPO's with-replacement pivot sampling.
        for pivot in rng.choices(candidates, k=config["retrieval"]["pivots"]):
            extend(pivot)
    return nodes


def standardize(values, epsilon=1e-6):
    if not values:
        return []
    mean = sum(values) / len(values)
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
    return [(x - mean) / (std + epsilon) for x in values]


def score_forest(forest, qa, config):
    """Equations 10-13: child mean (not descendant-leaf mean), two node populations."""
    cfg = config["retrieval"]
    gold_evidence = set(qa.get("evidence", []))
    for nodes in forest:
        for node in reversed(nodes[1:]):
            sources = {s for m in node.retrieved for s in flatten(m.get("source_turn_ids", []))}
            node.evidence = len(sources & gold_evidence) / len(gold_evidence) if gold_evidence else 0.0
            if node.children:
                node.performance = sum(nodes[c].performance for c in node.children) / len(node.children)
            else:
                node.performance = answer_f1(node.answer, qa["answer"]) if node.valid and node.terminal else 0.0
            node.reward = float(node.valid) * (cfg["evidence_alpha"] * node.evidence + node.performance)
    all_nodes = [n for nodes in forest for n in nodes[1:]]
    inter = standardize([n.reward for n in all_nodes])
    for node, advantage in zip(all_nodes, inter):
        node.inter = advantage
    for nodes in forest:
        intra = standardize([n.reward for n in nodes[1:]])
        for node, advantage in zip(nodes[1:], intra):
            node.intra = advantage
            node.advantage = float(cfg["use_intra"]) * node.intra + float(cfg["use_inter"]) * node.inter
    return forest


def sample_forest(policy, environment, qa, config, seed):
    if hasattr(policy, "set_seed"):
        policy.set_seed(seed)
    rng = random.Random(seed)
    count = config["retrieval"]["trees"]
    if config["retrieval"].get("algorithm", "mot_grpo") == "grpo":
        # Keep a matched maximum number of complete rollouts, then use outcome GRPO.
        count *= 1 + config["retrieval"]["expansion_rounds"] * config["retrieval"]["pivots"]
        independent = copy.deepcopy(config)
        independent["retrieval"]["expansion_rounds"] = 0
        forest = [sample_tree(policy, environment, qa, independent, rng) for _ in range(count)]
        leaves = [nodes[-1] for nodes in forest]
        advantages = standardize([answer_f1(n.answer, qa["answer"]) if n.valid and n.terminal else 0.0 for n in leaves])
        for nodes, advantage in zip(forest, advantages):
            for node in nodes[1:]:
                node.performance = answer_f1(nodes[-1].answer, qa["answer"]) if nodes[-1].valid and nodes[-1].terminal else 0.0
                node.reward, node.advantage = node.performance, advantage
        return forest
    forest = [sample_tree(policy, environment, qa, config, rng) for _ in range(count)]
    return score_forest(forest, qa, config)


def sample_forests(policy, environments, questions, config, seeds):
    """Batch active nodes across questions/trees, without sharing their environment states."""
    if not hasattr(policy, "complete_batch"):
        return [sample_forest(policy, env, qa, config, seed) for env, qa, seed in zip(environments, questions, seeds)]
    cfg = config["retrieval"]
    ordinary = cfg.get("algorithm", "mot_grpo") == "grpo"
    count = cfg["trees"] * (1 + cfg["expansion_rounds"] * cfg["pivots"]) if ordinary else cfg["trees"]
    forests = [[[Node(0, -1, 0, env.prompt(qa["question"], qa["category"]))] for _ in range(count)]
               for env, qa in zip(environments, questions)]
    rngs = [random.Random(seed) for seed in seeds]
    counters = [[0] * count for _ in questions]

    def extend(jobs):
        while jobs:
            prompts, request_seeds = [], []
            for qi, ti, pivot in jobs:
                parent = forests[qi][ti][pivot]
                prompts.append(next_messages(parent, questions[qi]))
                request_seeds.append((seeds[qi] * 1000003 + ti * 9176 + counters[qi][ti]) % 2147483647)
                counters[qi][ti] += 1
            completions = policy.complete_batch(prompts, request_seeds, max_new_tokens=config["model"]["max_new_tokens"])
            if len(completions) != len(jobs):
                raise ValueError("Rollout backend returned an incorrect batch size")
            following = []
            for (qi, ti, pivot), messages, completion in zip(jobs, prompts, completions):
                nodes = forests[qi][ti]
                node = append_node(nodes, nodes[pivot], messages, completion, environments[qi], policy)
                if not node.terminal and node.depth < cfg["max_depth"]:
                    following.append((qi, ti, node.id))
            jobs = following
    extend([(qi, ti, 0) for qi in range(len(questions)) for ti in range(count)])
    for _ in range(0 if ordinary else cfg["expansion_rounds"]):
        jobs = []
        for qi, forest in enumerate(forests):
            for ti, nodes in enumerate(forest):
                candidates = [n.id for n in nodes if not n.terminal and n.depth < cfg["max_depth"]]
                jobs.extend((qi, ti, pivot) for pivot in rngs[qi].choices(candidates, k=cfg["pivots"]))
        extend(jobs)
    for forest, qa in zip(forests, questions):
        if ordinary:
            leaves = [nodes[-1] for nodes in forest]
            advantages = standardize([answer_f1(n.answer, qa["answer"]) if n.valid and n.terminal else 0.0 for n in leaves])
            for nodes, advantage in zip(forest, advantages):
                for node in nodes[1:]:
                    node.performance = answer_f1(nodes[-1].answer, qa["answer"]) if nodes[-1].valid and nodes[-1].terminal else 0.0
                    node.reward, node.advantage = node.performance, advantage
        else:
            score_forest(forest, qa, config)
    return forests


def training_rows(forest, question_count=1):
    rows = []
    for tree_index, nodes in enumerate(forest):
        weight = 1 / (question_count * len(forest) * (len(nodes) - 1))
        for node in nodes[1:]:
            if not node.prompt_ids or not node.response_ids:
                raise ValueError("A training node must retain the actual sampled token IDs")
            rows.append({"prompt_ids": node.prompt_ids, "response_ids": node.response_ids,
                         "advantage": node.advantage, "weight": weight,
                         "tree": tree_index, "node": node.id})
    return rows


def forest_record(qa, forest, step, seed):
    return {"qa_id": qa["qa_id"], "sample_id": qa["sample_id"], "step": step, "seed": seed,
            "trees": [[asdict(node) for node in nodes] for nodes in forest]}
