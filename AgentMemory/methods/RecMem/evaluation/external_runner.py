"""Run MemoryOS, Mem0 OSS or A-Mem against RecMem's benchmark protocol.

Run each method in its own environment. Only the selected adapter is imported.
"""

import argparse
import hashlib
import json
import os
import sys
import uuid
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from evaluation.protocol import atomic_json, conversation_turns, load_selected, prepare_run, question_text
from evaluation.measurement import Measurement, capture_openai


ROOT = Path(__file__).resolve().parents[2]


def source_version(method, root):
    source = root / {"memoryos": "MemoryOS", "mem0": "mem0", "amem": "A-mem"}[method]
    library = source / {"memoryos": "eval", "mem0": "mem0", "amem": "agentic_memory"}[method]
    digest = hashlib.sha256()
    files = sorted(library.rglob("*.py"))
    if not library.is_dir() or not files:
        raise FileNotFoundError(library)
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    for name in ("external_runner.py", "measurement.py", "protocol.py", "conv_loader.py"):
        path = Path(__file__).with_name(name)
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return f"{method}:source-sha256:{digest.hexdigest()}"


class EmbeddingService:
    def __init__(self, base_url, model, api_key):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed(self, text):
        return self.client.embeddings.create(model=self.model, input=text).data[0].embedding

    def __call__(self, input):
        return [self.embed(text) for text in input]

    def embed_query(self, input):
        return self(input)

    def name(self):
        return self.model


class Answerer:
    def __init__(self, base_url, model, api_key, budget):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.budget = budget

    @staticmethod
    def estimate(text):
        return (len(text.encode("utf-8")) + 2) // 3

    def answer(self, context, question, bench):
        rule = "Answer in fewer than 5-6 words when possible." if bench == "locomo" else "Give a concise, direct answer."
        prefix = (
            "Answer the question using only the timestamped conversation or memories below. "
            "Prefer the latest explicit update when facts conflict. If the answer is absent, say so briefly.\n"
            f"{rule}\nContext:\n"
        )
        suffix = f"\nQuestion: {question}\nAnswer:"
        selected = list(context)
        removed = 0
        removed_tokens = 0

        def prompt():
            return prefix + "\n".join(selected or ["None"]) + suffix

        def shrink():
            nonlocal removed, removed_tokens
            if not selected:
                return False
            if len(selected) > 1:
                removed_tokens += self.estimate(selected.pop())
                removed += 1
            elif len(selected[0]) > 1:
                old = selected[0]
                selected[0] = old[: len(old) // 2]
                removed_tokens += self.estimate(old) - self.estimate(selected[0])
            else:
                removed_tokens += self.estimate(selected.pop())
                removed += 1
            return True

        while self.estimate(prompt()) > self.budget:
            if not shrink():
                raise ValueError("context_budget is smaller than the answer prompt")
        while True:
            try:
                reply = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "system", "content": prompt()}], temperature=0,
                )
                content = reply.choices[0].message.content
                if content is None:
                    raise ValueError("LLM returned an empty answer")
                return content.strip(), {"removed_chunks": removed, "removed_tokens_estimate": removed_tokens}
            except Exception as error:
                message = str(error).lower()
                context_error = any(word in message for word in ("context", "max_model_len", "max seq len")) and any(
                    word in message for word in ("token", "length", "seq len")
                )
                if not context_error or not shrink():
                    raise


class MemoryOSAdapter:
    def __init__(self, source, store, conversation, bench, conv_id, top_k):
        sys.path.insert(0, str(source / "eval"))
        from short_term_memory import ShortTermMemory
        from mid_term_memory import MidTermMemory
        from long_term_memory import LongTermMemory
        from dynamic_update import DynamicUpdate
        from retrieval_and_answer import RetrievalAndAnswer
        from main_loco_parse import update_user_profile_from_top_segment
        from utils import OpenAIClient

        store.mkdir(parents=True, exist_ok=False)
        self.speaker_a = conversation.speaker_a
        self.speaker_b = conversation.speaker_b
        self.conv_id = conv_id
        self.client = OpenAIClient()
        self.short = ShortTermMemory(max_capacity=1, file_path=str(store / "short.json"))
        self.mid = MidTermMemory(max_capacity=2000, file_path=str(store / "mid.json"))
        self.long = LongTermMemory(file_path=str(store / "long.json"))
        updater = DynamicUpdate(self.short, self.mid, self.long, topic_similarity_threshold=0.6, client=self.client)
        self.retriever = RetrievalAndAnswer(self.short, self.mid, self.long, updater, queue_capacity=top_k)
        for timestamp, turn in conversation_turns(conversation, bench):
            user = "\n".join(message.to_string() for message in turn if message.speaker == self.speaker_a)
            assistant = "\n".join(message.to_string() for message in turn if message.speaker != self.speaker_a)
            self.short.add_qa_pair({"user_input": user, "agent_response": assistant, "timestamp": timestamp})
            if self.short.is_full():
                updater.bulk_evict_and_update_mid_term()
            update_user_profile_from_top_segment(self.mid, self.long, conv_id, self.client)

    def context(self, question, top_k):
        found = self.retriever.retrieve(
            question, segment_threshold=0.1, page_threshold=0.1, knowledge_threshold=0.1, client=self.client,
        )
        recent = [
            f"[Message Timestamp]: {item.get('timestamp', '')} [Message]: "
            f"{self.speaker_a}: {item.get('user_input', '')}\n{self.speaker_b}: {item.get('agent_response', '')}"
            for item in self.short.get_all()
        ]
        retrieved = [
            f"[Message Timestamp]: {item.get('timestamp', '')} [Message]: "
            f"{self.speaker_a}: {item.get('user_input', '')}\n{self.speaker_b}: {item.get('agent_response', '')} "
            f"{item.get('meta_info', '')}"
            for item in found["retrieval_queue"][:top_k]
        ]
        profile = self.long.get_user_profile(self.conv_id)
        knowledge = [str(item.get("knowledge", "")) for item in found["long_term_knowledge"]]
        knowledge.extend(str(item.get("knowledge", "")) for item in self.long.get_assistant_knowledge())
        return recent + retrieved + ([f"User profile: {profile.get('data', '')}"] if profile else []) + knowledge


class Mem0Adapter:
    def __init__(self, source, store, conversation, bench, conv_id, model, embed_model, embed_dim, llm_url, embed_url):
        sys.path.insert(0, str(source))
        os.environ["MEM0_TELEMETRY"] = "false"
        os.environ["MEM0_DIR"] = str(store.parent / "runtime")
        from mem0 import Memory

        store.mkdir(parents=True, exist_ok=False)
        self.user_id = conv_id
        self.memory = Memory.from_config({
            "version": "v1.1",
            "llm": {"provider": "openai", "config": {
                "model": model, "api_key": os.getenv("OPENAI_API_KEY", "local"), "openai_base_url": llm_url,
            }},
            "embedder": {"provider": "openai", "config": {
                "model": embed_model, "api_key": os.getenv("EMBEDDING_API_KEY", "local"),
                "openai_base_url": embed_url,
            }},
            "vector_store": {"provider": "qdrant", "config": {
                "collection_name": f"mem0_{conv_id}", "path": str(store / "qdrant"),
                "embedding_model_dims": embed_dim,
            }},
            "history_db_path": str(store / "history.db"),
        })
        for timestamp, turn in conversation_turns(conversation, bench):
            messages = [
                {"role": "user" if message.speaker == conversation.speaker_a else "assistant",
                 "content": f"[{message.speaker}] ({timestamp}): {message.content}" +
                            (f" [Showing Image]: {message.image_caption}" if message.image_caption else "")}
                for message in turn
            ]
            self.memory.add(messages, user_id=self.user_id, metadata={"session_timestamp": timestamp})

    def context(self, question, top_k):
        found = self.memory.search(question, top_k=top_k, filters={"user_id": self.user_id})
        return [str(item.get("memory", "")) for item in found.get("results", [])[:top_k]]


class AMemAdapter:
    def __init__(self, source, conversation, bench, model, embedder):
        sys.path.insert(0, str(source))
        from agentic_memory.memory_system import AgenticMemorySystem

        self.memory = AgenticMemorySystem(
            llm_backend="openai", llm_model=model, api_key=os.getenv("OPENAI_API_KEY", "local"),
            embedding_function=embedder,
        )
        for timestamp, turn in conversation_turns(conversation, bench):
            note_time = datetime.fromisoformat(timestamp[:16].replace("/", "-").replace(" ", "T")).strftime("%Y%m%d%H%M")
            content = f"[Message Timestamp]: {timestamp} [Message]: " + "\n".join(message.to_string() for message in turn)
            self.memory.add_note(content, time=note_time)

    def context(self, question, top_k):
        return [str(item.get("content", "")) for item in self.memory.search_agentic(question, k=top_k)]


def build_adapter(args, source, store_root, conversation, bench, index, embedder, llm_url, embed_url):
    conv_id = f"{bench}_conv_{index}"
    attempt = store_root / f"{conv_id}_{uuid.uuid4().hex[:8]}"
    if args.method == "memoryos":
        return MemoryOSAdapter(source, attempt, conversation, bench, conv_id, args.top_k)
    if args.method == "mem0":
        return Mem0Adapter(source, attempt, conversation, bench, conv_id, args.model,
                           args.embedding_model, args.embedding_dim, llm_url, embed_url)
    return AMemAdapter(source, conversation, bench, args.model, embedder)


def check_personal_workspace(paths):
    personal = os.getenv("PERSONAL_ROOT")
    if not personal:
        return
    workspace = (Path(personal) / "workspace").resolve()
    for path in paths:
        try:
            Path(path).resolve().relative_to(workspace)
        except ValueError as error:
            raise ValueError(f"Path must stay under {workspace}: {path}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["memoryos", "mem0", "amem"])
    parser.add_argument("--bench", required=True, choices=["locomo", "longmemeval_s"])
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--store_root", required=True)
    parser.add_argument("--model", default=os.getenv("MODEL", "qwen3-vl-32b-instruct"))
    parser.add_argument("--embedding_model", default=os.getenv("EMBEDDING_MODEL", "qwen3-embedding-0.6b"))
    parser.add_argument("--embedding_dim", type=int, default=int(os.getenv("EMBEDDING_DIM", "1024")))
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--context_budget", type=int, default=28000)
    parser.add_argument("--conv_limit", type=int)
    parser.add_argument("--sample_size", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--measure", action="store_true", help="Save per-conversation token and timing metrics")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be combined")
    if any(value is not None and value <= 0 for value in
           (args.top_k, args.context_budget, args.embedding_dim, args.conv_limit, args.sample_size)):
        parser.error("counts, dimensions and budget must be positive")
    if not args.output_file.endswith(".json"):
        parser.error("--output_file must end in .json")
    if args.sample_size is not None and args.bench != "longmemeval_s":
        parser.error("--sample_size is LongMemEval-S only")
    llm_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
    embed_url = os.getenv("EMBEDDING_API_BASE") or os.getenv("EMBEDDING_BASE_URL")
    if not llm_url or not embed_url:
        parser.error("set OPENAI_API_BASE and EMBEDDING_API_BASE to the local services")
    os.environ["OPENAI_API_BASE"] = llm_url
    os.environ["OPENAI_BASE_URL"] = llm_url
    os.environ["EMBEDDING_API_BASE"] = embed_url
    os.environ["EMBEDDING_MODEL"] = args.embedding_model
    os.environ["MODEL"] = args.model

    convs, questions, ids = load_selected(args.data_path, args.bench, args.conv_limit, args.sample_size)
    source = ROOT / {"memoryos": "MemoryOS", "mem0": "mem0", "amem": "A-mem"}[args.method]
    if not source.is_dir():
        raise FileNotFoundError(source)
    store_root = Path(args.store_root).resolve()
    check_personal_workspace((source, args.data_path, args.output_file, store_root))
    store_root.mkdir(parents=True, exist_ok=True)
    config = {
        "method": args.method, "bench": args.bench, "model": args.model,
        "embedding_model": args.embedding_model, "embedding_dim": args.embedding_dim,
        "llm_base_url": llm_url, "embedding_base_url": embed_url,
        "top_k": args.top_k, "context_budget": args.context_budget,
        "conv_limit": args.conv_limit, "sample_size": args.sample_size,
        "dataset_path": str(Path(args.data_path).resolve()), "question_ids": ids,
        "source_version": source_version(args.method, ROOT), "store_root": str(store_root),
        "adaptation": "library_adapter" if args.method == "amem" else
                      "longmemeval_adapter" if args.method == "memoryos" and args.bench == "longmemeval_s" else "oss_adapter",
    }
    if args.measure:
        config.update({"measure": True, "measurement_schema": 1,
                       "conv_workers": 1, "question_max_workers": 1})
    prepare_run(args.output_file, args.data_path, config, args.resume, args.overwrite)
    with open(args.output_file, encoding="utf-8") as handle:
        answers = json.load(handle)
    meter = Measurement(args.output_file, answers, resume=args.resume) if args.measure else None
    context_path = args.output_file + ".context.json"
    with open(context_path, encoding="utf-8") as handle:
        stats = json.load(handle)
    embedder = (EmbeddingService(embed_url, args.embedding_model, os.getenv("EMBEDDING_API_KEY", "local"))
                if args.method == "amem" else None)
    answerer = Answerer(llm_url, args.model, os.getenv("OPENAI_API_KEY", "local"), args.context_budget)
    with capture_openai() if meter else nullcontext():
        for index, (conversation, conv_questions) in enumerate(zip(convs, questions)):
            key = str(index)
            if key in answers:
                if len(answers[key]) != len(conv_questions) or key not in stats:
                    raise ValueError(f"Invalid completed conversation: {key}")
                continue
            with meter.phase(index, None, "conversation") if meter else nullcontext():
                with meter.phase(index, None, "construct") if meter else nullcontext():
                    adapter = build_adapter(args, source, store_root, conversation, args.bench,
                                            index, embedder, llm_url, embed_url)
                if meter:
                    meter.set_sizes(index, sum(1 for _ in conversation_turns(conversation, args.bench)),
                                    len(adapter.memory.memories) if args.method == "amem" else None)
                rows, details = [], []
                for position, item in enumerate(conv_questions):
                    question = question_text(item)
                    with meter.phase(index, position, "query") if meter else nullcontext():
                        with meter.phase(index, position, "retrieve") if meter else nullcontext():
                            context = adapter.context(question, args.top_k)
                        with meter.phase(index, position, "answer") if meter else nullcontext():
                            response, info = answerer.answer(context, question, args.bench)
                    rows.append({"question": question, "answer": item.answer, "category": item.category,
                                 "response": response, "answerable": item.answerable})
                    details.append(info)
            if meter:
                meter.save_conversation(index)
            stats[key] = details
            atomic_json(context_path, stats)
            answers[key] = rows
            atomic_json(args.output_file, answers)
            print(f"Completed {index + 1}/{len(convs)} conversations", flush=True)


if __name__ == "__main__":
    main()
