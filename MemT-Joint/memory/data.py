"""Data boundaries: policy-facing episodes and scorer-only QA annotations."""
import re

from experiments.io import checked_output, read_json, sha256, write_json, write_jsonl


def locomo(path):
    raw = read_json(path)
    episodes, questions = [], []
    seen = set()
    for item in raw:
        sample_id = str(item["sample_id"])
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        conversation = item["conversation"]
        if not isinstance(conversation, dict):
            raise ValueError("Expected original LoCoMo conversation mapping")
        sessions, turn_ids = [], set()
        for key in sorted((k for k in conversation if re.fullmatch(r"session_\d+", k)),
                          key=lambda k: int(k.split("_")[1])):
            turns = []
            for turn in conversation[key]:
                turn_id = turn["dia_id"]
                if turn_id in turn_ids:
                    raise ValueError(f"Duplicate turn: {sample_id}/{turn_id}")
                turn_ids.add(turn_id)
                turns.append({"turn_id": turn_id, "speaker": turn["speaker"], "text": turn.get("text", "")})
            sessions.append({"session_id": key, "time": conversation.get(key + "_date_time", ""), "turns": turns})
        episodes.append({"sample_id": sample_id, "benchmark": "locomo", "sessions": sessions})
        for index, qa in enumerate(item["qa"]):
            if int(qa["category"]) == 5:
                continue
            raw_evidence = list(qa.get("evidence") or [])
            evidence = list(dict.fromkeys(ref for field in raw_evidence
                                          for ref in re.split(r"[;,\s]+", field.strip()) if ref))
            missing = sorted(set(evidence) - turn_ids)
            # Keep every answer-evaluation question. Incomplete gold evidence must not
            # become a partially correct reward target; preserve it for an audit.
            if missing:
                evidence = []
            questions.append({"sample_id": sample_id, "qa_id": f"{sample_id}:{index}",
                              "question": qa["question"], "answer": qa["answer"],
                              "evidence": evidence, "category": qa["category"], "benchmark": "locomo",
                              "raw_evidence": raw_evidence, "invalid_evidence": missing})
    return episodes, questions


def prepare_locomo(source, output):
    output = checked_output(output)
    episodes, questions = locomo(source)
    if len(episodes) != 10:
        raise ValueError("Paper split requires ten conversations; use a separate fixture directory for tests")
    write_json(output / "evidence_audit.json", [
        {"qa_id": q["qa_id"], "raw": q["raw_evidence"], "normalized": q["evidence"],
         "invalid": q["invalid_evidence"]} for q in questions
        if q["raw_evidence"] != q["evidence"] or q["invalid_evidence"]])
    # The released main.py uses this order, not a random QA-level split.
    split = {"train": [e["sample_id"] for e in episodes[:1]],
             "valid": [e["sample_id"] for e in episodes[1:2]],
             "test": [e["sample_id"] for e in episodes[2:]]}
    for name, ids in split.items():
        write_jsonl(output / name / "episodes.jsonl", (e for e in episodes if e["sample_id"] in ids))
        write_jsonl(output / name / "questions.jsonl", (q for q in questions if q["sample_id"] in ids))
        write_json(output / name / "partition.json", {"split": name, "source_sha256": sha256(source), "sample_ids": ids})
    write_json(output / "split.json", {"source_sha256": sha256(source), "unit": "conversation",
                                      "policy": "official-main-order-1-1-8", "ids": split,
                                      "counts": {s: sum(q["sample_id"] in ids for q in questions) for s, ids in split.items()}})


def validate_corpus(episodes, questions):
    episode_map = {e["sample_id"]: e for e in episodes}
    if len(episode_map) != len(episodes):
        raise ValueError("Duplicate episodes")
    ids = set()
    for qa in questions:
        if qa["qa_id"] in ids or qa["sample_id"] not in episode_map:
            raise ValueError("Duplicate question or missing episode")
        ids.add(qa["qa_id"])
        turns = {t["turn_id"] for s in episode_map[qa["sample_id"]]["sessions"] for t in s["turns"]}
        if not set(qa.get("evidence", [])) <= turns:
            raise ValueError("Evidence outside episode")
    return episode_map


def prepare_ood(source, output, benchmark):
    """OOD is evaluation-only. Preserve all supplied examples; never silently take 72."""
    output = checked_output(output)
    if benchmark == "hotpotqa":
        from dataset import load_hotpotqa_dataset
        raw = load_hotpotqa_dataset(str(source))
        episodes, questions = [], []
        for row in raw:
            sid = row["qa"][0]["sample_id"]
            episodes.append({"sample_id": sid, "benchmark": benchmark, "sessions": [
                {"session_id": s["session_id"], "time": "", "turns": s["session_turns"]} for s in row["conversation"]]})
            questions.extend(dict(q, qa_id=f"{sid}:0", benchmark=benchmark) for q in row["qa"])
    elif benchmark == "longmemeval":
        episodes, questions = [], []
        for item in read_json(source):
            sid = str(item["question_id"])
            sessions, evidence = [], []
            answer_sessions = set(item.get("answer_session_ids", []))
            for i, (session_id, date, turns) in enumerate(zip(item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"])):
                formatted = [{"turn_id": f"{session_id}:{j}", "speaker": t["role"], "text": t["content"]} for j, t in enumerate(turns)]
                sessions.append({"session_id": session_id, "time": date, "turns": formatted})
                if session_id in answer_sessions:
                    evidence.extend(t["turn_id"] for t in formatted)
            episodes.append({"sample_id": sid, "benchmark": benchmark, "sessions": sessions})
            questions.append({"sample_id": sid, "qa_id": sid, "question": item["question"], "answer": item["answer"],
                              "question_time": item.get("question_date", ""),
                              "evidence": evidence, "category": item["question_type"], "benchmark": benchmark})
    elif benchmark == "narrativeqa":
        import random
        import pyarrow.parquet as pq
        raw = pq.read_table(source).to_pylist()
        documents = {}
        for item in raw:
            documents.setdefault(str(item["document"]["id"]), []).append(item)
        if len(documents) < 10:
            raise ValueError("NarrativeQA paper evaluation requires ten documents")
        # The authors' exact ten-document selection is unavailable: persist this new selection.
        ids = random.Random(42).sample(sorted(documents), 10)
        episodes, questions = [], []
        for sid in ids:
            text = documents[sid][0]["document"]["text"]
            # Explicit 512-word blocks; not claimed to be the unpublished original preprocessing.
            words = text.split()
            sessions = [{"session_id": f"block-{i}", "time": "", "turns": [
                {"turn_id": f"block-{i}", "speaker": "document", "text": " ".join(words[i:i + 512])}]} for i in range(0, len(words), 512)]
            episodes.append({"sample_id": sid, "benchmark": benchmark, "sessions": sessions})
            for index, item in enumerate(documents[sid]):
                question = item["question"]
                answers = item["answers"]
                questions.append({"sample_id": sid, "qa_id": f"{sid}:{index}",
                                  "question": question["text"] if isinstance(question, dict) else question,
                                  "answer": [a["text"] for a in answers], "evidence": [],
                                  "category": benchmark, "benchmark": benchmark})
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    validate_corpus(episodes, questions)
    write_jsonl(output / "episodes.jsonl", episodes)
    write_jsonl(output / "questions.jsonl", questions)
    write_json(output / "partition.json", {"split": "ood_test", "source_sha256": sha256(source),
                                           "benchmark": benchmark, "sample_ids": [e["sample_id"] for e in episodes]})
