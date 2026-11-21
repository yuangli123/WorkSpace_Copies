"""Small, dependency-free pieces shared by the original and external runners."""

import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict

from evaluation.conv_loader import ConvLoader


def sample_lme_by_question_type(convs, questions, sample_size, seed=42):
    groups = defaultdict(list)
    for index, group in enumerate(questions):
        if group:
            groups[group[0].category].append(index)
    rng = random.Random(seed)
    indices = []
    for category in sorted(groups):
        group = groups[category]
        indices.extend(group if len(group) <= sample_size else rng.sample(group, sample_size))
    indices.sort()
    return [convs[i] for i in indices], [questions[i] for i in indices]


def load_selected(data_path, bench, conv_limit=None, sample_size=None):
    """Return conversations, questions and stable source IDs in matching order."""
    with open(data_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if bench == "locomo":
        convs, questions = ConvLoader.load_locomo(data_path)
        ids = [
            [f"{item['sample_id']}:{i}" for i, qa in enumerate(item["qa"]) if qa["category"] != 5]
            for item in raw
        ]
        if conv_limit is not None:
            return convs[:conv_limit], questions[:conv_limit], ids[:conv_limit]
        return convs, questions, ids
    if bench != "longmemeval_s":
        raise ValueError(f"Invalid benchmark name: {bench}")
    convs, questions = ConvLoader.load_longmemeval_s(data_path)
    ids = [[item["question_id"]] for item in raw]
    if sample_size is not None:
        selected, selected_questions = sample_lme_by_question_type(convs, questions, sample_size)
        lookup = {id(group): index for index, group in enumerate(questions)}
        selected_ids = [ids[lookup[id(group)]] for group in selected_questions]
        return selected, selected_questions, selected_ids
    if conv_limit is not None:
        return convs[:conv_limit], questions[:conv_limit], ids[:conv_limit]
    return convs, questions, ids


def conversation_turns(conversation, bench):
    """Yield the same turn groups and timestamps used by RecMem."""
    for session in conversation.sessions:
        timestamp = (
            ConvLoader.locomo_time_to_datetime(session.date_time).isoformat()
            if bench == "locomo" else session.date_time
        )
        messages = session.messages
        index = 0
        while index < len(messages):
            turn = [messages[index]]
            if index + 1 < len(messages) and (
                bench == "locomo" or
                (messages[index].speaker == "user" and messages[index + 1].speaker == "assistant")
            ):
                turn.append(messages[index + 1])
            yield timestamp, turn
            index += len(turn)


def question_text(question):
    return question.content + (f"\nThe date of the question is {question.date}" if question.date else "")


def atomic_json(path, value):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as handle:
            temp_path = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=4)
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def resume_matches(existing, expected, output_path):
    if existing == expected:
        return True
    record_path = output_path + ".migration.json"
    if not os.path.isfile(record_path):
        return False
    with open(record_path, encoding="utf-8") as handle:
        record = json.load(handle)
    if set(existing) != set(expected) or any(
        existing[key] != expected[key] for key in expected if key != "source_version"
    ):
        return False
    manifest_path = output_path + ".run.json"
    with open(manifest_path, "rb") as handle:
        manifest_sha = hashlib.sha256(handle.read()).hexdigest()
    backup = record.get("backup_manifest")
    if not backup or not os.path.isfile(backup):
        return False
    with open(backup, "rb") as handle:
        backup_sha = hashlib.sha256(handle.read()).hexdigest()
    return (
        record.get("answer_file") == os.path.realpath(output_path)
        and record.get("migrated_manifest_sha256") == manifest_sha
        and record.get("original_manifest_sha256") == backup_sha
        and record.get("old_source_version") == existing["source_version"]
        and record.get("new_source_version") == expected["source_version"]
        and record.get("dataset_sha256") == expected["dataset_sha256"]
    )


def prepare_run(output_path, data_path, config, resume=False, overwrite=False):
    if os.path.realpath(output_path) == os.path.realpath(data_path):
        raise ValueError("--output_file must not overwrite the dataset")
    digest = hashlib.sha256()
    with open(data_path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    manifest = {**config, "dataset_sha256": digest.hexdigest()}
    manifest_path = output_path + ".run.json"
    answer_exists = os.path.exists(output_path)
    manifest_exists = os.path.exists(manifest_path)
    if resume and answer_exists != manifest_exists:
        raise ValueError("Cannot resume: answer file and run manifest must both exist")
    if resume and answer_exists:
        with open(manifest_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if not resume_matches(existing, manifest, output_path):
            raise ValueError("Cannot resume: dataset or run parameters changed")
        return manifest
    if not overwrite and (answer_exists or manifest_exists):
        raise FileExistsError("Output already exists; use --resume or --overwrite")
    atomic_json(manifest_path, manifest)
    atomic_json(output_path, {})
    if config["method"] != "recmem":
        atomic_json(output_path + ".context.json", {})
    return manifest
