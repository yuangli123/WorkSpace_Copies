from __future__ import annotations

import re
import hashlib
import json

from memory.protocol import parse_calls, validate_call


def operation(sample_id, phase, completion, messages, source_turn_ids, call, index, valid):
    spans = list(re.finditer(r"<tool_call>.*?</tool_call>", completion.text, re.DOTALL))
    start = 0 if index == 0 else spans[index - 1].end()
    end = spans[index].end() if index < len(spans) else len(completion.text)
    identity = json.dumps([sample_id, phase, messages, source_turn_ids, completion.text, index], sort_keys=True, ensure_ascii=False)
    op_id = hashlib.sha256(identity.encode()).hexdigest()[:32]
    completion_id = hashlib.sha256(json.dumps([sample_id, phase, messages, completion.text], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:32]
    if getattr(completion, "protocol_version", "legacy-v1") == "paper-v2":
        # Identical outputs from independently seeded invocations are distinct actions.
        seed = completion.sampling_seed
        if seed is None:
            raise ValueError("paper-v2 action identity requires the invocation seed")
        op_id = hashlib.sha256(f"{op_id}:{seed}".encode()).hexdigest()[:32]
        completion_id = hashlib.sha256(f"{completion_id}:{seed}".encode()).hexdigest()[:32]
    return {"op_id": op_id, "sample_id": sample_id, "phase": phase,
            "category": call["name"] if call else "invalid", "valid": valid,
            "messages": messages, "response": completion.text, "target_span": [start, end],
            "source_turn_ids": list(source_turn_ids), "memory_refs": [], "deleted_refs": [],
            "parent_operation_ids": [], "call": call,
            "completion_id": completion_id,
            "protocol_version": getattr(completion, "protocol_version", "legacy-v1"),
            "sampling_seed": getattr(completion, "sampling_seed", None)}


def construct_episode(episode, policy, db, config, progress=None, checkpoint_fn=None):
    """Reuse official prompts/tools; keep atomic operation spans and all deleted versions."""
    from memory.formation import MemoryFormation, CreateRawTool
    from memory.update import MemoryUpdate

    sid = episode["sample_id"]
    collections = [f"{sid}_{kind}" for kind in ("turns", "facts", "experiences", "personas", "summary")]
    for collection in collections:
        db.create_collection(collection)
    protocol = config['construction'].get('protocol', 'legacy-v1')
    formation, evolution = MemoryFormation(policy, protocol), MemoryUpdate(policy, db)
    summary, personas = "", {}
    summary_sources, persona_sources = set(), {}
    progress = progress or {}
    summary = progress.get('summary', summary)
    personas = progress.get('personas', personas)
    summary_sources = set(progress.get('summary_sources', []))
    persona_sources = {k: set(v) for k, v in progress.get('persona_sources', {}).items()}
    operations = progress.get('operations', [])
    cursor = 0

    def complete(messages, phase, position):
        seed = int(hashlib.sha256(json.dumps([config['seed'], sid, phase, position]).encode()).hexdigest()[:8], 16)
        if protocol == 'paper-v2' and hasattr(policy, 'set_seed'):
            policy.set_seed(seed)
        result = policy.complete(messages, max_new_tokens=config['model']['max_new_tokens'], stop_at_tool=False)
        result.protocol_version = protocol
        result.sampling_seed = seed if protocol == 'paper-v2' else None
        return result

    def seal_completions():
        valid = {}
        for op in operations:
            valid[op['completion_id']] = valid.get(op['completion_id'], True) and op['valid']
        for op in operations:
            op['completion_valid'] = valid[op['completion_id']]
    source_text = {t["turn_id"]: f'{t["speaker"]} at {s["time"]}: {t["text"]}'
                   for s in episode["sessions"] for t in s["turns"]}
    width = config["memory"]["summary_context_turns"]
    for session_index, session in enumerate(episode["sessions"]):
        if not progress or session_index > progress.get('session_index', -1):
            summary, summary_sources = "", set()
        turns = session["turns"]
        for offset in range(0, len(turns), width):
            cursor += 1
            if cursor <= progress.get('cursor', 0):
                continue
            batch = turns[offset:offset + width]
            sources = [t["turn_id"] for t in batch]
            text = "\n".join(f'{t["speaker"]} at {session["time"]}: {t["text"]}' for t in batch)
            if protocol == "legacy-v1":
                db.add(f"{sid}_turns", ids=[f'{session["session_id"]}:{offset}'], documents=[text],
                   metadatas=[{"source_turn_ids": sources, "turn_time": session["time"]}])
            if protocol == 'paper-v2':
                text = "\n".join(f'[{t["turn_id"]}] {t["speaker"]} at {session["time"]}: {t["text"]}' for t in batch)
            messages = formation.construct_prompt(text, summary, "\n".join(f"Name: {k}, Profile: {v}" for k, v in personas.items()))
            completion = complete(messages, "formation", [session_index, offset])
            calls = parse_calls(completion.text)
            if not calls:
                operations.append(operation(sid, "formation", completion, messages, sources, None, 0, False))
            for index, call in enumerate(calls):
                tool = formation.tool_map.get(call["name"])
                valid = tool is not None and validate_call(call, tool)
                if call['name'] == 'create_raw':
                    valid = valid and CreateRawTool.valid_ids(call['arguments'].get('turn_ids'), sources)
                op = operation(sid, "formation", completion, messages, sources, call, index, valid)
                operations.append(op)
                if not valid:
                    continue
                item = formation.execute_tool(call["name"], call["arguments"])
                if not item:
                    op["valid"] = False
                    continue
                if item['type'] == 'raw':
                    chosen = [t['turn_id'] for t in batch if t['turn_id'] in item['turn_ids']]
                    op['source_turn_ids'] = chosen
                    db.begin(op)
                    db.add(f'{sid}_turns', ids=[op['op_id']],
                           documents=['\n'.join(source_text[k] for k in chosen)],
                           metadatas=[{'source_turn_ids': chosen, 'turn_time': session['time'],
                                       'original_turns': [source_text[k] for k in chosen]}])
                    db.end()
                    continue
                if item["type"] in {"summary", "persona"}:
                    if item["type"] == "summary":
                        collection, key = f"{sid}_summary", session["session_id"]
                        summary = item["document"]
                        summary_sources.update(sources)
                        all_sources = sorted(summary_sources)
                    else:
                        collection, key = f"{sid}_personas", item["name"]
                        personas[key] = item["document"]
                        persona_sources.setdefault(key, set()).update(sources)
                        all_sources = sorted(persona_sources[key])
                    db.begin(op)
                    db.upsert(collection, ids=[key], documents=[item["document"]],
                              metadatas=[{"source_turn_ids": all_sources, "memory_content": item["document"],
                                          "original_turns": [source_text[s] for s in all_sources],
                                          "updated_time": f"{session_index:08d}:{offset:08d}",
                                          "turn_time": session["time"], "name": key}])
                    db.end()
                    continue
                kind = "facts" if item["type"] == "fact" else "experiences"
                collection = f"{sid}_{kind}"
                related = db.search(collection, query_texts=[item["document"]],
                                    top_k=config["memory"]["update_retrieval_topk"])
                candidates = [{"id": key, "document": meta.get("memory_content", document),
                               "turn_time": meta.get("turn_time", ""), "start_time": meta.get("start_time", ""),
                               "end_time": meta.get("end_time", "")}
                              for key, document, meta in zip(related["ids"][0], related["documents"][0], related["metadatas"][0])]
                item["turn_time"] = session["time"]
                evo_messages = evolution.construct_prompt(item, candidates, evolution.get_tool_schemas(collection))
                response = complete(evo_messages, "evolution", [session_index, offset, index])
                decisions = parse_calls(response.text)
                if not decisions:
                    operations.append(operation(sid, "evolution", response, evo_messages, sources, None, 0, False))
                tools = {tool.name: tool for tool in evolution._get_tools(collection)}
                for j, decision in enumerate(decisions):
                    evo_tool = tools.get(decision["name"])
                    valid = evo_tool is not None and validate_call(decision, evo_tool)
                    if decision["name"] in {"update_item", "delete_item"}:
                        valid = valid and bool(db.get(collection, ids=[decision["arguments"].get("id", "")])["ids"])
                    evo_op = operation(sid, "evolution", response, evo_messages, sources, decision, j, valid)
                    evo_op["parent_operation_ids"] = [op["op_id"]]
                    operations.append(evo_op)
                    if not valid:
                        continue
                    args = dict(decision["arguments"], source_turn_ids=sources,
                                op_ids={"formation": op["op_id"], "update": evo_op["op_id"]},
                                original_text=text, original_turns=[text])
                    args["_memory_id"] = evo_op["op_id"]
                    db.begin(evo_op)
                    result = evolution.execute_tool(collection, decision["name"], args)
                    db.end()
                    evo_op["valid"] = result.get("status") == "success"
                    op["memory_refs"].extend(evo_op["memory_refs"])
            seal_completions()
            if checkpoint_fn:
                checkpoint_fn({'cursor': cursor, 'session_index': session_index, 'summary': summary,
                               'personas': personas, 'summary_sources': sorted(summary_sources),
                               'persona_sources': {k: sorted(v) for k, v in persona_sources.items()},
                               'operations': operations})
    seal_completions()
    return operations, collections
