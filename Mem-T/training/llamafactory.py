"""Export whole selected completions; the native trainer supports partial action masks."""
from pathlib import Path

from experiments.io import checked_output, write_json


def export_selected(selected, model_path, output, config):
    """Never silently supervise an unselected action in a multi-call completion."""
    from memory.protocol import parse_calls
    output = checked_output(output)
    grouped = {}
    for op in selected:
        key = (str(op["messages"]), op["response"])
        grouped.setdefault(key, []).append(op)
    rows, excluded = [], 0
    for (_, response), ops in grouped.items():
        if len(ops) != len(parse_calls(response)):
            excluded += len(ops)
            continue
        messages = ops[0]["messages"]
        rows.append({"system": messages[0]["content"] if messages[0]["role"] == "system" else "",
                     "conversations": [{"from": "human" if m["role"] == "user" else "gpt", "value": m["content"]}
                                       for m in messages if m["role"] != "system"] + [{"from": "gpt", "value": response}]})
    if len(rows) < 2:
        raise ValueError("Insufficient wholly selected completions; use the native action-mask SFT trainer")
    write_json(output / "memt_selected.json", rows)
    write_json(output / "dataset_info.json", {"memt_selected": {"file_name": "memt_selected.json", "formatting": "sharegpt",
               "columns": {"messages": "conversations", "system": "system"},
               "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"}}})
    import yaml
    cfg = config["construction"]
    settings = {"model_name_or_path": str(Path(model_path).resolve()), "stage": "sft", "do_train": True,
                "finetuning_type": "full", "dataset": "memt_selected", "dataset_dir": str(output.resolve()),
                "template": "qwen3", "enable_thinking": False, "cutoff_len": cfg["max_length"], "max_steps": cfg["steps"],
                "per_device_train_batch_size": cfg["per_device_batch_size"],
                "gradient_accumulation_steps": cfg["gradient_accumulation"], "learning_rate": cfg["learning_rate"],
                "lr_scheduler_type": "cosine", "warmup_ratio": cfg["warmup_ratio"], "val_size": cfg["validation_fraction"],
                "bf16": True, "flash_attn": "fa2", "deepspeed": cfg.get("deepspeed"),
                "output_dir": str((output / "model").resolve()), "save_steps": cfg["save_every"], "seed": config["seed"]}
    (output / "train.yaml").write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(output / "export_audit.json", {"whole_completions": len(rows), "excluded_partial_actions": excluded,
                                            "equivalent_to_native_selection": excluded == 0,
                                            "warning": "LLaMA-Factory whole-completion export omits partially selected completions. Use native SFT for exact atomic masks."})
