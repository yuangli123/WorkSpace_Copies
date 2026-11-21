from __future__ import annotations


import pytest
import torch

from models.hf import HFPolicy, render_prompt
from training.hf_rl import HFUpdater
from training.sft import sft_example, train_sft
from experiments.io import load_config


@pytest.fixture(params=["cpu", "cuda"])
def tiny(tmp_path, request):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(["system user assistant Paris Rome Alice lives in a city <tool_call> </tool_call> " * 20],
                                  trainers.BpeTrainer(vocab_size=300, special_tokens=["<pad>", "<eos>", "<unk>"],
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    fast.chat_template = "{% for m in messages %}{{m['role'] + ': ' + m['content'] + '\n'}}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(fast), n_positions=1024, n_embd=32, n_layer=1, n_head=2,
                                      eos_token_id=fast.eos_token_id, pad_token_id=fast.pad_token_id))
    path = tmp_path / "tiny-model"
    model.save_pretrained(path, safe_serialization=True)
    fast.save_pretrained(path)
    config = load_config("configs/memt4b.yaml")
    config["model"].update(dtype="float32", device=request.param, max_new_tokens=8, gradient_checkpointing=True)
    config["retrieval"].update(evidence_alpha=1.0, expansion_rounds=1, steps=3, warmup_ratio=0, learning_rate=0.001,
                               max_prompt_tokens=512, max_observation_tokens=512)
    config["construction"].update(steps=2, max_length=512, learning_rate=0.001, save_every=1, deepspeed=None,
                                  per_device_batch_size=1, gradient_accumulation=1)
    return path, config


def test_real_rl_update_checkpoint_exact_resume_and_generation(tiny, tmp_path):
    path, config = tiny
    policy = HFPolicy(path, config)
    tokens = policy.tokenizer
    prompt = tokens.encode("user: city?\nassistant: ", add_special_tokens=False)
    paris = tokens.encode("Paris", add_special_tokens=False)
    rome = tokens.encode("Rome", add_special_tokens=False)
    rows = [{"prompt_ids": prompt, "response_ids": paris, "advantage": 1, "weight": 0.5},
            {"prompt_ids": prompt, "response_ids": rome, "advantage": -1, "weight": 0.5}]
    updater = HFUpdater(policy, config)
    before = {k: v.detach().cpu().clone() for k, v in policy.model.state_dict().items()}
    metrics = updater.update(rows)
    assert metrics["grad_norm"] > 0
    assert any(not torch.equal(v, policy.model.state_dict()[k].cpu()) for k, v in before.items())
    assert all(torch.equal(v, updater.reference.state_dict()[k]) for k, v in before.items())
    checkpoint = tmp_path / "checkpoint"
    updater.save(checkpoint, {"step": 1, "question_cursor": 1, "order": [0]})
    updater.update(rows)
    expected = {k: v.detach().cpu().clone() for k, v in policy.model.state_dict().items()}
    resumed = HFPolicy(path, config)
    restored = HFUpdater(resumed, config)
    assert restored.restore(checkpoint)["step"] == 1
    restored.update(rows)
    assert all(torch.allclose(v, resumed.model.state_dict()[k].cpu(), atol=1e-7) for k, v in expected.items())
    completion = resumed.complete([{"role": "user", "content": "city"}], max_new_tokens=4)
    assert completion.prompt_ids and 0 < len(completion.response_ids) <= 4


def test_sft_masks_only_selected_action_and_really_updates(tiny, tmp_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path, config = tiny
    tokenizer = AutoTokenizer.from_pretrained(path)
    text = 'Thought: city. <tool_call>{"name":"finish","arguments":{"answer":"Paris"}}</tool_call>'
    split = text.index("<tool_call>")
    op = {"sample_id": "s", "messages": [{"role": "user", "content": "city?"}],
          "response": text, "target_span": [split, len(text)]}
    row = sft_example(tokenizer, op, 512)
    prompt_length = len(tokenizer.encode(render_prompt(tokenizer, op["messages"]), add_special_tokens=False))
    assert all(x == -100 for x in row["labels"][:prompt_length])
    assert any(x != -100 for x in row["labels"][prompt_length:])
    second = dict(op, messages=[{"role": "user", "content": "which city?"}])
    third = dict(op, messages=[{"role": "user", "content": "Alice city?"}])
    before = AutoModelForCausalLM.from_pretrained(path).state_dict()
    train_sft(config, [op, second, third], path, tmp_path / "sft")
    trained = AutoModelForCausalLM.from_pretrained(tmp_path / "sft/final").state_dict()
    assert any(not torch.equal(value, trained[key]) for key, value in before.items())
