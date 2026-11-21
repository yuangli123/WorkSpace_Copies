from __future__ import annotations

import torch
from memory.protocol import Completion


def render_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def effective_prompt_limit(config, model_context):
    nominal = config["retrieval"]["max_prompt_tokens"]
    actual = min(nominal, model_context - config["model"]["max_new_tokens"]) if model_context else nominal
    if actual <= 0:
        raise ValueError("Generation budget is larger than the model context")
    config["model"]["effective_prompt_tokens"] = actual
    return actual


class HFPolicy:
    def __init__(self, model_path, config, model=None, tokenizer=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.config = config
        cfg = config["model"]
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = getattr(torch, cfg.get("dtype", "bfloat16")) if self.device.type == "cuda" else torch.float32
        self.model = model or AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=False)
        self.model.to(self.device)
        context = getattr(self.model.config, "max_position_embeddings", getattr(self.model.config, "n_positions", None))
        self.prompt_limit = effective_prompt_limit(config, context)
        disable_dropout(self.model)
        self.model.eval()

    def set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def check_observation(self, text):
        count = len(self.tokenizer.encode(text, add_special_tokens=False))
        if count > self.config["retrieval"]["max_observation_tokens"]:
            raise ValueError(f"Tool observation has {count} tokens; increase explicit observation budget")

    def complete(self, messages, max_new_tokens=None, temperature=None, stop_at_tool=True):
        prompt = render_prompt(self.tokenizer, messages)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > self.prompt_limit:
            raise ValueError(f"Prompt has {len(ids)} tokens; exceeds explicit context budget")
        count = max_new_tokens or self.config["model"]["max_new_tokens"]
        context = getattr(self.model.config, "max_position_embeddings", getattr(self.model.config, "n_positions", None))
        if context and len(ids) + count > context:
            raise ValueError("Requested prompt plus generation exceeds model context")
        temperature = self.config["model"]["temperature"] if temperature is None else temperature
        inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
        kwargs = {"do_sample": temperature > 0, "max_new_tokens": count,
                  "pad_token_id": self.tokenizer.pad_token_id,
                  "eos_token_id": self.tokenizer.eos_token_id}
        if stop_at_tool:
            kwargs.update(stop_strings=["</tool_call>"], tokenizer=self.tokenizer)
        if temperature > 0:
            # No top-p/top-k/min-p warping: recomputed behavior log probabilities match sampling.
            kwargs.update(temperature=temperature, top_k=0, top_p=1.0)
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), **kwargs)
        response = generated[0, len(ids):].tolist()
        return Completion(self.tokenizer.decode(response, skip_special_tokens=True), ids, response)

    def get_completion(self, messages, **kwargs):
        return self.complete(messages).text

    def complete_batch(self, messages, seeds, max_new_tokens=None):
        completions = []
        for prompt, seed in zip(messages, seeds):
            self.set_seed(seed)
            completions.append(self.complete(prompt, max_new_tokens=max_new_tokens))
        return completions

    def save_model(self, path):
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)


