from __future__ import annotations

import os

from models.hf import render_prompt, effective_prompt_limit
from memory.protocol import Completion


class VLLMPolicy:
    """Inference-only local engine for construction, final-policy attribution and evaluation."""
    def __init__(self, model_path, config):
        # CUDA may already be initialized by the CLI seed setup.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from vllm import LLM
        from transformers import AutoConfig
        self.config = config
        model_config = AutoConfig.from_pretrained(model_path)
        self.prompt_limit = effective_prompt_limit(config, model_config.max_position_embeddings)
        self.engine = LLM(model=model_path, dtype=config["model"]["dtype"],
                          tensor_parallel_size=config["distributed"]["inference_tensor_parallel"],
                          gpu_memory_utilization=config["distributed"]["gpu_memory_utilization"],
                          max_model_len=self.prompt_limit + config["model"]["max_new_tokens"],
                          seed=config["seed"], trust_remote_code=False)
        self.tokenizer = self.engine.get_tokenizer()
        self.set_seed(config["seed"])

    def set_seed(self, seed):
        self.next_seed = seed

    def check_observation(self, text):
        if len(self.tokenizer.encode(text, add_special_tokens=False)) > self.config["retrieval"]["max_observation_tokens"]:
            raise ValueError("Observation exceeds configured token budget")

    def complete(self, messages, max_new_tokens=None, temperature=None, stop_at_tool=True):
        from vllm import SamplingParams
        ids = self.tokenizer.encode(render_prompt(self.tokenizer, messages), add_special_tokens=False)
        if len(ids) > self.prompt_limit:
            raise ValueError("Prompt exceeds configured token budget")
        parameters = SamplingParams(seed=self.next_seed, temperature=self.config["model"]["temperature"] if temperature is None else temperature,
                                    max_tokens=max_new_tokens or self.config["model"]["max_new_tokens"], top_p=1.0, top_k=-1,
                                    stop=["</tool_call>"] if stop_at_tool else None, include_stop_str_in_output=True)
        output = self.engine.generate(prompt_token_ids=[ids], sampling_params=parameters, use_tqdm=False)[0].outputs[0]
        self.next_seed += 1
        return Completion(output.text, ids, list(output.token_ids))

    def get_completion(self, messages, **kwargs):
        return self.complete(messages).text

    def complete_batch(self, messages, seeds, max_new_tokens=None):
        from vllm import SamplingParams
        prompts = [self.tokenizer.encode(render_prompt(self.tokenizer, prompt), add_special_tokens=False) for prompt in messages]
        if any(len(ids) > self.prompt_limit for ids in prompts):
            raise ValueError("Prompt exceeds configured token budget")
        parameters = [SamplingParams(seed=seed, temperature=self.config["model"]["temperature"],
                                     max_tokens=max_new_tokens or self.config["model"]["max_new_tokens"],
                                     top_p=1.0, top_k=-1, stop=["</tool_call>"], include_stop_str_in_output=True) for seed in seeds]
        outputs = self.engine.generate(prompt_token_ids=prompts, sampling_params=parameters, use_tqdm=False)
        return [Completion(out.outputs[0].text, prompt, list(out.outputs[0].token_ids)) for prompt, out in zip(prompts, outputs)]
