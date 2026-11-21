"""Pinned vLLM generation for the shared current policy."""
from contextlib import contextmanager
import os
import atexit

from memory.protocol import Completion
from models.hf import render_prompt, effective_prompt_limit


def _reload_policy_weights(worker, weights):
    # vLLM 0.8.5 public collective_rpc passes the WorkerBase as the first argument.
    worker.get_model().load_weights(weights.items())
    return True


def _cleanup_worker(worker):
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
    cleanup_dist_env_and_memory()
    return True


class _EngineWorker:
    """Keep vLLM's CUDA allocator and NCCL out of the HF training process."""
    def __init__(self, path, config, visible):
        os.environ['CUDA_VISIBLE_DEVICES'] = visible
        os.environ['VLLM_USE_V1'] = '0'
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        import torch
        from vllm import LLM
        assert torch.cuda.device_count() == config['distributed']['gpus'] == 2
        self.engine = LLM(model=path, dtype=config['model']['dtype'],
            tensor_parallel_size=config['distributed']['tensor_parallel'],
            distributed_executor_backend='mp', enable_sleep_mode=True, enforce_eager=True,
            disable_custom_all_reduce=True, enable_prefix_caching=False, enable_chunked_prefill=True,
            max_num_seqs=config['distributed']['generation_batch_size'],
            max_num_batched_tokens=config['distributed']['max_batched_tokens'],
            gpu_memory_utilization=config['distributed']['gpu_memory_utilization'],
            max_model_len=config['retrieval']['max_prompt_tokens']+config['model']['max_new_tokens'],
            seed=config['seed'], trust_remote_code=False)

    def wake(self):
        self.engine.wake_up()

    def sleep(self, level):
        self.engine.sleep(level=level)

    def rpc(self, method, args):
        return self.engine.collective_rpc(method, args=args)

    def generate(self, prompts, parameters):
        from vllm import SamplingParams
        outputs = self.engine.generate(prompt_token_ids=prompts,
            sampling_params=[SamplingParams(**p) for p in parameters], use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outputs]

    def close(self):
        import gc
        self.engine.collective_rpc(_cleanup_worker)
        self.engine.llm_engine.model_executor.shutdown()
        self.engine.llm_engine.model_executor = None
        self.engine = None
        gc.collect()
        return True


class _RayEngine:
    def __init__(self, path, config):
        from pathlib import Path
        import ray
        self.ray = ray
        if not ray.is_initialized():
            root = Path(__file__).resolve().parents[1]
            resources = {}
            if not config['distributed'].get('ray_address') and config.get('protocol', {}).get('name') == 'memt-execution-v2' and os.getenv('RAY_TMPDIR'):
                resources['_temp_dir'] = os.environ['RAY_TMPDIR']
            ray.init(address=config['distributed'].get('ray_address'), **resources, runtime_env={'env_vars':{
                'PYTHONPATH':str(root)+':'+str(root/'third_party/Tree-GRPO'), 'VLLM_USE_V1':'0'}})
        # Slurm owns exactly these two GPUs. Generation and HF/FSDP phases are serialized;
        # a second Ray GPU reservation would deadlock partial-role FSDP controls.
        visible = os.environ['CUDA_VISIBLE_DEVICES']
        self.actor = ray.remote(num_cpus=1, num_gpus=0)(_EngineWorker).remote(path, config, visible)

    def wake_up(self):
        self.ray.get(self.actor.wake.remote())

    def sleep(self, level):
        self.ray.get(self.actor.sleep.remote(level))

    def collective_rpc(self, method, args=()):
        return self.ray.get(self.actor.rpc.remote(method, args))

    def generate(self, prompts, parameters):
        return self.ray.get(self.actor.generate.remote(prompts, parameters))

    def close(self):
        actor, self.actor = self.actor, None
        try:
            self.ray.get(actor.close.remote())
        finally:
            self.ray.kill(actor, no_restart=True)


class VLLMPolicy:
    def __init__(self, model_path, config, owner=None, offload=None):
        from transformers import AutoConfig, AutoTokenizer
        self.path, self.config, self.owner, self.offload = str(model_path), config, owner, offload
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.prompt_limit = effective_prompt_limit(config, AutoConfig.from_pretrained(model_path).max_position_embeddings)
        self.engine, self.depth, self.dirty = None, 0, owner is not None
        self.next_seed = config['seed']

    def close(self):
        if self.engine is None:
            return
        engine, self.engine = self.engine, None
        engine.close()

    def invalidate(self):
        self.dirty = True

    def set_seed(self, seed):
        self.next_seed = seed

    @contextmanager
    def phase(self):
        if self.depth:
            yield
            return
        import torch
        self.depth += 1
        try:
            if self.offload:
                self.offload()
            elif self.owner is not None:
                self.owner.model.cpu()
                self.owner.device = torch.device('cpu')
                updater = getattr(self.owner, 'training_updater', None)
                if updater:
                    updater.move_optimizer('cpu')
            torch.cuda.empty_cache()
            if self.engine is None:
                self.engine = _RayEngine(self.path, self.config)
                atexit.register(self.close)
            else:
                self.engine.wake_up()
            if self.dirty:
                weights = {k: v.detach().cpu() for k, v in self.owner.model.state_dict().items()}
                self.engine.collective_rpc(_reload_policy_weights, args=(weights,))
                self.dirty = False
            yield
        finally:
            if self.engine is not None:
                self.engine.sleep(level=1)
            self.depth -= 1

    def complete_batch(self, messages, seeds, max_new_tokens=None, temperature=None, stop_at_tool=True):
        chunk = self.config['distributed']['generation_batch_size']
        if len(messages) > chunk:
            return [x for i in range(0, len(messages), chunk) for x in self.complete_batch(
                messages[i:i+chunk], seeds[i:i+chunk], max_new_tokens, temperature, stop_at_tool)]
        prompts = [self.tokenizer.encode(render_prompt(self.tokenizer, m), add_special_tokens=False) for m in messages]
        if any(len(p) > self.prompt_limit for p in prompts):
            raise ValueError('Prompt exceeds common token budget')
        count = len(prompts)
        padding = (-count) % self.config['distributed']['tensor_parallel']
        prompts += [prompts[0]] * padding
        seeds = list(seeds) + [seeds[0]] * padding
        parameters = [dict(seed=int(s),
            temperature=self.config['model']['temperature'] if temperature is None else temperature,
            max_tokens=max_new_tokens or self.config['model']['max_new_tokens'],
            top_p=1., top_k=-1, min_p=0., stop=['</tool_call>'] if stop_at_tool else [],
            include_stop_str_in_output=True, detokenize=True) for s in seeds]
        with self.phase():
            outputs = self.engine.generate(prompts, parameters)
        return [Completion(self.tokenizer.decode(tokens, skip_special_tokens=True), p, tokens)
                for p, tokens in list(zip(prompts, outputs))[:count]]

    def complete(self, messages, **kwargs):
        result = self.complete_batch([messages], [self.next_seed], **kwargs)[0]
        self.next_seed += 1
        return result


def attach_generation(policy, config, offload=None):
    if getattr(policy, 'generation', None) is None:
        policy.generation = VLLMPolicy(policy.model_path, config, owner=policy, offload=offload)
    return policy.generation
