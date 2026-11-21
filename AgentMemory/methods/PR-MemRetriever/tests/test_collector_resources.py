import copy
import pytest
from experiments.io import load_config, write_json
from experiments.construction_study import collector_config


def test_collector_single_gpu_preserves_method_and_sft_config(tmp_path):
    config = load_config('configs/experiments/construction-study-200-recovery.yaml')
    before = copy.deepcopy(config)
    assert collector_config(config) == config
    config['study']['collector_gpus'] = 1
    selected = collector_config(config)
    assert selected['distributed']['gpus'] == 1
    assert selected['distributed']['inference_tensor_parallel'] == 1
    assert selected['distributed']['tensor_parallel'] == 1
    assert selected['distributed']['gpu_memory_utilization'] == .8
    assert selected['retrieval'] == config['retrieval']
    assert selected['construction'] == config['construction']
    assert config['distributed'] == before['distributed']
    assert config['construction']['global_batch_size'] == 32
    assert selected['models'] == config['models']


def test_dispatch_selects_one_gpu_then_four(tmp_path):
    pytest.importorskip('fcntl')
    from experiments.study_dispatch import next_gpu_count
    config = load_config('configs/experiments/construction-study-200-recovery.yaml')
    config['study']['collector_gpus'] = 1
    assert next_gpu_count(config, tmp_path) == 1
    write_json(tmp_path / 'stages/collect-forests.json', {'status': 'running'})
    assert next_gpu_count(config, tmp_path) == 1
    write_json(tmp_path / 'stages/collect-forests.json', {'status': 'complete'})
    assert next_gpu_count(config, tmp_path) == 4


def test_collector_rejects_unsupported_gpu_counts(tmp_path):
    config = load_config('configs/experiments/construction-study-200-recovery.yaml')
    config['study']['collector_gpus'] = 2
    with pytest.raises(ValueError, match='collector_gpus'):
        collector_config(config)
