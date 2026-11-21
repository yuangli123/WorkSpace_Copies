"""Use the Mem-T prompts and schemas without adding policy-visible role tags."""
from pathlib import Path

from memory.retrieval import MemoryRetriever, get_final_result_format
from memory.settings import SystemConfig


def retriever(db, mode, top_k=5, max_steps=6, benchmark='locomo'):
    settings = SystemConfig()
    settings.data_name = benchmark
    settings.memory.retrieval_topk = top_k
    settings.memory.max_tool_steps = max_steps
    settings.traj_dir = str(Path(__file__).resolve().parents[2] / 'artifacts' / 'tool-logs')
    result = MemoryRetriever(db, None, settings)
    if mode == 'memory_only':
        result.tools = [tool for tool in result.tools if tool.name != 'search_turns']
        result.tool_map = {tool.name: tool for tool in result.tools}
    return result


def answer_rules(category='', benchmark='locomo'):
    return get_final_result_format(benchmark, str(category))


ANSWER_RULES = '\n'.join(answer_rules(category, benchmark)
                         for benchmark, category in [('locomo', ''), ('locomo', '3'),
                                                     ('longmemeval', ''), ('hotpotqa', ''), ('narrativeqa', '')])


def tool_schemas(mode, top_k):
    return retriever(None, mode, top_k).get_tool_schemas()


def retrieval_prompt(question, mode, top_k, max_steps):
    return retriever(None, mode, top_k, max_steps, question.get('benchmark', 'locomo')).construct_prompt(
        question['question'], str(question.get('category', '')))
