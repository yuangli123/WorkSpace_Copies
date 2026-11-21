"""Paper-v2 adds a learned raw-memory action; legacy-v1 uses released tools."""
from memory.formation_tools import MemoryFormation as ReleasedFormation
from memory.tools_base import BaseTool


class CreateRawTool(BaseTool):
    def __init__(self):
        super().__init__(name='create_raw',
                         description='Archive important original turns verbatim. Choose only turn_ids shown in the current input block.',
                         parameters={'turn_ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'uniqueItems': True}},
                         required=['turn_ids'])

    @staticmethod
    def valid_ids(ids, available):
        return (isinstance(ids, list) and bool(ids) and all(isinstance(x, str) for x in ids)
                and len(set(ids)) == len(ids) and set(ids) <= set(available))

    def __call__(self, turn_ids, **kwargs):
        return {'type': 'raw', 'turn_ids': turn_ids}


class MemoryFormation(ReleasedFormation):
    def __init__(self, executor, protocol='legacy-v1'):
        super().__init__(executor)
        self.protocol = protocol
        if protocol == 'paper-v2':
            self.tools.append(CreateRawTool())
            self.tool_map = {tool.name: tool for tool in self.tools}

    def construct_prompt(self, turn_text, prev_summary, prev_personas):
        messages = super().construct_prompt(turn_text, prev_summary, prev_personas)
        if self.protocol == 'paper-v2':
            messages[0]['content'] += ('\n[Raw Memory]\nUse create_raw to retain useful original information, especially details whose wording '
                                       'may matter for future queries. Select exact turn_ids from the current block. '
                                       'Raw text is copied by the tool; never invent IDs. Raw archiving is not automatic.\n')
        return messages
