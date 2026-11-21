"""Original LoCoMo judge protocol, preserved verbatim (generous topic matching)."""

LOCOMO_SYSTEM_PROMPT = 'Your task is to label an answer to a question as \'CORRECT\' or \'WRONG\'. \nYou will be given the following data: \n(1) a question (posed by one user to another user), \n(2) a \'gold\' (ground truth) answer, \n(3) a generated answer.\n\nThe point of the question is to ask about something one user should know about the other user based on their prior conversations. \nThe gold answer will usually be a concise and short answer that includes the referenced topic.\n\nGrading Rules:\n1. Be generous. As long as the generated answer touches on the same topic as the gold answer, it should be counted as CORRECT.\n2. For time-related questions, if it refers to the same time period as the gold answer (e.g., "last Tuesday" vs "May 7th"), mark it CORRECT.\n3. Ignore minor formatting differences.\n\nOutput Format:\nReturn the label in JSON format with the keys "reasoning" and "label".\nDo NOT output anything else outside the JSON object.\nExample: {"reasoning": "The answer identifies the correct item mentioned in the gold answer.", "label": "CORRECT"}\n'

LOCOMO_USER_PROMPT_TEMPLATE = '\nQuestion: {question} \nGold answer: {gold_answer} \nGenerated answer: {generated_answer} \n\nFirst, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.\n'

# User-supplied Memory-R2 text; keep its generous matching rule explicit.
MEMORY_R2_PROMPT = '''Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given: (1) a question, (2) a gold (ground truth) answer, (3) a generated answer.  The gold answer is usually concise; the generated answer may be longer. Be generous: if the generated answer touches on the same topic/date as the gold, count CORRECT. Different formats for the same date (e.g. "May 7" vs "7 May") are CORRECT.  Question: {question} Gold answer: {gold_answer} Generated answer: {generated_answer}  First give a one-sentence reasoning, then finish with CORRECT or WRONG. Do NOT include both.  Return a JSON object with key "label" whose value is exactly "CORRECT" or "WRONG".'''


def judge_prompts(protocol):
    if protocol == 'legacy-topic-v1':
        return LOCOMO_SYSTEM_PROMPT, LOCOMO_USER_PROMPT_TEMPLATE
    if protocol == 'memory-r2-generous-v1':
        return 'You are an evaluation assistant.', MEMORY_R2_PROMPT
    raise ValueError('Unknown judge protocol: ' + protocol)
