"""Build empty-state clips from TRAIN turns and explicitly source-reviewed QA."""
import argparse
import copy
import re
from pathlib import Path

from experiments.io import read_json, sha256, write_json, write_jsonl
from memory.data import validate_corpus


def normalized(text):
    words = re.sub(r'[^a-z0-9]+', ' ', text.lower()).split()
    numbers = dict(zip(('one','two','three','four','five','six','seven','eight','nine','ten'), map(str,range(1,11))))
    return ' '.join(numbers.get(word,word) for word in words)


def prepare(source, annotations, output):
    source, annotations, output = map(Path, (source, annotations, output))
    bundle = read_json(source)
    original, blocks = bundle['episode'], bundle['blocks']
    if original['sample_id'] != 'conv-26':
        raise ValueError('This reviewed annotation set is only for training conv-26')
    # The source is an exact block decomposition, not a generated memory snapshot.
    expected = [(i, offset, s['turns'][offset:offset+4])
                for i, s in enumerate(original['sessions']) for offset in range(0, len(s['turns']), 4)]
    if expected != [(b['session'], b['offset'], b['turns']) for b in blocks]:
        raise ValueError('Block source does not match original transcript')
    episodes, questions, reviews = [], [], []
    for start in range(0, len(blocks), 8):
        selected = blocks[start:start+8]
        sessions = []
        for block in selected:
            if not sessions or sessions[-1]['session_id'] != block['session_id']:
                sessions.append(dict(session_id=block['session_id'], time=block['time'], turns=[]))
            sessions[-1]['turns'].extend(copy.deepcopy(block['turns']))
        episodes.append(dict(sample_id='%s-short-%02d' % (original['sample_id'], start//8),
                             source_sample_id=original['sample_id'], benchmark=original['benchmark'],
                             sessions=sessions, source_block_range=[start, start+len(selected)]))
    for line in annotations.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        clip, question, answer, refs, category = line.split('|')
        episode = episodes[int(clip)]
        sources = {t['turn_id']: t for s in episode['sessions'] for t in s['turns']}
        evidence = refs.split(',')
        if not evidence or not set(evidence) <= set(sources):
            raise ValueError('QA evidence outside short episode: ' + question)
        quote = ' '.join(sources[ref]['text'] for ref in evidence)
        if normalized(answer) not in normalized(quote):
            raise ValueError('Answer must be a checked phrase in cited evidence: ' + question)
        sid = episode['sample_id']
        qa_id = '%s/qa-%02d' % (sid, sum(q['sample_id'] == sid for q in questions))
        questions.append(dict(sample_id=sid, qa_id=qa_id, benchmark='locomo', question=question,
                              answer=answer, evidence=evidence, category=int(category), reviewed=True,
                              eval_session=len(episode['sessions'])-1,
                              question_time=episode['sessions'][-1]['time'],
                              review={'kind': 'assistant_source_review', 'evidence_quote': quote}))
        reviews.append(dict(qa_id=qa_id, evidence=evidence, quote=quote, answer=answer))
    for episode in episodes:
        qs = [q for q in questions if q['sample_id'] == episode['sample_id']]
        if len(qs) != 8 or len({q['question'] for q in qs}) != 8:
            raise ValueError('Each clip needs eight distinct reviewed QA')
        source_blocks = [{t['turn_id'] for t in b['turns']} for b in blocks[slice(*episode['source_block_range'])]]
        if sum(any(set(q['evidence']) & ids for q in qs) for ids in source_blocks) < 2:
            raise ValueError('QA must cover multiple time points for Local write/retention credit')
    validate_corpus(episodes, questions)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be new; never overwrite reviewed data')
    write_jsonl(output/'episodes.jsonl', episodes)
    write_jsonl(output/'questions.jsonl', questions)
    write_json(output/'partition.json', dict(split='train', source_sample_ids=[original['sample_id']],
                                           source_sha256=sha256(source), initial_state='empty'))
    write_json(output/'DATA_AUDIT.json', dict(version='memt-short-empty-v1', review='assistant_source_review',
        scope='Assistant checked questions and short answers against cited TRAIN-only turns. No independent human/model adjudication.',
        source_sha256=sha256(source), annotations_sha256=sha256(annotations), blocks_max=8,
        episodes=len(episodes), questions=len(questions), source_sample_id=original['sample_id'],
        files={n:sha256(output/n) for n in ('episodes.jsonl','questions.jsonl','partition.json')}, checks=reviews))
    return len(episodes), len(questions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for arg in ('source', 'annotations', 'output'):
        parser.add_argument('--'+arg, required=True)
    args = parser.parse_args()
    print(prepare(args.source, args.annotations, args.output))
