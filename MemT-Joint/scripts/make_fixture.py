"""Create clearly labelled, synthetic partitions for CPU environment checks."""
import argparse
from pathlib import Path

from experiments.io import checked_output, write_json, write_jsonl


def create(output):
    output = checked_output(output)
    for split, cities in [('train', ['Beijing', 'Shanghai']), ('valid', ['Paris', 'Lyon']), ('test', ['Osaka', 'Tokyo'])]:
        sid = 'fixture-' + split
        episode = {'sample_id': sid, 'benchmark': 'fixture', 'sessions': [
            {'session_id': 'session_' + str(i), 'time': '2026-10-0' + str(i + 1),
             'turns': [{'turn_id': 's%d:u1' % i, 'speaker': 'User', 'text': 'residence=' + city}]}
            for i, city in enumerate(cities)]}
        questions = [{'sample_id': sid, 'qa_id': sid + '/q' + str(i), 'question': 'What is my current residence?',
                      'answer': city, 'evidence': ['s%d:u1' % i], 'category': 1, 'eval_session': i,
                      'question_time': '2026-10-0' + str(i + 1), 'reviewed': True}
                     for i, city in enumerate(cities)]
        folder = output / split
        write_jsonl(folder / 'episodes.jsonl', [episode])
        write_jsonl(folder / 'questions.jsonl', questions)
        write_json(folder / 'partition.json', {'split': split, 'fixture_only': True, 'sample_ids': [sid]})
    write_json(output / 'FIXTURE_ONLY.json', {'real_benchmark': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    create(parser.parse_args().output)
