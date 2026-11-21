"""Submit one owned Slurm continuation, only after an explicit walltime deferral."""
import argparse
import datetime
import fcntl
import os
from pathlib import Path
import subprocess

from .io import load_config, read_json, write_json


def next_gpu_count(config, output):
    receipt = Path(output) / 'stages/collect-forests.json'
    if receipt.exists() and read_json(receipt).get('status') == 'complete':
        return 4
    count = config.get('study', {}).get('collector_gpus', 4)
    if type(count) is not int or count not in (1, 4):
        raise ValueError('collector_gpus must be 1 or 4')
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--after-job')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root or os.getuid() != 2086:
        raise ValueError('Submit only from your own remote project')
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    output = Path(config['experiment']['output']).resolve()
    if root / 'artifacts' not in output.parents or output.is_symlink():
        raise ValueError('Unexpected formal run location')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'SUBMIT.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'COMPLETE.json').exists():
            raise ValueError('Formal experiment already complete')
        jobs = [read_json(p) for p in (output / 'jobs').glob('*.json')]
        live = set(subprocess.check_output(['squeue', '-h', '-u', str(os.getuid()), '-o', '%A'], text=True).split())
        active = [j['job'] for j in jobs if j['job'] in live and j['job'] != args.after_job]
        if active:
            print('Existing formal job: ' + ','.join(active), flush=True)
            return
        if args.after_job:
            if (args.after_job != os.environ.get('SLURM_JOB_ID')
                    or not any(j['job'] == args.after_job for j in jobs)
                    or read_json(output / 'DEFERRED.json')['job'] != args.after_job):
                raise ValueError('Continuation requires this allocation\'s explicit deferral receipt')
            if any(j.get('after_job') == args.after_job for j in jobs):
                print('Continuation already recorded', flush=True)
                return
        elif live & {j['job'] for j in jobs}:
            raise ValueError('Formal study already has a live allocation')
        if len(jobs) >= config['study']['maximum_jobs']:
            raise ValueError('Continuation job limit reached; inspect before extending')
        command = ['sbatch', '--parsable', '--job-name=memt-c200', '--gres=gpu:' + str(next_gpu_count(config, output))]
        if args.after_job:
            command += ['--dependency=afterok:' + args.after_job]
        command += ['scripts/slurm_train.sh', 'run-construction-study', '--config', str(config_path), '--resume']
        result = subprocess.check_output(command, text=True).strip()
        job = result.split(';')[0]
        if not job.isdecimal():
            raise ValueError('Unexpected sbatch response: ' + result)
        value = {'job': job, 'after_job': args.after_job, 'command': command,
                 'submitted_shanghai': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
        write_json(output / 'jobs' / (job + '.json'), value)
        write_json(output / 'SUBMITTED.json', value)
        print(value, flush=True)


if __name__ == '__main__':
    main()
