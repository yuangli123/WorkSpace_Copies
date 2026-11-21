"""Sealed chunk progress; a partial tool execution is never resumed as committed."""
from pathlib import Path

from .io import read_json, sha256, write_json
from memory.journal import JournalDB


def save_chunk(folder, state, journal, collections):
    folder = Path(folder)
    snapshot = {'versions': journal.versions, 'active': [[*key, ref] for key, ref in journal.active.items()],
                'collections': {name: journal.db.get(name) for name in collections}}
    path = folder / f'chunk-{state["cursor"]:08d}.json'
    previous = read_json(folder / 'CURRENT.json') if (folder / 'CURRENT.json').exists() else None
    write_json(path, {'state': state, 'snapshot': snapshot})
    write_json(folder / 'CURRENT.json', {'file': path.name, 'sha256': sha256(path)})
    # Only retire this run's preceding, sealed progress file after the new pointer
    # is durable. Final snapshots and operation logs are kept separately.
    if previous and previous['file'] != path.name:
        old = folder / previous['file']
        if old.is_file() and old.parent.resolve() == folder.resolve() and not old.is_symlink():
            old.unlink()


def read_chunk(folder):
    folder = Path(folder)
    pointer = read_json(folder / 'CURRENT.json')
    path = folder / pointer['file']
    if path.parent.resolve() != folder.resolve() or sha256(path) != pointer['sha256']:
        raise ValueError('Invalid construction progress seal')
    value = read_json(path)
    return value


def restore_chunk(folder, database):
    value = read_chunk(folder)
    journal = JournalDB(database)
    journal.versions = value['snapshot']['versions']
    journal.active = {(name, key): ref for name, key, ref in value['snapshot']['active']}
    for name, rows in value['snapshot']['collections'].items():
        database.create_collection(name)
        if rows['ids']:
            database.upsert(name, ids=rows['ids'], documents=rows['documents'], metadatas=rows['metadatas'])
    return value['state'], journal
