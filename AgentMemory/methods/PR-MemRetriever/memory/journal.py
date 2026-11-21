"""Versioned provenance around the released Mem-T tool interface."""
from __future__ import annotations

import copy
import re
from pathlib import Path

from experiments.io import read_json, write_json
from memory.protocol import flatten


class ListDB:
    """Small deterministic lexical backend for tests, never a BGE-M3 replacement in results."""
    def __init__(self):
        self.collections = {}

    def create_collection(self, name, **kwargs):
        self.collections.setdefault(name, {})

    def delete_collection(self, name):
        self.collections.pop(name, None)

    def add(self, name, ids, documents, metadatas=None, **kwargs):
        self.create_collection(name)
        for i, key in enumerate(ids):
            if key in self.collections[name]:
                raise ValueError(f"Duplicate memory {name}/{key}")
            self.collections[name][key] = (documents[i], copy.deepcopy((metadatas or [{}] * len(ids))[i]))

    def upsert(self, name, ids, documents, metadatas=None, **kwargs):
        self.create_collection(name)
        for key in ids:
            self.collections[name].pop(key, None)
        self.add(name, ids, documents, metadatas)

    def get(self, name, ids=None, **kwargs):
        collection = self.collections.get(name, {})
        keys = [key for key in (ids if ids is not None else collection) if key in collection]
        return {"ids": keys, "documents": [collection[k][0] for k in keys],
                "metadatas": [copy.deepcopy(collection[k][1]) for k in keys]}

    def search(self, name, query_texts, top_k=5, **kwargs):
        result = {"ids": [], "documents": [], "metadatas": []}
        for query in query_texts:
            tokens = set(re.findall(r"\w+", query.lower()))
            keys = sorted(self.collections.get(name, {}), key=lambda k: (
                -len(tokens & set(re.findall(r"\w+", self.collections[name][k][0].lower()))), k))[:top_k]
            data = self.get(name, keys)
            for key in result:
                result[key].append(data[key])
        return result

    def delete(self, name, ids, **kwargs):
        for key in ids:
            self.collections.get(name, {}).pop(key, None)


class JournalDB:
    def __init__(self, db):
        self.db = db
        self.versions = {}
        self.active = {}
        self.operations = {}
        self.context = None

    def __getattr__(self, name):
        return getattr(self.db, name)

    def begin(self, operation):
        self.context = operation
        self.operations[operation["op_id"]] = operation

    def end(self):
        self.context = None

    def _write(self, name, ids, documents, metadatas, update):
        new_meta = []
        for key, document, metadata in zip(ids, documents, metadatas or [{} for _ in ids]):
            old_ref = self.active.get((name, key))
            old = self.versions.get(old_ref, {})
            version = old.get("version", 0) + 1
            ref = f"{name}/{key}@{version}"
            meta = copy.deepcopy(metadata)
            provenance = set(old.get("operation_ids", []))
            if self.context:
                provenance.add(self.context["op_id"])
                provenance.update(self.context.get("parent_operation_ids", []))
                self.context["memory_refs"].append(ref)
            meta.update({"memory_ref": ref, "memory_version": version,
                         "operation_ids": sorted(provenance)})
            if old_ref:
                meta["source_turn_ids"] = sorted(set(flatten(old.get("source_turn_ids", []))) |
                                                 set(flatten(meta.get("source_turn_ids", []))))
            self.versions[ref] = {"collection": name, "id": key, "version": version,
                                  "document": document,
                                  "operation_ids": sorted(provenance),
                                  "source_turn_ids": flatten(meta.get("source_turn_ids", []))}
            self.active[(name, key)] = ref
            new_meta.append(meta)
        method = self.db.upsert if update else self.db.add
        return method(name, ids=ids, documents=documents, metadatas=new_meta)

    def add(self, name, ids, documents, metadatas=None, **kwargs):
        return self._write(name, ids, documents, metadatas, False)

    def upsert(self, name, ids, documents, metadatas=None, **kwargs):
        return self._write(name, ids, documents, metadatas, True)

    def delete(self, name, ids, **kwargs):
        for key in ids:
            ref = self.active.pop((name, key), None)
            if self.context and ref:
                self.context["deleted_refs"].append(ref)
        return self.db.delete(name, ids=ids, **kwargs)

    def save(self, path, collections):
        write_json(path, {"versions": self.versions, "active": [[*key, ref] for key, ref in self.active.items()],
                          "collections": {name: self.db.get(name) for name in collections}})

    @classmethod
    def restore(cls, path, db):
        snapshot = read_json(path)
        journal = cls(db)
        journal.versions = snapshot["versions"]
        journal.active = {(name, key): ref for name, key, ref in snapshot["active"]}
        for name, rows in snapshot["collections"].items():
            # Chroma's default 1000-record threshold leaves small restored segments
            # without an on-disk ID map, which is required when the cache reloads them.
            configuration = {"hnsw": {"sync_threshold": 1}} if rows["ids"] else None
            db.create_collection(name, configuration=configuration)
            if rows["ids"]:
                db.upsert(name, ids=rows["ids"], documents=rows["documents"], metadatas=rows["metadatas"])
        return journal


def open_db(config, path):
    backend = config["memory"]["backend"]
    if backend == "lexical_test":
        return ListDB()
    if backend != "chroma":
        raise ValueError(f"Unknown memory backend: {backend}")
    if not Path(config["memory"]["embedding_model"]).is_dir():
        raise ValueError("Use an existing local embedding model directory under your personal models/")
    from memory.settings import VectorDBConfig
    from memory.vector_db import ChromaVectorDB
    settings = VectorDBConfig()
    settings.db_type = "persistent"
    settings.path = str(Path(path).resolve())
    settings.embedding_model = config["memory"]["embedding_model"]
    settings.hnsw_num_threads = config['memory'].get('hnsw_num_threads')
    if settings.hnsw_num_threads is not None and (type(settings.hnsw_num_threads) is not int or settings.hnsw_num_threads <= 0):
        raise ValueError('hnsw_num_threads must be a positive integer')
    return ChromaVectorDB(settings)
