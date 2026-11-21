"""The layout migration must be previewable, auditable and resume-safe."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.compare_results import compare
from evaluation.migrate_layout import migrate, sha
from evaluation.protocol import load_selected, prepare_run as external_prepare, question_text
from tests.test_comparison import load_runner
from tests.test_external_protocol import locomo_sample


class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.runner = load_runner()

    def fixture(self, root):
        root = Path(root) / "AgentMemory"
        old = root / "RecMem"
        (root / "methods/RecMem").mkdir(parents=True)
        data = old / "dataset/Locomo/locomo10.json"
        data.parent.mkdir(parents=True)
        data.write_text(json.dumps(locomo_sample()), encoding="utf-8")
        question = load_selected(str(data), "locomo")[1][0][0]
        row = {"question": question_text(question), "answer": question.answer,
               "category": question.category, "response": "drawer", "answerable": question.answerable}
        scored_row = {**row, "llm_score": 1, "f1_score": 1, "bleu_score": 1}
        runs = {}
        for method in ("recmem", "mem0"):
            answer = old / f"results/comparison/locomo/smoke/{method}.json"
            score = old / f"results/Locomo/scores/{method}.json"
            answer.parent.mkdir(parents=True, exist_ok=True)
            score.parent.mkdir(parents=True, exist_ok=True)
            answer.write_text(json.dumps({"0": [row]}), encoding="utf-8")
            score.write_text(json.dumps({"0": [scored_row]}), encoding="utf-8")
            config = {"method": method, "bench": "locomo", "model": "model",
                      "llm_base_url": "http://llm", "embedding_model": "embed",
                      "embedding_base_url": "http://embed", "conv_limit": 1,
                      "sample_size": None, "dataset_path": str(data),
                      "source_version": f"old-{method}"}
            if method == "recmem":
                config["SEMANTIC_STORE"] = str(root.parent / "RecMem/results/qdrant/semantic")
            if method == "mem0":
                config["store_root"] = str(old / "results/comparison/locomo/smoke/stores/mem0")
            manifest = {**config, "dataset_sha256": sha(data)}
            Path(str(answer) + ".run.json").write_text(json.dumps(manifest), encoding="utf-8")
            Path(str(score) + ".run.json").write_text(json.dumps({
                "bench": "locomo", "judge_model": "judge", "dataset_sha256": sha(data),
                "answer_file": str(answer), "answer_sha256": sha(answer),
            }), encoding="utf-8")
            runs[method] = config
        store = old / "results/comparison/locomo/smoke/stores/mem0/db.txt"
        store.parent.mkdir(parents=True)
        store.write_text("db", encoding="utf-8")
        (old / ".env").write_text("MODEL=local\n", encoding="utf-8")
        return root, old, runs

    def test_preview_apply_resume_and_compare(self):
        with tempfile.TemporaryDirectory() as temp:
            root, old, configs = self.fixture(temp)
            versions = {"recmem": "new-recmem", "mem0": "new-mem0"}
            preview = migrate(old, root, versions=versions)
            self.assertFalse(preview["applied"])
            self.assertTrue((old / "dataset/Locomo/locomo10.json").exists())
            self.assertFalse((root / "artifacts/data/Locomo/locomo10.json").exists())
            self.assertEqual(migrate(old, root, apply=True, versions=versions)["manifests"], 4)
            self.assertEqual((root / ".env").read_text(encoding="utf-8"), "MODEL=local\n")
            self.assertTrue((root / "artifacts/stores/comparison/locomo/smoke/mem0/db.txt").exists())
            data = root / "artifacts/data/Locomo/locomo10.json"
            pairs = []
            for method in configs:
                answer = root / f"artifacts/results/comparison/locomo/smoke/{method}.json"
                score = root / f"artifacts/results/Locomo/scores/{method}.json"
                original = configs[method]
                config = {**original, "dataset_path": str(data), "source_version": versions[method]}
                if method == "recmem":
                    config["SEMANTIC_STORE"] = str(root / "artifacts/stores/recmem/semantic")
                if method == "mem0":
                    config["store_root"] = str(root / "artifacts/stores/comparison/locomo/smoke/mem0")
                prepare = self.runner.prepare_run if method == "recmem" else external_prepare
                prepare(str(answer), str(data), config, resume=True, overwrite=False)
                manifest = json.loads(Path(str(answer) + ".run.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["source_version"], f"old-{method}")
                record = json.loads(Path(str(answer) + ".migration.json").read_text(encoding="utf-8"))
                self.assertEqual(record["completed_before_migration"], ["0"])
                self.assertEqual(record["new_source_version"], versions[method])
                with self.assertRaisesRegex(ValueError, "parameters changed"):
                    prepare(str(answer), str(data), {**config, "model": "wrong"}, resume=True, overwrite=False)
                pairs.append((method, str(answer), str(score)))
            output = root / "artifacts/results/comparison.json"
            self.assertEqual(len(compare("locomo", str(data), pairs, str(output), conv_limit=1)), 4)
            answer = Path(pairs[0][1])
            record = Path(str(answer) + ".migration.json")
            record.unlink()
            config = {**configs["recmem"], "dataset_path": str(data), "source_version": "new-recmem",
                      "SEMANTIC_STORE": str(root / "artifacts/stores/recmem/semantic")}
            with self.assertRaisesRegex(ValueError, "parameters changed"):
                self.runner.prepare_run(str(answer), str(data), config, resume=True, overwrite=False)
            with self.assertRaisesRegex(ValueError, "No legacy artifacts"):
                migrate(old, root, versions=versions)

    def test_preflight_rejects_mismatch_unknown_field_and_target_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            root, old, _ = self.fixture(temp)
            versions = {"recmem": "new-recmem", "mem0": "new-mem0"}
            data = old / "dataset/Locomo/locomo10.json"
            data.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                migrate(old, root, versions=versions)
            data.write_text(json.dumps(locomo_sample()), encoding="utf-8")
            manifest = old / "results/comparison/locomo/smoke/mem0.json.run.json"
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["unexpected"] = True
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown manifest fields"):
                migrate(old, root, versions=versions)
            value.pop("unexpected")
            manifest.write_text(json.dumps(value), encoding="utf-8")
            target = root / "artifacts/data/Locomo/locomo10.json"
            target.parent.mkdir(parents=True)
            target.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                migrate(old, root, versions=versions)
            self.assertTrue(data.exists())


if __name__ == "__main__":
    unittest.main()
