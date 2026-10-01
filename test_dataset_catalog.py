import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from dataset_catalog import DatasetCatalog

ROOT = Path(__file__).resolve().parent


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "catalog.json"
        self.catalog = DatasetCatalog(self.path)
        self.raw = {"id": "orders", "fields": [{"name": "amount", "type": "number"}]}

    def test_register_persists_schema_and_direct_dependencies(self):
        self.catalog.register(self.raw)
        self.catalog.register({"id": "daily", "fields": [{"name": "total", "type": "number"}], "depends_on": ["orders"]})
        self.catalog.register({"id": "weekly", "fields": [{"name": "total", "type": "number"}], "depends_on": ["daily"]})
        fresh = DatasetCatalog(self.path)
        self.assertEqual([row["id"] for row in fresh.dependencies("weekly")], ["daily"])
        self.assertEqual(fresh.describe("orders")["fields"], self.raw["fields"])

    def test_bad_dependencies_and_duplicate_fields_do_not_write(self):
        for item in ({**self.raw, "depends_on": ["unknown"]}, {**self.raw, "fields": self.raw["fields"] * 2}):
            with self.assertRaises(ValueError):
                self.catalog.register(item)
        self.assertFalse(self.path.exists())

    def test_duplicate_and_unknown_dataset(self):
        self.catalog.register(self.raw)
        with self.assertRaises(ValueError):
            self.catalog.register(self.raw)
        with self.assertRaises(ValueError):
            self.catalog.dependencies("missing")

    def test_cli_query_and_registration(self):
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", str(self.path)]
        run = subprocess.run(prefix + ["register", str(ROOT / "samples/orders.json")], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        query = subprocess.run(prefix + ["describe", "orders"], capture_output=True, text=True)
        self.assertEqual(json.loads(query.stdout)["id"], "orders")
        self.assertEqual(subprocess.run(prefix + ["describe", "missing"], capture_output=True).returncode, 2)

    def test_impact_follows_downstream_with_shortest_path(self):
        self.catalog.register(self.raw)
        self.catalog.register({"id": "middle", "fields": [{"name": "total", "type": "number"}], "depends_on": ["orders"]})
        self.catalog.register({"id": "zed", "fields": [{"name": "total", "type": "number"}], "depends_on": ["orders"]})
        self.catalog.register({"id": "target", "fields": [{"name": "total", "type": "number"}], "depends_on": ["zed", "middle"]})
        self.catalog.register({"id": "leaf", "fields": [{"name": "total", "type": "number"}], "depends_on": ["target"]})
        self.catalog.register({"id": "isolated", "fields": [{"name": "total", "type": "number"}]})
        impact = self.catalog.impact("orders")
        self.assertEqual([item["dataset"]["id"] for item in impact], ["middle", "zed", "target", "leaf"])
        self.assertEqual(impact[0]["distance"], 1)
        self.assertEqual(impact[0]["path"], ["orders", "middle"])
        self.assertEqual(impact[2]["path"], ["orders", "middle", "target"])
        self.assertEqual(impact[3]["path"], ["orders", "middle", "target", "leaf"])
        self.assertEqual(impact[0]["dataset"], self.catalog.describe("middle"))
        self.assertEqual([item["dataset"]["id"] for item in self.catalog.impact("orders", max_depth=1)], ["middle", "zed"])
        self.assertEqual(self.catalog.impact("isolated"), [])

    def test_impact_rejects_invalid_max_depth(self):
        self.catalog.register(self.raw)
        for value in (True, 0, -1, 1.0, "1"):
            with self.assertRaises(ValueError):
                self.catalog.impact("orders", value)
        with self.assertRaises(ValueError):
            self.catalog.impact("missing", 0)
        with self.assertRaisesRegex(ValueError, "max_depth"):
            self.catalog.impact("missing", 0)

    def test_cli_impact(self):
        sample = ["--catalog", str(ROOT / "samples/catalog.json")]
        run = lambda *extra: subprocess.run(
            [sys.executable, str(ROOT / "dataset_catalog.py")] + sample + ["impact", *extra],
            capture_output=True, text=True)
        ok = run("orders")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        payload = json.loads(ok.stdout)
        self.assertEqual(payload, [{
            "dataset": DatasetCatalog(ROOT / "samples/catalog.json").describe("daily_totals"),
            "distance": 1, "path": ["orders", "daily_totals"]}])
        self.assertEqual(json.loads(run("daily_totals").stdout), [])
        self.assertEqual(run("orders", "--max-depth", "1").returncode, 0)
        for bad in ("0", "-1", "1.0", "ab", " 1", "1 "):
            failed = run("orders", "--max-depth", bad)
            self.assertEqual(failed.returncode, 2, bad)
            self.assertIn("error", json.loads(failed.stdout))
        missing = run("missing")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("error", json.loads(missing.stdout))
        depth_first = run("missing", "--max-depth", "0")
        self.assertEqual(depth_first.returncode, 2)
        self.assertNotIn("unknown", json.loads(depth_first.stdout)["error"])


if __name__ == "__main__":
    unittest.main()
