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

    def _register(self, identifier, depends_on=None):
        self.catalog.register({"id": identifier,
                               "fields": [{"name": "n", "type": "integer"}],
                               "depends_on": depends_on or []})

    def _seed_chain(self):
        for identifier, depends_on in (("orders", []), ("daily_totals", ["orders"]),
                                       ("weekly", ["daily_totals"]), ("report", ["orders"])):
            self._register(identifier, depends_on)

    def test_export_full_and_empty_catalogs(self):
        self.assertEqual(self.catalog.export(), {"datasets": []})
        self.assertFalse(self.path.exists())
        self._seed_chain()
        result = self.catalog.export()
        self.assertEqual([row["id"] for row in result["datasets"]],
                         ["orders", "daily_totals", "report", "weekly"])
        self.assertEqual(result["datasets"][1], self.catalog.describe("daily_totals"))
        self.assertEqual(self.catalog.export([]), {"datasets": []})

    def test_export_includes_upstream_only_in_dependency_order(self):
        self._seed_chain()
        result = self.catalog.export(["daily_totals"])
        self.assertEqual([row["id"] for row in result["datasets"]], ["orders", "daily_totals"])
        self.assertEqual([row["id"] for row in self.catalog.export(["orders"])["datasets"]], ["orders"])
        merged = self.catalog.export(["daily_totals", "orders", "daily_totals"])
        self.assertEqual([row["id"] for row in merged["datasets"]], ["orders", "daily_totals"])
        self.assertEqual([row["id"] for row in self.catalog.export(["weekly"])["datasets"]],
                         ["orders", "daily_totals", "weekly"])
        both = self.catalog.export(["report", "weekly"])
        self.assertEqual([row["id"] for row in both["datasets"]],
                         ["orders", "daily_totals", "report", "weekly"])

    def test_export_records_match_describe(self):
        self.catalog.register({"id": "orders",
                               "description": "root",
                               "fields": [{"name": "amount", "type": "number"}]})
        self.catalog.register({"id": "daily_totals",
                               "description": "rollup",
                               "fields": [{"name": "total", "type": "number"}],
                               "depends_on": ["orders"]})
        rows = self.catalog.export(["daily_totals"])["datasets"]
        self.assertEqual(rows, [self.catalog.describe("orders"), self.catalog.describe("daily_totals")])
        self.assertEqual([list(row) for row in rows],
                         [["id", "description", "fields", "depends_on"]] * 2)

    def test_export_round_trips_through_register(self):
        self._seed_chain()
        exported = self.catalog.export(["weekly", "report"])["datasets"]
        rebuilt_path = Path(self.temp.name) / "rebuilt.json"
        rebuilt = DatasetCatalog(rebuilt_path)
        for record in exported:
            rebuilt.register(record)
        fresh = DatasetCatalog(rebuilt_path)
        for identifier in ("orders", "daily_totals", "weekly", "report"):
            self.assertEqual(fresh.describe(identifier), self.catalog.describe(identifier))
        self.assertEqual([row["id"] for row in fresh.dependencies("weekly")], ["daily_totals"])
        self.assertEqual([row["dataset"]["id"] for row in fresh.impact("orders")],
                         ["daily_totals", "report", "weekly"])

    def test_export_rejects_bad_arguments(self):
        self._register("orders")
        for bad in ("orders", 1, 3.0, ("orders",), {"orders"}, True):
            with self.assertRaises(ValueError):
                self.catalog.export(bad)
        for bad in ([""], ["Orders"], ["orders", 1], ["bad id"], ["1abc"], ["-x"], [None]):
            with self.assertRaises(ValueError):
                self.catalog.export(bad)
        with self.assertRaises(ValueError):
            self.catalog.export(["missing"])
        with self.assertRaises(ValueError):
            self.catalog.export(["orders", "missing"])

    def test_export_missing_file_and_empty_object(self):
        missing = DatasetCatalog(Path(self.temp.name) / "absent.json")
        self.assertEqual(missing.export(), {"datasets": []})
        with self.assertRaises(ValueError):
            missing.export(["orders"])
        empty_path = Path(self.temp.name) / "empty.json"
        empty_path.write_text("{}", encoding="utf-8")
        empty = DatasetCatalog(empty_path)
        self.assertEqual(empty.export(None), {"datasets": []})
        with self.assertRaises(ValueError):
            empty.export(["orders"])
        with self.assertRaises(ValueError):
            empty.export([""])

    def test_export_detects_cycles_and_missing_dependencies(self):
        broken = Path(self.temp.name) / "broken.json"
        broken.write_text(json.dumps({
            "a": {"id": "a", "description": "", "fields": [{"name": "n", "type": "integer"}],
                  "depends_on": ["b"]},
            "b": {"id": "b", "description": "", "fields": [{"name": "n", "type": "integer"}],
                  "depends_on": ["a"]}}), encoding="utf-8")
        cyclic = DatasetCatalog(broken)
        with self.assertRaises(ValueError):
            cyclic.export(["a"])
        with self.assertRaises(ValueError):
            cyclic.export()
        broken.write_text(json.dumps({
            "a": {"id": "a", "description": "", "fields": [{"name": "n", "type": "integer"}],
                  "depends_on": ["ghost"]},
            "z": {"id": "z", "description": "", "fields": [{"name": "n", "type": "integer"}],
                  "depends_on": []}}), encoding="utf-8")
        gapped = DatasetCatalog(broken)
        with self.assertRaises(ValueError):
            gapped.export(["a"])
        with self.assertRaises(ValueError):
            gapped.export()
        self.assertEqual([row["id"] for row in gapped.export(["z"])["datasets"]], ["z"])

    def test_export_does_not_modify_catalog_file(self):
        self._seed_chain()
        before = self.path.read_text(encoding="utf-8")
        self.catalog.export(["weekly"])
        self.catalog.export()
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_cli_export(self):
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", str(self.path)]
        subprocess.run(prefix + ["register", str(ROOT / "samples/orders.json")], check=True, capture_output=True)
        subprocess.run(prefix + ["register", str(ROOT / "samples/daily.json")], check=True, capture_output=True)
        full = subprocess.run(prefix + ["export"], capture_output=True, text=True)
        self.assertEqual(full.returncode, 0, full.stderr)
        self.assertEqual([row["id"] for row in json.loads(full.stdout)["datasets"]],
                         ["orders", "daily_totals"])
        picked = subprocess.run(prefix + ["export", "--id", "daily_totals"],
                                capture_output=True, text=True)
        self.assertEqual([row["id"] for row in json.loads(picked.stdout)["datasets"]],
                         ["orders", "daily_totals"])
        orders_only = subprocess.run(prefix + ["export", "--id", "orders"],
                                     capture_output=True, text=True)
        self.assertEqual([row["id"] for row in json.loads(orders_only.stdout)["datasets"]], ["orders"])
        repeated = subprocess.run(prefix + ["export", "--id", "daily_totals", "--id", "orders",
                                            "--id", "daily_totals"], capture_output=True, text=True)
        self.assertEqual([row["id"] for row in json.loads(repeated.stdout)["datasets"]],
                         ["orders", "daily_totals"])
        for args in (["export", "--id", "missing"], ["export", "--id", ""]):
            failed = subprocess.run(prefix + args, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 2)
            self.assertIn("error", json.loads(failed.stdout))
        absent = [sys.executable, str(ROOT / "dataset_catalog.py"),
                  "--catalog", str(Path(self.temp.name) / "none.json")]
        ok = subprocess.run(absent + ["export"], capture_output=True, text=True)
        self.assertEqual(json.loads(ok.stdout), {"datasets": []})
        self.assertEqual(ok.returncode, 0)
        self.assertFalse((Path(self.temp.name) / "none.json").exists())

    def test_impact_lists_reachable_downstream_with_shortest_paths(self):
        for identifier, depends_on in (("orders", []), ("daily", ["orders"]),
                                       ("weekly", ["daily"]), ("report", ["orders"])):
            self._register(identifier, depends_on)
        result = self.catalog.impact("orders")
        self.assertEqual([(row["dataset"]["id"], row["distance"], row["path"]) for row in result],
                         [("daily", 1, ["orders", "daily"]),
                          ("report", 1, ["orders", "report"]),
                          ("weekly", 2, ["orders", "daily", "weekly"])])

    def test_impact_respects_max_depth_and_empty_leaf(self):
        for identifier, depends_on in (("orders", []), ("daily", ["orders"]),
                                       ("weekly", ["daily"])):
            self._register(identifier, depends_on)
        self.assertEqual([row["dataset"]["id"] for row in self.catalog.impact("orders", 1)], ["daily"])
        self.assertEqual(self.catalog.impact("weekly"), [])

    def test_impact_picks_lexicographically_smallest_shortest_path(self):
        for identifier, depends_on in (("orders", []), ("b", ["orders"]), ("c", ["orders"]),
                                       ("d", ["c", "b"])):
            self._register(identifier, depends_on)
        result = self.catalog.impact("orders")
        self.assertEqual([row["dataset"]["id"] for row in result], ["b", "c", "d"])
        self.assertEqual(result[-1]["path"], ["orders", "b", "d"])

    def test_impact_rejects_invalid_max_depth_and_unknown_source(self):
        self._register("orders")
        for bad in (True, False, 0, -1, 1.0, "1"):
            with self.assertRaises(ValueError):
                self.catalog.impact("orders", bad)
        with self.assertRaises(ValueError):
            self.catalog.impact("missing")

    def test_cli_impact(self):
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", str(self.path)]
        subprocess.run(prefix + ["register", str(ROOT / "samples/orders.json")], check=True, capture_output=True)
        subprocess.run(prefix + ["register", str(ROOT / "samples/daily.json")], check=True, capture_output=True)
        ok = subprocess.run(prefix + ["impact", "orders"], capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        result = json.loads(ok.stdout)
        self.assertEqual([row["dataset"]["id"] for row in result], ["daily_totals"])
        self.assertEqual(result[0]["distance"], 1)
        self.assertEqual(result[0]["path"], ["orders", "daily_totals"])
        self.assertEqual(subprocess.run(prefix + ["impact", "orders", "--max-depth", "0"],
                                       capture_output=True, text=True).returncode, 2)
        depth_error = subprocess.run(prefix + ["impact", "missing", "--max-depth", "x"],
                                     capture_output=True, text=True)
        self.assertEqual(depth_error.returncode, 2)
        self.assertIn("error", json.loads(depth_error.stdout))
        self.assertEqual(subprocess.run(prefix + ["impact", "missing"],
                                        capture_output=True).returncode, 2)

    def test_cli_query_and_registration(self):
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", str(self.path)]
        run = subprocess.run(prefix + ["register", str(ROOT / "samples/orders.json")], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        query = subprocess.run(prefix + ["describe", "orders"], capture_output=True, text=True)
        self.assertEqual(json.loads(query.stdout)["id"], "orders")
        self.assertEqual(subprocess.run(prefix + ["describe", "missing"], capture_output=True).returncode, 2)


if __name__ == "__main__":
    unittest.main()
