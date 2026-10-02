import copy
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

    def test_malformed_descriptions_all_raise_value_error(self):
        valid_field = {"name": "amount", "type": "number"}
        invalid = [
            None, True, False, 1, 1.5, "orders", ["orders"], (),
            {},
            {"fields": [valid_field]},
            {"id": "orders"},
            {"id": None, "fields": [valid_field]},
            {"id": 3, "fields": [valid_field]},
            {"id": ["orders"], "fields": [valid_field]},
            {"id": {"x": 1}, "fields": [valid_field]},
            {"id": "Orders", "fields": [valid_field]},
            {"id": "1orders", "fields": [valid_field]},
            {"id": "ord ers", "fields": [valid_field]},
            {"id": "orders", "fields": None},
            {"id": "orders", "fields": []},
            {"id": "orders", "fields": valid_field},
            {"id": "orders", "fields": "amount"},
            {"id": "orders", "fields": [["amount", "number"]]},
            {"id": "orders", "fields": ["amount"]},
            {"id": "orders", "fields": [1]},
            {"id": "orders", "fields": [None]},
            {"id": "orders", "fields": [{"type": "number"}]},
            {"id": "orders", "fields": [{"name": "amount"}]},
            {"id": "orders", "fields": [{"name": None, "type": "number"}]},
            {"id": "orders", "fields": [{"name": 1, "type": "number"}]},
            {"id": "orders", "fields": [{"name": ["amount"], "type": "number"}]},
            {"id": "orders", "fields": [{"name": {"x": 1}, "type": "number"}]},
            {"id": "orders", "fields": [{"name": "", "type": "number"}]},
            {"id": "orders", "fields": [{"name": "   ", "type": "number"}]},
            {"id": "orders", "fields": [{"name": "amount", "type": None}]},
            {"id": "orders", "fields": [{"name": "amount", "type": 1}]},
            {"id": "orders", "fields": [{"name": "amount", "type": "float"}]},
            {"id": "orders", "fields": [{"name": "amount", "type": ["number"]}]},
            {"id": "orders", "fields": [{"name": "amount", "type": {"t": "number"}}]},
            {"id": "orders", "fields": [dict(valid_field), dict(valid_field)]},
            {"id": "orders", "fields": [valid_field], "depends_on": "orders"},
            {"id": "orders", "fields": [valid_field], "depends_on": {"orders": 1}},
            {"id": "orders", "fields": [valid_field], "depends_on": [1]},
            {"id": "orders", "fields": [valid_field], "depends_on": [None]},
            {"id": "orders", "fields": [valid_field], "depends_on": [["orders"]]},
            # several problems at once must still surface the single exception type
            {"id": "BAD ID", "fields": [{"type": None}], "depends_on": [1, 1]},
        ]
        for bad in invalid:
            snapshot = copy.deepcopy(bad)
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.catalog.register(bad)
            self.assertEqual(bad, snapshot)
        self.assertFalse(self.path.exists())

    def test_failed_registration_creates_no_catalog_or_parent_directories(self):
        nested = Path(self.temp.name) / "nested" / "deeper" / "catalog.json"
        catalog = DatasetCatalog(nested)
        for bad in ("not-a-dict", ["not-a-dict"], 42,
                    {"id": "orders"}, {"id": "orders", "fields": []},
                    {"id": "orders", "fields": [{"name": "n", "type": "weird"}]}):
            with self.assertRaises(ValueError):
                catalog.register(copy.deepcopy(bad))
        self.assertFalse(nested.exists())
        self.assertFalse(nested.parent.exists())

    def test_failed_registration_keeps_existing_catalog_bytes_unchanged(self):
        self.catalog.register({"id": "orders", "fields": [{"name": "amount", "type": "number"}]})
        before = self.path.read_bytes()
        mtime = self.path.stat().st_mtime_ns
        for bad in (
            {"id": "orders", "fields": [{"name": "amount", "type": "number"}]},
            {"id": "daily", "fields": [{"name": "total", "type": "number"}], "depends_on": ["missing"]},
            {"id": "daily", "fields": [{"name": "total", "type": "number"}],
             "depends_on": ["orders", "orders"]},
            {"id": "daily", "fields": [{"name": "total", "type": "weird"}]},
            {"id": "daily", "fields": [{"name": " ", "type": "number"}]},
        ):
            with self.assertRaises(ValueError):
                self.catalog.register(copy.deepcopy(bad))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.path.stat().st_mtime_ns, mtime)

    def test_valid_registration_preserves_whitespace_order_and_defaults(self):
        descriptor = {
            "id": "orders", "description": 123, "meta": {"owner": "analytics"},
            "fields": [{"name": " order id ", "type": "string", "extra": "x"},
                       {"name": "amount", "type": "number"}],
            "depends_on": [],
        }
        snapshot = copy.deepcopy(descriptor)
        entry = self.catalog.register(descriptor)
        self.assertEqual(entry["fields"], [{"name": " order id ", "type": "string"},
                                           {"name": "amount", "type": "number"}])
        self.assertEqual(entry["description"], "123")
        self.assertEqual(entry["depends_on"], [])
        self.assertEqual(descriptor, snapshot)
        fresh = DatasetCatalog(self.path)
        self.assertEqual(fresh.describe("orders"), entry)
        minimal = self.catalog.register({"id": "bare",
                                         "fields": [{"name": "n", "type": "integer"}]})
        self.assertEqual(minimal["description"], "")
        self.assertEqual(minimal["depends_on"], [])

    def test_dependencies_sorted_and_identical_after_reopen(self):
        for descriptor in (
            {"id": "orders", "fields": [{"name": "n", "type": "integer"}]},
            {"id": "daily", "fields": [{"name": "n", "type": "integer"}], "depends_on": ["orders"]},
            {"id": "report", "fields": [{"name": "n", "type": "integer"}],
             "depends_on": ["daily", "orders"]},
        ):
            self.catalog.register(descriptor)
        record = self.catalog.describe("report")
        fresh = DatasetCatalog(self.path)
        self.assertEqual(fresh.describe("report"), record)
        self.assertEqual([row["id"] for row in fresh.dependencies("report")],
                         ["daily", "orders"])

    def test_queries_identical_before_and_after_failed_registration(self):
        for name in ("orders.json", "daily.json"):
            descriptor = json.loads((ROOT / "samples" / name).read_text(encoding="utf-8"))
            self.catalog.register(descriptor)

        def snapshot():
            return {
                "describe": self.catalog.describe("orders"),
                "dependencies": self.catalog.dependencies("daily_totals"),
                "impact": self.catalog.impact("orders"),
                "export_all": self.catalog.export(),
                "export_one": self.catalog.export(["daily_totals"]),
            }

        before = snapshot()
        for bad in ({"id": "broken", "fields": [{"name": "x", "type": "weird"}],
                     "depends_on": ["nope"]},
                    {"id": "orders", "fields": [{"name": "x", "type": "number"}]}):
            with self.assertRaises(ValueError):
                self.catalog.register(copy.deepcopy(bad))
        self.assertEqual(snapshot(), before)

    def test_cli_invalid_registration_prints_json_error_and_exits_2(self):
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", str(self.path)]
        descriptor = Path(self.temp.name) / "bad.json"
        for content in (json.dumps({"id": "bad", "fields": [{"name": "x", "type": "weird"}]}),
                        json.dumps(["not", "a", "dict"]),
                        "{not valid json"):
            descriptor.write_text(content, encoding="utf-8")
            run = subprocess.run(prefix + ["register", str(descriptor)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            payload = json.loads(run.stdout)
            self.assertEqual(set(payload), {"error"})
            self.assertTrue(payload["error"])
            self.assertFalse(self.path.exists())

    def _register(self, identifier, depends_on=None):
        self.catalog.register({"id": identifier,
                               "fields": [{"name": "n", "type": "integer"}],
                               "depends_on": depends_on or []})

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

    def _chain(self):
        for identifier, depends_on in (("orders", []), ("daily_totals", ["orders"]),
                                       ("weekly", ["daily_totals"]), ("report", ["orders"])):
            self._register(identifier, depends_on)

    def test_export_all_orders_dependencies_before_dependents(self):
        self._chain()
        result = self.catalog.export()
        self.assertEqual(set(result), {"datasets"})
        ids = [row["id"] for row in result["datasets"]]
        self.assertEqual(ids.index("orders"), 0)
        self.assertLess(ids.index("daily_totals"), ids.index("weekly"))
        self.assertEqual([row["id"] for row in self.catalog.export([])["datasets"]], [])

    def test_export_selected_includes_only_upstream_closure(self):
        self._chain()
        ids = [row["id"] for row in self.catalog.export(["daily_totals"])["datasets"]]
        self.assertEqual(ids, ["orders", "daily_totals"])
        self.assertEqual([row["id"] for row in self.catalog.export(["orders"])["datasets"]], ["orders"])
        ids = [row["id"] for row in self.catalog.export(["report", "weekly", "weekly"])["datasets"]]
        self.assertEqual(ids, ["orders", "daily_totals", "report", "weekly"])
        ids = [row["id"] for row in self.catalog.export(["weekly", "report"])["datasets"]]
        self.assertEqual(ids, ["orders", "daily_totals", "report", "weekly"])
        daily = self.catalog.describe("daily_totals")
        self.assertEqual(self.catalog.export(["daily_totals"])["datasets"][1], daily)
        self.assertEqual(list(self.catalog.export(["daily_totals"])["datasets"][1]),
                         ["id", "description", "fields", "depends_on"])

    def test_export_rejects_bad_arguments_and_unknown_ids(self):
        self._register("orders")
        for bad in ("orders", ("orders",), {1}, True, 1):
            with self.assertRaises(ValueError):
                self.catalog.export(bad)
        for bad in ([""], ["Orders"], ["ord ers"], [1], [None], ["orders", ""]):
            with self.assertRaises(ValueError):
                self.catalog.export(bad)
        with self.assertRaises(ValueError):
            self.catalog.export(["missing"])

    def _write_records(self, records):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(records), encoding="utf-8")

    def _record(self, identifier, depends_on):
        return {"id": identifier, "description": "",
                "fields": [{"name": "n", "type": "integer"}], "depends_on": depends_on}

    def test_export_fails_on_missing_dependency_or_cycle_within_selection(self):
        self._write_records({"a": self._record("a", ["ghost"]), "b": self._record("b", [])})
        with self.assertRaises(ValueError):
            self.catalog.export(["a"])
        self.assertEqual([row["id"] for row in self.catalog.export(["b"])["datasets"]], ["b"])
        self._write_records({"a": self._record("a", ["b"]), "b": self._record("b", ["a"]),
                             "c": self._record("c", [])})
        with self.assertRaises(ValueError):
            self.catalog.export(["a"])
        self.assertEqual([row["id"] for row in self.catalog.export(["c"])["datasets"]], ["c"])

    def test_export_missing_or_empty_catalog_file(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(self.catalog.export(), {"datasets": []})
        self.path.write_text("{}", encoding="utf-8")
        self.assertEqual(DatasetCatalog(self.path).export(), {"datasets": []})
        with self.assertRaises(ValueError):
            self.catalog.export(["orders"])

    def test_export_roundtrips_into_empty_catalog(self):
        sample = DatasetCatalog(ROOT / "samples" / "catalog.json")
        bundle = sample.export(["daily_totals"])
        for row in bundle["datasets"]:
            self.catalog.register(row)
        rebuilt = DatasetCatalog(self.path)
        original = DatasetCatalog(ROOT / "samples" / "catalog.json")
        for identifier in ("orders", "daily_totals"):
            self.assertEqual(rebuilt.describe(identifier), original.describe(identifier))
        self.assertEqual([row["id"] for row in rebuilt.dependencies("daily_totals")], ["orders"])
        self.assertEqual(rebuilt.dependencies("orders"), [])

    def test_export_does_not_modify_catalog_file(self):
        self._chain()
        before = self.path.read_text(encoding="utf-8")
        self.catalog.export(["weekly"])
        self.catalog.export()
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_cli_export(self):
        base = [sys.executable, str(ROOT / "dataset_catalog.py")]
        ok = subprocess.run(base + ["export", "--id", "daily_totals"], capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        result = json.loads(ok.stdout)
        self.assertEqual([row["id"] for row in result["datasets"]], ["orders", "daily_totals"])
        full = subprocess.run(base + ["export"], capture_output=True, text=True)
        self.assertEqual(full.returncode, 0, full.stderr)
        self.assertEqual({row["id"] for row in json.loads(full.stdout)["datasets"]},
                         {"orders", "daily_totals"})
        custom = base + ["--catalog", str(self.path)]
        missing = subprocess.run(custom + ["export", "--id", "nope"],
                                 capture_output=True, text=True)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("error", json.loads(missing.stdout))
        empty = subprocess.run(custom + ["export"], capture_output=True, text=True)
        self.assertEqual(json.loads(empty.stdout), {"datasets": []})
        bad = subprocess.run(base + ["export", "--id", ""], capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        self.assertIn("error", json.loads(bad.stdout))


if __name__ == "__main__":
    unittest.main()
