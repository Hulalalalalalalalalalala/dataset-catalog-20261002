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


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "catalog.json"
        self.catalog = DatasetCatalog(self.path)
        self.samples = DatasetCatalog(ROOT / "samples" / "catalog.json")

    def _copy_samples(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text((ROOT / "samples" / "catalog.json").read_text(encoding="utf-8"),
                             encoding="utf-8")
        return DatasetCatalog(self.path)

    def test_sample_daily_amount_finds_only_orders_with_amount_field(self):
        catalog = self._copy_samples()
        result = catalog.search("daily amount")
        self.assertEqual([row["dataset"]["id"] for row in result], ["orders"])
        self.assertEqual(result[0]["matched_fields"], ["amount"])
        self.assertEqual(result[0]["dataset"], catalog.describe("orders"))

    def test_words_must_all_match_but_may_target_different_places(self):
        catalog = self._copy_samples()
        self.assertEqual([row["dataset"]["id"] for row in catalog.search("orders total")],
                         ["daily_totals"])
        self.assertEqual(catalog.search("amount total"), [])
        result = catalog.search("order")
        self.assertEqual([row["dataset"]["id"] for row in result], ["daily_totals", "orders"])
        self.assertEqual(result[0]["matched_fields"], [])
        self.assertEqual(result[1]["matched_fields"], ["order_id"])

    def test_duplicate_and_whitespace_words_and_casefold(self):
        catalog = self._copy_samples()
        self.assertEqual([row["dataset"]["id"] for row in catalog.search("  ORDER   order ")],
                         ["daily_totals", "orders"])
        self.assertEqual(catalog.search(" \t\n ")[0]["matched_fields"], [])
        self.assertEqual({row["dataset"]["id"] for row in catalog.search("AMOUNT")}, {"orders"})

    def test_empty_query_matches_all_with_empty_matched_fields(self):
        catalog = self._copy_samples()
        for invocation in (catalog.search(""), catalog.search("   "),
                           catalog.search("", field_type="number")):
            self.assertEqual([row["dataset"]["id"] for row in invocation],
                             ["daily_totals", "orders"])
            self.assertTrue(all(row["matched_fields"] == [] for row in invocation))

    def test_field_type_filter_and_combined_keywords(self):
        catalog = self._copy_samples()
        self.assertEqual({row["dataset"]["id"] for row in catalog.search(field_type="string")},
                         {"daily_totals", "orders"})
        result = catalog.search("day", field_type="integer")
        self.assertEqual(result, [])
        result = catalog.search("total", field_type="number")
        self.assertEqual([row["dataset"]["id"] for row in result], ["daily_totals"])
        self.assertEqual(result[0]["matched_fields"], ["total"])

    def test_search_covers_id_description_and_field_names_but_not_dependencies(self):
        catalog = self._copy_samples()
        self.assertEqual({row["dataset"]["id"] for row in catalog.search("daily_totals")},
                         {"daily_totals"})
        self.assertEqual({row["dataset"]["id"] for row in catalog.search("amounts")},
                         {"orders"})
        self.assertEqual({row["dataset"]["id"] for row in catalog.search("number")}, set())

    def test_missing_or_empty_catalog_returns_empty_list(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(self.catalog.search("anything"), [])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{}", encoding="utf-8")
        self.assertEqual(DatasetCatalog(self.path).search("anything", field_type="string"), [])

    def test_invalid_arguments_raise_even_on_empty_catalog(self):
        self.assertFalse(self.path.exists())
        for bad in (1, 1.0, True, None, ["x"], {"x"}):
            with self.assertRaises(ValueError):
                self.catalog.search(bad)
        for bad in (1, True, "str", "STRING", "", "float"):
            with self.assertRaises(ValueError):
                self.catalog.search("", field_type=bad)

    def test_search_does_not_create_or_modify_catalog(self):
        self.assertFalse(self.path.exists())
        self.samples.search("orders", field_type="string")
        self.assertEqual([row["dataset"]["id"]
                          for row in self.samples.search("daily amount")], ["orders"])
        catalog = self._copy_samples()
        before = self.path.read_text(encoding="utf-8")
        catalog.search("", field_type="boolean")
        catalog.search("orders")
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_matched_fields_keep_order_without_duplicates(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "d": {"id": "d", "description": "",
                  "fields": [{"name": "alpha_id", "type": "string"},
                             {"name": "beta", "type": "integer"},
                             {"name": "alpha_two", "type": "string"}],
                  "depends_on": []}}), encoding="utf-8")
        result = DatasetCatalog(self.path).search("alpha id")
        self.assertEqual(result[0]["matched_fields"], ["alpha_id", "alpha_two"])

    def test_cli_search(self):
        catalog_path = str(ROOT / "samples" / "catalog.json")
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", catalog_path]
        ok = subprocess.run(prefix + ["search", "daily amount"], capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        result = json.loads(ok.stdout)
        self.assertEqual([row["dataset"]["id"] for row in result], ["orders"])
        self.assertEqual(result[0]["matched_fields"], ["amount"])
        empty_query = subprocess.run(prefix + ["search"], capture_output=True, text=True)
        self.assertEqual(empty_query.returncode, 0, empty_query.stderr)
        self.assertEqual([row["dataset"]["id"] for row in json.loads(empty_query.stdout)],
                         ["daily_totals", "orders"])
        typed = subprocess.run(prefix + ["search", "total", "--field-type", "number"],
                               capture_output=True, text=True)
        self.assertEqual(typed.returncode, 0, typed.stderr)
        self.assertEqual([row["dataset"]["id"] for row in json.loads(typed.stdout)],
                         ["daily_totals"])
        no_match = subprocess.run(prefix + ["search", "zzz"], capture_output=True, text=True)
        self.assertEqual(json.loads(no_match.stdout), [])
        missing_catalog = prefix[:3] + [str(Path(self.temp.name) / "nope" / "c.json")]
        missing = subprocess.run(missing_catalog + ["search", "x"],
                                 capture_output=True, text=True)
        self.assertEqual(missing.returncode, 0, missing.stderr)
        self.assertEqual(json.loads(missing.stdout), [])
        self.assertFalse(Path(self.temp.name, "nope").exists())
        bad = subprocess.run(prefix + ["search", "x", "--field-type", "float"],
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        self.assertIn("error", json.loads(bad.stdout))


if __name__ == "__main__":
    unittest.main()
