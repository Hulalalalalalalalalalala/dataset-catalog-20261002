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

    def test_search_matches_keywords_across_id_description_and_field_names(self):
        self._write_records({
            "orders": {"id": "orders", "description": "Order amounts from the daily export",
                       "fields": [{"name": "order_id", "type": "string"},
                                  {"name": "amount", "type": "number"}], "depends_on": []},
            "daily_totals": {"id": "daily_totals", "description": "Daily totals derived from orders",
                             "fields": [{"name": "day", "type": "string"},
                                        {"name": "total", "type": "number"}],
                             "depends_on": ["orders"]}})
        result = self.catalog.search("daily amount")
        self.assertEqual([row["dataset"]["id"] for row in result], ["orders"])
        self.assertEqual(result[0]["matched_fields"], ["amount"])
        self.assertEqual(result[0]["dataset"], self.catalog.describe("orders"))
        result = self.catalog.search("orders")
        self.assertEqual([row["dataset"]["id"] for row in result], ["daily_totals", "orders"])
        self.assertTrue(all(row["matched_fields"] == [] for row in result))

    def test_search_treats_repeated_and_split_words_with_casefold(self):
        self._write_records({
            "a": {"id": "a", "description": "first dataset",
                  "fields": [{"name": "user_name", "type": "string"},
                             {"name": "count", "type": "integer"}], "depends_on": []}})
        result = self.catalog.search("  USER   user  COUNT")
        self.assertEqual([row["dataset"]["id"] for row in result], ["a"])
        self.assertEqual(result[0]["matched_fields"], ["user_name", "count"])
        self.assertEqual(self.catalog.search("USERNAME"), [])
        self.assertEqual(self.catalog.search("count")[0]["matched_fields"], ["count"])

    def test_search_does_not_match_dependency_ids_or_field_type_text(self):
        self._write_records({
            "a": {"id": "a", "description": "", "fields": [{"name": "n", "type": "integer"}],
                  "depends_on": ["zz_hidden"]},
            "zz_hidden": {"id": "zz_hidden", "description": "",
                          "fields": [{"name": "flag", "type": "boolean"}], "depends_on": []}})
        self.assertEqual([row["dataset"]["id"] for row in self.catalog.search("zz_hidden")], ["zz_hidden"])
        self.assertEqual(self.catalog.search("integer"), [])

    def test_search_empty_query_and_field_type_filter(self):
        self._write_records({
            "a": {"id": "a", "description": "first",
                  "fields": [{"name": "n", "type": "integer"},
                             {"name": "label", "type": "string"}], "depends_on": []},
            "b": {"id": "b", "description": "second",
                  "fields": [{"name": "ratio", "type": "number"}], "depends_on": []}})
        self.assertEqual([row["dataset"]["id"] for row in self.catalog.search()], ["a", "b"])
        self.assertEqual([row["dataset"]["id"] for row in self.catalog.search("   ")], ["a", "b"])
        self.assertEqual([row["dataset"]["id"] for row in self.catalog.search(field_type="number")], ["b"])
        self.assertEqual([row["dataset"]["id"] for row in self.catalog.search(field_type="integer")], ["a"])
        result = self.catalog.search("first", field_type="number")
        self.assertEqual(result, [])
        result = self.catalog.search("n", field_type="integer")
        self.assertEqual(result[0]["matched_fields"], ["n"])

    def test_search_validates_arguments_before_reading_catalog(self):
        self.assertFalse(self.path.exists())
        for bad in (1, None, b"x", ["a"]):
            with self.assertRaises(ValueError):
                self.catalog.search(bad)
        for bad in ("float", "STRING", "", 1, True):
            with self.assertRaises(ValueError):
                self.catalog.search("q", bad)
        self.assertEqual(self.catalog.search(), [])
        self.assertEqual(self.catalog.search("q", "string"), [])
        self.assertFalse(self.path.exists())

    def test_search_does_not_modify_catalog(self):
        self._write_records({
            "a": {"id": "a", "description": "", "fields": [{"name": "n", "type": "integer"}],
                  "depends_on": []}})
        before = self.path.read_text(encoding="utf-8")
        self.catalog.search("a", "integer")
        self.catalog.search("")
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_cli_search(self):
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", str(self.path)]
        subprocess.run(prefix + ["register", str(ROOT / "samples/orders.json")], check=True, capture_output=True)
        subprocess.run(prefix + ["register", str(ROOT / "samples/daily.json")], check=True, capture_output=True)
        run = subprocess.run(prefix + ["search", "daily amount"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual([row["dataset"]["id"] for row in result], ["orders"])
        self.assertEqual(result[0]["matched_fields"], ["amount"])
        empty = subprocess.run(prefix + ["search"], capture_output=True, text=True)
        self.assertEqual(empty.returncode, 0)
        self.assertEqual([row["dataset"]["id"] for row in json.loads(empty.stdout)],
                         ["daily_totals", "orders"])
        none = subprocess.run(prefix + ["search", "nothing-matches"], capture_output=True, text=True)
        self.assertEqual(json.loads(none.stdout), [])
        bad = subprocess.run(prefix + ["search", "x", "--field-type", "float"],
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        self.assertIn("error", json.loads(bad.stdout))

    def test_cli_search_missing_catalog_file(self):
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", str(self.path)]
        run = subprocess.run(prefix + ["search", "x", "--field-type", "float"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("error", json.loads(run.stdout))
        run = subprocess.run(prefix + ["search"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0)
        self.assertEqual(json.loads(run.stdout), [])
        self.assertFalse(self.path.exists())

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


    def test_register_normalizes_and_persists_tags(self):
        entry = self.catalog.register({**self.raw, "tags": [" Finance ", "日汇总", "finance", "FINANCE"]})
        self.assertEqual(entry["tags"], ["Finance", "日汇总"])
        fresh = DatasetCatalog(self.path)
        self.assertEqual(fresh.describe("orders")["tags"], ["Finance", "日汇总"])
        entry = self.catalog.register({"id": "empty", "fields": [{"name": "n", "type": "integer"}], "tags": []})
        self.assertEqual(entry["tags"], [])
        entry = self.catalog.register({"id": "plain", "fields": [{"name": "n", "type": "integer"}]})
        self.assertNotIn("tags", entry)
        self.assertNotIn("tags", DatasetCatalog(self.path).describe("plain"))

    def test_register_rejects_invalid_tags_without_writing(self):
        for bad in ("Finance", 1, True, {"a": 1}, ["ok", 1], [None], [""], ["  "], ["ok", "\t"]):
            with self.assertRaises(ValueError):
                self.catalog.register({**self.raw, "tags": bad})
        self.assertFalse(self.path.exists())
        self.catalog.register(self.raw)
        before = self.path.read_text(encoding="utf-8")
        with self.assertRaises(ValueError):
            self.catalog.register({"id": "other", "fields": [{"name": "n", "type": "integer"}], "tags": [""]})
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def _tagged_records(self):
        return {
            "daily": {"id": "daily", "description": "offline export",
                      "fields": [{"name": "total", "type": "number"}], "depends_on": [],
                      "tags": ["Finance", "日汇总"]},
            "orders": {"id": "orders", "description": "offline export",
                       "fields": [{"name": "amount", "type": "number"}], "depends_on": [],
                       "tags": ["Finance"]},
            "legacy": {"id": "legacy", "description": "no tags key",
                       "fields": [{"name": "n", "type": "integer"}], "depends_on": []}}

    def test_search_requires_all_tags_with_casefold_full_match(self):
        self._write_records(self._tagged_records())
        result = self.catalog.search(tags=["finance", "日汇总"])
        self.assertEqual([row["dataset"]["id"] for row in result], ["daily"])
        result = self.catalog.search(tags=["FINANCE"])
        self.assertEqual([row["dataset"]["id"] for row in result], ["daily", "orders"])
        self.assertEqual(self.catalog.search(tags=["fin"]), [])
        self.assertEqual(self.catalog.search(tags=["finance", "missing"]), [])
        with self.assertRaises(ValueError):
            self.catalog.search(tags=["finance", ""])
        for tags in (None, []):
            self.assertEqual([row["dataset"]["id"] for row in self.catalog.search(tags=tags)],
                             ["daily", "legacy", "orders"])
        result = self.catalog.search("offline", field_type="number", tags=[" finance ", "Finance"])
        self.assertEqual([row["dataset"]["id"] for row in result], ["daily", "orders"])
        self.assertEqual(result[0]["matched_fields"], [])
        result = self.catalog.search("total", tags=["日汇总"])
        self.assertEqual(result[0]["matched_fields"], ["total"])

    def test_search_validates_tags_before_reading_catalog(self):
        self.assertFalse(self.path.exists())
        for bad in ("Finance", 1, True, {"a": 1}, ["ok", 1], [None], [""], ["  "]):
            with self.assertRaises(ValueError):
                self.catalog.search(tags=bad)
        self.assertEqual(self.catalog.search(tags=["finance"]), [])
        self.assertFalse(self.path.exists())

    def test_search_tags_does_not_modify_catalog(self):
        self._write_records(self._tagged_records())
        before = self.path.read_text(encoding="utf-8")
        self.catalog.search(tags=["finance"])
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        self.assertNotIn("tags", self.catalog.describe("legacy"))

    def test_tags_survive_export_roundtrip_and_queries(self):
        self.catalog.register({"id": "daily", "fields": [{"name": "total", "type": "number"}],
                               "tags": ["Finance", "日汇总"]})
        self.catalog.register({"id": "orders", "fields": [{"name": "amount", "type": "number"}],
                               "tags": ["Finance"]})
        bundle = self.catalog.export()
        other_path = self.path.parent / "rebuilt.json"
        rebuilt = DatasetCatalog(other_path)
        for row in bundle["datasets"]:
            rebuilt.register(row)
        for identifier in ("daily", "orders"):
            self.assertEqual(rebuilt.describe(identifier), self.catalog.describe(identifier))
        self.assertEqual(rebuilt.dependencies("orders"), [])
        impact = self.catalog.impact("daily")
        self.assertEqual(impact, [])
        self.assertEqual(self.catalog.describe("daily")["tags"], ["Finance", "日汇总"])

    def test_cli_search_tag_filter(self):
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", str(self.path)]
        first = {"id": "daily", "description": "offline",
                 "fields": [{"name": "total", "type": "number"}], "tags": ["Finance", "日汇总"]}
        second = {"id": "orders", "description": "offline",
                  "fields": [{"name": "amount", "type": "number"}], "tags": ["Finance"]}
        for row in (first, second):
            descriptor = Path(self.temp.name) / (row["id"] + ".json")
            descriptor.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            run = subprocess.run(prefix + ["register", str(descriptor)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
        both = subprocess.run(prefix + ["search", "--tag", "finance", "--tag", "日汇总"],
                              capture_output=True, text=True)
        self.assertEqual(both.returncode, 0, both.stderr)
        self.assertEqual([row["dataset"]["id"] for row in json.loads(both.stdout)], ["daily"])
        one = subprocess.run(prefix + ["search", "--tag", "finance"], capture_output=True, text=True)
        self.assertEqual([row["dataset"]["id"] for row in json.loads(one.stdout)], ["daily", "orders"])
        combo = subprocess.run(prefix + ["search", "offline", "--field-type", "number", "--tag", "finance"],
                               capture_output=True, text=True)
        self.assertEqual(combo.returncode, 0, combo.stderr)
        self.assertEqual([row["dataset"]["id"] for row in json.loads(combo.stdout)], ["daily", "orders"])
        plain = subprocess.run(prefix + ["search"], capture_output=True, text=True)
        self.assertEqual([row["dataset"]["id"] for row in json.loads(plain.stdout)], ["daily", "orders"])
        bad = subprocess.run(prefix + ["search", "--tag", "  "], capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        self.assertIn("error", json.loads(bad.stdout))
        described = subprocess.run(prefix + ["describe", "daily"], capture_output=True, text=True)
        self.assertEqual(json.loads(described.stdout)["tags"], ["Finance", "日汇总"])

    def test_cli_export_preserves_tags(self):
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", str(self.path)]
        descriptor = Path(self.temp.name) / "tagged.json"
        descriptor.write_text(json.dumps({"id": "daily", "fields": [{"name": "total", "type": "number"}],
                                          "tags": ["Finance", "日汇总"]}, ensure_ascii=False),
                              encoding="utf-8")
        subprocess.run(prefix + ["register", str(descriptor)], check=True, capture_output=True)
        run = subprocess.run(prefix + ["export"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["datasets"][0]["tags"], ["Finance", "日汇总"])

    def _desc(self, identifier, depends_on=None, **extra):
        descriptor = {"id": identifier,
                      "description": identifier + " dataset",
                      "fields": [{"name": "n", "type": "integer"}],
                      "depends_on": depends_on or []}
        descriptor.update(extra)
        return descriptor

    def test_import_orders_dependencies_first_independent_of_input_order(self):
        shuffled = [self._desc("c", ["a", "b"]), self._desc("b", ["a"]), self._desc("a")]
        expected = ["a", "b", "c"]
        orders = (shuffled, list(reversed(shuffled)), [shuffled[1], shuffled[2], shuffled[0]])
        for index, order in enumerate(orders):
            path = self.path.parent / ("catalog-order-" + str(index) + ".json")
            catalog = DatasetCatalog(path)
            result = catalog.import_bundle({"datasets": order})
            self.assertEqual([row["id"] for row in result["datasets"]], expected)
            self.assertEqual([row["id"] for row in DatasetCatalog(path).export()["datasets"]],
                             expected)

    def test_import_breaks_ready_ties_by_lexicographic_id(self):
        bundle = {"datasets": [self._desc("z"), self._desc("a"), self._desc("m"),
                               self._desc("root", ["z", "a", "m"])]}
        result = self.catalog.import_bundle(bundle)
        self.assertEqual([row["id"] for row in result["datasets"]], ["a", "m", "z", "root"])

    def test_import_resolves_batch_dependencies_in_any_position(self):
        result = self.catalog.import_bundle({"datasets": [
            self._desc("top", ["mid"]), self._desc("mid", ["base"]), self._desc("base")]})
        self.assertEqual([row["id"] for row in result["datasets"]], ["base", "mid", "top"])
        fresh = DatasetCatalog(self.path)
        self.assertEqual([row["id"] for row in fresh.dependencies("top")], ["mid"])
        self.assertEqual([row["id"] for row in fresh.dependencies("mid")], ["base"])

    def test_import_mixes_existing_and_batch_dependencies(self):
        self.catalog.register(self._desc("base"))
        result = self.catalog.import_bundle({"datasets": [
            self._desc("c", ["base", "a"]), self._desc("a", ["base"])]})
        self.assertEqual([row["id"] for row in result["datasets"]], ["a", "c"])
        fresh = DatasetCatalog(self.path)
        self.assertEqual([row["id"] for row in fresh.dependencies("c")], ["a", "base"])
        self.assertEqual({row["id"] for row in fresh.export()["datasets"]},
                         {"a", "base", "c"})
        self.assertEqual([row["dataset"]["id"] for row in fresh.impact("base")], ["a", "c"])
        self.assertEqual([row["dataset"]["id"] for row in fresh.search("dataset")],
                         ["a", "base", "c"])

    def test_import_returns_only_newly_added_normalized_records(self):
        self.catalog.register(self._desc("base", tags=["Base"]))
        bundle = {"datasets": [self._desc("a", ["base"], description=5,
                                          tags=[" Finance ", "finance"])],
                  "ignored": "top-level key"}
        result = self.catalog.import_bundle(bundle)
        self.assertEqual(set(result), {"datasets"})
        self.assertEqual([row["id"] for row in result["datasets"]], ["a"])
        entry = result["datasets"][0]
        self.assertEqual(entry, {"id": "a", "description": "5",
                                 "fields": [{"name": "n", "type": "integer"}],
                                 "depends_on": ["base"], "tags": ["Finance"]})

    def test_import_distinguishes_missing_tags_from_empty_tags(self):
        result = self.catalog.import_bundle({"datasets": [
            {"id": "plain", "fields": [{"name": "n", "type": "integer"}], "depends_on": []},
            {"id": "tagged", "fields": [{"name": "n", "type": "integer"}],
             "depends_on": [], "tags": []}]})
        records = {row["id"]: row for row in result["datasets"]}
        self.assertNotIn("tags", records["plain"])
        self.assertEqual(records["tagged"]["tags"], [])
        fresh = DatasetCatalog(self.path)
        self.assertNotIn("tags", fresh.describe("plain"))
        self.assertEqual(fresh.describe("tagged")["tags"], [])

    def test_import_empty_array_changes_nothing(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(self.catalog.import_bundle({"datasets": []}), {"datasets": []})
        self.assertFalse(self.path.exists())
        self.catalog.register(self.raw)
        before = self.path.read_text(encoding="utf-8")
        self.assertEqual(self.catalog.import_bundle({"datasets": [], "extra": 1}),
                         {"datasets": []})
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_import_rejects_bad_bundle_shapes(self):
        for bad in ([], "x", 1, None, True, {"datasets": {}}, {"records": []}, {}):
            with self.assertRaises(ValueError):
                self.catalog.import_bundle(bad)
        self.assertFalse(self.path.exists())

    def test_import_rejects_duplicate_missing_self_and_cyclic_dependencies(self):
        invalid_batches = [
            [self._desc("a"), self._desc("a")],
            [self._desc("a", ["ghost"])],
            [self._desc("a", ["a"])],
            [self._desc("a", ["b"]), self._desc("b", ["a"])],
            [self._desc("a", ["b"]), self._desc("b", ["c"]), self._desc("c", ["a"])],
        ]
        for index, datasets in enumerate(invalid_batches):
            path = self.path.parent / ("bad-" + str(index) + ".json")
            with self.assertRaises(ValueError):
                DatasetCatalog(path).import_bundle({"datasets": datasets})
            self.assertFalse(path.exists())
        self.catalog.register(self._desc("base"))
        before = self.path.read_text(encoding="utf-8")
        for datasets in invalid_batches + [[self._desc("base")]]:
            with self.assertRaises(ValueError):
                self.catalog.import_bundle({"datasets": datasets})
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_import_applies_register_field_and_tag_rules_without_overwrite(self):
        self.catalog.register(self._desc("base"))
        before = self.path.read_text(encoding="utf-8")
        bad_batches = [
            [{"id": "new", "fields": [{"name": "n", "type": "float"}], "depends_on": []}],
            [{"id": "new", "fields": [{"name": "n", "type": "integer"},
                                      {"name": "n", "type": "string"}], "depends_on": []}],
            [{"id": "new", "fields": [{"name": "n", "type": "integer"}], "depends_on": ["x"]}],
            [{"id": "new", "fields": [{"name": "n", "type": "integer"}],
              "depends_on": ["base", "base"]}],
            [{"id": "new", "fields": [{"name": "n", "type": "integer"}], "depends_on": [],
              "tags": [""]}],
            [{"id": "new", "fields": [{"name": "n", "type": "integer"}], "depends_on": [],
              "tags": "x"}],
            [self._desc("base")],
            ["not-a-dict"],
        ]
        for datasets in bad_batches:
            with self.assertRaises(ValueError):
                self.catalog.import_bundle({"datasets": datasets})
            self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_import_failure_leaves_existing_file_byte_identical(self):
        self.catalog.register(self._desc("base"))
        before = self.path.read_text(encoding="utf-8")
        with self.assertRaises(ValueError):
            self.catalog.import_bundle({"datasets": [self._desc("a", ["ghost"])]})
        with self.assertRaises(ValueError):
            self.catalog.import_bundle({"datasets": [self._desc("a"), self._desc("a")]})
        with self.assertRaises(ValueError):
            self.catalog.import_bundle({"datasets": [self._desc("a", ["b"]),
                                                     self._desc("b", ["a"])]})
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        self.assertEqual(DatasetCatalog(self.path).describe("base")["id"], "base")

    def test_import_does_not_modify_input_object(self):
        bundle = {"datasets": [self._desc("b", ["a"]), self._desc("a")], "kept": True}
        snapshot = json.loads(json.dumps(bundle))
        self.catalog.import_bundle(bundle)
        self.assertEqual(bundle, snapshot)

    def test_import_roundtrip_via_export(self):
        sample = DatasetCatalog(ROOT / "samples" / "catalog.json")
        bundle = sample.export(["daily_totals"])
        result = self.catalog.import_bundle(json.loads(json.dumps(bundle)))
        self.assertEqual([row["id"] for row in result["datasets"]], ["orders", "daily_totals"])
        rebuilt = DatasetCatalog(self.path)
        original = DatasetCatalog(ROOT / "samples" / "catalog.json")
        for identifier in ("orders", "daily_totals"):
            self.assertEqual(rebuilt.describe(identifier), original.describe(identifier))
        self.assertEqual([row["id"] for row in rebuilt.dependencies("daily_totals")], ["orders"])

    def test_cli_import(self):
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", str(self.path)]
        bundle_path = Path(self.temp.name) / "bundle.json"
        bundle = {"datasets": [
            {"id": "daily_totals", "description": "Daily totals derived from orders",
             "fields": [{"name": "day", "type": "string"}, {"name": "total", "type": "number"}],
             "depends_on": ["orders"]},
            {"id": "orders", "description": "Order amounts from the daily export",
             "fields": [{"name": "order_id", "type": "string"},
                        {"name": "amount", "type": "number"}], "depends_on": []}]}
        bundle_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
        run = subprocess.run(prefix + ["import", str(bundle_path)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual([row["id"] for row in json.loads(run.stdout)["datasets"]],
                         ["orders", "daily_totals"])
        described = subprocess.run(prefix + ["describe", "daily_totals"],
                                   capture_output=True, text=True)
        self.assertEqual(json.loads(described.stdout)["depends_on"], ["orders"])

        missing = subprocess.run(prefix + ["import", str(self.path.parent / "nope.json")],
                                 capture_output=True, text=True)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("error", json.loads(missing.stdout))

        bad_json = Path(self.temp.name) / "bad.json"
        bad_json.write_text("{not json", encoding="utf-8")
        run = subprocess.run(prefix + ["import", str(bad_json)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("error", json.loads(run.stdout))

        invalid = Path(self.temp.name) / "invalid.json"
        invalid.write_text(json.dumps({"datasets": [bundle["datasets"][1]]}), encoding="utf-8")
        run = subprocess.run(prefix + ["import", str(invalid)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("error", json.loads(run.stdout))

        empty = Path(self.temp.name) / "empty.json"
        empty.write_text(json.dumps({"datasets": []}), encoding="utf-8")
        run = subprocess.run(prefix + ["import", str(empty)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0)
        self.assertEqual(json.loads(run.stdout), {"datasets": []})


if __name__ == "__main__":
    unittest.main()
