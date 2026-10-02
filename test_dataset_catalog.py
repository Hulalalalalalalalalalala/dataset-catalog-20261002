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

    def _bundle_record(self, identifier, depends_on=None, tags=None):
        record = {"id": identifier, "description": identifier + " desc",
                  "fields": [{"name": "n", "type": "integer"}], "depends_on": depends_on or []}
        if tags is not None:
            record["tags"] = tags
        return record

    def test_import_accepts_any_order_and_sorts_dependencies_first(self):
        bundle = {"datasets": [
            self._bundle_record("c", ["a", "b"]),
            self._bundle_record("b", ["a"]),
            self._bundle_record("a", [])],
            "ignored": 1}
        result = self.catalog.import_bundle(bundle)
        self.assertEqual([row["id"] for row in result["datasets"]], ["a", "b", "c"])
        self.assertEqual(self.catalog.import_bundle({"datasets": []}), {"datasets": []})
        for order in (["a", "b", "c"], ["c", "b", "a"], ["b", "c", "a"]):
            other = Path(self.temp.name) / ("catalog-" + "-".join(order) + ".json")
            catalog = DatasetCatalog(other)
            rows = [self._bundle_record(identifier, {"a": [], "b": ["a"], "c": ["a", "b"]}[identifier])
                    for identifier in order]
            got = catalog.import_bundle({"datasets": rows})
            self.assertEqual([row["id"] for row in got["datasets"]], ["a", "b", "c"])

    def test_import_tie_breaks_ready_candidates_by_id(self):
        result = self.catalog.import_bundle({"datasets": [
            self._bundle_record("d", ["b", "c"]),
            self._bundle_record("c", ["a"]),
            self._bundle_record("b", ["a"]),
            self._bundle_record("a", [])]})
        self.assertEqual([row["id"] for row in result["datasets"]], ["a", "b", "c", "d"])

    def test_import_returns_only_new_records_in_normalized_form(self):
        self.catalog.register(self._bundle_record("old"))
        with self.assertRaises(ValueError):
            self.catalog.import_bundle({"datasets": [
                self._bundle_record("solo", ["solo"])]})
        with self.assertRaises(ValueError):
            self.catalog.import_bundle({"datasets": [
                {"id": "dup", "fields": [{"name": "n", "type": "integer"}],
                 "depends_on": ["old", "old"]}]})
        result = self.catalog.import_bundle({"datasets": [
            {"id": "new", "description": 5,
             "fields": [{"name": " V ", "type": "integer", "extra": "x"}],
             "depends_on": ["old"], "tags": []}]})
        self.assertEqual([row["id"] for row in result["datasets"]], ["new"])
        self.assertEqual(result["datasets"][0],
                         {"id": "new", "description": "5",
                          "fields": [{"name": " V ", "type": "integer"}],
                          "depends_on": ["old"], "tags": []})

    def test_import_normalizes_like_register_and_keeps_tags_distinction(self):
        result = self.catalog.import_bundle({"datasets": [
            {"id": "tagged", "fields": [{"name": "n", "type": "integer"}],
             "tags": [" Finance ", "finance"]},
            {"id": "empty", "fields": [{"name": "n", "type": "integer"}], "tags": []},
            {"id": "plain", "fields": [{"name": "n", "type": "integer"}]}]})
        rows = {row["id"]: row for row in result["datasets"]}
        self.assertEqual(rows["tagged"]["tags"], ["Finance"])
        self.assertEqual(rows["empty"]["tags"], [])
        self.assertNotIn("tags", rows["plain"])
        fresh = DatasetCatalog(self.path)
        self.assertEqual(fresh.describe("tagged")["tags"], ["Finance"])
        self.assertEqual(fresh.describe("empty")["tags"], [])
        self.assertNotIn("tags", fresh.describe("plain"))

    def test_import_rejects_batch_conflicts_and_bad_graph_without_writing(self):
        self.catalog.register(self._bundle_record("old"))
        before = self.path.read_text(encoding="utf-8")
        good = self._bundle_record("new")
        bad_bundles = [
            {"datasets": [good, good]},                                   # duplicate id in batch
            {"datasets": [self._bundle_record("old")]},                   # conflicts with existing
            {"datasets": [self._bundle_record("new", ["ghost"])]},        # missing dependency
            {"datasets": [self._bundle_record("new", ["new"])]},          # self dependency
            {"datasets": [self._bundle_record("a", ["b"]),
                          self._bundle_record("b", ["a"])]},              # cycle
            {"datasets": [{"id": "new", "fields": [{"name": "n", "type": "float"}]}]},
            {"datasets": [{"id": "new", "fields": [{"name": "n", "type": "integer"}], "tags": [""]}]},
            {"datasets": [{"id": "new", "fields": [{"name": "n", "type": "integer"}],
                           "depends_on": ["old", "old"]}]},
        ]
        for bundle in bad_bundles:
            with self.assertRaises(ValueError):
                self.catalog.import_bundle(bundle)
            self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        with self.assertRaises(ValueError):
            self.catalog.import_bundle({"datasets": [
                self._bundle_record("old", []), self._bundle_record("old", [])]})
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_import_validates_bundle_shape_without_creating_file(self):
        nested = Path(self.temp.name) / "state" / "nested" / "catalog.json"
        catalog = DatasetCatalog(nested)
        for bad in ([], None, "x", 1, True, {}, {"datasets": None}, {"datasets": {}},
                    {"datasets": "x"}, {"datasets": 1}, {"other": []}):
            with self.assertRaises(ValueError):
                catalog.import_bundle(bad)
        self.assertFalse(nested.exists())
        self.assertFalse(nested.parent.exists())
        self.assertEqual(catalog.import_bundle({"datasets": []}), {"datasets": []})
        self.assertFalse(nested.exists())
        self.assertFalse(nested.parent.exists())

    def test_import_does_not_mutate_input_and_persists_for_all_queries(self):
        bundle = {"datasets": [
            {"id": "c", "description": "cycle",
             "fields": [{"name": "total", "type": "number"}], "depends_on": ["a", "b"],
             "tags": ["Finance"]},
            {"id": "b", "description": "", "fields": [{"name": "n", "type": "integer"}],
             "depends_on": ["a"]},
            {"id": "a", "description": "base", "fields": [{"name": "n", "type": "integer"}]}]}
        snapshot = json.loads(json.dumps(bundle, ensure_ascii=False))
        result = self.catalog.import_bundle(bundle)
        self.assertEqual(bundle, snapshot)
        self.assertEqual([row["id"] for row in result["datasets"]], ["a", "b", "c"])
        fresh = DatasetCatalog(self.path)
        self.assertEqual([row["id"] for row in fresh.dependencies("c")], ["a", "b"])
        self.assertEqual([row["dataset"]["id"] for row in fresh.impact("a")], ["b", "c"])
        self.assertEqual({row["id"] for row in fresh.export()["datasets"]}, {"a", "b", "c"})
        self.assertEqual([row["id"] for row in fresh.export(["c"])["datasets"]], ["a", "b", "c"])
        self.assertEqual([row["dataset"]["id"] for row in fresh.search(tags=["finance"])], ["c"])
        self.assertEqual([row["dataset"]["id"] for row in fresh.search("total")], ["c"])

    def test_import_same_bundle_twice_conflicts_and_leaves_catalog_intact(self):
        bundle = {"datasets": [self._bundle_record("a"), self._bundle_record("b", ["a"])]}
        self.catalog.import_bundle(bundle)
        before = self.path.read_text(encoding="utf-8")
        with self.assertRaises(ValueError):
            self.catalog.import_bundle(bundle)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        self.assertEqual([row["id"] for row in self.catalog.export()["datasets"]], ["a", "b"])

    def test_export_bundle_roundtrips_through_import(self):
        sample = DatasetCatalog(ROOT / "samples" / "catalog.json")
        bundle = sample.export(["daily_totals"])
        snapshot = json.loads(json.dumps(bundle, ensure_ascii=False))
        result = self.catalog.import_bundle(bundle)
        self.assertEqual(bundle, snapshot)
        self.assertEqual([row["id"] for row in result["datasets"]], ["orders", "daily_totals"])
        rebuilt = DatasetCatalog(self.path)
        original = DatasetCatalog(ROOT / "samples" / "catalog.json")
        for identifier in ("orders", "daily_totals"):
            self.assertEqual(rebuilt.describe(identifier), original.describe(identifier))
        self.assertEqual([row["id"] for row in rebuilt.dependencies("daily_totals")], ["orders"])

    def test_cli_import(self):
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", str(self.path)]
        sample = DatasetCatalog(ROOT / "samples" / "catalog.json")
        bundle_path = Path(self.temp.name) / "bundle.json"
        bundle_path.write_text(json.dumps(sample.export(["daily_totals"]), ensure_ascii=False),
                               encoding="utf-8")
        run = subprocess.run(prefix + ["import", str(bundle_path)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual([row["id"] for row in json.loads(run.stdout)["datasets"]],
                         ["orders", "daily_totals"])
        empty = Path(self.temp.name) / "empty.json"
        empty.write_text(json.dumps({"datasets": []}), encoding="utf-8")
        run = subprocess.run(prefix + ["import", str(empty)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0)
        self.assertEqual(json.loads(run.stdout), {"datasets": []})
        missing = subprocess.run(prefix + ["import", str(Path(self.temp.name) / "nope.json")],
                                 capture_output=True, text=True)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("error", json.loads(missing.stdout))
        bad_json = Path(self.temp.name) / "bad.json"
        bad_json.write_text("{not json", encoding="utf-8")
        run = subprocess.run(prefix + ["import", str(bad_json)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("error", json.loads(run.stdout))
        invalid = Path(self.temp.name) / "invalid.json"
        invalid.write_text(json.dumps({"datasets": [{"id": "x"}]}), encoding="utf-8")
        before = self.path.read_text(encoding="utf-8")
        run = subprocess.run(prefix + ["import", str(invalid)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("error", json.loads(run.stdout))
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)


    def _snapshot_record(self, identifier, depends_on=None, tags=..., description=...,
                         fields=None, owner=...):
        record = {"id": identifier,
                  "description": identifier + " desc" if description is ... else description,
                  "fields": fields or [{"name": "n", "type": "integer"}],
                  "depends_on": depends_on or []}
        if tags is not ... and tags is not None:
            record["tags"] = tags
        if owner is not ...:
            record["owner"] = owner
        return record

    def test_diff_lists_added_removed_and_changed_sorted_by_id(self):
        self.catalog.import_bundle({"datasets": [
            self._snapshot_record("a"), self._snapshot_record("b"),
            self._snapshot_record("c", ["b"])]})
        bundle = {"datasets": [
            self._snapshot_record("c", ["b"], description="updated"),
            self._snapshot_record("b"),
            self._snapshot_record("d", ["b"]),
            self._snapshot_record("e", ["d"])]}
        result = self.catalog.diff_bundle(bundle)
        self.assertEqual(set(result), {"added", "removed", "changed"})
        self.assertEqual([row["id"] for row in result["added"]], ["d", "e"])
        self.assertEqual([row["id"] for row in result["removed"]], ["a"])
        self.assertEqual([row["id"] for row in result["changed"]], ["c"])
        changed = result["changed"][0]
        self.assertEqual(changed["changed_keys"], ["description"])
        self.assertEqual(changed["before"]["description"], "c desc")
        self.assertEqual(changed["after"]["description"], "updated")
        self.assertEqual(changed["before"]["id"], changed["after"]["id"])

    def test_diff_empty_arrays_when_identical(self):
        bundle = {"datasets": [self._snapshot_record("a"), self._snapshot_record("b", ["a"])]}
        self.catalog.import_bundle(bundle)
        result = self.catalog.diff_bundle(bundle)
        self.assertEqual(result, {"added": [], "removed": [], "changed": []})

    def test_diff_empty_snapshot_removes_everything(self):
        self.catalog.import_bundle({"datasets": [self._snapshot_record("a"),
                                                 self._snapshot_record("b", ["a"])]})
        result = self.catalog.diff_bundle({"datasets": []})
        self.assertEqual(result["added"], [])
        self.assertEqual(result["changed"], [])
        self.assertEqual([row["id"] for row in result["removed"]], ["a", "b"])
        self.assertEqual(result["removed"][1],
                         {"id": "b", "description": "b desc",
                          "fields": [{"name": "n", "type": "integer"}], "depends_on": ["a"]})

    def test_diff_missing_or_empty_catalog_treats_snapshot_as_all_added(self):
        self.assertFalse(self.path.exists())
        result = self.catalog.diff_bundle({"datasets": [self._snapshot_record("a")]})
        self.assertEqual([row["id"] for row in result["added"]], ["a"])
        self.assertEqual(result["removed"], [])
        self.assertEqual(result["changed"], [])
        self.assertFalse(self.path.exists())
        self.path.write_text("{}", encoding="utf-8")
        result = DatasetCatalog(self.path).diff_bundle({"datasets": []})
        self.assertEqual(result, {"added": [], "removed": [], "changed": []})

    def test_diff_partial_snapshot_is_the_complete_after_state(self):
        self.catalog.import_bundle({"datasets": [
            self._snapshot_record("orders"), self._snapshot_record("daily", ["orders"]),
            self._snapshot_record("weekly", ["daily"])]})
        partial = self.catalog.export(["daily"])
        result = self.catalog.diff_bundle(partial)
        self.assertEqual([row["id"] for row in result["added"]], [])
        self.assertEqual([row["id"] for row in result["removed"]], ["weekly"])
        self.assertEqual(result["changed"], [])

    def test_diff_ignores_ordering_and_extra_data_but_detects_field_and_tag_changes(self):
        self.catalog.import_bundle({"datasets": [
            self._snapshot_record("a", ["b"]), self._snapshot_record("b")]})
        reordered = {"datasets": [
            {"id": "b", "description": "b desc", "extra_top": 1,
             "fields": [{"type": "integer", "name": "n", "ignored": True}], "depends_on": []},
            {"id": "a", "description": "a desc",
             "fields": [{"name": "n", "type": "integer"}], "depends_on": ["b", "b"][:1]}]}
        self.assertEqual(self.catalog.diff_bundle(reordered)["changed"], [])

        resorted_deps = {"datasets": [self._snapshot_record("b"),
                                      self._snapshot_record("a", ["b"])]}
        self.assertEqual(self.catalog.diff_bundle(resorted_deps)["changed"], [])

        field_order = {"datasets": [
            self._snapshot_record("b", fields=[{"name": "n", "type": "integer"}]),
            self._snapshot_record("a", ["b"],
                                  fields=[{"name": "z", "type": "integer"},
                                          {"name": "n", "type": "integer"}])]}
        result = self.catalog.diff_bundle(field_order)
        self.assertEqual(result["changed"][0]["changed_keys"], ["fields"])

    def test_diff_detects_tag_presence_spelling_and_order(self):
        cases = [
            (None, [], ["tags"]),
            ([], ["x"], ["tags"]),
            (["Finance"], ["finance"], ["tags"]),
            (["a", "b"], ["b", "a"], ["tags"]),
            ([" Finance ", "finance"], ["Finance"], []),
            (None, None, []),
        ]
        for index, (before_tags, after_tags, expected_keys) in enumerate(cases):
            other = Path(self.temp.name) / ("tags-" + str(index) + ".json")
            catalog = DatasetCatalog(other)
            catalog.register(self._snapshot_record("a", tags=before_tags))
            result = catalog.diff_bundle({"datasets": [self._snapshot_record("a", tags=after_tags)]})
            self.assertEqual([row["changed_keys"] for row in result["changed"]],
                             [expected_keys] if expected_keys else [])

    def test_diff_normalizes_both_sides_before_comparing(self):
        self.path.write_text(json.dumps({"a": {
            "id": "a", "description": 5, "depends_on": [],
            "fields": [{"name": "n", "type": "integer", "extra": 9}], "unknown": 123}}),
            encoding="utf-8")
        result = self.catalog.diff_bundle({"datasets": [
            {"depends_on": [], "fields": [{"type": "integer", "name": "n"}],
             "id": "a", "description": 5}]})
        self.assertEqual(result, {"added": [], "removed": [], "changed": []})

    def test_diff_changed_keys_sorted_and_records_are_full(self):
        self.catalog.register(self._snapshot_record("a"))
        bundle = {"datasets": [{
            "id": "a", "description": "new",
            "fields": [{"name": "n", "type": "string"}], "depends_on": [], "tags": ["x"]}]}
        changed = self.catalog.diff_bundle(bundle)["changed"][0]
        self.assertEqual(changed["changed_keys"], ["description", "fields", "tags"])
        self.assertEqual(set(changed["before"]), {"id", "description", "fields", "depends_on"})
        self.assertEqual(set(changed["after"]),
                         {"id", "description", "fields", "depends_on", "tags"})

    def test_diff_validates_snapshot_graph_without_partial_results(self):
        self.catalog.register(self._snapshot_record("old"))
        bad_bundles = [
            [], None, "x", 1, True, {},
            {"datasets": None}, {"datasets": {}}, {"datasets": "x"}, {"datasets": 1},
            {"other": []},
            {"datasets": [None]},
            {"datasets": ["x"]},
            {"datasets": [{"id": "bad id", "fields": [{"name": "n", "type": "integer"}]}]},
            {"datasets": [{"id": "a"}]},
            {"datasets": [{"id": "a", "fields": [{"name": "n", "type": "float"}]}]},
            {"datasets": [self._snapshot_record("a"), self._snapshot_record("a")]},
            {"datasets": [self._snapshot_record("a", ["a"])]},
            {"datasets": [self._snapshot_record("a", ["b", "b"])]},
            {"datasets": [self._snapshot_record("a", ["ghost"])]},
            {"datasets": [self._snapshot_record("a", ["b"]), self._snapshot_record("b", ["a"])]},
            {"datasets": [self._snapshot_record("a", tags=[""])]},
        ]
        for bundle in bad_bundles:
            with self.assertRaises(ValueError):
                self.catalog.diff_bundle(bundle)

    def test_diff_allows_forward_references_inside_snapshot(self):
        result = self.catalog.diff_bundle({"datasets": [
            self._snapshot_record("c", ["a", "b"]), self._snapshot_record("b", ["a"]),
            self._snapshot_record("a")]})
        self.assertEqual([row["id"] for row in result["added"]], ["a", "b", "c"])
        self.assertEqual(result["added"][2]["depends_on"], ["a", "b"])

    def test_diff_does_not_mutate_inputs_or_write_files(self):
        self.catalog.register(self._snapshot_record("a"))
        self.catalog.register(self._snapshot_record("b", ["a"]))
        catalog_before = self.path.read_text(encoding="utf-8")
        bundle = {"datasets": [
            self._snapshot_record("a"),
            self._snapshot_record("b", ["a"], description="changed",
                                  fields=[{"name": "n", "type": "integer"}]),
            self._snapshot_record("c", ["a"])], "ignored": True}
        snapshot = json.loads(json.dumps(bundle))
        result = self.catalog.diff_bundle(bundle)
        self.assertEqual(bundle, snapshot)
        self.assertEqual(self.path.read_text(encoding="utf-8"), catalog_before)
        result["added"][0]["description"] = "tampered"
        result["changed"][0]["after"]["description"] = "tampered"
        again = self.catalog.diff_bundle(bundle)
        self.assertEqual(again["added"][0]["description"], "c desc")
        self.assertEqual(again["changed"][0]["after"]["description"], "changed")

        nested = Path(self.temp.name) / "state" / "nested" / "catalog.json"
        empty_catalog = DatasetCatalog(nested)
        self.assertEqual(empty_catalog.diff_bundle({"datasets": []}),
                         {"added": [], "removed": [], "changed": []})
        self.assertFalse(nested.exists())
        self.assertFalse(nested.parent.exists())

    def test_cli_diff_success_and_errors(self):
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", str(self.path)]
        subprocess.run(prefix + ["register", str(ROOT / "samples/orders.json")],
                       check=True, capture_output=True)
        subprocess.run(prefix + ["register", str(ROOT / "samples/daily.json")],
                       check=True, capture_output=True)
        catalog_before = self.path.read_text(encoding="utf-8")

        snapshot_path = Path(self.temp.name) / "snapshot.json"
        snapshot_path.write_text(json.dumps({"datasets": [
            {"id": "orders", "description": "Order amounts from the daily export",
             "fields": [{"name": "order_id", "type": "string"},
                        {"name": "amount", "type": "number"}], "depends_on": []},
            {"id": "daily_totals", "description": "updated description",
             "fields": [{"name": "day", "type": "string"}, {"name": "total", "type": "number"}],
             "depends_on": ["orders"]},
            {"id": "report", "description": "new",
             "fields": [{"name": "id", "type": "string"}], "depends_on": ["orders"]}]},
            ensure_ascii=False), encoding="utf-8")
        run = subprocess.run(prefix + ["diff", str(snapshot_path)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual(set(result), {"added", "removed", "changed"})
        self.assertEqual([row["id"] for row in result["added"]], ["report"])
        self.assertEqual(result["removed"], [])
        self.assertEqual(result["changed"][0]["id"], "daily_totals")
        self.assertEqual(result["changed"][0]["changed_keys"], ["description"])
        self.assertEqual(self.path.read_text(encoding="utf-8"), catalog_before)

        empty = subprocess.run(prefix + ["diff", str(ROOT / "samples" / "orders.json")],
                               capture_output=True, text=True)
        self.assertEqual(empty.returncode, 2)
        self.assertIn("error", json.loads(empty.stdout))

        missing = subprocess.run(prefix + ["diff", str(Path(self.temp.name) / "nope.json")],
                                 capture_output=True, text=True)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("error", json.loads(missing.stdout))

        bad_json = Path(self.temp.name) / "bad.json"
        bad_json.write_text("{not json", encoding="utf-8")
        run = subprocess.run(prefix + ["diff", str(bad_json)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("error", json.loads(run.stdout))

        invalid = Path(self.temp.name) / "invalid.json"
        invalid.write_text(json.dumps({"datasets": [
            {"id": "a", "fields": [{"name": "n", "type": "integer"}], "depends_on": ["a"]}]}),
            encoding="utf-8")
        run = subprocess.run(prefix + ["diff", str(invalid)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("error", json.loads(run.stdout))
        self.assertEqual(self.path.read_text(encoding="utf-8"), catalog_before)

    def test_preview_diff_matches_diff_bundle_and_structure(self):
        self.catalog.import_bundle({"datasets": [
            self._snapshot_record("a"), self._snapshot_record("b", ["a"])]})
        bundle = {"datasets": [
            self._snapshot_record("a", description="updated"),
            self._snapshot_record("b", ["a"]), self._snapshot_record("c", ["b"])]}
        result = self.catalog.preview_bundle(bundle)
        self.assertEqual(set(result), {"diff", "affected"})
        self.assertEqual(result["diff"], self.catalog.diff_bundle(bundle))
        self.assertEqual([item["id"] for item in result["affected"]], ["b", "c"])
        b, c = result["affected"]
        self.assertEqual(set(b), {"id", "causes"})
        self.assertEqual(set(c), {"id", "causes"})
        self.assertEqual(b["causes"], [
            {"id": "a", "before": {"distance": 1, "path": ["a", "b"]},
             "after": {"distance": 1, "path": ["a", "b"]}}])
        self.assertEqual(c["causes"], [
            {"id": "a", "before": None,
             "after": {"distance": 2, "path": ["a", "b", "c"]}}])
        for item in result["affected"]:
            for cause in item["causes"]:
                self.assertEqual(set(cause), {"id", "before", "after"})
                for side in ("before", "after"):
                    if cause[side] is not None:
                        self.assertEqual(set(cause[side]), {"distance", "path"})

    def test_preview_tag_change_propagates_and_source_is_self_excluded(self):
        self.catalog.import_bundle({"datasets": [
            self._snapshot_record("a"), self._snapshot_record("b", ["a"]),
            self._snapshot_record("c", ["b"])]})
        result = self.catalog.preview_bundle({"datasets": [
            self._snapshot_record("a", tags=["x"]), self._snapshot_record("b", ["a"]),
            self._snapshot_record("c", ["b"])]})
        self.assertEqual([row["changed_keys"] for row in result["diff"]["changed"]], [["tags"]])
        self.assertEqual([item["id"] for item in result["affected"]], ["b", "c"])
        self.assertNotIn("a", [item["id"] for item in result["affected"]])

    def test_preview_empty_affected_when_no_diff(self):
        bundle = {"datasets": [self._snapshot_record("a"), self._snapshot_record("b", ["a"])]}
        self.catalog.import_bundle(bundle)
        self.assertEqual(self.catalog.preview_bundle(bundle)["affected"], [])

    def test_preview_removed_source_propagates_only_before(self):
        self.catalog.import_bundle({"datasets": [
            self._snapshot_record("a"), self._snapshot_record("b", ["a"]),
            self._snapshot_record("c", ["b"])]})
        result = self.catalog.preview_bundle({"datasets": []})
        affected = {item["id"]: {cause["id"]: cause for cause in item["causes"]}
                    for item in result["affected"]}
        self.assertEqual(sorted(affected), ["b", "c"])
        self.assertEqual(affected["b"]["a"]["before"], {"distance": 1, "path": ["a", "b"]})
        self.assertIsNone(affected["b"]["a"]["after"])
        self.assertEqual(affected["c"]["a"]["before"], {"distance": 2, "path": ["a", "b", "c"]})
        self.assertIsNone(affected["c"]["a"]["after"])
        self.assertEqual(affected["c"]["b"]["before"], {"distance": 1, "path": ["b", "c"]})
        self.assertEqual([cause["id"] for cause in result["affected"][1]["causes"]], ["a", "b"])

    def test_preview_added_source_propagates_only_after(self):
        result = self.catalog.preview_bundle({"datasets": [
            self._snapshot_record("a"), self._snapshot_record("b", ["a"])]})
        affected = {item["id"]: {cause["id"]: cause for cause in item["causes"]}
                    for item in result["affected"]}
        self.assertEqual(sorted(affected), ["b"])
        self.assertIsNone(affected["b"]["a"]["before"])
        self.assertEqual(affected["b"]["a"]["after"], {"distance": 1, "path": ["a", "b"]})
        self.assertFalse(self.path.exists())

    def test_preview_source_can_be_affected_by_other_sources(self):
        self.catalog.import_bundle({"datasets": [
            self._snapshot_record("a"), self._snapshot_record("b", ["a"])]})
        # both a (changed) and b (changed to depend on a in the same way still reachable)
        result = self.catalog.preview_bundle({"datasets": [
            self._snapshot_record("a", description="changed"),
            self._snapshot_record("b", ["a"], description="changed too")]})
        b = [item for item in result["affected"] if item["id"] == "b"]
        self.assertEqual(len(b), 1)
        self.assertEqual([cause["id"] for cause in b[0]["causes"]], ["a"])

    def test_preview_queries_each_side_independently(self):
        self.catalog.import_bundle({"datasets": [
            self._snapshot_record("a"), self._snapshot_record("b", ["a"]),
            self._snapshot_record("c", ["b"])]})
        result = self.catalog.preview_bundle({"datasets": [
            self._snapshot_record("a", description="changed"),
            self._snapshot_record("b", ["a"]),
            self._snapshot_record("c", ["a", "b"])]})
        cause = [item for item in result["affected"] if item["id"] == "c"][0]["causes"][0]
        self.assertEqual(cause["before"], {"distance": 2, "path": ["a", "b", "c"]})
        self.assertEqual(cause["after"], {"distance": 1, "path": ["a", "c"]})

    def test_preview_tie_breaks_equal_shortest_paths_by_id_sequence(self):
        self.catalog.import_bundle({"datasets": [
            self._snapshot_record("a"), self._snapshot_record("b", ["a"]),
            self._snapshot_record("c", ["a"]), self._snapshot_record("d", ["b", "c"])]})
        result = self.catalog.preview_bundle({"datasets": [
            self._snapshot_record("a", description="changed"),
            self._snapshot_record("b", ["a"]), self._snapshot_record("c", ["a"]),
            self._snapshot_record("d", ["b", "c"])]})
        self.assertEqual([item["id"] for item in result["affected"]], ["b", "c", "d"])
        cause = [item for item in result["affected"] if item["id"] == "d"][0]["causes"][0]
        self.assertEqual(cause["before"]["path"], ["a", "b", "d"])
        self.assertEqual(cause["after"]["path"], ["a", "b", "d"])

    def test_preview_deleting_a_leaf_source_affects_nothing(self):
        self.catalog.import_bundle({"datasets": [
            self._snapshot_record("a"), self._snapshot_record("b", ["a"])]})
        result = self.catalog.preview_bundle({"datasets": [self._snapshot_record("a")]})
        self.assertEqual([row["id"] for row in result["diff"]["removed"]], ["b"])
        self.assertEqual(result["affected"], [])

    def test_preview_validates_current_catalog_graph(self):
        self._snapshot_record("a")
        good = json.dumps({"datasets": [self._snapshot_record("a")]})
        bad_states = [
            json.dumps([self._snapshot_record("a")]),
            json.dumps({"a": "x"}),
            json.dumps({"a": self._snapshot_record("a", ["ghost"])}),
            json.dumps({"a": self._snapshot_record("a", ["a"])}),
            json.dumps({"a": self._snapshot_record("a", ["b", "b"]),
                        "b": self._snapshot_record("b")}),
            json.dumps({"a": {**self._snapshot_record("a"), "id": "other"}}),
            json.dumps({"a": self._snapshot_record("a", ["b"]),
                        "b": self._snapshot_record("b", ["a"])}),
        ]
        for raw in bad_states:
            other = Path(self.temp.name) / "bad-catalog.json"
            other.write_text(raw, encoding="utf-8")
            with self.assertRaises(ValueError):
                DatasetCatalog(other).preview_bundle(json.loads(good))

    def test_preview_rejects_invalid_snapshot_like_diff(self):
        self.catalog.register(self._snapshot_record("old"))
        for bundle in ([], None, "x", 1, True, {}, {"datasets": None}, {"datasets": "x"},
                       {"datasets": [self._snapshot_record("a"), self._snapshot_record("a")]},
                       {"datasets": [self._snapshot_record("a", ["a"])]},
                       {"datasets": [self._snapshot_record("a", ["ghost"])]},
                       {"datasets": [self._snapshot_record("a", ["b"]),
                                     self._snapshot_record("b", ["a"])]}):
            with self.assertRaises(ValueError):
                self.catalog.preview_bundle(bundle)

    def test_preview_does_not_mutate_inputs_or_write_files(self):
        self.catalog.import_bundle({"datasets": [
            self._snapshot_record("a"), self._snapshot_record("b", ["a"])]})
        catalog_before = self.path.read_text(encoding="utf-8")
        bundle = {"datasets": [
            self._snapshot_record("a", description="changed"),
            self._snapshot_record("b", ["a"]), self._snapshot_record("c", ["a"])]}
        snapshot = json.loads(json.dumps(bundle))
        result = self.catalog.preview_bundle(bundle)
        self.assertEqual(bundle, snapshot)
        self.assertEqual(self.path.read_text(encoding="utf-8"), catalog_before)
        result["affected"][0]["causes"][0]["before"]["path"] = ["tampered"]
        again = self.catalog.preview_bundle(bundle)
        self.assertEqual(again["affected"][0]["causes"][0]["before"]["path"], ["a", "b"])

        nested = Path(self.temp.name) / "state" / "nested" / "catalog.json"
        empty_catalog = DatasetCatalog(nested)
        self.assertEqual(empty_catalog.preview_bundle({"datasets": []}),
                         {"diff": {"added": [], "removed": [], "changed": []}, "affected": []})
        self.assertFalse(nested.exists())
        self.assertFalse(nested.parent.exists())

    def test_preview_is_independent_of_input_order(self):
        self.catalog.import_bundle({"datasets": [
            self._snapshot_record("a"), self._snapshot_record("b", ["a"]),
            self._snapshot_record("c", ["b"])]})
        descriptors = [self._snapshot_record("a", description="changed"),
                       self._snapshot_record("b", ["a"]), self._snapshot_record("c", ["b"])]
        first = self.catalog.preview_bundle({"datasets": descriptors})
        second = self.catalog.preview_bundle({"datasets": list(reversed(descriptors))})
        self.assertEqual(first, second)

    def test_cli_preview_success_and_errors(self):
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", str(self.path)]
        subprocess.run(prefix + ["register", str(ROOT / "samples/orders.json")],
                       check=True, capture_output=True)
        subprocess.run(prefix + ["register", str(ROOT / "samples/daily.json")],
                       check=True, capture_output=True)
        catalog_before = self.path.read_text(encoding="utf-8")

        snapshot_path = Path(self.temp.name) / "snapshot.json"
        snapshot_path.write_text(json.dumps({"datasets": [
            {"id": "orders", "description": "Order amounts from the daily export",
             "fields": [{"name": "order_id", "type": "string"},
                        {"name": "amount", "type": "number"}], "depends_on": []},
            {"id": "daily_totals", "description": "updated description",
             "fields": [{"name": "day", "type": "string"}, {"name": "total", "type": "number"}],
             "depends_on": ["orders"]}]}, ensure_ascii=False), encoding="utf-8")
        run = subprocess.run(prefix + ["preview", str(snapshot_path)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual(set(result), {"diff", "affected"})
        self.assertEqual(result["diff"]["changed"][0]["changed_keys"], ["description"])
        self.assertEqual(result["affected"], [])
        self.assertEqual(self.path.read_text(encoding="utf-8"), catalog_before)

        missing = subprocess.run(prefix + ["preview", str(Path(self.temp.name) / "nope.json")],
                                 capture_output=True, text=True)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("error", json.loads(missing.stdout))
        bad_json = Path(self.temp.name) / "bad.json"
        bad_json.write_text("{not json", encoding="utf-8")
        run = subprocess.run(prefix + ["preview", str(bad_json)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("error", json.loads(run.stdout))
        invalid = Path(self.temp.name) / "invalid.json"
        invalid.write_text(json.dumps({"datasets": [
            {"id": "a", "fields": [{"name": "n", "type": "integer"}], "depends_on": ["a"]}]}),
            encoding="utf-8")
        run = subprocess.run(prefix + ["preview", str(invalid)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("error", json.loads(run.stdout))
        self.assertEqual(self.path.read_text(encoding="utf-8"), catalog_before)


    def test_register_normalizes_and_persists_owner(self):
        entry = self.catalog.register({**self.raw, "owner": "  Jane  Q "})
        self.assertEqual(entry["owner"], "Jane  Q")
        fresh = DatasetCatalog(self.path)
        self.assertEqual(fresh.describe("orders")["owner"], "Jane  Q")
        plain = self.catalog.register({"id": "plain", "fields": [{"name": "n", "type": "integer"}]})
        self.assertNotIn("owner", plain)
        self.assertNotIn("owner", DatasetCatalog(self.path).describe("plain"))
        unicode = self.catalog.register({"id": "u", "fields": [{"name": "n", "type": "integer"}],
                                         "owner": "日 本　"})
        self.assertEqual(unicode["owner"], "日 本")

    def test_register_rejects_invalid_owner_without_writing(self):
        for bad in (None, 1, True, ["Jane"], {"name": "Jane"}, "", "   ", "\t \n"):
            with self.assertRaises(ValueError):
                self.catalog.register({**self.raw, "owner": bad})
        self.assertFalse(self.path.exists())
        self.catalog.register(self.raw)
        before = self.path.read_text(encoding="utf-8")
        with self.assertRaises(ValueError):
            self.catalog.register({"id": "other", "fields": [{"name": "n", "type": "integer"}],
                                   "owner": " "})
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def _owner_records(self):
        return {
            "a": {"id": "a", "description": "alpha report",
                  "fields": [{"name": "n", "type": "integer"}], "depends_on": [],
                  "tags": ["Finance"], "owner": "Jane Q"},
            "b": {"id": "b", "description": "beta",
                  "fields": [{"name": "label", "type": "string"}], "depends_on": [],
                  "owner": "JANE q"},
            "legacy": {"id": "legacy", "description": "no owner",
                       "fields": [{"name": "n", "type": "integer"}], "depends_on": []}}

    def test_search_owner_full_casefold_match(self):
        self._write_records(self._owner_records())
        self.assertEqual([row["dataset"]["id"] for row in self.catalog.search(owner="jane q")],
                         ["a", "b"])
        self.assertEqual([row["dataset"]["id"] for row in self.catalog.search(owner="  jane q  ")],
                         ["a", "b"])
        self.assertEqual([row["dataset"]["id"] for row in self.catalog.search(owner="Jane Q")],
                         ["a", "b"])
        self.assertEqual(self.catalog.search(owner="jane"), [])
        self.assertEqual(self.catalog.search(owner="jane  q"), [])
        self.assertEqual(self.catalog.search(owner="jane q extra"), [])
        self.assertEqual(self.catalog.search(owner="nobody"), [])
        self.assertEqual([row["dataset"]["id"] for row in self.catalog.search(owner=None)],
                         ["a", "b", "legacy"])

    def test_search_owner_combines_with_other_conditions(self):
        self._write_records(self._owner_records())
        result = self.catalog.search("alpha", owner="jane q")
        self.assertEqual([row["dataset"]["id"] for row in result], ["a"])
        result = self.catalog.search("beta", owner="jane q")
        self.assertEqual([row["dataset"]["id"] for row in result], ["b"])
        result = self.catalog.search(field_type="string", owner="jane q")
        self.assertEqual([row["dataset"]["id"] for row in result], ["b"])
        result = self.catalog.search(tags=["finance"], owner="jane q")
        self.assertEqual([row["dataset"]["id"] for row in result], ["a"])
        result = self.catalog.search("alpha", field_type="integer", tags=["finance"], owner="JANE Q")
        self.assertEqual([row["dataset"]["id"] for row in result], ["a"])
        result = self.catalog.search("alpha", owner="nobody")
        self.assertEqual(result, [])
        # owner text is not part of the keyword scope or matched_fields
        result = self.catalog.search("jane")
        self.assertEqual(result, [])
        result = self.catalog.search("alpha", owner="jane q")
        self.assertEqual(result[0]["matched_fields"], [])

    def test_search_validates_owner_before_reading_catalog(self):
        self.assertFalse(self.path.exists())
        for bad in ("", "   ", 1, True, ["jane"], 0):
            with self.assertRaises(ValueError):
                self.catalog.search(owner=bad)
        self.assertEqual(self.catalog.search(owner="jane q"), [])
        self.assertFalse(self.path.exists())

    def test_search_owner_does_not_modify_catalog(self):
        self._write_records(self._owner_records())
        before = self.path.read_text(encoding="utf-8")
        self.catalog.search(owner="jane q")
        self.catalog.search(owner="nobody")
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        self.assertNotIn("owner", self.catalog.describe("legacy"))

    def test_owner_survives_export_and_import_roundtrip(self):
        self._write_records(self._owner_records())
        bundle = self.catalog.export()
        other_path = self.path.parent / "rebuilt.json"
        rebuilt = DatasetCatalog(other_path)
        result = rebuilt.import_bundle(bundle)
        saved = {row["id"]: row for row in result["datasets"]}
        self.assertEqual(saved["a"]["owner"], "Jane Q")
        self.assertEqual(saved["b"]["owner"], "JANE q")
        self.assertNotIn("owner", saved["legacy"])
        for identifier in ("a", "b", "legacy"):
            self.assertEqual(rebuilt.describe(identifier), self.catalog.describe(identifier))

    def test_import_rejects_invalid_owner_batch_without_writing(self):
        good = {"id": "ok", "fields": [{"name": "n", "type": "integer"}]}
        bad_bundles = [
            {"datasets": [good, {**good, "id": "bad", "owner": ""}]},
            {"datasets": [{**good, "id": "bad", "owner": None}]},
            {"datasets": [{**good, "id": "bad", "owner": 1}]},
            {"datasets": [{**good, "id": "bad", "owner": "   "}]},
        ]
        for bundle in bad_bundles:
            with self.assertRaises(ValueError):
                self.catalog.import_bundle(bundle)
        self.assertFalse(self.path.exists())
        snapshot = {"datasets": [{**good, "id": "x", "owner": " A "}]}
        snap = json.loads(json.dumps(snapshot))
        result = self.catalog.import_bundle(snapshot)
        self.assertEqual(result["datasets"][0]["owner"], "A")
        self.assertEqual(snapshot, snap)

    def test_diff_detects_owner_changes_with_whitespace_normalization(self):
        self.catalog.register({"id": "a", "fields": [{"name": "n", "type": "integer"}],
                               "owner": " Alice "})
        same = {"datasets": [{"id": "a", "fields": [{"name": "n", "type": "integer"}],
                              "owner": "Alice"}]}
        self.assertEqual(self.catalog.diff_bundle(same)["changed"], [])
        for after_owner, expected in ((None, ["owner"]), ("Bob", ["owner"]),
                                      ("alice", ["owner"])):
            record = {"id": "a", "fields": [{"name": "n", "type": "integer"}]}
            if after_owner is not None:
                record["owner"] = after_owner
            changed = self.catalog.diff_bundle({"datasets": [record]})["changed"]
            self.assertEqual([row["changed_keys"] for row in changed], [expected])
        # adding an owner where none existed
        plain = DatasetCatalog(self.path.parent / "plain.json")
        plain.register({"id": "a", "fields": [{"name": "n", "type": "integer"}]})
        changed = plain.diff_bundle(same)["changed"]
        self.assertEqual([row["changed_keys"] for row in changed], [["owner"]])
        self.assertNotIn("owner", changed[0]["before"])
        self.assertEqual(changed[0]["after"]["owner"], "Alice")

    def test_diff_and_preview_reject_invalid_owner(self):
        self.catalog.register({"id": "a", "fields": [{"name": "n", "type": "integer"}]})
        for bad in (None, "", "   ", 1, ["x"]):
            record = {"id": "a", "fields": [{"name": "n", "type": "integer"}], "owner": bad}
            with self.assertRaises(ValueError):
                self.catalog.diff_bundle({"datasets": [record]})
            with self.assertRaises(ValueError):
                self.catalog.preview_bundle({"datasets": [record]})

    def test_preview_owner_only_change_propagates(self):
        self.catalog.import_bundle({"datasets": [
            self._snapshot_record("a"), self._snapshot_record("b", ["a"]),
            self._snapshot_record("c", ["b"])]})
        result = self.catalog.preview_bundle({"datasets": [
            self._snapshot_record("a", owner="Alice"), self._snapshot_record("b", ["a"]),
            self._snapshot_record("c", ["b"])]})
        self.assertEqual([row["changed_keys"] for row in result["diff"]["changed"]], [["owner"]])
        self.assertEqual([item["id"] for item in result["affected"]], ["b", "c"])
        self.assertEqual(result["affected"][0]["causes"][0]["id"], "a")

    def test_cli_search_owner_filter(self):
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", str(self.path)]
        first = {"id": "a", "description": "alpha",
                 "fields": [{"name": "n", "type": "integer"}], "owner": "Jane Q"}
        second = {"id": "b", "description": "beta",
                  "fields": [{"name": "n", "type": "integer"}]}
        for row in (first, second):
            descriptor = Path(self.temp.name) / (row["id"] + ".json")
            descriptor.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            run = subprocess.run(prefix + ["register", str(descriptor)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
        run = subprocess.run(prefix + ["search", "--owner", "jane q"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual([row["dataset"]["id"] for row in json.loads(run.stdout)], ["a"])
        run = subprocess.run(prefix + ["search", "alpha", "--owner", "JANE Q"],
                             capture_output=True, text=True)
        self.assertEqual([row["dataset"]["id"] for row in json.loads(run.stdout)], ["a"])
        run = subprocess.run(prefix + ["search", "--owner", "jane"],
                             capture_output=True, text=True)
        self.assertEqual(json.loads(run.stdout), [])
        run = subprocess.run(prefix + ["search"], capture_output=True, text=True)
        self.assertEqual([row["dataset"]["id"] for row in json.loads(run.stdout)], ["a", "b"])
        bad = subprocess.run(prefix + ["search", "--owner", "   "],
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        self.assertIn("error", json.loads(bad.stdout))
        described = subprocess.run(prefix + ["describe", "a"], capture_output=True, text=True)
        self.assertEqual(json.loads(described.stdout)["owner"], "Jane Q")


if __name__ == "__main__":
    unittest.main()
