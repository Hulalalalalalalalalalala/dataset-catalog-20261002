import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
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

    def test_upstream_lists_direct_and_indirect_with_shortest_paths(self):
        for identifier, depends_on in (("orders", []), ("daily", ["orders"]),
                                       ("weekly", ["daily"]), ("report", ["orders"])):
            self._register(identifier, depends_on)
        result = self.catalog.upstream("weekly")
        self.assertEqual([(row["dataset"]["id"], row["distance"], row["path"]) for row in result],
                         [("daily", 1, ["weekly", "daily"]),
                          ("orders", 2, ["weekly", "daily", "orders"])])
        self.assertEqual(result[0]["dataset"], self.catalog.describe("daily"))
        self.assertEqual([set(row) for row in result],
                         [{"dataset", "distance", "path"}] * 2)

    def test_upstream_diamond_picks_lexicographically_smallest_shortest_path(self):
        for identifier, depends_on in (("base", []), ("left", ["base"]), ("right", ["base"]),
                                       ("report", ["left", "right"])):
            self._register(identifier, depends_on)
        result = self.catalog.upstream("report")
        self.assertEqual([(row["dataset"]["id"], row["distance"]) for row in result],
                         [("left", 1), ("right", 1), ("base", 2)])
        self.assertEqual(result[-1]["path"], ["report", "left", "base"])

    def test_upstream_respects_max_depth_and_empty_root(self):
        for identifier, depends_on in (("orders", []), ("daily", ["orders"]),
                                       ("weekly", ["daily"])):
            self._register(identifier, depends_on)
        self.assertEqual([row["dataset"]["id"] for row in self.catalog.upstream("weekly", 1)],
                         ["daily"])
        self.assertEqual(self.catalog.upstream("weekly", None),
                         self.catalog.upstream("weekly"))
        self.assertEqual(self.catalog.upstream("orders"), [])

    def test_upstream_rejects_invalid_arguments_before_reading_catalog(self):
        self._register("orders")
        for bad in (True, False, 0, -1, 1.0, "1"):
            with self.assertRaises(ValueError):
                self.catalog.upstream("orders", bad)
        for bad in ("", "Orders", "ord ers", 1, None, True):
            with self.assertRaises(ValueError):
                self.catalog.upstream(bad)
        with self.assertRaises(ValueError):
            self.catalog.upstream("missing")
        missing = DatasetCatalog(Path(self.temp.name) / "absent" / "catalog.json")
        for bad in ("Orders", 1):
            with self.assertRaises(ValueError):
                missing.upstream(bad)
        with self.assertRaises(ValueError):
            missing.upstream("orders", 0)
        self.assertFalse(missing.path.exists())

    def test_upstream_validates_full_closure_beyond_depth_limit(self):
        self._write_records({
            "top": {"id": "top", "fields": [{"name": "n", "type": "integer"}],
                    "depends_on": ["mid"]},
            "mid": {"id": "mid", "fields": [{"name": "n", "type": "integer"}],
                    "depends_on": ["broken"]},
            "broken": {"id": "broken", "fields": [{"name": "n", "type": "integer"}],
                       "depends_on": ["missing"]},
        })
        for depth in (None, 1, 2):
            with self.assertRaises(ValueError):
                self.catalog.upstream("top", depth)

    def test_upstream_ignores_records_outside_the_closure(self):
        self._write_records({
            "base": {"id": "base", "fields": [{"name": "n", "type": "integer"}]},
            "top": {"id": "top", "fields": [{"name": "n", "type": "integer"}],
                    "depends_on": ["base"]},
            "corrupt": "not a descriptor",
            "cyclic": {"id": "cyclic", "fields": [{"name": "n", "type": "integer"}],
                       "depends_on": ["cyclic"]},
        })
        result = self.catalog.upstream("top")
        self.assertEqual([(row["dataset"]["id"], row["distance"], row["path"]) for row in result],
                         [("base", 1, ["top", "base"])])

    def test_upstream_rejects_invalid_catalog_states(self):
        bad_states = (
            ["not", "an", "object"],
            {"top": {"id": "other", "fields": [{"name": "n", "type": "integer"}]}},
            {"top": {"id": "top", "fields": [{"name": "n", "type": "integer"}],
                     "depends_on": ["top"]}},
            {"top": {"id": "top", "fields": [{"name": "n", "type": "integer"}],
                     "depends_on": ["mid", "mid"]},
             "mid": {"id": "mid", "fields": [{"name": "n", "type": "integer"}]}},
            {"top": {"id": "top", "fields": [{"name": "n", "type": "integer"}],
                     "depends_on": ["mid"]},
             "mid": {"id": "mid", "fields": [{"name": "n", "type": "integer"}],
                     "depends_on": ["top"]}},
        )
        for state in bad_states:
            self._write_records(state)
            with self.assertRaises(ValueError):
                self.catalog.upstream("top")
        self._write_records({"top": {"id": "top", "fields": [{"name": "n", "type": "integer"}],
                                     "depends_on": ["missing"]}})
        with self.assertRaises(ValueError):
            self.catalog.upstream("top")

    def test_upstream_is_independent_of_input_order_and_read_only(self):
        self._write_records({
            "report": {"id": "report", "fields": [{"name": "n", "type": "integer"}],
                       "depends_on": ["right", "left"]},
            "right": {"id": "right", "fields": [{"name": "n", "type": "integer"}],
                      "depends_on": ["base"]},
            "base": {"id": "base", "fields": [{"name": "n", "type": "integer"}]},
            "left": {"id": "left", "fields": [{"name": "n", "type": "integer"}],
                     "depends_on": ["base"]},
        })
        before = self.path.read_text(encoding="utf-8")
        result = self.catalog.upstream("report")
        self.assertEqual([(row["dataset"]["id"], row["distance"], row["path"]) for row in result],
                         [("left", 1, ["report", "left"]),
                          ("right", 1, ["report", "right"]),
                          ("base", 2, ["report", "left", "base"])])
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_cli_upstream(self):
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", str(self.path)]
        subprocess.run(prefix + ["register", str(ROOT / "samples/orders.json")], check=True, capture_output=True)
        subprocess.run(prefix + ["register", str(ROOT / "samples/daily.json")], check=True, capture_output=True)
        ok = subprocess.run(prefix + ["upstream", "daily_totals"], capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        result = json.loads(ok.stdout)
        self.assertEqual([row["dataset"]["id"] for row in result], ["orders"])
        self.assertEqual(result[0]["distance"], 1)
        self.assertEqual(result[0]["path"], ["daily_totals", "orders"])
        empty = subprocess.run(prefix + ["upstream", "orders"], capture_output=True, text=True)
        self.assertEqual(empty.returncode, 0, empty.stderr)
        self.assertEqual(json.loads(empty.stdout), [])
        limited = subprocess.run(prefix + ["upstream", "daily_totals", "--max-depth", "1"],
                                 capture_output=True, text=True)
        self.assertEqual(limited.returncode, 0, limited.stderr)
        self.assertEqual([row["dataset"]["id"] for row in json.loads(limited.stdout)], ["orders"])
        for bad_args in (["upstream", "daily_totals", "--max-depth", "0"],
                         ["upstream", "daily_totals", "--max-depth", "x"],
                         ["upstream", "daily_totals", "--max-depth", "1.5"],
                         ["upstream", "missing"],
                         ["upstream", "Orders"]):
            run = subprocess.run(prefix + bad_args, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, bad_args)
            self.assertEqual(set(json.loads(run.stdout)), {"error"}, bad_args)

    def test_common_upstream_sample_orders_and_daily_totals(self):
        result = DatasetCatalog(ROOT / "samples" / "catalog.json").common_upstream(
            ["orders", "daily_totals"])
        self.assertEqual([row["dataset"]["id"] for row in result], ["orders"])
        self.assertEqual(result[0]["dataset"],
                         DatasetCatalog(ROOT / "samples" / "catalog.json").describe("orders"))
        self.assertEqual(result[0]["targets"], [
            {"id": "daily_totals", "distance": 1, "path": ["daily_totals", "orders"]},
            {"id": "orders", "distance": 0, "path": ["orders"]}])
        self.assertEqual(set(result[0]), {"dataset", "targets"})
        for target in result[0]["targets"]:
            self.assertEqual(set(target), {"id", "distance", "path"})

    def test_common_upstream_merges_duplicates_and_ignores_input_order(self):
        self._write_records({
            "base": {"id": "base", "fields": [{"name": "n", "type": "integer"}]},
            "a": {"id": "a", "fields": [{"name": "n", "type": "integer"}],
                  "depends_on": ["base"]},
            "b": {"id": "b", "fields": [{"name": "n", "type": "integer"}],
                  "depends_on": ["base"]}})
        first = self.catalog.common_upstream(["b", "a", "b"])
        second = self.catalog.common_upstream(["a", "b"])
        self.assertEqual(first, second)
        self.assertEqual([(row["dataset"]["id"],
                           [(t["id"], t["distance"], t["path"]) for t in row["targets"]])
                          for row in first],
                         [("base", [("a", 1, ["a", "base"]), ("b", 1, ["b", "base"])])])

    def test_common_upstream_keeps_independent_sources_and_drops_transitive_ones(self):
        self._write_records({
            "x": self._record("x", []), "z": self._record("z", []),
            "m": self._record("m", ["z"]), "y": self._record("y", ["m"]),
            "l": self._record("l", ["x", "y"]), "r": self._record("r", ["x", "y"])})
        result = self.catalog.common_upstream(["l", "r"])
        self.assertEqual([row["dataset"]["id"] for row in result], ["x", "y"])
        y = [row for row in result if row["dataset"]["id"] == "y"][0]
        self.assertEqual([(t["id"], t["distance"]) for t in y["targets"]],
                         [("l", 1), ("r", 1)])

    def test_common_upstream_no_shared_source_returns_empty(self):
        self._write_records({
            "a": self._record("a", []), "b": self._record("b", []),
            "u": self._record("u", ["a"]), "v": self._record("v", ["b"])})
        self.assertEqual(self.catalog.common_upstream(["u", "v"]), [])
        self.assertEqual(self.catalog.common_upstream(["a", "b"]), [])

    def test_common_upstream_target_can_be_a_source_at_distance_zero(self):
        self._write_records({
            "s": self._record("s", []),
            "a": self._record("a", ["s"]), "b": self._record("b", ["s"])})
        result = self.catalog.common_upstream(["s", "a", "b"])
        self.assertEqual([row["dataset"]["id"] for row in result], ["s"])
        self.assertEqual([(t["id"], t["distance"], t["path"]) for t in result[0]["targets"]],
                         [("a", 1, ["a", "s"]), ("b", 1, ["b", "s"]),
                          ("s", 0, ["s"])])

    def test_common_upstream_picks_lexicographically_smallest_shortest_path(self):
        self._write_records({
            "w": self._record("w", []),
            "aaa": self._record("aaa", ["w"]), "bbb": self._record("bbb", ["w"]),
            "t1": self._record("t1", ["aaa", "bbb"]),
            "t2": self._record("t2", ["bbb", "aaa"])})
        result = self.catalog.common_upstream(["t1", "t2"])
        self.assertEqual([row["dataset"]["id"] for row in result], ["aaa", "bbb"])
        self.assertEqual([t["path"] for t in result[0]["targets"]],
                         [["t1", "aaa"], ["t2", "aaa"]])
        self.assertEqual([t["path"] for t in result[1]["targets"]],
                         [["t1", "bbb"], ["t2", "bbb"]])

    def test_common_upstream_validates_arguments_before_reading_catalog(self):
        self.assertFalse(self.path.exists())
        missing = DatasetCatalog(self.path)
        for bad in ("x", None, True, 1, ("a", "b"), {1, 2}):
            with self.assertRaises(ValueError):
                missing.common_upstream(bad)
        for bad in ([""], ["Orders"], ["a b"], [1], [None], [True], ["a", ""],
                    ["a", "Orders"], ["a", 1]):
            with self.assertRaises(ValueError):
                missing.common_upstream(bad)
        with self.assertRaises(ValueError):
            missing.common_upstream(["a"])
        with self.assertRaises(ValueError):
            missing.common_upstream(["a", "a"])
        self.assertFalse(self.path.exists())
        # registered ids are still unknown in an empty catalog and never create it
        with self.assertRaises(ValueError):
            missing.common_upstream(["a", "b"])
        self.assertFalse(self.path.exists())

    def test_common_upstream_validates_each_target_closure(self):
        self._write_records({
            "good": {"id": "good", "fields": [{"name": "n", "type": "integer"}]},
            "top": {"id": "top", "fields": [{"name": "n", "type": "integer"}],
                    "depends_on": ["broken"]},
            "broken": {"id": "broken", "fields": [{"name": "n", "type": "integer"}],
                       "depends_on": ["missing"]}})
        with self.assertRaises(ValueError):
            self.catalog.common_upstream(["top", "good"])

    def test_common_upstream_ignores_records_outside_target_closures(self):
        self._write_records({
            "a": {"id": "a", "fields": [{"name": "n", "type": "integer"}]},
            "b": {"id": "b", "fields": [{"name": "n", "type": "integer"}]},
            "corrupt": "not a descriptor",
            "cyclic": {"id": "cyclic", "fields": [{"name": "n", "type": "integer"}],
                       "depends_on": ["cyclic"]}})
        self.assertEqual(self.catalog.common_upstream(["a", "b"]), [])

    def test_common_upstream_rejects_invalid_catalog_states(self):
        bad_states = (
            ["not", "an", "object"],
            {"top": {"id": "other", "fields": [{"name": "n", "type": "integer"}]},
             "ok": {"id": "ok", "fields": [{"name": "n", "type": "integer"}]}},
            {"top": {"id": "top", "fields": [{"name": "n", "type": "integer"}],
                     "depends_on": ["top"]},
             "ok": {"id": "ok", "fields": [{"name": "n", "type": "integer"}]}},
            {"top": {"id": "top", "fields": [{"name": "n", "type": "integer"}],
                     "depends_on": ["mid", "mid"]},
             "mid": {"id": "mid", "fields": [{"name": "n", "type": "integer"}]},
             "ok": {"id": "ok", "fields": [{"name": "n", "type": "integer"}]}},
            {"top": {"id": "top", "fields": [{"name": "n", "type": "integer"}],
                     "depends_on": ["mid"]},
             "mid": {"id": "mid", "fields": [{"name": "n", "type": "integer"}],
                     "depends_on": ["top"]},
             "ok": {"id": "ok", "fields": [{"name": "n", "type": "integer"}]}},
        )
        for state in bad_states:
            self._write_records(state)
            with self.assertRaises(ValueError):
                self.catalog.common_upstream(["top", "ok"])

    def test_common_upstream_is_read_only(self):
        self._write_records({
            "base": {"id": "base", "fields": [{"name": "n", "type": "integer"}]},
            "a": {"id": "a", "fields": [{"name": "n", "type": "integer"}],
                  "depends_on": ["base"]},
            "b": {"id": "b", "fields": [{"name": "n", "type": "integer"}],
                  "depends_on": ["base"]}})
        before = self.path.read_text(encoding="utf-8")
        self.catalog.common_upstream(["a", "b"])
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_cli_common_upstream(self):
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", str(self.path)]
        subprocess.run(prefix + ["register", str(ROOT / "samples/orders.json")],
                       check=True, capture_output=True)
        subprocess.run(prefix + ["register", str(ROOT / "samples/daily.json")],
                       check=True, capture_output=True)
        before = self.path.read_text(encoding="utf-8")
        run = subprocess.run(prefix + ["common-upstream", "--id", "orders",
                                       "--id", "daily_totals"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual([row["dataset"]["id"] for row in result], ["orders"])
        self.assertEqual([(t["id"], t["distance"], t["path"]) for t in result[0]["targets"]],
                         [("daily_totals", 1, ["daily_totals", "orders"]),
                          ("orders", 0, ["orders"])])
        # duplicate and reversed input produce the same JSON
        reversed_run = subprocess.run(prefix + ["common-upstream", "--id", "daily_totals",
                                               "--id", "orders", "--id", "daily_totals"],
                                     capture_output=True, text=True)
        self.assertEqual(json.loads(reversed_run.stdout), result)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        for bad_args in (["common-upstream"],
                         ["common-upstream", "--id", "orders"],
                         ["common-upstream", "--id", "orders", "--id", "nope"],
                         ["common-upstream", "--id", "Bad", "--id", "orders"],
                         ["common-upstream", "--id", "orders", "--id", "orders"]):
            failed = subprocess.run(prefix + bad_args, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 2, bad_args)
            self.assertEqual(set(json.loads(failed.stdout)), {"error"}, bad_args)

    def test_cli_common_upstream_uses_default_catalog(self):
        base = [sys.executable, str(ROOT / "dataset_catalog.py")]
        run = subprocess.run(base + ["common-upstream", "--id", "orders",
                                     "--id", "daily_totals"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual([row["dataset"]["id"] for row in json.loads(run.stdout)], ["orders"])

    def test_between_collects_every_path_node_and_edge(self):
        for identifier, depends_on in (("orders", []), ("left", ["orders"]), ("right", ["orders"]),
                                       ("daily", ["left", "right"]), ("side", ["orders"]),
                                       ("island", [])):
            self._register(identifier, depends_on)
        result = self.catalog.between("orders", "daily")
        self.assertEqual(set(result), {"datasets", "edges"})
        self.assertEqual([row["id"] for row in result["datasets"]],
                         ["daily", "left", "orders", "right"])
        self.assertEqual(result["datasets"][0], self.catalog.describe("daily"))
        self.assertEqual(result["edges"], [{"source": "left", "target": "daily"},
                                           {"source": "orders", "target": "left"},
                                           {"source": "orders", "target": "right"},
                                           {"source": "right", "target": "daily"}])

    def test_between_same_endpoints_and_unreachable_pairs(self):
        for identifier, depends_on in (("orders", []), ("daily", ["orders"]), ("island", [])):
            self._register(identifier, depends_on)
        same = self.catalog.between("orders", "orders")
        self.assertEqual([row["id"] for row in same["datasets"]], ["orders"])
        self.assertEqual(same["edges"], [])
        empty = {"datasets": [], "edges": []}
        self.assertEqual(self.catalog.between("daily", "orders"), empty)
        self.assertEqual(self.catalog.between("orders", "island"), empty)

    def test_between_validates_arguments_before_reading_catalog(self):
        for bad in ("Orders", "", "1abc", 1, None, ["orders"]):
            with self.assertRaises(ValueError):
                self.catalog.between(bad, "orders")
            with self.assertRaises(ValueError):
                self.catalog.between("orders", bad)
        self.assertFalse(self.path.exists())
        with self.assertRaises(ValueError):
            self.catalog.between("orders", "daily")
        self.path.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.catalog.between("orders", "daily")

    def test_between_validates_whole_catalog_graph(self):
        self._register("orders")
        self._register("daily", ["orders"])
        records = json.loads(self.path.read_text(encoding="utf-8"))
        records["broken"] = {"id": "other", "fields": [{"name": "n", "type": "integer"}],
                             "depends_on": []}
        self.path.write_text(json.dumps(records), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.catalog.between("orders", "daily")

    def test_between_is_read_only(self):
        self._register("orders")
        self._register("daily", ["orders"])
        before = self.path.read_text(encoding="utf-8")
        self.catalog.between("orders", "daily")
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_cli_between(self):
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", str(self.path)]
        subprocess.run(prefix + ["register", str(ROOT / "samples/orders.json")],
                       check=True, capture_output=True)
        subprocess.run(prefix + ["register", str(ROOT / "samples/daily.json")],
                       check=True, capture_output=True)
        ok = subprocess.run(prefix + ["between", "orders", "daily_totals"],
                            capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        result = json.loads(ok.stdout)
        self.assertEqual([row["id"] for row in result["datasets"]], ["daily_totals", "orders"])
        self.assertEqual(result["edges"], [{"source": "orders", "target": "daily_totals"}])
        bad = subprocess.run(prefix + ["between", "orders", "missing"],
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        self.assertEqual(set(json.loads(bad.stdout)), {"error"})

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
                         fields=None):
        record = {"id": identifier,
                  "description": identifier + " desc" if description is ... else description,
                  "fields": fields or [{"name": "n", "type": "integer"}],
                  "depends_on": depends_on or []}
        if tags is not ... and tags is not None:
            record["tags"] = tags
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

    def test_apply_commits_snapshot_and_returns_pre_commit_diff(self):
        self.catalog.import_bundle({"datasets": [
            self._snapshot_record("a"), self._snapshot_record("b"),
            self._snapshot_record("c", ["b"])]})
        bundle = {"datasets": [
            self._snapshot_record("c", ["b"], description="updated"),
            self._snapshot_record("b"),
            self._snapshot_record("d", ["b"])]}
        expected = self.catalog.diff_bundle(bundle)
        result = self.catalog.apply_bundle(bundle)
        self.assertEqual(result, expected)
        self.assertEqual([row["id"] for row in result["added"]], ["d"])
        self.assertEqual([row["id"] for row in result["removed"]], ["a"])
        self.assertEqual([row["id"] for row in result["changed"]], ["c"])
        self.assertEqual(result["changed"][0]["changed_keys"], ["description"])
        fresh = DatasetCatalog(self.path)
        self.assertEqual([row["id"] for row in fresh.export()["datasets"]], ["b", "c", "d"])
        self.assertEqual(fresh.describe("c")["description"], "updated")
        self.assertEqual(self.catalog.describe("c")["description"], "updated")
        with self.assertRaises(ValueError):
            fresh.describe("a")
        self.assertEqual(self.catalog.diff_bundle(bundle),
                         {"added": [], "removed": [], "changed": []})

    def test_apply_normalizes_and_allows_forward_references(self):
        result = self.catalog.apply_bundle({"datasets": [
            {"id": "c", "description": 7, "extra": True,
             "fields": [{"name": "n", "type": "integer", "ignored": 1}],
             "depends_on": ["b", "a"], "tags": [" X ", "x"], "owner": "  Team A "},
            {"id": "b", "fields": [{"name": "n", "type": "integer"}], "depends_on": ["a"]},
            {"id": "a", "fields": [{"name": "n", "type": "integer"}]}],
            "ignored": 1})
        self.assertEqual([row["id"] for row in result["added"]], ["a", "b", "c"])
        self.assertEqual(result["removed"], [])
        self.assertEqual(result["changed"], [])
        fresh = DatasetCatalog(self.path)
        self.assertEqual(fresh.describe("c"),
                         {"id": "c", "description": "7",
                          "fields": [{"name": "n", "type": "integer"}],
                          "depends_on": ["a", "b"], "tags": ["X"], "owner": "Team A"})
        self.assertEqual([row["id"] for row in fresh.export()["datasets"]], ["a", "b", "c"])
        saved = self.path.read_text(encoding="utf-8")
        self.assertTrue(saved.endswith("\n"))
        self.assertEqual(set(json.loads(saved)), {"a", "b", "c"})

    def test_apply_empty_snapshot_clears_catalog(self):
        self.catalog.import_bundle({"datasets": [
            self._snapshot_record("a"), self._snapshot_record("b", ["a"])]})
        result = self.catalog.apply_bundle({"datasets": []})
        self.assertEqual(result["added"], [])
        self.assertEqual(result["changed"], [])
        self.assertEqual([row["id"] for row in result["removed"]], ["a", "b"])
        self.assertEqual(self.catalog.export(), {"datasets": []})
        self.assertEqual(self.catalog.search(), [])

    def test_apply_without_differences_never_writes_or_creates_dirs(self):
        self.catalog.import_bundle({"datasets": [
            self._snapshot_record("a"), self._snapshot_record("b", ["a"])]})
        before = self.path.read_text(encoding="utf-8")
        result = self.catalog.apply_bundle({"datasets": [
            self._snapshot_record("b", ["a"]), self._snapshot_record("a")]})
        self.assertEqual(result, {"added": [], "removed": [], "changed": []})
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

        nested = Path(self.temp.name) / "state" / "nested" / "catalog.json"
        catalog = DatasetCatalog(nested)
        self.assertEqual(catalog.apply_bundle({"datasets": []}),
                         {"added": [], "removed": [], "changed": []})
        self.assertFalse(nested.exists())
        self.assertFalse(nested.parent.exists())

    def test_apply_rejects_invalid_snapshot_without_writing(self):
        nested = Path(self.temp.name) / "state" / "nested" / "catalog.json"
        empty_catalog = DatasetCatalog(nested)
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
            {"datasets": [self._snapshot_record("a", ["b"]),
                          self._snapshot_record("b", ["a"])]},
            {"datasets": [self._snapshot_record("a", tags=[""])]},
        ]
        for bundle in bad_bundles:
            with self.assertRaises(ValueError):
                empty_catalog.apply_bundle(bundle)
        self.assertFalse(nested.exists())
        self.assertFalse(nested.parent.exists())

        self.catalog.register(self._snapshot_record("old"))
        before = self.path.read_text(encoding="utf-8")
        for bundle in bad_bundles:
            with self.assertRaises(ValueError):
                self.catalog.apply_bundle(bundle)
            self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_apply_validates_current_catalog_like_preview(self):
        good = {"datasets": [self._snapshot_record("a")]}
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
        for index, raw in enumerate(bad_states):
            other = Path(self.temp.name) / ("bad-apply-" + str(index) + ".json")
            other.write_text(raw, encoding="utf-8")
            with self.assertRaises(ValueError):
                DatasetCatalog(other).apply_bundle(good)
            self.assertEqual(other.read_text(encoding="utf-8"), raw)

    def test_apply_does_not_mutate_input_and_result_is_detached(self):
        self.catalog.import_bundle({"datasets": [self._snapshot_record("a")]})
        bundle = {"datasets": [
            self._snapshot_record("a", description="changed"),
            self._snapshot_record("b", ["a"], tags=[" Finance ", "finance"])]}
        snapshot = json.loads(json.dumps(bundle))
        result = self.catalog.apply_bundle(bundle)
        self.assertEqual(bundle, snapshot)
        result["added"][0]["tags"].append("tampered")
        result["changed"][0]["after"]["description"] = "tampered"
        fresh = DatasetCatalog(self.path)
        self.assertEqual(fresh.describe("b")["tags"], ["Finance"])
        self.assertEqual(fresh.describe("a")["description"], "changed")
        again = self.catalog.diff_bundle(bundle)
        self.assertEqual(again, {"added": [], "removed": [], "changed": []})

    def test_apply_save_failure_raises_oserror_and_preserves_state(self):
        self.catalog.register(self._snapshot_record("old"))
        before = self.path.read_text(encoding="utf-8")
        bundle = {"datasets": [self._snapshot_record("new")]}
        with mock.patch("dataset_catalog.os.replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                self.catalog.apply_bundle(bundle)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        self.assertEqual([row["id"] for row in self.catalog.export()["datasets"]], ["old"])
        leftovers = [item for item in self.path.parent.iterdir() if item != self.path]
        self.assertEqual(leftovers, [])

        nested = Path(self.temp.name) / "empty" / "catalog.json"
        nested.parent.mkdir(parents=True)
        missing = DatasetCatalog(nested)
        with mock.patch("dataset_catalog.os.replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                missing.apply_bundle(bundle)
        self.assertFalse(nested.exists())
        self.assertEqual(list(nested.parent.iterdir()), [])

    def test_cli_apply(self):
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", str(self.path)]
        subprocess.run(prefix + ["register", str(ROOT / "samples/orders.json")],
                       check=True, capture_output=True)
        subprocess.run(prefix + ["register", str(ROOT / "samples/daily.json")],
                       check=True, capture_output=True)

        snapshot_path = Path(self.temp.name) / "snapshot.json"
        snapshot_path.write_text(json.dumps({"datasets": [
            {"id": "orders", "description": "Order amounts from the daily export",
             "fields": [{"name": "order_id", "type": "string"},
                        {"name": "amount", "type": "number"}], "depends_on": []},
            {"id": "report", "description": "new",
             "fields": [{"name": "id", "type": "string"}], "depends_on": ["orders"]}]},
            ensure_ascii=False), encoding="utf-8")
        run = subprocess.run(prefix + ["apply", str(snapshot_path)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual(set(result), {"added", "removed", "changed"})
        self.assertEqual([row["id"] for row in result["added"]], ["report"])
        self.assertEqual([row["id"] for row in result["removed"]], ["daily_totals"])
        self.assertEqual(result["changed"], [])
        exported = subprocess.run(prefix + ["export"], capture_output=True, text=True)
        self.assertEqual([row["id"] for row in json.loads(exported.stdout)["datasets"]],
                         ["orders", "report"])

        rerun = subprocess.run(prefix + ["apply", str(snapshot_path)],
                               capture_output=True, text=True)
        self.assertEqual(rerun.returncode, 0)
        self.assertEqual(json.loads(rerun.stdout), {"added": [], "removed": [], "changed": []})

        before = self.path.read_text(encoding="utf-8")
        missing = subprocess.run(prefix + ["apply", str(Path(self.temp.name) / "nope.json")],
                                 capture_output=True, text=True)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("error", json.loads(missing.stdout))
        bad_json = Path(self.temp.name) / "bad.json"
        bad_json.write_text("{not json", encoding="utf-8")
        run = subprocess.run(prefix + ["apply", str(bad_json)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("error", json.loads(run.stdout))
        invalid = Path(self.temp.name) / "invalid.json"
        invalid.write_text(json.dumps({"datasets": [
            {"id": "a", "fields": [{"name": "n", "type": "integer"}], "depends_on": ["ghost"]}]}),
            encoding="utf-8")
        run = subprocess.run(prefix + ["apply", str(invalid)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("error", json.loads(run.stdout))
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_apply_expected_match_commits_and_returns_diff(self):
        self.catalog.import_bundle({"datasets": [
            self._snapshot_record("a"), self._snapshot_record("b", ["a"])]})
        expected = {"datasets": [
            {**self._snapshot_record("b", ["a"]), "ignored": 1},
            {"fields": [{"name": "n", "type": "integer", "extra": 2}],
             "description": "a desc", "id": "a", "depends_on": []}]}
        bundle = {"datasets": [
            self._snapshot_record("a", description="updated"),
            self._snapshot_record("c", ["a"])]}
        result = self.catalog.apply_bundle(bundle, expected)
        self.assertEqual([row["id"] for row in result["added"]], ["c"])
        self.assertEqual([row["id"] for row in result["removed"]], ["b"])
        self.assertEqual([row["id"] for row in result["changed"]], ["a"])
        self.assertEqual([row["id"] for row in self.catalog.export()["datasets"]], ["a", "c"])

    def test_apply_expected_mismatch_rejected_without_writing(self):
        self.catalog.import_bundle({"datasets": [
            self._snapshot_record("a"), self._snapshot_record("b", ["a"])]})
        before = self.path.read_text(encoding="utf-8")
        expected = {"datasets": [self._snapshot_record("a")]}
        with self.assertRaises(ValueError) as caught:
            self.catalog.apply_bundle({"datasets": [self._snapshot_record("a")]}, expected)
        self.assertEqual(str(caught.exception), "catalog does not match expected snapshot")
        # mismatch is rejected even when the target equals the current state
        with self.assertRaises(ValueError) as caught:
            self.catalog.apply_bundle({"datasets": [
                self._snapshot_record("a"), self._snapshot_record("b", ["a"])]}, expected)
        self.assertEqual(str(caught.exception), "catalog does not match expected snapshot")
        # field order, tag order/spelling and cleaned owner spelling still count
        for index, record in enumerate((
                {**self._snapshot_record("a"), "fields": [
                    {"name": "m", "type": "string"}, {"name": "n", "type": "integer"}]},
                {**self._snapshot_record("a"), "tags": ["x", "y"]},
                {**self._snapshot_record("a"), "owner": "Team A"})):
            catalog = DatasetCatalog(Path(self.temp.name) / ("other-" + str(index) + ".json"))
            catalog.import_bundle({"datasets": [record]})
            variant = json.loads(json.dumps(record))
            if "fields" in record and len(record["fields"]) == 2:
                variant["fields"] = list(reversed(variant["fields"]))
            if "tags" in record:
                variant["tags"] = list(reversed(variant["tags"]))
            if "owner" in record:
                variant["owner"] = "team a"
            with self.assertRaises(ValueError):
                catalog.apply_bundle({"datasets": [record]}, {"datasets": [variant]})
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_apply_expected_validated_and_empty_matches_only_empty(self):
        nested = Path(self.temp.name) / "state" / "nested" / "catalog.json"
        empty_catalog = DatasetCatalog(nested)
        result = empty_catalog.apply_bundle({"datasets": []}, {"datasets": []})
        self.assertEqual(result, {"added": [], "removed": [], "changed": []})
        self.assertFalse(nested.exists())
        self.assertFalse(nested.parent.exists())
        with self.assertRaises(ValueError) as caught:
            empty_catalog.apply_bundle({"datasets": []}, {"datasets": [self._snapshot_record("a")]})
        self.assertEqual(str(caught.exception), "catalog does not match expected snapshot")
        self.assertFalse(nested.exists())

        self.catalog.register(self._snapshot_record("a"))
        before = self.path.read_text(encoding="utf-8")
        with self.assertRaises(ValueError):
            self.catalog.apply_bundle({"datasets": []}, {"datasets": []})
        bad_expected = [
            [], "x", 1, True, {},
            {"datasets": None}, {"other": []},
            {"datasets": [{"id": "bad id", "fields": [{"name": "n", "type": "integer"}]}]},
            {"datasets": [self._snapshot_record("a"), self._snapshot_record("a")]},
            {"datasets": [self._snapshot_record("b", ["b"])]},
            {"datasets": [self._snapshot_record("b", ["ghost"])]},
            {"datasets": [self._snapshot_record("b", ["c", "c"]),
                          self._snapshot_record("c")]},
            {"datasets": [self._snapshot_record("b", ["c"]),
                          self._snapshot_record("c", ["b"])]},
        ]
        for expected in bad_expected:
            with self.assertRaises(ValueError):
                self.catalog.apply_bundle({"datasets": [self._snapshot_record("a")]}, expected)
            self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_cli_apply_expected(self):
        prefix = [sys.executable, str(ROOT / "dataset_catalog.py"), "--catalog", str(self.path)]
        subprocess.run(prefix + ["register", str(ROOT / "samples/orders.json")],
                       check=True, capture_output=True)
        subprocess.run(prefix + ["register", str(ROOT / "samples/daily.json")],
                       check=True, capture_output=True)

        expected_path = Path(self.temp.name) / "expected.json"
        expected_path.write_text(json.dumps(self.catalog.export()), encoding="utf-8")
        snapshot_path = Path(self.temp.name) / "snapshot.json"
        snapshot_path.write_text(json.dumps({"datasets": [
            {"id": "orders", "fields": [{"name": "order_id", "type": "string"}],
             "depends_on": []}]}), encoding="utf-8")
        run = subprocess.run(prefix + ["apply", str(snapshot_path), "--expected", str(expected_path)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual([row["id"] for row in result["removed"]], ["daily_totals"])
        self.assertEqual([row["id"] for row in self.catalog.export()["datasets"]], ["orders"])

        # the catalog changed, so the same expected snapshot no longer matches
        before = self.path.read_text(encoding="utf-8")
        run = subprocess.run(prefix + ["apply", str(snapshot_path), "--expected", str(expected_path)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertEqual(json.loads(run.stdout)["error"],
                         "catalog does not match expected snapshot")
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

        null_expected = Path(self.temp.name) / "null.json"
        null_expected.write_text("null", encoding="utf-8")
        run = subprocess.run(prefix + ["apply", str(snapshot_path), "--expected", str(null_expected)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("error", json.loads(run.stdout))
        missing = subprocess.run(
            prefix + ["apply", str(snapshot_path), "--expected",
                      str(Path(self.temp.name) / "nope.json")],
            capture_output=True, text=True)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("error", json.loads(missing.stdout))
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)


    def _schema_record(self, identifier, fields, depends_on=None, description="s", **extra):
        record = {"id": identifier, "description": description,
                  "fields": fields, "depends_on": depends_on or []}
        record.update(extra)
        return record

    def test_schema_diff_add_remove_and_type_change_structure(self):
        self.catalog.import_bundle({"datasets": [
            self._schema_record("a", [{"name": "keep", "type": "string"},
                                      {"name": "gone", "type": "integer"},
                                      {"name": "widened", "type": "integer"},
                                      {"name": "narrowed", "type": "number"},
                                      {"name": "switched", "type": "string"}]),
            self._schema_record("b", [{"name": "n", "type": "integer"}], ["a"])]})
        bundle = {"datasets": [
            self._schema_record("a", [{"name": "fresh", "type": "boolean"},
                                      {"name": "keep", "type": "string"},
                                      {"name": "widened", "type": "number"},
                                      {"name": "narrowed", "type": "integer"},
                                      {"name": "switched", "type": "boolean"}]),
            self._schema_record("b", [{"name": "n", "type": "integer"}], ["a"])]}
        result = self.catalog.schema_diff_bundle(bundle)
        self.assertEqual(set(result), {"changes"})
        self.assertEqual([row["id"] for row in result["changes"]], ["a"])
        change = result["changes"][0]
        self.assertEqual(set(change),
                         {"id", "added_fields", "removed_fields", "type_changes", "breaking"})
        self.assertEqual(change["added_fields"], [{"name": "fresh", "type": "boolean"}])
        self.assertEqual(change["removed_fields"], [{"name": "gone", "type": "integer"}])
        self.assertEqual(change["type_changes"], [
            {"name": "narrowed", "before": "number", "after": "integer"},
            {"name": "switched", "before": "string", "after": "boolean"},
            {"name": "widened", "before": "integer", "after": "number"}])
        self.assertIs(change["breaking"], True)

    def test_schema_diff_integer_to_number_is_not_breaking(self):
        self.catalog.import_bundle({"datasets": [
            self._schema_record("a", [{"name": "x", "type": "integer"},
                                      {"name": "y", "type": "integer"},
                                      {"name": "z", "type": "string"}])]})
        bundle = {"datasets": [
            self._schema_record("a", [{"name": "x", "type": "number"},
                                      {"name": "y", "type": "integer"},
                                      {"name": "z", "type": "string"}])]}
        change = self.catalog.schema_diff_bundle(bundle)["changes"]
        self.assertEqual(len(change), 1)
        self.assertEqual(change[0]["added_fields"], [])
        self.assertEqual(change[0]["removed_fields"], [])
        self.assertEqual(change[0]["type_changes"],
                         [{"name": "x", "before": "integer", "after": "number"}])
        self.assertIs(change[0]["breaking"], False)

        for before_type, after_type in (("number", "integer"), ("string", "integer"),
                                        ("integer", "string"), ("boolean", "string"),
                                        ("string", "boolean"), ("number", "string")):
            other = Path(self.temp.name) / ("types-" + before_type + "-" + after_type + ".json")
            catalog = DatasetCatalog(other)
            catalog.import_bundle({"datasets": [
                self._schema_record("a", [{"name": "f", "type": before_type}])]})
            row = catalog.schema_diff_bundle({"datasets": [
                self._schema_record("a", [{"name": "f", "type": after_type}])]})["changes"][0]
            self.assertIs(row["breaking"], True, (before_type, after_type))
            self.assertEqual(row["type_changes"],
                             [{"name": "f", "before": before_type, "after": after_type}])

    def test_schema_diff_new_dataset_all_added_and_deleted_all_removed(self):
        self.catalog.import_bundle({"datasets": [
            self._schema_record("old", [{"name": "a", "type": "integer"},
                                        {"name": "b", "type": "string"}])]})
        result = self.catalog.schema_diff_bundle({"datasets": [
            self._schema_record("old", [{"name": "a", "type": "integer"},
                                        {"name": "b", "type": "string"}]),
            self._schema_record("new", [{"name": "x", "type": "number"},
                                        {"name": "y", "type": "boolean"}], ["old"])]})
        changes = result["changes"]
        self.assertEqual([row["id"] for row in changes], ["new"])
        added_only = changes[0]
        self.assertEqual(added_only["added_fields"],
                         [{"name": "x", "type": "number"}, {"name": "y", "type": "boolean"}])
        self.assertEqual(added_only["removed_fields"], [])
        self.assertEqual(added_only["type_changes"], [])
        self.assertIs(added_only["breaking"], False)
        # deleting a dataset means all its fields are removed
        empty = self.catalog.schema_diff_bundle({"datasets": []})["changes"]
        self.assertEqual(len(empty), 1)
        self.assertEqual(empty[0]["id"], "old")
        self.assertEqual(empty[0]["added_fields"], [])
        self.assertEqual(empty[0]["type_changes"], [])
        self.assertEqual(empty[0]["removed_fields"],
                         [{"name": "a", "type": "integer"}, {"name": "b", "type": "string"}])
        self.assertIs(empty[0]["breaking"], True)

    def test_schema_diff_arrays_sorted_by_unicode_codepoint(self):
        fields = [{"name": name, "type": "string"}
                  for name in ("b", "a", "A", "B", "01", "日", "é", "abc")]
        self.catalog.import_bundle({"datasets": [self._schema_record("a", fields)]})
        result = self.catalog.schema_diff_bundle({"datasets": []})
        removed = [row["name"] for row in result["changes"][0]["removed_fields"]]
        self.assertEqual(removed, sorted(removed))
        self.assertEqual(removed, ["01", "A", "B", "a", "abc", "b", "é", "日"])

        other = Path(self.temp.name) / "unicode-add.json"
        catalog = DatasetCatalog(other)
        after = [{"name": name, "type": "integer"} for name in ("中", "Z", "a", "!")]
        changes = catalog.schema_diff_bundle({"datasets": [
            self._schema_record("a", after)]})["changes"]
        added = [row["name"] for row in changes[0]["added_fields"]]
        self.assertEqual(added, ["!", "Z", "a", "中"])
        self.assertEqual(changes[0]["added_fields"][0], {"name": "!", "type": "integer"})

    def test_schema_diff_rename_is_remove_plus_add_and_matches_names_exactly(self):
        self.catalog.import_bundle({"datasets": [
            self._schema_record("a", [{"name": "Amount", "type": "integer"},
                                      {"name": " id ", "type": "string"}])]})
        bundle = {"datasets": [
            self._schema_record("a", [{"name": "amount", "type": "integer"},
                                      {"name": "id", "type": "string"}])]}
        change = self.catalog.schema_diff_bundle(bundle)["changes"][0]
        self.assertEqual(change["added_fields"], [{"name": "amount", "type": "integer"},
                                                  {"name": "id", "type": "string"}])
        self.assertEqual(change["removed_fields"], [{"name": " id ", "type": "string"},
                                                    {"name": "Amount", "type": "integer"}])
        self.assertEqual(change["type_changes"], [])
        self.assertIs(change["breaking"], True)

        # case-only or whitespace-only differences never line up as type changes
        other = Path(self.temp.name) / "exact.json"
        catalog = DatasetCatalog(other)
        catalog.import_bundle({"datasets": [
            self._schema_record("a", [{"name": "x", "type": "integer"}])]})
        row = catalog.schema_diff_bundle({"datasets": [
            self._schema_record("a", [{"name": "x", "type": "integer"},
                                      {"name": "x ", "type": "number"}])]})["changes"][0]
        self.assertEqual(row["added_fields"], [{"name": "x ", "type": "number"}])
        self.assertEqual(row["type_changes"], [])

    def test_schema_diff_ignores_metadata_and_field_order_but_diff_keeps_order_rule(self):
        records = {"datasets": [
            self._schema_record("a", [{"name": "z", "type": "integer"},
                                      {"name": "n", "type": "integer"}], ["b"],
                                tags=["t"], owner="team"),
            self._schema_record("b", [{"name": "n", "type": "integer"}])]}
        self.catalog.import_bundle(records)
        changed = {"datasets": [
            self._schema_record("b", [{"name": "n", "type": "integer"}], description="new desc"),
            self._schema_record("a", [{"name": "n", "type": "integer"},
                                      {"name": "z", "type": "integer"}], ["b"],
                                description="other", tags=["t"], owner="team")]}
        self.assertEqual(self.catalog.schema_diff_bundle(changed), {"changes": []})
        # the legacy diff still treats a pure field reorder as a change
        self.assertEqual(
            [row["changed_keys"] for row in self.catalog.diff_bundle(changed)["changed"]],
            [["description", "fields"], ["description"]])

        reordered_only = {"datasets": [
            self._schema_record("a", [{"name": "n", "type": "integer"},
                                      {"name": "z", "type": "integer"}], ["b"],
                                tags=["t"], owner="team"),
            self._schema_record("b", [{"name": "n", "type": "integer"}])]}
        self.assertEqual(self.catalog.schema_diff_bundle(reordered_only), {"changes": []})
        self.assertEqual(
            [row["changed_keys"] for row in self.catalog.diff_bundle(reordered_only)["changed"]],
            [["fields"]])

    def test_schema_diff_empty_arrays_preserved_on_field_change_only(self):
        self.catalog.import_bundle({"datasets": [
            self._schema_record("a", [{"name": "n", "type": "integer"}])]})
        change = self.catalog.schema_diff_bundle({"datasets": [
            self._schema_record("a", [{"name": "n", "type": "number"}])]})["changes"][0]
        self.assertEqual(change["added_fields"], [])
        self.assertEqual(change["removed_fields"], [])
        self.assertEqual(change["type_changes"],
                         [{"name": "n", "before": "integer", "after": "number"}])
        self.assertIs(change["breaking"], False)

    def test_schema_diff_changes_sorted_by_id_and_unchanged_omitted(self):
        self.catalog.import_bundle({"datasets": [
            self._schema_record("a", [{"name": "n", "type": "integer"}]),
            self._schema_record("b", [{"name": "n", "type": "integer"}]),
            self._schema_record("c", [{"name": "n", "type": "integer"}])]})
        result = self.catalog.schema_diff_bundle({"datasets": [
            self._schema_record("a", [{"name": "n", "type": "string"}]),
            self._schema_record("b", [{"name": "n", "type": "integer"}]),
            self._schema_record("c", [{"name": "n", "type": "integer"},
                                      {"name": "x", "type": "boolean"}])]})
        self.assertEqual([row["id"] for row in result["changes"]], ["a", "c"])

    def test_schema_diff_missing_or_empty_catalog_and_empty_snapshot(self):
        self.assertFalse(self.path.exists())
        result = self.catalog.schema_diff_bundle({"datasets": [
            self._schema_record("a", [{"name": "n", "type": "integer"}])]})
        self.assertEqual([row["id"] for row in result["changes"]], ["a"])
        self.assertEqual(result["changes"][0]["added_fields"],
                         [{"name": "n", "type": "integer"}])
        self.assertFalse(self.path.exists())

        self.path.write_text("{}", encoding="utf-8")
        self.assertEqual(DatasetCatalog(self.path).schema_diff_bundle({"datasets": []}),
                         {"changes": []})

        self.catalog.import_bundle({"datasets": [
            self._schema_record("a", [{"name": "n", "type": "integer"}])]})
        self.assertEqual(self.catalog.schema_diff_bundle({"datasets": []})["changes"][0]["id"], "a")

    def test_schema_diff_validates_both_graphs_even_for_metadata_only_changes(self):
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
                self.catalog.schema_diff_bundle(bundle)

        # validation still runs when the invalid snapshot would otherwise only change metadata
        for index, bundle in enumerate(bad_bundles):
            other = Path(self.temp.name) / ("bad-schema-snapshot-" + str(index) + ".json")
            catalog = DatasetCatalog(other)
            catalog.register(self._snapshot_record("old"))
            with self.assertRaises(ValueError):
                catalog.schema_diff_bundle(bundle)

        # an invalid current catalog is rejected like in preview
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
            other = Path(self.temp.name) / "bad-schema-catalog.json"
            other.write_text(raw, encoding="utf-8")
            with self.assertRaises(ValueError):
                DatasetCatalog(other).schema_diff_bundle(json.loads(good))

    def test_schema_diff_does_not_mutate_inputs_or_write_files(self):
        self.catalog.import_bundle({"datasets": [
            self._schema_record("a", [{"name": "n", "type": "integer"}])]})
        catalog_before = self.path.read_text(encoding="utf-8")
        bundle = {"datasets": [
            self._schema_record("a", [{"name": "n", "type": "number"},
                                      {"name": "x", "type": "boolean"}])], "ignored": True}
        snapshot = json.loads(json.dumps(bundle))
        result = self.catalog.schema_diff_bundle(bundle)
        self.assertEqual(bundle, snapshot)
        self.assertEqual(self.path.read_text(encoding="utf-8"), catalog_before)
        result["changes"][0]["added_fields"].append({"name": "tampered", "type": "string"})
        result["changes"][0]["type_changes"][0]["after"] = "string"
        again = self.catalog.schema_diff_bundle(bundle)["changes"][0]
        self.assertEqual(again["added_fields"], [{"name": "x", "type": "boolean"}])
        self.assertEqual(again["type_changes"][0]["after"], "number")

        nested = Path(self.temp.name) / "state" / "nested" / "catalog.json"
        empty_catalog = DatasetCatalog(nested)
        self.assertEqual(empty_catalog.schema_diff_bundle({"datasets": []}), {"changes": []})
        self.assertFalse(nested.exists())
        self.assertFalse(nested.parent.exists())

    def test_cli_schema_diff_success_and_errors(self):
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
            {"id": "daily_totals", "description": "same fields, new description",
             "fields": [{"name": "day", "type": "string"}, {"name": "total", "type": "number"}],
             "depends_on": ["orders"]},
            {"id": "report", "description": "new",
             "fields": [{"name": "id", "type": "string"}], "depends_on": ["orders"]}]},
            ensure_ascii=False), encoding="utf-8")
        run = subprocess.run(prefix + ["schema-diff", str(snapshot_path)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual(set(result), {"changes"})
        self.assertEqual([row["id"] for row in result["changes"]], ["report"])
        self.assertEqual(result["changes"][0]["added_fields"],
                         [{"name": "id", "type": "string"}])
        self.assertEqual(result["changes"][0]["removed_fields"], [])
        self.assertEqual(result["changes"][0]["type_changes"], [])
        self.assertIs(result["changes"][0]["breaking"], False)
        self.assertEqual(self.path.read_text(encoding="utf-8"), catalog_before)

        removed = subprocess.run(prefix + ["schema-diff", str(ROOT / "samples" / "orders.json")],
                                 capture_output=True, text=True)
        self.assertEqual(removed.returncode, 2)
        self.assertIn("error", json.loads(removed.stdout))

        missing = subprocess.run(prefix + ["schema-diff", str(Path(self.temp.name) / "nope.json")],
                                 capture_output=True, text=True)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("error", json.loads(missing.stdout))

        bad_json = Path(self.temp.name) / "bad.json"
        bad_json.write_text("{not json", encoding="utf-8")
        run = subprocess.run(prefix + ["schema-diff", str(bad_json)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("error", json.loads(run.stdout))

        invalid = Path(self.temp.name) / "invalid.json"
        invalid.write_text(json.dumps({"datasets": [
            {"id": "a", "fields": [{"name": "n", "type": "integer"}], "depends_on": ["a"]}]}),
            encoding="utf-8")
        run = subprocess.run(prefix + ["schema-diff", str(invalid)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("error", json.loads(run.stdout))
        self.assertEqual(self.path.read_text(encoding="utf-8"), catalog_before)


if __name__ == "__main__":
    unittest.main()
