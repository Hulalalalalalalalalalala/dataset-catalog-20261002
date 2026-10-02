"""Register dataset schemas and inspect their direct dependencies."""
import argparse
import copy
import heapq
import json
import re
from pathlib import Path


FIELD_TYPES = ("string", "integer", "number", "boolean")


def _normalize_tags(tags):
    if not isinstance(tags, list):
        raise ValueError("tags must be a list of strings")
    normalized = []
    seen = set()
    for tag in tags:
        if not isinstance(tag, str):
            raise ValueError("tags must be a list of strings")
        cleaned = tag.strip()
        if not cleaned:
            raise ValueError("tags must be nonempty strings")
        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            normalized.append(cleaned)
    return normalized


def _normalized_record(dataset, known_ids):
    if not isinstance(dataset, dict):
        raise ValueError("invalid dataset descriptor")
    identifier = dataset.get("id")
    if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", identifier):
        raise ValueError("invalid dataset id")
    fields = dataset.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("at least one field is required")
    names = []
    for field in fields:
        if not isinstance(field, dict) or "name" not in field or "type" not in field:
            raise ValueError("invalid field descriptor")
        name = field["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("field names must be nonempty strings")
        if field["type"] not in FIELD_TYPES:
            raise ValueError("unsupported field type")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("field names must be unique")
    dependencies = dataset.get("depends_on", [])
    if not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
        raise ValueError("depends_on must be a list of dataset ids")
    if len(set(dependencies)) != len(dependencies) or any(item not in known_ids for item in dependencies):
        raise ValueError("dependencies must be unique, already registered dataset ids")
    entry = {"id": identifier, "description": str(dataset.get("description", "")),
             "fields": [{"name": field["name"], "type": field["type"]} for field in fields],
             "depends_on": sorted(dependencies)}
    if "tags" in dataset:
        entry["tags"] = _normalize_tags(dataset["tags"])
    return entry


def _toposort(entries):
    remaining = {key: set(entry["depends_on"]) for key, entry in entries.items()}
    ready = [key for key, deps in remaining.items() if not deps]
    heapq.heapify(ready)
    ordered = []
    while ready:
        key = heapq.heappop(ready)
        ordered.append(key)
        for other, deps in remaining.items():
            if key in deps:
                deps.remove(key)
                if not deps:
                    heapq.heappush(ready, other)
    return ordered if len(ordered) == len(entries) else None


def _normalized_snapshot(bundle):
    """Validate a complete-snapshot bundle and return its normalized records."""
    if not isinstance(bundle, dict):
        raise ValueError("bundle must be an object")
    descriptors = bundle.get("datasets")
    if not isinstance(descriptors, list):
        raise ValueError("bundle must contain a datasets array")

    snapshot_ids = set()
    for dataset in descriptors:
        if not isinstance(dataset, dict):
            raise ValueError("invalid dataset descriptor")
        identifier = dataset.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", identifier):
            raise ValueError("invalid dataset id")
        if identifier in snapshot_ids:
            raise ValueError("dataset already registered")
        snapshot_ids.add(identifier)

    snapshot = {}
    for dataset in descriptors:
        entry = _normalized_record(dataset, snapshot_ids)
        if entry["id"] in entry["depends_on"]:
            raise ValueError("dependencies must be unique, already registered dataset ids")
        snapshot[entry["id"]] = entry
    if _toposort(snapshot) is None:
        raise ValueError("dependency cycle detected")
    return snapshot


def _normalized_catalog(records):
    """Validate the persisted catalog as a self-contained dependency graph."""
    if not isinstance(records, dict):
        raise ValueError("invalid catalog")
    known_ids = set(records)
    current = {}
    for key, entry in records.items():
        normalized = _normalized_record(entry, known_ids)
        if normalized["id"] != key or normalized["id"] in current:
            raise ValueError("invalid dataset id")
        if normalized["id"] in normalized["depends_on"]:
            raise ValueError("dependencies must be unique, already registered dataset ids")
        current[normalized["id"]] = normalized
    if _toposort(current) is None:
        raise ValueError("dependency cycle detected")
    return current


def _diff_records(current, snapshot):
    before_ids = set(current)
    after_ids = set(snapshot)
    added = [copy.deepcopy(snapshot[key]) for key in sorted(after_ids - before_ids)]
    removed = [copy.deepcopy(current[key]) for key in sorted(before_ids - after_ids)]
    changed = []
    for key in sorted(before_ids & after_ids):
        before = current[key]
        after = snapshot[key]
        changed_keys = sorted(name for name in set(before) | set(after)
                              if before.get(name) != after.get(name))
        if changed_keys:
            changed.append({"id": key, "before": copy.deepcopy(before),
                            "after": copy.deepcopy(after), "changed_keys": changed_keys})
    return {"added": added, "removed": removed, "changed": changed}


def _downstream_index(records):
    downstream = {}
    for key, entry in records.items():
        for source in entry["depends_on"]:
            downstream.setdefault(source, set()).add(key)
    return downstream


def _shortest_downstream(downstream, source):
    """Map each reached id to its shortest downstream path from source.

    Distances are positive edge counts and the path includes source and target;
    among equal-distance paths the lexicographically smallest id sequence wins.
    """
    reached = {}
    frontier = {source: [source]}
    distance = 0
    while frontier:
        distance += 1
        candidates = {}
        for node, path in frontier.items():
            for child in downstream.get(node, ()):  # unconnected records are never reached
                if child in reached:
                    continue
                candidate = path + [child]
                if child not in candidates or candidate < candidates[child]:
                    candidates[child] = candidate
        for child, path in candidates.items():
            reached[child] = {"distance": distance, "path": path}
        frontier = candidates
    return reached


class DatasetCatalog:
    def __init__(self, path):
        self.path = Path(path)

    def entries(self):
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def _normalize_entry(self, dataset, known_ids):
        return _normalized_record(dataset, known_ids)

    def register(self, dataset):
        records = self.entries()
        if not isinstance(dataset, dict):
            raise ValueError("invalid dataset descriptor")
        identifier = dataset.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", identifier):
            raise ValueError("invalid dataset id")
        if identifier in records:
            raise ValueError("dataset already registered")
        entry = self._normalize_entry(dataset, set(records))
        records[identifier] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return entry

    def describe(self, identifier):
        records = self.entries()
        if identifier not in records:
            raise ValueError("unknown dataset")
        return records[identifier]

    def search(self, query="", field_type=None, tags=None):
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if field_type is not None and field_type not in FIELD_TYPES:
            raise ValueError("field_type must be None or one of string, integer, number, boolean")
        required_tags = _normalize_tags(tags) if tags is not None else []
        words = {word.casefold() for word in query.split()}
        records = self.entries()
        results = []
        for identifier in sorted(records):
            entry = records[identifier]
            if field_type is not None and not any(field["type"] == field_type for field in entry["fields"]):
                continue
            if required_tags:
                available = {tag.casefold() for tag in entry.get("tags", [])}
                if not all(tag.casefold() in available for tag in required_tags):
                    continue
            matched_indices = set()
            matched = True
            dataset_haystack = (entry["id"] + "\n" + entry.get("description", "")).casefold()
            for word in words:
                hits = {index for index, field in enumerate(entry["fields"])
                        if word in field["name"].casefold()}
                matched_indices.update(hits)
                if not hits and word not in dataset_haystack:
                    matched = False
                    break
            if matched:
                results.append({"dataset": entry,
                                "matched_fields": [entry["fields"][index]["name"]
                                                   for index in sorted(matched_indices)]})
        return results

    def dependencies(self, identifier):
        entry = self.describe(identifier)
        return [self.describe(key) for key in entry["depends_on"]]

    def impact(self, identifier, max_depth=None):
        if max_depth is not None and (isinstance(max_depth, bool)
                                      or not isinstance(max_depth, int) or max_depth <= 0):
            raise ValueError("max_depth must be None or a positive integer")
        records = self.entries()
        if identifier not in records:
            raise ValueError("unknown dataset")
        downstream = {}
        for key, entry in records.items():
            for source in entry["depends_on"]:
                downstream.setdefault(source, set()).add(key)
        reached = {identifier: [identifier]}
        frontier = {identifier: [identifier]}
        result = []
        distance = 0
        while frontier and (max_depth is None or distance < max_depth):
            distance += 1
            candidates = {}
            for node, path in frontier.items():
                for child in downstream.get(node, ()):  # unconnected records are never reached
                    if child in reached:
                        continue
                    candidate = path + [child]
                    if child not in candidates or candidate < candidates[child]:
                        candidates[child] = candidate
            if not candidates:
                break
            frontier = candidates
            reached.update(candidates)
            for child, path in candidates.items():
                result.append({"dataset": records[child], "distance": distance, "path": path})
        result.sort(key=lambda item: (item["distance"], item["dataset"]["id"]))
        return result

    def export(self, identifiers=None):
        if identifiers is not None and not isinstance(identifiers, list):
            raise ValueError("identifiers must be None or a list of dataset ids")
        selected = []
        if identifiers is not None:
            for item in identifiers:
                if not isinstance(item, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", item):
                    raise ValueError("invalid dataset id")
                if item not in selected:
                    selected.append(item)
        records = self.entries()
        unknown = [item for item in selected if item not in records]
        if unknown:
            raise ValueError("unknown dataset")
        if identifiers is None:
            selected = list(records)

        included = set()
        stack = list(selected)
        while stack:
            key = stack.pop()
            if key in included:
                continue
            included.add(key)
            for source in records[key]["depends_on"]:
                if source not in records:
                    raise ValueError("missing dependency")
                if source not in included:
                    stack.append(source)

        subset = {key: records[key] for key in included}
        ordered = _toposort(subset)
        if ordered is None:
            raise ValueError("dependency cycle detected")
        return {"datasets": [records[key] for key in ordered]}

    def import_bundle(self, bundle):
        if not isinstance(bundle, dict):
            raise ValueError("bundle must be an object")
        descriptors = bundle.get("datasets")
        if not isinstance(descriptors, list):
            raise ValueError("bundle must contain a datasets array")

        records = self.entries()
        batch_ids = set()
        for dataset in descriptors:
            if not isinstance(dataset, dict):
                raise ValueError("invalid dataset descriptor")
            identifier = dataset.get("id")
            if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", identifier):
                raise ValueError("invalid dataset id")
            if identifier in records or identifier in batch_ids:
                raise ValueError("dataset already registered")
            batch_ids.add(identifier)

        known = set(records) | batch_ids
        new_entries = {}
        for dataset in descriptors:
            entry = self._normalize_entry(dataset, known)
            if entry["id"] in entry["depends_on"]:
                raise ValueError("dependencies must be unique, already registered dataset ids")
            new_entries[entry["id"]] = entry

        batch = {key: {"depends_on": [dep for dep in entry["depends_on"] if dep in batch_ids]}
                 for key, entry in new_entries.items()}
        ordered = _toposort(batch)
        if ordered is None:
            raise ValueError("dependency cycle detected")

        imported = [new_entries[key] for key in ordered]
        if new_entries:
            for entry in imported:
                records[entry["id"]] = entry
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        return {"datasets": imported}

    def diff_bundle(self, bundle):
        snapshot = _normalized_snapshot(bundle)
        records = self.entries()
        current = {key: _normalized_record(entry, set(records))
                   for key, entry in records.items()}
        return _diff_records(current, snapshot)

    def preview_bundle(self, bundle):
        snapshot = _normalized_snapshot(bundle)
        current = _normalized_catalog(self.entries())
        diff = _diff_records(current, snapshot)

        source_ids = ([row["id"] for row in diff["added"]]
                      + [row["id"] for row in diff["removed"]]
                      + [row["id"] for row in diff["changed"]])

        before_graph = _downstream_index(current)
        after_graph = _downstream_index(snapshot)
        before_reach = {}
        after_reach = {}
        for source in source_ids:
            before_reach[source] = (
                _shortest_downstream(before_graph, source) if source in current else None)
            after_reach[source] = (
                _shortest_downstream(after_graph, source) if source in snapshot else None)

        affected_map = {}
        for source in source_ids:
            before_hits = before_reach[source] or {}
            after_hits = after_reach[source] or {}
            for target in set(before_hits) | set(after_hits):
                if target == source:  # a source is not affected by its own change
                    continue
                causes = affected_map.setdefault(target, {})
                before = before_hits.get(target)
                after = after_hits.get(target)
                if before is not None or after is not None:
                    causes[source] = {"id": source, "before": before, "after": after}

        affected = []
        for target in sorted(affected_map):
            causes = affected_map[target]
            affected.append({"id": target,
                             "causes": [causes[source] for source in sorted(causes)]})
        return {"diff": diff, "affected": affected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="samples/catalog.json")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("register").add_argument("file")
    commands.add_parser("import", help="import a bundle of dataset metadata").add_argument("file")
    commands.add_parser("diff", help="compare the catalog with a read-only snapshot bundle").add_argument("file")
    commands.add_parser("preview", help="diff a snapshot and preview its downstream impact").add_argument("file")
    commands.add_parser("describe").add_argument("id")
    commands.add_parser("dependencies").add_argument("id")
    impact = commands.add_parser("impact")
    impact.add_argument("id")
    impact.add_argument("--max-depth")
    export = commands.add_parser("export")
    export.add_argument("--id", action="append", dest="ids")
    search = commands.add_parser("search")
    search.add_argument("query", nargs="?", default="")
    search.add_argument("--field-type", dest="field_type")
    search.add_argument("--tag", action="append", dest="tags")
    args = parser.parse_args()
    try:
        catalog = DatasetCatalog(args.catalog)
        if args.command == "register":
            result = catalog.register(json.loads(Path(args.file).read_text(encoding="utf-8")))
        elif args.command == "import":
            result = catalog.import_bundle(json.loads(Path(args.file).read_text(encoding="utf-8")))
        elif args.command == "diff":
            result = catalog.diff_bundle(json.loads(Path(args.file).read_text(encoding="utf-8")))
        elif args.command == "preview":
            result = catalog.preview_bundle(json.loads(Path(args.file).read_text(encoding="utf-8")))
        elif args.command == "export":
            result = catalog.export(None if args.ids is None else args.ids)
        elif args.command == "impact":
            raw_depth = args.max_depth
            if raw_depth is not None:
                if not re.fullmatch(r"[0-9]+", raw_depth) or int(raw_depth) <= 0:
                    raise ValueError("invalid max_depth")
                max_depth = int(raw_depth)
            else:
                max_depth = None
            result = catalog.impact(args.id, max_depth=max_depth)
        elif args.command == "search":
            result = catalog.search(args.query, args.field_type, args.tags)
        else:
            result = getattr(catalog, args.command)(args.id)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
