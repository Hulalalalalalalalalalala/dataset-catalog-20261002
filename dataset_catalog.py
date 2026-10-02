"""Register dataset schemas and inspect their direct dependencies."""
import argparse
import heapq
import json
import re
from pathlib import Path


class DatasetCatalog:
    def __init__(self, path):
        self.path = Path(path)

    def entries(self):
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def register(self, dataset):
        records = self.entries()
        identifier = dataset["id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", identifier):
            raise ValueError("invalid dataset id")
        if identifier in records:
            raise ValueError("dataset already registered")
        fields = dataset["fields"]
        if not isinstance(fields, list) or not fields:
            raise ValueError("at least one field is required")
        names = [field["name"] for field in fields]
        if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
            raise ValueError("field names must be nonempty and unique")
        if any(field["type"] not in ("string", "integer", "number", "boolean") for field in fields):
            raise ValueError("unsupported field type")
        dependencies = dataset.get("depends_on", [])
        if not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
            raise ValueError("depends_on must be a list of dataset ids")
        if len(set(dependencies)) != len(dependencies) or any(item not in records for item in dependencies):
            raise ValueError("dependencies must be unique, already registered dataset ids")
        entry = {"id": identifier, "description": str(dataset.get("description", "")),
                 "fields": [{"name": field["name"], "type": field["type"]} for field in fields],
                 "depends_on": sorted(dependencies)}
        records[identifier] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return entry

    def describe(self, identifier):
        records = self.entries()
        if identifier not in records:
            raise ValueError("unknown dataset")
        return records[identifier]

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
        if identifiers is not None:
            if not isinstance(identifiers, list):
                raise ValueError("identifiers must be None or a list of dataset ids")
            if any(not isinstance(item, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", item)
                   for item in identifiers):
                raise ValueError("identifiers must be valid dataset ids")
        records = self.entries()
        if identifiers is None:
            selected = set(records)
        else:
            selected = set(identifiers)
            unknown = selected - records.keys()
            if unknown:
                raise ValueError("unknown dataset")
        included = set()

        def collect(identifier):
            if identifier in included:
                return
            included.add(identifier)
            for dependency in records[identifier]["depends_on"]:
                if dependency not in records:
                    raise ValueError("missing dependency")
                collect(dependency)

        for identifier in selected:
            collect(identifier)
        ready = []
        remaining = {}
        dependents = {key: [] for key in included}
        for key in included:
            deps = set(records[key]["depends_on"]) & included
            remaining[key] = deps
            if not deps:
                heapq.heappush(ready, key)
            for dependency in deps:
                dependents[dependency].append(key)
        ordered = []
        while ready:
            key = heapq.heappop(ready)
            ordered.append(records[key])
            for dependent in dependents[key]:
                remaining[dependent].discard(key)
                if not remaining[dependent]:
                    heapq.heappush(ready, dependent)
        if len(ordered) != len(included):
            raise ValueError("dependency cycle")
        return {"datasets": ordered}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="samples/catalog.json")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("register").add_argument("file")
    commands.add_parser("describe").add_argument("id")
    commands.add_parser("dependencies").add_argument("id")
    impact = commands.add_parser("impact")
    impact.add_argument("id")
    impact.add_argument("--max-depth")
    export = commands.add_parser("export")
    export.add_argument("--id", action="append", dest="ids")
    args = parser.parse_args()
    try:
        catalog = DatasetCatalog(args.catalog)
        if args.command == "register":
            result = catalog.register(json.loads(Path(args.file).read_text(encoding="utf-8")))
        elif args.command == "impact":
            raw_depth = args.max_depth
            if raw_depth is not None:
                if not re.fullmatch(r"[0-9]+", raw_depth) or int(raw_depth) <= 0:
                    raise ValueError("invalid max_depth")
                max_depth = int(raw_depth)
            else:
                max_depth = None
            result = catalog.impact(args.id, max_depth=max_depth)
        elif args.command == "export":
            result = catalog.export(args.ids)
        else:
            result = getattr(catalog, args.command)(args.id)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
