"""Register dataset schemas and inspect their direct dependencies."""
import argparse
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
        if max_depth is not None and (
            isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth <= 0
        ):
            raise ValueError("max_depth must be None or a positive integer")
        records = self.entries()
        if identifier not in records:
            raise ValueError("unknown dataset")
        downstream = {key: [] for key in records}
        for key, entry in records.items():
            for dependency in entry["depends_on"]:
                downstream[dependency].append(key)
        # Level-synchronous BFS; paths[id] is the lexicographically smallest
        # shortest id sequence reaching id from the source.
        paths = {identifier: [identifier]}
        frontier = [identifier]
        while frontier:
            following = {}
            for current in frontier:
                for target in downstream[current]:
                    if target in paths:
                        continue
                    candidate = paths[current] + [target]
                    if target not in following or candidate < following[target]:
                        following[target] = candidate
            paths.update(following)
            frontier = list(following)
        items = []
        for target, path in paths.items():
            if target == identifier:
                continue
            distance = len(path) - 1
            if max_depth is not None and distance > max_depth:
                continue
            items.append({"dataset": records[target], "distance": distance, "path": path})
        items.sort(key=lambda item: (item["distance"], item["dataset"]["id"]))
        return items


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
    args = parser.parse_args()
    try:
        catalog = DatasetCatalog(args.catalog)
        if args.command == "register":
            result = catalog.register(json.loads(Path(args.file).read_text(encoding="utf-8")))
        elif args.command == "impact":
            raw_depth = args.max_depth
            if raw_depth is not None and (
                re.fullmatch(r"[0-9]+", raw_depth) is None or int(raw_depth) <= 0
            ):
                raise ValueError("max-depth must be a string of digits 0-9 with a value greater than zero")
            result = catalog.impact(args.id, None if raw_depth is None else int(raw_depth))
        else:
            result = getattr(catalog, args.command)(args.id)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
