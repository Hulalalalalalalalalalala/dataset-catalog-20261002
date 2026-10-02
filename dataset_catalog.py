"""Register dataset schemas and inspect their direct dependencies."""
import argparse
import copy
import csv
import heapq
import io
import json
import os
import re
import sys
import tempfile
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


def _normalize_owner(owner):
    if not isinstance(owner, str):
        raise ValueError("owner must be a string")
    cleaned = owner.strip()
    if not cleaned:
        raise ValueError("owner must be a nonempty string")
    return cleaned


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
    if "owner" in dataset:
        entry["owner"] = _normalize_owner(dataset["owner"])
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


def _downstream_index(records):
    downstream = {}
    for key, entry in records.items():
        for source in entry["depends_on"]:
            downstream.setdefault(source, set()).add(key)
    return downstream


def _shortest_downstream_paths(downstream, source):
    reached = {source: [source]}
    frontier = {source: [source]}
    paths = {}
    while frontier:
        candidates = {}
        for node, path in frontier.items():
            for child in downstream.get(node, ()):
                if child in reached:
                    continue
                candidate = path + [child]
                if child not in candidates or candidate < candidates[child]:
                    candidates[child] = candidate
        if not candidates:
            break
        frontier = candidates
        reached.update(candidates)
        paths.update(candidates)
    return paths


def _shortest_upstream_paths(closure, start):
    reached = {start: [start]}
    frontier = {start: [start]}
    while frontier:
        candidates = {}
        for node, path in frontier.items():
            for parent in closure[node]["depends_on"]:
                if parent in reached:
                    continue
                candidate = path + [parent]
                if parent not in candidates or candidate < candidates[parent]:
                    candidates[parent] = candidate
        if not candidates:
            break
        frontier = candidates
        reached.update(candidates)
    return reached


def _validated_upstream_closure(records, start):
    """Normalize start and every reachable upstream record as one subgraph."""
    known_ids = set(records)
    closure = {}
    stack = [start]
    while stack:
        key = stack.pop()
        if key in closure:
            continue
        entry = _normalized_record(records[key], known_ids)
        if entry["id"] != key:
            raise ValueError("invalid dataset descriptor")
        if entry["id"] in entry["depends_on"]:
            raise ValueError("dependencies must be unique, already registered dataset ids")
        closure[key] = entry
        stack.extend(entry["depends_on"])
    if _toposort(closure) is None:
        raise ValueError("dependency cycle detected")
    return closure


CSV_HEADER = ("dataset_id", "description", "owner", "tags", "depends_on",
              "field_name", "field_type")


def _validated_upstream_closure_many(records, starts):
    """Normalize every start and its reachable upstream records as one subgraph."""
    known_ids = set(records)
    closure = {}
    stack = list(starts)
    while stack:
        key = stack.pop()
        if key in closure:
            continue
        entry = _normalized_record(records[key], known_ids)
        if entry["id"] != key:
            raise ValueError("invalid dataset descriptor")
        if entry["id"] in entry["depends_on"]:
            raise ValueError("dependencies must be unique, already registered dataset ids")
        closure[key] = entry
        stack.extend(entry["depends_on"])
    if _toposort(closure) is None:
        raise ValueError("dependency cycle detected")
    return closure


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

    def search(self, query="", field_type=None, tags=None, owner=None):
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if field_type is not None and field_type not in FIELD_TYPES:
            raise ValueError("field_type must be None or one of string, integer, number, boolean")
        required_tags = _normalize_tags(tags) if tags is not None else []
        required_owner = _normalize_owner(owner).casefold() if owner is not None else None
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
            if required_owner is not None:
                entry_owner = entry.get("owner")
                if not isinstance(entry_owner, str) or entry_owner.casefold() != required_owner:
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

    def upstream(self, identifier, max_depth=None):
        if max_depth is not None and (isinstance(max_depth, bool)
                                      or not isinstance(max_depth, int) or max_depth <= 0):
            raise ValueError("max_depth must be None or a positive integer")
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", identifier):
            raise ValueError("invalid dataset id")
        records = self.entries()
        if not isinstance(records, dict):
            raise ValueError("invalid catalog state")
        if identifier not in records:
            raise ValueError("unknown dataset")
        closure = _validated_upstream_closure(records, identifier)
        all_paths = _shortest_upstream_paths(closure, identifier)
        result = []
        for parent, path in all_paths.items():
            if parent == identifier:
                continue
            distance = len(path) - 1
            if max_depth is None or distance <= max_depth:
                result.append({"dataset": records[parent], "distance": distance, "path": path})
        result.sort(key=lambda item: (item["distance"], item["dataset"]["id"]))
        return result

    def common_upstream(self, identifiers):
        if not isinstance(identifiers, list):
            raise ValueError("identifiers must be a list of dataset ids")
        targets = []
        for item in identifiers:
            if not isinstance(item, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", item):
                raise ValueError("invalid dataset id")
            if item not in targets:
                targets.append(item)
        if len(targets) < 2:
            raise ValueError("at least two distinct dataset ids are required")
        records = self.entries()
        if not isinstance(records, dict):
            raise ValueError("invalid catalog state")
        if any(item not in records for item in targets):
            raise ValueError("unknown dataset")
        closures = {target: _validated_upstream_closure(records, target) for target in targets}
        target_paths = {target: _shortest_upstream_paths(closures[target], target)
                        for target in targets}
        shared = set(target_paths[targets[0]])
        for target in targets[1:]:
            shared &= set(target_paths[target])
        # a candidate is dropped when another shared source depends on it, even
        # transitively through nodes that are not themselves shared
        merged = {key: entry for closure in closures.values()
                  for key, entry in closure.items()}
        downstream = _downstream_index(merged)

        def nearest(source):
            return not (set(_shortest_downstream_paths(downstream, source)) & shared)

        result = []
        for source in sorted(shared):
            if not nearest(source):
                continue
            source_targets = [
                {"id": target, "distance": len(target_paths[target][source]) - 1,
                 "path": target_paths[target][source]}
                for target in sorted(targets)]
            result.append({"dataset": records[source], "targets": source_targets})
        return result

    def between(self, source, target):
        for endpoint in (source, target):
            if not isinstance(endpoint, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", endpoint):
                raise ValueError("invalid dataset id")
        records = self.entries()
        if not isinstance(records, dict):
            raise ValueError("invalid catalog state")
        if source not in records or target not in records:
            raise ValueError("unknown dataset")
        current = self._current_state()
        if source == target:
            return {"datasets": [records[source]], "edges": []}
        downstream = _downstream_index(current)
        forward = set()
        stack = [source]
        while stack:
            node = stack.pop()
            if node in forward:
                continue
            forward.add(node)
            stack.extend(downstream.get(node, ()))
        backward = set()
        stack = [target]
        while stack:
            node = stack.pop()
            if node in backward:
                continue
            backward.add(node)
            stack.extend(current[node]["depends_on"])
        chain = forward & backward  # every member sits on at least one source -> target path
        if not chain:
            return {"datasets": [], "edges": []}
        edges = []
        for node in chain:
            for parent in current[node]["depends_on"]:
                if parent in chain:
                    edges.append({"source": parent, "target": node})
        edges.sort(key=lambda edge: (edge["source"], edge["target"]))
        return {"datasets": [records[key] for key in sorted(chain)], "edges": edges}

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

    def export_csv(self, identifiers=None):
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
        if not isinstance(records, dict):
            raise ValueError("invalid catalog state")
        unknown = [item for item in selected if item not in records]
        if unknown:
            raise ValueError("unknown dataset")
        if identifiers is None:
            selected = list(records)

        closure = _validated_upstream_closure_many(records, selected)
        ordered = _toposort(closure)

        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(CSV_HEADER)
        for key in ordered:
            entry = closure[key]
            owner = entry.get("owner", "")
            tags = json.dumps(entry.get("tags", []), ensure_ascii=False, separators=(",", ":"))
            depends_on = json.dumps(entry["depends_on"], ensure_ascii=False, separators=(",", ":"))
            for field in entry["fields"]:
                writer.writerow([entry["id"], entry["description"], owner, tags, depends_on,
                                 field["name"], field["type"]])
        return output.getvalue()

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

    def _snapshot_state(self, bundle):
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

    @staticmethod
    def _diff_states(current, snapshot):
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

    def diff_bundle(self, bundle):
        snapshot = self._snapshot_state(bundle)
        records = self.entries()
        current = {key: _normalized_record(entry, set(records))
                   for key, entry in records.items()}
        return self._diff_states(current, snapshot)

    def _current_state(self):
        records = self.entries()
        if not isinstance(records, dict):
            raise ValueError("invalid catalog state")
        current = {}
        for key, entry in records.items():
            normalized = _normalized_record(entry, set(records))
            if normalized["id"] != key:
                raise ValueError("invalid dataset descriptor")
            if normalized["id"] in normalized["depends_on"]:
                raise ValueError("dependencies must be unique, already registered dataset ids")
            current[key] = normalized
        if _toposort(current) is None:
            raise ValueError("dependency cycle detected")
        return current

    def preview_bundle(self, bundle):
        snapshot = self._snapshot_state(bundle)
        current = self._current_state()

        diff = self._diff_states(current, snapshot)
        sources = {row["id"] for row in diff["added"]}
        sources.update(row["id"] for row in diff["removed"])
        sources.update(row["id"] for row in diff["changed"])

        # target id -> source id -> {"before"/"after": propagation explanation}
        reached_by = {}
        for side, state in (("before", current), ("after", snapshot)):
            downstream = _downstream_index(state)
            for source in sources:
                if source not in state:
                    continue
                for target, path in _shortest_downstream_paths(downstream, source).items():
                    propagation = {"distance": len(path) - 1, "path": path}
                    reached_by.setdefault(target, {}).setdefault(source, {})[side] = propagation

        affected = []
        for target in sorted(reached_by):
            causes = []
            for source in sorted(reached_by[target]):
                sides = reached_by[target][source]
                before = sides.get("before")
                after = sides.get("after")
                if before is None and after is None:
                    continue
                causes.append({"id": source, "before": before, "after": after})
            if causes:
                affected.append({"id": target, "causes": causes})
        return {"diff": diff, "affected": affected}

    @staticmethod
    def _field_map(entry):
        return {field["name"]: field["type"] for field in entry["fields"]}

    @staticmethod
    def _schema_diff_states(current, snapshot):
        changes = []
        for key in sorted(set(current) | set(snapshot)):
            if key in current and key in snapshot:
                before_fields = DatasetCatalog._field_map(current[key])
                after_fields = DatasetCatalog._field_map(snapshot[key])
                added_names = sorted(set(after_fields) - set(before_fields))
                removed_names = sorted(set(before_fields) - set(after_fields))
                shared_names = sorted(set(before_fields) & set(after_fields))
            elif key in snapshot:
                before_fields = {}
                after_fields = DatasetCatalog._field_map(snapshot[key])
                added_names = sorted(after_fields)
                removed_names = []
                shared_names = []
            else:
                before_fields = DatasetCatalog._field_map(current[key])
                after_fields = {}
                added_names = []
                removed_names = sorted(before_fields)
                shared_names = []
            added_fields = [{"name": name, "type": after_fields[name]} for name in added_names]
            removed_fields = [{"name": name, "type": before_fields[name]} for name in removed_names]
            type_changes = [{"name": name, "before": before_fields[name], "after": after_fields[name]}
                            for name in shared_names
                            if before_fields[name] != after_fields[name]]
            if not added_fields and not removed_fields and not type_changes:
                continue
            breaking = bool(removed_fields) or any(
                not (change["before"] == "integer" and change["after"] == "number")
                for change in type_changes)
            changes.append({"id": key, "added_fields": added_fields,
                            "removed_fields": removed_fields, "type_changes": type_changes,
                            "breaking": breaking})
        return {"changes": changes}

    @staticmethod
    def _relation_pairs(state):
        downstream = _downstream_index(state)
        pairs = {}
        for source in state:
            for target, path in _shortest_downstream_paths(downstream, source).items():
                pairs[(source, target)] = {"distance": len(path) - 1, "path": path}
        return pairs

    def relation_diff_bundle(self, bundle):
        snapshot = self._snapshot_state(bundle)
        current = self._current_state()
        before = self._relation_pairs(current)
        after = self._relation_pairs(snapshot)
        added = []
        removed = []
        changed = []
        for source, target in sorted(set(before) | set(after)):
            prior = before.get((source, target))
            follow = after.get((source, target))
            item = {"source": source, "target": target, "before": prior, "after": follow}
            if prior is None:
                added.append(item)
            elif follow is None:
                removed.append(item)
            elif prior != follow:
                changed.append(item)
        return {"added": added, "removed": removed, "changed": changed}

    def schema_diff_bundle(self, bundle):
        snapshot = self._snapshot_state(bundle)
        current = self._current_state()
        return self._schema_diff_states(current, snapshot)

    def _save_records(self, records):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(records, ensure_ascii=False, indent=2) + "\n"
        fd, temp_name = tempfile.mkstemp(dir=str(self.path.parent),
                                         prefix=self.path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(data)
            os.replace(temp_name, self.path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    def apply_bundle(self, bundle, expected_bundle=None):
        snapshot = self._snapshot_state(bundle)
        expected = None if expected_bundle is None else self._snapshot_state(expected_bundle)
        current = self._current_state()
        if expected is not None and current != expected:
            raise ValueError("catalog does not match expected snapshot")
        diff = self._diff_states(current, snapshot)
        if diff["added"] or diff["removed"] or diff["changed"]:
            ordered = _toposort(snapshot)
            self._save_records({key: snapshot[key] for key in ordered})
        return diff


def _parse_max_depth(raw):
    if raw is None:
        return None
    if not re.fullmatch(r"[0-9]+", raw) or int(raw) <= 0:
        raise ValueError("invalid max_depth")
    return int(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="samples/catalog.json")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("register").add_argument("file")
    commands.add_parser("import", help="import a bundle of dataset metadata").add_argument("file")
    commands.add_parser("diff", help="compare the catalog with a read-only snapshot bundle").add_argument("file")
    commands.add_parser("preview",
                        help="diff a snapshot bundle and preview the downstream impact of the change"
                        ).add_argument("file")
    commands.add_parser("schema-diff",
                        help="review read-only field additions, removals and type changes against "
                             "a complete snapshot bundle"
                        ).add_argument("file")
    commands.add_parser("relation-diff",
                        help="review read-only transitive dependency relation changes against "
                             "a complete snapshot bundle"
                        ).add_argument("file")
    apply_cmd = commands.add_parser("apply",
                                    help="apply a snapshot bundle as the complete catalog state")
    apply_cmd.add_argument("file")
    apply_cmd.add_argument("--expected",
                           help="expected current-state snapshot file; the apply is rejected "
                                "unless the catalog matches it")
    commands.add_parser("describe").add_argument("id")
    commands.add_parser("dependencies").add_argument("id")
    impact = commands.add_parser("impact")
    impact.add_argument("id")
    impact.add_argument("--max-depth")
    upstream = commands.add_parser("upstream")
    upstream.add_argument("id")
    upstream.add_argument("--max-depth")
    export = commands.add_parser("export")
    export.add_argument("--id", action="append", dest="ids")
    export_csv = commands.add_parser(
        "export-csv",
        help="export the selected datasets and their upstream closure as a field-level CSV "
             "data dictionary")
    export_csv.add_argument("--id", action="append", dest="ids")
    common_upstream = commands.add_parser(
        "common-upstream",
        help="find the common upstream datasets shared by repeated --id targets")
    common_upstream.add_argument("--id", action="append", dest="ids")
    between = commands.add_parser(
        "between",
        help="report every node and edge lying on a directed path from SOURCE to TARGET")
    between.add_argument("source")
    between.add_argument("target")
    search = commands.add_parser("search")
    search.add_argument("query", nargs="?", default="")
    search.add_argument("--field-type", dest="field_type")
    search.add_argument("--tag", action="append", dest="tags")
    search.add_argument("--owner", dest="owner")
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
        elif args.command == "schema-diff":
            result = catalog.schema_diff_bundle(json.loads(Path(args.file).read_text(encoding="utf-8")))
        elif args.command == "relation-diff":
            result = catalog.relation_diff_bundle(json.loads(Path(args.file).read_text(encoding="utf-8")))
        elif args.command == "apply":
            bundle = json.loads(Path(args.file).read_text(encoding="utf-8"))
            if args.expected is None:
                result = catalog.apply_bundle(bundle)
            else:
                expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
                if expected is None:
                    raise ValueError("bundle must be an object")
                result = catalog.apply_bundle(bundle, expected)
        elif args.command == "export":
            result = catalog.export(None if args.ids is None else args.ids)
        elif args.command == "export-csv":
            sys.stdout.write(catalog.export_csv(None if args.ids is None else args.ids))
            return 0
        elif args.command == "common-upstream":
            result = catalog.common_upstream([] if args.ids is None else args.ids)
        elif args.command == "between":
            result = catalog.between(args.source, args.target)
        elif args.command == "impact":
            result = catalog.impact(args.id, max_depth=_parse_max_depth(args.max_depth))
        elif args.command == "upstream":
            result = catalog.upstream(args.id, max_depth=_parse_max_depth(args.max_depth))
        elif args.command == "search":
            result = catalog.search(args.query, args.field_type, args.tags, args.owner)
        else:
            result = getattr(catalog, args.command)(args.id)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
