#!/usr/bin/env python3
"""Fail closed unless an on-disk model snapshot matches its pinned hub manifest."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    checked, failures = [], []
    for item in manifest["files"]:
        path = args.snapshot / item["path"]
        record = {"path": item["path"], "expected_bytes": item["size"], "actual_bytes": None, "expected_sha256": item.get("sha256"), "actual_sha256": None, "ok": False}
        if path.is_file():
            record["actual_bytes"] = path.stat().st_size
            if record["actual_bytes"] == record["expected_bytes"] and item.get("sha256"):
                record["actual_sha256"] = sha256(path)
                record["ok"] = record["actual_sha256"] == item["sha256"]
            elif record["actual_bytes"] == record["expected_bytes"]:
                record["ok"] = True
        checked.append(record)
        if not record["ok"]:
            failures.append(record["path"])
    report = {"repository": manifest["repository"], "revision": manifest["revision"], "files_checked": len(checked), "failures": failures, "files": checked}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"repository": report["repository"], "revision": report["revision"], "files_checked": len(checked), "failures": failures}, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
