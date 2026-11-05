"""Verify every audited Python process entered and exited without the old bridge."""

import hashlib
import json
import sys
from pathlib import Path


def main():
    output = Path(sys.argv[1])
    summaries = []
    for name in sys.argv[2:]:
        path = Path(name)
        raw = path.read_bytes()
        by_pid = {}
        for line in raw.splitlines():
            item = json.loads(line)
            assert item["module_clean"] and item["maps_clean"], item
            assert item["stage"] in ("entry", "exit"), item
            by_pid.setdefault(item["pid"], []).append(item["stage"])
        assert by_pid, path.name
        assert all(sorted(stages) == ["entry", "exit"] for stages in by_pid.values()), by_pid
        summaries.append({"input": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                          "process_count": len(by_pid), "all_entries_and_exits_clean": True})
    assert summaries
    output.write_text(json.dumps(summaries, sort_keys=True, indent=2) + "\n")
    print(json.dumps(summaries, sort_keys=True))


if __name__ == "__main__":
    main()
