"""Audit legacy bridge isolation at Python process entry and exit."""

import atexit
import json
import os
import sys

path = os.environ.get("GAUDIRT_SITE_AUDIT_LOG")
if path:
    assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
    last_exit_pid = None

    def audit(stage):
        global last_exit_pid
        pid = os.getpid()
        if stage == "exit" and last_exit_pid == pid:
            return
        bad_module = any(name == "habana_frameworks" or name.startswith("habana_frameworks.")
                         for name in sys.modules)
        with open("/proc/self/maps", encoding="utf-8") as maps:
            mapped = maps.read().lower()
        bad_map = any(name in mapped for name in ("libhabana_pytorch", "gaudi-pytorch-bridge", "habana_frameworks"))
        record = json.dumps({"pid": pid, "stage": stage, "module_clean": not bad_module,
                             "maps_clean": not bad_map}, sort_keys=True) + "\n"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
        try:
            os.write(descriptor, record.encode())
        finally:
            os.close(descriptor)
        if bad_module or bad_map:
            os._exit(99)
        if stage == "exit":
            last_exit_pid = pid

    audit("entry")
    atexit.register(audit, "exit")
    from multiprocessing.util import Finalize
    Finalize(None, audit, args=("exit",), exitpriority=1000)
