"""Shared macOS physical-memory and current-RSS telemetry; no model loading."""

import os
import re
import subprocess
import sys
from typing import Any

def parse_vm_stat(output: str) -> int:
    """Conservative physical availability: free + speculative pages only.

    Inactive, compressed and purgeable pages are deliberately excluded. This
    lower bound can reject a host whose caches could in fact be reclaimed.
    """
    page_size = re.search(r"page size of (\d+) bytes", output)
    if page_size is None:
        raise RuntimeError("vm_stat page size unavailable")
    pages = []
    for name in ("Pages free", "Pages speculative"):
        match = re.search(rf"^{name}:\s+(\d+)\.?$", output, re.MULTILINE)
        if match is None:
            raise RuntimeError("vm_stat free/speculative page reading unavailable")
        pages.append(int(match.group(1)))
    return int(page_size.group(1)) * sum(pages)


def _command(args: list[str]) -> str:
    return subprocess.run(
        args, check=True, capture_output=True, text=True, timeout=10,
        env={**os.environ, "LC_ALL": "C"},
    ).stdout


def host_memory_snapshot() -> dict[str, Any]:
    if sys.platform != "darwin":
        raise RuntimeError("physical preflight requires macOS memory telemetry")
    available = parse_vm_stat(_command(["/usr/bin/vm_stat"]))
    pressure = _command(["/usr/bin/memory_pressure", "-Q"])
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", pressure)
    if match is None:
        raise RuntimeError("system memory-pressure reading unavailable")
    rss = int(_command(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())]).strip()) * 1024
    return {
        "available_physical_memory_lower_bound_bytes": available,
        "availability_basis": "vm_stat free + speculative pages; excludes reclaimable caches",
        "system_memory_free_percentage": int(match.group(1)),
        "process_current_rss_bytes": rss,
    }
