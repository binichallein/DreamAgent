"""System status API routes."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/system", tags=["system"])


class CPUStatus(BaseModel):
    model: str | None = None
    logical_cores: int = Field(default=0, ge=0)
    load_average: list[float] = Field(default_factory=list)


class MemoryStatus(BaseModel):
    total_bytes: int = Field(default=0, ge=0)
    available_bytes: int = Field(default=0, ge=0)
    used_percent: float | None = None


class GPUStatus(BaseModel):
    name: str
    utilization_percent: int | None = None
    memory_used_mib: int | None = None
    memory_total_mib: int | None = None
    temperature_c: int | None = None


class SystemStatus(BaseModel):
    hostname: str
    platform: str
    os: str
    arch: str
    uptime_seconds: float | None = None
    cpu: CPUStatus
    memory: MemoryStatus
    gpus: list[GPUStatus] = Field(default_factory=list)


def _read_cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        processor = platform.processor()
        return processor or None

    try:
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                _, value = line.split(":", 1)
                model = value.strip()
                return model or None
    except OSError:
        return None
    return None


def _read_uptime_seconds() -> float | None:
    uptime = Path("/proc/uptime")
    if not uptime.exists():
        return None
    try:
        first = uptime.read_text(encoding="utf-8").split()[0]
        return float(first)
    except (OSError, IndexError, ValueError):
        return None


def _read_memory_status() -> MemoryStatus:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return MemoryStatus()

    values: dict[str, int] = {}
    try:
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition(":")
            amount = value.strip().split(maxsplit=1)[0]
            if amount.isdigit():
                values[key] = int(amount) * 1024
    except OSError:
        return MemoryStatus()

    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    used_percent = None
    if total > 0:
        used_percent = round(((total - available) / total) * 100, 1)
    return MemoryStatus(
        total_bytes=total,
        available_bytes=available,
        used_percent=used_percent,
    )


def _parse_int(value: str) -> int | None:
    value = value.strip()
    return int(value) if value.isdigit() else None


def _read_gpus() -> list[GPUStatus]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        return []

    gpus: list[GPUStatus] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5 or not parts[0]:
            continue
        gpus.append(
            GPUStatus(
                name=parts[0],
                utilization_percent=_parse_int(parts[1]),
                memory_used_mib=_parse_int(parts[2]),
                memory_total_mib=_parse_int(parts[3]),
                temperature_c=_parse_int(parts[4]),
            )
        )
    return gpus


@router.get("/status", summary="Get controller machine status")
async def get_system_status() -> SystemStatus:
    """Return a lightweight snapshot of the machine running this Web UI."""
    load_average = (
        [round(value, 2) for value in os.getloadavg()]
        if hasattr(os, "getloadavg")
        else []
    )
    return SystemStatus(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        os=platform.system(),
        arch=platform.machine(),
        uptime_seconds=_read_uptime_seconds(),
        cpu=CPUStatus(
            model=_read_cpu_model(),
            logical_cores=os.cpu_count() or 0,
            load_average=load_average,
        ),
        memory=_read_memory_status(),
        gpus=_read_gpus(),
    )
