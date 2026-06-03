import { useEffect, useMemo, useState } from "react";
import { Activity, Cpu, HardDrive, Server } from "lucide-react";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { getApiBaseUrl } from "@/hooks/utils";
import { getAuthHeader } from "@/lib/auth";
import { cn } from "@/lib/utils";

type CPUStatus = {
  model: string | null;
  logical_cores: number;
  load_average: number[];
};

type MemoryStatus = {
  total_bytes: number;
  available_bytes: number;
  used_percent: number | null;
};

type GPUStatus = {
  name: string;
  utilization_percent: number | null;
  memory_used_mib: number | null;
  memory_total_mib: number | null;
  temperature_c: number | null;
};

type SystemStatus = {
  hostname: string;
  platform: string;
  os: string;
  arch: string;
  uptime_seconds: number | null;
  cpu: CPUStatus;
  memory: MemoryStatus;
  gpus: GPUStatus[];
};

const POLL_INTERVAL_MS = 15_000;

async function fetchSystemStatus(signal?: AbortSignal): Promise<SystemStatus> {
  const response = await fetch(`${getApiBaseUrl()}/api/system/status`, {
    headers: getAuthHeader(),
    signal,
  });

  if (!response.ok) {
    throw new Error("Failed to load system status");
  }

  return response.json() as Promise<SystemStatus>;
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let amount = value;
  let unitIndex = 0;
  while (amount >= 1024 && unitIndex < units.length - 1) {
    amount /= 1024;
    unitIndex += 1;
  }
  return `${amount.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatMib(value: number | null): string {
  if (value === null) {
    return "-";
  }
  if (value >= 1024) {
    return `${(value / 1024).toFixed(1)} GiB`;
  }
  return `${value} MiB`;
}

function formatUptime(seconds: number | null): string {
  if (seconds === null || seconds < 0) {
    return "-";
  }
  const days = Math.floor(seconds / 86_400);
  const hours = Math.floor((seconds % 86_400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (days > 0) {
    return `${days}d ${hours}h`;
  }
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m`;
}

function formatGpuSummary(gpus: GPUStatus[]): string {
  const gpu = gpus[0];
  if (!gpu) {
    return "none";
  }
  const utilization =
    gpu.utilization_percent === null ? "-" : `${gpu.utilization_percent}%`;
  const memory =
    gpu.memory_used_mib === null || gpu.memory_total_mib === null
      ? "-"
      : `${formatMib(gpu.memory_used_mib)}/${formatMib(gpu.memory_total_mib)}`;
  return `${utilization} ${memory}`;
}

export function SystemStatusBadge() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    let disposed = false;
    let controller: AbortController | null = null;

    const loadStatus = async () => {
      controller?.abort();
      controller = new AbortController();
      try {
        const nextStatus = await fetchSystemStatus(controller.signal);
        if (!disposed) {
          setStatus(nextStatus);
          setHasError(false);
        }
      } catch (error) {
        if (disposed) {
          return;
        }
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
        setHasError(true);
      }
    };

    void loadStatus();
    const intervalId = window.setInterval(() => {
      void loadStatus();
    }, POLL_INTERVAL_MS);

    return () => {
      disposed = true;
      controller?.abort();
      window.clearInterval(intervalId);
    };
  }, []);

  const cpuSummary = useMemo(() => {
    if (!status) {
      return "CPU -";
    }
    const load = status.cpu.load_average[0];
    const loadText = typeof load === "number" ? load.toFixed(2) : "-";
    return `CPU ${loadText}/${status.cpu.logical_cores || "-"}`;
  }, [status]);

  const memorySummary = useMemo(() => {
    if (!status || status.memory.used_percent === null) {
      return "MEM -";
    }
    return `MEM ${status.memory.used_percent.toFixed(1)}%`;
  }, [status]);

  const gpuSummary = useMemo(
    () => (status ? formatGpuSummary(status.gpus) : "-"),
    [status],
  );

  const hostname = status?.hostname ?? (hasError ? "System offline" : "System");

  return (
    <div className="pointer-events-none fixed top-1.5 right-28 z-30 hidden max-w-[calc(100vw-20rem)] xl:flex">
      <HoverCard openDelay={200} closeDelay={120}>
        <HoverCardTrigger asChild>
          <button
            type="button"
            className={cn(
              "pointer-events-auto inline-flex h-8 max-w-full items-center gap-2 overflow-hidden rounded-md border border-border bg-background/95 px-2.5 text-[11px] text-muted-foreground shadow-sm backdrop-blur transition-colors hover:bg-secondary/70 hover:text-foreground",
              hasError && "border-red-200 text-red-700 dark:border-red-900/70 dark:text-red-300",
            )}
          >
            <Server className="size-3.5 shrink-0" />
            <span className="max-w-32 truncate font-medium text-foreground">
              {hostname}
            </span>
            <span className="h-3 w-px shrink-0 bg-border" />
            <Cpu className="size-3.5 shrink-0" />
            <span className="shrink-0">{cpuSummary}</span>
            <HardDrive className="size-3.5 shrink-0" />
            <span className="shrink-0">{memorySummary}</span>
            <Activity className="size-3.5 shrink-0" />
            <span className="max-w-40 truncate">GPU {gpuSummary}</span>
          </button>
        </HoverCardTrigger>
        <HoverCardContent align="end" className="w-96 p-3">
          <div className="space-y-3 text-xs">
            <div>
              <div className="font-semibold text-foreground">
                {status?.hostname ?? "System status"}
              </div>
              <div className="mt-1 text-muted-foreground">
                {status ? `${status.platform} · ${status.arch}` : "Loading"}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div className="rounded-md border bg-muted/30 p-2">
                <div className="font-medium text-foreground">CPU</div>
                <div className="mt-1 text-muted-foreground">
                  {status?.cpu.model ?? "-"}
                </div>
                <div className="mt-1">
                  Load {status?.cpu.load_average.join(" / ") || "-"} ·{" "}
                  {status?.cpu.logical_cores ?? "-"} cores
                </div>
              </div>
              <div className="rounded-md border bg-muted/30 p-2">
                <div className="font-medium text-foreground">Memory</div>
                <div className="mt-1">
                  {status?.memory.used_percent === null ||
                  status?.memory.used_percent === undefined
                    ? "-"
                    : `${status.memory.used_percent.toFixed(1)}% used`}
                </div>
                <div className="mt-1 text-muted-foreground">
                  {status
                    ? `${formatBytes(
                        status.memory.total_bytes -
                          status.memory.available_bytes,
                      )} / ${formatBytes(status.memory.total_bytes)}`
                    : "-"}
                </div>
              </div>
            </div>
            <div className="rounded-md border bg-muted/30 p-2">
              <div className="flex items-center justify-between gap-2">
                <div className="font-medium text-foreground">GPU</div>
                <div className="text-muted-foreground">
                  Uptime {formatUptime(status?.uptime_seconds ?? null)}
                </div>
              </div>
              <div className="mt-2 space-y-1">
                {status?.gpus.length ? (
                  status.gpus.map((gpu, index) => (
                    <div
                      key={`${gpu.name}-${index}`}
                      className="flex items-center justify-between gap-3"
                    >
                      <span className="min-w-0 truncate">{gpu.name}</span>
                      <span className="shrink-0 text-muted-foreground">
                        {gpu.utilization_percent ?? "-"}% ·{" "}
                        {formatMib(gpu.memory_used_mib)}/
                        {formatMib(gpu.memory_total_mib)} ·{" "}
                        {gpu.temperature_c ?? "-"}C
                      </span>
                    </div>
                  ))
                ) : (
                  <div className="text-muted-foreground">
                    No GPU detected by nvidia-smi
                  </div>
                )}
              </div>
            </div>
          </div>
        </HoverCardContent>
      </HoverCard>
    </div>
  );
}
