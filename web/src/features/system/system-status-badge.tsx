import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  Cpu,
  HardDrive,
  Monitor,
  Thermometer,
  type LucideIcon,
} from "lucide-react";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { Progress } from "@/components/ui/progress";
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

function clampPercent(value: number | null | undefined): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return 0;
  }
  return Math.min(100, Math.max(0, value));
}

function formatPercent(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "-";
  }
  return `${value.toFixed(1)}%`;
}

function ratioPercent(
  used: number | null | undefined,
  total: number | null | undefined,
): number | null {
  if (
    typeof used !== "number" ||
    typeof total !== "number" ||
    !Number.isFinite(used) ||
    !Number.isFinite(total) ||
    total <= 0
  ) {
    return null;
  }
  return (used / total) * 100;
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

type MetricBarProps = {
  icon: LucideIcon;
  label: string;
  valueText: string;
  percent: number | null | undefined;
};

function MetricBar({
  icon: Icon,
  label,
  valueText,
  percent,
}: MetricBarProps) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-1.5 text-muted-foreground">
          <Icon className="size-3.5 shrink-0" />
          <span className="truncate">{label}</span>
        </div>
        <span className="shrink-0 font-medium text-foreground">
          {valueText}
        </span>
      </div>
      <Progress value={clampPercent(percent)} className="h-1.5 bg-muted" />
    </div>
  );
}

function getCpuLoadPercent(status: SystemStatus): number | null {
  const load = status.cpu.load_average[0];
  const cores = status.cpu.logical_cores;
  if (typeof load !== "number" || cores <= 0) {
    return null;
  }
  return (load / cores) * 100;
}

export function SystemStatusBadge() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [hasError, setHasError] = useState(false);
  const [isCardOpen, setIsCardOpen] = useState(false);

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

    loadStatus().catch(() => undefined);
    const intervalId = window.setInterval(() => {
      loadStatus().catch(() => undefined);
    }, POLL_INTERVAL_MS);

    return () => {
      disposed = true;
      controller?.abort();
      window.clearInterval(intervalId);
    };
  }, []);

  const cpuLoadText = useMemo(() => {
    if (!status) {
      return "-";
    }
    const load = status.cpu.load_average[0];
    const loadText = typeof load === "number" ? load.toFixed(2) : "-";
    return `${loadText} / ${status.cpu.logical_cores || "-"}`;
  }, [status]);

  const usedMemoryBytes = status
    ? Math.max(0, status.memory.total_bytes - status.memory.available_bytes)
    : 0;

  const hostname = status?.hostname ?? (hasError ? "System offline" : "System");

  return (
    <div className="pointer-events-none fixed top-1.5 right-28 z-30 hidden xl:flex">
      <HoverCard
        open={isCardOpen}
        openDelay={200}
        closeDelay={120}
        onOpenChange={setIsCardOpen}
      >
        <HoverCardTrigger asChild>
          <button
            type="button"
            aria-label={`System status: ${hostname}`}
            onClick={() => setIsCardOpen((open) => !open)}
            className={cn(
              "pointer-events-auto relative inline-flex size-8 items-center justify-center rounded-md border border-border bg-background/95 text-muted-foreground shadow-sm backdrop-blur transition-colors hover:bg-secondary/70 hover:text-foreground",
              hasError &&
                "border-red-200 text-red-700 dark:border-red-900/70 dark:text-red-300",
            )}
          >
            <Monitor className="size-4" />
            <span
              className={cn(
                "absolute top-1.5 right-1.5 size-1.5 rounded-full",
                status && !hasError ? "bg-emerald-500" : "bg-muted-foreground",
              )}
            />
          </button>
        </HoverCardTrigger>
        <HoverCardContent align="end" sideOffset={8} className="w-[380px] p-3">
          <div className="space-y-3 text-xs">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="truncate font-semibold text-foreground">
                  {status?.hostname ?? "System status"}
                </div>
                <div className="mt-1 line-clamp-2 text-muted-foreground">
                  {status ? `${status.platform} · ${status.arch}` : "Loading"}
                </div>
              </div>
              <div className="shrink-0 rounded-md border bg-muted/30 px-2 py-1 text-muted-foreground">
                Uptime {formatUptime(status?.uptime_seconds ?? null)}
              </div>
            </div>

            {status ? (
              <>
                <div className="rounded-md border bg-muted/30 p-2.5">
                  <div className="mb-2 min-w-0 truncate font-medium text-foreground">
                    {status.cpu.model ?? "CPU"}
                  </div>
                  <MetricBar
                    icon={Cpu}
                    label="CPU load"
                    valueText={cpuLoadText}
                    percent={getCpuLoadPercent(status)}
                  />
                </div>

                <div className="rounded-md border bg-muted/30 p-2.5">
                  <MetricBar
                    icon={HardDrive}
                    label="Memory"
                    valueText={`${formatBytes(usedMemoryBytes)} / ${formatBytes(
                      status.memory.total_bytes,
                    )}`}
                    percent={status.memory.used_percent}
                  />
                </div>

                <div className="space-y-2">
                  {status.gpus.length ? (
                    status.gpus.map((gpu, index) => (
                      <div
                        key={`${gpu.name}-${index}`}
                        className="space-y-2 rounded-md border bg-muted/30 p-2.5"
                      >
                        <div className="truncate font-medium text-foreground">
                          {gpu.name}
                        </div>
                        <MetricBar
                          icon={Activity}
                          label="GPU utilization"
                          valueText={formatPercent(gpu.utilization_percent)}
                          percent={gpu.utilization_percent}
                        />
                        <MetricBar
                          icon={HardDrive}
                          label="GPU memory"
                          valueText={`${formatMib(gpu.memory_used_mib)} / ${formatMib(
                            gpu.memory_total_mib,
                          )}`}
                          percent={ratioPercent(
                            gpu.memory_used_mib,
                            gpu.memory_total_mib,
                          )}
                        />
                        <MetricBar
                          icon={Thermometer}
                          label="GPU temperature"
                          valueText={
                            gpu.temperature_c === null
                              ? "-"
                              : `${gpu.temperature_c}C`
                          }
                          percent={
                            gpu.temperature_c === null
                              ? null
                              : (gpu.temperature_c / 100) * 100
                          }
                        />
                      </div>
                    ))
                  ) : (
                    <div className="rounded-md border bg-muted/30 p-2.5 text-muted-foreground">
                      No GPU detected by nvidia-smi
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="rounded-md border bg-muted/30 p-2.5 text-muted-foreground">
                {hasError ? "Failed to load system status" : "Loading"}
              </div>
            )}
          </div>
        </HoverCardContent>
      </HoverCard>
    </div>
  );
}
