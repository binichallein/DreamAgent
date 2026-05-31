import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactElement } from "react";
import {
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  Cpu,
  Database,
  RefreshCw,
  Search,
  Wrench,
  XCircle,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import { fetchDreamMemories } from "./api";
import {
  DREAM_CATEGORIES,
  filterDreamMemories,
  formatUsefulRate,
  groupDreamMemories,
} from "./dream-utils";
import type { DreamMemory, DreamMemoryCategory } from "./types";

type CategoryFilter = DreamMemoryCategory | "all";

type Field = {
  label: string;
  value: unknown;
};

const optimizationFields: FieldSpec[] = [
  ["Environment", "environment"],
  ["Model Type", "model_type"],
  ["Model Arch", "model_arch"],
  ["Model Name", "model_name"],
  ["Model Size", "model_size"],
  ["Inference Backend", "inference_backend"],
  ["Serving Framework", "serving_framework"],
  ["Precision", "precision"],
  ["Workload", "workload"],
  ["Before Optimize", "metrics_before"],
  ["After Optimize", "metrics_after"],
  ["Objective Metric", "objective_metric"],
  ["Detail Description", "detail_description"],
  ["Applicability", "applicability"],
  ["Caveats", "caveats"],
  ["Failure Reason", "failure_reason"],
];

const environmentDebugFields: FieldSpec[] = [
  ["Environment", "environment"],
  ["Debug Type", "debug_type"],
  ["Component", "component"],
  ["Hardware", "hardware"],
  ["OS", "os"],
  ["Driver", "driver"],
  ["Runtime", "runtime"],
  ["Dependency Stack", "dependency_stack"],
  ["Inference Backend", "inference_backend"],
  ["Issue Signature", "issue_signature"],
  ["Symptoms", "symptoms"],
  ["Root Cause", "root_cause"],
  ["Solution", "solution"],
  ["Verification", "verification"],
  ["Related Backend", "related_backend"],
  ["Commands", "commands"],
  ["Error Messages", "error_messages"],
  ["Diagnostic Steps", "diagnostic_steps"],
  ["Prevention", "prevention"],
  ["Caveats", "caveats"],
  ["Risk", "risk"],
  ["Artifacts", "artifacts"],
];

type FieldSpec = [string, keyof DreamMemory];

function isEmptyValue(value: unknown): boolean {
  if (value === null || value === undefined || value === "") {
    return true;
  }
  if (Array.isArray(value)) {
    return value.length === 0;
  }
  if (typeof value === "object") {
    return Object.keys(value).length === 0;
  }
  return false;
}

function formatValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.join("\n");
  }
  if (typeof value === "object" && value !== null) {
    return Object.entries(value)
      .map(([key, entryValue]) => `${key}: ${String(entryValue)}`)
      .join("\n");
  }
  return String(value);
}

function buildFields(memory: DreamMemory): Field[] {
  const specs =
    memory.category === "optimization"
      ? optimizationFields
      : environmentDebugFields;
  return specs
    .map(([label, key]) => ({ label, value: memory[key] }))
    .filter((field) => !isEmptyValue(field.value));
}

function statusBadge(memory: DreamMemory): ReactElement | null {
  if (memory.success === true) {
    return (
      <Badge className="border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/50 dark:text-emerald-300">
        <CheckCircle2 className="size-3" />
        success
      </Badge>
    );
  }
  if (memory.success === false) {
    return (
      <Badge className="border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/50 dark:text-red-300">
        <XCircle className="size-3" />
        failed
      </Badge>
    );
  }
  return null;
}

function MemoryRow({ memory }: { memory: DreamMemory }): ReactElement {
  const fields = buildFields(memory);
  const usefulRate = formatUsefulRate(memory.useful_rate);

  return (
    <Collapsible>
      <div className="rounded-md border border-border bg-background">
        <CollapsibleTrigger className="group flex w-full items-center gap-3 px-3 py-2 text-left transition-colors hover:bg-secondary/50">
          <ChevronDown className="size-4 shrink-0 text-muted-foreground transition-transform group-data-[state=closed]:-rotate-90" />
          <div className="min-w-0 flex-1">
            <div className="flex min-w-0 flex-wrap items-center gap-1.5">
              {statusBadge(memory)}
              {memory.environment ? (
                <Badge variant="outline">{memory.environment}</Badge>
              ) : null}
              {memory.inference_backend ? (
                <Badge variant="outline">{memory.inference_backend}</Badge>
              ) : null}
              {memory.model_type ? (
                <Badge variant="outline">{memory.model_type}</Badge>
              ) : null}
            </div>
            <div className="mt-1 truncate text-sm font-medium">{memory.title}</div>
            {memory.summary ? (
              <div className="mt-0.5 truncate text-xs text-muted-foreground">
                {memory.summary}
              </div>
            ) : null}
          </div>
          <div className="hidden shrink-0 text-right text-xs text-muted-foreground md:block">
            <div>useful {usefulRate}</div>
            <div>
              chosen {memory.chosen ?? 0}
              {typeof memory.token_used === "number"
                ? ` · ${memory.token_used.toLocaleString()} tokens`
                : ""}
            </div>
          </div>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="border-t border-border px-3 py-3">
            <dl className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {fields.map((field) => (
                <div key={field.label} className="min-w-0">
                  <dt className="text-[11px] font-medium text-muted-foreground">
                    {field.label}
                  </dt>
                  <dd className="mt-1 whitespace-pre-wrap break-words text-xs leading-5 text-foreground">
                    {formatValue(field.value)}
                  </dd>
                </div>
              ))}
              <div className="min-w-0">
                <dt className="text-[11px] font-medium text-muted-foreground">
                  Stats
                </dt>
                <dd className="mt-1 whitespace-pre-wrap text-xs leading-5">
                  chosen {memory.chosen ?? 0}
                  {"\n"}useful {memory.useful_when_chosen ?? 0}
                  {"\n"}useful rate {usefulRate}
                  {memory.time ? `\ntime ${memory.time}` : ""}
                  {typeof memory.token_used === "number"
                    ? `\ntoken used ${memory.token_used.toLocaleString()}`
                    : ""}
                </dd>
              </div>
            </dl>
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}

function CategorySection({
  category,
  memories,
}: {
  category: DreamMemoryCategory;
  memories: DreamMemory[];
}): ReactElement {
  const label =
    DREAM_CATEGORIES.find((item) => item.id === category)?.label ?? category;
  const Icon = category === "optimization" ? Cpu : Wrench;

  return (
    <Collapsible>
      <section className="rounded-md border border-border bg-card">
        <CollapsibleTrigger className="group flex w-full items-center justify-between gap-3 px-4 py-3 text-left transition-colors hover:bg-secondary/50">
          <span className="flex min-w-0 items-center gap-2">
            <ChevronDown className="size-4 shrink-0 text-muted-foreground transition-transform group-data-[state=closed]:-rotate-90" />
            <Icon className="size-4 shrink-0 text-muted-foreground" />
            <span className="truncate text-sm font-medium">{label}</span>
          </span>
          <Badge variant="secondary">{memories.length}</Badge>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="space-y-2 border-t border-border p-3">
            {memories.length > 0 ? (
              memories.map((memory) => (
                <MemoryRow key={memory.id} memory={memory} />
              ))
            ) : (
              <div className="rounded-md border border-dashed border-border px-3 py-6 text-center text-xs text-muted-foreground">
                No memories in this category.
              </div>
            )}
          </div>
        </CollapsibleContent>
      </section>
    </Collapsible>
  );
}

export function DreamPage(): ReactElement {
  const [memories, setMemories] = useState<DreamMemory[]>([]);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<CategoryFilter>("all");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadMemories = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetchDreamMemories();
      setMemories(response.memories);
    } catch (loadError) {
      setError(
        loadError instanceof Error
          ? loadError.message
          : "Failed to load Dream memories.",
      );
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadMemories();
  }, [loadMemories]);

  const filtered = useMemo(
    () => filterDreamMemories(memories, query, category),
    [memories, query, category],
  );
  const grouped = useMemo(() => groupDreamMemories(filtered), [filtered]);

  return (
    <main className="flex h-full min-h-0 w-full flex-col bg-background">
      <header className="flex h-10 shrink-0 items-center justify-between border-b border-border px-4">
        <div className="flex min-w-0 items-center gap-2">
          <BrainCircuit className="size-4 text-muted-foreground" />
          <h1 className="truncate text-sm font-semibold">Dream</h1>
        </div>
        <Button
          variant="ghost"
          size="icon-xs"
          type="button"
          onClick={() => {
            loadMemories();
          }}
          aria-label="Refresh Dream memories"
          disabled={isLoading}
        >
          <RefreshCw className={cn("size-4", isLoading && "animate-spin")} />
        </Button>
      </header>

      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-4 px-4 py-4">
          <div className="flex flex-col gap-3 border-b border-border pb-4 md:flex-row md:items-center md:justify-between">
            <div className="min-w-0">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Database className="size-3.5" />
                Infer optimize memory library
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                {memories.length} memories · {filtered.length} shown
              </p>
            </div>
            <div className="flex min-w-0 flex-col gap-2 md:flex-row md:items-center">
              <div className="flex items-center gap-1 rounded-md border border-border bg-background p-0.5">
                {[
                  ["all", "全部"],
                  ["optimization", "优化记忆"],
                  ["environment_debug", "环境 Debug"],
                ].map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setCategory(value as CategoryFilter)}
                    className={cn(
                      "h-7 rounded px-2 text-xs transition-colors",
                      category === value
                        ? "bg-secondary text-foreground"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <label className="flex h-8 min-w-0 items-center gap-2 rounded-md border border-border px-2 text-xs text-muted-foreground md:w-64">
                <Search className="size-4 shrink-0" />
                <input
                  type="text"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="搜索记忆"
                  className="min-w-0 flex-1 bg-transparent text-foreground outline-none placeholder:text-muted-foreground"
                />
              </label>
            </div>
          </div>

          {error ? (
            <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
              {error}
            </div>
          ) : null}

          {!isLoading && memories.length === 0 ? (
            <div className="rounded-md border border-dashed border-border px-4 py-16 text-center">
              <BrainCircuit className="mx-auto size-8 text-muted-foreground" />
              <div className="mt-3 text-sm font-medium">No Dream memories yet</div>
              <div className="mt-1 text-xs text-muted-foreground">
                Memories will appear here after infer-optimize Dream episodes are
                written to the memory library.
              </div>
            </div>
          ) : null}

          <div className="space-y-3">
            <CategorySection
              category="optimization"
              memories={grouped.optimization}
            />
            <CategorySection
              category="environment_debug"
              memories={grouped.environment_debug}
            />
          </div>
        </div>
      </div>
    </main>
  );
}
