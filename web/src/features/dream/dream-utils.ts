import type { DreamMemory, DreamMemoryCategory } from "./types";

export const DREAM_CATEGORIES: Array<{
  id: DreamMemoryCategory;
  label: string;
}> = [
  { id: "optimization", label: "优化记忆" },
  { id: "environment_debug", label: "环境部署 Debug 记忆" },
];

export function formatUsefulRate(rate?: number): string {
  if (typeof rate !== "number" || Number.isNaN(rate)) {
    return "0%";
  }
  return `${Math.round(rate * 100)}%`;
}

export function groupDreamMemories(
  memories: DreamMemory[],
): Record<DreamMemoryCategory, DreamMemory[]> {
  return {
    optimization: memories.filter((memory) => memory.category === "optimization"),
    environment_debug: memories.filter(
      (memory) => memory.category === "environment_debug",
    ),
  };
}

export function dreamMemorySearchText(memory: DreamMemory): string {
  return [
    memory.title,
    memory.summary,
    memory.environment,
    memory.model_type,
    memory.model_arch,
    memory.model_name,
    memory.inference_backend,
    memory.serving_framework,
    memory.detail_description,
    memory.debug_type,
    memory.component,
    memory.issue_signature,
    memory.root_cause,
    memory.solution,
    memory.commands?.join(" "),
    memory.error_messages?.join(" "),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

export function filterDreamMemories(
  memories: DreamMemory[],
  query: string,
  category: DreamMemoryCategory | "all",
): DreamMemory[] {
  const normalizedQuery = query.trim().toLowerCase();
  return memories.filter((memory) => {
    if (category !== "all" && memory.category !== category) {
      return false;
    }
    if (!normalizedQuery) {
      return true;
    }
    return dreamMemorySearchText(memory).includes(normalizedQuery);
  });
}
