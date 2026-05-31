export type DreamMemoryCategory = "optimization" | "environment_debug";

export type DreamMemory = {
  id: string;
  category: DreamMemoryCategory;
  title: string;
  summary?: string;
  environment?: string | null;
  model_type?: string | null;
  model_arch?: string | null;
  model_name?: string | null;
  model_size?: string | null;
  inference_backend?: string | null;
  serving_framework?: string | null;
  precision?: Record<string, unknown> | null;
  workload?: Record<string, unknown> | null;
  metrics_before?: Record<string, unknown> | null;
  metrics_after?: Record<string, unknown> | null;
  objective_metric?: string | null;
  success?: boolean | null;
  detail_description?: string;
  applicability?: string | null;
  caveats?: string | null;
  failure_reason?: string | null;
  debug_type?:
    | "install"
    | "build"
    | "runtime"
    | "dependency"
    | "driver"
    | "network"
    | "auth"
    | "filesystem"
    | "performance"
    | "other"
    | null;
  component?: string | null;
  hardware?: string | null;
  os?: string | null;
  driver?: string | null;
  runtime?: string | null;
  dependency_stack?: Record<string, unknown> | null;
  issue_signature?: string | null;
  symptoms?: string | null;
  root_cause?: string | null;
  solution?: string | null;
  verification?: string | null;
  related_backend?: string | null;
  risk?: string | null;
  commands?: string[];
  error_messages?: string[];
  diagnostic_steps?: string[];
  prevention?: string | null;
  artifacts?: string[];
  chosen?: number;
  useful_when_chosen?: number;
  useful_rate?: number;
  time?: string | null;
  token_used?: number | null;
  source_session_id?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type DreamMemoriesResponse = {
  memories: DreamMemory[];
};
