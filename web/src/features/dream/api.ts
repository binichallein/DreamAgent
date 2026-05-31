import { getApiBaseUrl } from "@/hooks/utils";
import { getAuthHeader } from "@/lib/auth";
import type { DreamMemoriesResponse } from "./types";

export async function fetchDreamMemories(): Promise<DreamMemoriesResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/dream/memories`, {
    headers: getAuthHeader(),
  });

  if (!response.ok) {
    let message = "Failed to load Dream memories.";
    try {
      const body = await response.json();
      if (typeof body.detail === "string") {
        message = body.detail;
      }
    } catch {
      // Keep generic message.
    }
    throw new Error(message);
  }

  return response.json() as Promise<DreamMemoriesResponse>;
}
