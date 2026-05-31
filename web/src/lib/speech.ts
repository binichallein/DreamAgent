import { getApiBaseUrl } from "@/hooks/utils";
import { getAuthHeader } from "./auth";

export type SpeechTranscriptionResponse = {
  text: string;
  language?: string | null;
  duration_ms: number;
};

async function readErrorMessage(response: Response): Promise<string> {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") {
      return data.detail;
    }
    if (typeof data.msg === "string") {
      return data.msg;
    }
  } catch {
    // Fall through to generic status text.
  }
  return response.statusText || "Speech transcription failed";
}

export async function transcribeSpeechBlob(
  blob: Blob,
): Promise<SpeechTranscriptionResponse> {
  const formData = new FormData();
  formData.append("file", blob, "voice.webm");

  const response = await fetch(`${getApiBaseUrl()}/api/speech/transcribe`, {
    method: "POST",
    headers: getAuthHeader(),
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json() as Promise<SpeechTranscriptionResponse>;
}
