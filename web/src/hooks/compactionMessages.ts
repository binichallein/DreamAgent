import type { LiveMessage } from "./types";

export function removeCompactionIndicator(
  messages: LiveMessage[],
  compactionMessageId: string | null,
): LiveMessage[] {
  if (!compactionMessageId) {
    return messages;
  }
  return messages.filter((message) => message.id !== compactionMessageId);
}
