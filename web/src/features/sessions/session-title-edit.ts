export function shouldCommitSessionTitleEdit(
  key: string,
  isComposing = false,
): boolean {
  return key === "Enter" && !isComposing;
}

export function shouldCancelSessionTitleEdit(key: string): boolean {
  return key === "Escape";
}
