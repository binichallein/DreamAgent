import type { FileUIPart } from "ai";

export type PromptInputSubmittedFile = FileUIPart & {
  sourceFile?: File;
};

export type PromptInputAttachmentItem = PromptInputSubmittedFile & {
  id: string;
};

export function createPromptInputAttachment(
  file: File,
  id: string,
): PromptInputAttachmentItem {
  return {
    id,
    type: "file",
    url: URL.createObjectURL(file),
    mediaType: file.type,
    filename: file.name,
    sourceFile: file,
  };
}

export function preparePromptInputFiles(
  files: PromptInputAttachmentItem[],
): PromptInputSubmittedFile[] {
  return files.map(({ id: _id, ...file }) => file);
}

export async function getPromptInputUploadFile(
  filePart: PromptInputSubmittedFile,
): Promise<File> {
  if (filePart.sourceFile) {
    return filePart.sourceFile;
  }

  if (!filePart.url) {
    throw new Error("Attachment is missing file data.");
  }

  const response = await fetch(filePart.url);
  if (!response.ok) {
    throw new Error(response.statusText || "Failed to fetch attachment data.");
  }
  const blob = await response.blob();
  return new File([blob], filePart.filename ?? "unnamed_file", {
    type: filePart.mediaType ?? blob.type,
  });
}
