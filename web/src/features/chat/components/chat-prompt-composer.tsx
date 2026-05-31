import {
  PromptInput,
  PromptInputAttachment,
  PromptInputAttachments,
  PromptInputBody,
  PromptInputButton,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
  usePromptInputAttachments,
  usePromptInputController,
} from "@ai-elements";
import type { ChatStatus } from "ai";
import type { PromptInputMessage } from "@ai-elements";
import type { GitDiffStats, Session } from "@/lib/api/models";
import type { TokenUsage } from "@/hooks/wireTypes";
import type { ActivityDetail } from "./activity-status-indicator";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { MEDIA_CONFIG } from "@/config/media";

import { FileMentionMenu } from "../file-mention-menu";
import { useFileMentions } from "../useFileMentions";
import { SlashCommandMenu } from "../slash-command-menu";
import { useSlashCommands, type SlashCommandDef } from "../useSlashCommands";
import { PromptToolbar } from "./prompt-toolbar";
import {
  ArrowUpIcon,
  Loader2Icon,
  MicIcon,
  SquareIcon,
  Maximize2Icon,
  Minimize2Icon,
} from "lucide-react";
import { toast } from "sonner";
import { transcribeSpeechBlob } from "@/lib/speech";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { GlobalConfigControls } from "@/features/chat/global-config-controls";
import {
  type ChangeEvent,
  type KeyboardEvent,
  type ReactElement,
  type SyntheticEvent,
  memo,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import type { SessionFileEntry } from "@/hooks/useSessions";

const TRAILING_WHITESPACE_REGEX = /\s$/;

type ChatPromptComposerProps = {
  status: ChatStatus;
  onSubmit: (message: PromptInputMessage) => Promise<void>;
  canSendMessage: boolean;
  currentSession?: Session;
  isUploading: boolean;
  isStreaming: boolean;
  isAwaitingIdle: boolean;
  isReplayingHistory: boolean;
  onCancel?: () => void;
  onListSessionDirectory?: (
    sessionId: string,
    path?: string,
  ) => Promise<SessionFileEntry[]>;
  gitDiffStats?: GitDiffStats | null;
  isGitDiffLoading?: boolean;
  slashCommands?: SlashCommandDef[];
  planMode?: boolean;
  onPlanModeChange?: (enabled: boolean) => void;
  dreamMode?: boolean;
  onDreamModeChange?: (enabled: boolean) => void;
  activityStatus?: ActivityDetail;
  usagePercent?: number;
  usedTokens?: number;
  maxTokens?: number;
  tokenUsage?: TokenUsage | null;
};

export const ChatPromptComposer = memo(function ChatPromptComposerComponent({
  status,
  onSubmit,
  canSendMessage,
  currentSession,
  isUploading,
  isStreaming,
  isAwaitingIdle,
  isReplayingHistory,
  onCancel,
  onListSessionDirectory,
  gitDiffStats,
  isGitDiffLoading,
  slashCommands = [],
  planMode = false,
  onPlanModeChange,
  dreamMode = false,
  onDreamModeChange,
  activityStatus,
  usagePercent,
  usedTokens,
  maxTokens,
  tokenUsage,
}: ChatPromptComposerProps): ReactElement {
  const promptController = usePromptInputController();
  const attachmentContext = usePromptInputAttachments();
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordedMimeTypeRef = useRef("audio/webm");
  const [isExpanded, setIsExpanded] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);

  const {
    isOpen: isMentionOpen,
    query: mentionQuery,
    sections: mentionSections,
    flatOptions: mentionOptions,
    activeIndex: mentionActiveIndex,
    setActiveIndex: setMentionActiveIndex,
    handleTextChange: handleMentionTextChange,
    handleCaretChange: handleMentionCaretChange,
    handleKeyDown: handleMentionKeyDown,
    selectOption: selectMentionOption,
    closeMenu: closeMentionMenu,
    workspaceStatus: mentionWorkspaceStatus,
    workspaceError: mentionWorkspaceError,
    retryWorkspace: retryMentionWorkspace,
    workspaceFileCount: mentionWorkspaceFileCount,
  } = useFileMentions({
    text: promptController.textInput.value,
    setText: promptController.textInput.setInput,
    textareaRef,
    attachments: attachmentContext.files,
    sessionId: currentSession?.sessionId,
    listDirectory: onListSessionDirectory,
  });

  const {
    isOpen: isSlashOpen,
    query: slashQuery,
    options: slashOptions,
    activeIndex: slashActiveIndex,
    setActiveIndex: setSlashActiveIndex,
    handleTextChange: handleSlashTextChange,
    handleCaretChange: handleSlashCaretChange,
    handleKeyDown: handleSlashKeyDown,
    selectOption: selectSlashOption,
    closeMenu: closeSlashMenu,
  } = useSlashCommands({
    text: promptController.textInput.value,
    setText: promptController.textInput.setInput,
    textareaRef,
    commands: slashCommands,
  });

  const handleTextareaChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => {
      const value = event.currentTarget.value;
      const caret = event.currentTarget.selectionStart;
      handleMentionTextChange(value, caret);
      handleSlashTextChange(value, caret);
    },
    [handleMentionTextChange, handleSlashTextChange],
  );

  const handleTextareaSelection = useCallback(
    (event: SyntheticEvent<HTMLTextAreaElement>) => {
      const caret = event.currentTarget.selectionStart;
      handleMentionCaretChange(caret);
      handleSlashCaretChange(caret);
    },
    [handleMentionCaretChange, handleSlashCaretChange],
  );

  const handleTextareaBlur = useCallback(() => {
    closeMentionMenu();
    closeSlashMenu();
  }, [closeMentionMenu, closeSlashMenu]);

  const handleTextareaKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      // Priority: slash menu first, then mention menu
      if (isSlashOpen) {
        handleSlashKeyDown(event);
        return;
      }
      if (isMentionOpen) {
        handleMentionKeyDown(event);
        return;
      }
    },
    [isSlashOpen, isMentionOpen, handleSlashKeyDown, handleMentionKeyDown],
  );

  const handleFileError = useCallback(
    (err: { code: string; message: string }) => {
      toast.error("File Error", { description: err.message });
    },
    [],
  );

  const handleToggleExpand = useCallback(() => {
    setIsExpanded((prev) => !prev);
  }, []);

  const stopMediaStream = useCallback(() => {
    for (const track of mediaStreamRef.current?.getTracks() ?? []) {
      track.stop();
    }
    mediaStreamRef.current = null;
  }, []);

  const appendTranscript = useCallback(
    (transcript: string) => {
      const cleaned = transcript.trim();
      if (!cleaned) {
        toast.info("No speech detected");
        return;
      }

      const current = promptController.textInput.value;
      const separator =
        current.length > 0 && !TRAILING_WHITESPACE_REGEX.test(current)
          ? " "
          : "";
      const nextValue = `${current}${separator}${cleaned}`;
      promptController.textInput.setInput(nextValue);

      window.requestAnimationFrame(() => {
        const textarea = textareaRef.current;
        if (!textarea) {
          return;
        }
        textarea.focus();
        textarea.setSelectionRange(nextValue.length, nextValue.length);
        handleMentionCaretChange(nextValue.length);
        handleSlashCaretChange(nextValue.length);
      });
    },
    [
      promptController.textInput,
      handleMentionCaretChange,
      handleSlashCaretChange,
    ],
  );

  const handleVoiceRecordingStop = useCallback(async () => {
    const audioBlob = new Blob(audioChunksRef.current, {
      type: recordedMimeTypeRef.current,
    });
    audioChunksRef.current = [];
    setIsRecording(false);
    mediaRecorderRef.current = null;
    stopMediaStream();

    if (audioBlob.size === 0) {
      toast.info("No speech recorded");
      return;
    }

    setIsTranscribing(true);
    try {
      const result = await transcribeSpeechBlob(audioBlob);
      appendTranscript(result.text);
    } catch (error) {
      toast.error("Voice input failed", {
        description:
          error instanceof Error ? error.message : "Speech transcription failed",
      });
    } finally {
      setIsTranscribing(false);
    }
  }, [appendTranscript, stopMediaStream]);

  const startVoiceRecording = useCallback(async () => {
    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices?.getUserMedia ||
      typeof MediaRecorder === "undefined"
    ) {
      toast.error("Voice input is not available in this browser");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const preferredMimeType =
        MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : MediaRecorder.isTypeSupported("audio/webm")
            ? "audio/webm"
            : "";
      const recorder = preferredMimeType
        ? new MediaRecorder(stream, { mimeType: preferredMimeType })
        : new MediaRecorder(stream);

      audioChunksRef.current = [];
      recordedMimeTypeRef.current =
        recorder.mimeType || preferredMimeType || "audio/webm";
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      recorder.onstop = () => {
        handleVoiceRecordingStop().catch((error: unknown) => {
          console.error("[ChatPromptComposer] Voice transcription failed", error);
        });
      };
      recorder.onerror = () => {
        setIsRecording(false);
        stopMediaStream();
        toast.error("Voice recording failed");
      };

      mediaStreamRef.current = stream;
      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
    } catch (error) {
      stopMediaStream();
      toast.error("Microphone access failed", {
        description:
          error instanceof Error ? error.message : "Unable to start recording",
      });
    }
  }, [handleVoiceRecordingStop, stopMediaStream]);

  const handleToggleVoiceRecording = useCallback(() => {
    if (isRecording) {
      if (mediaRecorderRef.current?.state === "recording") {
        mediaRecorderRef.current.stop();
      }
      return;
    }

    startVoiceRecording().catch((error: unknown) => {
      console.error("[ChatPromptComposer] Voice recording failed", error);
    });
  }, [isRecording, startVoiceRecording]);

  useEffect(() => {
    return () => {
      const recorder = mediaRecorderRef.current;
      if (recorder?.state === "recording") {
        recorder.onstop = null;
        recorder.stop();
      }
      stopMediaStream();
    };
  }, [stopMediaStream]);

  return (
    <div className="w-full">
      <PromptToolbar
        gitDiffStats={gitDiffStats}
        isGitDiffLoading={isGitDiffLoading}
        workDir={currentSession?.workDir}
        planMode={planMode}
        activityStatus={activityStatus}
        usagePercent={usagePercent}
        usedTokens={usedTokens}
        maxTokens={maxTokens}
        tokenUsage={tokenUsage}
      />

      <PromptInput
        accept="*"
        className={cn(
          "w-full [&_[data-slot=input-group]]:rounded-lg [&_[data-slot=input-group]]:border [&_[data-slot=input-group]]:border-border [&_[data-slot=input-group]]:bg-background [&_[data-slot=input-group]]:shadow-[0_1px_12px_rgba(15,23,42,0.08)]",
          planMode && "[&_[data-slot=input-group]]:border-dashed [&_[data-slot=input-group]]:!border-blue-200 dark:[&_[data-slot=input-group]]:!border-blue-600"
        )}
        multiple
        maxFiles={MEDIA_CONFIG.maxCount}
        onSubmit={onSubmit}
        onError={handleFileError}
      >
        <PromptInputBody className="w-full relative">
          {/* Expand/Collapse button - positioned relative to entire input body */}
          <button
            type="button"
            onClick={handleToggleExpand}
            disabled={!(canSendMessage && currentSession)}
            className="absolute top-2 right-2 z-10 p-1 cursor-pointer rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary/50 transition-colors disabled:opacity-50 disabled:pointer-events-none"
            aria-label={isExpanded ? "Collapse input" : "Expand input"}
          >
            {isExpanded ? (
              <Minimize2Icon className="size-4" />
            ) : (
              <Maximize2Icon className="size-4" />
            )}
          </button>
          <PromptInputAttachments>
            {(file) => <PromptInputAttachment data={file} />}
          </PromptInputAttachments>
          {isUploading ? (
            <Badge
              className="mb-2 bg-secondary/70 text-muted-foreground"
              variant="secondary"
            >
              <Loader2Icon className="size-4 animate-spin text-primary" />
              <span>Uploading files…</span>
            </Badge>
          ) : null}
          <div className="relative w-full flex items-start">
            <div className="flex-1 relative">
              <PromptInputTextarea
                ref={textareaRef}
                className={cn(
                  "transition-all duration-200 pr-8",
                  isExpanded
                    ? "min-h-[220px] max-h-[60vh] sm:min-h-[300px]"
                    : "min-h-10 max-h-36 sm:min-h-16 sm:max-h-48",
                )}
                placeholder={
                  !currentSession
                    ? "创建会话后开始..."
                    : isAwaitingIdle
                      ? isReplayingHistory
                        ? "正在连接..."
                        : "正在启动环境..."
                      : isStreaming
                        ? "继续输入..."
                        : "回复助手....."
                }
                aria-busy={isUploading}
                disabled={!canSendMessage || isUploading || !currentSession || isAwaitingIdle}
                onChange={handleTextareaChange}
                onSelect={handleTextareaSelection}
                onKeyUp={handleTextareaSelection}
                onClick={handleTextareaSelection}
                onBlur={handleTextareaBlur}
                onKeyDown={handleTextareaKeyDown}
              />
              {/* Slash command menu - mutually exclusive with file mention menu */}
              <SlashCommandMenu
                open={isSlashOpen && canSendMessage && !isMentionOpen}
                query={slashQuery}
                options={slashOptions}
                activeIndex={slashActiveIndex}
                onSelect={selectSlashOption}
                onHover={setSlashActiveIndex}
              />
              {/* File mention menu - only show when slash menu is not open */}
              <FileMentionMenu
                open={isMentionOpen && canSendMessage && !isSlashOpen}
                query={mentionQuery}
                sections={mentionSections}
                flatOptions={mentionOptions}
                activeIndex={mentionActiveIndex}
                onSelect={selectMentionOption}
                onHover={setMentionActiveIndex}
                workspaceStatus={mentionWorkspaceStatus}
                workspaceError={mentionWorkspaceError}
                onRetryWorkspace={retryMentionWorkspace}
                isWorkspaceAvailable={Boolean(
                  currentSession && onListSessionDirectory,
                )}
                workspaceFileCount={mentionWorkspaceFileCount}
              />
            </div>
          </div>
        </PromptInputBody>
        <PromptInputFooter className="w-full gap-2 py-1 border-none bg-transparent shadow-none">
          <PromptInputTools className="flex-1 min-w-0 flex-wrap">
            <GlobalConfigControls
              planMode={planMode}
              onPlanModeChange={onPlanModeChange}
              dreamMode={dreamMode}
              onDreamModeChange={onDreamModeChange}
            />
          </PromptInputTools>
          {isStreaming ? (
            <div className="flex items-center gap-1.5 shrink-0">
              <PromptInputButton
                aria-label="Stop generation"
                disabled={!onCancel}
                onClick={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  onCancel?.();
                }}
                size="icon-sm"
                variant="default"
                className="shrink-0"
              >
                <SquareIcon className="size-4" />
              </PromptInputButton>
              <Tooltip>
                <TooltipTrigger asChild>
                  <PromptInputSubmit
                    aria-label="Queue message"
                    size="icon-sm"
                    variant="outline"
                    className="shrink-0"
                    disabled={!(canSendMessage && currentSession)}
                  >
                    <ArrowUpIcon className="size-4" />
                  </PromptInputSubmit>
                </TooltipTrigger>
                <TooltipContent>Queue message</TooltipContent>
              </Tooltip>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 shrink-0">
              <Tooltip>
                <TooltipTrigger asChild>
                  <PromptInputButton
                    aria-label={isRecording ? "Stop voice input" : "Start voice input"}
                    disabled={
                      !canSendMessage ||
                      isAwaitingIdle ||
                      isUploading ||
                      isTranscribing ||
                      !currentSession
                    }
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      handleToggleVoiceRecording();
                    }}
                    size="icon-sm"
                    variant={isRecording ? "default" : "ghost"}
                    className={cn(
                      "shrink-0",
                      isRecording && "animate-pulse",
                    )}
                  >
                    {isTranscribing ? (
                      <Loader2Icon className="size-4 animate-spin" />
                    ) : (
                      <MicIcon className="size-4" />
                    )}
                  </PromptInputButton>
                </TooltipTrigger>
                <TooltipContent>
                  {isRecording
                    ? "Stop voice input"
                    : isTranscribing
                      ? "Transcribing..."
                      : "Voice input"}
                </TooltipContent>
              </Tooltip>
              <PromptInputSubmit
                status={isUploading ? "submitted" : status}
                disabled={
                  !canSendMessage ||
                  isAwaitingIdle ||
                  isUploading ||
                  !currentSession
                }
                className="shrink-0"
              />
            </div>
          )}
        </PromptInputFooter>
      </PromptInput>
    </div>
  );
});
