import { useState, useCallback } from "react";
import { Input } from "@/components/ui/input";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Kbd, KbdGroup } from "@/components/ui/kbd";
import type { Session } from "@/lib/api/models";
import { shortenTitle } from "@/lib/utils";
import {
  ChevronsDownUpIcon,
  ChevronsUpDownIcon,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  SearchIcon,
} from "lucide-react";
import { SessionInfoPopover } from "./session-info-popover";
import { OpenInMenu } from "./open-in-menu";
import { isMacOS } from "@/hooks/utils";
import {
  shouldCancelSessionTitleEdit,
  shouldCommitSessionTitleEdit,
} from "@/features/sessions/session-title-edit";

type ChatWorkspaceHeaderProps = {
  currentStep: number;
  sessionDescription?: string;
  currentSession?: Session;
  selectedSessionId?: string;
  isFilesPanelOpen?: boolean;
  blocksExpanded: boolean;
  onToggleBlocks: () => void;
  onToggleFilesPanel?: () => void;
  onOpenSearch: () => void;
  onOpenSidebar?: () => void;
  onRenameSession?: (sessionId: string, newTitle: string) => Promise<boolean>;
};

export function ChatWorkspaceHeader({
  currentStep: _,
  sessionDescription,
  currentSession,
  selectedSessionId,
  isFilesPanelOpen = false,
  blocksExpanded,
  onToggleBlocks,
  onToggleFilesPanel,
  onOpenSearch,
  onOpenSidebar,
  onRenameSession,
}: ChatWorkspaceHeaderProps) {
  const searchShortcutModifier = isMacOS() ? "Cmd" : "Ctrl";
  const displayTitle = sessionDescription || "Untitled session";

  // Editing state
  const [isEditing, setIsEditing] = useState(false);
  const [editingTitle, setEditingTitle] = useState("");

  const handleDoubleClick = useCallback(() => {
    if (!(onRenameSession && selectedSessionId && sessionDescription)) return;
    setIsEditing(true);
    setEditingTitle(sessionDescription);
  }, [onRenameSession, selectedSessionId, sessionDescription]);

  const handleCancelEdit = useCallback(() => {
    setIsEditing(false);
    setEditingTitle("");
  }, []);

  const handleSaveEdit = useCallback(async () => {
    if (!(selectedSessionId && onRenameSession)) {
      handleCancelEdit();
      return;
    }

    const trimmedTitle = editingTitle.trim();
    if (!trimmedTitle) {
      handleCancelEdit();
      return;
    }

    const success = await onRenameSession(selectedSessionId, trimmedTitle);
    if (success) {
      handleCancelEdit();
    }
  }, [selectedSessionId, editingTitle, onRenameSession, handleCancelEdit]);

  return (
    <div className="flex min-h-9 min-w-0 flex-col gap-2 border-b px-3 py-2 sm:flex-row sm:items-center sm:justify-between sm:px-4 lg:px-4">
      <div className="flex min-w-0 items-center gap-2">
        {onOpenSidebar ? (
          <button
            type="button"
            aria-label="Open sessions sidebar"
            className="inline-flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-secondary/60 hover:text-foreground lg:hidden"
            onClick={onOpenSidebar}
          >
            <PanelLeftOpen className="size-4" />
          </button>
        ) : null}
        <div className="min-w-0 flex-1">
          {isEditing ? (
            <Input
              autoFocus
              value={editingTitle}
              onChange={(e) => setEditingTitle(e.target.value)}
              onBlur={handleSaveEdit}
              onKeyDown={(e) => {
                e.stopPropagation();
                if (
                  shouldCommitSessionTitleEdit(
                    e.key,
                    e.nativeEvent.isComposing,
                  )
                ) {
                  e.preventDefault();
                  handleSaveEdit();
                }
                if (shouldCancelSessionTitleEdit(e.key)) {
                  e.preventDefault();
                  handleCancelEdit();
                }
              }}
              className="h-7 text-xs font-bold"
            />
          ) : (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  className="truncate bg-transparent p-0 text-left text-xs font-semibold text-foreground hover:text-primary"
                  onDoubleClick={handleDoubleClick}
                >
                  {shortenTitle(displayTitle, 60)}
                </button>
              </TooltipTrigger>
              <TooltipContent side="bottom" className="max-w-md">
                <div>{displayTitle}</div>
                {onRenameSession && sessionDescription ? (
                  <div className="text-muted-foreground text-[10px] mt-1">
                    Double-click to rename
                  </div>
                ) : null}
              </TooltipContent>
            </Tooltip>
          )}
        </div>
      </div>
      <div className="flex items-center justify-end gap-2">
        {selectedSessionId && (
          <>
            {currentSession?.workDir ? (
              <div className="hidden lg:block">
                <OpenInMenu workDir={currentSession.workDir} />
              </div>
            ) : null}

            <SessionInfoPopover
              sessionId={selectedSessionId}
              session={currentSession}
            />

            {onToggleFilesPanel ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    aria-label={
                      isFilesPanelOpen
                        ? "Hide workspace files"
                        : "Show workspace files"
                    }
                    className="relative inline-flex items-center cursor-pointer justify-center rounded-md p-2 text-muted-foreground transition-colors hover:bg-secondary/60 hover:text-foreground"
                    onClick={onToggleFilesPanel}
                  >
                    {isFilesPanelOpen ? (
                      <PanelRightClose className="size-4" />
                    ) : (
                      <PanelRightOpen className="size-4" />
                    )}
                  </button>
                </TooltipTrigger>
                <TooltipContent side="bottom">
                  {isFilesPanelOpen
                    ? "Hide workspace files"
                    : "Show workspace files"}
                </TooltipContent>
              </Tooltip>
            ) : null}

            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  aria-label="Search messages"
                  className="inline-flex items-center cursor-pointer justify-center rounded-md p-2 text-muted-foreground transition-colors hover:bg-secondary/60 hover:text-foreground"
                  onClick={onOpenSearch}
                >
                  <SearchIcon className="size-4" />
                </button>
              </TooltipTrigger>
              <TooltipContent className="flex items-center gap-2" side="bottom">
                <span>Search messages</span>
                <KbdGroup>
                  <Kbd>{searchShortcutModifier}</Kbd>
                  <span className="text-muted-foreground">+</span>
                  <Kbd>F</Kbd>
                </KbdGroup>
              </TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  aria-label={
                    blocksExpanded ? "Fold all blocks" : "Unfold all blocks"
                  }
                  className="inline-flex items-center cursor-pointer justify-center rounded-md p-2 text-muted-foreground transition-colors hover:bg-secondary/60 hover:text-foreground"
                  onClick={onToggleBlocks}
                >
                  {blocksExpanded ? (
                    <ChevronsDownUpIcon className="size-4" />
                  ) : (
                    <ChevronsUpDownIcon className="size-4" />
                  )}
                </button>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                {blocksExpanded ? "Fold all blocks" : "Unfold all blocks"}
              </TooltipContent>
            </Tooltip>
          </>
        )}
      </div>
    </div>
  );
}
