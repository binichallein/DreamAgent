"""API routes."""

from kimi_cli.web.api import config, dream, open_in, sessions, speech, system

config_router = config.router
dream_router = dream.router
sessions_router = sessions.router
speech_router = speech.router
system_router = system.router
work_dirs_router = sessions.work_dirs_router
open_in_router = open_in.router

__all__ = [
    "config_router",
    "dream_router",
    "open_in_router",
    "sessions_router",
    "speech",
    "speech_router",
    "system_router",
    "work_dirs_router",
]
