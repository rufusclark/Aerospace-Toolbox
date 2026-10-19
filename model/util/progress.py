from datetime import timedelta
from time import time
import sys


def format_elapsed(td: timedelta):
    seconds = int(td.total_seconds())
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)


class ProgressBar:
    def __init__(self, tasks: int, *, name: str = "") -> None:
        self.name = name
        self.tasks = tasks
        self.start_time = time()
        self.update(0)

    def update(self, tasks_completed: int, msg: str = "") -> None:
        percent = tasks_completed/self.tasks
        elapsed = time() - self.start_time

        bar_width = 20
        filled = int(bar_width * percent)
        bar = "#" * filled + " " * (bar_width - filled)
        name = self.name

        remaining = (elapsed / tasks_completed) * (
            self.tasks - tasks_completed) if tasks_completed else 0

        elapsed_td = timedelta(seconds=elapsed)
        remaining_td = timedelta(seconds=remaining)

        sys.stdout.write(
            f"\r\033[2K"
            f"{f"{name} " if name else ""}[{bar}] {percent:.2%} "
            f"(elapsed: {format_elapsed(elapsed_td)} - "
            f"remaining: {format_elapsed(remaining_td)})"
            f"{f" - {msg}" if msg else ""}")
        sys.stdout.flush()

        if self.tasks >= tasks_completed:
            sys.stdout.write(f"\n")
            sys.stdout.flush()

    def complete(self) -> None:
        self.update(self.tasks)
