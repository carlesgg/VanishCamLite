from enum import Enum, auto
import time


class Visibility(Enum):
    VISIBLE = auto()
    DISAPPEARING = auto()
    INVISIBLE = auto()
    APPEARING = auto()


class VisibilityController:
    def __init__(self, transition_seconds: float = 0.5):
        self.state = Visibility.VISIBLE
        self.transition_seconds = transition_seconds
        self.transition_started = 0.0

    def shush_detected(self) -> None:
        if self.state is Visibility.VISIBLE:
            self.state = Visibility.DISAPPEARING
            self.transition_started = time.monotonic()

    def reappear(self) -> None:
        if self.state is Visibility.INVISIBLE:
            self.state = Visibility.APPEARING
            self.transition_started = time.monotonic()

    def alpha(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        elapsed = now - self.transition_started
        if self.state is Visibility.DISAPPEARING:
            progress = min(1.0, elapsed / self.transition_seconds)
            if progress >= 1.0:
                self.state = Visibility.INVISIBLE
            return 1.0 - progress
        if self.state is Visibility.APPEARING:
            progress = min(1.0, elapsed / self.transition_seconds)
            if progress >= 1.0:
                self.state = Visibility.VISIBLE
            return progress
        return 0.0 if self.state is Visibility.INVISIBLE else 1.0
