"""Lazily build and cache one model runner per demo.

The registry keeps the set of demos (``app.demos.DEMOS``) and, on first request
for a demo, builds its runner and caches it. Models aren't all loaded at boot —
each is loaded on its demo's first hit, then reused. Runners can also be injected
up front (used by tests) to avoid loading real weights.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from app.demos import DEMOS, Demo, get_demo


def default_runner_factory(demo: Demo, weights_dir: Path, device_name: str):
    """Build the runner for a demo, dispatching on its task."""
    if demo.task == "completion":
        from app.services.completer import LLMCompleter

        return LLMCompleter.from_demo(demo, weights_dir, device_name)
    raise NotImplementedError(f"Task '{demo.task}' is not supported yet (demo '{demo.slug}').")


class DemoRegistry:
    def __init__(
        self,
        weights_dir,
        device_name: str = "auto",
        runner_factory=default_runner_factory,
        runners: dict | None = None,
    ) -> None:
        self.weights_dir = Path(weights_dir)
        self.device_name = device_name
        self._runner_factory = runner_factory
        self._runners: dict[str, object] = dict(runners or {})
        self._lock = Lock()

    def is_available(self, demo: Demo) -> bool:
        """A demo is available if a runner is injected or its weights file exists."""
        if demo.slug in self._runners:
            return True
        return (self.weights_dir / demo.weights_filename).exists()

    def available(self) -> list[Demo]:
        return [demo for demo in DEMOS if self.is_available(demo)]

    def get(self, slug: str) -> Demo | None:
        demo = get_demo(slug)
        if demo is None or not self.is_available(demo):
            return None
        return demo

    def runner_for(self, demo: Demo):
        with self._lock:
            if demo.slug not in self._runners:
                self._runners[demo.slug] = self._runner_factory(
                    demo, self.weights_dir, self.device_name
                )
            return self._runners[demo.slug]
