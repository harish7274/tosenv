"""
ToS Env Client — Python SDK
==============================
Provides synchronous (and optionally async-style) access to
the TosEnvironment server.

Usage:
    from tos_env import TosEnvClient, TosAction

    with TosEnvClient(base_url="http://localhost:8000") as client:
        obs    = client.reset(task="binary_risk", seed=42)
        result = client.step(TosAction(verdict="risky"))
        state  = client.state()
"""

from __future__ import annotations

import requests
from typing import Optional

from models import TosAction, TosObservation, TosState, TosStepResult


class TosEnvClient:
    """
    Synchronous HTTP client for the ToS OpenEnv server.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self._session: Optional[requests.Session] = None

    # ── Context manager ──────────────────────────────────────────────────

    def __enter__(self) -> "TosEnvClient":
        self._session = requests.Session()
        return self

    def __exit__(self, *_):
        if self._session:
            self._session.close()
            self._session = None

    def _s(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    # ── OpenEnv API ───────────────────────────────────────────────────────

    def reset(self, task: str = "binary_risk", seed: int = 42) -> TosObservation:
        """Start a new episode, returns initial observation."""
        resp = self._s().post(
            f"{self.base_url}/reset",
            params={"task": task, "seed": seed},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return TosObservation(**resp.json())

    def step(self, action: TosAction) -> TosStepResult:
        """Send an action, receive step result."""
        resp = self._s().post(
            f"{self.base_url}/step",
            json=action.model_dump(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return TosStepResult(**resp.json())

    def state(self) -> TosState:
        """Fetch current episode state."""
        resp = self._s().get(f"{self.base_url}/state", timeout=self.timeout)
        resp.raise_for_status()
        return TosState(**resp.json())

    def health(self) -> bool:
        """Return True if server is up."""
        try:
            resp = self._s().get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    # ── Convenience close ─────────────────────────────────────────────────

    def close(self):
        if self._session:
            self._session.close()
            self._session = None
