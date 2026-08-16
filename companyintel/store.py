"""Authoritative CompanyIntel store facade; projections stay read-only."""
from __future__ import annotations
import json
from typing import Any
from tools.companyintel_graph_tool import companyintel_graph

class CompanyIntelStore:
    def call(self, action: str, run_id: str, **kwargs: Any) -> dict[str, Any]:
        result = companyintel_graph({"action": action, "run_id": run_id, **kwargs})
        payload = json.loads(result)
        if not payload.get("ok"):
            raise RuntimeError(payload.get("error", "CompanyIntel operation failed"))
        return payload

    def init_run(self, run_id: str, target_url: str, **kwargs: Any) -> dict[str, Any]:
        return self.call("init_run", run_id, target_url=target_url, **kwargs)

    def projection(self, run_id: str, view: str = "graph") -> dict[str, Any]:
        return self.call("projection", run_id, view=view)

    def summary(self, run_id: str) -> dict[str, Any]:
        return self.call("summary", run_id)

    def replay_events(self, run_id: str, after: int = 0, limit: int = 200) -> dict[str, Any]:
        return self.call("replay_events", run_id, after_event_id=after, limit=limit)

    def resume(self, run_id: str, worker_id: str = "companyintel-api") -> dict[str, Any]:
        return self.call("resume_frontier", run_id, worker_id=worker_id)
