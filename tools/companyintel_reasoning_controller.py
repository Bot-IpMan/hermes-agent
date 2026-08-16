"""Code-owned validation facade for CompanyIntel LLM reasoning proposals.

The controller does not call an LLM and does not mutate graph state. It validates
LLM-shaped proposals, returns typed decisions, and builds report metadata that can
be persisted by the durable graph runtime.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


_ALLOWED_AMBIGUITY_DECISIONS = {
    "KEEP_BOTH_CANDIDATES",
    "MERGE_CANDIDATES",
    "REJECT_CANDIDATE",
    "REQUIRE_MORE_EVIDENCE",
    "MARK_INCONCLUSIVE",
}


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}:{hashlib.sha256(payload.encode()).hexdigest()[:32]}"


def _clean_list(values: Iterable[Any]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


class CompanyIntelReasoningController:
    """Validate reasoning proposals without granting them graph authority."""

    def __init__(self, *, llm_request=None):
        self._llm_request = llm_request

    def _request(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._llm_request is not None:
            return self._llm_request(operation, payload)
        from tools.companyintel_openrouter_reasoning import request_json
        return request_json(operation=operation, system="You are the CompanyIntel reasoning controller. Return JSON only.", payload=payload)

    def synthesize_report_with_llm(self, *, graph_digest: dict[str, Any]) -> dict[str, Any]:
        result = self._request("synthesize_report", graph_digest)
        text = str(result.get("report_text", "")).strip()
        claims = result.get("claims", [])
        if not text or not isinstance(claims, list):
            raise ValueError("reasoning synthesis must return report_text and claims")
        evidence_ids = set(graph_digest.get("evidence_ids", []))
        for claim in claims:
            if not isinstance(claim, dict) or not claim.get("text"):
                raise ValueError("reasoning claim must contain text")
            unknown = set(claim.get("evidence_ids", [])) - evidence_ids
            if unknown:
                raise ValueError("reasoning claim references unknown evidence")
        return {"report_text": text[:200000], "claims": claims, "model": "nvidia/nemotron-3-ultra-550b-a55b:free", "provider": "openrouter"}

    def review_report_with_llm(self, *, report: dict[str, Any]) -> dict[str, Any]:
        result = self._request("review_report", report)
        verdict = str(result.get("verdict", "")).strip().upper()
        if verdict not in {"PASS", "PASS_WITH_LIMITATIONS", "REVISE", "BLOCK"}:
            raise ValueError("reasoning review returned invalid verdict")
        return {"verdict": verdict, "findings": result.get("findings", []), "review_text": str(result.get("review_text", "")).strip()[:50000], "review_label": "same-model-review", "independent_review": False, "model": "nvidia/nemotron-3-ultra-550b-a55b:free", "provider": "openrouter"}

    def generate_hypotheses_with_llm(self, *, payload: dict[str, Any], evidence_ids: set[str]) -> list[dict[str, Any]]:
        result = self._request("generate_hypotheses", payload)
        raw = result.get("hypotheses", [])
        if not isinstance(raw, list):
            raise ValueError("reasoning hypotheses must be a list")
        normalized = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("each reasoning hypothesis must be an object")
            normalized.append(self.generate_hypothesis(
                run_id=payload["run_id"],
                kind=str(item.get("kind", "")).strip(),
                subject_node_ids=item.get("subject_node_ids", []),
                supporting_evidence_ids=item.get("supporting_evidence_ids", []),
                evidence_ids=evidence_ids,
                claim=item.get("claim", ""),
                confidence=float(item.get("confidence", 0)),
            ))
        return normalized

    def select_equal_priority_pivot_with_llm(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        result = self._request("select_equal_priority_pivot", payload)
        return self.select_equal_priority_pivot(run_id=payload["run_id"], node_id=payload["node_id"], eligible_pivots=payload["eligible_pivots"], selected_pivot=str(result.get("selected_pivot", "")), rationale=str(result.get("rationale", "")))

    def interpret_ambiguity_with_llm(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        result = self._request("interpret_ambiguity", payload)
        return self.interpret_ambiguity(run_id=payload["run_id"], ambiguity_type=payload["ambiguity_type"], subject_ids=payload["subject_ids"], evidence_ids=payload["evidence_ids"], decision=str(result.get("decision", "")), conflicts=payload.get("conflicts", []), rationale=str(result.get("rationale", "")))

    def decide_weak_branch_with_llm(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        result = self._request("decide_weak_branch", payload)
        return self.decide_weak_branch(run_id=payload["run_id"], clue_id=payload["clue_id"], clue_strength=payload["clue_strength"], allowed_pivots=payload["allowed_pivots"], decision=str(result.get("decision", "")), max_new_tasks=int(payload["max_new_tasks"]), max_budget_cost=int(payload["max_budget_cost"]), current_depth=int(payload["current_depth"]), max_depth=int(payload["max_depth"]), rationale=str(result.get("rationale", "")))

    def generate_hypothesis(
        self, *, run_id: str, kind: str, subject_node_ids: list[str],
        supporting_evidence_ids: list[str], evidence_ids: set[str],
        claim: str = "", confidence: float = 0.0,
    ) -> dict[str, Any]:
        if not run_id or not kind or not subject_node_ids:
            raise ValueError("run_id, kind and subject_node_ids are required")
        supporting = _clean_list(supporting_evidence_ids)
        unknown = sorted(set(supporting) - set(evidence_ids))
        if unknown:
            raise ValueError(f"hypothesis references unknown evidence: {', '.join(unknown)}")
        if not 0 <= float(confidence) <= 1:
            raise ValueError("hypothesis confidence must be between 0 and 1")
        return {
            "decision_type": "GENERATE_HYPOTHESIS",
            "hypothesis_id": _stable_id("hypothesis", run_id, kind, subject_node_ids, supporting),
            "run_id": run_id,
            "kind": kind,
            "claim": str(claim).strip()[:4000],
            "subject_node_ids": _clean_list(subject_node_ids),
            "supporting_evidence_ids": supporting,
            "confidence": float(confidence),
            "status": "HYPOTHESIS",
        }

    def select_equal_priority_pivot(
        self, *, run_id: str, node_id: str, eligible_pivots: list[dict[str, Any]],
        selected_pivot: str, rationale: str,
    ) -> dict[str, Any]:
        if not eligible_pivots or not rationale.strip():
            raise ValueError("eligible_pivots and rationale are required")
        highest = max(int(item.get("priority", -1)) for item in eligible_pivots)
        selected = next((item for item in eligible_pivots if item.get("pivot_type") == selected_pivot), None)
        if selected is None:
            raise ValueError("selected pivot is not eligible")
        if int(selected.get("priority", -1)) != highest:
            raise ValueError("selected pivot is not highest-priority")
        return {
            "decision_type": "SELECT_EQUAL_PRIORITY_PIVOT",
            "decision_id": _stable_id("pivot-decision", run_id, node_id, selected_pivot),
            "run_id": run_id,
            "node_id": node_id,
            "selected_pivot": selected_pivot,
            "eligible_pivots": eligible_pivots,
            "rationale": rationale.strip()[:2000],
        }

    def interpret_ambiguity(
        self, *, run_id: str, ambiguity_type: str, subject_ids: list[str],
        evidence_ids: list[str], decision: str, conflicts: list[str], rationale: str,
    ) -> dict[str, Any]:
        if decision not in _ALLOWED_AMBIGUITY_DECISIONS:
            raise ValueError("unsupported ambiguity decision")
        if decision == "MERGE_CANDIDATES" and conflicts:
            raise ValueError("cannot merge candidates with hard conflict")
        if not subject_ids or not evidence_ids or not rationale.strip():
            raise ValueError("ambiguity requires subjects, evidence and rationale")
        return {
            "decision_type": "INTERPRET_AMBIGUITY",
            "decision_id": _stable_id("ambiguity", run_id, ambiguity_type, subject_ids, evidence_ids),
            "run_id": run_id,
            "ambiguity_type": ambiguity_type,
            "subject_ids": _clean_list(subject_ids),
            "evidence_ids": _clean_list(evidence_ids),
            "decision": decision,
            "conflicts": _clean_list(conflicts),
            "rationale": rationale.strip()[:4000],
            "status": "PROPOSED",
        }

    def compare_candidates_with_llm(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        result = self._request("compare_candidates", payload)
        comparison = self.compare_candidates(result.get("candidates", payload.get("candidates", [])))
        comparison["decision_type"] = "COMPARE_CANDIDATES"
        comparison["run_id"] = payload["run_id"]
        comparison["rationale"] = str(result.get("rationale", "")).strip()[:4000]
        return comparison

    def analyze_contradictions_with_llm(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        result = self._request("analyze_contradictions", payload)
        analyzed = self.analyze_contradictions(run_id=payload["run_id"], contradictions=result.get("contradictions", payload.get("contradictions", [])))
        analyzed["rationale"] = str(result.get("rationale", "")).strip()[:4000]
        return analyzed

    def compare_candidates(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        normalized = []
        for candidate in sorted((dict(candidate) for candidate in candidates), key=lambda item: str(item.get("candidate_id", ""))):
            if candidate.get("conflicting_dimensions") or candidate.get("missing_dimensions") or candidate.get("status") == "AMBIGUOUS":
                candidate["status"] = "AMBIGUOUS"
            normalized.append(candidate)
        if not normalized:
            return {"candidates": [], "dominance": "NONE", "recommended_decision": "REQUIRE_MORE_EVIDENCE", "verified_candidate_ids": []}
        safe_verified = [
            item["candidate_id"] for item in normalized
            if item.get("status") in {"HIGH_CONFIDENCE", "VERIFIED"}
            and not item.get("conflicting_dimensions")
            and not item.get("missing_dimensions")
        ]
        scores = {float(item.get("score", 0)) for item in normalized}
        dominance = "SINGLE" if len(safe_verified) == 1 and len(scores) > 1 else "NONE"
        return {
            "candidates": normalized,
            "dominance": dominance,
            "recommended_decision": "SELECT_CANDIDATE" if dominance == "SINGLE" else "REQUIRE_MORE_EVIDENCE",
            "verified_candidate_ids": safe_verified if dominance == "SINGLE" else [],
        }

    def analyze_contradictions(self, *, run_id: str, contradictions: list[dict[str, Any]]) -> dict[str, Any]:
        normalized = []
        for item in contradictions:
            evidence_ids = _clean_list(item.get("evidence_ids", []))
            normalized.append({**dict(item), "contradiction_id": item.get("contradiction_id") or _stable_id("contradiction", run_id, item), "evidence_ids": evidence_ids, "status": item.get("status", "UNRESOLVED")})
        return {"decision_type": "ANALYZE_CONTRADICTIONS", "run_id": run_id, "contradictions": normalized, "unresolved": [item["contradiction_id"] for item in normalized if item["status"] == "UNRESOLVED"]}

    def decide_weak_branch(
        self, *, run_id: str, clue_id: str, clue_strength: str,
        allowed_pivots: list[dict[str, Any]], decision: str,
        max_new_tasks: int, max_budget_cost: int, current_depth: int,
        max_depth: int, rationale: str,
    ) -> dict[str, Any]:
        if clue_strength != "weak":
            raise ValueError("weak branch decision requires weak clue")
        if decision not in {"BRANCH", "DO_NOT_BRANCH", "REQUEST_STRONGER_EVIDENCE"}:
            raise ValueError("unsupported weak branch decision")
        if decision == "BRANCH" and current_depth >= max_depth:
            raise ValueError("weak branch depth limit exceeded")
        if max_new_tasks < 0 or max_budget_cost < 0 or not rationale.strip():
            raise ValueError("weak branch limits and rationale are required")
        return {
            "decision_type": "WEAK_CLUE_BRANCH",
            "decision_id": _stable_id("weak-branch", run_id, clue_id),
            "run_id": run_id,
            "clue_id": clue_id,
            "branch": decision == "BRANCH",
            "decision": decision,
            "allowed_pivots": allowed_pivots,
            "max_new_tasks": int(max_new_tasks),
            "max_budget_cost": int(max_budget_cost),
            "current_depth": int(current_depth),
            "max_depth": int(max_depth),
            "rationale": rationale.strip()[:2000],
        }

    def build_report_artifact(self, *, run_id: str, graph_revision: str, evidence_ids: list[str], candidates: list[dict[str, Any]], contradictions: list[dict[str, Any]], unresolved_ambiguity: dict[str, Any], source_limitations: list[str], report_text: str, model: str, provider: str, review_text: str, gate: dict[str, Any]) -> dict[str, Any]:
        comparison = self.compare_candidates(candidates)
        contradiction_result = self.analyze_contradictions(run_id=run_id, contradictions=contradictions)
        review_label = "same-model-review" if review_text.strip() else "not_reviewed"
        return {
            "report_id": _stable_id("report", run_id, graph_revision),
            "run_id": run_id,
            "graph_revision": graph_revision,
            "evidence_ids": _clean_list(evidence_ids),
            "candidate_comparison": comparison["candidates"],
            "candidate_decision": comparison["recommended_decision"],
            "contradictions": contradiction_result["contradictions"],
            "unresolved_ambiguity": unresolved_ambiguity,
            "source_limitations": _clean_list(source_limitations),
            "model": str(model).strip()[:256],
            "provider": str(provider).strip()[:128],
            "review_label": review_label,
            "independent_review": False if review_text.strip() else None,
            "review_independence": "not_independent_same_model" if review_text.strip() else "not_reviewed",
            "review_text": review_text.strip()[:50000],
            "gate_result": gate,
            "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "text": str(report_text).strip()[:200000],
        }
