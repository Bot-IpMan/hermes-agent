# CompanyIntel automatic retry policy

Status: implemented and bounded.

## Scope

This policy applies to the native Hermes `companyintel` profile and the durable
frontier stored in each run's `graph.sqlite3`. It does not apply to the generic
LookData intelligence worker.

## Current policy

1. A transient frontier/provider failure moves the affected frontier task to
   `RETRY_WAIT` while its `attempt` is below the task's `max_attempts` (default
   `3`, bounded to `1..10` when explicitly requested).
2. The durable run stores `retry_at`, `retry_reason`, and increments
   `run_state.retry_count`. The task retains its checkpoint, attempt, and
   logical task identity.
3. During the active CompanyIntel gateway execution, after a `RETRY_WAIT`
   summary the gateway waits until `retry_at`, capped at `60` seconds, and
   performs one `resume_frontier` call with the same `run_id` and worker batch.
4. The gateway continues with the bounded batch loop. It does not sleep or
   retry indefinitely. If the retry is still unavailable, the run remains
   `retry_wait`/`RESUMABLE` and the current invocation returns a deferred,
   incomplete result.
5. A later retry becomes claimable only when its durable `available_at`/`retry_at`
   deadline has elapsed. Expired leases are recovered by the frontier claim
   path, subject to the task attempt limit.
6. The UI/API `POST /v1/runs/{run_id}/resume` is the operator-controlled
   continuation path. It preserves the same `run_id`, graph, frontier,
   evidence, checkpoints, and task identities. It does not create a new run.
7. There is currently no independent background scheduler that wakes every
   `retry_wait` run after the gateway process has returned. Therefore a run in
   `retry_wait`/`RESUMABLE` is intentionally not presented as completed and
   requires the manual Resume action after the retry deadline.

## State contract

```text
transient task failure
  -> task RETRY_WAIT + retry_at/reason
  -> run RETRY_WAIT
  -> inline bounded wait (<= 60s)
  -> same-run resume_frontier once
  -> RUNNING, or RETRY_WAIT/RESUMABLE deferred
  -> manual POST /v1/runs/{run_id}/resume
  -> RUNNING, or terminal FAILED/PARTIAL/COMPLETED through normal gates
```

`RETRY_WAIT`, `RESUMABLE`, `PARTIAL`, and an open frontier are incomplete
states. They must not be projected as `completed` or `100%`.

## Operational limits

- Task attempts: default `3`; explicit task policy is clamped to `1..10`.
- Run frontier batches: at most `10` batches per active gateway invocation.
- Frontier batch size: `50` tasks.
- Inline retry sleep: at most `60` seconds per retry observation.
- Resume deadline default: `3600` seconds unless the resume request supplies a
  bounded value accepted by the runtime.
- Fixed provider/model routing is preserved; retry does not select a fallback
  model or provider.

These are safety bounds, not a completion guarantee. Provider capacity,
network access, open frontier tasks, and report gates can still leave a run
resumable.

## Authoritative implementation points

- `tools/companyintel_graph_tool.py`: task attempt limits, `RETRY_WAIT`,
  `retry_at`, `retry_reason`, lease recovery, and `resume_frontier`.
- `gateway/platforms/api_server.py`: bounded inline wait and same-run
  `resume_frontier` call during CompanyIntel execution, plus the manual Resume
  endpoint.
- `server/intelligence/hermes.go` in LookData: history/status projection must
  preserve `retry_wait`/`resumable` as incomplete states.

## Verification expectations

For a retry/resume change, verify separately:

- a task failure persists `RETRY_WAIT`, attempt, retry deadline, and checkpoint;
- an automatic inline retry uses the same `run_id` and does not create a new
  graph database;
- repeated unavailable retries stop at the bounded policy and return
  `RETRY_WAIT`/`RESUMABLE`;
- manual Resume is idempotently bound to the same run and preserves graph,
  frontier, evidence, and event history;
- the UI exposes Resume for resumable states and never reports them as
  completed.

A future background scheduler would be a separate P3/P2 follow-up requiring a
persistent wake/lease contract, duplicate-claim tests, restart recovery, and a
new operator-visible configuration. It is intentionally not implied by this
policy.
