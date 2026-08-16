# CompanyIntel implementation waves

Authoritative graph contract: `corporate-intelligence-graph/v2`.

## Wave 0 — contract unification

Implemented in `companyintel/schema.py` and `tools/validate_graph.py`:

- strict canonical node/edge/evidence fields;
- deterministic IDs;
- canonical relation enum plus relation-specific endpoint compatibility checks;
- legacy v1 `id/type/from/to` import adapter;
- canonical chain fixture in `tests/fixtures/companyintel/`;
- report validator in `tools/validate_report.py` with fail-closed completed-report gate.

## Wave 1 — deterministic graph core

Implemented in `tools/companyintel_graph_tool.py`:

- profile-local SQLite owner;
- deterministic node/edge/evidence IDs;
- immutable evidence triggers;
- priority frontier and typed mutation tools;
- checkpoint after observation and every fifth retrieval;
- reopen regression proves checkpoint, search log, graph nodes/edges/evidence survive the five-retrieval boundary;
- graph materialization validates v2 before writing projection;
- report projection stays unavailable until the durable gate is satisfied.

## Wave 2 — Pivot Investigator v2

Implemented in `tools/companyintel_pivots.py`, `companyintel/extractors/`, and
`companyintel/planners.py`:

- typed registry with explicit plans for all canonical high-value node types;
- unsupported provider/source pivots remain explicit fail-closed tasks;
- bounded site/HTML/JSON-LD/document/JS/image inventory;
- phone, email, address, name and domain-variant planners;
- explicit source outcome taxonomy.

## Wave 3 — durable run/resume

Implemented in the graph tool and API server:

- SQLite run state, checkpoints and events;
- replayable SSE endpoint;
- idempotent stop/terminal transitions;
- automatic retry policy: bounded inline retry, durable retry metadata, and explicit manual-resume boundary (`docs/companyintel-automatic-retry-policy.md`);
- terminal state persisted before cleanup;
- Hub state is a projection, not the run owner.

## Wave 4 — LookData graph UI

Implemented in the LookData frontend/API projection layer:

- graph/frontier/evidence/report routes;
- live run state and provider/retry information;
- graph, evidence, frontier, candidate and timeline views;
- same-model resume action;
- historical projection browsing;
- Playwright acceptance coverage.

## Wave 5 — source expansion

The source matrix is implemented in `companyintel/source_matrix.py`.
Marketplace, maps, social, Wayback/RDAP/DNS/CT, official registries,
Prozorro/Spending, court/IP and reverse-image classes are explicit
`SKIPPED_UNSUPPORTED` entries until a bounded, reviewed provider adapter exists.
No generic web search is reported as authoritative coverage for these classes.

## Verification commands

```text
uv run --extra dev --extra messaging --extra acp python -m pytest \
  tests/tools/test_companyintel_schema.py \
  tests/tools/test_companyintel_validators.py \
  tests/tools/test_companyintel_planners.py \
  tests/tools/test_companyintel_extractors.py \
  tests/tools/test_companyintel_graph_tool.py -q -o addopts=

python3 tools/validate_graph.py tests/fixtures/companyintel/url-phone-marketplace-legal-registry.v2.json
```
