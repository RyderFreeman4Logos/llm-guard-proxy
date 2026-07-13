# Architecture

This document distinguishes the state boundary implemented by issue #141 from
the remaining target ownership changes for configuration and workflow I/O.

## Current implementation

The workspace has three crates. The service depends directly on both state and
core, while state depends one-way on core:

```text
OpenAI-compatible clients
           |
           v
llm-guard-proxy service crate
  Axum + Tokio + Reqwest + ProxyState
           |
           v
llm-guard-proxy-state crate
  observability + evidence + budget
           |
           v
llm-guard-proxy-core crate
  policy + config + workflow + data models
```

`ProxyState` is presently a service-owned composition object. It holds the
HTTP client, listener scope, limiters, recovery coordinators, caches,
observability/evidence stores, counters, and live-request registry. Its clones
share those bounded coordination objects; `for_listener` changes only listener
scope.

The state crate owns SQLite-backed observability and evidence stores, budget
storage, redaction, retention, metrics, and live-request tracking. The core
crate still owns configuration loading/reload and stdio workflow execution;
moving those adapters to the service is outside the state-crate milestone.

## Dependency direction

The enforced dependency DAG is service -> {state, core} and state -> core:

```text
OpenAI-compatible clients                 Upstream model services
           |                                         ^
           | HTTP/SSE                                | HTTP/SSE
           v                                         |
+--------------------------- service ----------------+|
| Axum routes, CLI, listener/signal lifecycle         ||
| Reqwest upstream adapters                           ||
+-----------------------------|----------------------+|
                              v                       |
+---------------------------- state -----------------+|
| observability | evidence | budget                   ||
| stores, retention, redaction, metrics, live registry||
+-----------------------------|-----------------------+
                              v
+----------------------------- core ------------------+
| policy | configuration source/reload | stdio workflow|
| thinking, loop detection, retry decisions, models   |
+-----------------------------------------------------+

state -> SQLite observability / SQLite evidence / budget database
core  -> configuration files / stdio workflow processes
service -> core policy, configuration, and workflow contracts
```

The state layer owns durable and live operational state. It is the only layer
that owns observability, evidence, and budget storage. It may depend on core
contracts, but core must not depend on state or service. The service composes
the state and core layers at the binary entry point.

## Remaining target ownership

| Area | Target owner | Responsibility |
| --- | --- | --- |
| HTTP/SSE, CLI, TCP listeners, signals | service | Translate the OpenAI-compatible protocol, manage process lifecycle, and render responses. |
| Configuration source and reload | service | Locate/read configuration, watch or poll its source, validate new snapshots through core, and publish them to state. |
| Upstream HTTP and embedding clients | service | Own Reqwest clients and translate upstream HTTP/SSE into core/state inputs and outputs. |
| Stdio workflow execution | service | Implement the core workflow port with process/stdio I/O. |
| Shared runtime and durable state | state | Own request coordination, limits, live registry, observability, evidence, budgets, SQLite access, retention, and redaction at persistence boundaries. |
| Policy, shared models, configuration handle | core | Own pure thinking/loop/retry policy, domain types, validated configuration snapshots/handle, and the workflow port contract. |

## Target ports and adapters

| Port | Core contract | Target adapter/owner |
| --- | --- | --- |
| Configuration | `ConfigHandle` exposes validated immutable snapshots to policy and state. | Service supplies the configuration source and reload driver; state consumes the published handle. |
| Workflow | A workflow port describes the operations policy needs without process APIs. | Service supplies the stdio/process adapter. |
| Inbound OpenAI-compatible API | Core defines request-policy and result models, not HTTP routes. | Service Axum routes parse/render HTTP, JSON, and SSE. |
| Upstream generation, metadata, and embeddings | Core exposes typed inputs, decisions, and the embedding boundary. | Service HTTP adapters own Reqwest and network failure translation. |
| Observability, evidence, and budget | No core port; operational records belong to state. | State owns records, storage implementations, retention, redaction, and live coordination. |

## Target feature placement

- Core: thinking policy, loop detection, retry decisions, configuration handle,
  workflow port, and protocol-independent models.
- State: observability/evidence/budget records and stores, retention, redaction
  at persistence boundaries, request admission state, caches, and live-request
  tracking.
- Service: generic forwarding, shielded HTTP/SSE release behavior, config
  source/reload, upstream transport, process lifecycle, and the stdio workflow
  adapter.

Raw payload capture remains opt-in. The state layer must enforce redaction and
retention before sensitive data reaches durable storage.

## Target forbidden dependency edges

1. Core must not depend on state or service, nor on Axum, Reqwest, Tokio,
   filesystem/process APIs, SQLite implementations, HTTP types, or routing.
2. State must not depend on service framework or transport types. It may depend
   on core contracts and its own persistence implementation.
3. Service must not own observability, evidence, or budget persistence, and it
   must not bypass state with ad-hoc SQLite or raw-payload writes.
4. Core policy must not read configuration sources, start reload tasks, bind
   ports, construct HTTP clients, spawn processes, or emit protocol responses.
5. The stdio workflow adapter belongs in service; core owns only its port.

The state boundary is implemented cohesively. Configuration-source and workflow
adapter ownership remain future changes and must preserve the same dependency
direction when implemented.
