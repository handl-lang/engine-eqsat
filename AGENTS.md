<!-- myco:agent-awareness -->

## Agent awareness

- Session start / any work discussion: run `myco status --prompt`
  before proposing next steps — know the board first.
- User describes new work → offer to capture it (`myco todo add`,
  `issue new`, `proposal new`). Don't wait to be asked.
- Picking up a task: `myco resume <ID>` is the briefing.
- Wrapping up: `myco handoff <ID>` anything resumable; keep statuses
  honest (claimed → in-progress → blocked/done).

<!-- /myco:agent-awareness -->

# Agent guide — engine-eqsat

The e-graph **search interpretation** over kernel `Term`s — ADR-0174
phase P2 / ADR-0175's `eqsat → engine-eqsat-* + kit-eqsat` dissolution
row. Crate `hnl-engine-eqsat`; the spike evidence and open questions
live in `handl-lang/host:project/reports/engine-eqsat-spike.md`.

## Layout

- `src/pat.rs` — the law pattern language (`Pat`/`OpPat`/`PatSource`/
  `AttrsSpec`). Kernel `Term` has no metavariables, so laws are
  authored here and compiled into `handl_egraph`'s internal term
  domain — which never appears in the public API. Module docs carry
  the bridge-encoding rules (dim = child slot 0 with LHS-root-dim
  inheritance on the RHS, exact-keyset attr matching, JSON-valued
  `Const` leaves).
- `src/law.rs` — `Law` (`lhs -> rhs` + `law`/`tags` metadata) and
  `LawError` (unbound RHS metavariable). **Unconditional only** — the
  `cond.py` guard port is an open design question (spike §6.2).
- `src/builtin.rs` — the first law pack: `id_add`, `id_mul`,
  `double_neg`, `sub_to_add` (guard-free `SIMPLIFICATION_RULES`
  ports from `mlopt_core.laws.tensor`) + `reshape_flatten` (the
  attrs-bind demo).
- `src/engine.rs` — the public surface: `saturate`, `extract_best`,
  `optimize`, `size_cost`, `SaturateOptions`/`OptimizeOptions`/
  `StopPolicy`, `Saturated`/`OptimizeResult`, `Error`. Kernel
  `Term`s in and out; the saturated `EGraph` (with `ProofEdge`/
  `Application` provenance) stays inspectable on the result.

## Substrate deps

`handl-egraph` (`handl-lang/egraph` — the finished e-graph port),
`hnl-engine` (`handl-lang/engine` — the `Term` kernel), `serde_json`.
All git `branch = "main"` deps; the `[patch]` dev-link tables in
`Cargo.toml` resolve them to sibling checkouts locally (regenerate
with `scripts/dev-link.sh` if the dep set changes — the markers are
`# dev-link: begin/end`).

## Verification

```sh
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps
```

`ci.sh` runs the same gate (fmt → clippy → test → doc is *not* the
layout here — it is egraph's `doctor|check|test|ci` stage model;
run `bash ci.sh` for the full gate).
