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
- `src/law.rs` — `Law` (`lhs -> rhs` + `law`/`tags`/`cond`/`dspec`
  metadata) and `LawError` (unbound RHS metavariable, guard
  refusals). `compile` wires declarative guards into
  `Rewrite::with_check`/`with_derive`.
- `src/guard.rs` — the `cond.py` guards-as-data interpreter over
  `handl_egraph::BoundEnv`: attr/scalar predicates, the shape-reading
  predicates (dispatched to `src/shape.rs`'s `pred` table), and the
  non-shape `dspec` expr algebra. The flat-map/repeat pairing guards
  and `tval`/`tapp` tensor channels refuse by name
  (`GuardError::Unsupported`) — never approximated.
- `src/shape.rs` — stage-2/3 shape inference: `Shape`
  (`Tuple`/`Unknown`/`Invalid`, `None` extents for unproven dims) +
  the `cond.py` shape-spec/predicate vocabulary + `broadcast`/
  `matmul_shape`/`numel` helpers, ported from `mlopt_core.typing`.
  `cost.rs` `{"flops": …}` formulas bind `out`/`o<i>`/`m`/`n`/`k`/
  conv dims against it — `+∞` on anything unproven.
- `src/builtin.rs` — the law packs ported from
  `mlopt_core.laws.tensor`: `simplification`/`categorical`/`all`
  (unguarded + `cond`/`dspec`-guarded members whose guards fit the
  ported vocabulary, incl. the shape-table predicates and the
  per-key `attrs_fields` attr metavariables), the opt-in
  `prior_art`/`pointwise_factor`/`attention` presets, and
  `reshape_flatten_demo` (the UNSOUND attrs-bind demo — matcher
  tests only, never a pack). Each pack doc carries the census of
  the Python members that stay out and why.
- `src/engine.rs` — the public surface: `saturate`, `extract_best`,
  `optimize`, `size_cost`, `SaturateOptions`/`OptimizeOptions`/
  `StopPolicy`, `Saturated`/`OptimizeResult`, `Error`. Kernel
  `Term`s in and out; the saturated `EGraph` (with `ProofEdge`/
  `Application` provenance) stays inspectable on the result.
- `laws/` — only `reshape_flatten_demo.json` (the UNSOUND attrs-bind
  matcher demo, quarantined engine-side). The `LAW_FORMAT = 2`
  corpus is canonical in `handl-kit-eqsat` (`data/laws/*.json`,
  ADR-0177 §1): tests decode it via the `handl-kit-eqsat`
  dev-dependency (`law_pack(name).json`), and
  `examples/emit_laws.rs` is the regeneration path from the `builtin`
  constructors — its output is synced into kit-eqsat, never vendored
  back. `prim_folds` is kit-side minted-record data with no emit arm.

## Substrate deps

`handl-egraph` (`handl-lang/egraph` — the finished e-graph port),
`hnl-engine` (`handl-lang/engine` — the `Term` kernel), `serde_json`;
dev-dep `handl-kit-eqsat` (`handl-lang/kit-eqsat` — the law corpus).
All git `branch = "main"` deps; the manifest carries no `[patch]`
tables — local sibling resolution is the gitignored workspace-root
`.cargo/config.toml` (`/home/luis/GitHub/handl-lang/.cargo/config.toml`).

## Verification

```sh
cargo run --manifest-path xtask/Cargo.toml -- gate   # fmt → clippy → test → doc
```

or `workflows/gate.yaml` for the same gate (fmt → clippy → test → doc).
