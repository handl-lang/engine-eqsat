# engine-eqsat

`hnl-engine-eqsat` — the e-graph **search interpretation** over kernel
`Term`s.

A thin interpretation crate: it consumes `handl-egraph` (the Rust e-graph —
union-find, compiled matcher, bounded saturation, deterministic extraction,
`ProofEdge` provenance, and the total kernel↔pattern `term_bridge`) and
`hnl-engine` as libraries. The public API speaks only `hnl_engine::Term`;
law patterns live in the crate's own `pat` language and compile into the
e-graph's internal term domain, which never crosses the boundary.

```rust
use hnl_engine::Term;
use hnl_engine_eqsat::{self as eqsat, builtin};
use serde_json::json;

// add(mul(2, 1), 0)
let input = Term::op_native(
    "add",
    vec![
        Term::op_native("mul", vec![Term::constant(json!(2)), Term::constant(json!(1))]),
        Term::constant(json!(0)),
    ],
);

let out = eqsat::optimize(
    &input,
    &builtin::simplification(),   // id_add, id_mul, double_neg, sub_to_add
    &mut eqsat::size_cost,
    &eqsat::OptimizeOptions::spec(),
).unwrap();
assert_eq!(out.term, Term::constant(json!(2)));
```

## API surface

- `optimize` / `saturate` / `extract_best` — lift → saturate →
  extract → lower over kernel `Term`s; the saturated
  `EGraph` (provenance included) is returned for inspection.
- `Law` + `pat` — the law/pattern language, plus the contracts:
  - **law records** — `LAW_FORMAT = 2` JSON via
    `Law::to_data`/`from_data` and `lawset::LawSet::from_json` /
    `from_data`; `cond`/`dspec` guards are *interpreted* by
    `guard` over `handl_egraph::BoundEnv` (the `cond.py` port,
    incl. flat-map/repeat pairing guards and shape-reading
    dspec channels). The tensor channels (`tval`/`tapp`) and
    procedural `check`/`derive` hooks refuse by name — the
    `missing_hooks` honesty posture, never a silent drop.
  - **shape inference** — `shape` ports `mlopt_core.typing`'s shape
    algebra (`Tuple`/`Unknown`/`Invalid`; `None` extents for
    unproven dims).
  - **cost specs** — `cost::CostSpec` (`count`/`size`/`op_table`)
    with `optimize_spec`/`extract_best_spec`; `{"flops": …}`
    formulas price `+∞` on anything unproven.
  - **engine trait** — `engine::SearchEngine` over `LawSet` +
    `CostSpec` data, `EGraphEngine` the sole impl; `src/bin/
    eqsat-run.rs` is the CLI front end.
- **law corpus** — the canonical `LAW_FORMAT = 2` corpus (243
  records, 8 packs) lives in `handl-kit-eqsat`
  (`data/laws/*.json`), consumed here as a dev-dependency in the
  contract tests. `laws/` keeps only `reshape_flatten_demo.json`
  (the UNSOUND attrs-bind matcher demo, quarantined engine-side);
  `examples/emit_laws.rs` regenerates pack records from the `builtin`
  constructors — its output is synced into kit-eqsat, never vendored
  back.

## Status

Unconditional *and* `cond`/`dspec`-guarded law packs ported from
`mlopt_core.laws.tensor` (`simplification`, `categorical`, opt-in
`prior_art`/`pointwise_factor`/`attention`, plus the kit-side
`prim_folds` minted records), the serialized law format, the guard
interpreter, shape inference, cost specs, and the `SearchEngine` trait.
Git-install only (`publish = false`).

## Development

```bash
cargo run --manifest-path xtask/Cargo.toml -- gate   # fmt → clippy → test → doc
```

The repo's `xtask/` crate provides the gate tasks (`fmt`, `clippy`, `test`, `doc`,
`check`, `gate`); `ci.sh` is the gate entry. Deps resolve from git, so no
sibling checkouts are needed; a plain clone test-runs as-is.

## License

MIT.
