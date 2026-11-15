# hnl-engine-eqsat

The e-graph **search interpretation** over kernel `Term`s —
ADR-0174 phase P2 / ADR-0175 (`eqsat → engine-eqsat-* + kit-eqsat`).

A thin interpretation crate: it consumes [`handl-egraph`](https://github.com/handl-lang/egraph)
(the finished Rust e-graph — union-find, compiled matcher, bounded
saturation, deterministic extraction, `ProofEdge` provenance, and the
total kernel↔pattern `term_bridge`) and [`hnl-engine`](https://github.com/handl-lang/engine)
as libraries. The public API speaks only `hnl_engine::Term`; law
patterns live in the crate's own [`pat`] language and compile into
the e-graph's internal term domain, which never crosses the boundary.

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

- [`optimize`] — lift → saturate → extract → lower, one call.
- [`saturate`] / [`extract_best`] — the staged form; the saturated
  [`EGraph`](https://docs.rs/handl-egraph) (provenance included) is
  returned for inspection.
- [`Law`] + [`pat`] — the law/pattern language. Unconditional laws
  only for now; guards (`cond.py`), serialized law data
  (`kit-eqsat`), and cost-model DSLs are the open P2 questions (see
  `handl-lang/host:project/reports/engine-eqsat-spike.md` §6).

## Status

v1 scaffold + first working slice: the interpretation shell, the
pattern compiler, four ported `SIMPLIFICATION_RULES` identities, and
the attrs-bind round-trip. Git-install only (`publish = false`).

## License

MIT OR Apache-2.0
