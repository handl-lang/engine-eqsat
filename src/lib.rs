//! # hnl-engine-eqsat
//!
//! The e-graph **search interpretation** over kernel `Term`s —
//! ADR-0174 phase P2 / ADR-0175's `eqsat → engine-eqsat-* + kit-eqsat`
//! dissolution row. It consumes [`handl_egraph`] (the finished Rust
//! e-graph substrate: union-find + congruence closure, compiled
//! pattern matcher, bounded saturation, deterministic extraction,
//! `ProofEdge` provenance, and the total kernel↔pattern
//! `term_bridge`) and [`hnl_engine`] (the `Term` kernel) as
//! libraries, exactly as `hnl-engine-transform-rewrite` consumes
//! `hnl-engine-transform`.
//!
//! ## Boundary contract
//!
//! The public API speaks **only** [`hnl_engine::Term`]: [`optimize`],
//! [`saturate`], and [`extract_best`] take and return kernel terms.
//! `handl_egraph::Term` — which carries the metavariables kernel
//! terms lack — exists only inside [`pat`]/[`law`] compilation and
//! never appears in a signature (ADR-0174: "`egraph::Term` stays
//! engine-internal"). Laws are authored in [`pat`]'s pattern
//! language; see its module docs for the bridge-encoding rules a law
//! author must know (dim rides child slot 0, attr matching is
//! exact-keyset, `Const` leaves are JSON-valued).
//!
//! ## The P2 contracts (stage 1, ADR-0177)
//!
//! * **Law records** — [`law::Law::to_data`]/[`law::Law::from_data`]
//!   and [`lawset::LawSet::from_json`] implement the `LAW_FORMAT = 2`
//!   JSON record (design doc §1), patterns spelled in the [`pat`]
//!   authoring view. `cond`/`dspec` load **deferred**: a law carrying
//!   them refuses to fire ([`engine::Error::GuardUnsupported`]) until
//!   the stage-2 guard interpreter lands; `serializable: false` /
//!   `missing_hooks` refuse with [`engine::Error::UnserializableLaw`]
//!   — the `missing_hooks` honesty posture, never a silent drop.
//!   The reference corpus lives in `laws/*.json`.
//! * **Cost specs** — [`cost::CostSpec`] (`count`/`size`/`op_table`)
//!   prices internal terms engine-side via [`optimize_spec`] /
//!   [`extract_best_spec`]; unknown kinds are refused at load
//!   ([`cost::CostError::UnknownKind`]). The `FnMut` kernel-cost
//!   overload stays on [`optimize`]/[`extract_best`].
//! * **Engine trait** — [`engine::SearchEngine`] (`saturate`/
//!   `optimize`/`name` over [`lawset::LawSet`] + [`cost::CostSpec`]
//!   data) with [`engine::EGraphEngine`] as the sole impl.
//!
//! ## Quick start
//!
//! ```
//! use hnl_engine::Term;
//! use hnl_engine_eqsat::{self as eqsat, builtin, engine};
//! use serde_json::json;
//!
//! // add(mul(2, 1), 0)  -- all native ops, default dim
//! let input = Term::op_native(
//!     "add",
//!     vec![
//!         Term::op_native("mul", vec![Term::constant(json!(2)), Term::constant(json!(1))]),
//!         Term::constant(json!(0)),
//!     ],
//! );
//!
//! let laws = builtin::simplification();
//! let out = eqsat::optimize(
//!     &input,
//!     &laws,
//!     &mut eqsat::size_cost,
//!     &eqsat::OptimizeOptions::spec(),
//! ).unwrap();
//! assert_eq!(out.term, Term::constant(json!(2)));
//! ```
//!
//! ## What is deliberately absent (stage 1)
//!
//! * **Guard interpretation.** `cond`/`dspec` trees are carried as
//!   deferred record data — no interpreter yet (design doc §2); laws
//!   that have them load but refuse to fire.
//! * **`flops`/shape costs.** `CostSpec` ships `count`/`size`/
//!   `op_table`; parametric flops formulas need the stage-2 shape
//!   inference and price as `+∞` until then. Roofline/fusion/
//!   executor models stay Python-side as backend-profiled specs.
//! * **Certificates.** `ProofEdge`/`Application` provenance is
//!   inspectable on [`engine::Saturated::graph`]; level-2 certificate
//!   reconstruction (`proof.py`) is not ported (design doc §4).

#![warn(missing_docs)]

pub mod builtin;
pub mod cost;
pub mod engine;
pub mod law;
pub mod lawset;
pub mod pat;

pub use cost::{CostError, CostSpec, OpEntry, OpTable};
pub use engine::{
    EGraphEngine, Error, OptimizeOptions, OptimizeResult, SaturateOptions, Saturated, SearchEngine,
    StopPolicy, extract_best, extract_best_spec, optimize, optimize_spec, saturate, size_cost,
};
pub use law::{LAW_FORMAT, Law, LawError, RecordError};
pub use lawset::{LawSet, all_tags};
pub use pat::{AttrsSpec, OpPat, Pat, PatError, PatSource};
