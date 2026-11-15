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
//! ## The P2 contracts (stages 1–3, ADR-0177)
//!
//! * **Law records** — [`law::Law::to_data`]/[`law::Law::from_data`]
//!   and [`lawset::LawSet::from_json`] implement the `LAW_FORMAT = 2`
//!   JSON record (design doc §1), patterns spelled in the [`pat`]
//!   authoring view. `cond`/`dspec` are **interpreted** by
//!   [`guard`] — the `cond.py` port over `handl_egraph::BoundEnv`
//!   (attr/scalar predicates, derive exprs, `with_check`/
//!   `with_derive` wiring) — dispatching shape-reading atoms to
//!   [`shape::pred`]. The flat-map/repeat pairing guards
//!   (`flat-pair-unsq`, `repeat-chain`, …) and the tensor channels
//!   (`tval`/`tapp`, tensor `$leaf:` mints) remain unported and
//!   refuse by name ([`engine::Error::GuardUnsupported`]).
//!   `serializable: false` / `missing_hooks` refuse with
//!   [`engine::Error::UnserializableLaw`] — the `missing_hooks`
//!   honesty posture, never a silent drop. The reference corpus
//!   lives in `laws/*.json`.
//! * **Shape inference** — [`shape`] ports `mlopt_core.typing`'s
//!   `_shape_of`/`_broadcast`/`_matmul_shape`/`_numel` over the
//!   bridge encoding (design doc §2 stage 2): `Shape::Tuple` with
//!   `None` extents for unproven dims, `Shape::Unknown` for
//!   undecidable, `Shape::Invalid` for provably ill-typed — never a
//!   guessed dim. Broadcasting, matmul, reshape/`-1` folding, view
//!   ops, reductions, conv/pool windows, and the leaf conventions
//!   (`{"__tensor__": {"shape": …}}`, `{"shape": …}`) are covered.
//! * **Cost specs** — [`cost::CostSpec`] (`count`/`size`/
//!   `op_table`) prices internal terms engine-side via
//!   [`optimize_spec`] / [`extract_best_spec`]; unknown kinds are
//!   refused at load ([`cost::CostError::UnknownKind`]).
//!   `op_table` `{"flops": "…"}` formulas evaluate a small
//!   arithmetic language over [`shape`]-inferred dims (`out`,
//!   `o<i>`, `m`/`n`/`k`, `in`/`outf`, conv `n`/`oc`/`ic`/`kh`/`kw`/
//!   `groups`) — **`+∞` whenever a referenced name is unbound**
//!   (unknown/`Invalid`/non-concrete shapes), never under-priced.
//!   The `FnMut` kernel-cost overload stays on [`optimize`]/
//!   [`extract_best`]. Roofline/fusion/executor models stay
//!   Python-side as backend-profiled specs.
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
//! ## What is deliberately absent
//!
//! * **Flat-map/repeat guards + tensor channels.** The
//!   `cond.py`/`dspec` machinery that pairs flattened axes
//!   (`flat-pair-unsq`, `flat-map-unsq`, `repeat-chain`,
//!   `repeat-heads`, `axis-align-eq`, `bcast-dim-inv`) and the
//!   tensor-value channels (`tval`/`tapp`, tensor `$leaf:` mints)
//!   are not ported — they refuse by name at compile
//!   ([`engine::Error::GuardUnsupported`]), never approximated.
//! * **Backend-profiled cost models.** Roofline/fusion/executor
//!   costs are backend algorithms, not tables — they stay
//!   Python-side until a `CostSpec` kind with a `backend` profile
//!   field is needed (design doc §3).
//! * **Certificates.** `ProofEdge`/`Application` provenance is
//!   inspectable on [`engine::Saturated::graph`]; level-2 certificate
//!   reconstruction (`proof.py`) is not ported (design doc §4).

#![warn(missing_docs)]

pub mod builtin;
pub mod cost;
pub mod engine;
pub mod guard;
pub mod law;
pub mod lawset;
pub mod pat;
pub mod shape;

pub use cost::{CostError, CostSpec, OpEntry, OpTable};
pub use engine::{
    EGraphEngine, Error, OptimizeOptions, OptimizeResult, SaturateOptions, Saturated, SearchEngine,
    StopPolicy, extract_best, extract_best_spec, optimize, optimize_spec, saturate, size_cost,
};
pub use guard::{Cond, Dspec, GuardError};
pub use law::{LAW_FORMAT, Law, LawError, RecordError};
pub use lawset::{LawSet, all_tags};
pub use pat::{AttrField, AttrsSpec, OpPat, Pat, PatError, PatSource};
