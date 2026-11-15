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
//! ## What is deliberately absent (v1)
//!
//! * **Guards.** `cond.py`'s `check`/`derive`/`dspec` predicates have
//!   no kit-data spelling yet — [`law::Law`] is unconditional-only
//!   (spike §6.2).
//! * **Serialized laws / cost models.** Laws are Rust values; the
//!   `kit-eqsat` data format and interpretable cost DSL are open
//!   questions (spike §6.1/§6.3).
//! * **Certificates.** `ProofEdge`/`Application` provenance is
//!   inspectable on [`engine::Saturated::graph`]; level-2 certificate
//!   reconstruction (`proof.py`) is not ported (spike §6.4).

#![warn(missing_docs)]

pub mod builtin;
pub mod engine;
pub mod law;
pub mod pat;

pub use engine::{
    Error, OptimizeOptions, OptimizeResult, SaturateOptions, Saturated, StopPolicy, extract_best,
    optimize, saturate, size_cost,
};
pub use law::{Law, LawError};
pub use pat::{AttrsSpec, OpPat, Pat, PatSource};
