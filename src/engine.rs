//! The interpretation surface: kernel `Term`s in, saturated
//! [`EGraph`] + extracted kernel `Term` out.
//!
//! Everything here speaks `hnl_engine::Term` (re-exported as
//! [`KernelTerm`]). `handl_egraph::Term`/patterns stay inside —
//! [`crate::law::Law`] compiles them privately.

use std::collections::BTreeMap;

use handl_egraph::{
    AddTermError, ClassId, EGraph, KernelTerm, RunError, RunOptions, SaturationStats, Stop,
    StopReason, Term as ETerm, from_kernel, to_kernel,
};

use crate::cost::CostSpec;
use crate::law::{self, Law, LawError};
use crate::lawset::LawSet;

/// Why an interpretation call failed.
#[derive(Debug)]
pub enum Error {
    /// A [`Law`] failed to compile (e.g. unbound RHS metavariable).
    Law(law::LawError),
    /// Lifting a kernel term into the e-graph failed.
    AddTerm(AddTermError),
    /// Saturation failed before iterating.
    Run(RunError),
    /// Extraction found no acyclic representative.
    Extract,
    /// A law record flagged itself non-serializable
    /// (`serializable: false` / `missing_hooks`) — firing it would
    /// silently drop procedural hooks. Named refusal (design doc §1).
    UnserializableLaw(String),
    /// A law carries a `cond`/`dspec` guard as deferred data with no
    /// stage-1 interpreter — firing would silently drop the side
    /// condition. Named refusal.
    GuardUnsupported(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Law(e) => write!(f, "law compile: {e}"),
            Error::AddTerm(e) => write!(f, "add_term: {e}"),
            Error::Run(e) => write!(f, "saturate: {e}"),
            Error::Extract => f.write_str("no extractable term in e-class"),
            Error::UnserializableLaw(name) => {
                write!(f, "law '{name}' is not serializable — refusing to fire")
            }
            Error::GuardUnsupported(name) => write!(
                f,
                "law '{name}' carries a deferred cond/dspec guard — no interpreter yet"
            ),
        }
    }
}

impl std::error::Error for Error {}

impl From<law::LawError> for Error {
    fn from(e: law::LawError) -> Self {
        Error::Law(e)
    }
}

impl From<AddTermError> for Error {
    fn from(e: AddTermError) -> Self {
        Error::AddTerm(e)
    }
}

impl From<RunError> for Error {
    fn from(e: RunError) -> Self {
        Error::Run(e)
    }
}

/// Saturation bounds — the spec's `RunOptions` minus the cost hook
/// (extraction cost lives on [`optimize`]/[`extract_best`]).
#[derive(Clone, Debug)]
pub struct SaturateOptions {
    /// Iteration cap (spec default 100).
    pub max_iterations: usize,
    /// Enode cap (spec default 100_000).
    pub max_nodes: usize,
    /// `rule name -> max NEW enodes the rule may contribute` —
    /// bounded saturation for explosive rules (comm/assoc).
    pub rule_budgets: BTreeMap<String, usize>,
}

impl SaturateOptions {
    /// Spec defaults: 100 iterations / 100k enodes, no budgets.
    pub fn spec() -> Self {
        Self {
            max_iterations: 100,
            max_nodes: 100_000,
            rule_budgets: BTreeMap::new(),
        }
    }
}

impl Default for SaturateOptions {
    fn default() -> Self {
        Self::spec()
    }
}

/// A saturated e-graph plus the e-class roots of the input terms.
pub struct Saturated {
    /// The saturated graph — provenance (`merge_log`,
    /// `applications`, `enode_app`) is inspectable here.
    pub graph: EGraph,
    /// Canonical e-class id per input term, same order.
    pub roots: Vec<ClassId>,
    /// The final [`SaturationStats`].
    pub stats: SaturationStats,
}

/// The stop policy for [`optimize`] — `FixedPoint` saturates exactly;
/// `Improving` is lazy saturation (extract each iteration, stop after
/// `patience` non-improving rounds — heuristic, can stop early).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum StopPolicy {
    /// Saturate to the exact fixed point.
    #[default]
    FixedPoint,
    /// Lazy saturation with the given patience.
    Improving {
        /// Iterations without strict cost improvement before stopping.
        patience: usize,
    },
}

/// Options for [`optimize`].
#[derive(Clone, Debug)]
pub struct OptimizeOptions {
    /// Saturation bounds.
    pub saturate: SaturateOptions,
    /// Stop policy (default [`StopPolicy::FixedPoint`]).
    pub stop: StopPolicy,
}

impl OptimizeOptions {
    /// Spec-default options.
    pub fn spec() -> Self {
        OptimizeOptions {
            saturate: SaturateOptions::spec(),
            stop: StopPolicy::FixedPoint,
        }
    }
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        Self::spec()
    }
}

/// The result of [`optimize`]: the extracted best kernel `Term` plus
/// the saturated graph it came from (provenance stays inspectable).
pub struct OptimizeResult {
    /// The minimum-cost member of the root e-class, lowered back to a
    /// kernel `Term` (`to_kernel` is total — always a `Term`).
    pub term: KernelTerm,
    /// The saturated graph and its stats.
    pub saturated: Saturated,
}

/// Compile laws to rewrites (the only place [`Law`] meets
/// `handl_egraph::Rewrite` — private to the interpretation). The
/// record-honesty refusals surface as their named [`Error`]
/// variants, not wrapped [`Error::Law`].
fn compile_rules(laws: &[Law]) -> Result<Vec<handl_egraph::Rewrite>, Error> {
    laws.iter()
        .map(|law| match law.compile() {
            Ok(rule) => Ok(rule),
            Err(LawError::GuardDeferred { law, .. }) => Err(Error::GuardUnsupported(law)),
            Err(LawError::Unserializable { law, .. }) => Err(Error::UnserializableLaw(law)),
            Err(e) => Err(Error::Law(e)),
        })
        .collect()
}

/// Build the `RunOptions` for one saturation pass. The lifetime is
/// generic — `Stop::Improving` callers graft a `cost_fn` borrow on
/// afterwards (`RunOptions` is invariant in its lifetime).
fn run_options<'a>(opts: &SaturateOptions) -> RunOptions<'a> {
    RunOptions {
        max_iterations: opts.max_iterations,
        max_nodes: opts.max_nodes,
        rule_budgets: opts.rule_budgets.clone(),
        ..RunOptions::default()
    }
}

/// Lift kernel terms into a fresh [`EGraph`] and saturate under `laws`.
///
/// Each input term's root e-class is returned in [`Saturated::roots`],
/// same order — extraction targets those, not "the" root.
///
/// # Errors
///
/// [`Error::Law`] on a compile failure, [`Error::AddTerm`] if lifting
/// fails, [`Error::Run`] on a `run` failure.
pub fn saturate(
    terms: &[KernelTerm],
    laws: &[Law],
    opts: &SaturateOptions,
) -> Result<Saturated, Error> {
    let rules = compile_rules(laws)?;
    let mut graph = EGraph::new();
    let mut roots = Vec::with_capacity(terms.len());
    for t in terms {
        roots.push(graph.add_term(&from_kernel(t), "input")?);
    }
    // One `run` per root over the shared graph — the dirty frontier
    // makes repeat runs incremental.
    let mut stats = None;
    for &root in &roots {
        stats = Some(graph.run(&rules, root, &mut run_options(opts))?);
    }
    let stats = stats.unwrap_or_else(|| SaturationStats {
        iterations: 0,
        n_enodes: graph.n_enodes(),
        n_classes: graph.n_classes(),
        n_proof_edges: graph.n_proof_edges(),
        truncation_level: 0,
        rule_budgets: BTreeMap::new(),
        budget_suspended: Vec::new(),
        stop: StopReason::FixedPoint,
        improved: None,
    });
    Ok(Saturated {
        graph,
        roots,
        stats,
    })
}

/// Extract the minimum-cost representative of `root`'s e-class and
/// lower it to a kernel `Term`.
///
/// `cost` prices **kernel** terms — the wrapper lowers each candidate
/// e-graph term through `to_kernel` before pricing (total, but not
/// free; cost models that need the internal term should live
/// substrate-side for now — spike §6.3).
///
/// `None` when the class has no acyclic extractable member.
pub fn extract_best(
    graph: &mut EGraph,
    root: ClassId,
    cost: &mut dyn FnMut(&KernelTerm) -> f64,
) -> Option<KernelTerm> {
    let mut adapted = |t: &ETerm| {
        let k = to_kernel(t);
        cost(&k)
    };
    let best = graph.extract_best(root, &mut adapted, None, None)?;
    Some(to_kernel(&best))
}

/// The one-shot interpretation over **internal** terms: lift `term`,
/// saturate under `laws`, extract the cheapest member under
/// `cost_fn` (priced on `handl_egraph::Term` — no bridge crossing),
/// return it as a kernel `Term` alongside the saturated graph.
///
/// This is the shared body of [`optimize`] (kernel-priced costs) and
/// [`optimize_spec`] ([`CostSpec`] data priced engine-side).
///
/// # Errors
///
/// See [`saturate`]; plus [`Error::Extract`] when the root class has
/// no extractable member.
fn optimize_inner(
    term: &KernelTerm,
    laws: &[Law],
    cost_fn: &mut dyn FnMut(&ETerm) -> f64,
    opts: &OptimizeOptions,
) -> Result<OptimizeResult, Error> {
    let rules = compile_rules(laws)?;
    let mut graph = EGraph::new();
    let root = graph.add_term(&from_kernel(term), "input")?;

    // One adapter prices internal terms — reused by the `Improving`
    // stop's in-run extraction and the final `extract_best`.
    let mut adapted = |t: &ETerm| cost_fn(t);

    let mut run_opts = run_options(&opts.saturate);
    match opts.stop {
        StopPolicy::FixedPoint => run_opts.stop = Stop::FixedPoint,
        StopPolicy::Improving { patience } => {
            run_opts.stop = Stop::Improving;
            run_opts.patience = patience;
            run_opts.cost_fn = Some(&mut adapted);
        }
    }
    let stats = graph.run(&rules, root, &mut run_opts)?;

    let best = graph
        .extract_best(root, &mut adapted, None, None)
        .ok_or(Error::Extract)?;

    Ok(OptimizeResult {
        term: to_kernel(&best),
        saturated: Saturated {
            graph,
            roots: vec![root],
            stats,
        },
    })
}

/// The one-shot interpretation: lift `term`, saturate under `laws`,
/// extract the cheapest member under `cost`, return it as a kernel
/// `Term` alongside the saturated graph.
///
/// `cost` prices **kernel** terms — the wrapper lowers each candidate
/// e-graph term through `to_kernel` before pricing (total, but not
/// free). For serializable costs use [`optimize_spec`], which prices
/// the internal term directly (design doc §3).
///
/// # Errors
///
/// See `optimize_inner`.
pub fn optimize(
    term: &KernelTerm,
    laws: &[Law],
    cost: &mut dyn FnMut(&KernelTerm) -> f64,
    opts: &OptimizeOptions,
) -> Result<OptimizeResult, Error> {
    let mut adapted = |t: &ETerm| {
        let k = to_kernel(t);
        cost(&k)
    };
    optimize_inner(term, laws, &mut adapted, opts)
}

/// [`optimize`] over a serializable [`CostSpec`]: the spec prices
/// internal e-graph terms directly — `count`/`size`/`op_table` never
/// cross the `to_kernel` bridge per node (design doc §3's
/// engine-side interpretation).
///
/// # Errors
///
/// See `optimize_inner`.
pub fn optimize_spec(
    term: &KernelTerm,
    laws: &[Law],
    cost: &CostSpec,
    opts: &OptimizeOptions,
) -> Result<OptimizeResult, Error> {
    let mut adapted = |t: &ETerm| cost.price(t);
    optimize_inner(term, laws, &mut adapted, opts)
}

/// [`extract_best`] over a serializable [`CostSpec`] — prices
/// internal terms directly (see [`optimize_spec`]).
pub fn extract_best_spec(graph: &mut EGraph, root: ClassId, cost: &CostSpec) -> Option<KernelTerm> {
    let mut adapted = |t: &ETerm| cost.price(t);
    let best = graph.extract_best(root, &mut adapted, None, None)?;
    Some(to_kernel(&best))
}

/// The search-engine contract of design doc §5 (ADR-0177):
/// `saturate`/`optimize`/`name` over [`LawSet`] + [`CostSpec`]
/// **data** — not closures — which is what makes a second search
/// interpretation implementable without re-deriving the contract.
///
/// Minted inside this crate (not `hnl-engine`, not a contract crate):
/// promote only when a second interpretation lands. [`EGraphEngine`]
/// is the sole impl — the current free functions, wrapped.
pub trait SearchEngine {
    /// Saturate `terms` under `laws`; return roots + provenance.
    ///
    /// # Errors
    ///
    /// See [`saturate`] — including [`Error::UnserializableLaw`] and
    /// [`Error::GuardUnsupported`] for records whose honesty flags
    /// or deferred guards forbid firing.
    fn saturate(
        &self,
        terms: &[KernelTerm],
        laws: &LawSet,
        opts: &SaturateOptions,
    ) -> Result<Saturated, Error>;

    /// One-shot: lift, saturate, extract min-cost under `cost`,
    /// lower.
    ///
    /// # Errors
    ///
    /// See [`optimize_spec`].
    fn optimize(
        &self,
        term: &KernelTerm,
        laws: &LawSet,
        cost: &CostSpec,
        opts: &OptimizeOptions,
    ) -> Result<OptimizeResult, Error>;

    /// Name for provenance/diagnostics (`"eqsat"`).
    fn name(&self) -> &'static str;
}

/// The e-graph search interpretation — the crate's free functions
/// behind the [`SearchEngine`] contract.
#[derive(Clone, Copy, Debug, Default)]
pub struct EGraphEngine;

impl SearchEngine for EGraphEngine {
    fn saturate(
        &self,
        terms: &[KernelTerm],
        laws: &LawSet,
        opts: &SaturateOptions,
    ) -> Result<Saturated, Error> {
        saturate(terms, laws.laws(), opts)
    }

    fn optimize(
        &self,
        term: &KernelTerm,
        laws: &LawSet,
        cost: &CostSpec,
        opts: &OptimizeOptions,
    ) -> Result<OptimizeResult, Error> {
        optimize_spec(term, laws.laws(), cost, opts)
    }

    fn name(&self) -> &'static str {
        "eqsat"
    }
}

/// A size cost priced on **kernel** terms: `Const`/`Opaque` leaves
/// cost 0, every other node costs 1 plus its children — mirrors
/// `handl_egraph::Term::size` through the bridge (the `dim` child is
/// counted, so kernel `Op`s price one higher than their arg count —
/// the bridge encoding's honest cost; a dedicated kernel-side cost
/// model is kit-eqsat data, spike §6.3).
pub fn size_cost(term: &KernelTerm) -> f64 {
    fn go(t: &KernelTerm) -> usize {
        match t {
            KernelTerm::Const { .. } | KernelTerm::Opaque { .. } => 0,
            KernelTerm::Op { dim, args, .. } => 1 + go(dim) + args.iter().map(go).sum::<usize>(),
            KernelTerm::Arr { dim, .. } => 1 + go(dim),
            KernelTerm::Seq { axis, fst, snd } => 1 + go(axis) + go(fst) + go(snd),
            KernelTerm::First { inner } => 1 + go(inner),
            KernelTerm::Choice { l, r } => 1 + go(l) + go(r),
            KernelTerm::Loop { body } => 1 + go(body),
            KernelTerm::App { callee, args } => 1 + go(callee) + args.iter().map(go).sum::<usize>(),
            KernelTerm::Iter {
                fuel,
                predicate,
                step,
                initial,
            } => 1 + go(fuel) + go(predicate) + go(step) + go(initial),
            KernelTerm::Child { .. } => 1,
            KernelTerm::Handle { body, impl_, .. } => 1 + go(body) + go(impl_),
            KernelTerm::Rewrite { source, target, .. } => 1 + go(source) + go(target),
            KernelTerm::Src { cell } | KernelTerm::Tgt { cell } => 1 + go(cell),
            KernelTerm::RecordProj { record, .. } => 1 + go(record),
            KernelTerm::RecordWith { record, value, .. } => 1 + go(record) + go(value),
            KernelTerm::Tag { value, .. } => 1 + go(value),
            KernelTerm::Cobind { grade, body } => 1 + go(grade) + go(body),
        }
    }
    go(term) as f64
}
