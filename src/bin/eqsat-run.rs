//! `eqsat-run` — the JSON-in/JSON-out driver for the
//! [`hnl_engine_eqsat`] interpretation, the engine-eqsat half of the
//! Python↔Rust differential harness
//! (`graphopt:bench/drivers/audit/eqsat_differential.py`).
//!
//! Spec on stdin::
//!
//! ```json
//! {
//!   "name": "case-id",
//!   "terms": [<kernel Term JSON>, ...],
//!   "root": 0,
//!   "laws": [<LAW_FORMAT=2 record>, ...],
//!   "options": {"max_iterations": 100, "max_nodes": 100000,
//!               "rule_budgets": {"comm_add": 400}},
//!   "cost": {"kind": "op_table", "default": 1.0, "leaf": 0.0,
//!            "dim_leaf": 0.0, "ops": {"add": 1.0}},
//!   "check_terms": [<kernel Term JSON>, ...]
//! }
//! ```
//!
//! * `terms`/`check_terms` are `hnl_engine::Term` serde JSON
//!   (externally-tagged `{"Op": …}` / `{"Const": …}` / …). `terms`
//!   additionally accepts a top-level
//!   `{"Opaque": {"__tag__": "dag.program", "nodes": [...], "root": n}}`
//!   record — the ADR-0173 shared-subtree envelope — which lifts
//!   node-table-first into e-class ids (see `hnl_engine_eqsat::dag`'s
//!   decoding rules in `src/dag.rs`); `check_terms` stays
//!   kernel-`Term`-only.
//! * `laws` are `LAW_FORMAT = 2` records in the `pat` authoring
//!   spelling — the same corpus the `handl-kit-eqsat` packs ship.
//! * `cost` is a `CostSpec` record; `null`/absent defaults to
//!   `{"kind": "size"}`.
//!
//! Report on stdout::
//!
//! ```json
//! {"ok": true, "name": "…", "term": <Term JSON|null>,
//!  "cost": 0.0, "stats": {…}, "fired": ["law", …],
//!  "rule_fires": {"law": <merge_count>, …},
//!  "checks": [{"in_root": true}, …]}
//! ```
//!
//! On any failure the report is `{"ok": false, "error": "…"}` and the
//! exit status is 1 — the driver treats a nonzero exit as a hard
//! failure, never a verdict.

use std::io::Read;

use handl_egraph::from_kernel;
use hnl_engine::Term as KernelTerm;
use hnl_engine_eqsat::{self as eqsat, CostSpec, LawSet, SaturateOptions};
use serde_json::{Value, json};

/// Why the run failed before a report could be produced.
#[derive(Debug)]
enum RunError {
    /// stdin was not a JSON object.
    BadSpec(String),
    /// A `terms`/`check_terms` entry was not a kernel `Term`.
    BadTerm(String),
    /// The `laws` array did not decode as a `LawSet`.
    BadLaws(String),
    /// The `cost` record did not decode as a `CostSpec`.
    BadCost(String),
    /// The interpretation itself failed.
    Engine(eqsat::Error),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::BadSpec(e) => write!(f, "bad spec: {e}"),
            RunError::BadTerm(e) => write!(f, "bad term: {e}"),
            RunError::BadLaws(e) => write!(f, "bad laws: {e}"),
            RunError::BadCost(e) => write!(f, "bad cost: {e}"),
            RunError::Engine(e) => write!(f, "{e}"),
        }
    }
}

impl From<eqsat::Error> for RunError {
    fn from(e: eqsat::Error) -> Self {
        RunError::Engine(e)
    }
}

/// Decode a JSON array of kernel `Term`s — `check_terms` only; the
/// `terms` array goes through [`eqsat::saturate_json`] so `dag.program`
/// records are accepted there.
fn terms_of(value: Option<&Value>, field: &str) -> Result<Vec<KernelTerm>, RunError> {
    let Some(v) = value else {
        return Ok(Vec::new());
    };
    let arr = v
        .as_array()
        .ok_or_else(|| RunError::BadSpec(format!("\"{field}\" must be an array")))?;
    arr.iter()
        .map(|t| {
            serde_json::from_value::<KernelTerm>(t.clone())
                .map_err(|e| RunError::BadTerm(format!("{field}: {e}")))
        })
        .collect()
}

/// Decode the optional `cost` record (default `{"kind": "size"}`).
fn cost_of(spec: &Value) -> Result<CostSpec, RunError> {
    match spec.get("cost") {
        None | Some(Value::Null) => Ok(CostSpec::Size),
        Some(v) => CostSpec::from_data(v).map_err(|e| RunError::BadCost(e.to_string())),
    }
}

/// Decode `options` into [`SaturateOptions`].
fn options_of(spec: &Value) -> SaturateOptions {
    let mut opts = SaturateOptions::spec();
    if let Some(o) = spec.get("options").and_then(Value::as_object) {
        if let Some(n) = o.get("max_iterations").and_then(Value::as_u64) {
            opts.max_iterations = n as usize;
        }
        if let Some(n) = o.get("max_nodes").and_then(Value::as_u64) {
            opts.max_nodes = n as usize;
        }
        if let Some(b) = o.get("rule_budgets").and_then(Value::as_object) {
            opts.rule_budgets = b
                .iter()
                .filter_map(|(k, v)| v.as_u64().map(|n| (k.clone(), n as usize)))
                .collect();
        }
    }
    opts
}

/// The `stats` block — the fields the differential report compares.
fn stats_json(stats: &handl_egraph::SaturationStats) -> Value {
    json!({
        "iterations": stats.iterations,
        "n_enodes": stats.n_enodes,
        "n_classes": stats.n_classes,
        "n_proof_edges": stats.n_proof_edges,
        "truncation_level": stats.truncation_level,
        "stop": format!("{:?}", stats.stop),
        "improved": stats.improved,
        "rule_budgets": stats.rule_budgets,
        "budget_suspended": stats.budget_suspended,
    })
}

/// Sorted unique rule names named by the merge log — provenance the
/// driver diffs against Python's `merge_log`.
fn fired_json(graph: &handl_egraph::EGraph) -> Value {
    let mut names: Vec<String> = graph
        .merge_log()
        .iter()
        .filter_map(|e| e.rule.as_deref().map(str::to_string))
        .collect();
    names.sort();
    names.dedup();
    json!(names)
}

/// Per-rule merge counts — the `{rule_name: merge_count}` record the
/// Python `Engine` port's `rule_fires` reports (ADR-0178 WP3: the
/// `engine="eqsat"` adapter surfaces this verbatim).
fn rule_fires_json(graph: &handl_egraph::EGraph) -> Value {
    let mut counts: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    for e in graph.merge_log() {
        if let Some(r) = e.rule.as_deref() {
            *counts.entry(r.to_string()).or_default() += 1;
        }
    }
    json!(counts)
}

/// Decode the `terms` array as raw wire records — each entry is a
/// kernel-`Term` tree JSON or a top-level `dag.program` envelope;
/// `saturate_json` dispatches per record. Also reports whether any
/// entry was a `dag.program` record (the shape-cost refusal input).
fn term_records_of(spec: &Value) -> Result<(Vec<Value>, bool), RunError> {
    let Some(v) = spec.get("terms") else {
        return Err(RunError::BadSpec("\"terms\" is required".into()));
    };
    let arr = v
        .as_array()
        .ok_or_else(|| RunError::BadSpec("\"terms\" must be an array".into()))?;
    let mut saw_dag = false;
    for t in arr {
        if t.get("Opaque")
            .and_then(|p| p.get("__tag__"))
            .and_then(Value::as_str)
            == Some("dag.program")
        {
            saw_dag = true;
        }
    }
    Ok((arr.clone(), saw_dag))
}

/// Run the spec; returns the report object.
fn run(spec: &Value) -> Result<Value, RunError> {
    let (terms, saw_dag) = term_records_of(spec)?;
    if terms.is_empty() {
        return Err(RunError::BadSpec("\"terms\" must be non-empty".into()));
    }
    let check_terms = terms_of(spec.get("check_terms"), "check_terms")?;
    let laws = match spec.get("laws") {
        None | Some(Value::Null) => LawSet::new(),
        Some(v) => LawSet::from_data(v).map_err(|e| RunError::BadLaws(e.to_string()))?,
    };
    let cost = cost_of(spec)?;
    // `flops` formulas price through shape inference's *structural*
    // term memo — exponential on the shared terms a dag.program input
    // produces. Refuse by name up front instead of hanging.
    if saw_dag && cost.needs_shapes() {
        return Err(RunError::BadCost(
            "flops cost formulas are unsupported over dag.program inputs \
             (shape inference is tree-priced) — use count/size/op_table \
             weights or engine=\"python-egraph\""
                .into(),
        ));
    }
    let opts = options_of(spec);
    let root_idx = spec.get("root").and_then(Value::as_u64).unwrap_or(0) as usize;
    if root_idx >= terms.len() {
        return Err(RunError::BadSpec(format!(
            "\"root\" {root_idx} out of range ({} terms)",
            terms.len()
        )));
    }

    // saturate_json takes the law slice — compile refusals
    // (unserializable records, unsupported guards) surface as named
    // engine errors; dag.program records decode node-table-first.
    let mut sat = eqsat::saturate_json(&terms, laws.laws(), &opts)?;
    let root = sat.roots[root_idx];

    let t_extract = std::time::Instant::now();
    // extract_best_json prices DAG-safely (identity-memoised
    // CostSpec::price_dag) and serialises tree-or-dag.program by the
    // expansion bounds — the plain `to_kernel` path tree-expands
    // shared terms exponentially.
    let (term_json, cost_json) = match eqsat::extract_best_json(&mut sat.graph, root, &cost) {
        Ok(Some((t, c))) => (Some(t), json!(c)),
        Ok(None) => (None, Value::Null),
        Err(e) => return Err(RunError::Engine(e)),
    };
    if std::env::var_os("EQSAT_TIMING").is_some() {
        eprintln!(
            "[eqsat] extract_best_json: {:.1}ms",
            t_extract.elapsed().as_secs_f64() * 1e3
        );
    }

    let checks: Vec<Value> = check_terms
        .iter()
        .map(|kt| match sat.graph.class_of_term(&from_kernel(kt)) {
            Some(c) => json!({
                "in_root": sat.graph.find_ref(c) == sat.graph.find_ref(root),
            }),
            None => json!({"in_root": false, "absent": true}),
        })
        .collect();

    Ok(json!({
        "ok": true,
        "name": spec.get("name").cloned().unwrap_or(Value::Null),
        "term": term_json,
        "cost": cost_json,
        "stats": stats_json(&sat.stats),
        "fired": fired_json(&sat.graph),
        "rule_fires": rule_fires_json(&sat.graph),
        "checks": checks,
    }))
}

fn main() {
    let mut input = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut input) {
        println!("{}", json!({"ok": false, "error": format!("stdin: {e}")}));
        std::process::exit(1);
    }
    let report = match serde_json::from_str::<Value>(&input) {
        Err(e) => Err(RunError::BadSpec(format!("invalid JSON: {e}"))),
        Ok(spec) => run(&spec),
    };
    match report {
        Ok(v) => println!("{v}"),
        Err(e) => {
            println!("{}", json!({"ok": false, "error": e.to_string()}));
            std::process::exit(1);
        }
    }
}
