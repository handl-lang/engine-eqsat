//! [`CostSpec`] — the serializable cost model of design doc §3
//! (ADR-0177): named cost *kinds* plus a per-op table, interpreted
//! engine-side over the e-graph's internal [`Term`] — no per-node
//! `to_kernel` bridge crossing.
//!
//! Kinds, in landing order:
//!
//! * [`CostSpec::Count`] — every node (leaf or op) costs 1: the plain
//!   node count.
//! * [`CostSpec::Size`] — the shipped `size_cost` semantics:
//!   `Const`/`Opaque` leaves cost 0, every other node costs 1 plus its
//!   children, the `dim` slot counted — the honest bridge-adjusted
//!   cost (identical to [`handl_egraph::Term::size`]).
//! * [`CostSpec::OpTable`] — per-op-name weights with `default`/`leaf`
//!   fallbacks, `dim_leaf` for leaves in the dim slot, and view-op
//!   zeroing (the `_VIEW_OPS` port). Covers `count_cost`,
//!   `param_bytes_cost` (leaf = param bytes), and launch-penalty
//!   variants once a per-term constant exists.
//!
//! `flops`/roofline/fusion/executor models are **not** here — they
//! need shape inference (doc §2 stage 2) or are backend algorithms.
//! A record whose `kind` is not `count`/`size`/`op_table` is refused
//! at load with [`CostError::UnknownKind`], never approximated. An
//! `op_table` entry carrying a `"flops"` formula string is kept as
//! [`OpEntry::Formula`] and prices as `+∞` until shape inference
//! lands — deferred, never under-priced into a wrong extraction.
//!
//! The `FnMut(&KernelTerm) -> f64` procedural overload stays on
//! [`crate::engine::optimize`]/[`crate::extract_best`] for genuinely
//! procedural costs — the same named, flagged posture.

use std::collections::BTreeMap;

use handl_egraph::{Op, Term};
use serde_json::Value;

/// Why a [`CostSpec`] record failed to decode — or why a kind was
/// refused.
#[derive(Clone, Debug, PartialEq)]
pub enum CostError {
    /// The record is not an object, or a required field is missing
    /// or mistyped.
    Malformed(String),
    /// `kind` is not `count`/`size`/`op_table` — refused at load,
    /// never approximated (a `flops`/roofline/fusion/executor kind
    /// lands only when its interpreter does).
    UnknownKind(String),
    /// An `ops` entry is neither a number nor a recognised object
    /// (`{"view": true}`, `{"cost": n}`, `{"flops": "…"}`).
    BadEntry {
        /// The op name the entry was filed under.
        op: String,
        /// What was wrong with it.
        detail: String,
    },
}

impl std::fmt::Display for CostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CostError::Malformed(msg) => write!(f, "malformed cost spec: {msg}"),
            CostError::UnknownKind(kind) => {
                write!(f, "unknown cost kind {kind:?} — refused, not approximated")
            }
            CostError::BadEntry { op, detail } => {
                write!(f, "bad ops entry for {op:?}: {detail}")
            }
        }
    }
}

impl std::error::Error for CostError {}

/// One `ops` table entry: the per-op weight spec.
#[derive(Clone, Debug, PartialEq)]
pub enum OpEntry {
    /// A flat weight added to the op's children costs.
    Weight(f64),
    /// A view op — always costs 0 (the `_VIEW_OPS` zeroing port:
    /// reshape/transpose/slice-class ops are free).
    View,
    /// A deferred parametric formula (e.g. `"2*m*n*k"`) — needs the
    /// stage-2 shape inference to resolve; prices as `+∞` until then
    /// so the op is never *under*-priced into a wrong extraction.
    Formula(String),
}

/// The `op_table` kind's data: per-op-name weights plus fallbacks.
#[derive(Clone, Debug, PartialEq)]
pub struct OpTable {
    /// An optional label carried for provenance (`"flops-v1"`).
    pub name: Option<String>,
    /// Weight for ops with no `ops` entry.
    pub default: f64,
    /// Weight for [`Term::Leaf`] nodes in argument position.
    pub leaf: f64,
    /// Weight for [`Term::Leaf`] nodes in the **dim slot** (child 0
    /// of a `Term::Op`-encoded node — one carrying a `"source"`
    /// attr). `None` prices dim leaves at [`OpTable::leaf`].
    pub dim_leaf: Option<f64>,
    /// Per-op-name entries.
    pub ops: BTreeMap<String, OpEntry>,
}

/// A serializable cost spec — the `CostSpec` record of design doc §3.
#[derive(Clone, Debug, PartialEq)]
pub enum CostSpec {
    /// Node count: every node costs 1.
    Count,
    /// The shipped `size_cost` semantics over internal terms:
    /// leaves 0, `Op` = 1 + children (dim slot counted).
    Size,
    /// Per-op-name weights with `default`/`leaf`/`dim_leaf`
    /// fallbacks and view-op zeroing.
    OpTable(OpTable),
}

/// The `"source"` attr that marks a node as a `Term::Op`-encoded
/// kernel op (child 0 is then the dim slot) — the same heuristic
/// `handl_egraph::to_kernel` uses.
fn is_op_encoded(op: &Op) -> bool {
    op.attrs.iter().any(|(k, _)| k.as_ref() == "source")
}

impl OpTable {
    /// The weight an op head contributes (its *local* cost — the
    /// children are priced separately by [`OpTable::price`]).
    fn head_weight(&self, op: &str) -> f64 {
        match self.ops.get(op) {
            None => self.default,
            Some(OpEntry::Weight(w)) => *w,
            Some(OpEntry::View) => 0.0,
            Some(OpEntry::Formula(_)) => f64::INFINITY,
        }
    }

    /// Price one node; `in_dim` marks the dim-slot position (child 0
    /// of an op-encoded node) so [`OpTable::dim_leaf`] applies to
    /// leaves there.
    fn price_node(&self, t: &Term, in_dim: bool) -> f64 {
        match t {
            Term::Leaf(_) => {
                if in_dim {
                    self.dim_leaf.unwrap_or(self.leaf)
                } else {
                    self.leaf
                }
            }
            // A metavariable is pattern-only and never reaches
            // extraction; price it at `default` rather than guess.
            Term::Var(_) => self.default,
            Term::Op(o) => {
                let mut total = self.head_weight(&o.op);
                let sourced = is_op_encoded(o);
                for (i, child) in o.args.iter().enumerate() {
                    total += self.price_node(child, sourced && i == 0);
                }
                total
            }
        }
    }
}

impl CostSpec {
    /// Price a (sub)term — the additive total-cost function
    /// `extract_best`'s local-cost recovery expects.
    pub fn price(&self, t: &Term) -> f64 {
        match self {
            CostSpec::Count => count(t),
            CostSpec::Size => t.size() as f64,
            CostSpec::OpTable(table) => table.price_node(t, false),
        }
    }

    /// Encode as the `CostSpec` record (design doc §3).
    pub fn to_data(&self) -> Value {
        match self {
            CostSpec::Count => serde_json::json!({"kind": "count"}),
            CostSpec::Size => serde_json::json!({"kind": "size"}),
            CostSpec::OpTable(t) => {
                let ops: serde_json::Map<String, Value> = t
                    .ops
                    .iter()
                    .map(|(op, e)| {
                        let v = match e {
                            OpEntry::Weight(w) => Value::from(*w),
                            OpEntry::View => serde_json::json!({"view": true, "cost": 0.0}),
                            OpEntry::Formula(f) => serde_json::json!({"flops": f}),
                        };
                        (op.clone(), v)
                    })
                    .collect();
                let mut rec = serde_json::Map::new();
                rec.insert("kind".to_string(), Value::from("op_table"));
                if let Some(n) = &t.name {
                    rec.insert("name".to_string(), Value::from(n.as_str()));
                }
                rec.insert("default".to_string(), num(t.default));
                rec.insert("leaf".to_string(), num(t.leaf));
                rec.insert(
                    "dim_leaf".to_string(),
                    t.dim_leaf.map(num).unwrap_or(Value::Null),
                );
                rec.insert("ops".to_string(), Value::Object(ops));
                Value::Object(rec)
            }
        }
    }

    /// Decode a [`CostSpec`] from its record form.
    ///
    /// # Errors
    ///
    /// [`CostError::UnknownKind`] for any `kind` outside
    /// `count`/`size`/`op_table` — refused at load, never approximated.
    /// [`CostError::Malformed`]/[`CostError::BadEntry`] on field-level
    /// problems.
    pub fn from_data(value: &Value) -> Result<CostSpec, CostError> {
        let obj = value
            .as_object()
            .ok_or_else(|| CostError::Malformed(format!("{value}")))?;
        let kind = obj
            .get("kind")
            .and_then(Value::as_str)
            .ok_or_else(|| CostError::Malformed("\"kind\" must be a string".into()))?;
        match kind {
            "count" => Ok(CostSpec::Count),
            "size" => Ok(CostSpec::Size),
            "op_table" => op_table_from_data(obj).map(CostSpec::OpTable),
            other => Err(CostError::UnknownKind(other.to_string())),
        }
    }

    /// Decode from JSON text.
    ///
    /// # Errors
    ///
    /// [`CostError::Malformed`] on unparsable input, else whatever
    /// [`CostSpec::from_data`] reports.
    pub fn from_json(text: &str) -> Result<CostSpec, CostError> {
        let value: Value = serde_json::from_str(text)
            .map_err(|e| CostError::Malformed(format!("invalid JSON: {e}")))?;
        CostSpec::from_data(&value)
    }
}

/// `f64` → JSON number (`null` for non-finite — costs are finite by
/// contract; a non-finite weight would poison extraction anyway).
fn num(v: f64) -> Value {
    serde_json::Number::from_f64(v)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

/// The node-count cost: every node (leaf, var, op) costs 1 plus its
/// children — the plain "how many nodes" metric.
fn count(t: &Term) -> f64 {
    match t {
        Term::Var(_) | Term::Leaf(_) => 1.0,
        Term::Op(o) => 1.0 + o.args.iter().map(count).sum::<f64>(),
    }
}

/// Decode the `op_table` record body (the `kind` key is already
/// consumed).
fn op_table_from_data(obj: &serde_json::Map<String, Value>) -> Result<OpTable, CostError> {
    let malformed = |msg: &str| CostError::Malformed(msg.to_string());
    let num_field = |field: &str, default: f64| -> Result<f64, CostError> {
        match obj.get(field) {
            None | Some(Value::Null) => Ok(default),
            Some(v) => v
                .as_f64()
                .ok_or_else(|| malformed(&format!("\"{field}\" must be a number"))),
        }
    };
    let name = obj.get("name").and_then(Value::as_str).map(str::to_string);
    let default = num_field("default", 1.0)?;
    let leaf = num_field("leaf", 0.0)?;
    let dim_leaf = match obj.get("dim_leaf") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_f64()
                .ok_or_else(|| malformed("\"dim_leaf\" must be a number"))?,
        ),
    };
    let mut ops = BTreeMap::new();
    if let Some(raw) = obj.get("ops") {
        let entries = raw
            .as_object()
            .ok_or_else(|| malformed("\"ops\" must be an object"))?;
        for (op, v) in entries {
            ops.insert(op.clone(), op_entry_from_data(op, v)?);
        }
    }
    Ok(OpTable {
        name,
        default,
        leaf,
        dim_leaf,
        ops,
    })
}

/// Decode one `ops` entry: a bare number is a [`OpEntry::Weight`]; an
/// object is `{"view": true}` → [`OpEntry::View`], `{"cost": n}` →
/// [`OpEntry::Weight`], `{"flops": "…"}` → [`OpEntry::Formula`]
/// (deferred — see module docs).
fn op_entry_from_data(op: &str, v: &Value) -> Result<OpEntry, CostError> {
    if let Some(w) = v.as_f64() {
        return Ok(OpEntry::Weight(w));
    }
    let obj = v.as_object().ok_or_else(|| CostError::BadEntry {
        op: op.to_string(),
        detail: format!("expected a number or object, got {v}"),
    })?;
    if let Some(f) = obj.get("flops") {
        return match f.as_str() {
            Some(s) => Ok(OpEntry::Formula(s.to_string())),
            None => Err(CostError::BadEntry {
                op: op.to_string(),
                detail: "\"flops\" must be a formula string".to_string(),
            }),
        };
    }
    if obj.get("view").and_then(Value::as_bool) == Some(true) {
        return Ok(OpEntry::View);
    }
    if let Some(c) = obj.get("cost") {
        return c
            .as_f64()
            .map(OpEntry::Weight)
            .ok_or_else(|| CostError::BadEntry {
                op: op.to_string(),
                detail: "\"cost\" must be a number".to_string(),
            });
    }
    Err(CostError::BadEntry {
        op: op.to_string(),
        detail: format!("unrecognised entry {v}"),
    })
}
