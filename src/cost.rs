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
//! `{"flops": "…"}` formula entries resolve against
//! [`crate::shape`]-inferred operand/output shapes — a small
//! arithmetic expression language over bound dim names (see
//! [`OpEntry::Formula`] for the binding table). **A formula prices
//! `+∞` whenever a name it references is unbound** — unknown or
//! `Invalid` shapes, unproven dims, malformed expressions — so an
//! op is never *under*-priced into a wrong extraction. Roofline/
//! fusion/executor models are **not** here — they need shape
//! inference (doc §2 stage 2) or are backend algorithms.
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
    /// A parametric flops formula (e.g. `"2*m*n*k"`) evaluated
    /// against [`crate::shape`]-inferred shapes — see
    /// `formula_bindings` for the name table. Prices `+∞` whenever
    /// a referenced name is unbound (unknown/`Invalid`/non-concrete
    /// shapes) or the expression is malformed — never under-priced.
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
    /// Price one node; `in_dim` marks the dim-slot position (child 0
    /// of an op-encoded node) so [`OpTable::dim_leaf`] applies to
    /// leaves there. `smemo` is the shared [`crate::shape::Memo`]
    /// formula evaluation draws operand shapes from (created lazily
    /// by [`OpTable::price`]).
    fn price_node(&self, t: &Term, in_dim: bool, smemo: &mut Option<crate::shape::Memo>) -> f64 {
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
                let mut total = self.head_cost(o, smemo);
                let sourced = is_op_encoded(o);
                for (i, child) in o.args.iter().enumerate() {
                    total += self.price_node(child, sourced && i == 0, smemo);
                }
                total
            }
        }
    }

    /// [`OpTable::price_node`] memoised by node *identity* — the same
    /// value as the plain recursion (the memo caches, the per-edge
    /// sums still count every occurrence) but linear in distinct
    /// nodes on a shared DAG where the plain walk is exponential.
    /// `in_dim` is part of the key: a leaf prices differently in the
    /// dim slot.
    fn price_node_dag(
        &self,
        t: &Term,
        in_dim: bool,
        smemo: &mut Option<crate::shape::Memo>,
        memo: &mut std::collections::HashMap<(crate::dag::NodeKey, bool), f64>,
    ) -> f64 {
        match t {
            Term::Leaf(_) | Term::Var(_) => self.price_node(t, in_dim, smemo),
            Term::Op(o) => {
                let key = (crate::dag::node_key(t), in_dim);
                if let Some(&c) = memo.get(&key) {
                    return c;
                }
                let mut total = self.head_cost(o, smemo);
                let sourced = is_op_encoded(o);
                for (i, child) in o.args.iter().enumerate() {
                    total += self.price_node_dag(child, sourced && i == 0, smemo, memo);
                }
                memo.insert(key, total);
                total
            }
        }
    }

    /// Price the whole term — one postorder pass sharing a shape
    /// memo across every `flops` formula it meets.
    fn price(&self, t: &Term) -> f64 {
        let mut smemo = None;
        self.price_node(t, false, &mut smemo)
    }

    /// [`OpTable::price`] over a possibly-shared term — identical
    /// values, linear in distinct nodes. `flops` formulas still draw
    /// on the structural [`crate::shape::Memo`]; see
    /// [`CostSpec::needs_shapes`] for the honest boundary.
    fn price_dag(&self, t: &Term) -> f64 {
        let mut smemo = None;
        let mut memo = std::collections::HashMap::default();
        self.price_node_dag(t, false, &mut smemo, &mut memo)
    }

    /// The op head's local cost: flat weights, view zeroing, or a
    /// `flops` formula over inferred shapes.
    fn head_cost(&self, o: &Op, smemo: &mut Option<crate::shape::Memo>) -> f64 {
        match self.ops.get(o.op.as_ref()) {
            None => self.default,
            Some(OpEntry::Weight(w)) => *w,
            Some(OpEntry::View) => 0.0,
            Some(OpEntry::Formula(f)) => {
                let memo = smemo.get_or_insert_with(crate::shape::Memo::new);
                formula_cost(f, o, memo)
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
            CostSpec::OpTable(table) => table.price(t),
        }
    }

    /// [`CostSpec::price`] over a possibly-shared term — identical
    /// values, but each walk memoises by node *identity* (pointer-
    /// equal `Rc` allocations), so a DAG prices in linear time where
    /// the plain recursion tree-expands exponentially. This is the
    /// cost function the `dag.program` wire path (`saturate_json` →
    /// `extract_best_json`) requires; `flops` formulas still evaluate
    /// through the structural shape memo — see
    /// [`CostSpec::needs_shapes`].
    pub fn price_dag(&self, t: &Term) -> f64 {
        match self {
            CostSpec::Count => {
                let mut memo = std::collections::HashMap::default();
                count_dag(t, &mut memo)
            }
            CostSpec::Size => {
                let mut memo = std::collections::HashMap::default();
                size_dag(t, &mut memo)
            }
            CostSpec::OpTable(table) => table.price_dag(t),
        }
    }

    /// Whether this spec's `op_table` carries `flops` formulas —
    /// the entries whose pricing draws on [`crate::shape`]'s
    /// *structural* term memo (exponential on a shared DAG). Callers
    /// saturating `dag.program` inputs should refuse such specs by
    /// name rather than hang inside shape inference.
    pub fn needs_shapes(&self) -> bool {
        match self {
            CostSpec::OpTable(table) => {
                table.ops.values().any(|e| matches!(e, OpEntry::Formula(_)))
            }
            _ => false,
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

/// [`count`] memoised by node identity — same occurrence-priced
/// value, linear in distinct nodes on a shared DAG.
fn count_dag(t: &Term, memo: &mut std::collections::HashMap<crate::dag::NodeKey, f64>) -> f64 {
    match t {
        Term::Var(_) | Term::Leaf(_) => 1.0,
        Term::Op(o) => {
            let key = crate::dag::node_key(t);
            if let Some(&c) = memo.get(&key) {
                return c;
            }
            let total = 1.0 + o.args.iter().map(|c| count_dag(c, memo)).sum::<f64>();
            memo.insert(key, total);
            total
        }
    }
}

/// [`Term::size`] memoised by node identity — same value.
fn size_dag(t: &Term, memo: &mut std::collections::HashMap<crate::dag::NodeKey, f64>) -> f64 {
    match t {
        Term::Var(_) | Term::Leaf(_) => 0.0,
        Term::Op(o) => {
            let key = crate::dag::node_key(t);
            if let Some(&c) = memo.get(&key) {
                return c;
            }
            let total = 1.0 + o.args.iter().map(|c| size_dag(c, memo)).sum::<f64>();
            memo.insert(key, total);
            total
        }
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

// ---------------------------------------------------------------------------
//  `flops` formulas — arithmetic over shape-inferred dim names
// ---------------------------------------------------------------------------

/// Evaluate a `{"flops": "…"}` formula for one op node.
///
/// The name table (`formula_bindings`) binds `out` (the output
/// numel), `o<i>` (output dims), the per-family contraction names
/// (`m`/`n`/`k` for matmul, `in`/`outf`/`m` for linear,
/// `n`/`oc`/`ic`/`kh`/`kw`/`groups` for conv), all as concrete
/// numbers — a name binds **only** when the dim is provably concrete.
/// Any referenced-but-unbound name, a non-finite result, or a
/// malformed expression prices `+∞`: the op is never under-priced.
fn formula_cost(formula: &str, o: &Op, memo: &mut crate::shape::Memo) -> f64 {
    let out_shape = crate::shape::shape_of_with(&Term::Op(o.clone()), memo);
    let bindings = formula_bindings(o, memo, &out_shape);
    eval_formula(formula, &bindings).unwrap_or(f64::INFINITY)
}

/// The dim-name bindings a formula may reference, per op family.
/// Every value is a *concrete* extent — `None` dims and non-tuple
/// shapes simply leave the name unbound.
fn formula_bindings(
    o: &Op,
    memo: &mut crate::shape::Memo,
    out_shape: &crate::shape::Shape,
) -> BTreeMap<String, f64> {
    use crate::shape::{Dim, Shape};
    let mut b = BTreeMap::new();
    let dim_f = |d: Dim| d.map(|n| n as f64);
    if let Some(n) = crate::shape::numel_concrete(out_shape) {
        b.insert("out".to_string(), n as f64);
    }
    if let Shape::Tuple(dims) = out_shape {
        for (i, d) in dims.iter().enumerate() {
            if let Some(v) = dim_f(*d) {
                b.insert(format!("o{i}"), v);
            }
        }
    }
    // Real operand shapes (the dim slot is never an operand).
    let args: Vec<&Term> = if is_op_encoded(o) {
        o.args.iter().skip(1).collect()
    } else {
        o.args.iter().collect()
    };
    let shapes: Vec<Shape> = args
        .iter()
        .map(|a| crate::shape::shape_of_with(a, memo))
        .collect();
    let tup = |i: usize| shapes.get(i).and_then(Shape::as_tuple);
    let name = o.op.strip_prefix("tensor.").unwrap_or(&o.op);
    match name {
        // matmul: m = a[-2], n = b[-1], k = contraction extent —
        // b[-2] for rank ≥ 2, b[0] for a rank-1 rhs (the
        // `_flops_of` matmul arm's k_dim read).
        "matmul" => {
            if let Some(a) = tup(0)
                && a.len() >= 2
                && let Some(v) = dim_f(a[a.len() - 2])
            {
                b.insert("m".to_string(), v);
            }
            if let Some(bm) = tup(1) {
                if let Some(v) = bm.last().and_then(|d| dim_f(*d)) {
                    b.insert("n".to_string(), v);
                }
                let k = if bm.len() >= 2 {
                    dim_f(bm[bm.len() - 2])
                } else if bm.len() == 1 {
                    dim_f(bm[0])
                } else {
                    None
                };
                if let Some(v) = k {
                    b.insert("k".to_string(), v);
                }
            }
        }
        // linear: m = leading-batch numel of x, outf = w[0],
        // in = x[-1] — `2*out*in` reproduces `_flops_of`'s
        // `2 * n_out * shapes[0][-1]`.
        "linear" => {
            if let Some(x) = tup(0)
                && !x.is_empty()
            {
                if let Some(v) = dim_f(x[x.len() - 1]) {
                    b.insert("in".to_string(), v);
                }
                let batch: i64 = x[..x.len() - 1]
                    .iter()
                    .try_fold(1i64, |acc, d| acc.checked_mul((*d)?))
                    .unwrap_or(0);
                if batch > 0 {
                    b.insert("m".to_string(), batch as f64);
                }
            }
            if let Some(w) = tup(1)
                && !w.is_empty()
                && let Some(v) = dim_f(w[0])
            {
                b.insert("outf".to_string(), v);
            }
        }
        // conv1d/conv2d: n = N, oc = O, ic = C, kh/kw the kernel
        // extents, groups the attr — `2*out*ic*kh*kw/groups`
        // reproduces `_flops_of`'s conv arm.
        "conv1d" | "conv2d" => {
            let spatial = if name == "conv1d" { 1 } else { 2 };
            if let Some(x) = tup(0) {
                if let Some(v) = x.first().and_then(|d| dim_f(*d)) {
                    b.insert("n".to_string(), v);
                }
                if x.len() > 1
                    && let Some(v) = dim_f(x[1])
                {
                    b.insert("ic".to_string(), v);
                }
            }
            if let Some(w) = tup(1) {
                if let Some(v) = w.first().and_then(|d| dim_f(*d)) {
                    b.insert("oc".to_string(), v);
                }
                if w.len() > spatial + 1
                    && let Some(v) = dim_f(w[1])
                {
                    b.insert("ic_w".to_string(), v);
                }
                for (i, key) in ["kh", "kw"].iter().enumerate().take(spatial) {
                    if let Some(v) = w.get(2 + i).and_then(|d| dim_f(*d)) {
                        b.insert((*key).to_string(), v);
                    }
                }
            }
            if let Some(g) = o.attrs.iter().find_map(|(k, v)| {
                if k.as_ref() == "__attrs__"
                    && let handl_egraph::TermAttr::Lit(handl_egraph::AttrValue::Object(map)) = v
                {
                    return map.get("groups").and_then(|g| match g {
                        handl_egraph::AttrValue::Int(n) => Some(*n as f64),
                        handl_egraph::AttrValue::UInt(n) => {
                            Some(f64::from(u32::try_from(*n).unwrap_or(u32::MAX)))
                        }
                        _ => None,
                    });
                }
                None
            }) && g > 0.0
            {
                b.insert("groups".to_string(), g);
            }
        }
        _ => {}
    }
    b
}

/// A flops formula is a small arithmetic expression: numbers,
/// identifiers, `+ - * /`, parentheses, unary minus. Anything else —
/// or a name absent from `formula_bindings` — is `None` → `+∞`.
fn eval_formula(src: &str, bindings: &BTreeMap<String, f64>) -> Option<f64> {
    let toks = lex_formula(src)?;
    let mut p = FormulaParser {
        toks: &toks,
        pos: 0,
        bindings,
    };
    let v = p.expr()?;
    if p.pos != toks.len() {
        return None;
    }
    v.is_finite().then_some(v)
}

/// One formula token.
#[derive(Clone, Debug, PartialEq)]
enum FTok {
    /// A numeric literal.
    Num(f64),
    /// A bound dim name.
    Name(String),
    /// `+`.
    Add,
    /// `-`.
    Sub,
    /// `*`.
    Mul,
    /// `/`.
    Div,
    /// `(`.
    LParen,
    /// `)`.
    RParen,
}

/// Tokenize a formula string; `None` on any illegal byte.
fn lex_formula(src: &str) -> Option<Vec<FTok>> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        match b[i] {
            b' ' | b'\t' | b'\n' => i += 1,
            b'+' => {
                out.push(FTok::Add);
                i += 1;
            }
            b'-' => {
                out.push(FTok::Sub);
                i += 1;
            }
            b'*' => {
                out.push(FTok::Mul);
                i += 1;
            }
            b'/' => {
                out.push(FTok::Div);
                i += 1;
            }
            b'(' => {
                out.push(FTok::LParen);
                i += 1;
            }
            b')' => {
                out.push(FTok::RParen);
                i += 1;
            }
            c if c.is_ascii_digit() || c == b'.' => {
                let start = i;
                while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
                    i += 1;
                }
                let text = std::str::from_utf8(&b[start..i]).ok()?;
                out.push(FTok::Num(text.parse::<f64>().ok()?));
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                let start = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                out.push(FTok::Name(
                    std::str::from_utf8(&b[start..i]).ok()?.to_string(),
                ));
            }
            _ => return None,
        }
    }
    Some(out)
}

/// Recursive-descent evaluator: `expr := term (('+'|'-') term)*`,
/// `term := factor (('*'|'/') factor)*`,
/// `factor := '-' factor | '(' expr ')' | num | name`.
struct FormulaParser<'a> {
    toks: &'a [FTok],
    pos: usize,
    bindings: &'a BTreeMap<String, f64>,
}

impl FormulaParser<'_> {
    fn peek(&self) -> Option<&FTok> {
        self.toks.get(self.pos)
    }

    fn expr(&mut self) -> Option<f64> {
        let mut v = self.term()?;
        while let Some(op) = self.peek() {
            match op {
                FTok::Add => {
                    self.pos += 1;
                    v += self.term()?;
                }
                FTok::Sub => {
                    self.pos += 1;
                    v -= self.term()?;
                }
                _ => break,
            }
        }
        Some(v)
    }

    fn term(&mut self) -> Option<f64> {
        let mut v = self.factor()?;
        while let Some(op) = self.peek() {
            match op {
                FTok::Mul => {
                    self.pos += 1;
                    v *= self.factor()?;
                }
                FTok::Div => {
                    self.pos += 1;
                    let d = self.factor()?;
                    if d == 0.0 {
                        return None;
                    }
                    v /= d;
                }
                _ => break,
            }
        }
        Some(v)
    }

    fn factor(&mut self) -> Option<f64> {
        match self.peek()? {
            FTok::Sub => {
                self.pos += 1;
                Some(-self.factor()?)
            }
            FTok::LParen => {
                self.pos += 1;
                let v = self.expr()?;
                if self.peek() != Some(&FTok::RParen) {
                    return None;
                }
                self.pos += 1;
                Some(v)
            }
            FTok::Num(n) => {
                let v = *n;
                self.pos += 1;
                Some(v)
            }
            FTok::Name(name) => {
                let v = *self.bindings.get(name)?;
                self.pos += 1;
                Some(v)
            }
            _ => None,
        }
    }
}
