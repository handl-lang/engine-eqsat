//! Shape inference over e-graph [`Term`]s — the dependency stage-2
//! shape-reading guards (design doc §2) and the `flops` [`crate::cost`]
//! formula kind (§3) share.
//!
//! This is the Rust port of `mlopt_core.typing._shape_of` /
//! `_broadcast` / `_matmul_shape` / `_numel` and the shape-spec half of
//! `mlopt_core.laws.cond` (`_SPEC_OPS`), reproduced over the
//! `term_bridge` encoding rather than the typed `Op` IR. The honesty
//! contract is ported verbatim:
//!
//! * A shape is [`Shape::Tuple`] — a rank plus per-dim [`Dim`]s
//!   (`Some(i64)` concrete, `None` unknown extent, mirroring Python's
//!   `int | None` dims — `TensorType` rejects symbolic/negative dims
//!   at construction, so none appear here) — [`Shape::Unknown`]
//!   (Python `None`: merely unknown) or [`Shape::Invalid`]
//!   (Python `_INVALID`: *provably* ill-typed — poisonous, priced
//!   near-infinite, never a wildcard).
//! * **Unknown is never guessed.** Broadcast binaries decline to
//!   [`Shape::Unknown`] when *any* operand shape is unknown (the
//!   kokoro `transpose_noop` fix); only `ARG0_SHAPE_OPS` may answer
//!   operand 0's shape on a partial-unknown read, and never a `()`
//!   (scalar) where the op cannot produce one.
//! * Attr positions that are unbound, mistyped, or metavar strings
//!   resolve to "unknown", never a fabricated axis — the same
//!   `TypeError → None` contract `_shape_node` wraps `_infer_op_shape`
//!   in. Where Python would *raise* a non-`TypeError` (e.g. `int("D")`
//!   in `permute`), this port declines to [`Shape::Unknown`] — a
//!   documented, strictly-more-honest choice.
//!
//! # The bridge encoding, as shape inference sees it
//!
//! * A domain op is a `Term::Op` carrying a `"source"` attr; its
//!   **child 0 is the dim slot** and is never a tensor operand — the
//!   real args are `args[1..]` (the same `is_op_encoded` heuristic
//!   [`crate::cost`] uses). Non-sourced enodes (`arr`/`seq`/`app`/…)
//!   treat every child as an argument and land on the default arm.
//! * Op attrs live under the `"__attrs__"` key as a JSON object
//!   (`{"dim": 0, "shape": [2, 3], …}`); the `"lit"` attr is consulted
//!   as a fallback object. [`AttrArg`] positions in [`ShapeSpec`]s and
//!   the [`pred`] predicates resolve through [`BoundEnv::attrs`] —
//!   bare metavar names, no `$attr:` prefix (that prefix is only the
//!   record spelling).
//! * Op names are canonicalised by stripping a leading `tensor.` —
//!   the host fragment spells `tensor.matmul`, the builtin law packs
//!   spell `matmul`; both name the same shape rule.
//! * Leaves decode through `to_kernel`: `Const` reports `()` (a
//!   `{"__tensor__": {"shape": …}}` wire value reports its declared
//!   shape instead — the host-tensor leaf convention), `Opaque`
//!   reports a `"shape"`/`"__tensor__"` payload when present, and
//!   anything else is [`Shape::Unknown`]. A `Var` is pattern-only and
//!   never reaches inference — [`Shape::Unknown`].
//!
//! # The guard-predicate seam (stage 2)
//!
//! [`pred`] exposes the shape-reading half of `cond.py`'s predicate
//! table as functions over [`BoundEnv`] — the resolved-binding
//! environment `Rewrite::check` hooks already receive. Stage 2's
//! `cond`/`dspec` interpreter maps JSON atoms (`("shape-compat", A,
//! B)`, …) onto these; the non-shape atoms (`attr-*`, `const-*`,
//! `term-eq`, `op-in`, `leaf`, `const`, the `and`/`or`/`not`
//! combinators, and the `dspec` expr algebra) are stage-2's own
//! surface — this module deliberately does not pre-empt them. Every
//! predicate here **declines** (`false`) on anything it cannot prove,
//! the `cond.py` strictness contract.
//!
//! # Not ported (honest gaps)
//!
//! * `_SHAPE_RULES` — the carrier-family registrations (`om`,
//!   `applyd`, `aff`, `trace`, …) live in `mlopt_carriers` and are
//!   per-op *data* pushed in from outside core. They land here only
//!   when a carrier domain needs them; until then those ops take the
//!   default arm (`operand 0 or unknown`).
//! * `value_shape_of` / the `("vshape", T)` resolver — the
//!   carrier-aware value shape needs a registered backend resolver
//!   (the `tval`/`tapp` channel, design doc §2's out-of-scope list).
//!   [`ShapeSpec::Vshape`] resolves as the cost convention and says
//!   so.

use handl_egraph::{AttrValue, BoundEnv, KernelTerm, Op, Term, TermAttr, from_kernel, to_kernel};
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};

/// One shape extent: `Some(d)` a concrete dim, `None` an unknown
/// extent (Python's `None` *inside* a shape tuple — distinct from
/// [`Shape::Unknown`], the wholly-unknown wildcard).
pub type Dim = Option<i64>;

/// An inferred tensor shape — the port of `tuple | _INVALID | None`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Shape {
    /// A rank plus per-dim extents (`()` is the scalar shape — a
    /// perfectly good [`Shape::Tuple`], distinct from
    /// [`Shape::Unknown`]).
    Tuple(Vec<Dim>),
    /// Provably ill-typed — Python's `_INVALID` sentinel. Poisonous:
    /// cost models price it near-infinite and guards decline on it.
    Invalid,
    /// Unknown — Python's `None`. A wildcard *only* where a rule says
    /// so (`broadcast`'s missing-operand arm); everywhere else it
    /// propagates as unknown, never as a guessed shape.
    Unknown,
}

impl Shape {
    /// The shape as a dim slice, when it is a tuple (scalar `()`
    /// included — `Some(&[])`).
    pub fn as_tuple(&self) -> Option<&[Dim]> {
        match self {
            Shape::Tuple(d) => Some(d),
            _ => None,
        }
    }

    /// `_dim_at`: `(in_range, dim)` for index `i` (negative indexes
    /// from the end, Python semantics).
    pub fn dim_at(&self, i: i64) -> (bool, Dim) {
        let Some(d) = self.as_tuple() else {
            return (false, None);
        };
        let n = d.len() as i64;
        if n == 0 || !(-n..n).contains(&i) {
            return (false, None);
        }
        (true, d[i.rem_euclid(n) as usize])
    }

    /// Whether every extent is a concrete dim (Python's
    /// `all(isinstance(d, int) for d in s)`).
    pub fn is_concrete(&self) -> bool {
        matches!(self, Shape::Tuple(d) if d.iter().all(Option::is_some))
    }
}

/// `x or None` for shapes: `()` (scalar) reads as [`Shape::Unknown`]
/// — the carrier-internal-member convention.
fn non_scalar(s: Shape) -> Shape {
    match s {
        Shape::Tuple(d) if d.is_empty() => Shape::Unknown,
        other => other,
    }
}

/// Per-node inference result: the shape plus the best-effort dtype
/// (the `{"__tensor__": {"dtype": …}}` / op-attr `"dtype"` read —
/// `None` when nothing declares one; dtype is honest-unknown, never
/// guessed from op semantics).
#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    /// The inferred shape.
    pub shape: Shape,
    /// The declared dtype, when one is carried.
    pub dtype: Option<AttrValue>,
}

/// The memo table [`infer`]/[`shape_of`] share — keyed by node
/// *identity*, not structure.
///
/// `Term`'s derived `Hash`/`Eq` walks the fully-expanded tree, which
/// is exponential on the Rc-shared DAGs `dag.program` input and
/// `min_term_cached` extraction produce (the same blow-up
/// `handl_egraph::extract`'s `TermKey` and `crate::dag::NodeKey`
/// solved for cost/pricing walks). The key is the
/// `(tag, ptr, len-or-ptr2, ptr3)` identity tuple those modules
/// spell as a struct — inlined here so the public alias names only
/// public types. Pointer-equal `Rc` allocations are the same node;
/// pointer-distinct terms merely miss and recompute — the memo never
/// changes a *value*, only dedup work, so e-class merges (which never
/// mutate terms) cannot stale an entry.
///
/// The value keeps the `Term` alongside its [`TensorInfo`] so the
/// `Rc`s the keys point into cannot be freed and their addresses
/// reused while the memo lives.
pub type Memo = HashMap<(u8, usize, usize, usize), (TensorInfo, Term)>;

/// `(tag, ptr, len-or-ptr2, ptr3)` identity of a term node — the
/// `crate::dag::NodeKey` layout, kept tuple-shaped for the public
/// [`Memo`] alias.
fn term_key(t: &Term) -> (u8, usize, usize, usize) {
    match t {
        // `Rc<str>`'s own `as_ptr` is the *fat* `*const str` — go
        // through `str::as_bytes` for the thin data pointer.
        Term::Var(n) => (0, n.as_bytes().as_ptr() as usize, n.len(), 0),
        Term::Leaf(k) => (1, k.as_bytes().as_ptr() as usize, k.len(), 0),
        Term::Op(o) => (
            2,
            o.op.as_bytes().as_ptr() as usize,
            o.args.as_ptr() as usize,
            o.attrs.as_ptr() as usize,
        ),
    }
}

/// Look up a memoised [`TensorInfo`] by node identity.
fn memo_info<'m>(memo: &'m Memo, t: &Term) -> Option<&'m TensorInfo> {
    memo.get(&term_key(t)).map(|(info, _)| info)
}

/// Work budget for one [`infer_postorder`] walk — a cap on distinct
/// node inferences, not a time limit. Generous by construction: the
/// e-graph's own node caps (`max_nodes`, typically ~10⁴) bound every
/// legitimate term far below this; the cap exists so a pathological
/// input *declines* — unvisited nodes read as [`Shape::Unknown`],
/// the standard "cannot prove" channel every guard already honours —
/// instead of hanging the search.
const INFER_NODE_BUDGET: usize = 1 << 18;

/// Infer [`TensorInfo`] for `t` — shape and dtype, one postorder DAG
/// walk (the `_shape_of` iterative-frontier port; deep terms cannot
/// hit a recursion limit).
pub fn infer(t: &Term) -> TensorInfo {
    let mut memo = Memo::new();
    infer_with(t, &mut memo)
}

/// [`infer`] with a caller-shared memo — cost models pricing many
/// candidates reuse one table.
pub fn infer_with(t: &Term, memo: &mut Memo) -> TensorInfo {
    infer_postorder(t, memo);
    memo_info(memo, t).cloned().unwrap_or(TensorInfo {
        shape: Shape::Unknown,
        dtype: None,
    })
}

/// The shape half of [`infer`] — `_shape_of`.
pub fn shape_of(t: &Term) -> Shape {
    infer(t).shape
}

/// [`shape_of`] with a shared memo.
pub fn shape_of_with(t: &Term, memo: &mut Memo) -> Shape {
    infer_with(t, memo).shape
}

/// The dtype half of [`infer`].
pub fn dtype_of(t: &Term) -> Option<AttrValue> {
    infer(t).dtype
}

/// Postorder walk filling `memo` for `t` and every descendant —
/// children before parents, pruning at already-memoised nodes (the
/// `postorder(term, prune=…)` port).
fn infer_postorder(t: &Term, memo: &mut Memo) {
    let mut stack: Vec<(Term, bool)> = vec![(t.clone(), false)];
    let mut budget = INFER_NODE_BUDGET;
    while let Some((node, expanded)) = stack.pop() {
        let key = term_key(&node);
        if memo.contains_key(&key) {
            continue;
        }
        if !expanded {
            stack.push((node.clone(), true));
            for child in node_children(&node) {
                if !memo.contains_key(&term_key(child)) {
                    stack.push((child.clone(), false));
                }
            }
            continue;
        }
        if budget == 0 {
            // Budget exhausted: stop the walk. Everything left
            // unvisited reads as `Shape::Unknown` downstream — the
            // honest decline — so a pathological term vetoes guards
            // rather than hanging them.
            if std::env::var_os("EQSAT_DEBUG").is_some() {
                eprintln!("[eqsat-dbg] shape infer node budget exhausted");
            }
            break;
        }
        budget -= 1;
        let info = infer_node(&node, memo);
        // The value keeps `node` alive so `key`'s pointers stay
        // valid for the memo's lifetime.
        memo.insert(key, (info, node));
    }
}

/// All child terms of a node (the dim slot included — it is never
/// *read* as a tensor operand, but memoising it keeps the walk total).
fn node_children(t: &Term) -> &[Term] {
    match t {
        Term::Var(_) | Term::Leaf(_) => &[],
        Term::Op(o) => &o.args,
    }
}

/// The canonical `Term::Arr` e-graph spelling — an `arr` enode with
/// exactly one child (the dim slot) and exactly one `Lit` attr named
/// `value`, the inverse of `dag.rs`'s `arr_leaf_record`. Returns the
/// `value` payload as JSON when the enode has that shape; `None` for
/// any other `arr` enode (a different arity/attr set is *not* the
/// `Term::Arr` spelling and falls through to ordinary op handling).
fn arr_leaf_value(o: &Op) -> Option<Value> {
    if &*o.op != "arr" || o.args.len() != 1 || o.attrs.len() != 1 {
        return None;
    }
    let (k, v) = &o.attrs[0];
    if k.as_ref() != "value" {
        return None;
    }
    match v {
        TermAttr::Lit(av) => Some(av.to_json()),
        TermAttr::Meta(_) => None,
    }
}

/// Whether `o` is the canonical `arr` enode spelling of an mlopt
/// bridge leaf — a `{"__mlopt_leaf__": …}` record inside `Arr.value`.
/// These enodes *are* `Var`/`Param` leaves wearing the canonical
/// `Arr` wire spelling (the `leaf="arr"` profile); every leaf-shaped
/// read — declared shape, `leaf-dim`, leaf pricing — must see them
/// exactly as it saw the `Term::Leaf` `const-str` spelling, or the
/// two leaf profiles stop canonicalising identically.
pub(crate) fn is_bridge_leaf_enode(o: &Op) -> bool {
    arr_leaf_value(o).is_some_and(|v| v.get("__mlopt_leaf__").is_some())
}

/// Compute one node's [`TensorInfo`]; every operand is already
/// memoised by the postorder driver.
fn infer_node(t: &Term, memo: &Memo) -> TensorInfo {
    match t {
        // A metavariable is pattern-only — `add_term` rejects it, so
        // this is defensive: unknown, never a guess.
        Term::Var(_) => TensorInfo {
            shape: Shape::Unknown,
            dtype: None,
        },
        Term::Leaf(key) => leaf_info(key),
        Term::Op(o) => match arr_leaf_value(o) {
            // The canonical `arr` leaf spelling carries its declared
            // shape inside `Arr.value` — read it through the same
            // leaf channels a `Term::Leaf` key decodes through, so
            // `arr` and `const-str` leaves infer identically.
            Some(v) => json_leaf_info(&v),
            None => infer_op(o, memo),
        },
    }
}

/// Whether `o` is a domain op in the `Term::Op` encoding — child 0 is
/// the dim slot (the `"source"` attr heuristic `to_kernel`/`cost`
/// share).
fn is_op_encoded(o: &Op) -> bool {
    o.attrs.iter().any(|(k, _)| k.as_ref() == "source")
}

/// The real tensor operands of an op node: `args[1..]` for
/// op-encoded nodes (child 0 is the dim slot), `args` otherwise.
fn op_args(o: &Op) -> &[Term] {
    if is_op_encoded(o) && !o.args.is_empty() {
        &o.args[1..]
    } else {
        &o.args[..]
    }
}

/// Canonical op name for shape dispatch — `tensor.matmul` and
/// `matmul` name the same rule (the host fragment's prefix is a
/// namespace, not a different operator).
fn canon_op(op: &str) -> &str {
    op.strip_prefix("tensor.").unwrap_or(op)
}

/// Read a named attr: the `"__attrs__"` JSON object first, then the
/// `"lit"` attr when it is itself an object (the two payload slots a
/// kernel `Term::Op` carries).
fn attr_of<'a>(o: &'a Op, names: &[&str]) -> Option<&'a AttrValue> {
    for key in ["__attrs__", "lit"] {
        let Some(AttrValue::Object(map)) = lit_attr(o, key) else {
            continue;
        };
        for name in names {
            if let Some(v) = map.get(*name) {
                return Some(v);
            }
        }
    }
    None
}

/// Fetch a literal attr value by exact key.
fn lit_attr<'a>(o: &'a Op, key: &str) -> Option<&'a AttrValue> {
    o.attrs.iter().find_map(|(k, v)| {
        if k.as_ref() == key
            && let TermAttr::Lit(v) = v
        {
            return Some(v);
        }
        None
    })
}

/// `AttrValue` → `i64` under the Python `isinstance(d, int)` contract:
/// ints (and bools — `isinstance(True, int)` in Python) admit, floats
/// and strings do not.
fn av_int(v: &AttrValue) -> Option<i64> {
    match v {
        AttrValue::Int(n) => Some(*n),
        AttrValue::UInt(n) => i64::try_from(*n).ok(),
        AttrValue::Bool(b) => Some(i64::from(*b)),
        _ => None,
    }
}

/// The mlopt attr wire envelopes — `mlopt_core.ir.attr_to_data`'s
/// `{"__tuple__": […]}` / `{"__list__": […]}` encoding of Python
/// `tuple`/`list` attr values. Every shape read below admits
/// Python's `isinstance(x, (tuple, list))` union interchangeably,
/// so both envelopes — and a bare `Array` (the port's tuple normal
/// form, e.g. a dspec-derived attr before re-encoding) — yield the
/// same item view. `None` for anything else, including a marker key
/// with a non-array payload or a multi-key map.
fn seq_items(v: &AttrValue) -> Option<&[AttrValue]> {
    match v {
        AttrValue::Array(items) => Some(items),
        AttrValue::Object(map) if map.len() == 1 => {
            if let Some(AttrValue::Array(items)) = map.get("__tuple__") {
                Some(items)
            } else if let Some(AttrValue::Array(items)) = map.get("__list__") {
                Some(items)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// `AttrValue` → int tuple under the `isinstance(dim, (tuple, list))`
/// contract — a non-int *element* fails the whole read (Python's
/// `all(isinstance(d, int))` gates).
fn av_ints(v: &AttrValue) -> Option<Vec<i64>> {
    let items = seq_items(v)?;
    items.iter().map(av_int).collect()
}

/// A shape-attr array (`"shape"`/`"sizes"`/`"repeats"`) → dims:
/// ints admit, `null` is an unknown extent, anything else is an
/// unknown extent too (Python would carry the raw value; `None` is
/// the honest analog — a metavar string is "not yet known", never a
/// fabricated dim).
fn attr_dims(v: &AttrValue) -> Option<Vec<Dim>> {
    let items = seq_items(v)?;
    Some(
        items
            .iter()
            .map(|d| match d {
                AttrValue::Int(n) => Some(*n),
                AttrValue::UInt(n) => i64::try_from(*n).ok(),
                AttrValue::Bool(b) => Some(i64::from(*b)),
                _ => None,
            })
            .collect(),
    )
}

/// Python `bool(attr)` truthiness for `keepdim`/`or`-default flags.
/// Sequences read through the wire envelopes first —
/// `{"__tuple__": []}` is the empty tuple, falsy like Python's `()`,
/// not a non-empty object.
fn av_truthy(v: &AttrValue) -> bool {
    if let Some(items) = seq_items(v) {
        return !items.is_empty();
    }
    match v {
        AttrValue::Null => false,
        AttrValue::Bool(b) => *b,
        AttrValue::Int(n) => *n != 0,
        AttrValue::UInt(n) => *n != 0,
        AttrValue::Float(f) => *f != 0.0,
        AttrValue::Str(s) => !s.is_empty(),
        AttrValue::Array(_) => unreachable!("seq_items covers Array"),
        AttrValue::Object(o) => !o.is_empty(),
    }
}

/// `_conv_scalar`: a 1-D conv window attr — a sequence's first
/// element, else the value itself.
fn conv_scalar(v: Option<&AttrValue>) -> Option<AttrValue> {
    match v {
        Some(v) => match seq_items(v) {
            Some([first, ..]) => Some(first.clone()),
            _ => Some(v.clone()),
        },
        None => None,
    }
}

/// `_conv_pair`: a 2-D conv window attr as a pair; `(None, None)` when
/// malformed (non-pair sequence).
fn conv_pair(v: Option<&AttrValue>) -> (Option<AttrValue>, Option<AttrValue>) {
    match v {
        Some(v) => match seq_items(v) {
            Some([a, b]) => (Some(a.clone()), Some(b.clone())),
            Some(_) => (None, None),
            None => (Some(v.clone()), Some(v.clone())),
        },
        None => (None, None),
    }
}

/// `_ax_tuple`: an axis attr — an int or a list/tuple of ints — as a
/// vec; `None` for anything else (including non-int elements — the
/// honest decline where Python's `int(d)` could raise).
fn ax_tuple(v: Option<&AttrValue>) -> Option<Vec<i64>> {
    match v {
        Some(a) => av_ints(a).or_else(|| av_int(a).map(|d| vec![d])),
        None => None,
    }
}

/// `_numel`: product of dims, `None` dims counting 1, non-tuple → 1.
/// Saturating at `i64::MAX` — Python ints are unbounded; a saturating
/// product can only *over*price, never under-price.
pub fn numel(shape: &Shape) -> i64 {
    let Shape::Tuple(dims) = shape else {
        return 1;
    };
    dims.iter()
        .fold(1i64, |acc, d| acc.saturating_mul(d.unwrap_or(1).max(0)))
}

/// The strict numel: `Some(product)` only when the shape is a tuple
/// with *every* dim concrete — the "shapes fully known" precondition
/// the `flops` formulas require (an unknown extent is `None`, never 1).
pub fn numel_concrete(shape: &Shape) -> Option<i64> {
    let Shape::Tuple(dims) = shape else {
        return None;
    };
    dims.iter().try_fold(1i64, |acc, d| acc.checked_mul((*d)?))
}

/// `_cat_empty`: a provably numel-0 shape — a literal `0` dim.
fn cat_empty(shape: &Shape) -> bool {
    matches!(shape, Shape::Tuple(d) if !d.is_empty() && d.contains(&Some(0)))
}

/// `broadcast`: numpy/PyTorch broadcasting of two shapes — the
/// `_broadcast` port.
///
/// `Unknown` is a wildcard — `broadcast(Unknown, s) = s` (the
/// missing-operand arm; callers that must decline on unknown check
/// before calling). `Invalid` is poisonous and propagates.
pub fn broadcast(a: &Shape, b: &Shape) -> Shape {
    match (a, b) {
        (Shape::Unknown, other) | (other, Shape::Unknown) => other.clone(),
        (Shape::Invalid, _) | (_, Shape::Invalid) => Shape::Invalid,
        (Shape::Tuple(x), Shape::Tuple(y)) => {
            let ndim = x.len().max(y.len());
            let x_pad = ndim - x.len();
            let y_pad = ndim - y.len();
            let mut out = Vec::with_capacity(ndim);
            for i in 0..ndim {
                // Left-pad with literal 1s (broadcast alignment).
                let da = if i >= x_pad { x[i - x_pad] } else { Some(1) };
                let db = if i >= y_pad { y[i - y_pad] } else { Some(1) };
                // Negative dims are unresolved -1 markers: unknown,
                // not a mismatch.
                let da = da.filter(|d| *d >= 0);
                let db = db.filter(|d| *d >= 0);
                out.push(match (da, db) {
                    (None, _) | (_, None) => None,
                    (Some(1), Some(db)) => Some(db),
                    (Some(da), Some(1)) => Some(da),
                    (Some(da), Some(db)) if da == db => Some(da),
                    _ => return Shape::Invalid,
                });
            }
            Shape::Tuple(out)
        }
    }
}

/// `_dim_eq`: `None` (unknown) dims are wildcards.
fn dim_eq(x: Dim, y: Dim) -> bool {
    x.is_none() || y.is_none() || x == y
}

/// The simple matmul-shape read shared by the `matmul` match arm and
/// `addmm`'s inline GEMM — `(*a[:-1], b[-1])` for rank ≥ 2, the
/// rank-1 promotions, `()` for the dot product, `None` otherwise.
/// Deliberately *not* `_matmul_shape` (which batch-broadcasts); the
/// Python `_infer_op_shape` `matmul` arm is this simpler read — the
/// guard-facing [`matmul_shape`] exposes full torch semantics.
fn mm_simple(a: &[Dim], b: &[Dim]) -> Option<Vec<Dim>> {
    if a.len() >= 2 && b.len() >= 2 {
        let mut out = a[..a.len() - 1].to_vec();
        out.push(b[b.len() - 1]);
        return Some(out);
    }
    if a.len() >= 2 && b.len() == 1 {
        return Some(a[..a.len() - 1].to_vec());
    }
    if a.len() == 1 && b.len() >= 2 {
        let mut out = b[..b.len() - 2].to_vec();
        out.push(b[b.len() - 1]);
        return Some(out);
    }
    if a.len() == 1 && b.len() == 1 {
        return Some(Vec::new());
    }
    None
}

/// `_matmul_shape`: `torch.matmul` output shape for operand shapes —
/// full torch semantics (rank-1 promotion/demotion, batch-dim
/// broadcasting).
///
/// Returns `Some(dims)` on a valid contraction (the result may carry
/// `None` extents), `None` on an invalid one (inner-dim mismatch or a
/// scalar/empty/non-tuple operand) — the `mm-shape-ok` predicate's
/// `is_some`. `Invalid`/`Unknown` inputs are not tuples → `None`.
pub fn matmul_shape(a: &Shape, b: &Shape) -> Option<Vec<Dim>> {
    let (Shape::Tuple(a), Shape::Tuple(b)) = (a, b) else {
        return None;
    };
    if a.is_empty() || b.is_empty() {
        return None;
    }
    if a.len() == 1 && b.len() == 1 {
        return dim_eq(a[0], b[0]).then(Vec::new);
    }
    if b.len() == 1 {
        if a.len() < 2 || !dim_eq(a[a.len() - 1], b[0]) {
            return None;
        }
        return Some(a[..a.len() - 1].to_vec());
    }
    if a.len() == 1 {
        if !dim_eq(a[0], b[b.len() - 2]) {
            return None;
        }
        let mut out = b[..b.len() - 2].to_vec();
        out.push(b[b.len() - 1]);
        return Some(out);
    }
    let batch = broadcast(
        &Shape::Tuple(a[..a.len() - 2].to_vec()),
        &Shape::Tuple(b[..b.len() - 2].to_vec()),
    );
    if batch == Shape::Invalid || !dim_eq(a[a.len() - 1], b[b.len() - 2]) {
        return None;
    }
    let Shape::Tuple(batch) = batch else {
        return None;
    };
    let mut out = batch;
    out.push(a[a.len() - 2]);
    out.push(b[b.len() - 1]);
    Some(out)
}

/// `_conv_window`: one conv output extent —
/// `(n + 2*pd - dl*(k-1) - 1) // st + 1`.
enum ConvWindow {
    /// Concrete extent.
    Extent(i64),
    /// Concrete non-positive stride/dilation — torch refuses both:
    /// provably ill-typed, not merely unknown.
    Invalid,
    /// Undecidable — unknown operand dim, malformed spelling, or an
    /// attr metavar.
    Unknown,
}

fn conv_window(
    n: Dim,
    k: Dim,
    st: Option<&AttrValue>,
    pd: Option<&AttrValue>,
    dl: Option<&AttrValue>,
) -> ConvWindow {
    let st_i = st.and_then(av_int);
    let pd_i = pd.and_then(av_int);
    let dl_i = dl.and_then(av_int);
    if st_i.is_some_and(|s| s <= 0) || dl_i.is_some_and(|d| d <= 0) {
        return ConvWindow::Invalid;
    }
    let (Some(st), Some(pd), Some(dl)) = (st_i, pd_i, dl_i) else {
        return ConvWindow::Unknown;
    };
    let (Some(n), Some(k)) = (n, k) else {
        return ConvWindow::Unknown;
    };
    // Python `//` is floor division.
    ConvWindow::Extent((n + 2 * pd - dl * (k - 1) - 1).div_euclid(st) + 1)
}

/// `_index_shape`: numpy-style advanced indexing — the `layout` attr
/// marks which key positions carry index operands (`True`) vs full
/// slices (`False`); contiguous index positions splice the broadcast
/// index shape in place, separated ones move it to the front.
fn index_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let Some(base) = shapes.first().and_then(Shape::as_tuple) else {
        return Shape::Unknown;
    };
    if base.is_empty() {
        return Shape::Unknown;
    }
    let idx_shapes = &shapes[1..];
    let mut ib = idx_shapes
        .first()
        .cloned()
        .unwrap_or(Shape::Tuple(Vec::new()));
    for s in &idx_shapes[1..] {
        ib = broadcast(&ib, s);
        if ib == Shape::Invalid {
            return Shape::Invalid;
        }
    }
    // `layout`: one flag per key position. Absent or empty → the
    // `x[i]` spelling (every index operand on the leading axes).
    let layout: Vec<bool> = match attr_of(o, &["layout"]).and_then(seq_items) {
        Some(flags) if !flags.is_empty() => flags.iter().map(av_truthy).collect(),
        _ => vec![true; idx_shapes.len()],
    };
    let consumed = layout.len().min(base.len());
    let trailing: Vec<Dim> = base[consumed..].to_vec();
    let idx_pos: Vec<usize> = (0..layout.len()).filter(|&i| layout[i]).collect();
    let contiguous =
        !idx_pos.is_empty() && idx_pos.last().unwrap() - idx_pos[0] + 1 == idx_pos.len();
    let ib_dims: Vec<Dim> = ib.as_tuple().map(<[Dim]>::to_vec).unwrap_or_default();
    let mut kept: Vec<Dim> = Vec::new();
    if contiguous {
        let first_idx = idx_pos[0];
        let mut kept_before = Vec::new();
        let mut kept_after = Vec::new();
        for (i, flag) in layout[..consumed].iter().enumerate() {
            if *flag {
                continue;
            }
            if i < first_idx {
                kept_before.push(base[i]);
            } else {
                kept_after.push(base[i]);
            }
        }
        kept.extend(kept_before);
        kept.extend(ib_dims.iter().copied());
        kept.extend(kept_after);
    } else {
        // Separated index positions: the index block moves to front.
        let mut front = ib_dims;
        for (i, flag) in layout[..consumed].iter().enumerate() {
            if !flag {
                front.push(base[i]);
            }
        }
        kept = front;
    }
    kept.extend(trailing);
    Shape::Tuple(kept)
}

/// `_fold_minus1`: fold `-1` dims of a reshape target through the base
/// numel; an unresolvable fold leaves `None` in that slot.
fn fold_minus1(target: &[Dim], base_n: i64) -> Vec<Dim> {
    if !target.contains(&Some(-1)) {
        return target.to_vec();
    }
    let mut known: i64 = 1;
    for d in target {
        if *d != Some(-1) {
            known = known.saturating_mul(match d {
                Some(n) if *n > 0 => *n,
                _ => 1,
            });
        }
    }
    let inferred = if known != 0 && base_n % known == 0 {
        Some(base_n / known)
    } else {
        None
    };
    target
        .iter()
        .map(|d| if *d == Some(-1) { inferred } else { *d })
        .collect()
}

/// The `reshape` numel check (`_infer_op_shape`'s reshape arm):
/// `Invalid` when the declared target provably mismatches the input
/// numel — both sides fully concrete non-negative ints.
fn reshape_check(target: &[Dim], base: &Shape) -> Shape {
    if target.iter().flatten().any(|d| *d < 0) {
        return Shape::Invalid;
    }
    let base_dims = base.as_tuple();
    let concrete = |d: &[Dim]| d.iter().all(|x| x.is_some_and(|n| n >= 0));
    if let Some(b) = base_dims
        && concrete(target)
        && concrete(b)
    {
        let tn: i64 = target.iter().flatten().product();
        let bn: i64 = b.iter().flatten().product();
        if tn != bn {
            return Shape::Invalid;
        }
    }
    Shape::Tuple(target.to_vec())
}

/// Decode a positional shape-list argument — a `Const` leaf whose
/// value is an array of ints/nulls (the `tensor.reshape` fragment
/// signature `(Tensor * (List Int)) -> Tensor` spells the target as a
/// child, not an attr).
fn positional_shape_arg(args: &[Term]) -> Option<Vec<Dim>> {
    let arg = args.get(1)?; // args[0] is the tensor operand
    let Term::Leaf(key) = arg else {
        return None;
    };
    let KernelTerm::Const { value } = to_kernel(&Term::Leaf(key.clone())) else {
        return None;
    };
    let Value::Array(items) = value else {
        return None;
    };
    // Non-int elements → unknown dims, never a guess.
    Some(
        items
            .iter()
            .map(|v| match v {
                Value::Number(n) => n
                    .as_i64()
                    .or_else(|| n.as_u64().and_then(|u| i64::try_from(u).ok())),
                _ => None,
            })
            .collect(),
    )
}

/// The `Op` dispatch — `_infer_op_shape`'s match arms over bridge
/// terms. Returns `(shape, dtype)`; `dtype` inherits operand 0's
/// declaration unless an attr overrides.
fn infer_op(o: &Op, memo: &Memo) -> TensorInfo {
    let name = canon_op(&o.op);
    let args = op_args(o);
    let shapes: Vec<Shape> = args
        .iter()
        .map(|a| {
            memo_info(memo, a)
                .map(|i| i.shape.clone())
                .unwrap_or(Shape::Unknown)
        })
        .collect();
    let dtype0 = args
        .first()
        .and_then(|a| memo_info(memo, a))
        .and_then(|i| i.dtype.clone());
    let dtype = attr_of(o, &["dtype"]).cloned().or(dtype0);
    let shape = infer_op_shape(name, o, args, &shapes);
    TensorInfo { shape, dtype }
}

/// `_infer_op_shape`: the per-op match. `args`/`shapes` are the real
/// operands (dim slot already stripped); inside the match every
/// operand shape is a [`Shape::Tuple`] (the early return filtered
/// `Unknown`/`Invalid`).
fn infer_op_shape(name: &str, o: &Op, args: &[Term], shapes: &[Shape]) -> Shape {
    // `_INVALID` propagates before anything else.
    if shapes.contains(&Shape::Invalid) {
        return Shape::Invalid;
    }
    if shapes.is_empty() || shapes.contains(&Shape::Unknown) {
        // Broadcast binaries FIRST: output is the broadcast of EVERY
        // operand — a `None` sibling means genuinely unknown (the
        // kokoro transpose_noop fix).
        if BROADCAST_BINOPS.contains(&name) {
            return Shape::Unknown;
        }
        // Operand 0's shape answers ONLY for `_ARG0_SHAPE_OPS`.
        if let Some(first) = shapes.first()
            && *first != Shape::Unknown
            && ARG0_SHAPE_OPS.contains(&name)
        {
            return first.clone();
        }
        return Shape::Unknown;
    }
    match name {
        "matmul" => {
            let (Some(a), Some(b)) = (shapes.first(), shapes.get(1)) else {
                return Shape::Unknown;
            };
            match (a.as_tuple(), b.as_tuple()) {
                (Some(a), Some(b)) => mm_simple(a, b).map(Shape::Tuple).unwrap_or(Shape::Unknown),
                _ => Shape::Unknown,
            }
        }
        "add" | "sub" | "mul" | "div" | "pow" | "maximum" | "minimum" | "fmax" | "fmin"
        | "fmod" | "remainder" | "xlogy" | "atan2" | "heaviside" | "isclose" | "eq" | "ne"
        | "lt" | "le" | "gt" | "ge" | "logical_and" | "logical_or" | "logical_xor"
        | "bitwise_and" | "bitwise_or" => {
            // `_broadcast(shapes[0], shapes[1] if len>1 else None)` —
            // a missing second operand broadcasts against unknown
            // (wildcard → operand 0's shape).
            let a = shapes.first().cloned().unwrap_or(Shape::Unknown);
            let b = shapes.get(1).cloned().unwrap_or(Shape::Unknown);
            broadcast(&a, &b)
        }
        "lerp" | "addcmul" | "addcdiv" => {
            let mut out = shapes.first().cloned().unwrap_or(Shape::Unknown);
            for s in shapes.iter().skip(1) {
                out = broadcast(&out, s);
                if out == Shape::Invalid {
                    return Shape::Invalid;
                }
            }
            out
        }
        "where" => {
            let out = broadcast(
                shapes.get(1).unwrap_or(&Shape::Unknown),
                shapes.get(2).unwrap_or(&Shape::Unknown),
            );
            broadcast(&out, shapes.first().unwrap_or(&Shape::Unknown))
        }
        name if UNARY_PASSTHROUGH.contains(&name) => {
            shapes.first().cloned().unwrap_or(Shape::Unknown)
        }
        name if AXIS_ANCHORED.contains(&name) => {
            // `shapes[0] or None` — a `()` operand means a
            // carrier-internal member read as a tensor: unknown.
            non_scalar(shapes.first().cloned().unwrap_or(Shape::Unknown))
        }
        "linear" => linear_shape(shapes),
        name if REDUCE_OPS.contains(&name) => reduce_shape(o, shapes),
        "transpose" => transpose_shape(o, shapes),
        "reshape" => reshape_shape(o, shapes, args),
        "unsqueeze" => unsqueeze_shape(o, shapes),
        "squeeze" => squeeze_shape(o, shapes),
        "expand" | "broadcast_to" => expand_shape(o, shapes),
        "repeat" => repeat_shape(o, shapes),
        "permute" => permute_shape(o, shapes),
        "movedim" => movedim_shape(o, shapes),
        "expand_as" => shapes
            .get(1)
            .cloned()
            .unwrap_or_else(|| shapes.first().cloned().unwrap_or(Shape::Unknown)),
        "stack" => stack_shape(o, args, shapes),
        "unbind" => drop_axis_shape(o, shapes, -1),
        "getitem" => shapes.first().cloned().unwrap_or(Shape::Unknown),
        "select" => drop_axis_shape(o, shapes, 0),
        "slice" => slice_shape(o, shapes),
        "embedding" => embedding_shape(shapes),
        "index_select" => index_select_shape(o, shapes),
        "gather" | "take_along_dim" | "searchsorted" => {
            if shapes.len() < 2 {
                return non_scalar(shapes.first().cloned().unwrap_or(Shape::Unknown));
            }
            non_scalar(shapes[1].clone())
        }
        "index" => index_shape(o, shapes),
        "unflatten" => unflatten_shape(o, shapes),
        "narrow" => narrow_shape(o, shapes),
        "pad" => pad_shape(o, shapes),
        "glu" => glu_shape(o, shapes),
        "nonzero" => {
            let rank = shapes
                .first()
                .and_then(Shape::as_tuple)
                .map(<[Dim]>::len)
                .unwrap_or(0);
            Shape::Tuple(vec![None, Some(rank as i64)])
        }
        "one_hot" => {
            let mut out = shapes[0]
                .as_tuple()
                .map(<[Dim]>::to_vec)
                .unwrap_or_default();
            let nc = attr_of(o, &["num_classes"]).and_then(av_int);
            out.push(nc.filter(|n| *n > 0));
            Shape::Tuple(out)
        }
        "outer" => Shape::Tuple(vec![Some(numel(&shapes[0])), Some(numel(&shapes[1]))]),
        "diag_sum" | "item" | "numel" => Shape::Tuple(Vec::new()),
        // The equation string is opaque to the structural walker —
        // honest unknown rather than a wrong guess.
        "einsum" => Shape::Unknown,
        "addmm" | "addbmm" | "baddbmm" | "addmv" => addmm_shape(shapes),
        "conv1d" => conv_shape(o, shapes, 1),
        "conv2d" => conv_shape(o, shapes, 2),
        "tensor_split" => tensor_split_shape(o, shapes),
        "unfold" => unfold_shape(o, shapes),
        "pixel_shuffle" => pixel_shuffle_shape(o, shapes, "upscale_factor", true),
        "pixel_unshuffle" => pixel_shuffle_shape(o, shapes, "downscale_factor", false),
        "hstack" | "vstack" => hstack_vstack_shape(name, args, shapes),
        "flatten" => flatten_shape(o, shapes),
        "contiguous" | "to" | "type_as" | "float" | "dropout" | "alias" | "sdpa" | "inv" => {
            shapes.first().cloned().unwrap_or(Shape::Unknown)
        }
        "concat" => concat_shape(o, shapes),
        "chunk" => chunk_shape(o, shapes),
        "split" => split_shape(o, shapes),
        // Unknown ops pass through operand 0's shape — `()` reads as
        // carrier-internal → unknown (the `case _` arm).
        _ => non_scalar(shapes.first().cloned().unwrap_or(Shape::Unknown)),
    }
}

/// `F.linear(x[..., in], W[out, in]) -> [..., out]` — the bias
/// (arg 2) broadcasts on, with the column-bias `(o,1)` squeeze rule.
fn linear_shape(shapes: &[Shape]) -> Shape {
    let (Some(x), Some(w)) = (shapes.first(), shapes.get(1)) else {
        return Shape::Unknown;
    };
    let (Some(x), Some(w)) = (x.as_tuple(), w.as_tuple()) else {
        return Shape::Unknown;
    };
    if x.is_empty() || w.is_empty() {
        return Shape::Unknown;
    }
    let mut out: Vec<Dim> = x[..x.len() - 1].to_vec();
    out.push(w[0]);
    let Some(b) = shapes.get(2) else {
        return Shape::Tuple(out);
    };
    let Shape::Tuple(b) = b else {
        return Shape::Unknown;
    };
    // Column bias (o,1): prefer the squeezed (o,) broadcast so the
    // bias lands on the last axis; fall back to raw on Invalid.
    if b.len() >= 2 && b[b.len() - 1] == Some(1) {
        let squeezed = Shape::Tuple(b[..b.len() - 1].to_vec());
        let mut ob = broadcast(&Shape::Tuple(out.clone()), &squeezed);
        if ob == Shape::Invalid {
            ob = broadcast(&Shape::Tuple(out.clone()), &Shape::Tuple(b.clone()));
        }
        return ob;
    }
    broadcast(&Shape::Tuple(out), &Shape::Tuple(b.clone()))
}

/// The reduction family (`sum`/`mean`/`prod`/…): honor `dim`/`axis`
/// and `keepdim`; no dim → scalar `()`.
fn reduce_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let base = match shapes.first() {
        Some(Shape::Tuple(d)) => d.clone(),
        _ => return Shape::Tuple(Vec::new()),
    };
    let Some(dim) = attr_of(o, &["dim", "axis"]) else {
        return Shape::Tuple(Vec::new());
    };
    let keep = attr_of(o, &["keepdim"]).is_some_and(av_truthy);
    if base.is_empty() {
        // Explicit-dim reduce over a scalar — the carrier-internal
        // `()` case: unknown, never scalar.
        return Shape::Unknown;
    }
    let dims: Vec<i64> = match av_ints(dim).or_else(|| av_int(dim).map(|d| vec![d])) {
        Some(d) => d,
        None => return Shape::Unknown,
    };
    let ndim = base.len() as i64;
    let norm: BTreeSet<i64> = dims.iter().map(|d| d.rem_euclid(ndim)).collect();
    if keep {
        Shape::Tuple(
            base.iter()
                .enumerate()
                .map(|(i, d)| {
                    if norm.contains(&(i as i64)) {
                        Some(1)
                    } else {
                        *d
                    }
                })
                .collect(),
        )
    } else {
        Shape::Tuple(
            base.iter()
                .enumerate()
                .filter(|(i, _)| !norm.contains(&(*i as i64)))
                .map(|(_, d)| *d)
                .collect(),
        )
    }
}

/// `transpose`: swap `dim0`/`dim1` (defaults -2/-1) of the base.
/// A present-but-non-int axis attr is the metavar case → unknown.
fn transpose_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let Some(Shape::Tuple(base)) = shapes.first() else {
        return Shape::Unknown;
    };
    if base.is_empty() {
        return Shape::Unknown;
    }
    let axis = |names: &[&str], default: i64| -> Option<i64> {
        match attr_of(o, names) {
            None => Some(default),
            Some(v) => av_int(v),
        }
    };
    let (Some(d0), Some(d1)) = (axis(&["dim0"], -2), axis(&["dim1"], -1)) else {
        return Shape::Unknown;
    };
    let n = base.len() as i64;
    let (d0, d1) = (d0.rem_euclid(n) as usize, d1.rem_euclid(n) as usize);
    let mut out = base.clone();
    out.swap(d0, d1);
    Shape::Tuple(out)
}

/// `reshape`: the `"shape"` attr (or a positional `Const` shape-list
/// arg — the `tensor.reshape` fragment signature) with `-1` folding
/// and the numel check; no declared shape → operand 0's shape.
fn reshape_shape(o: &Op, shapes: &[Shape], args: &[Term]) -> Shape {
    let target: Option<Vec<Dim>> = match attr_of(o, &["shape"]) {
        Some(v) => attr_dims(v), // present but non-array → None → Unknown
        None => positional_shape_arg(args),
    };
    let Some(target) = target else {
        return if attr_of(o, &["shape"]).is_some() {
            Shape::Unknown
        } else {
            shapes.first().cloned().unwrap_or(Shape::Unknown)
        };
    };
    let folded = fold_minus1(&target, numel(&shapes[0]));
    reshape_check(&folded, &shapes[0])
}

/// `unsqueeze`: insert a 1 at `dim % (rank+1)`; non-int dim (attr
/// metavar) → unknown.
fn unsqueeze_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let Some(Shape::Tuple(base)) = shapes.first() else {
        return Shape::Unknown;
    };
    let d = match attr_of(o, &["dim"]) {
        None => -1,
        Some(v) => match av_int(v) {
            Some(d) => d,
            None => return Shape::Unknown,
        },
    };
    let nd = d.rem_euclid(base.len() as i64 + 1) as usize;
    let mut out = base[..nd].to_vec();
    out.push(Some(1));
    out.extend_from_slice(&base[nd..]);
    Shape::Tuple(out)
}

/// `squeeze`: no dim → drop every known size-1 axis (unknown extents
/// stay); with dim → drop that axis.
fn squeeze_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let Some(Shape::Tuple(base)) = shapes.first() else {
        return Shape::Unknown;
    };
    if base.is_empty() {
        return Shape::Unknown;
    }
    match attr_of(o, &["dim"]) {
        None => Shape::Tuple(base.iter().copied().filter(|d| *d != Some(1)).collect()),
        Some(v) => {
            let Some(d) = av_int(v) else {
                return Shape::Unknown;
            };
            let nd = d.rem_euclid(base.len() as i64) as usize;
            Shape::Tuple(
                base.iter()
                    .enumerate()
                    .filter(|(i, _)| *i != nd)
                    .map(|(_, d)| *d)
                    .collect(),
            )
        }
    }
}

/// `expand`/`broadcast_to`: the `shape` attr, `-1` keeping the input
/// extent right-aligned; other negatives → `Invalid`.
fn expand_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let Some(s) = attr_of(o, &["shape"]).and_then(attr_dims) else {
        return shapes.first().cloned().unwrap_or(Shape::Unknown);
    };
    let base = shapes[0]
        .as_tuple()
        .map(<[Dim]>::to_vec)
        .unwrap_or_default();
    let mut resolved: Vec<Dim> = Vec::with_capacity(s.len());
    for (j, d) in s.iter().enumerate() {
        match d {
            Some(-1) => {
                // -1 keeps the input extent at the right-aligned slot.
                let k = j as i64 - (s.len() as i64 - base.len() as i64);
                if !(0..base.len() as i64).contains(&k) {
                    return Shape::Invalid;
                }
                resolved.push(base[k as usize]);
            }
            Some(n) if *n < 0 => return Shape::Invalid,
            other => resolved.push(*other),
        }
    }
    Shape::Tuple(resolved)
}

/// `repeat`: `repeat(*sizes)` — leading dims are new, covered dims
/// multiply (None-aware). A non-int element declines the whole read.
fn repeat_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let base = match shapes.first() {
        Some(Shape::Tuple(d)) => d.clone(),
        _ => return Shape::Unknown,
    };
    let rep = match attr_of(o, &["shape", "repeats"]) {
        None => return Shape::Tuple(base),
        Some(v) => match seq_items(v) {
            Some(items) => {
                if items.is_empty() {
                    return Shape::Tuple(base);
                }
                let mut out = Vec::with_capacity(items.len());
                for it in items.iter() {
                    match it {
                        AttrValue::Int(n) => out.push(Some(*n)),
                        AttrValue::UInt(n) => match i64::try_from(*n) {
                            Ok(n) => out.push(Some(n)),
                            Err(_) => return Shape::Unknown,
                        },
                        AttrValue::Null => out.push(None),
                        _ => return Shape::Unknown,
                    }
                }
                out
            }
            None => return Shape::Tuple(base),
        },
    };
    if rep.len() < base.len() {
        // torch requires len(reps) >= ndim — malformed; report base.
        return Shape::Tuple(base);
    }
    let pad = rep.len() - base.len();
    Shape::Tuple(
        (0..rep.len())
            .map(|i| {
                let b = if i < pad { Some(1) } else { base[i - pad] };
                match (b, rep[i]) {
                    (Some(b), Some(r)) if b >= 0 => Some(b.saturating_mul(r)),
                    _ => None,
                }
            })
            .collect(),
    )
}

/// `permute`: reorder axes by the `dim`/`dims`/`order` attr; absent
/// or malformed → reverse all axes (the `not isinstance(dims, …)`
/// arm); a non-int element declines (Python's `int(d)` could raise —
/// the honest-unknown choice).
fn permute_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let Some(Shape::Tuple(base)) = shapes.first() else {
        return Shape::Unknown;
    };
    if base.is_empty() {
        return Shape::Unknown;
    }
    let reversed = || Shape::Tuple(base.iter().rev().copied().collect());
    let Some(v) = attr_of(o, &["dim", "dims", "order"]) else {
        return reversed();
    };
    let Some(dims) = av_ints(v) else {
        return if seq_items(v).is_some() {
            Shape::Unknown
        } else {
            reversed()
        };
    };
    if dims.is_empty() {
        return reversed();
    }
    let n = base.len() as i64;
    Shape::Tuple(
        dims.iter()
            .map(|d| base[d.rem_euclid(n) as usize])
            .collect(),
    )
}

/// `movedim`: move `source` axes to `destination` positions.
fn movedim_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let Some(Shape::Tuple(base)) = shapes.first() else {
        return Shape::Unknown;
    };
    if base.is_empty() {
        return Shape::Unknown;
    }
    let (Some(src), Some(dst)) = (
        ax_tuple(attr_of(o, &["source"])),
        ax_tuple(attr_of(o, &["destination"])),
    ) else {
        return Shape::Tuple(base.clone());
    };
    if src.is_empty() || src.len() != dst.len() {
        return Shape::Tuple(base.clone());
    }
    let n = base.len() as i64;
    let src_n: Vec<i64> = src.iter().map(|s| s.rem_euclid(n)).collect();
    let mut order: Vec<usize> = (0..base.len())
        .filter(|i| !src_n.contains(&(*i as i64)))
        .collect();
    let mut pairs: Vec<(i64, i64)> = src_n
        .iter()
        .copied()
        .zip(dst.iter().map(|d| d.rem_euclid(n)))
        .collect();
    pairs.sort_by_key(|p| p.1);
    for (s, d) in pairs {
        order.insert(d as usize, s as usize);
    }
    Shape::Tuple(order.iter().map(|&i| base[i]).collect())
}

/// `stack`: all inputs share a shape; insert the arg count at `dim`.
fn stack_shape(o: &Op, args: &[Term], shapes: &[Shape]) -> Shape {
    let Some(Shape::Tuple(base)) = shapes.first() else {
        return Shape::Unknown;
    };
    let d = match attr_of(o, &["dim"]) {
        None => 0,
        Some(v) => match av_int(v) {
            Some(d) => d,
            None => return Shape::Unknown,
        },
    };
    let nd = d.rem_euclid(base.len() as i64 + 1) as usize;
    let mut out = base[..nd].to_vec();
    out.push(Some(args.len() as i64));
    out.extend_from_slice(&base[nd..]);
    Shape::Tuple(out)
}

/// `unbind`/`select`: drop the `dim` axis (`unbind` defaults -1,
/// `select` defaults 0). Rank-0 → unknown.
fn drop_axis_shape(o: &Op, shapes: &[Shape], default: i64) -> Shape {
    let Some(Shape::Tuple(base)) = shapes.first() else {
        return Shape::Unknown;
    };
    if base.is_empty() {
        return Shape::Unknown;
    }
    let d = match attr_of(o, &["dim"]) {
        None => default,
        Some(v) => match av_int(v) {
            Some(d) => d,
            None => return Shape::Unknown,
        },
    };
    let nd = d.rem_euclid(base.len() as i64) as usize;
    Shape::Tuple(
        base.iter()
            .enumerate()
            .filter(|(i, _)| *i != nd)
            .map(|(_, d)| *d)
            .collect(),
    )
}

/// `slice`: `aten.slice(t, dim, start, end, step)` — step bounds the
/// stride; a non-int extent leaves the dim unchanged.
fn slice_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let Some(Shape::Tuple(base)) = shapes.first() else {
        return Shape::Unknown;
    };
    if base.is_empty() {
        return Shape::Unknown;
    }
    let d = match attr_of(o, &["dim"]) {
        None => 0,
        Some(v) => match av_int(v) {
            Some(d) => d,
            None => return Shape::Unknown,
        },
    };
    // `attrs.get("start", 0) or 0` / `get("step", 1) or 1` — falsy
    // values collapse to the default; a truthy non-int declines.
    let int_or_default = |names: &[&str], default: i64| -> Option<i64> {
        match attr_of(o, names) {
            None => Some(default),
            Some(v) if !av_truthy(v) => Some(default),
            Some(v) => av_int(v),
        }
    };
    let (Some(lo), Some(step)) = (int_or_default(&["start"], 0), int_or_default(&["step"], 1))
    else {
        return Shape::Unknown;
    };
    // `end` has no default and no `or`: non-int reads as absent.
    let hi = attr_of(o, &["end"]).and_then(av_int);
    let mut out = base.clone();
    let nd = d.rem_euclid(base.len() as i64) as usize;
    if let Some(bd) = base[nd]
        && step != 0
    {
        let hi_eff = hi.map(|h| h.min(bd)).unwrap_or(bd);
        let n = hi_eff - lo;
        // ceil(n / step) clamped at 0 — Python `max(0, -(-n // step))`.
        let q = if step > 0 {
            n.div_euclid(step) + i64::from(n.rem_euclid(step) != 0)
        } else {
            -((-n).div_euclid(step))
        };
        out[nd] = Some(q.max(0));
    }
    Shape::Tuple(out)
}

/// `embedding`: row gather — `idx.shape + (d,)` where W is `(v, d)`.
fn embedding_shape(shapes: &[Shape]) -> Shape {
    if shapes.len() < 2 {
        return Shape::Unknown;
    }
    let Some(w) = shapes[0].as_tuple() else {
        return Shape::Unknown;
    };
    if w.len() != 2 {
        return Shape::Unknown;
    }
    let mut out = shapes[1]
        .as_tuple()
        .map(<[Dim]>::to_vec)
        .unwrap_or_default();
    out.push(w[1]);
    Shape::Tuple(out)
}

/// `index_select`: gather along `dim`; that axis resizes to
/// `len(index)`.
fn index_select_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let Some(Shape::Tuple(base)) = shapes.first() else {
        return Shape::Unknown;
    };
    if base.is_empty() {
        return Shape::Unknown;
    }
    let d = match attr_of(o, &["dim"]) {
        None => 0,
        Some(v) => match av_int(v) {
            Some(d) => d,
            None => return Shape::Unknown,
        },
    };
    let idx_len = attr_of(o, &["index"]).and_then(|v| seq_items(v).map(|a| a.len() as i64));
    let mut out = base.clone();
    out[d.rem_euclid(base.len() as i64) as usize] = idx_len;
    Shape::Tuple(out)
}

/// `unflatten`: splice the `sizes` tuple in at `dim`, `-1` folded
/// against `base[dim]`.
fn unflatten_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let Some(Shape::Tuple(base)) = shapes.first() else {
        return Shape::Unknown;
    };
    if base.is_empty() {
        return Shape::Unknown;
    }
    let Some(sizes) = attr_of(o, &["sizes"]).and_then(av_ints) else {
        return Shape::Tuple(base.clone());
    };
    if sizes.is_empty() {
        return Shape::Tuple(base.clone());
    }
    let d = match attr_of(o, &["dim"]) {
        None => 0,
        Some(v) => match av_int(v) {
            Some(d) => d,
            None => return Shape::Unknown,
        },
    };
    let nd = d.rem_euclid(base.len() as i64) as usize;
    let mut sz: Vec<i64> = sizes;
    if sz.contains(&-1)
        && let Some(bd) = base[nd]
    {
        let mut known: i64 = 1;
        for s in &sz {
            if *s != -1 {
                known = known.saturating_mul(if *s > 0 { *s } else { 1 });
            }
        }
        let inf = if known != 0 && bd % known == 0 {
            Some(bd / known)
        } else {
            None
        };
        for s in &mut sz {
            if *s == -1 {
                *s = inf.unwrap_or(-1);
            }
        }
    }
    let mut out: Vec<Dim> = base[..nd].to_vec();
    out.extend(sz.iter().map(|s| if *s < 0 { None } else { Some(*s) }));
    out.extend_from_slice(&base[nd + 1..]);
    Shape::Tuple(out)
}

/// `narrow`: `narrow(t, dim, start, length)` — a length-bounded slice.
fn narrow_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let Some(Shape::Tuple(base)) = shapes.first() else {
        return Shape::Unknown;
    };
    if base.is_empty() {
        return Shape::Unknown;
    }
    let d = match attr_of(o, &["dim"]) {
        None => 0,
        Some(v) => match av_int(v) {
            Some(d) => d,
            None => return Shape::Unknown,
        },
    };
    let mut out = base.clone();
    out[d.rem_euclid(base.len() as i64) as usize] = attr_of(o, &["length"]).and_then(av_int);
    Shape::Tuple(out)
}

/// `pad`: `F.pad` constant — trailing dims get `+left/+right`.
/// Non-int pad elements read as unknown extents on the touched axes.
fn pad_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let Some(Shape::Tuple(base)) = shapes.first() else {
        return Shape::Unknown;
    };
    if base.is_empty() {
        return Shape::Unknown;
    }
    let Some(pads) = attr_of(o, &["pad"]).and_then(seq_items) else {
        return Shape::Tuple(base.clone());
    };
    if pads.is_empty() {
        return Shape::Tuple(base.clone());
    }
    let pad_int = |v: &AttrValue| av_int(v);
    let mut out = base.clone();
    for i in 0..pads.len() / 2 {
        let Some(ax) = out.len().checked_sub(1 + i) else {
            break;
        };
        let (lo, hi) = (pad_int(&pads[2 * i]), pad_int(&pads[2 * i + 1]));
        out[ax] = match (out[ax], lo, hi) {
            (Some(d), Some(lo), Some(hi)) => Some(d.saturating_add(lo.saturating_add(hi))),
            _ => None,
        };
    }
    Shape::Tuple(out)
}

/// `glu`: halve the `dim` axis (default -1).
fn glu_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let Some(Shape::Tuple(base)) = shapes.first() else {
        return Shape::Unknown;
    };
    if base.is_empty() {
        return Shape::Unknown;
    }
    let d = match attr_of(o, &["dim"]) {
        None => -1,
        Some(v) => match av_int(v) {
            Some(d) => d,
            None => return Shape::Unknown,
        },
    };
    let mut out = base.clone();
    let nd = d.rem_euclid(base.len() as i64) as usize;
    out[nd] = out[nd].map(|d| d / 2);
    Shape::Tuple(out)
}

/// `addmm`/`addbmm`/`baddbmm`/`addmv`: bias + a@b (addmv: bias +
/// a@v) — the GEMM part shapes like the simple matmul read on args
/// 1/2, then the bias broadcasts.
fn addmm_shape(shapes: &[Shape]) -> Shape {
    let (Some(bias), Some(a), Some(b)) = (shapes.first(), shapes.get(1), shapes.get(2)) else {
        return Shape::Unknown;
    };
    let mm = match (a.as_tuple(), b.as_tuple()) {
        (Some(a), Some(b)) => mm_simple(a, b),
        _ => None,
    };
    let Some(mm) = mm else {
        return non_scalar(bias.clone());
    };
    broadcast(&Shape::Tuple(mm), bias)
}

/// `conv1d`/`conv2d`: `(N,C,L…) @ (O,C,k…) -> (N,O,L'…)` with the
/// `_conv_window` extent rule; malformed operands degrade to the
/// `(x0, w0, None…)` partial shape exactly like Python.
fn conv_shape(o: &Op, shapes: &[Shape], spatial: usize) -> Shape {
    let need = spatial + 2;
    let x = shapes.first().cloned().unwrap_or(Shape::Unknown);
    let w = shapes.get(1).cloned().unwrap_or(Shape::Unknown);
    let malformed = match (x.as_tuple(), w.as_tuple()) {
        (Some(xt), Some(wt)) => {
            xt.len() < need
                || wt.len() < need
                || matches!(attr_of(o, &["padding"]), Some(AttrValue::Str(_)))
        }
        _ => true,
    };
    if malformed {
        // Partial honest shape: (x0, w0, None…) when both operand
        // ranks admit, else operand 0's shape (Python's fallback).
        if let (Some(xt), Some(wt)) = (x.as_tuple(), w.as_tuple())
            && !xt.is_empty()
            && !wt.is_empty()
        {
            let mut out = vec![xt[0], wt[0]];
            out.resize(need, None);
            return Shape::Tuple(out);
        }
        return non_scalar(x);
    }
    let xt = x.as_tuple().unwrap();
    let wt = w.as_tuple().unwrap();
    // conv1d attrs are scalars (`_conv_scalar`), conv2d pairs
    // (`_conv_pair`) — collect per-axis window attr values.
    let window_attrs = |name: &str| -> Vec<Option<AttrValue>> {
        if spatial == 1 {
            vec![conv_scalar(attr_of(o, &[name]))]
        } else {
            let (a, b) = conv_pair(attr_of(o, &[name]));
            vec![a, b]
        }
    };
    let st = window_attrs("stride");
    let pd = window_attrs("padding");
    let dl = window_attrs("dilation");
    let mut out = vec![xt[0], wt[0]];
    for i in 0..spatial {
        match conv_window(
            xt[2 + i],
            wt[2 + i],
            st[i].as_ref(),
            pd[i].as_ref(),
            dl[i].as_ref(),
        ) {
            ConvWindow::Extent(e) => out.push(Some(e)),
            ConvWindow::Invalid => return Shape::Invalid,
            ConvWindow::Unknown => out.push(None),
        }
    }
    Shape::Tuple(out)
}

/// `tensor_split`: the folded `split` — `sections` is the count (or
/// index list), `index` the folded section.
fn tensor_split_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let Some(Shape::Tuple(base)) = shapes.first() else {
        return Shape::Unknown;
    };
    if base.is_empty() {
        return Shape::Unknown;
    }
    let d = match attr_of(o, &["dim"]) {
        None => 0,
        Some(v) => match av_int(v) {
            Some(d) => d,
            None => return Shape::Unknown,
        },
    };
    let idx = match attr_of(o, &["index"]) {
        None => 0,
        Some(v) if !av_truthy(v) => 0,
        Some(v) => match av_int(v) {
            Some(i) => i,
            None => return Shape::Unknown,
        },
    };
    let nd = d.rem_euclid(base.len() as i64) as usize;
    let mut out = base.clone();
    match attr_of(o, &["sections"]) {
        Some(v) => match av_int(v) {
            Some(n) => {
                if n <= 0 {
                    return Shape::Invalid; // sections must be positive
                }
                out[nd] = base[nd].map(|b| (b + n - 1) / n);
            }
            None => match seq_items(v) {
                Some(bounds) => {
                    let ints: Option<Vec<i64>> = bounds.iter().map(av_int).collect();
                    out[nd] = match (ints, base[nd]) {
                        (Some(v), Some(bd)) => {
                            let mut b = vec![0];
                            b.extend(v);
                            b.push(bd);
                            if (idx as usize) + 1 < b.len() {
                                Some(b[idx as usize + 1] - b[idx as usize])
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };
                }
                _ => out[nd] = None,
            },
        },
        None => out[nd] = None,
    }
    Shape::Tuple(out)
}

/// `unfold`: `unfold(dim, size, step)` — dim shrinks to sliding
/// windows and a trailing axis of `size` appears.
fn unfold_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let Some(Shape::Tuple(base)) = shapes.first() else {
        return Shape::Unknown;
    };
    if base.is_empty() {
        return Shape::Unknown;
    }
    let axis_int = |names: &[&str], default: i64| -> Option<i64> {
        match attr_of(o, names) {
            None => Some(default),
            Some(v) if !av_truthy(v) => Some(default),
            Some(v) => av_int(v),
        }
    };
    // `size` is read raw (`attr_of(op, "size")` — non-int → the dim
    // stays unknown rather than declining the whole op, matching the
    // `isinstance(size, int)` gates below).
    let size = match attr_of(o, &["size"]) {
        None => None,
        Some(v) => av_int(v),
    };
    let (Some(d), Some(step)) = (axis_int(&["dim"], 0), axis_int(&["step"], 1)) else {
        return Shape::Unknown;
    };
    let nd = d.rem_euclid(base.len() as i64) as usize;
    let mut out = base.clone();
    out[nd] = match (base[nd], size) {
        (Some(n), Some(s)) if s > 0 && step != 0 => Some(((n - s) / step + 1).max(0)),
        _ => None,
    };
    out.push(size);
    Shape::Tuple(out)
}

/// `pixel_shuffle`/`pixel_unshuffle`: the `(N,C·r²,H,W)` space↔depth
/// trade, `r` from `upscale_factor`/`downscale_factor`.
fn pixel_shuffle_shape(o: &Op, shapes: &[Shape], factor_attr: &str, shuffle: bool) -> Shape {
    let Some(Shape::Tuple(base)) = shapes.first() else {
        return Shape::Tuple(Vec::new());
    };
    let r = match attr_of(o, &[factor_attr]) {
        None => 1,
        Some(v) => match av_int(v) {
            Some(r) => r,
            None => return Shape::Tuple(base.clone()),
        },
    };
    if base.len() < 4 {
        return Shape::Tuple(base.clone());
    }
    let c = base[1];
    if shuffle {
        Shape::Tuple(vec![
            base[0],
            c.map(|c| c / (r * r)),
            base[2].map(|d| d * r),
            base[3].map(|d| d * r),
        ])
    } else {
        Shape::Tuple(vec![
            base[0],
            c.map(|c| c * r * r),
            base[2].map(|d| d / r),
            base[3].map(|d| d / r),
        ])
    }
}

/// `hstack`/`vstack`: cat along the stacked axis.
fn hstack_vstack_shape(name: &str, args: &[Term], shapes: &[Shape]) -> Shape {
    let Some(a) = shapes.first().and_then(Shape::as_tuple) else {
        return Shape::Unknown;
    };
    if a.is_empty() {
        return Shape::Unknown;
    }
    let dim = if name == "hstack" {
        if a.len() == 1 { 0 } else { 1 }
    } else {
        // vstack promotes 1-D rows to (1, d) then cats dim 0.
        if a.len() < 2 {
            return Shape::Tuple(vec![Some(args.len() as i64), a[0]]);
        }
        0
    };
    let mut out = a.to_vec();
    out[dim] = Some(0);
    for s in shapes {
        let Some(st) = s.as_tuple() else {
            return Shape::Tuple(a.to_vec());
        };
        if st.len() != a.len() {
            return Shape::Tuple(a.to_vec());
        }
        out[dim] = Some(out[dim].unwrap_or(0).saturating_add(st[dim].unwrap_or(0)));
    }
    Shape::Tuple(out)
}

/// `flatten`: merge `start_dim..=end_dim` into their numel.
fn flatten_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let Some(Shape::Tuple(base)) = shapes.first() else {
        return Shape::Unknown;
    };
    if base.is_empty() {
        return Shape::Unknown;
    }
    let axis_int = |names: &[&str], default: i64| -> Option<i64> {
        match attr_of(o, names) {
            None => Some(default),
            Some(v) if !av_truthy(v) => Some(default),
            Some(v) => av_int(v),
        }
    };
    let (Some(d0), Some(d1)) = (axis_int(&["start_dim"], 0), axis_int(&["end_dim"], -1)) else {
        return Shape::Unknown;
    };
    let n = base.len() as i64;
    let (d0, d1) = (d0.rem_euclid(n) as usize, d1.rem_euclid(n) as usize);
    // `_numel` of the merged range — None dims count 1 (the Python
    // convention verbatim, warts and all).
    let merged: i64 = base[d0..=d1]
        .iter()
        .fold(1i64, |acc, d| acc.saturating_mul(d.unwrap_or(1).max(0)));
    let mut out: Vec<Dim> = base[..d0].to_vec();
    out.push(Some(merged));
    out.extend_from_slice(&base[d1 + 1..]);
    Shape::Tuple(out)
}

/// `concat`: variadic cat summing every operand along the cat axis;
/// provably-empty (`0`-dim) operands don't set rank or the axis.
fn concat_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let live: Vec<&Shape> = shapes.iter().filter(|s| !cat_empty(s)).collect();
    let pool: Vec<&Shape> = if live.is_empty() {
        shapes.iter().collect()
    } else {
        live
    };
    let Some(a) = pool.first().and_then(|s| s.as_tuple()) else {
        // `()` has no cat axis — carrier-internal operand.
        return Shape::Unknown;
    };
    if a.is_empty() {
        return Shape::Unknown;
    }
    let d = match attr_of(o, &["dim"]) {
        None => 0,
        Some(v) => match av_int(v) {
            Some(d) => d,
            None => return Shape::Unknown,
        },
    };
    let nd = d.rem_euclid(a.len() as i64) as usize;
    let mut out = a.to_vec();
    out[nd] = Some(0);
    for s in &pool {
        let Some(st) = s.as_tuple() else {
            return Shape::Tuple(a.to_vec());
        };
        if st.len() != a.len() {
            return Shape::Tuple(a.to_vec());
        }
        out[nd] = Some(out[nd].unwrap_or(0).saturating_add(st[nd].unwrap_or(0)));
    }
    Shape::Tuple(out)
}

/// `chunk`: `s[dim] // chunks` — a non-positive chunk count is
/// `Invalid`, a metavar/malformed one `Unknown`.
fn chunk_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let Some(Shape::Tuple(base)) = shapes.first() else {
        return Shape::Unknown;
    };
    if base.is_empty() {
        return Shape::Unknown;
    }
    let d = match attr_of(o, &["dim"]) {
        None => -1,
        Some(v) => match av_int(v) {
            Some(d) => d,
            None => return Shape::Unknown,
        },
    };
    let n = match attr_of(o, &["chunks"]) {
        None => 2,
        Some(v) => match av_int(v) {
            Some(n) => n,
            None => return Shape::Unknown,
        },
    };
    if n <= 0 {
        return Shape::Invalid;
    }
    let nd = d.rem_euclid(base.len() as i64) as usize;
    let mut out = base.clone();
    // `(base[dim] or 0) // n` — a None extent reads 0 → 0.
    out[nd] = Some(base[nd].unwrap_or(0) / n);
    Shape::Tuple(out)
}

/// `split`: `sizes` is the per-section size (or list); `index` the
/// folded section (`or 0` — falsy collapses to 0).
fn split_shape(o: &Op, shapes: &[Shape]) -> Shape {
    let Some(Shape::Tuple(base)) = shapes.first() else {
        return Shape::Unknown;
    };
    if base.is_empty() {
        return Shape::Unknown;
    }
    let d = match attr_of(o, &["dim"]) {
        None => -1,
        Some(v) => match av_int(v) {
            Some(d) => d,
            None => return Shape::Unknown,
        },
    };
    let idx = match attr_of(o, &["index"]) {
        None => 0,
        Some(v) if !av_truthy(v) => 0,
        Some(v) => match av_int(v) {
            Some(i) => i,
            None => return Shape::Unknown,
        },
    };
    let nd = d.rem_euclid(base.len() as i64) as usize;
    let mut out = base.clone();
    out[nd] = match attr_of(o, &["sizes", "split_size"]) {
        Some(v) => match seq_items(v) {
            Some(a) => {
                let ints: Option<Vec<i64>> = a.iter().map(av_int).collect();
                match ints {
                    Some(v) if (idx as usize) < v.len() => Some(v[idx as usize]),
                    _ => None,
                }
            }
            None => av_int(v),
        },
        None => None,
    };
    Shape::Tuple(out)
}

// ---------------------------------------------------------------------------
//  Op-name tables (the `_BROADCAST_BINOPS` / `_ARG0_SHAPE_OPS` sets,
//  verbatim from `mlopt_core.typing`)
// ---------------------------------------------------------------------------

/// `_BROADCAST_BINOPS` — output is the broadcast of EVERY operand.
const BROADCAST_BINOPS: &[&str] = &[
    "add",
    "mul",
    "sub",
    "div",
    "pow",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    "fmod",
    "remainder",
    "xlogy",
    "atan2",
    "heaviside",
    "isclose",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "logical_and",
    "logical_or",
    "logical_xor",
    "bitwise_and",
    "bitwise_or",
];

/// `_ARG0_SHAPE_OPS` — result shape IS operand 0's shape; the only
/// ops the partial-unknown early return may answer `shapes[0]` for.
const ARG0_SHAPE_OPS: &[&str] = &[
    "square",
    "sqrt",
    "neg",
    "sigmoid",
    "silu",
    "tanh",
    "gelu",
    "rsqrt",
    "exp",
    "exp2",
    "expm1",
    "softmax",
    "log_softmax",
    "masked_fill",
    "logical_not",
    "relu",
    "leaky_relu",
    "elu",
    "celu",
    "softplus",
    "softsign",
    "hardswish",
    "hardsigmoid",
    "mish",
    "relu6",
    "prelu",
    "hardtanh",
    "abs",
    "sign",
    "floor",
    "ceil",
    "round",
    "frac",
    "trunc",
    "log",
    "log2",
    "log10",
    "log1p",
    "erf",
    "erfc",
    "erfinv",
    "gammaln",
    "digamma",
    "i0",
    "sinc",
    "reciprocal",
    "nan_to_num",
    "isnan",
    "isinf",
    "isfinite",
    "isposinf",
    "isneginf",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "asinh",
    "acosh",
    "atanh",
    "detach",
    "detach_",
    "copy",
    "cumsum",
    "cumprod",
    "logcumsumexp",
    "cummax",
    "cummin",
    "argsort",
    "flip",
    "roll",
    "triu",
    "tril",
    "scatter",
    "scatter_add",
    "scatter_reduce",
    "index_add",
    "index_put",
    "slice_scatter",
    "select_scatter",
    "batch_norm",
    "group_norm",
    "instance_norm",
    "sort",
    "topk",
    "median",
    "kthvalue",
    "mode",
    "_assert_tensor_metadata",
    "contiguous",
    "to",
    "type_as",
    "float",
    "dropout",
    "alias",
    "inv",
    "getitem",
    "sdpa",
];

/// The shape-preserving unary family (`return shapes[0]` arms).
const UNARY_PASSTHROUGH: &[&str] = &[
    "square",
    "sqrt",
    "neg",
    "sigmoid",
    "silu",
    "tanh",
    "gelu",
    "rsqrt",
    "exp",
    "exp2",
    "expm1",
    "softmax",
    "log_softmax",
    "masked_fill",
    "logical_not",
    "relu",
    "leaky_relu",
    "elu",
    "celu",
    "softplus",
    "softsign",
    "hardswish",
    "hardsigmoid",
    "mish",
    "relu6",
    "prelu",
    "hardtanh",
    "abs",
    "sign",
    "floor",
    "ceil",
    "round",
    "frac",
    "trunc",
    "log",
    "log2",
    "log10",
    "log1p",
    "erf",
    "erfc",
    "erfinv",
    "gammaln",
    "digamma",
    "i0",
    "sinc",
    "reciprocal",
    "nan_to_num",
    "isnan",
    "isinf",
    "isfinite",
    "isposinf",
    "isneginf",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "asinh",
    "acosh",
    "atanh",
    "detach",
    "detach_",
    "copy",
];

/// Axis-anchored rank-preserving ops (`shapes[0] or None` arms).
const AXIS_ANCHORED: &[&str] = &[
    "cumsum",
    "cumprod",
    "logcumsumexp",
    "cummax",
    "cummin",
    "argsort",
    "flip",
    "roll",
    "triu",
    "tril",
    "scatter",
    "scatter_add",
    "scatter_reduce",
    "index_add",
    "index_put",
    "slice_scatter",
    "select_scatter",
    "batch_norm",
    "group_norm",
    "instance_norm",
    "sort",
    "topk",
    "median",
    "kthvalue",
    "mode",
    "_assert_tensor_metadata",
];

/// The reduction family (`sum`/`mean`/… `keepdim`/`dim` arms).
const REDUCE_OPS: &[&str] = &[
    "sum",
    "mean",
    "prod",
    "max",
    "min",
    "amax",
    "amin",
    "var_mean",
    "std_mean",
    "argmax",
    "argmin",
    "var",
    "std",
    "any",
    "all",
    "nansum",
    "nanmean",
    "count_nonzero",
    "linalg_vector_norm",
];

// ---------------------------------------------------------------------------
//  Leaves
// ---------------------------------------------------------------------------

/// A leaf's [`TensorInfo`]: decode the self-describing key — a
/// serialized kernel `Term` round-trips through `to_kernel`; anything
/// else is inspected as raw JSON (`{"__tensor__": …}` wire values and
/// `{"shape": […]}` payloads both read).
fn leaf_info(key: &str) -> TensorInfo {
    if let Ok(kt) = serde_json::from_str::<KernelTerm>(key) {
        return kernel_leaf_info(&kt);
    }
    if let Ok(v) = serde_json::from_str::<Value>(key) {
        return json_leaf_info(&v);
    }
    TensorInfo {
        shape: Shape::Unknown,
        dtype: None,
    }
}

/// The mlopt-search bridge leaf convention —
/// `mlopt_orchestrator.eqsat._LEAF_PREFIX`: a `Const` *string*
/// `__mlopt_leaf__:{"k":…,"n":…,"s":…}` carrying a typed `Var`/`Param`
/// leaf, where `s` is the declared `TensorType` shape (`null` = the
/// shape is wholly unknown — never a scalar).  The kernel `Term` has
/// no variable leaf, so the Python bridge encodes leaves this way;
/// decoding the payload here is what keeps shape guards honest on
/// bridged terms.  Without the read every bridge leaf fell through to
/// the scalar-`()` `Const` default, and broadcast guards
/// (`bcast-into`/`bcast-eq`/`unsq-out`) treated phantom scalars as
/// broadcast-compatible — the WP4 `mul_unsq_nat` parity failure: the
/// pad laws fired on real `(4,6)` operands and extraction produced
/// `mul(a, unsqueeze(b, -1))`, a term that cannot materialise
/// (`4` vs `6` at dim 1).
pub(crate) const MLOPT_LEAF_PREFIX: &str = "__mlopt_leaf__:";

/// Whether `v` is a `__mlopt_leaf__:` bridge-leaf string — the
/// `Var`/`Param` encoding. These are *leaves*, not `Const`s:
/// cond.py's `isinstance(t, Const)` is `False` for `Var`/`Param`, so
/// the guard layer's const reads ([`crate::guard`]) must decline on
/// them even though they arrive as `Const`-shaped wire leaves. The
/// prefix alone classifies — a matched-but-malformed payload is
/// still a leaf (honest decline, never a guess).
pub(crate) fn is_bridge_leaf(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.starts_with(MLOPT_LEAF_PREFIX))
}

/// Decode a `__mlopt_leaf__:` payload's declared shape.  `Some` iff
/// `v` is a string carrying the prefix — a matched-but-malformed
/// payload reads [`Shape::Unknown`] (honest decline), and `s: null`/
/// absent reads [`Shape::Unknown`] too: a shape-less `Var`/`Param` is
/// an unknown-rank tensor, *not* a scalar.
fn mlopt_leaf_info(v: &Value) -> Option<TensorInfo> {
    let s = v.as_str()?;
    let payload = s.strip_prefix(MLOPT_LEAF_PREFIX)?;
    let shape = serde_json::from_str::<Value>(payload)
        .ok()
        .and_then(|rec| rec.get("s").cloned())
        .and_then(|s| match s {
            Value::Null => None,
            other => json_dims(&other),
        });
    Some(TensorInfo {
        shape: shape.map_or(Shape::Unknown, Shape::Tuple),
        dtype: None,
    })
}

/// The leaf info of a decoded kernel term: `Const` reports `()`
/// (a `__tensor__` wire value reports its declared shape, and a
/// `__mlopt_leaf__:` bridge leaf its `TensorType` shape); `Opaque`
/// reads `shape`/`__tensor__` payloads; other variants recurse through
/// the bridge (they cannot loop — `Const`/`Opaque` are the only
/// leaf-producing variants).
fn kernel_leaf_info(kt: &KernelTerm) -> TensorInfo {
    match kt {
        KernelTerm::Const { value } => {
            if let Some(info) = tensor_wire_info(value) {
                return info;
            }
            if let Some(info) = mlopt_leaf_info(value) {
                return info;
            }
            TensorInfo {
                shape: Shape::Tuple(Vec::new()),
                dtype: None,
            }
        }
        KernelTerm::Opaque { data } => json_leaf_info(data),
        other => infer(&from_kernel(other)),
    }
}

/// The leaf info of a raw JSON payload: `__tensor__` wire values,
/// `__mlopt_leaf__:` bridge-leaf strings, and bare `{"shape": […]}`
/// records all read; `dtype` rides along.
fn json_leaf_info(v: &Value) -> TensorInfo {
    if let Some(info) = tensor_wire_info(v) {
        return info;
    }
    if let Some(info) = mlopt_leaf_info(v) {
        return info;
    }
    if let Value::Object(map) = v
        && let Some(shape) = map.get("shape").and_then(json_dims)
    {
        let dtype = map.get("dtype").map(AttrValue::from);
        return TensorInfo {
            shape: Shape::Tuple(shape),
            dtype,
        };
    }
    TensorInfo {
        shape: Shape::Unknown,
        dtype: None,
    }
}

/// The `{"__tensor__": {"shape": […], "dtype": …}}` wire convention
/// (host-tensor's tensor literal encoding).
fn tensor_wire_info(v: &Value) -> Option<TensorInfo> {
    let wire = v.get("__tensor__")?;
    let shape = wire.get("shape").and_then(json_dims)?;
    let dtype = wire.get("dtype").map(AttrValue::from);
    Some(TensorInfo {
        shape: Shape::Tuple(shape),
        dtype,
    })
}

/// The leaf's *declared* shape as concrete dims — the
/// `("leaf-dim", T, i)` dspec read (`term.typ.shape` in `cond.py`):
/// only a [`Term::Leaf`] — or the canonical `arr` enode spelling of
/// one — whose payload declares a non-empty, all-int shape answers;
/// `Op` bindings, shape-less leaves and `None`-dim declarations all
/// decline. This is the *declared* read — never the inferred op
/// shape — matching `_d_leaf_dim`'s strictness (a domain `Op` has no
/// `.typ` and declines there too; the `arr` enode is not a domain
/// op, it is the leaf's canonical spelling).
pub(crate) fn leaf_declared_dims(t: &Term) -> Option<Vec<i64>> {
    let shape = match t {
        Term::Leaf(key) => leaf_info(key).shape,
        Term::Op(o) => json_leaf_info(&arr_leaf_value(o)?).shape,
        Term::Var(_) => return None,
    };
    match shape {
        Shape::Tuple(dims) => dims
            .into_iter()
            .collect::<Option<Vec<i64>>>()
            .filter(|d| !d.is_empty()),
        _ => None,
    }
}

/// A JSON array of ints/nulls → dims (non-int elements → unknown
/// extents, never a guess).
fn json_dims(v: &Value) -> Option<Vec<Dim>> {
    let Value::Array(items) = v else {
        return None;
    };
    Some(
        items
            .iter()
            .map(|d| match d {
                Value::Number(n) => n
                    .as_i64()
                    .or_else(|| n.as_u64().and_then(|u| i64::try_from(u).ok())),
                Value::Null => None,
                _ => None,
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
//  Shape specs — the `_SPEC_OPS` half of `cond.py`, over `BoundEnv`
// ---------------------------------------------------------------------------

/// An attribute-position argument in a [`ShapeSpec`] or [`pred`]
/// predicate: a bound attr metavariable name or a literal value.
///
/// `Name` resolves through [`BoundEnv::attrs`] under the **bare**
/// metavar name (the `$attr:` prefix is record spelling only).
/// `Lit` carries the literal — `AttrValue::Int` for axis positions,
/// `AttrValue::Array` for tuple positions (a reshape target, an index
/// list). Python's third spelling — an inline derive *expr* — is
/// stage-2's `dspec` concern and declines here as unresolvable.
#[derive(Clone, Debug, PartialEq)]
pub enum AttrArg {
    /// `$attr:NAME` — looked up in `BoundEnv::attrs`.
    Name(String),
    /// A literal attr value.
    Lit(AttrValue),
}

impl AttrArg {
    /// Convenience: a literal int axis.
    pub fn int(n: i64) -> AttrArg {
        AttrArg::Lit(AttrValue::Int(n))
    }

    /// Convenience: a named attr metavar.
    pub fn named(name: impl Into<String>) -> AttrArg {
        AttrArg::Name(name.into())
    }

    /// Resolve against the bound environment: `Name` → the bound
    /// `AttrValue` (`None` when unbound — the `_MISSING` case), `Lit`
    /// → itself.
    pub fn resolve(&self, env: &BoundEnv) -> Option<AttrValue> {
        match self {
            AttrArg::Name(name) => env.attrs.get(name.as_str()).cloned(),
            AttrArg::Lit(v) => Some(v.clone()),
        }
    }
}

/// A shape-spec argument — the `_SPEC_OPS` grammar of `cond.py`:
/// a bound-term metavar or a computed spec. Specs compose.
#[derive(Clone, Debug, PartialEq)]
pub enum ShapeSpec {
    /// A position that resolves to [`Shape::Unknown`] — the
    /// `_shape`/`_tshape` decline for a non-metavar, non-spec ref.
    Unknown,
    /// A bound term metavariable — its inferred shape.
    Var(String),
    /// `("mm-out", a, b)` — `_matmul_shape` of two specs.
    MmOut(Box<ShapeSpec>, Box<ShapeSpec>),
    /// `("bcast", a, b)` — broadcast of two specs (non-tuple reads
    /// wildcard, `_s_bcast` verbatim).
    Bcast(Box<ShapeSpec>, Box<ShapeSpec>),
    /// `("unsq-out", t, K)` — insert a 1 at `K mod (rank+1)`.
    UnsqOut(Box<ShapeSpec>, AttrArg),
    /// `("reshape-out", t, NAME)` — resolved reshape target under the
    /// bound attr (`-1` folded, numel checked — mismatch declines).
    ReshapeOut(Box<ShapeSpec>, AttrArg),
    /// `("getitem-out", t)` — tensor-index output: `s[1:]`.
    GetitemOut(Box<ShapeSpec>),
    /// `("select-out", t, D)` — drop axis `D mod rank`.
    SelectOut(Box<ShapeSpec>, AttrArg),
    /// `("slice-out", t, D, S, E, STEP)` — axis `D` replaced by the
    /// sliced extent; `None` slots take the torch defaults
    /// (`0` / the dim / `1`).
    SliceOut(
        Box<ShapeSpec>,
        AttrArg,
        Option<AttrArg>,
        Option<AttrArg>,
        Option<AttrArg>,
    ),
    /// `("chunk-out", t, C, D)` — `s[D] // C`.
    ChunkOut(Box<ShapeSpec>, AttrArg, AttrArg),
    /// `("transpose-out", t, D0, D1)` — swap axes `D0`/`D1`
    /// (defaults -2/-1).
    TransposeOut(Box<ShapeSpec>, AttrArg, AttrArg),
    /// `("tail-block", t, NAME)` — `s[-k:]` iff the bound reduce dims
    /// name exactly T's last `k` axes.
    TailBlock(Box<ShapeSpec>, String),
    /// `("isel-out", t, D, I)` — `index_select` output: axis `D`
    /// replaced by `len(I)`.
    IselOut(Box<ShapeSpec>, AttrArg, AttrArg),
    /// `("vshape", t)` — the value-shape read. There is no registered
    /// carrier resolver on this side, so it resolves exactly as
    /// [`ShapeSpec::Var`] — the cost convention (documented, not a
    /// silent weakening: `vshape` exists for carriers that never
    /// reach this engine).
    Vshape(Box<ShapeSpec>),
}

/// Resolve a [`ShapeSpec`] against the bound environment — `_shape`.
/// Anything unresolvable is [`Shape::Unknown`] (the decline).
pub fn spec_shape(env: &BoundEnv, spec: &ShapeSpec) -> Shape {
    let mut memo = Memo::new();
    spec_shape_with(env, spec, &mut memo)
}

/// [`spec_shape`] with a shared memo (predicate batches reuse one).
pub fn spec_shape_with(env: &BoundEnv, spec: &ShapeSpec, memo: &mut Memo) -> Shape {
    /// `_tshape`: tuple-or-unknown.
    fn tshape(s: Shape) -> Option<Vec<Dim>> {
        s.as_tuple().map(<[Dim]>::to_vec)
    }
    /// `_aval` + `isinstance(v, int)`.
    fn aval_int(env: &BoundEnv, a: &AttrArg) -> Option<i64> {
        a.resolve(env).as_ref().and_then(av_int)
    }
    /// `_bound_attr`: unbound/absent/Null → default.
    fn bound_int(env: &BoundEnv, a: &AttrArg, default: i64) -> Option<i64> {
        match a.resolve(env) {
            None | Some(AttrValue::Null) => Some(default),
            Some(v) => av_int(&v),
        }
    }
    match spec {
        ShapeSpec::Unknown => Shape::Unknown,
        ShapeSpec::Var(name) => env
            .terms
            .get(name.as_str())
            .map(|t| shape_of_with(t, memo))
            .unwrap_or(Shape::Unknown),
        ShapeSpec::Vshape(inner) => spec_shape_with(env, inner, memo),
        ShapeSpec::MmOut(a, b) => {
            let sa = spec_shape_with(env, a, memo);
            let sb = spec_shape_with(env, b, memo);
            matmul_shape(&sa, &sb)
                .map(Shape::Tuple)
                .unwrap_or(Shape::Unknown)
        }
        ShapeSpec::Bcast(a, b) => {
            let sa = spec_shape_with(env, a, memo);
            let sb = spec_shape_with(env, b, memo);
            // `_s_bcast`: non-tuple operands read as None (wildcard).
            let ta = tshape(sa).map(Shape::Tuple).unwrap_or(Shape::Unknown);
            let tb = tshape(sb).map(Shape::Tuple).unwrap_or(Shape::Unknown);
            broadcast(&ta, &tb)
        }
        ShapeSpec::UnsqOut(t, k) => {
            let s = spec_shape_with(env, t, memo);
            let Some(base) = tshape(s) else {
                return Shape::Unknown;
            };
            let Some(k) = aval_int(env, k) else {
                return Shape::Unknown;
            };
            let nd = k.rem_euclid(base.len() as i64 + 1) as usize;
            let mut out = base[..nd].to_vec();
            out.push(Some(1));
            out.extend_from_slice(&base[nd..]);
            Shape::Tuple(out)
        }
        ShapeSpec::ReshapeOut(t, name) => {
            let s = spec_shape_with(env, t, memo);
            let Some(base) = tshape(s.clone()) else {
                return Shape::Unknown;
            };
            let Some(v) = name.resolve(env) else {
                return Shape::Unknown;
            };
            // The target is a shape attr value (tuple of ints/-1).
            let target = seq_items(&v).and_then(|items| {
                items
                    .iter()
                    .map(|d| match d {
                        AttrValue::Int(n) => Some(Some(*n)),
                        AttrValue::UInt(n) => i64::try_from(*n).ok().map(Some),
                        AttrValue::Bool(b) => Some(Some(i64::from(*b))),
                        _ => Some(None),
                    })
                    .collect::<Option<Vec<Dim>>>()
            });
            let Some(target) = target else {
                return Shape::Unknown;
            };
            // `_reshape_shape`: fold -1 via numel; numel-mismatch →
            // decline (None — the guard contract, not INVALID). The
            // numel check is gated on `_nonneg` (every dim a
            // non-negative int) exactly like Python — a `None` or
            // negative dim skips the check and returns raw.
            let folded = fold_minus1(&target, numel(&s));
            let nonneg = |d: &[Dim]| d.iter().all(|x| x.is_some_and(|n| n >= 0));
            if nonneg(&folded) && nonneg(&base) && numel(&Shape::Tuple(folded.clone())) != numel(&s)
            {
                return Shape::Unknown;
            }
            Shape::Tuple(folded)
        }
        ShapeSpec::GetitemOut(t) => {
            let s = spec_shape_with(env, t, memo);
            match tshape(s) {
                Some(d) if !d.is_empty() => Shape::Tuple(d[1..].to_vec()),
                _ => Shape::Unknown,
            }
        }
        ShapeSpec::SelectOut(t, d) => {
            let s = spec_shape_with(env, t, memo);
            let Some(base) = tshape(s) else {
                return Shape::Unknown;
            };
            if base.is_empty() {
                return Shape::Unknown;
            }
            let Some(d) = aval_int(env, d) else {
                return Shape::Unknown;
            };
            let nd = d.rem_euclid(base.len() as i64) as usize;
            Shape::Tuple(
                base.iter()
                    .enumerate()
                    .filter(|(i, _)| *i != nd)
                    .map(|(_, d)| *d)
                    .collect(),
            )
        }
        ShapeSpec::SliceOut(t, d, lo, hi, step) => {
            let s = spec_shape_with(env, t, memo);
            let Some(base) = tshape(s) else {
                return Shape::Unknown;
            };
            if base.is_empty() {
                return Shape::Unknown;
            }
            // `_bound_attr` per slot: absent/unbound/Null → default;
            // a bound non-int reads raw and simply fails the
            // `isinstance(int)` rewrite gate (the dim stays
            // unchanged — the spec still answers the tuple).
            let opt_int = |a: &Option<AttrArg>, default: Option<i64>| -> Option<i64> {
                match a {
                    None => default,
                    Some(arg) => match arg.resolve(env) {
                        None | Some(AttrValue::Null) => default,
                        Some(v) => av_int(&v),
                    },
                }
            };
            let (Some(dv), lo, hi, st) = (
                aval_int(env, d),
                opt_int(lo, Some(0)),
                opt_int(hi, None),
                opt_int(step, Some(1)),
            ) else {
                return Shape::Unknown;
            };
            let nd = dv.rem_euclid(base.len() as i64) as usize;
            let mut out = base.clone();
            if let Some(bd) = base[nd]
                && let (Some(lo), Some(st)) = (lo, st)
                && st > 0
            {
                let hi_eff = hi.map(|h| h.min(bd)).unwrap_or(bd);
                let n = hi_eff - lo;
                let q = n.div_euclid(st) + i64::from(n.rem_euclid(st) != 0);
                out[nd] = Some(q.max(0));
            }
            Shape::Tuple(out)
        }
        ShapeSpec::ChunkOut(t, c, d) => {
            let s = spec_shape_with(env, t, memo);
            let Some(base) = tshape(s) else {
                return Shape::Unknown;
            };
            if base.is_empty() {
                return Shape::Unknown;
            }
            let (Some(c), Some(d)) = (aval_int(env, c), aval_int(env, d)) else {
                return Shape::Unknown;
            };
            if c <= 0 {
                return Shape::Unknown;
            }
            let nd = d.rem_euclid(base.len() as i64) as usize;
            let mut out = base.clone();
            // `_s_chunk_out`: a None extent stays None (the guard
            // declines on unproven, unlike the typing rule's `or 0`).
            out[nd] = base[nd].map(|b| b / c);
            Shape::Tuple(out)
        }
        ShapeSpec::TransposeOut(t, d0, d1) => {
            let s = spec_shape_with(env, t, memo);
            let Some(base) = tshape(s) else {
                return Shape::Unknown;
            };
            if base.is_empty() {
                return Shape::Unknown;
            }
            let (Some(d0), Some(d1)) = (bound_int(env, d0, -2), bound_int(env, d1, -1)) else {
                return Shape::Unknown;
            };
            let n = base.len() as i64;
            let (a, b) = (d0.rem_euclid(n) as usize, d1.rem_euclid(n) as usize);
            let mut out = base.clone();
            out.swap(a, b);
            Shape::Tuple(out)
        }
        ShapeSpec::TailBlock(t, name) => {
            // `_attr_dims`: bare non-bool int or non-empty int tuple.
            let dims = env.attrs.get(name.as_str()).and_then(|v| match v {
                // `_attr_dims`: bare non-bool int, or a non-empty
                // tuple of non-bool ints.
                AttrValue::Int(n) => Some(vec![*n]),
                _ => seq_items(v).filter(|a| !a.is_empty()).and_then(|a| {
                    a.iter()
                        .map(|i| match i {
                            AttrValue::Int(n) => Some(*n),
                            AttrValue::UInt(n) => i64::try_from(*n).ok(),
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()
                }),
            });
            let s = spec_shape_with(env, t, memo);
            let Some(su) = tshape(s) else {
                return Shape::Unknown;
            };
            let (Some(dims), true) = (dims, su.iter().all(|d| d.is_some())) else {
                return Shape::Unknown;
            };
            // `_trailing_block`: k distinct in-range axes whose minimum
            // is rank-k IS the trailing block.
            let rank = su.len() as i64;
            let k = dims.len() as i64;
            if !(1..=rank).contains(&k) || !dims.iter().all(|d| (-rank..rank).contains(d)) {
                return Shape::Unknown;
            }
            let norm: BTreeSet<i64> = dims.iter().map(|d| d.rem_euclid(rank)).collect();
            if norm.len() as i64 != k || *norm.iter().next().unwrap_or(&0) != rank - k {
                return Shape::Unknown;
            }
            Shape::Tuple(su[su.len() - k as usize..].to_vec())
        }
        ShapeSpec::IselOut(t, d, idx) => {
            let s = spec_shape_with(env, t, memo);
            let Some(base) = tshape(s) else {
                return Shape::Unknown;
            };
            if base.is_empty() {
                return Shape::Unknown;
            }
            let (Some(d), Some(v)) = (aval_int(env, d), idx.resolve(env)) else {
                return Shape::Unknown;
            };
            let Some(items) = seq_items(&v) else {
                return Shape::Unknown;
            };
            let mut out = base.clone();
            out[d.rem_euclid(base.len() as i64) as usize] = Some(items.len() as i64);
            Shape::Tuple(out)
        }
    }
}

pub mod pred {
    //! Shape-reading guard predicates over [`BoundEnv`] — the
    //! stage-2 seam: each function mirrors one `cond.py` `_OPS`
    //! entry that reads shapes, taking [`ShapeSpec`] / [`AttrArg`]
    //! arguments exactly as the JSON grammar spells them (a metavar
    //! name, a computed spec, an attr metavar, or a literal). Stage
    //! 2's `cond` interpreter parses the JSON atom and dispatches
    //! here; every predicate **declines** (`false`) on anything it
    //! cannot prove — the `cond.py` strictness contract.
    //!
    //! Ported: `shaped`, `concrete`, `scalar`, `uniform`,
    //! `ones-but-last`, `rank`, `rank-eq`, `shape-eq`, `shape-compat`,
    //! `dim-eq`, `dim-compat`, `dim-eq-const`, `dim-eq-attr`,
    //! `dim-mod`, `bcast-into`, `mm-shape-ok`, `bcast-eq`,
    //! `ones-before`, `axes-last2`, `axes-distinct`, `axes-eq`,
    //! `axes-noop`, `axis`, `attr-cmp-dim`, `attr-range`, and the
    //! flat-map/repeat pairing guards (`axis-align-eq`,
    //! `bcast-dim-inv`, `flat-pair-unsq`, `flat-map-unsq`,
    //! `repeat-chain`, `repeat-heads`) — the whole `cond.py` `_OPS`
    //! table.
    //!
    //! The non-shape atoms, `and`/`or`/`not` combinators, and the
    //! `dspec` expr algebra live in [`crate::guard`]. The only
    //! remaining unported channels are the tensor-value ones
    //! (`tval`/`tapp` dspec exprs) — named refusals there.

    use super::{
        AttrArg, Memo, Shape, ShapeSpec, av_int, broadcast, matmul_shape, seq_items,
        spec_shape_with,
    };
    use handl_egraph::AttrValue;
    use handl_egraph::BoundEnv;
    use std::collections::BTreeMap;

    /// `_tshape`: resolve a spec to a tuple shape or `None`.
    fn tshape(env: &BoundEnv, spec: &ShapeSpec, memo: &mut Memo) -> Option<Vec<Option<i64>>> {
        spec_shape_with(env, spec, memo)
            .as_tuple()
            .map(<[Option<i64>]>::to_vec)
    }

    /// `_shape`: resolve a spec keeping `Invalid` distinct —
    /// predicates that compare shapes treat `Invalid` as a decline.
    fn shape(env: &BoundEnv, spec: &ShapeSpec, memo: &mut Memo) -> Shape {
        spec_shape_with(env, spec, memo)
    }

    /// `_aval` → int, `None` on unbound/non-int.
    fn aval_int(env: &BoundEnv, a: &AttrArg) -> Option<i64> {
        a.resolve(env).as_ref().and_then(av_int)
    }

    /// `_dim_at` on a resolved tuple.
    fn dim_at(s: &[Option<i64>], i: i64) -> Option<Option<i64>> {
        let n = s.len() as i64;
        if n == 0 || !(-n..n).contains(&i) {
            return None;
        }
        Some(s[i.rem_euclid(n) as usize])
    }

    /// `("shaped", T)` — `_shape_of(T)` is a tuple.
    pub fn shaped(env: &BoundEnv, t: &ShapeSpec) -> bool {
        let mut memo = Memo::new();
        matches!(spec_shape_with(env, t, &mut memo), Shape::Tuple(_))
    }

    /// `("concrete", T)` — tuple, every dim an int.
    pub fn concrete(env: &BoundEnv, t: &ShapeSpec) -> bool {
        let mut memo = Memo::new();
        spec_shape_with(env, t, &mut memo).is_concrete()
    }

    /// `("scalar", T)` — shape `== ()`.
    pub fn scalar(env: &BoundEnv, t: &ShapeSpec) -> bool {
        let mut memo = Memo::new();
        matches!(spec_shape_with(env, t, &mut memo), Shape::Tuple(d) if d.is_empty())
    }

    /// `("uniform", T)` — tuple, every dim `== 1` (scalar ok).
    pub fn uniform(env: &BoundEnv, t: &ShapeSpec) -> bool {
        let mut memo = Memo::new();
        matches!(spec_shape_with(env, t, &mut memo), Shape::Tuple(d) if d.iter().all(|x| *x == Some(1)))
    }

    /// `("ones-but-last", T)` — `s[:-1]` all `== 1`.
    pub fn ones_but_last(env: &BoundEnv, t: &ShapeSpec) -> bool {
        let mut memo = Memo::new();
        match spec_shape_with(env, t, &mut memo) {
            Shape::Tuple(d) => d[..d.len().saturating_sub(1)].iter().all(|x| *x == Some(1)),
            _ => false,
        }
    }

    /// `("rank", T, CMP, k)` — `len(shape) CMP k`.
    pub fn rank(env: &BoundEnv, t: &ShapeSpec, cmp: &str, k: i64) -> bool {
        let mut memo = Memo::new();
        let Some(s) = tshape(env, t, &mut memo) else {
            return false;
        };
        cmp_ord(cmp, s.len() as i64, k)
    }

    /// `("rank-eq", A, B)` — equal ranks, both shaped.
    pub fn rank_eq(env: &BoundEnv, a: &ShapeSpec, b: &ShapeSpec) -> bool {
        let mut memo = Memo::new();
        match (tshape(env, a, &mut memo), tshape(env, b, &mut memo)) {
            (Some(x), Some(y)) => x.len() == y.len(),
            _ => false,
        }
    }

    /// `("shape-eq", A, B)` — `sa == sb` exactly (`None` dims compare
    /// equal); A must be a tuple (`_tshape`), B may resolve to
    /// anything `_shape` returns (a tuple compares, `Invalid`/
    /// `Unknown` decline).
    pub fn shape_eq(env: &BoundEnv, a: &ShapeSpec, b: &ShapeSpec) -> bool {
        let mut memo = Memo::new();
        let Some(sa) = tshape(env, a, &mut memo) else {
            return false;
        };
        shape(env, b, &mut memo) == Shape::Tuple(sa)
    }

    /// `("shape-compat", A, B)` — same rank, `None` dims are
    /// wildcards.
    pub fn shape_compat(env: &BoundEnv, a: &ShapeSpec, b: &ShapeSpec) -> bool {
        let mut memo = Memo::new();
        let (Some(sa), Some(sb)) = (tshape(env, a, &mut memo), tshape(env, b, &mut memo)) else {
            return false;
        };
        sa.len() == sb.len()
            && sa
                .iter()
                .zip(sb.iter())
                .all(|(da, db)| da.is_none() || db.is_none() || da == db)
    }

    /// `("dim-eq", A, i, B, j)` — `sa[i] == sb[j]` (`None == None` ok).
    pub fn dim_eq(env: &BoundEnv, a: &ShapeSpec, i: i64, b: &ShapeSpec, j: i64) -> bool {
        let mut memo = Memo::new();
        let (Some(sa), Some(sb)) = (tshape(env, a, &mut memo), tshape(env, b, &mut memo)) else {
            return false;
        };
        match (dim_at(&sa, i), dim_at(&sb, j)) {
            (Some(da), Some(db)) => da == db,
            _ => false,
        }
    }

    /// `("dim-compat", A, i, B, j)` — `sa[i]`/`sb[j]` `None`-wildcard
    /// equality.
    pub fn dim_compat(env: &BoundEnv, a: &ShapeSpec, i: i64, b: &ShapeSpec, j: i64) -> bool {
        let mut memo = Memo::new();
        let (Some(sa), Some(sb)) = (tshape(env, a, &mut memo), tshape(env, b, &mut memo)) else {
            return false;
        };
        match (dim_at(&sa, i), dim_at(&sb, j)) {
            (Some(da), Some(db)) => da.is_none() || db.is_none() || da == db,
            _ => false,
        }
    }

    /// `("dim-eq-const", T, i, k)` — `s[i] == k`.
    pub fn dim_eq_const(env: &BoundEnv, t: &ShapeSpec, i: i64, k: i64) -> bool {
        let mut memo = Memo::new();
        let Some(s) = tshape(env, t, &mut memo) else {
            return false;
        };
        dim_at(&s, i) == Some(Some(k))
    }

    /// `("dim-eq-attr", A, K, B, K2)` — `sa[K] == sb[K2]` with the
    /// axes named by bound attr metavars.
    pub fn dim_eq_attr(
        env: &BoundEnv,
        a: &ShapeSpec,
        ka: &AttrArg,
        b: &ShapeSpec,
        kb: &AttrArg,
    ) -> bool {
        let mut memo = Memo::new();
        let (Some(sa), Some(sb)) = (tshape(env, a, &mut memo), tshape(env, b, &mut memo)) else {
            return false;
        };
        let (Some(ia), Some(ib)) = (aval_int(env, ka), aval_int(env, kb)) else {
            return false;
        };
        match (dim_at(&sa, ia), dim_at(&sb, ib)) {
            (Some(da), Some(db)) => da == db,
            _ => false,
        }
    }

    /// `("dim-mod", T, K, m, r)` — `sa[K % rank] % m == r`.
    pub fn dim_mod(env: &BoundEnv, t: &ShapeSpec, k: &AttrArg, m: i64, r: i64) -> bool {
        let mut memo = Memo::new();
        let Some(s) = tshape(env, t, &mut memo) else {
            return false;
        };
        let Some(d) = aval_int(env, k) else {
            return false;
        };
        if m == 0 || s.is_empty() || !(-(s.len() as i64)..s.len() as i64).contains(&d) {
            return false;
        }
        match dim_at(&s, d) {
            Some(Some(n)) => n % m == r,
            _ => false,
        }
    }

    /// `("bcast-into", T, U)` — `broadcast(s_T, s_U) == s_U`.
    pub fn bcast_into(env: &BoundEnv, t: &ShapeSpec, u: &ShapeSpec) -> bool {
        let mut memo = Memo::new();
        let small = shape(env, t, &mut memo);
        let big = shape(env, u, &mut memo);
        let Some(small) = small.as_tuple() else {
            return false;
        };
        let Shape::Tuple(big_dims) = &big else {
            return false;
        };
        super::broadcast(
            &Shape::Tuple(small.to_vec()),
            &Shape::Tuple(big_dims.clone()),
        ) == big
    }

    /// `("mm-shape-ok", A, B)` — `_matmul_shape(sa, sb)` is a tuple.
    pub fn mm_shape_ok(env: &BoundEnv, a: &ShapeSpec, b: &ShapeSpec) -> bool {
        let mut memo = Memo::new();
        let sa = shape(env, a, &mut memo);
        let sb = shape(env, b, &mut memo);
        matmul_shape(&sa, &sb).is_some()
    }

    /// `("bcast-eq", A, B, C, D)` — `bcast(sA,sB) == bcast(sC,sD)`,
    /// every spec a tuple and both broadcasts succeeding.
    pub fn bcast_eq(
        env: &BoundEnv,
        a: &ShapeSpec,
        b: &ShapeSpec,
        c: &ShapeSpec,
        d: &ShapeSpec,
    ) -> bool {
        let mut memo = Memo::new();
        let specs = [a, b, c, d].map(|s| shape(env, s, &mut memo));
        if !specs.iter().all(|s| matches!(s, Shape::Tuple(_))) {
            return false;
        }
        let g1 = super::broadcast(&specs[0], &specs[1]);
        let g2 = super::broadcast(&specs[2], &specs[3]);
        matches!(g1, Shape::Tuple(_)) && g1 == g2
    }

    /// `("ones-before", T, K)` — `s[i] == 1` for every dim before the
    /// normalized attr axis `K mod (rank+1)`.
    pub fn ones_before(env: &BoundEnv, t: &ShapeSpec, k: &AttrArg) -> bool {
        let mut memo = Memo::new();
        let (Some(s), Some(k)) = (tshape(env, t, &mut memo), aval_int(env, k)) else {
            return false;
        };
        let nd = k.rem_euclid(s.len() as i64 + 1) as usize;
        s[..nd].iter().all(|d| *d == Some(1))
    }

    /// `_axes_pair`: normalize a bound `transpose` axis pair against
    /// `rank` — both attrs missing means the bare `t()` last-two
    /// swap; otherwise each must be an int in `[-rank, rank)`.
    fn axes_pair(env: &BoundEnv, k0: &AttrArg, k1: &AttrArg, rank: usize) -> Option<(i64, i64)> {
        if rank < 2 {
            return None;
        }
        let a0 = k0.resolve(env);
        let a1 = k1.resolve(env);
        // `_MISSING` is *unbound only* — a bound `null` attr reads
        // `None`, which `_axis_pair` rejects as non-int (decline),
        // it does NOT take the bare-`t()` default.
        let missing = |v: &Option<AttrValue>| v.is_none();
        if missing(&a0) && missing(&a1) {
            return Some((rank as i64 - 2, rank as i64 - 1));
        }
        let (Some(d0), Some(d1)) = (a0.as_ref().and_then(av_int), a1.as_ref().and_then(av_int))
        else {
            return None;
        };
        let r = rank as i64;
        if !(-r..r).contains(&d0) || !(-r..r).contains(&d1) {
            return None;
        }
        Some((d0.rem_euclid(r), d1.rem_euclid(r)))
    }

    /// `("axes-last2", T, D0, D1)` — the transpose pair is the
    /// last-two swap.
    pub fn axes_last2(env: &BoundEnv, t: &ShapeSpec, d0: &AttrArg, d1: &AttrArg) -> bool {
        let mut memo = Memo::new();
        let Some(s) = tshape(env, t, &mut memo) else {
            return false;
        };
        let Some(pair) = axes_pair(env, d0, d1, s.len()) else {
            return false;
        };
        let r = s.len() as i64;
        // `set(pair) == {rank-2, rank-1}` — either order counts.
        (pair.0 == r - 2 && pair.1 == r - 1) || (pair.0 == r - 1 && pair.1 == r - 2)
    }

    /// `("axes-distinct", T, D0, D1)` — the pair normalizes and
    /// `d0 != d1`.
    pub fn axes_distinct(env: &BoundEnv, t: &ShapeSpec, d0: &AttrArg, d1: &AttrArg) -> bool {
        let mut memo = Memo::new();
        let Some(s) = tshape(env, t, &mut memo) else {
            return false;
        };
        axes_pair(env, d0, d1, s.len()).is_some_and(|p| p.0 != p.1)
    }

    /// `("axes-eq", T, A0, A1, B0, B1)` — two distinct pairs naming
    /// the same axis set.
    pub fn axes_eq(
        env: &BoundEnv,
        t: &ShapeSpec,
        a0: &AttrArg,
        a1: &AttrArg,
        b0: &AttrArg,
        b1: &AttrArg,
    ) -> bool {
        let mut memo = Memo::new();
        let Some(s) = tshape(env, t, &mut memo) else {
            return false;
        };
        let (Some(p1), Some(p2)) = (
            axes_pair(env, a0, a1, s.len()),
            axes_pair(env, b0, b1, s.len()),
        ) else {
            return false;
        };
        p1.0 != p1.1
            && p2.0 != p2.1
            && (p1.0 == p2.0 && p1.1 == p2.1 || p1.0 == p2.1 && p1.1 == p2.0)
    }

    /// `("axes-noop", T, D0, D1)` — `transpose(u, D0, D1)` is a
    /// semantic no-op: same axis, or both swapped extents are 1.
    pub fn axes_noop(env: &BoundEnv, t: &ShapeSpec, d0: &AttrArg, d1: &AttrArg) -> bool {
        let mut memo = Memo::new();
        let Some(s) = tshape(env, t, &mut memo) else {
            return false;
        };
        if s.is_empty() {
            return false;
        }
        if s.len() == 1 {
            // Rank-1: every valid pair is (0, 0) — both attrs must be
            // ints in [-1, 1).
            return aval_int(env, d0).is_some_and(|d| (-1..1).contains(&d))
                && aval_int(env, d1).is_some_and(|d| (-1..1).contains(&d));
        }
        let Some((d0, d1)) = axes_pair(env, d0, d1, s.len()) else {
            return false;
        };
        d0 == d1 || (s[d0 as usize] == Some(1) && s[d1 as usize] == Some(1))
    }

    /// `("axis", T, NAME, k)` — the bound attr sits at the same
    /// right-aligned axis position: `v % rank == k % rank`.
    pub fn axis(env: &BoundEnv, t: &ShapeSpec, name: &AttrArg, k: i64) -> bool {
        let mut memo = Memo::new();
        let (Some(s), Some(v)) = (tshape(env, t, &mut memo), aval_int(env, name)) else {
            return false;
        };
        if s.is_empty() {
            return false;
        }
        v.rem_euclid(s.len() as i64) == k.rem_euclid(s.len() as i64)
    }

    /// `("attr-cmp-dim", NAME, CMP, T, K)` — `$attr:NAME CMP
    /// shape(T)[$attr:K]`; both sides must resolve to ints.
    pub fn attr_cmp_dim(
        env: &BoundEnv,
        name: &AttrArg,
        cmp: &str,
        t: &ShapeSpec,
        k: &AttrArg,
    ) -> bool {
        let mut memo = Memo::new();
        let (Some(v), Some(s), Some(ki)) = (
            aval_int(env, name),
            tshape(env, t, &mut memo),
            aval_int(env, k),
        ) else {
            return false;
        };
        match dim_at(&s, ki) {
            Some(Some(d)) => cmp_ord(cmp, v, d),
            _ => false,
        }
    }

    /// `("attr-range", V, T, K)` — bound attr `V` IS `range(s[K])`:
    /// a tuple of non-bool ints `0..d-1` where `d = s[K % rank]`.
    pub fn attr_range(env: &BoundEnv, v: &AttrArg, t: &ShapeSpec, k: &AttrArg) -> bool {
        let mut memo = Memo::new();
        let (Some(val), Some(s), Some(ki)) =
            (v.resolve(env), tshape(env, t, &mut memo), aval_int(env, k))
        else {
            return false;
        };
        if s.is_empty() {
            return false;
        }
        let Some(items) = seq_items(&val) else {
            return false;
        };
        // `isinstance(i, int) and not isinstance(i, bool)` — bools are
        // excluded here even though they ride `isinstance(int)`
        // elsewhere.
        let ints: Option<Vec<i64>> = items
            .iter()
            .map(|i| match i {
                AttrValue::Int(n) => Some(*n),
                AttrValue::UInt(n) => i64::try_from(*n).ok(),
                _ => None,
            })
            .collect();
        let Some(ints) = ints else {
            return false;
        };
        match dim_at(&s, ki) {
            Some(Some(d)) if d >= 0 => ints == (0..d).collect::<Vec<_>>(),
            _ => false,
        }
    }

    // -----------------------------------------------------------------
    //  The flat-map / repeat pairing guards — `_p_*` ports over
    //  concrete int shapes. Every unprovable piece declines.
    // -----------------------------------------------------------------

    /// `_aval` → an int-tuple read for the repeat specs: an `Array`
    /// attr whose elements read int-like (`Some`) or not (`None` —
    /// a non-int element never equals a concrete dim, so the strict
    /// comparisons below decline exactly where Python's `!=`/`==`
    /// on mixed types would; float-vs-int numeric equality is the one
    /// place this reads *stricter* than Python — a safe direction).
    fn aval_elems(env: &BoundEnv, a: &AttrArg) -> Option<Vec<Option<i64>>> {
        let v = a.resolve(env)?;
        seq_items(&v).map(|items| items.iter().map(av_int).collect())
    }

    /// `_concrete_specs`: every spec must resolve to an all-int tuple.
    fn concrete_specs(
        env: &BoundEnv,
        specs: &[&ShapeSpec],
        memo: &mut Memo,
    ) -> Option<Vec<Vec<i64>>> {
        specs
            .iter()
            .map(|s| {
                let d = tshape(env, s, memo)?;
                d.into_iter().collect::<Option<Vec<i64>>>()
            })
            .collect()
    }

    /// `_concrete_rank2`: rank ≥ 2, all dims concrete.
    fn concrete_rank2(env: &BoundEnv, spec: &ShapeSpec, memo: &mut Memo) -> Option<Vec<i64>> {
        let d = tshape(env, spec, memo)?;
        if d.len() < 2 {
            return None;
        }
        d.into_iter().collect()
    }

    /// `_unsq_shape` on a concrete int tuple: insert a 1 at
    /// `k % (rank+1)`.
    fn unsq_vec(shape: &[i64], k: i64) -> Vec<i64> {
        let nd = k.rem_euclid(shape.len() as i64 + 1) as usize;
        let mut out = shape[..nd].to_vec();
        out.push(1);
        out.extend_from_slice(&shape[nd..]);
        out
    }

    /// `_stride`: flat-index stride of `shape[i]` — the product of
    /// the trailing dims. `i128` with a checked product: an
    /// overflow declines (never a wrapped stride that could
    /// fabricate an equality).
    fn stride(shape: &[i64], i: usize) -> Option<i128> {
        shape[i + 1..]
            .iter()
            .try_fold(1i128, |acc, d| acc.checked_mul(i128::from(*d)))
    }

    /// Concrete-int broadcast of two dim tuples — `_broadcast`'s
    /// int-only slice; `Invalid`/unknown results decline.
    fn broadcast_ints(a: &[i64], b: &[i64]) -> Option<Vec<i64>> {
        let ta = Shape::Tuple(a.iter().map(|d| Some(*d)).collect());
        let tb = Shape::Tuple(b.iter().map(|d| Some(*d)).collect());
        match broadcast(&ta, &tb) {
            Shape::Tuple(d) => d.into_iter().collect(),
            _ => None,
        }
    }

    /// `_viewed_map`: flat-read coeff map of `src` broadcast into
    /// `grid` — `{grid position: flat stride}` per non-1 dim.
    fn viewed_map(src: &[i64], grid: &[i64]) -> Option<BTreeMap<i64, i128>> {
        let off = grid.len() as i64 - src.len() as i64;
        let mut out = BTreeMap::new();
        for (j, &dj) in src.iter().enumerate() {
            if dj != 1 {
                out.insert(off + j as i64, stride(src, j)?);
            }
        }
        Some(out)
    }

    /// `_shifted_map`: `src`'s coeff map through an axis insertion —
    /// positions at or past `nd` read one slot further right.
    fn shifted_map(src: &[i64], grid: &[i64], nd: i64) -> Option<BTreeMap<i64, i128>> {
        let off = grid.len() as i64 - src.len() as i64;
        let mut out = BTreeMap::new();
        for (j, &dj) in src.iter().enumerate() {
            if dj != 1 {
                let p = off + j as i64;
                out.insert(p + i64::from(p >= nd), stride(src, j)?);
            }
        }
        Some(out)
    }

    /// `_flat_map_ok`: coeff maps agree at every grid position the
    /// grid actually varies (`w[p] > 1`; size-1 positions read the
    /// constant 0 either way).
    fn flat_map_ok(lmap: &BTreeMap<i64, i128>, rmap: &BTreeMap<i64, i128>, w: &[i64]) -> bool {
        w.iter().enumerate().all(|(p, &wd)| {
            wd == 1
                || lmap.get(&(p as i64)).copied().unwrap_or(0)
                    == rmap.get(&(p as i64)).copied().unwrap_or(0)
        })
    }

    /// `_repeat_merge_ok`: the chain's shape equation at the
    /// normalized axis — expand is exactly the unsqueezed shape with
    /// an int factor `r > 1` at `d`, reshape merges `d-1`/`d`.
    fn repeat_merge_ok(bs: &[i64], d: i64, es: &[Option<i64>], rs: &[Option<i64>]) -> bool {
        let nd = bs.len();
        let d = d.rem_euclid(nd as i64 + 1);
        let us = unsq_vec(bs, d);
        if es.len() != us.len() || rs.len() != nd || d == 0 {
            return false;
        }
        let d = d as usize;
        let Some(r) = es[d] else {
            return false; // `isinstance(r, int)`
        };
        if r <= 1 {
            return false;
        }
        for (i, &u) in us.iter().enumerate() {
            if i != d && es[i] != Some(u) {
                return false;
            }
        }
        // merged = us[..d-1] + (us[d-1]*r,) + us[d+1..]
        let mut merged: Vec<i128> = us[..d - 1].iter().map(|&x| i128::from(x)).collect();
        let Some(prod) = i128::from(us[d - 1]).checked_mul(i128::from(r)) else {
            return false;
        };
        merged.push(prod);
        merged.extend(us[d + 1..].iter().map(|&x| i128::from(x)));
        rs.iter()
            .zip(merged.iter())
            .all(|(&e, &m)| e.map(i128::from) == Some(m))
    }

    /// `_rep_factor`: the expand factor at the normalized axis.
    fn rep_factor(es: &[Option<i64>], d: i64, rank: usize) -> Option<i64> {
        let dd = d.rem_euclid(rank as i64 + 1);
        if dd as usize >= es.len() {
            return None;
        }
        es[dd as usize]
    }

    /// `("axis-align-eq", A, B, K)` — `_p_axis_align_eq`: the bound
    /// axis sits at the same right-aligned position in both shapes.
    /// Only the ranks are read — dims need not be concrete.
    pub fn axis_align_eq(env: &BoundEnv, a: &ShapeSpec, b: &ShapeSpec, k: &AttrArg) -> bool {
        let mut memo = Memo::new();
        let (Some(sa), Some(sb), Some(d)) = (
            tshape(env, a, &mut memo),
            tshape(env, b, &mut memo),
            aval_int(env, k),
        ) else {
            return false;
        };
        if sa.is_empty() || sb.is_empty() {
            return false;
        }
        d.rem_euclid(sa.len() as i64) - sa.len() as i64
            == d.rem_euclid(sb.len() as i64) - sb.len() as i64
    }

    /// `("bcast-dim-inv", V, U, K)` — `_p_bcast_dim_inv`: `V`
    /// right-aligns into `U`'s grid and is invariant (extent 1 or
    /// absent) along `U`'s axis `K`.
    pub fn bcast_dim_inv(env: &BoundEnv, v: &ShapeSpec, u: &ShapeSpec, k: &AttrArg) -> bool {
        let mut memo = Memo::new();
        let (Some(sv), Some(su), Some(d)) = (
            tshape(env, v, &mut memo),
            tshape(env, u, &mut memo),
            aval_int(env, k),
        ) else {
            return false;
        };
        if su.is_empty() {
            return false;
        }
        let pos = d.rem_euclid(su.len() as i64) - su.len() as i64 + sv.len() as i64;
        if !(0..sv.len() as i64).contains(&pos) {
            return true; // V has no such axis — broadcasts the constant
        }
        sv[pos as usize] == Some(1) // a None extent declines (not provably 1)
    }

    /// `("flat-pair-unsq", T, K, G, S)` — `_p_flat_pair_unsq`: the
    /// *wr* naturality's u-side — `u`'s flat reads agree across
    /// `unsq→S` vs `bcast→G`. All specs must be concrete ints.
    pub fn flat_pair_unsq(
        env: &BoundEnv,
        t: &ShapeSpec,
        k: &AttrArg,
        g: &ShapeSpec,
        s: &ShapeSpec,
    ) -> bool {
        let mut memo = Memo::new();
        let (Some(k), Some(specs)) = (aval_int(env, k), concrete_specs(env, &[t, g, s], &mut memo))
        else {
            return false;
        };
        let mut it = specs.into_iter();
        let (Some(su), Some(gr), Some(sh)) = (it.next(), it.next(), it.next()) else {
            return false;
        };
        let (r, n, n2) = (su.len() as i64, gr.len() as i64, sh.len() as i64);
        let nd = k.rem_euclid(r + 1) as usize;
        for (kk, &suk) in su.iter().enumerate() {
            if suk == 1 {
                continue;
            }
            let j = if kk < nd { kk } else { kk + 1 };
            let (p1, p2) = (n - r + kk as i64, n2 - (r + 1) + j as i64);
            if !(0..n).contains(&p1) || !(0..n2).contains(&p2) {
                return false;
            }
            let (p1, p2) = (p1 as usize, p2 as usize);
            if gr[p1] != suk || sh[p2] != suk {
                return false;
            }
            match (stride(&gr, p1), stride(&sh, p2)) {
                (Some(a), Some(b)) if a == b => {}
                _ => return false,
            }
        }
        true
    }

    /// `("flat-map-unsq", T, K, V, S)` — `_p_flat_map_unsq`: the
    /// *wl* naturality — `mul(unsq(u,K), reshape(v,S)) ==
    /// unsq(mul(u,v), K)` at shape level: the output grids coincide
    /// and both operands' flat-read coeff maps agree across the two
    /// orders. All shapes must be concrete ints.
    pub fn flat_map_unsq(
        env: &BoundEnv,
        t: &ShapeSpec,
        k: &AttrArg,
        v: &ShapeSpec,
        s: &ShapeSpec,
    ) -> bool {
        let mut memo = Memo::new();
        let (Some(k), Some(specs)) = (aval_int(env, k), concrete_specs(env, &[t, v, s], &mut memo))
        else {
            return false;
        };
        let mut it = specs.into_iter();
        let (Some(su), Some(sv), Some(sh)) = (it.next(), it.next(), it.next()) else {
            return false;
        };
        let su_p = unsq_vec(&su, k);
        let (Some(gu), Some(gl)) = (broadcast_ints(&su, &sv), broadcast_ints(&su_p, &sh)) else {
            return false;
        };
        let nd = k.rem_euclid(gu.len() as i64 + 1) as usize;
        let mut w = gu[..nd].to_vec();
        w.push(1);
        w.extend_from_slice(&gu[nd..]);
        if w != gl {
            return false;
        }
        let nd = nd as i64;
        let (Some(l1), Some(r1), Some(l2), Some(r2)) = (
            viewed_map(&su_p, &w),
            shifted_map(&su, &gu, nd),
            viewed_map(&sh, &w),
            shifted_map(&sv, &gu, nd),
        ) else {
            return false;
        };
        flat_map_ok(&l1, &r1, &w) && flat_map_ok(&l2, &r2, &w)
    }

    /// `("repeat-chain", T, UD, ES, RS)` — `_p_repeat_chain`: the
    /// `unsqueeze → expand → reshape` copy-map chain is a
    /// `repeat_interleave` on `T`'s dim `d-1`. `T` must be a concrete
    /// tuple; `UD`/`ES`/`RS` resolve through `_aval`.
    pub fn repeat_chain(
        env: &BoundEnv,
        t: &ShapeSpec,
        ud: &AttrArg,
        es: &AttrArg,
        rs: &AttrArg,
    ) -> bool {
        let mut memo = Memo::new();
        let (Some(d), Some(es), Some(rs), Some(bs)) = (
            aval_int(env, ud),
            aval_elems(env, es),
            aval_elems(env, rs),
            tshape(env, t, &mut memo),
        ) else {
            return false;
        };
        let Some(bs) = bs.into_iter().collect::<Option<Vec<i64>>>() else {
            return false; // `any(x is None)` — a `None` dim declines
        };
        repeat_merge_ok(&bs, d, &es, &rs)
    }

    /// `("repeat-heads", A, B, UD, ES)` — `_p_repeat_heads`: the
    /// repeated-head relation `sa[-2] == sb[-2] * r` where `r` is the
    /// expand factor at the normalized unsqueeze axis. Both shapes
    /// rank ≥ 2 and fully concrete.
    pub fn repeat_heads(
        env: &BoundEnv,
        a: &ShapeSpec,
        b: &ShapeSpec,
        ud: &AttrArg,
        es: &AttrArg,
    ) -> bool {
        let mut memo = Memo::new();
        let (Some(sa), Some(sb), Some(d), Some(es)) = (
            concrete_rank2(env, a, &mut memo),
            concrete_rank2(env, b, &mut memo),
            aval_int(env, ud),
            aval_elems(env, es),
        ) else {
            return false;
        };
        let Some(r) = rep_factor(&es, d, sb.len()) else {
            return false;
        };
        i128::from(sa[sa.len() - 2]) == i128::from(sb[sb.len() - 2]) * i128::from(r)
    }

    /// The six comparison spellings (`_CMPS` on ints).
    fn cmp_ord(cmp: &str, a: i64, b: i64) -> bool {
        match cmp {
            "==" => a == b,
            "!=" => a != b,
            "<" => a < b,
            "<=" => a <= b,
            ">" => a > b,
            ">=" => a >= b,
            _ => false,
        }
    }
}
