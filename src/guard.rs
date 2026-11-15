//! Guards-as-data — the stage-1 `cond`/`dspec` interpreter of design
//! doc §2 (`mlopt_core.laws.cond` ported over
//! [`handl_egraph::BoundEnv`]).
//!
//! A law record's `cond` is a predicate tree and its `dspec` a
//! `{NAME: expr}` derive spec — both pure JSON data. This module
//! parses them once at [`crate::law::Law::compile`] into typed ASTs
//! ([`Cond`]/[`Dspec`]) and interprets them per firing through
//! [`handl_egraph::Rewrite::with_check`]/`with_derive`.
//!
//! # Vocabulary (stage 1 — attr/scalar only)
//!
//! * Combinators: `and`, `or`, `not`, plus bare `true`/`false`.
//! * Term predicates: `op-in`, `leaf`, `const`, `term-eq`,
//!   `const-num`, `const-cmp` — read [`BoundEnv::terms`]. A "kernel
//!   op" is a `Term::Op` carrying the bridge's `source` attr (the same
//!   `sourced` test `term_bridge::to_kernel` uses); `Leaf` terms
//!   decode through `to_kernel` for the `Const` reads (`const`,
//!   `const-num`, `const-cmp`, the `("const", T)` expr).
//! * Attr predicates: `attr-is`, `attr-eq`, `attr-in`, `attr-type`,
//!   `attr-len`, `attr-eq-attr`, `attr-cmp`, `attr-ints` — read
//!   [`BoundEnv::attrs`] (bare names; the Python `"$attr:"` key
//!   mangling is the `Subst::attrs` split). Attr positions accept the
//!   same three spellings as `_aval`: a string names a bound attr
//!   metavariable, an array is an inline derive [`Expr`], anything
//!   else is a literal.
//! * Exprs: literals, `lit`, `attr`, `attr0`, `const`, `len`, `tuple`,
//!   `concat`, `gather`, `unique`, `posmap`, `nth`, `take`, `sum`,
//!   `guard`, `if`, the arithmetic/comparison ops `add` `sub` `mul`
//!   `fdiv` `floordiv` `neg` `recip` `float` `int` `sqrt` `rsqrt`
//!   `max` `min` `pow` `mod` — every node `cond.py`'s `_DOPS` table
//!   has that needs no shape inference and no tensor backend.
//!
//! # Strictness (ported verbatim)
//!
//! * Predicates that cannot be *proven* decline to `false`; derive
//!   exprs that cannot be computed veto the whole spec (`None`) —
//!   the `check`/`derive` contract the e-graph honours.
//! * Python's `TypeError`/`ArithmeticError` arms (unorderable
//!   comparisons, bad indexing, division by zero, `math.sqrt` of a
//!   negative) are declines here too — never a fabricated value.
//!   Where Python would *raise* out of `eval_cond` (a `TypeError`
//!   `_p_attr_cmp` doesn't catch is already a decline there; the
//!   remaining raise-sites are malformed-node `ValueError`s, which
//!   this port refuses at parse time instead).
//!
//! # Shape-reading predicates (stage 2 — [`crate::shape`])
//!
//! The shape predicates and shape specs port with
//! [`crate::shape`]'s inference: `shaped`, `concrete`, `scalar`,
//! `uniform`, `ones-but-last`, `rank`, `rank-eq`, `shape-eq`,
//! `shape-compat`, `dim-eq`, `dim-compat`, `dim-eq-const`,
//! `dim-eq-attr`, `dim-mod`, `bcast-into`, `mm-shape-ok`,
//! `axes-last2`, `axes-distinct`, `axes-eq`, `axis`, `attr-cmp-dim`,
//! `bcast-eq`, `ones-before`, `axes-noop`, `attr-range`, and the
//! `_SPEC_OPS` specs (`mm-out`/`bcast`/`unsq-out`/`reshape-out`/
//! `getitem-out`/`select-out`/`slice-out`/`chunk-out`/
//! `transpose-out`/`tail-block`/`isel-out`/`vshape`) — parsed into
//! [`ShapePred`] and evaluated through [`crate::shape::pred`]. Attr
//! positions accept the name/literal spellings; an inline derive
//! *expr* inside a shape attr position is a named
//! [`GuardError::Unsupported`] (shape specs do not evaluate the
//! stage-1 `Aval` exprs).
//!
//! # Deliberately absent (named, never approximated)
//!
//! * The flat-map/repeat pairing guards: `axis-align-eq`,
//!   `bcast-dim-inv`, `flat-pair-unsq`, `flat-map-unsq`,
//!   `repeat-chain`, `repeat-heads` — they need the carrier-side
//!   flat-map machinery, not landed here.
//! * The shape-reading *exprs* `shape`, `vshape`, `dim`, `leaf-dim`,
//!   `bcast` inside `dspec` — the derive channel's shape reads stay
//!   [`GuardError::Unsupported`] (predicates cover the guard side;
//!   derived shape bindings are a separate surface).
//! * The tensor-value channels `tval`/`tapp` and the `$leaf:` tensor
//!   mint — they need a registered [`TensorAlgebra`](cond.py)-equivalent
//!   backend (kit/backend territory). `$leaf:` exprs evaluating to
//!   scalars mint `Const` leaves exactly like `_mint_leaf`; anything
//!   else declines.
//! * The tensor-value channels `tval`/`tapp` and the `$leaf:` tensor
//!   mint — they need a registered [`TensorAlgebra`](cond.py)-equivalent
//!   backend (kit/backend territory). `$leaf:` exprs evaluating to
//!   scalars mint `Const` leaves exactly like `_mint_leaf`; anything
//!   else declines.
//!
//! # Documented divergences (all conservative)
//!
//! * Python ints are unbounded; [`V::Int`] is `i64`. Arithmetic that
//!   would overflow declines (Python computes) — dims/attrs never
//!   approach the bound, and `pow` keeps cond.py's own
//!   `_POW_MAX_BITS`-style budget via checked `i128` math before the
//!   `i64` narrowing.
//! * `int`/`float` casts of strings implement a strict subset of
//!   Python's `int()`/`float()` (whitespace-trimmed, no underscores,
//!   no `0x`/`1_000` spellings) — a spelling Python accepts but this
//!   port declines is a *decline*, never a wrong value.
//! * `attr-is`'s Python `is` identity is approximated: `None`/`bool`
//!   singletons and CPython's small-int cache (`-5..=256`) compare by
//!   value; every other pair is `false` — matching `is` on distinct
//!   objects (the bank only ever spells `attr-is` against `true`/
//!   `false`).
//! * Non-scalar `add`/`mul` keep Python's operator fallback with no
//!   tensor algebra registered: `Seq`/`Str` concatenation and
//!   sequence repetition; every other non-scalar operand declines
//!   (a tensor payload cannot exist here — there is no `tval`).

use std::collections::BTreeMap;
use std::rc::Rc;

use handl_egraph::{AttrValue, BoundEnv, Derived, KernelTerm, Term, TermAttr, to_kernel};
use serde_json::Value;

/// Why a `cond`/`dspec` tree failed to parse — a compile-time
/// refusal, never an eval-time surprise.
#[derive(Clone, Debug, PartialEq)]
pub enum GuardError {
    /// The node is malformed — a bad shape, arity, literal type, or
    /// an op name this DSL does not define (cond.py's `ValueError`
    /// arm: a bug in the record, refused at load).
    Malformed(String),
    /// A well-formed node the interpreter does not cover — the
    /// flat-map/repeat pairing guards or a tensor-backend channel
    /// (shape-reading predicates *are* covered — see [`Pred::Shape`]).
    /// Refused at compile, named (design doc §2: declines are named,
    /// never approximated).
    Unsupported(String),
}

impl std::fmt::Display for GuardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GuardError::Malformed(d) => write!(f, "malformed guard node: {d}"),
            GuardError::Unsupported(d) => write!(f, "guard op unsupported: {d}"),
        }
    }
}

impl std::error::Error for GuardError {}

// ---------------------------------------------------------------------------
//  The value domain — cond.py's int/float/bool/None/str/tuple lifted
// ---------------------------------------------------------------------------

/// A guard-dsl value: the Python `int | float | bool | None | str |
/// tuple` domain the `cond`/`dspec` exprs compute over.
///
/// `Seq` is the tuple/list normal form (the port's
/// tuple-normalised attrs); `Obj` carries map payloads (the
/// `__attrs__` object a whole-payload `attrs_bind` produces) so
/// `attr-eq`-style comparisons stay honest.
#[derive(Clone, Debug, PartialEq)]
pub enum V {
    /// Python `None` / JSON `null`.
    Null,
    /// Python `bool` (admitted wherever `isinstance(x, int)` is —
    /// the numeric tower).
    Bool(bool),
    /// Python `int` (i64 — see the module docs' divergence note).
    Int(i64),
    /// Python `float`.
    Float(f64),
    /// Python `str`.
    Str(Rc<str>),
    /// Python `tuple`/`list` — sequence values.
    Seq(Rc<[V]>),
    /// A string-keyed map (JSON object / `AttrValue::Object`).
    Obj(Rc<BTreeMap<Rc<str>, V>>),
}

impl V {
    /// Lift a JSON literal into the value domain.
    ///
    /// Numbers keep their JSON spelling: integral `i64`/`u64` become
    /// [`V::Int`] (a `u64` past `i64::MAX` degrades to `Float` —
    /// documented), everything else `Float`.
    pub fn from_json(v: &Value) -> V {
        match v {
            Value::Null => V::Null,
            Value::Bool(b) => V::Bool(*b),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    V::Int(i)
                } else if let Some(u) = n.as_u64() {
                    match i64::try_from(u) {
                        Ok(i) => V::Int(i),
                        Err(_) => V::Float(u as f64),
                    }
                } else {
                    V::Float(n.as_f64().unwrap_or(f64::NAN))
                }
            }
            Value::String(s) => V::Str(Rc::from(s.as_str())),
            Value::Array(items) => {
                V::Seq(items.iter().map(V::from_json).collect::<Vec<_>>().into())
            }
            Value::Object(map) => V::Obj(Rc::new(
                map.iter()
                    .map(|(k, v)| (Rc::from(k.as_str()), V::from_json(v)))
                    .collect(),
            )),
        }
    }

    /// Lower into the e-graph's [`AttrValue`] — the `Derived::Attr`
    /// payload channel.
    pub fn to_attr(&self) -> AttrValue {
        match self {
            V::Null => AttrValue::Null,
            V::Bool(b) => AttrValue::Bool(*b),
            V::Int(n) => AttrValue::Int(*n),
            V::Float(f) => AttrValue::Float(*f),
            V::Str(s) => AttrValue::Str(s.clone()),
            V::Seq(items) => {
                AttrValue::Array(items.iter().map(V::to_attr).collect::<Vec<_>>().into())
            }
            V::Obj(map) => AttrValue::Object(Rc::new(
                map.iter().map(|(k, v)| (k.clone(), v.to_attr())).collect(),
            )),
        }
    }

    /// Lift a bound [`AttrValue`] into the value domain.
    pub fn from_attr(a: &AttrValue) -> V {
        match a {
            AttrValue::Null => V::Null,
            AttrValue::Bool(b) => V::Bool(*b),
            AttrValue::Int(n) => V::Int(*n),
            AttrValue::UInt(n) => match i64::try_from(*n) {
                Ok(i) => V::Int(i),
                Err(_) => V::Float(*n as f64),
            },
            AttrValue::Float(f) => V::Float(*f),
            AttrValue::Str(s) => V::Str(s.clone()),
            AttrValue::Array(items) => {
                V::Seq(items.iter().map(V::from_attr).collect::<Vec<_>>().into())
            }
            AttrValue::Object(map) => V::Obj(Rc::new(
                map.iter()
                    .map(|(k, v)| (k.clone(), V::from_attr(v)))
                    .collect(),
            )),
        }
    }
}

/// Python `==` over the value domain — the numeric tower
/// (`True == 1 == 1.0`), structural sequences/objects, `NaN != NaN`.
pub(crate) fn v_eq(a: &V, b: &V) -> bool {
    use V::*;
    match (a, b) {
        (Null, Null) => true,
        (Bool(x), Bool(y)) => x == y,
        // The numeric tower: bool/int/float compare by value.
        (Bool(x), Int(n)) | (Int(n), Bool(x)) => (*x as i64) == *n,
        (Bool(x), Float(f)) | (Float(f), Bool(x)) => int_eq_float(*x as i64, *f),
        (Int(x), Int(y)) => x == y,
        (Int(n), Float(f)) | (Float(f), Int(n)) => int_eq_float(*n, *f),
        (Float(x), Float(y)) => x == y,
        (Str(x), Str(y)) => x == y,
        (Seq(x), Seq(y)) => x.len() == y.len() && x.iter().zip(y.iter()).all(|(p, q)| v_eq(p, q)),
        (Obj(x), Obj(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| v_eq(v, w)))
        }
        _ => false,
    }
}

/// `int == float` exactly — the `egraph::value` port: range-check
/// then compare, so `i64::MAX as f64` never equals `i64::MAX`.
fn int_eq_float(n: i64, f: f64) -> bool {
    const TWO63: f64 = 9_223_372_036_854_775_808.0;
    f.fract() == 0.0 && (-TWO63..TWO63).contains(&f) && (f as i64) == n
}

/// The numeric view of a value: `Some` for `Bool`/`Int`/`Float`
/// (bool is int in the Python tower), `None` otherwise.
fn num(v: &V) -> Option<Num> {
    match v {
        V::Bool(b) => Some(Num::Int(if *b { 1 } else { 0 })),
        V::Int(n) => Some(Num::Int(i128::from(*n))),
        V::Float(f) => Some(Num::Float(*f)),
        _ => None,
    }
}

/// A numeric operand — int-valued (`i128` for headroom) or float.
#[derive(Clone, Copy)]
enum Num {
    /// Integer (bool coerces to 0/1 upstream).
    Int(i128),
    /// Float.
    Float(f64),
}

impl Num {
    /// `as_f64` — Python's numeric coercion for `/`, `float()`, roots.
    fn f64(self) -> f64 {
        match self {
            Num::Int(n) => n as f64,
            Num::Float(f) => f,
        }
    }
}

/// Python ordering over the value domain — `None` where Python
/// raises `TypeError` (unorderable pair → the predicate declines).
fn v_cmp(a: &V, b: &V) -> Option<std::cmp::Ordering> {
    use V::*;
    match (a, b) {
        // Numeric tower: exact for int pairs, f64 otherwise.
        (x, y) if num(x).is_some() && num(y).is_some() => {
            match (num(x).unwrap(), num(y).unwrap()) {
                (Num::Int(p), Num::Int(q)) => Some(p.cmp(&q)),
                (p, q) => p.f64().partial_cmp(&q.f64()),
            }
        }
        (Str(x), Str(y)) => Some(x.cmp(y)),
        // Tuple ordering: first non-equal pair decides; a prefix
        // compares by length. An unorderable element pair propagates
        // the TypeError → decline.
        (Seq(x), Seq(y)) => {
            for (p, q) in x.iter().zip(y.iter()) {
                if !v_eq(p, q) {
                    return v_cmp(p, q);
                }
            }
            Some(x.len().cmp(&y.len()))
        }
        _ => None,
    }
}

/// The `_CMPS` table — `==`/`!=` never fail (Python `==` is total),
/// orderings decline on unorderable pairs.
fn cmp_vals(cmp: &str, a: &V, b: &V) -> bool {
    use std::cmp::Ordering::*;
    match cmp {
        "==" => v_eq(a, b),
        "!=" => !v_eq(a, b),
        "<" => matches!(v_cmp(a, b), Some(Less)),
        "<=" => matches!(v_cmp(a, b), Some(Less | Equal)),
        ">" => matches!(v_cmp(a, b), Some(Greater)),
        ">=" => matches!(v_cmp(a, b), Some(Greater | Equal)),
        _ => false, // unreachable — the parser validates CMPs
    }
}

/// `is_scalar` — `bool | int | float` (None is NOT scalar).
fn is_scalar(v: &V) -> bool {
    matches!(v, V::Bool(_) | V::Int(_) | V::Float(_))
}

// ---------------------------------------------------------------------------
//  Term references — the `bound.get(name)` positions
// ---------------------------------------------------------------------------

/// A term-metavariable reference in a cond/expr node. Python does
/// `bound.get(args[i])` — a non-string key simply misses (a list
/// would raise `TypeError`; declined here as unbound, the
/// conservative arm).
#[derive(Clone, Debug)]
pub enum TermRef {
    /// A metavar name looked up in [`BoundEnv::terms`].
    Name(Rc<str>),
    /// Anything else — always unbound.
    Unbound,
}

impl TermRef {
    /// Resolve against the bound environment.
    fn term<'a>(&self, env: &'a BoundEnv) -> Option<&'a Term> {
        match self {
            TermRef::Name(n) => env.terms.get(n.as_ref()),
            TermRef::Unbound => None,
        }
    }
}

/// The `.value` of a bound `Const` leaf — `getattr(t, "value",
/// None)`: only a `Term::Leaf` decoding to `KernelTerm::Const`
/// answers; every other term shape declines.
fn const_value(t: &Term) -> Option<V> {
    if !matches!(t, Term::Leaf(_)) {
        return None;
    }
    match to_kernel(t) {
        KernelTerm::Const { value } => Some(V::from_json(&value)),
        _ => None,
    }
}

/// Whether the bound term is a *kernel* `Op` — the bridge's
/// `source`-attr marker (structural encodings like `seq`/`first`
/// lower without it, exactly the `to_kernel` `sourced` test).
fn is_kernel_op(t: &Term) -> bool {
    match t {
        Term::Op(op) => op
            .attrs
            .iter()
            .any(|(k, v)| k.as_ref() == "source" && matches!(v, TermAttr::Lit(_))),
        _ => false,
    }
}

/// `_num_value` — the bound term's leaf `.value` when it is a Const.
fn num_value(env: &BoundEnv, r: &TermRef) -> Option<V> {
    r.term(env).and_then(const_value)
}

/// Whether `v` is Python-`isinstance(x, (int, float))` — bool is int.
fn is_number(v: &V) -> bool {
    matches!(v, V::Bool(_) | V::Int(_) | V::Float(_))
}

// ---------------------------------------------------------------------------
//  Attribute-value resolution — `_aval`
// ---------------------------------------------------------------------------

/// An attr-position argument: attr-metavar name, inline derive expr,
/// or literal — `_aval`'s three spellings.
#[derive(Clone, Debug)]
pub enum Aval {
    /// `"NAME"` — `bound["$attr:NAME"]` (bare key in `Subst::attrs`).
    Name(Rc<str>),
    /// A tuple/list node — an inline derive expr, evaluated against
    /// the same `bound`; a decline surfaces as missing.
    Expr(Box<Expr>),
    /// Any other JSON value — the literal itself.
    Lit(V),
}

impl Aval {
    /// `_aval` — resolve to a value; `None` is `_MISSING`.
    fn eval(&self, env: &BoundEnv) -> Option<V> {
        match self {
            Aval::Name(n) => env.attrs.get(n.as_ref()).map(V::from_attr),
            Aval::Expr(e) => e.eval(env),
            Aval::Lit(v) => Some(v.clone()),
        }
    }
}

// ---------------------------------------------------------------------------
//  The cond AST
// ---------------------------------------------------------------------------

/// The six `_CMPS` spellings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp {
    /// `==`.
    Eq,
    /// `!=`.
    Ne,
    /// `<`.
    Lt,
    /// `<=`.
    Le,
    /// `>`.
    Gt,
    /// `>=`.
    Ge,
}

impl Cmp {
    /// The `_CMPS` key spelling.
    fn name(self) -> &'static str {
        match self {
            Cmp::Eq => "==",
            Cmp::Ne => "!=",
            Cmp::Lt => "<",
            Cmp::Le => "<=",
            Cmp::Gt => ">",
            Cmp::Ge => ">=",
        }
    }

    /// Parse a CMP literal; `None` on a bad spelling (malformed).
    fn parse(v: &Value) -> Option<Cmp> {
        match v.as_str()? {
            "==" => Some(Cmp::Eq),
            "!=" => Some(Cmp::Ne),
            "<" => Some(Cmp::Lt),
            "<=" => Some(Cmp::Le),
            ">" => Some(Cmp::Gt),
            ">=" => Some(Cmp::Ge),
            _ => None,
        }
    }

    /// Apply to two values (see [`cmp_vals`]).
    fn apply(self, a: &V, b: &V) -> bool {
        cmp_vals(self.name(), a, b)
    }
}

/// The `_ATTR_TYPES` keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttrKind {
    /// `int` — `isinstance(x, int)` (bool admitted).
    Int,
    /// `float`.
    Float,
    /// `number` — `int | float` (bool admitted).
    Number,
    /// `bool`.
    Bool,
    /// `str`.
    Str,
    /// `tuple`.
    Tuple,
}

impl AttrKind {
    /// Parse an `attr-type` kind literal.
    fn parse(v: &Value) -> Option<AttrKind> {
        match v.as_str()? {
            "int" => Some(AttrKind::Int),
            "float" => Some(AttrKind::Float),
            "number" => Some(AttrKind::Number),
            "bool" => Some(AttrKind::Bool),
            "str" => Some(AttrKind::Str),
            "tuple" => Some(AttrKind::Tuple),
            _ => None,
        }
    }

    /// `isinstance(v, _ATTR_TYPES[k])`.
    fn holds(self, v: &V) -> bool {
        match self {
            // isinstance(True, int) is True — bool rides the int arm.
            AttrKind::Int => matches!(v, V::Int(_) | V::Bool(_)),
            AttrKind::Float => matches!(v, V::Float(_)),
            AttrKind::Number => is_number(v),
            AttrKind::Bool => matches!(v, V::Bool(_)),
            AttrKind::Str => matches!(v, V::Str(_)),
            AttrKind::Tuple => matches!(v, V::Seq(_)),
        }
    }
}

/// A parsed `cond` tree — `eval_cond`'s total predicate.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)] // the shape predicates embed big specs; boxing churns every eval site
pub enum Cond {
    /// A bare bool literal.
    Bool(bool),
    /// `("and", c…)`.
    And(Vec<Cond>),
    /// `("or", c…)`.
    Or(Vec<Cond>),
    /// `("not", c)`.
    Not(Box<Cond>),
    /// A stage-1 predicate (see [`Pred`]).
    Pred(Pred),
}

/// The stage-1 predicate set — every `_OPS` entry that reads no
/// shape. Shape-reading ops parse as [`GuardError::Unsupported`].
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)] // `Shape(ShapePred)` carries the whole predicate tree
pub enum Pred {
    /// `("op-in", T, ops)` — kernel-Op head in the literal set.
    OpIn(TermRef, V),
    /// `("leaf", T)` — bound term is not a kernel `Op`.
    Leaf(TermRef),
    /// `("const", T)` — bound term is a `Const` leaf.
    Const(TermRef),
    /// `("term-eq", A, B)` — bound terms equal (`None == None`).
    TermEq(TermRef, TermRef),
    /// `("const-num", T)` — `.value` is int|float.
    ConstNum(TermRef),
    /// `("const-cmp", T, CMP, v)` — numeric `.value` CMP literal.
    ConstCmp(TermRef, Cmp, V),
    /// `("attr-is", A, v)` — `_aval(A) is v` (identity approx).
    AttrIs(Aval, V),
    /// `("attr-eq", A, v)` — `_aval(A) == v`, missing reads `None`.
    AttrEq(Aval, V),
    /// `("attr-in", A, vs)` — `_aval(A) in vs`.
    AttrIn(Aval, V),
    /// `("attr-type", A, K)` — `isinstance` per kind.
    AttrType(Aval, AttrKind),
    /// `("attr-len", A, CMP, k)` — tuple attr length CMP k.
    AttrLen(Aval, Cmp, V),
    /// `("attr-eq-attr", A, B)` — two resolved attrs equal.
    AttrEqAttr(Aval, Aval),
    /// `("attr-cmp", A, CMP, B)` — resolved comparison.
    AttrCmp(Aval, Cmp, Aval),
    /// `("attr-ints", A)` — tuple/list of ints (bool admitted).
    AttrInts(Aval),
    /// A shape-reading predicate — the stage-2/3 `_OPS` subset ported
    /// over [`crate::shape`] (see [`ShapePred`]).
    Shape(ShapePred),
}

/// A parsed shape-reading predicate — the `cond.py` `_OPS` entries
/// that read [`crate::shape`]-inferred shapes, with arguments already
/// resolved to [`crate::shape::ShapeSpec`]/[`crate::shape::AttrArg`]
/// data. Evaluation delegates to [`crate::shape::pred`].
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)] // spec-carrying variants dominate; boxing is not worth it here
pub enum ShapePred {
    /// `("shaped", T)`.
    Shaped(crate::shape::ShapeSpec),
    /// `("concrete", T)`.
    Concrete(crate::shape::ShapeSpec),
    /// `("scalar", T)`.
    Scalar(crate::shape::ShapeSpec),
    /// `("uniform", T)`.
    Uniform(crate::shape::ShapeSpec),
    /// `("ones-but-last", T)`.
    OnesButLast(crate::shape::ShapeSpec),
    /// `("rank", T, CMP, k)`.
    Rank(crate::shape::ShapeSpec, Cmp, i64),
    /// `("rank-eq", A, B)`.
    RankEq(crate::shape::ShapeSpec, crate::shape::ShapeSpec),
    /// `("shape-eq", A, B)`.
    ShapeEq(crate::shape::ShapeSpec, crate::shape::ShapeSpec),
    /// `("shape-compat", A, B)`.
    ShapeCompat(crate::shape::ShapeSpec, crate::shape::ShapeSpec),
    /// `("dim-eq", A, i, B, j)`.
    DimEq(crate::shape::ShapeSpec, i64, crate::shape::ShapeSpec, i64),
    /// `("dim-compat", A, i, B, j)`.
    DimCompat(crate::shape::ShapeSpec, i64, crate::shape::ShapeSpec, i64),
    /// `("dim-eq-const", T, i, k)`.
    DimEqConst(crate::shape::ShapeSpec, i64, i64),
    /// `("dim-eq-attr", A, K, B, K2)`.
    DimEqAttr(
        crate::shape::ShapeSpec,
        crate::shape::AttrArg,
        crate::shape::ShapeSpec,
        crate::shape::AttrArg,
    ),
    /// `("dim-mod", T, K, m, r)`.
    DimMod(crate::shape::ShapeSpec, crate::shape::AttrArg, i64, i64),
    /// `("bcast-into", T, U)`.
    BcastInto(crate::shape::ShapeSpec, crate::shape::ShapeSpec),
    /// `("mm-shape-ok", A, B)`.
    MmShapeOk(crate::shape::ShapeSpec, crate::shape::ShapeSpec),
    /// `("bcast-eq", A, B, C, D)`.
    BcastEq(
        crate::shape::ShapeSpec,
        crate::shape::ShapeSpec,
        crate::shape::ShapeSpec,
        crate::shape::ShapeSpec,
    ),
    /// `("ones-before", T, K)`.
    OnesBefore(crate::shape::ShapeSpec, crate::shape::AttrArg),
    /// `("axes-last2", T, D0, D1)`.
    AxesLast2(
        crate::shape::ShapeSpec,
        crate::shape::AttrArg,
        crate::shape::AttrArg,
    ),
    /// `("axes-distinct", T, D0, D1)`.
    AxesDistinct(
        crate::shape::ShapeSpec,
        crate::shape::AttrArg,
        crate::shape::AttrArg,
    ),
    /// `("axes-eq", T, A0, A1, B0, B1)`.
    AxesEq(
        crate::shape::ShapeSpec,
        crate::shape::AttrArg,
        crate::shape::AttrArg,
        crate::shape::AttrArg,
        crate::shape::AttrArg,
    ),
    /// `("axes-noop", T, D0, D1)`.
    AxesNoop(
        crate::shape::ShapeSpec,
        crate::shape::AttrArg,
        crate::shape::AttrArg,
    ),
    /// `("axis", T, NAME, k)`.
    Axis(crate::shape::ShapeSpec, crate::shape::AttrArg, i64),
    /// `("attr-cmp-dim", NAME, CMP, T, K)`.
    AttrCmpDim(
        crate::shape::AttrArg,
        Cmp,
        crate::shape::ShapeSpec,
        crate::shape::AttrArg,
    ),
    /// `("attr-range", V, T, K)`.
    AttrRange(
        crate::shape::AttrArg,
        crate::shape::ShapeSpec,
        crate::shape::AttrArg,
    ),
}

impl ShapePred {
    /// Evaluate one shape-reading predicate through
    /// [`crate::shape::pred`] — every unproven case declines to
    /// `false` there.
    fn eval(&self, env: &BoundEnv) -> bool {
        use crate::shape::pred as sp;
        match self {
            ShapePred::Shaped(t) => sp::shaped(env, t),
            ShapePred::Concrete(t) => sp::concrete(env, t),
            ShapePred::Scalar(t) => sp::scalar(env, t),
            ShapePred::Uniform(t) => sp::uniform(env, t),
            ShapePred::OnesButLast(t) => sp::ones_but_last(env, t),
            ShapePred::Rank(t, cmp, k) => sp::rank(env, t, cmp.name(), *k),
            ShapePred::RankEq(a, b) => sp::rank_eq(env, a, b),
            ShapePred::ShapeEq(a, b) => sp::shape_eq(env, a, b),
            ShapePred::ShapeCompat(a, b) => sp::shape_compat(env, a, b),
            ShapePred::DimEq(a, i, b, j) => sp::dim_eq(env, a, *i, b, *j),
            ShapePred::DimCompat(a, i, b, j) => sp::dim_compat(env, a, *i, b, *j),
            ShapePred::DimEqConst(t, i, k) => sp::dim_eq_const(env, t, *i, *k),
            ShapePred::DimEqAttr(a, ka, b, kb) => sp::dim_eq_attr(env, a, ka, b, kb),
            ShapePred::DimMod(t, k, m, r) => sp::dim_mod(env, t, k, *m, *r),
            ShapePred::BcastInto(t, u) => sp::bcast_into(env, t, u),
            ShapePred::MmShapeOk(a, b) => sp::mm_shape_ok(env, a, b),
            ShapePred::BcastEq(a, b, c, d) => sp::bcast_eq(env, a, b, c, d),
            ShapePred::OnesBefore(t, k) => sp::ones_before(env, t, k),
            ShapePred::AxesLast2(t, d0, d1) => sp::axes_last2(env, t, d0, d1),
            ShapePred::AxesDistinct(t, d0, d1) => sp::axes_distinct(env, t, d0, d1),
            ShapePred::AxesEq(t, a0, a1, b0, b1) => sp::axes_eq(env, t, a0, a1, b0, b1),
            ShapePred::AxesNoop(t, d0, d1) => sp::axes_noop(env, t, d0, d1),
            ShapePred::Axis(t, name, k) => sp::axis(env, t, name, *k),
            ShapePred::AttrCmpDim(name, cmp, t, k) => sp::attr_cmp_dim(env, name, cmp.name(), t, k),
            ShapePred::AttrRange(v, t, k) => sp::attr_range(env, v, t, k),
        }
    }
}

/// `attr-is`'s Python `is` — singletons and the small-int cache by
/// value; everything else `false` (distinct objects).
fn aval_is(v: Option<&V>, lit: &V) -> bool {
    let v = match v {
        // absent reads as None — the optional-attr contract
        None => &V::Null,
        Some(v) => v,
    };
    match (v, lit) {
        (V::Null, V::Null) => true,
        (V::Bool(a), V::Bool(b)) => a == b,
        (V::Int(n), V::Int(m)) => (-5..=256).contains(n) && n == m,
        _ => false,
    }
}

/// `v in vs` — Python membership: sequence elementwise `==`, string
/// substring, mapping key-membership; anything else is a `TypeError`
/// Python would raise — declined (`false`) here.
fn v_contains(hay: &V, needle: &V) -> bool {
    match hay {
        V::Seq(items) => items.iter().any(|x| v_eq(x, needle)),
        V::Str(s) => match needle {
            V::Str(n) => s.contains(n.as_ref()),
            _ => false,
        },
        V::Obj(map) => match needle {
            V::Str(n) => map.contains_key(n.as_ref()),
            _ => false,
        },
        _ => false,
    }
}

impl Cond {
    /// Parse a `cond` tree from its record JSON (the
    /// `cond_from_data` inverse — lists are already canonical).
    ///
    /// # Errors
    ///
    /// [`GuardError::Malformed`] on a bad node shape/arity/op-name;
    /// [`GuardError::Unsupported`] on an unported guard op.
    pub fn from_data(value: &Value) -> Result<Cond, GuardError> {
        match value {
            Value::Bool(b) => Ok(Cond::Bool(*b)),
            Value::Array(items) if !items.is_empty() => {
                let op = items[0]
                    .as_str()
                    .ok_or_else(|| GuardError::Malformed(format!("{value}")))?;
                match op {
                    "and" => Ok(Cond::And(
                        items[1..]
                            .iter()
                            .map(Cond::from_data)
                            .collect::<Result<_, _>>()?,
                    )),
                    "or" => Ok(Cond::Or(
                        items[1..]
                            .iter()
                            .map(Cond::from_data)
                            .collect::<Result<_, _>>()?,
                    )),
                    "not" => {
                        if items.len() != 2 {
                            return Err(GuardError::Malformed(format!(
                                "malformed not node: {value}"
                            )));
                        }
                        Ok(Cond::Not(Box::new(Cond::from_data(&items[1])?)))
                    }
                    _ => pred_from_data(op, &items[1..]).map(Cond::Pred),
                }
            }
            _ => Err(GuardError::Malformed(format!("{value}"))),
        }
    }

    /// `eval_cond` — the total predicate over a [`BoundEnv`].
    pub fn eval(&self, env: &BoundEnv) -> bool {
        match self {
            Cond::Bool(b) => *b,
            Cond::And(cs) => cs.iter().all(|c| c.eval(env)),
            Cond::Or(cs) => cs.iter().any(|c| c.eval(env)),
            Cond::Not(c) => !c.eval(env),
            Cond::Pred(p) => p.eval(env),
        }
    }
}

impl Pred {
    /// Evaluate one stage-1 predicate.
    fn eval(&self, env: &BoundEnv) -> bool {
        match self {
            Pred::OpIn(t, ops) => {
                let Some(term) = t.term(env) else {
                    return false;
                };
                is_kernel_op(term)
                    && match term {
                        Term::Op(op) => v_contains(ops, &V::Str(op.op.clone())),
                        _ => false,
                    }
            }
            Pred::Leaf(t) => t.term(env).is_none_or(|t| !is_kernel_op(t)),
            Pred::Const(t) => t.term(env).is_some_and(|t| const_value(t).is_some()),
            Pred::TermEq(a, b) => match (a.term(env), b.term(env)) {
                (None, None) => true, // bound.get miss == bound.get miss
                (Some(x), Some(y)) => x == y,
                _ => false,
            },
            Pred::ConstNum(t) => num_value(env, t).is_some_and(|v| is_number(&v)),
            Pred::ConstCmp(t, cmp, lit) => num_value(env, t)
                .filter(is_number)
                .is_some_and(|v| cmp.apply(&v, lit)),
            Pred::AttrIs(a, lit) => aval_is(a.eval(env).as_ref(), lit),
            Pred::AttrEq(a, lit) => {
                // absent reads as None — the optional-attr contract
                let v = a.eval(env).unwrap_or(V::Null);
                v_eq(&v, lit)
            }
            Pred::AttrIn(a, vs) => {
                let v = a.eval(env).unwrap_or(V::Null);
                v_contains(vs, &v)
            }
            Pred::AttrType(a, k) => a.eval(env).is_some_and(|v| k.holds(&v)),
            Pred::AttrLen(a, cmp, k) => match a.eval(env) {
                Some(V::Seq(items)) => cmp.apply(&V::Int(items.len() as i64), k),
                _ => false,
            },
            Pred::AttrEqAttr(a, b) => match (a.eval(env), b.eval(env)) {
                (Some(x), Some(y)) => v_eq(&x, &y),
                _ => false,
            },
            Pred::AttrCmp(a, cmp, b) => match (a.eval(env), b.eval(env)) {
                (Some(x), Some(y)) => cmp.apply(&x, &y),
                _ => false,
            },
            Pred::AttrInts(a) => match a.eval(env) {
                Some(V::Seq(items)) => items.iter().all(|x| matches!(x, V::Int(_) | V::Bool(_))),
                _ => false,
            },
            Pred::Shape(sp) => sp.eval(env),
        }
    }
}

// ---------------------------------------------------------------------------
//  The dspec AST — `eval_derive`'s expr algebra
// ---------------------------------------------------------------------------

/// A parsed derive expr — `_dexpr`'s node set minus the
/// shape/tensor channels (not ported — named refusals).
#[derive(Clone, Debug)]
pub enum Expr {
    /// A bare `null`/`bool`/number literal.
    Lit(V),
    /// `("lit", v)` — an explicit literal of any JSON shape.
    LitWrap(V),
    /// `("attr", x)` — `_aval(x)`; missing declines.
    Attr(Aval),
    /// `("attr0", x)` — `v[0]` for a seq, else `v`.
    Attr0(Aval),
    /// `("const", T)` — the bound leaf's `.value`.
    Const(TermRef),
    /// `("len", e)` — length of seq/str.
    Len(Box<Expr>),
    /// `("tuple", e…)` — tuple literal.
    Tuple(Vec<Expr>),
    /// `("concat", e…)` — sequence concatenation.
    Concat(Vec<Expr>),
    /// `("gather", table, sel)` — index-of-index.
    Gather(Box<Expr>, Box<Expr>),
    /// `("unique", e)` — first-occurrence uniques.
    Unique(Box<Expr>),
    /// `("posmap", dom, keys)` — `dom.index(k)` per key.
    Posmap(Box<Expr>, Box<Expr>),
    /// `("nth", seq, i)` — `seq[i]`, Python negative reach.
    Nth(Box<Expr>, Box<Expr>),
    /// `("take", seq, k)` — `seq[:k]`.
    Take(Box<Expr>, Box<Expr>),
    /// `("sum", e)` — sum of a numeric sequence.
    Sum(Box<Expr>),
    /// `("guard", COND, e)` — veto unless COND holds.
    Guard(Box<Cond>, Box<Expr>),
    /// `("if", COND, t, f)` — branch on a cond tree.
    If(Box<Cond>, Box<Expr>, Box<Expr>),
    /// `("max"|"min", e, e)` — int-only selection.
    Max(Box<Expr>, Box<Expr>),
    /// `("min", e, e)`.
    Min(Box<Expr>, Box<Expr>),
    /// `("pow", e, e)` — int ** int, exp >= 0, bit-budgeted.
    Pow(Box<Expr>, Box<Expr>),
    /// `("mod", e, e)` — int %, nonzero divisor.
    Mod(Box<Expr>, Box<Expr>),
    /// `("neg", e)`.
    Neg(Box<Expr>),
    /// `("recip", e)` — `1.0 / a`.
    Recip(Box<Expr>),
    /// `("float", e)`.
    ToFloat(Box<Expr>),
    /// `("int", e)`.
    ToInt(Box<Expr>),
    /// `("sqrt", e)`.
    Sqrt(Box<Expr>),
    /// `("rsqrt", e)` — `1.0 / sqrt(a)`.
    Rsqrt(Box<Expr>),
    /// `("add"|"sub"|"mul"|"fdiv"|"floordiv", e, e)`.
    Bin(BinOp, Box<Expr>, Box<Expr>),
}

/// The `_DBINOPS`/`_DUNOPS` arithmetic ops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    /// `add` — `+`.
    Add,
    /// `sub` — `-`.
    Sub,
    /// `mul` — `*`.
    Mul,
    /// `fdiv` — `/` (always float).
    Fdiv,
    /// `floordiv` — `//`.
    Floordiv,
}

/// Result bit budget for `pow` — cond.py's `_POW_MAX_BITS`
/// (`1 << 16` bits); checked `i128` math bounds it tighter still.
const POW_MAX_BITS: u64 = 1 << 16;

/// Python floor division on `i128` (`a // b`, sign of the divisor).
fn py_floordiv(a: i128, b: i128) -> Option<i128> {
    if b == 0 {
        return None;
    }
    let q = a / b;
    let r = a % b;
    Some(if r != 0 && ((r < 0) != (b < 0)) {
        q - 1
    } else {
        q
    })
}

/// Python `%` on `i128` — floored, sign of the divisor.
fn py_mod(a: i128, b: i128) -> Option<i128> {
    if b == 0 {
        return None;
    }
    Some(a - b * py_floordiv(a, b)?)
}

/// Narrow an `i128` to `V::Int`; overflow declines (Python ints are
/// unbounded — see the module docs).
fn int_v(n: i128) -> Option<V> {
    i64::try_from(n).ok().map(V::Int)
}

/// Hashability for `unique`/`posmap` element compares — Python
/// `dict.fromkeys`/`list.index` need hashable elements; a map is
/// not, and a sequence is iff its elements are.
fn hashable(v: &V) -> bool {
    match v {
        V::Obj(_) => false,
        V::Seq(items) => items.iter().all(hashable),
        _ => true,
    }
}

impl Expr {
    /// Parse one derive expr node. Bare `null`/`bool`/number are
    /// literals; a bare string/object is malformed (cond.py's
    /// `ValueError` arm).
    ///
    /// # Errors
    ///
    /// As [`Cond::from_data`].
    pub fn from_data(value: &Value) -> Result<Expr, GuardError> {
        match value {
            Value::Null | Value::Bool(_) | Value::Number(_) => Ok(Expr::Lit(V::from_json(value))),
            Value::String(_) | Value::Object(_) => Err(GuardError::Malformed(format!(
                "malformed derive expr: {value}"
            ))),
            Value::Array(items) => {
                let op = items
                    .first()
                    .and_then(Value::as_str)
                    .ok_or_else(|| GuardError::Malformed(format!("{value}")))?;
                let args = &items[1..];
                let need = |n: usize| -> Result<(), GuardError> {
                    if args.len() < n {
                        Err(GuardError::Malformed(format!("{value}")))
                    } else {
                        Ok(())
                    }
                };
                // Stage-3 channels — named refusals, not approximations.
                if matches!(
                    op,
                    "shape" | "vshape" | "dim" | "leaf-dim" | "bcast" | "tval" | "tapp"
                ) {
                    return Err(GuardError::Unsupported(format!(
                        "{op}: not ported — the dspec shape/tensor channels need a tensor backend"
                    )));
                }
                match op {
                    "lit" => {
                        need(1)?;
                        Ok(Expr::LitWrap(V::from_json(&args[0])))
                    }
                    "attr" => {
                        need(1)?;
                        Ok(Expr::Attr(aval_from_data(&args[0])?))
                    }
                    "attr0" => {
                        need(1)?;
                        Ok(Expr::Attr0(aval_from_data(&args[0])?))
                    }
                    "const" => {
                        need(1)?;
                        Ok(Expr::Const(term_ref_from_data(&args[0])))
                    }
                    "len" => {
                        need(1)?;
                        Ok(Expr::Len(Box::new(Expr::from_data(&args[0])?)))
                    }
                    "tuple" => Ok(Expr::Tuple(
                        args.iter().map(Expr::from_data).collect::<Result<_, _>>()?,
                    )),
                    "concat" => Ok(Expr::Concat(
                        args.iter().map(Expr::from_data).collect::<Result<_, _>>()?,
                    )),
                    "gather" => {
                        need(2)?;
                        Ok(Expr::Gather(
                            Box::new(Expr::from_data(&args[0])?),
                            Box::new(Expr::from_data(&args[1])?),
                        ))
                    }
                    "unique" => {
                        need(1)?;
                        Ok(Expr::Unique(Box::new(Expr::from_data(&args[0])?)))
                    }
                    "posmap" => {
                        need(2)?;
                        Ok(Expr::Posmap(
                            Box::new(Expr::from_data(&args[0])?),
                            Box::new(Expr::from_data(&args[1])?),
                        ))
                    }
                    "nth" => {
                        need(2)?;
                        Ok(Expr::Nth(
                            Box::new(Expr::from_data(&args[0])?),
                            Box::new(Expr::from_data(&args[1])?),
                        ))
                    }
                    "take" => {
                        need(2)?;
                        Ok(Expr::Take(
                            Box::new(Expr::from_data(&args[0])?),
                            Box::new(Expr::from_data(&args[1])?),
                        ))
                    }
                    "sum" => {
                        need(1)?;
                        Ok(Expr::Sum(Box::new(Expr::from_data(&args[0])?)))
                    }
                    "guard" => {
                        need(2)?;
                        Ok(Expr::Guard(
                            Box::new(Cond::from_data(&args[0])?),
                            Box::new(Expr::from_data(&args[1])?),
                        ))
                    }
                    "if" => {
                        need(3)?;
                        Ok(Expr::If(
                            Box::new(Cond::from_data(&args[0])?),
                            Box::new(Expr::from_data(&args[1])?),
                            Box::new(Expr::from_data(&args[2])?),
                        ))
                    }
                    "max" | "min" | "pow" | "mod" | "add" | "sub" | "mul" | "fdiv" | "floordiv" => {
                        need(2)?;
                        let a = Box::new(Expr::from_data(&args[0])?);
                        let b = Box::new(Expr::from_data(&args[1])?);
                        Ok(match op {
                            "max" => Expr::Max(a, b),
                            "min" => Expr::Min(a, b),
                            "pow" => Expr::Pow(a, b),
                            "mod" => Expr::Mod(a, b),
                            "add" => Expr::Bin(BinOp::Add, a, b),
                            "sub" => Expr::Bin(BinOp::Sub, a, b),
                            "mul" => Expr::Bin(BinOp::Mul, a, b),
                            "fdiv" => Expr::Bin(BinOp::Fdiv, a, b),
                            _ => Expr::Bin(BinOp::Floordiv, a, b),
                        })
                    }
                    "neg" | "recip" | "float" | "int" | "sqrt" | "rsqrt" => {
                        need(1)?;
                        let a = Box::new(Expr::from_data(&args[0])?);
                        Ok(match op {
                            "neg" => Expr::Neg(a),
                            "recip" => Expr::Recip(a),
                            "float" => Expr::ToFloat(a),
                            "int" => Expr::ToInt(a),
                            "sqrt" => Expr::Sqrt(a),
                            _ => Expr::Rsqrt(a),
                        })
                    }
                    _ => Err(GuardError::Malformed(format!("unknown derive op: {op:?}"))),
                }
            }
        }
    }

    /// `_dexpr` — evaluate to a value; `None` is `_Decline` (the
    /// whole derive vetoes).
    pub fn eval(&self, env: &BoundEnv) -> Option<V> {
        match self {
            Expr::Lit(v) | Expr::LitWrap(v) => Some(v.clone()),
            Expr::Attr(a) => a.eval(env),
            Expr::Attr0(a) => match a.eval(env)? {
                V::Seq(items) => {
                    if items.is_empty() {
                        None
                    } else {
                        Some(items[0].clone())
                    }
                }
                v => Some(v),
            },
            Expr::Const(t) => num_value(env, t),
            Expr::Len(e) => match e.eval(env)? {
                V::Seq(items) => Some(V::Int(items.len() as i64)),
                V::Str(s) => Some(V::Int(s.chars().count() as i64)),
                _ => None,
            },
            Expr::Tuple(es) => Some(V::Seq(
                es.iter()
                    .map(|e| e.eval(env))
                    .collect::<Option<Vec<_>>>()?
                    .into(),
            )),
            Expr::Concat(es) => {
                let mut out = Vec::new();
                for e in es {
                    match e.eval(env)? {
                        V::Seq(items) => out.extend(items.iter().cloned()),
                        _ => return None,
                    }
                }
                Some(V::Seq(out.into()))
            }
            Expr::Gather(table, sel) => {
                let table = table.eval(env)?;
                let sel = sel.eval(env)?;
                let (V::Seq(table), V::Seq(sel)) = (&table, &sel) else {
                    return None;
                };
                // Selectors: non-bool ints in [0, len(table)).
                let mut idx = Vec::with_capacity(sel.len());
                for j in sel.iter() {
                    let V::Int(n) = j else {
                        return None;
                    };
                    if *n < 0 || *n as usize >= table.len() {
                        return None;
                    }
                    idx.push(*n as usize);
                }
                Some(V::Seq(
                    idx.iter()
                        .map(|&j| table[j].clone())
                        .collect::<Vec<_>>()
                        .into(),
                ))
            }
            Expr::Unique(e) => {
                let V::Seq(items) = e.eval(env)? else {
                    return None;
                };
                if !items.iter().all(hashable) {
                    return None;
                }
                // dict.fromkeys — first-occurrence uniques under ==.
                let mut out: Vec<V> = Vec::with_capacity(items.len());
                'outer: for item in items.iter() {
                    for kept in out.iter() {
                        if v_eq(kept, item) {
                            continue 'outer;
                        }
                    }
                    out.push(item.clone());
                }
                Some(V::Seq(out.into()))
            }
            Expr::Posmap(dom, keys) => {
                let (V::Seq(dom), V::Seq(keys)) = (dom.eval(env)?, keys.eval(env)?) else {
                    return None;
                };
                let mut out = Vec::with_capacity(keys.len());
                for k in keys.iter() {
                    let pos = dom.iter().position(|d| v_eq(d, k))?;
                    out.push(V::Int(pos as i64));
                }
                Some(V::Seq(out.into()))
            }
            Expr::Nth(seq, i) => {
                let (V::Seq(seq), i) = (seq.eval(env)?, i.eval(env)?) else {
                    return None;
                };
                // isinstance(i, int) — bool admitted; negative reach.
                let n = match i {
                    V::Int(n) => n,
                    V::Bool(b) => i64::from(b),
                    _ => return None,
                };
                let len = seq.len() as i64;
                if !(-len..len).contains(&n) {
                    return None;
                }
                let idx = if n < 0 { len + n } else { n } as usize;
                Some(seq[idx].clone())
            }
            Expr::Take(seq, k) => {
                let (V::Seq(seq), k) = (seq.eval(env)?, k.eval(env)?) else {
                    return None;
                };
                let n = match k {
                    V::Int(n) => n,
                    V::Bool(b) => i64::from(b),
                    _ => return None,
                };
                // Python seq[:k] — negative k counts from the end.
                let len = seq.len() as i64;
                let end = if n < 0 { (len + n).max(0) } else { n.min(len) } as usize;
                Some(V::Seq(seq[..end].to_vec().into()))
            }
            Expr::Sum(e) => {
                let V::Seq(items) = e.eval(env)? else {
                    return None;
                };
                // sum() starts at int 0; a float operand promotes
                // the accumulator (and stays float).
                let mut acc = Num::Int(0i128);
                let mut is_float = false;
                for item in items.iter() {
                    let n = num(item)?;
                    if matches!(n, Num::Float(_)) {
                        is_float = true;
                    }
                    acc = match (acc, n) {
                        (Num::Int(a), Num::Int(b)) => Num::Int(a + b),
                        (a, b) => Num::Float(a.f64() + b.f64()),
                    };
                }
                match acc {
                    Num::Int(n) if !is_float => int_v(n),
                    Num::Int(n) => Some(V::Float(n as f64)),
                    Num::Float(f) => Some(V::Float(f)),
                }
            }
            Expr::Guard(c, e) => {
                if !c.eval(env) {
                    return None;
                }
                e.eval(env)
            }
            Expr::If(c, t, f) => {
                if c.eval(env) {
                    t.eval(env)
                } else {
                    f.eval(env)
                }
            }
            Expr::Max(a, b) | Expr::Min(a, b) => {
                let is_max = matches!(self, Expr::Max(..));
                let (a, b) = (a.eval(env)?, b.eval(env)?);
                // int-only selection — `isinstance(x, int)` admits
                // bool, declines float (num() maps both to Num::Int
                // only for int-typed values).
                let (Num::Int(x), Num::Int(y)) = (num(&a)?, num(&b)?) else {
                    return None;
                };
                int_v(if is_max { x.max(y) } else { x.min(y) })
            }
            Expr::Pow(a, b) => {
                let (a, b) = (a.eval(env)?, b.eval(env)?);
                let (Num::Int(base), Num::Int(exp)) = (num(&a)?, num(&b)?) else {
                    return None;
                };
                if exp < 0 {
                    return None;
                }
                // cond.py's bit budget: exp * bit_length(base) must
                // stay under _POW_MAX_BITS — i128 arithmetic keeps
                // the product exact for any i64 exponent.
                let bits = i128::from(128 - base.unsigned_abs().leading_zeros());
                if !matches!(base, -1..=1) && exp * bits > i128::from(POW_MAX_BITS) {
                    return None;
                }
                // ±1/0 bases are budget-exempt but still need exact
                // handling for exponents past u32 — compute directly.
                let v = match base {
                    0 => {
                        if exp == 0 {
                            V::Int(1) // Python 0**0 == 1
                        } else {
                            V::Int(0)
                        }
                    }
                    1 => V::Int(1),
                    -1 => {
                        if exp % 2 == 0 {
                            V::Int(1)
                        } else {
                            V::Int(-1)
                        }
                    }
                    _ => {
                        // Budget bounds exp <= POW_MAX_BITS here.
                        let e = u32::try_from(exp).ok()?;
                        int_v(base.checked_pow(e)?)?
                    }
                };
                Some(v)
            }
            Expr::Mod(a, b) => {
                let (a, b) = (a.eval(env)?, b.eval(env)?);
                let (Num::Int(x), Num::Int(y)) = (num(&a)?, num(&b)?) else {
                    return None;
                };
                int_v(py_mod(x, y)?)
            }
            Expr::Neg(a) => unop_scalar(a.eval(env)?, |n| match n {
                Num::Int(x) => int_v(-x),
                Num::Float(f) => Some(V::Float(-f)),
            }),
            Expr::Recip(a) => {
                let v = a.eval(env)?;
                let n = scalar_num(&v)?;
                if n.f64() == 0.0 {
                    return None; // 1.0/0 — ZeroDivisionError → decline
                }
                Some(V::Float(1.0 / n.f64()))
            }
            Expr::ToFloat(a) => {
                let v = a.eval(env)?;
                match v {
                    V::Str(s) => parse_float(&s).map(V::Float),
                    _ => scalar_num(&v).map(|n| V::Float(n.f64())),
                }
            }
            Expr::ToInt(a) => {
                let v = a.eval(env)?;
                match v {
                    V::Str(s) => parse_int(&s).and_then(int_v),
                    _ => {
                        let n = scalar_num(&v)?;
                        match n {
                            Num::Int(x) => int_v(x),
                            // int() truncates toward zero.
                            Num::Float(f) if f.is_finite() => {
                                let t = f.trunc();
                                // i64::MAX as f64 rounds up to 2^63 —
                                // bound strictly below it.
                                const TWO63: f64 = 9_223_372_036_854_775_808.0;
                                if (-TWO63..TWO63).contains(&t) {
                                    Some(V::Int(t as i64))
                                } else {
                                    None
                                }
                            }
                            Num::Float(_) => None, // nan/inf → ValueError/OverflowError
                        }
                    }
                }
            }
            Expr::Sqrt(a) => {
                let v = a.eval(env)?;
                let n = scalar_num(&v)?;
                if n.f64() < 0.0 {
                    return None; // math.sqrt domain error → decline
                }
                Some(V::Float(n.f64().sqrt()))
            }
            Expr::Rsqrt(a) => {
                let v = a.eval(env)?;
                let n = scalar_num(&v)?;
                if n.f64() <= 0.0 {
                    return None; // sqrt domain / 1.0÷0 → decline
                }
                Some(V::Float(1.0 / n.f64().sqrt()))
            }
            Expr::Bin(op, a, b) => binop(*op, a.eval(env)?, b.eval(env)?),
        }
    }
}

/// The scalar operand of a unop — `None` on non-scalars (the
/// tensor-fallback arm has no analogue here: a non-scalar that is
/// not `Str` declines, matching `_tcall` on the Python operators).
fn scalar_num(v: &V) -> Option<Num> {
    if is_scalar(v) { num(v) } else { None }
}

/// `_unop`'s scalar path with a per-op body; the non-scalar arm only
/// reaches `float`/`int` (handled at the call site) — everything else
/// declines like `_tcall`'s `TypeError`.
fn unop_scalar(v: V, f: impl FnOnce(Num) -> Option<V>) -> Option<V> {
    f(scalar_num(&v)?)
}

/// `_binop` — scalar arithmetic on the numeric tower, then the
/// Python-operator fallback for non-scalars (no tensor algebra is
/// registered on this side, so `_top(op)` is always `None`).
fn binop(op: BinOp, a: V, b: V) -> Option<V> {
    if is_scalar(&a) && is_scalar(&b) {
        let (x, y) = (num(&a)?, num(&b)?);
        return match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul => match (x, y) {
                (Num::Int(p), Num::Int(q)) => {
                    let r = match op {
                        BinOp::Add => p.checked_add(q),
                        BinOp::Sub => p.checked_sub(q),
                        _ => p.checked_mul(q),
                    }?;
                    int_v(r)
                }
                (p, q) => {
                    let r = match op {
                        BinOp::Add => p.f64() + q.f64(),
                        BinOp::Sub => p.f64() - q.f64(),
                        _ => p.f64() * q.f64(),
                    };
                    Some(V::Float(r))
                }
            },
            BinOp::Fdiv => {
                if y.f64() == 0.0 {
                    return None; // ZeroDivisionError → decline
                }
                Some(V::Float(x.f64() / y.f64()))
            }
            BinOp::Floordiv => match (x, y) {
                (Num::Int(p), Num::Int(q)) => int_v(py_floordiv(p, q)?),
                (p, q) => {
                    if q.f64() == 0.0 {
                        return None;
                    }
                    Some(V::Float((p.f64() / q.f64()).floor()))
                }
            },
        };
    }
    // The `_top(op) or _DBINOPS[op]` fallback — Python's own
    // operators over non-scalars.
    match op {
        BinOp::Add => match (&a, &b) {
            (V::Seq(x), V::Seq(y)) => {
                let mut out = x.to_vec();
                out.extend(y.iter().cloned());
                Some(V::Seq(out.into()))
            }
            (V::Str(x), V::Str(y)) => Some(V::Str(Rc::from(format!("{x}{y}")))),
            _ => None,
        },
        BinOp::Mul => seq_repeat(&a, &b),
        // sub/fdiv/floordiv have no non-scalar Python meaning here.
        _ => None,
    }
}

/// `a * b` for non-scalars — sequence/string repetition by an
/// int-like count (negative counts give the empty sequence, as in
/// Python).
fn seq_repeat(a: &V, b: &V) -> Option<V> {
    fn rep(unit: &V, n: i64) -> V {
        if n <= 0 {
            return V::Seq(Rc::from(Vec::new()));
        }
        match unit {
            V::Seq(items) => {
                let mut out = Vec::with_capacity(items.len() * n as usize);
                for _ in 0..n {
                    out.extend(items.iter().cloned());
                }
                V::Seq(out.into())
            }
            V::Str(s) => V::Str(Rc::from(s.repeat(n as usize))),
            _ => unreachable!("rep called on non-repeatable"),
        }
    }
    let count = |v: &V| -> Option<i64> {
        match v {
            V::Int(n) => Some(*n),
            V::Bool(b) => Some(i64::from(*b)),
            _ => None,
        }
    };
    match (a, b) {
        (V::Seq(_) | V::Str(_), _) if count(b).is_some() => Some(rep(a, count(b)?)),
        (_, V::Seq(_) | V::Str(_)) if count(a).is_some() => Some(rep(b, count(a)?)),
        _ => None,
    }
}

/// Python `int(str)` — trimmed, optional sign, ASCII digits only
/// (underscores/`0x` spellings decline — conservative).
fn parse_int(s: &str) -> Option<i128> {
    let t = s.trim();
    let (neg, digits) = match t.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n: i128 = digits.parse().ok()?;
    Some(if neg { -n } else { n })
}

/// Python `float(str)` — trimmed; `f64::from_str` covers the
/// `inf`/`nan`/exponent spellings (underscores decline).
fn parse_float(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok()
}

// ---------------------------------------------------------------------------
//  Attr-position parsing (`_aval`'s three spellings)
// ---------------------------------------------------------------------------

/// Parse an `_aval` argument: string → attr name, array → inline
/// expr, else literal.
fn aval_from_data(value: &Value) -> Result<Aval, GuardError> {
    match value {
        Value::String(s) => Ok(Aval::Name(Rc::from(s.as_str()))),
        Value::Array(_) => Ok(Aval::Expr(Box::new(Expr::from_data(value)?))),
        _ => Ok(Aval::Lit(V::from_json(value))),
    }
}

/// Parse a term-position argument — `bound.get(x)`: only a string
/// names a metavar; anything else is unbound.
fn term_ref_from_data(value: &Value) -> TermRef {
    match value {
        Value::String(s) => TermRef::Name(Rc::from(s.as_str())),
        _ => TermRef::Unbound,
    }
}

/// Fetch `args[i]` or report malformed (Python would `IndexError`).
fn arg<'a>(args: &'a [Value], i: usize, whole: &Value) -> Result<&'a Value, GuardError> {
    args.get(i)
        .ok_or_else(|| GuardError::Malformed(format!("{whole}")))
}

/// Stage-3 predicate names still refused — the `_OPS` entries that
/// read shapes through machinery this port does not cover: the
/// flat-map/repeat pairing guards (`axis-align-eq`, `bcast-dim-inv`,
/// `flat-pair-unsq`, `flat-map-unsq`, `repeat-chain`, `repeat-heads`).
/// Named so the refusal says *why*.
const SHAPE_PREDS_UNPORTED: &[&str] = &[
    "axis-align-eq",
    "bcast-dim-inv",
    "flat-pair-unsq",
    "flat-map-unsq",
    "repeat-chain",
    "repeat-heads",
];

/// A strict int literal for the shape predicates' `i`/`j`/`k`/`m`/`r`
/// positions — Python's handlers `isinstance`-check these and
/// decline; a non-int literal is a record bug, refused as malformed
/// (the conservative arm — never a silently-wrong comparison).
fn lit_int(v: &Value) -> Result<i64, GuardError> {
    v.as_i64()
        .or_else(|| v.as_u64().and_then(|u| i64::try_from(u).ok()))
        .ok_or_else(|| GuardError::Malformed(format!("expected int literal, got {v}")))
}

/// Parse a shape-spec argument — `_shape`'s ref grammar: a string is
/// a bound-term metavar ([`crate::shape::ShapeSpec::Var`]), a
/// non-empty tuple dispatches through the `_SPEC_OPS` table
/// ([`crate::shape::ShapeSpec`]'s computed variants), anything else
/// resolves to [`crate::shape::ShapeSpec::Unknown`] — Python's `None`
/// decline. A spec op with a bad arity resolves unknown the same
/// way; an inline derive *expr* in an attr position is a named
/// [`GuardError::Unsupported`] (shape attrs do not evaluate stage-1
/// dspec exprs).
fn shape_spec_from_data(value: &Value) -> Result<crate::shape::ShapeSpec, GuardError> {
    use crate::shape::ShapeSpec as S;
    let Value::String(name) = value else {
        let Value::Array(items) = value else {
            return Ok(S::Unknown);
        };
        let Some(head) = items.first().and_then(Value::as_str) else {
            return Ok(S::Unknown);
        };
        let spec_arg = |i: usize| -> Result<S, GuardError> {
            items
                .get(i + 1)
                .map(shape_spec_from_data)
                .unwrap_or(Ok(S::Unknown))
        };
        let attr_arg = |i: usize| -> Result<crate::shape::AttrArg, GuardError> {
            items
                .get(i + 1)
                .map(shape_attr_arg_from_data)
                .unwrap_or(Ok(crate::shape::AttrArg::Lit(AttrValue::Null)))
        };
        let name_arg = |i: usize| -> String {
            match items.get(i + 1) {
                Some(Value::String(s)) => s.clone(),
                // A non-string NAME resolves the spec to unknown —
                // `_aval`'s decline arm.
                _ => String::new(),
            }
        };
        return match head {
            "mm-out" => Ok(S::MmOut(Box::new(spec_arg(0)?), Box::new(spec_arg(1)?))),
            "bcast" => Ok(S::Bcast(Box::new(spec_arg(0)?), Box::new(spec_arg(1)?))),
            "unsq-out" => Ok(S::UnsqOut(Box::new(spec_arg(0)?), attr_arg(1)?)),
            "reshape-out" => Ok(S::ReshapeOut(Box::new(spec_arg(0)?), attr_arg(1)?)),
            "getitem-out" => Ok(S::GetitemOut(Box::new(spec_arg(0)?))),
            "select-out" => Ok(S::SelectOut(Box::new(spec_arg(0)?), attr_arg(1)?)),
            "slice-out" => Ok(S::SliceOut(
                Box::new(spec_arg(0)?),
                attr_arg(1)?,
                opt_attr_arg(items, 2)?,
                opt_attr_arg(items, 3)?,
                opt_attr_arg(items, 4)?,
            )),
            "chunk-out" => Ok(S::ChunkOut(
                Box::new(spec_arg(0)?),
                attr_arg(1)?,
                attr_arg(2)?,
            )),
            "transpose-out" => Ok(S::TransposeOut(
                Box::new(spec_arg(0)?),
                attr_arg(1)?,
                attr_arg(2)?,
            )),
            "tail-block" => {
                let name = name_arg(1);
                if name.is_empty() {
                    Ok(S::Unknown)
                } else {
                    Ok(S::TailBlock(Box::new(spec_arg(0)?), name))
                }
            }
            "isel-out" => Ok(S::IselOut(
                Box::new(spec_arg(0)?),
                attr_arg(1)?,
                attr_arg(2)?,
            )),
            "vshape" => Ok(S::Vshape(Box::new(spec_arg(0)?))),
            _ => Ok(S::Unknown),
        };
    };
    Ok(S::Var(name.clone()))
}

/// An optional [`crate::shape::AttrArg`] spec slot — a missing or
/// JSON-`null` arg maps to the `None` slot (the torch-default
/// marker).
fn opt_attr_arg(items: &[Value], i: usize) -> Result<Option<crate::shape::AttrArg>, GuardError> {
    match items.get(i + 1) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => shape_attr_arg_from_data(v).map(Some),
    }
}

/// Parse an attr-position argument for the shape predicates/specs —
/// `_aval`'s spellings restricted to what [`crate::shape::AttrArg`]
/// carries: a string names a bound attr metavariable, `("lit", v)`
/// and scalars are literals. An inline derive *expr* (any other
/// array) is a named refusal — the stage-1 `Aval` sites evaluate
/// exprs; inside shape specs they stay [`GuardError::Unsupported`].
fn shape_attr_arg_from_data(value: &Value) -> Result<crate::shape::AttrArg, GuardError> {
    use crate::shape::AttrArg as A;
    match value {
        Value::String(s) => Ok(A::Name(s.clone())),
        Value::Array(items) if items.len() == 2 && items[0].as_str() == Some("lit") => {
            Ok(A::Lit(AttrValue::from(&items[1])))
        }
        Value::Array(_) => Err(GuardError::Unsupported(format!(
            "inline derive expr in a shape attr position: {value}"
        ))),
        _ => Ok(A::Lit(AttrValue::from(value))),
    }
}

/// Parse one predicate node `("op", args…)` — arity checked like
/// Python's indexing (short → malformed; extra args ignored, as the
/// handlers do).
fn pred_from_data(op: &str, args: &[Value]) -> Result<Pred, GuardError> {
    let whole = Value::Array(
        std::iter::once(Value::from(op))
            .chain(args.iter().cloned())
            .collect(),
    );
    if SHAPE_PREDS_UNPORTED.contains(&op) {
        return Err(GuardError::Unsupported(format!(
            "{op}: needs the flat-map/repeat guard machinery (not ported)"
        )));
    }
    let aval = |i: usize| aval_from_data(arg(args, i, &whole)?);
    let tref =
        |i: usize| -> Result<TermRef, GuardError> { Ok(term_ref_from_data(arg(args, i, &whole)?)) };
    let lit = |i: usize| Ok::<V, GuardError>(V::from_json(arg(args, i, &whole)?));
    let cmp = |i: usize| {
        Cmp::parse(arg(args, i, &whole)?)
            .ok_or_else(|| GuardError::Malformed(format!("bad CMP in {whole}")))
    };
    let spec = |i: usize| shape_spec_from_data(arg(args, i, &whole)?);
    let sattr = |i: usize| shape_attr_arg_from_data(arg(args, i, &whole)?);
    let int = |i: usize| lit_int(arg(args, i, &whole)?);
    match op {
        // The ported shape-reading `_OPS` — evaluated through
        // `crate::shape::pred` at fire time.
        "shaped" => Ok(Pred::Shape(ShapePred::Shaped(spec(0)?))),
        "concrete" => Ok(Pred::Shape(ShapePred::Concrete(spec(0)?))),
        "scalar" => Ok(Pred::Shape(ShapePred::Scalar(spec(0)?))),
        "uniform" => Ok(Pred::Shape(ShapePred::Uniform(spec(0)?))),
        "ones-but-last" => Ok(Pred::Shape(ShapePred::OnesButLast(spec(0)?))),
        "rank" => Ok(Pred::Shape(ShapePred::Rank(spec(0)?, cmp(1)?, int(2)?))),
        "rank-eq" => Ok(Pred::Shape(ShapePred::RankEq(spec(0)?, spec(1)?))),
        "shape-eq" => Ok(Pred::Shape(ShapePred::ShapeEq(spec(0)?, spec(1)?))),
        "shape-compat" => Ok(Pred::Shape(ShapePred::ShapeCompat(spec(0)?, spec(1)?))),
        "dim-eq" => Ok(Pred::Shape(ShapePred::DimEq(
            spec(0)?,
            int(1)?,
            spec(2)?,
            int(3)?,
        ))),
        "dim-compat" => Ok(Pred::Shape(ShapePred::DimCompat(
            spec(0)?,
            int(1)?,
            spec(2)?,
            int(3)?,
        ))),
        "dim-eq-const" => Ok(Pred::Shape(ShapePred::DimEqConst(
            spec(0)?,
            int(1)?,
            int(2)?,
        ))),
        "dim-eq-attr" => Ok(Pred::Shape(ShapePred::DimEqAttr(
            spec(0)?,
            sattr(1)?,
            spec(2)?,
            sattr(3)?,
        ))),
        "dim-mod" => Ok(Pred::Shape(ShapePred::DimMod(
            spec(0)?,
            sattr(1)?,
            int(2)?,
            int(3)?,
        ))),
        "bcast-into" => Ok(Pred::Shape(ShapePred::BcastInto(spec(0)?, spec(1)?))),
        "mm-shape-ok" => Ok(Pred::Shape(ShapePred::MmShapeOk(spec(0)?, spec(1)?))),
        "bcast-eq" => Ok(Pred::Shape(ShapePred::BcastEq(
            spec(0)?,
            spec(1)?,
            spec(2)?,
            spec(3)?,
        ))),
        "ones-before" => Ok(Pred::Shape(ShapePred::OnesBefore(spec(0)?, sattr(1)?))),
        "axes-last2" => Ok(Pred::Shape(ShapePred::AxesLast2(
            spec(0)?,
            sattr(1)?,
            sattr(2)?,
        ))),
        "axes-distinct" => Ok(Pred::Shape(ShapePred::AxesDistinct(
            spec(0)?,
            sattr(1)?,
            sattr(2)?,
        ))),
        "axes-eq" => Ok(Pred::Shape(ShapePred::AxesEq(
            spec(0)?,
            sattr(1)?,
            sattr(2)?,
            sattr(3)?,
            sattr(4)?,
        ))),
        "axes-noop" => Ok(Pred::Shape(ShapePred::AxesNoop(
            spec(0)?,
            sattr(1)?,
            sattr(2)?,
        ))),
        "axis" => Ok(Pred::Shape(ShapePred::Axis(spec(0)?, sattr(1)?, int(2)?))),
        "attr-cmp-dim" => Ok(Pred::Shape(ShapePred::AttrCmpDim(
            sattr(0)?,
            cmp(1)?,
            spec(2)?,
            sattr(3)?,
        ))),
        "attr-range" => Ok(Pred::Shape(ShapePred::AttrRange(
            sattr(0)?,
            spec(1)?,
            sattr(2)?,
        ))),
        "op-in" => {
            // ops is a literal container — Seq elements, Str
            // substring, Obj keys (v_contains handles all three).
            Ok(Pred::OpIn(tref(0)?, lit(1)?))
        }
        "leaf" => Ok(Pred::Leaf(tref(0)?)),
        "const" => Ok(Pred::Const(tref(0)?)),
        "term-eq" => Ok(Pred::TermEq(tref(0)?, tref(1)?)),
        "const-num" => Ok(Pred::ConstNum(tref(0)?)),
        "const-cmp" => Ok(Pred::ConstCmp(tref(0)?, cmp(1)?, lit(2)?)),
        "attr-is" => Ok(Pred::AttrIs(aval(0)?, lit(1)?)),
        "attr-eq" => Ok(Pred::AttrEq(aval(0)?, lit(1)?)),
        "attr-in" => Ok(Pred::AttrIn(aval(0)?, lit(1)?)),
        "attr-type" => {
            let k = AttrKind::parse(arg(args, 1, &whole)?)
                .ok_or_else(|| GuardError::Malformed(format!("bad attr-type in {whole}")))?;
            Ok(Pred::AttrType(aval(0)?, k))
        }
        "attr-len" => Ok(Pred::AttrLen(aval(0)?, cmp(1)?, lit(2)?)),
        "attr-eq-attr" => Ok(Pred::AttrEqAttr(aval(0)?, aval(1)?)),
        "attr-cmp" => Ok(Pred::AttrCmp(aval(0)?, cmp(1)?, aval(2)?)),
        "attr-ints" => Ok(Pred::AttrInts(aval(0)?)),
        _ => Err(GuardError::Malformed(format!("unknown cond op: {op:?}"))),
    }
}

// ---------------------------------------------------------------------------
//  The dspec — `{NAME: expr}` → `Derived` bindings
// ---------------------------------------------------------------------------

/// A parsed derive spec — `derive_from_data`'s canonical pairs.
#[derive(Clone, Debug)]
pub struct Dspec {
    /// `(name, expr)` pairs, sorted by name (the canonical form —
    /// `"$attr:"` stripped, `"$leaf:"` kept, exactly
    /// `derive_from_data`).
    entries: Vec<(String, Expr)>,
}

impl Dspec {
    /// Parse a `dspec` from its record JSON — a `{NAME: expr}`
    /// mapping or a `[[NAME, expr], …]` pair list.
    ///
    /// # Errors
    ///
    /// [`GuardError::Malformed`] on a bad spec shape, non-string
    /// name, or malformed expr; [`GuardError::Unsupported`] on a
    /// unported expr op.
    pub fn from_data(value: &Value) -> Result<Dspec, GuardError> {
        let mut items: Vec<(String, &Value)> = Vec::new();
        match value {
            Value::Object(map) => {
                for (k, v) in map {
                    items.push((k.clone(), v));
                }
            }
            Value::Array(pairs) => {
                for p in pairs {
                    let pair = p.as_array().filter(|a| a.len() == 2).ok_or_else(|| {
                        GuardError::Malformed(format!("malformed derive spec: {value}"))
                    })?;
                    let name = pair[0].as_str().ok_or_else(|| {
                        GuardError::Malformed(format!("derive spec key not a name: {value}"))
                    })?;
                    items.push((name.to_string(), &pair[1]));
                }
            }
            _ => {
                return Err(GuardError::Malformed(format!(
                    "malformed derive spec: {value}"
                )));
            }
        }
        let mut entries = Vec::with_capacity(items.len());
        for (name, expr) in items {
            // derive_from_data: "$attr:" loses its prefix, "$leaf:"
            // keeps its own; every other key is a bare attr name.
            let canon = if let Some(rest) = name.strip_prefix("$attr:") {
                rest.to_string()
            } else {
                name
            };
            entries.push((canon, Expr::from_data(expr)?));
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(Dspec { entries })
    }

    /// The metavariable names this spec binds, in the
    /// [`crate::pat`] metavar namespace: a `"$leaf:H"` key binds the
    /// term metavariable `H`; every other key binds the attribute
    /// metavariable `"$attr:<name>"` (the `"$attr:"` prefix was
    /// already stripped at parse).
    pub(crate) fn bound_vars(&self) -> std::collections::BTreeSet<String> {
        self.entries
            .iter()
            .map(|(name, _)| {
                if let Some(hint) = name.strip_prefix("$leaf:") {
                    hint.to_string()
                } else {
                    format!("$attr:{name}")
                }
            })
            .collect()
    }

    /// `eval_derive` — the extra-substitution map; `None` vetoes the
    /// whole spec on the first declining expr.
    pub fn eval(&self, env: &BoundEnv) -> Option<Vec<Derived>> {
        let mut out = Vec::with_capacity(self.entries.len());
        for (name, expr) in &self.entries {
            let v = expr.eval(env)?;
            if let Some(hint) = name.strip_prefix("$leaf:") {
                out.push(Derived::Term(Rc::from(hint), mint_leaf(&v, hint)?));
            } else {
                out.push(Derived::Attr(Rc::from(name.as_str()), v.to_attr()));
            }
        }
        Some(out)
    }
}

/// `_mint_leaf` — package a `$leaf:` value as a leaf term.
///
/// Scalars mint `Const(v)` — the `Const` contract is int/float
/// exactly, so `bool`/`None`/`str` decline rather than smuggle a
/// non-scalar in. Non-scalars need the registered tensor `mint`,
/// which does not exist on this side — decline (unported/backend).
fn mint_leaf(v: &V, hint: &str) -> Option<Term> {
    let _ = hint;
    let json = match v {
        V::Int(n) => Value::from(*n),
        V::Float(f) => serde_json::Number::from_f64(*f).map(Value::Number)?,
        _ => return None,
    };
    Some(handl_egraph::from_kernel(&KernelTerm::constant(json)))
}

// ---------------------------------------------------------------------------
//  Tests — the interpreter over a hand-built BoundEnv
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use handl_egraph::attr;
    use serde_json::json;
    use std::collections::BTreeMap;

    fn konst(v: Value) -> Term {
        handl_egraph::from_kernel(&hnl_engine::Term::constant(v))
    }

    /// A kernel-`Op` term — the `source` attr is the bridge's marker.
    fn kop(op: &str, args: Vec<Term>) -> Term {
        Term::op(
            op,
            args,
            [attr("source", AttrValue::Str(Rc::from("native")))],
        )
    }

    /// A structural (non-kernel) op term.
    fn sop(op: &str, args: Vec<Term>) -> Term {
        Term::opl(op, args)
    }

    fn env(terms: &[(&str, Term)], attrs: &[(&str, AttrValue)]) -> BoundEnv {
        BoundEnv {
            terms: terms
                .iter()
                .map(|(k, v)| (Rc::from(*k), v.clone()))
                .collect::<BTreeMap<_, _>>(),
            attrs: attrs
                .iter()
                .map(|(k, v)| (Rc::from(*k), v.clone()))
                .collect::<BTreeMap<_, _>>(),
        }
    }

    fn eval_cond(v: Value, env: &BoundEnv) -> bool {
        Cond::from_data(&v).unwrap().eval(env)
    }

    fn eval_expr(v: Value, env: &BoundEnv) -> Option<V> {
        Expr::from_data(&v).unwrap().eval(env)
    }

    #[test]
    fn cond_parse_refuses_bad_trees() {
        // Bare non-bool scalars and empty nodes are malformed.
        for bad in [
            json!(5),
            json!("x"),
            json!(["and"]),
            json!([]),
            json!({"op": "leaf"}),
            json!([5, "x"]),
        ] {
            // ["and"] with zero children is actually legal (all([])=true)
            // — handled below; the rest must refuse.
            if bad == json!(["and"]) {
                continue;
            }
            assert!(
                matches!(Cond::from_data(&bad), Err(GuardError::Malformed(_))),
                "{bad}"
            );
        }
        // `not` arity, missing pred args, unknown ops.
        for bad in [
            json!(["not"]),
            json!(["not", true, false]),
            json!(["leaf"]),
            json!(["attr-eq", "A"]),
            json!(["bogus", "x"]),
            json!(["attr-type", "A", "complex"]),
            json!(["const-cmp", "C", "===", 1]),
        ] {
            assert!(
                matches!(Cond::from_data(&bad), Err(GuardError::Malformed(_))),
                "{bad}"
            );
        }
        // The unported flat-map/repeat guards refuse as unsupported,
        // named — the ported shape preds (`shaped`, `rank-eq`, …)
        // parse into `Pred::Shape` and evaluate at fire time.
        for bad in [
            json!(["flat-pair-unsq", "x"]),
            json!(["axis-align-eq", "a", "b"]),
            json!(["repeat-chain", "x"]),
            json!(["repeat-heads", "x"]),
        ] {
            assert!(
                matches!(Cond::from_data(&bad), Err(GuardError::Unsupported(_))),
                "{bad}"
            );
        }
        // and/or/not combinators.
        let e = env(&[], &[]);
        assert!(eval_cond(json!(["and"]), &e));
        assert!(!eval_cond(json!(["or"]), &e));
        assert!(eval_cond(json!(["not", false]), &e));
        assert!(eval_cond(json!(true), &e));
        assert!(!eval_cond(json!(false), &e));
    }

    #[test]
    fn term_predicates_read_bound_terms() {
        let add = kop("add", vec![konst(json!(1)), konst(json!(2))]);
        let leaf = konst(json!(7));
        let e = env(
            &[
                ("u", add.clone()),
                ("v", leaf.clone()),
                ("w", sop("seq", vec![])),
            ],
            &[],
        );

        // op-in: kernel-op head membership; structural ops decline.
        assert!(eval_cond(json!(["op-in", "u", ["add", "mul"]]), &e));
        assert!(!eval_cond(json!(["op-in", "u", ["mul"]]), &e));
        assert!(!eval_cond(json!(["op-in", "w", ["seq"]]), &e)); // no source attr
        assert!(!eval_cond(json!(["op-in", "missing", ["add"]]), &e));
        // leaf: anything that is not a kernel op (const leaf, structural op, unbound).
        assert!(eval_cond(json!(["leaf", "v"]), &e));
        assert!(eval_cond(json!(["leaf", "w"]), &e));
        assert!(eval_cond(json!(["leaf", "missing"]), &e));
        assert!(!eval_cond(json!(["leaf", "u"]), &e));
        // const / const-num / const-cmp over the Const leaf's .value.
        assert!(eval_cond(json!(["const", "v"]), &e));
        assert!(!eval_cond(json!(["const", "u"]), &e));
        assert!(eval_cond(json!(["const-num", "v"]), &e));
        assert!(!eval_cond(json!(["const-num", "u"]), &e));
        assert!(eval_cond(json!(["const-cmp", "v", "==", 7]), &e));
        assert!(eval_cond(json!(["const-cmp", "v", ">=", 7.0]), &e));
        assert!(!eval_cond(json!(["const-cmp", "v", "<", 7]), &e));
        assert!(!eval_cond(json!(["const-cmp", "u", "==", 1]), &e));
        // term-eq: structural equality; unbound == unbound.
        assert!(eval_cond(json!(["term-eq", "u", "u"]), &e));
        assert!(!eval_cond(json!(["term-eq", "u", "v"]), &e));
        assert!(eval_cond(json!(["term-eq", "missing", "missing2"]), &e));
        assert!(!eval_cond(json!(["term-eq", "u", "missing"]), &e));
    }

    #[test]
    fn attr_predicates_read_bound_attrs() {
        let e = env(
            &[],
            &[
                ("I", AttrValue::Int(3)),
                ("F", AttrValue::Float(0.5)),
                ("B", AttrValue::Bool(true)),
                ("S", AttrValue::Str(Rc::from("hi"))),
                (
                    "T",
                    AttrValue::Array(vec![AttrValue::Int(1), AttrValue::Int(2)].into()),
                ),
                ("N", AttrValue::Null),
            ],
        );
        // attr-is: bool/None singletons; int only inside the small-int cache.
        assert!(eval_cond(json!(["attr-is", "B", true]), &e));
        assert!(!eval_cond(json!(["attr-is", "B", false]), &e));
        assert!(eval_cond(json!(["attr-is", "N", null]), &e));
        assert!(eval_cond(json!(["attr-is", "missing", null]), &e)); // absent reads None
        assert!(eval_cond(json!(["attr-is", "I", 3]), &e)); // 3 in the cache
        assert!(!eval_cond(json!(["attr-is", "I", 300]), &e)); // 300 not cached... wait I=3
        // attr-eq: Python == incl. numeric tower and missing→None.
        assert!(eval_cond(json!(["attr-eq", "I", 3]), &e));
        assert!(eval_cond(json!(["attr-eq", "I", 3.0]), &e)); // 3 == 3.0
        assert!(eval_cond(json!(["attr-eq", "B", 1]), &e)); // True == 1
        assert!(eval_cond(json!(["attr-eq", "missing", null]), &e));
        assert!(!eval_cond(json!(["attr-eq", "I", 4]), &e));
        assert!(eval_cond(json!(["attr-eq", "T", [1, 2]]), &e));
        // attr-in / attr-type / attr-len / attr-ints.
        assert!(eval_cond(json!(["attr-in", "I", [1, 3, 5]]), &e));
        assert!(!eval_cond(json!(["attr-in", "I", [1, 2]]), &e));
        assert!(eval_cond(json!(["attr-in", "S", "hi"]), &e)); // str substring
        assert!(eval_cond(json!(["attr-type", "I", "int"]), &e));
        assert!(eval_cond(json!(["attr-type", "B", "int"]), &e)); // bool is int
        assert!(eval_cond(json!(["attr-type", "F", "float"]), &e));
        assert!(eval_cond(json!(["attr-type", "B", "bool"]), &e));
        assert!(eval_cond(json!(["attr-type", "T", "tuple"]), &e));
        assert!(!eval_cond(json!(["attr-type", "missing", "int"]), &e));
        assert!(eval_cond(json!(["attr-len", "T", "==", 2]), &e));
        assert!(eval_cond(json!(["attr-len", "T", ">", 1]), &e));
        assert!(!eval_cond(json!(["attr-len", "I", "==", 2]), &e));
        assert!(eval_cond(json!(["attr-ints", "T"]), &e));
        assert!(!eval_cond(json!(["attr-ints", "S"]), &e));
        // attr-eq-attr / attr-cmp over aval positions (incl. inline exprs).
        assert!(eval_cond(json!(["attr-eq-attr", "I", "I"]), &e));
        assert!(!eval_cond(json!(["attr-eq-attr", "I", "F"]), &e));
        assert!(eval_cond(json!(["attr-cmp", "I", ">", "F"]), &e)); // 3 > 0.5
        assert!(eval_cond(json!(["attr-cmp", ["attr", "I"], "==", 3]), &e)); // inline expr
        assert!(!eval_cond(json!(["attr-cmp", "missing", "==", 1]), &e));
        // Boolean structure.
        assert!(eval_cond(
            json!(["and", ["attr-eq", "I", 3], ["attr-eq", "S", "hi"]]),
            &e
        ));
        assert!(eval_cond(
            json!(["or", ["attr-eq", "I", 4], ["attr-eq", "S", "hi"]]),
            &e
        ));
        assert!(!eval_cond(json!(["not", ["attr-eq", "I", 3]]), &e));
    }

    #[test]
    fn exprs_evaluate_the_scalar_algebra() {
        let e = env(
            &[("u", kop("add", vec![])), ("c", konst(json!(4)))],
            &[
                ("A", AttrValue::Int(3)),
                (
                    "T",
                    AttrValue::Array(
                        vec![AttrValue::Int(10), AttrValue::Int(20), AttrValue::Int(30)].into(),
                    ),
                ),
            ],
        );
        // literals / attr / attr0 / const.
        assert_eq!(eval_expr(json!(null), &e), Some(V::Null));
        assert_eq!(eval_expr(json!(2.5), &e), Some(V::Float(2.5)));
        assert_eq!(
            eval_expr(json!(["lit", {"k": [1]}]), &e),
            Some(V::Obj(Rc::new(
                [(Rc::from("k"), V::Seq(Rc::from(vec![V::Int(1)])))]
                    .into_iter()
                    .collect()
            )))
        );
        assert_eq!(eval_expr(json!(["attr", "A"]), &e), Some(V::Int(3)));
        assert_eq!(eval_expr(json!(["attr0", "T"]), &e), Some(V::Int(10)));
        assert_eq!(eval_expr(json!(["attr0", "A"]), &e), Some(V::Int(3))); // non-seq → itself
        assert_eq!(eval_expr(json!(["attr", "missing"]), &e), None); // missing declines
        assert_eq!(eval_expr(json!(["const", "c"]), &e), Some(V::Int(4)));
        // len / tuple / concat / nth / take / sum.
        assert_eq!(
            eval_expr(json!(["len", ["attr", "T"]]), &e),
            Some(V::Int(3))
        );
        // Bare strings are malformed exprs — a literal spells ("lit", v).
        assert!(matches!(
            Expr::from_data(&json!(["tuple", 1, "x"])),
            Err(GuardError::Malformed(_))
        ));
        assert_eq!(
            eval_expr(json!(["tuple", 1, ["lit", "x"]]), &e),
            Some(V::Seq(Rc::from(vec![V::Int(1), V::Str(Rc::from("x"))])))
        );
        assert_eq!(
            eval_expr(json!(["concat", ["attr", "T"], ["attr", "T"]]), &e),
            Some(V::Seq(Rc::from(vec![
                V::Int(10),
                V::Int(20),
                V::Int(30),
                V::Int(10),
                V::Int(20),
                V::Int(30),
            ])))
        );
        assert_eq!(
            eval_expr(json!(["nth", ["attr", "T"], 1]), &e),
            Some(V::Int(20))
        );
        assert_eq!(
            eval_expr(json!(["nth", ["attr", "T"], -1]), &e),
            Some(V::Int(30))
        );
        assert_eq!(eval_expr(json!(["nth", ["attr", "T"], 3]), &e), None); // out of range
        assert_eq!(
            eval_expr(json!(["take", ["attr", "T"], 2]), &e),
            Some(V::Seq(Rc::from(vec![V::Int(10), V::Int(20)])))
        );
        assert_eq!(
            eval_expr(json!(["take", ["attr", "T"], -1]), &e),
            Some(V::Seq(Rc::from(vec![V::Int(10), V::Int(20)])))
        ); // seq[:-1]
        assert_eq!(
            eval_expr(json!(["sum", ["attr", "T"]]), &e),
            Some(V::Int(60))
        );
        assert_eq!(eval_expr(json!(["sum", ["tuple"]]), &e), Some(V::Int(0)));
        // gather / unique / posmap.
        assert_eq!(
            eval_expr(json!(["gather", ["attr", "T"], ["lit", [0, 2]]]), &e),
            Some(V::Seq(Rc::from(vec![V::Int(10), V::Int(30)])))
        );
        assert_eq!(
            eval_expr(json!(["gather", ["attr", "T"], ["lit", [true]]]), &e),
            None // bool selectors decline (unlike nth)
        );
        assert_eq!(
            eval_expr(json!(["unique", ["lit", [1, 2, 1, 3, 2]]]), &e),
            Some(V::Seq(Rc::from(vec![V::Int(1), V::Int(2), V::Int(3)])))
        );
        assert_eq!(
            eval_expr(
                json!(["posmap", ["lit", ["a", "b"]], ["lit", ["b", "a", "b"]]]),
                &e
            ),
            Some(V::Seq(Rc::from(vec![V::Int(1), V::Int(0), V::Int(1)])))
        );
        assert_eq!(
            eval_expr(json!(["posmap", ["lit", ["a"]], ["lit", ["z"]]]), &e),
            None // absent key declines
        );
        // arithmetic — the numeric tower.
        assert_eq!(eval_expr(json!(["add", 1, 2]), &e), Some(V::Int(3)));
        assert_eq!(eval_expr(json!(["add", 1, 2.5]), &e), Some(V::Float(3.5)));
        assert_eq!(eval_expr(json!(["sub", 1, 2]), &e), Some(V::Int(-1)));
        assert_eq!(eval_expr(json!(["mul", true, 3]), &e), Some(V::Int(3)));
        assert_eq!(eval_expr(json!(["fdiv", 1, 2]), &e), Some(V::Float(0.5)));
        assert_eq!(eval_expr(json!(["fdiv", 1, 0]), &e), None); // ZeroDivision → decline
        assert_eq!(eval_expr(json!(["floordiv", -3, 2]), &e), Some(V::Int(-2)));
        assert_eq!(eval_expr(json!(["floordiv", 7, 2]), &e), Some(V::Int(3)));
        assert_eq!(eval_expr(json!(["mod", -3, 2]), &e), Some(V::Int(1))); // Python sign
        assert_eq!(eval_expr(json!(["mod", 3, 0]), &e), None);
        assert_eq!(eval_expr(json!(["neg", 5]), &e), Some(V::Int(-5)));
        assert_eq!(eval_expr(json!(["neg", true]), &e), Some(V::Int(-1)));
        assert_eq!(eval_expr(json!(["recip", 4]), &e), Some(V::Float(0.25)));
        assert_eq!(eval_expr(json!(["recip", 0]), &e), None);
        assert_eq!(eval_expr(json!(["float", 3]), &e), Some(V::Float(3.0)));
        assert_eq!(eval_expr(json!(["int", 2.9]), &e), Some(V::Int(2)));
        assert_eq!(eval_expr(json!(["int", -2.9]), &e), Some(V::Int(-2)));
        assert_eq!(eval_expr(json!(["sqrt", 4]), &e), Some(V::Float(2.0)));
        assert_eq!(eval_expr(json!(["sqrt", -1]), &e), None); // domain error
        assert_eq!(eval_expr(json!(["rsqrt", 4]), &e), Some(V::Float(0.5)));
        assert_eq!(eval_expr(json!(["rsqrt", 0]), &e), None);
        assert_eq!(eval_expr(json!(["max", 3, 7]), &e), Some(V::Int(7)));
        assert_eq!(eval_expr(json!(["min", 3.0, 7]), &e), None); // float declines
        assert_eq!(eval_expr(json!(["pow", 2, 10]), &e), Some(V::Int(1024)));
        assert_eq!(eval_expr(json!(["pow", 2, -1]), &e), None); // negative exp declines
        // non-scalar operator fallback: seq concat / str repeat.
        assert_eq!(
            eval_expr(json!(["add", ["attr", "T"], ["attr", "T"]]), &e),
            eval_expr(json!(["concat", ["attr", "T"], ["attr", "T"]]), &e)
        );
        assert_eq!(
            eval_expr(json!(["mul", ["lit", [1, 2]], 2]), &e),
            Some(V::Seq(Rc::from(vec![
                V::Int(1),
                V::Int(2),
                V::Int(1),
                V::Int(2)
            ])))
        );
        // guard / if embed cond trees.
        assert_eq!(
            eval_expr(json!(["guard", ["const-num", "c"], ["const", "c"]]), &e),
            Some(V::Int(4))
        );
        assert_eq!(
            eval_expr(json!(["guard", ["const-num", "u"], ["const", "c"]]), &e),
            None // u is an op — the guard vetoes
        );
        assert_eq!(
            eval_expr(json!(["if", ["attr-eq", "A", 3], 10, 20]), &e),
            Some(V::Int(10))
        );
        assert_eq!(
            eval_expr(json!(["if", ["attr-eq", "A", 4], 10, 20]), &e),
            Some(V::Int(20))
        );
        // declines propagate: a declining subexpr vetoes the whole spec.
        assert_eq!(eval_expr(json!(["add", ["attr", "missing"], 1]), &e), None);
        assert_eq!(
            eval_expr(json!(["tuple", 1, ["attr", "missing"]]), &e),
            None
        );
    }

    #[test]
    fn dspec_mints_attrs_and_leaf_terms() {
        let e = env(&[("c", konst(json!(4)))], &[("A", AttrValue::Int(3))]);
        // Object and pair-list forms canonicalize identically.
        let d1 = Dspec::from_data(
            &json!({"$attr:X": ["attr", "A"], "$leaf:N": ["neg", ["const", "c"]]}),
        )
        .unwrap();
        // The pair-list spelling parses identically.
        let _d2 = Dspec::from_data(&json!([
            ["$attr:X", ["attr", "A"]],
            ["$leaf:N", ["neg", ["const", "c"]]]
        ]))
        .unwrap();
        let out = d1.eval(&e).unwrap();
        assert_eq!(out.len(), 2);
        // A declining expr vetoes the whole spec.
        let bad = Dspec::from_data(&json!({"$attr:X": ["attr", "missing"]})).unwrap();
        assert!(bad.eval(&e).is_none());
        // $leaf: scalars mint Const leaves; non-scalars decline.
        let leaf = Dspec::from_data(&json!({"$leaf:N": ["add", ["const", "c"], 1]})).unwrap();
        let out = leaf.eval(&e).unwrap();
        let [Derived::Term(name, term)] = out.as_slice() else {
            panic!("expected one term binding: {out:?}")
        };
        assert_eq!(name.as_ref(), "N");
        // The minted term is a Const leaf decoding to 5.
        assert_eq!(
            const_value(term),
            Some(V::Int(5)),
            "minted leaf should decode to Const(5)"
        );
        let str_leaf = Dspec::from_data(&json!({"$leaf:S": ["lit", "x"]})).unwrap();
        assert!(str_leaf.eval(&e).is_none()); // str can't mint a Const
        // $attr: lowers through to_attr.
        let attr_d = Dspec::from_data(&json!({"A2": ["add", ["attr", "A"], 1]})).unwrap();
        let out = attr_d.eval(&e).unwrap();
        let [Derived::Attr(name, value)] = out.as_slice() else {
            panic!("expected one attr binding: {out:?}")
        };
        assert_eq!(name.as_ref(), "A2");
        assert_eq!(*value, AttrValue::Int(4));
        // Malformed specs refuse at parse.
        for bad in [json!(5), json!(["not-a-pair"]), json!([["ok", 1], [2, 3]])] {
            assert!(
                matches!(Dspec::from_data(&bad), Err(GuardError::Malformed(_))),
                "{bad}"
            );
        }
        // Stage-3 exprs refuse as unsupported.
        for bad in [
            json!({"X": ["shape", "u"]}),
            json!({"X": ["dim", "u", 0]}),
            json!({"X": ["bcast", "u"]}),
        ] {
            assert!(
                matches!(Dspec::from_data(&bad), Err(GuardError::Unsupported(_))),
                "{bad}"
            );
        }
        // Unknown expr ops are malformed.
        assert!(matches!(
            Dspec::from_data(&json!({"X": ["bogus", 1]})),
            Err(GuardError::Malformed(_))
        ));
    }
}
