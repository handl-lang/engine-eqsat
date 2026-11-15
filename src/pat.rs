//! The law pattern language — the interpretation's *only* pattern
//! surface.
//!
//! Kernel `Term` has no metavariables (`Term::Var`/`TermAttr::Meta`
//! exist only inside `handl_egraph`), so laws are authored here in
//! [`Pat`], a small pattern AST that `compile` lowers into the
//! e-graph's internal term domain. `handl_egraph::Term` never appears
//! in this crate's public API — the ADR-0174 non-goal ("`egraph::Term`
//! stays engine-internal") holds by construction.
//!
//! # The bridge encoding, as a law author sees it
//!
//! [`Pat::Op`] describes a kernel `Term::Op` and the compiler applies
//! the `term_bridge` encoding:
//!
//! * `dim` rides **child slot 0**. [`OpPat::dim`] controls it:
//!   - LHS `None` → a fresh don't-care metavariable (nested nodes) or
//!     the law's *root dim variable* (the LHS root op).
//!   - RHS `None` → the LHS root's dim variable — i.e. **rewritten ops
//!     inherit the matched root's dim** (the dim-preserving default;
//!     laws that change dim are not yet expressible — spike §6).
//!   - `Some(pat)` on either side compiles `pat` in dim position
//!     (a `Pat::Var` binds/reuses, a `Pat::Const` pins).
//!
//! If the LHS root is not an op (or pins a non-var dim), RHS ops get
//!   an unbound dim metavariable and the firing is skipped — honest,
//!   never a wrong dim.
//! * The `source` tag is pinned as an attr: [`PatSource::Native`] (the
//!   default) emits `source: "native"`; `Foreign(n)` additionally emits
//!   `foreign: n`; `Judgment`/`Dynamic` likewise. Attr matching is
//!   **exact-keyset** — a pattern that omits `lit`/`__attrs__` will
//!   *not* match a node that carries them. Use [`OpPat::lit`] /
//!   [`OpPat::attrs_lit`] / [`OpPat::attrs_bind`] to pin or bind them.
//! * [`Pat::Const`] mints the bridge's self-describing leaf key — a
//!   `Const` is matched by its JSON value, never by a raw string.

use std::collections::BTreeSet;
use std::rc::Rc;

use handl_egraph::{AttrValue, KernelTerm, Term, TermAttr, attr, attr_meta, from_kernel};
use serde_json::Value;

/// One pattern node — metavariable, constant leaf, or op application.
#[derive(Clone, Debug, PartialEq)]
pub enum Pat {
    /// A term metavariable: binds an e-class. Repeated uses of the same
    /// name must match the *same* class.
    Var(String),
    /// A `Const` leaf pinned to a JSON value (the bridge's
    /// self-describing leaf key — minted via `from_kernel`, never
    /// spelled as a string).
    Const(Value),
    /// An op application in the kernel `Term::Op` bridge encoding.
    Op(OpPat),
}

impl Pat {
    /// Shorthand for [`Pat::Var`].
    pub fn var(name: impl Into<String>) -> Pat {
        Pat::Var(name.into())
    }

    /// Shorthand for [`Pat::Const`].
    pub fn constant(value: impl Into<Value>) -> Pat {
        Pat::Const(value.into())
    }

    /// Shorthand for a native-source [`Pat::Op`] with the given args.
    pub fn op(name: impl Into<String>, args: Vec<Pat>) -> Pat {
        Pat::Op(OpPat::new(name).args(args))
    }
}

/// The `source` tag of a [`Pat::Op`] — mirrors the kernel
/// `ArrSource` lattice; compiled to the bridge's `source`/`foreign`
/// attrs.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum PatSource {
    /// `ArrSource::Native` — `source: "native"`. The default.
    #[default]
    Native,
    /// `ArrSource::Foreign(name)` — `source: "foreign"` +
    /// `foreign: name`.
    Foreign(String),
    /// `ArrSource::Judgment` — `source: "judgment"`.
    Judgment,
    /// `ArrSource::DynamicForeign` — `source: "dynamic"`.
    Dynamic,
}

/// How a [`Pat::Op`]'s `attrs` payload (the kernel `Term::Op.attrs`
/// `Value`, bridged under `"__attrs__"`) is treated.
#[derive(Clone, Debug, PartialEq)]
pub enum AttrsSpec {
    /// Pin the payload to a literal JSON value (exact-keyset match).
    Lit(Value),
    /// Bind the whole payload to an attribute metavariable — re-emitted
    /// verbatim wherever the same name appears on a RHS.
    Bind(String),
}

/// An op-application pattern: `op(args…)` in the kernel `Term::Op`
/// encoding (dim slot explicit via [`OpPat::dim`], see module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct OpPat {
    /// The operator name (the domain alphabet — data, not code).
    pub op: String,
    /// The dim-slot pattern. `None` = don't-care on the LHS (fresh
    /// metavariable; the LHS *root* gets the law's root dim var) and
    /// "inherit the LHS root's dim" on the RHS. Boxed — `Pat`/`OpPat`
    /// are mutually recursive.
    pub dim: Option<Box<Pat>>,
    /// Arguments *excluding* the dim slot.
    pub args: Vec<Pat>,
    /// The source tag; default [`PatSource::Native`].
    pub source: PatSource,
    /// The `lit` literal payload to pin, if any.
    pub lit: Option<Value>,
    /// The `attrs` payload spec, if any. `None` means the pattern
    /// carries no `__attrs__` key and will not match attr-carrying
    /// nodes (exact-keyset).
    pub attrs: Option<AttrsSpec>,
}

impl OpPat {
    /// A native-source op pattern with no args.
    pub fn new(op: impl Into<String>) -> OpPat {
        OpPat {
            op: op.into(),
            dim: None,
            args: Vec::new(),
            source: PatSource::Native,
            lit: None,
            attrs: None,
        }
    }

    /// Set the dim-slot pattern explicitly.
    pub fn dim(mut self, dim: Pat) -> OpPat {
        self.dim = Some(Box::new(dim));
        self
    }

    /// Set the argument list (replaces any previous).
    pub fn args(mut self, args: Vec<Pat>) -> OpPat {
        self.args = args;
        self
    }

    /// Append one argument.
    pub fn arg(mut self, arg: Pat) -> OpPat {
        self.args.push(arg);
        self
    }

    /// Set the source tag.
    pub fn source(mut self, source: PatSource) -> OpPat {
        self.source = source;
        self
    }

    /// Pin the `lit` literal payload.
    pub fn lit(mut self, lit: impl Into<Value>) -> OpPat {
        self.lit = Some(lit.into());
        self
    }

    /// Pin the `attrs` payload to a literal JSON value.
    pub fn attrs_lit(mut self, attrs: impl Into<Value>) -> OpPat {
        self.attrs = Some(AttrsSpec::Lit(attrs.into()));
        self
    }

    /// Bind the whole `attrs` payload to an attribute metavariable.
    pub fn attrs_bind(mut self, name: impl Into<String>) -> OpPat {
        self.attrs = Some(AttrsSpec::Bind(name.into()));
        self
    }
}

/// Why a [`Pat`] failed to decode from its JSON record form.
#[derive(Clone, Debug, PartialEq)]
pub enum PatError {
    /// The node is not a `{"mvar"}`/`{"const"}`/`{"op"}` record.
    UnknownNode(String),
    /// An `{"op"}` record is missing a field or carries a wrong type.
    MalformedOp(String),
    /// The `source` tag is not one of
    /// `native`/`foreign`/`judgment`/`dynamic`.
    UnknownSource(String),
    /// The `attrs` spec is not `null`, `{"lit": v}`, or `{"bind": n}`.
    UnknownAttrs(String),
}

impl std::fmt::Display for PatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatError::UnknownNode(desc) => write!(f, "not a pattern record: {desc}"),
            PatError::MalformedOp(msg) => write!(f, "malformed op pattern: {msg}"),
            PatError::UnknownSource(s) => write!(f, "unknown pattern source tag: {s:?}"),
            PatError::UnknownAttrs(desc) => write!(f, "unknown attrs spec: {desc}"),
        }
    }
}

impl std::error::Error for PatError {}

/// The `source` tag string of a [`PatSource`] (the record spelling).
fn source_name(source: &PatSource) -> &'static str {
    match source {
        PatSource::Native => "native",
        PatSource::Foreign(_) => "foreign",
        PatSource::Judgment => "judgment",
        PatSource::Dynamic => "dynamic",
    }
}

/// Parse a `source` record tag; `foreign` requires the sibling
/// `"foreign"` name field (checked by the caller).
fn source_from(name: &str, foreign: Option<&str>) -> Result<PatSource, PatError> {
    match name {
        "native" => Ok(PatSource::Native),
        "foreign" => match foreign {
            Some(n) => Ok(PatSource::Foreign(n.to_string())),
            None => Err(PatError::MalformedOp(
                "source \"foreign\" requires a \"foreign\" name".to_string(),
            )),
        },
        "judgment" => Ok(PatSource::Judgment),
        "dynamic" => Ok(PatSource::Dynamic),
        other => Err(PatError::UnknownSource(other.to_string())),
    }
}

impl Pat {
    /// Encode this pattern as its JSON record form — the `Pat`
    /// authoring view, never the compiled bridge encoding (design
    /// doc §1): `{"mvar": name}`, `{"const": <json>}`, or
    /// `{"op", "dim", "args", "source", "foreign"?, "lit", "attrs"}`.
    pub fn to_data(&self) -> Value {
        match self {
            Pat::Var(name) => serde_json::json!({"mvar": name}),
            Pat::Const(value) => serde_json::json!({"const": value}),
            Pat::Op(op) => {
                let mut rec = serde_json::Map::new();
                rec.insert("op".to_string(), Value::from(op.op.as_str()));
                rec.insert(
                    "dim".to_string(),
                    match &op.dim {
                        Some(d) => d.to_data(),
                        None => Value::Null,
                    },
                );
                rec.insert(
                    "args".to_string(),
                    Value::Array(op.args.iter().map(Pat::to_data).collect()),
                );
                rec.insert("source".to_string(), Value::from(source_name(&op.source)));
                if let PatSource::Foreign(name) = &op.source {
                    rec.insert("foreign".to_string(), Value::from(name.as_str()));
                }
                rec.insert(
                    "lit".to_string(),
                    match &op.lit {
                        Some(v) => v.clone(),
                        None => Value::Null,
                    },
                );
                rec.insert(
                    "attrs".to_string(),
                    match &op.attrs {
                        Some(AttrsSpec::Lit(v)) => serde_json::json!({"lit": v}),
                        Some(AttrsSpec::Bind(name)) => serde_json::json!({"bind": name}),
                        None => Value::Null,
                    },
                );
                Value::Object(rec)
            }
        }
    }

    /// Decode a [`Pat`] from its JSON record form (see
    /// [`Pat::to_data`]).
    ///
    /// # Errors
    ///
    /// [`PatError`] on any malformed node — unknown shape, missing
    /// field, bad `source`/`attrs` spelling. Nothing is approximated.
    pub fn from_data(value: &Value) -> Result<Pat, PatError> {
        let obj = value
            .as_object()
            .ok_or_else(|| PatError::UnknownNode(format!("{value}")))?;
        if let Some(mvar) = obj.get("mvar") {
            let name = mvar
                .as_str()
                .ok_or_else(|| PatError::UnknownNode(format!("{value}")))?;
            return Ok(Pat::Var(name.to_string()));
        }
        if let Some(c) = obj.get("const") {
            return Ok(Pat::Const(c.clone()));
        }
        if obj.contains_key("op") {
            return op_pat_from_data(obj);
        }
        Err(PatError::UnknownNode(format!("{value}")))
    }
}

/// Decode the `{"op", ...}` record form into an [`OpPat`].
fn op_pat_from_data(obj: &serde_json::Map<String, Value>) -> Result<Pat, PatError> {
    let bad = |msg: &str| PatError::MalformedOp(msg.to_string());
    let op = obj
        .get("op")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("\"op\" must be a string"))?
        .to_string();
    let dim = match obj.get("dim") {
        None | Some(Value::Null) => None,
        Some(v) => Some(Box::new(Pat::from_data(v)?)),
    };
    let args = match obj.get("args") {
        None => Vec::new(),
        Some(Value::Array(items)) => items
            .iter()
            .map(Pat::from_data)
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => return Err(bad("\"args\" must be an array")),
    };
    let source_name = obj
        .get("source")
        .map(|v| v.as_str().ok_or_else(|| bad("\"source\" must be a string")))
        .transpose()?
        .unwrap_or("native");
    let foreign = obj.get("foreign").and_then(Value::as_str);
    let source = source_from(source_name, foreign)?;
    let lit = match obj.get("lit") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.clone()),
    };
    let attrs = match obj.get("attrs") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let spec = v
                .as_object()
                .ok_or_else(|| PatError::UnknownAttrs(format!("{v}")))?;
            if let Some(l) = spec.get("lit") {
                Some(AttrsSpec::Lit(l.clone()))
            } else if let Some(b) = spec.get("bind") {
                let name = b
                    .as_str()
                    .ok_or_else(|| PatError::UnknownAttrs(format!("{v}")))?;
                Some(AttrsSpec::Bind(name.to_string()))
            } else {
                return Err(PatError::UnknownAttrs(format!("{v}")));
            }
        }
    };
    Ok(Pat::Op(OpPat {
        op,
        dim,
        args,
        source,
        lit,
        attrs,
    }))
}

/// Which side of a law a pattern compiles for — controls the `dim`
/// default (see module docs).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Side {
    Lhs,
    Rhs,
}

/// Mutable state shared across one law's LHS+RHS compilation: the
/// don't-care name mint and the LHS root's dim metavariable (what RHS
/// `dim: None` resolves to).
pub(crate) struct Ctx {
    fresh: u64,
    /// The metavariable name bound to the LHS root op's dim slot —
    /// `None` when the LHS root is not an op or pins a non-var dim.
    pub(crate) root_dim: Option<Rc<str>>,
}

impl Ctx {
    pub(crate) fn new() -> Self {
        Ctx {
            fresh: 0,
            root_dim: None,
        }
    }

    /// Mint a don't-care metavariable name (`__dim_N`, `__fresh_N` —
    /// namespaced so they cannot collide with law-authored vars).
    fn fresh(&mut self) -> Rc<str> {
        let name = format!("__fresh_{}", self.fresh);
        self.fresh += 1;
        Rc::from(name)
    }
}

/// Compile a [`Pat`] into the e-graph's internal pattern term.
///
/// `is_root` marks the law-side root node (its `dim: None` becomes the
/// root dim variable on the LHS / the inherited dim on the RHS).
pub(crate) fn compile(pat: &Pat, ctx: &mut Ctx, side: Side, is_root: bool) -> Term {
    match pat {
        Pat::Var(name) => Term::var(name.as_str()),
        Pat::Const(value) => from_kernel(&KernelTerm::constant(value.clone())),
        Pat::Op(op) => {
            let dim = match &op.dim {
                Some(d) => compile(d, ctx, side, false),
                None => match side {
                    // LHS: the root op's dim IS the root dim variable
                    // (recorded in `ctx.root_dim` by [`crate::law`]);
                    // nested dims are fresh don't-cares.
                    Side::Lhs if is_root => ctx
                        .root_dim
                        .clone()
                        .map(Term::var)
                        .unwrap_or_else(|| Term::var(ctx.fresh())),
                    Side::Lhs => Term::var(ctx.fresh()),
                    // RHS: inherit the matched root's dim. When there is
                    // no root dim var the metavariable stays unbound
                    // and the firing is skipped (honest, never wrong).
                    Side::Rhs => ctx
                        .root_dim
                        .clone()
                        .map(Term::var)
                        .unwrap_or_else(|| Term::var(ctx.fresh())),
                },
            };
            let mut args = Vec::with_capacity(op.args.len() + 1);
            args.push(dim);
            args.extend(op.args.iter().map(|a| compile(a, ctx, side, false)));

            let mut attrs: Vec<(Rc<str>, TermAttr)> =
                vec![(Rc::from("source"), TermAttr::Lit(source_attr(&op.source)))];
            if let PatSource::Foreign(name) = &op.source {
                attrs.push(attr("foreign", name.as_str()));
            }
            if let Some(lit) = &op.lit {
                attrs.push(attr("lit", AttrValue::from(lit)));
            }
            match &op.attrs {
                Some(AttrsSpec::Lit(v)) => attrs.push(attr("__attrs__", AttrValue::from(v))),
                Some(AttrsSpec::Bind(name)) => attrs.push(attr_meta("__attrs__", name)),
                None => {}
            }
            Term::op(op.op.as_str(), args, attrs)
        }
    }
}

/// The `source` attr literal for a [`PatSource`].
fn source_attr(source: &PatSource) -> AttrValue {
    AttrValue::from(match source {
        PatSource::Native => "native",
        PatSource::Foreign(_) => "foreign",
        PatSource::Judgment => "judgment",
        PatSource::Dynamic => "dynamic",
    })
}

/// Determine the metavariable name the LHS root op's dim slot binds
/// to — what RHS `dim: None` inherits. `None` when the LHS root is
/// not an op or pins a non-var dim.
pub(crate) fn root_dim_var(lhs: &Pat, ctx: &mut Ctx) -> Option<Rc<str>> {
    match lhs {
        Pat::Op(op) => match op.dim.as_deref() {
            None => Some(ctx.fresh()),
            Some(Pat::Var(name)) => Some(Rc::from(name.as_str())),
            Some(_) => None,
        },
        _ => None,
    }
}

/// Collect the metavariable names a pattern mentions (term vars and
/// attr binds, `"$attr:<name>"`-prefixed) — used by [`crate::law`]
/// to validate that every RHS metavariable is LHS-bound.
pub(crate) fn metavar_names(pat: &Pat, out: &mut BTreeSet<String>) {
    match pat {
        Pat::Var(name) => {
            out.insert(name.clone());
        }
        Pat::Const(_) => {}
        Pat::Op(op) => {
            if let Some(dim) = &op.dim {
                metavar_names(dim, out);
            }
            if let Some(AttrsSpec::Bind(name)) = &op.attrs {
                out.insert(format!("$attr:{name}"));
            }
            for a in &op.args {
                metavar_names(a, out);
            }
        }
    }
}
