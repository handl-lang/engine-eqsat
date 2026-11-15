//! [`Law`] — a named rewrite rule in the [`crate::pat`] pattern
//! language, compiled to the e-graph's internal [`Rewrite`].
//!
//! The law **record** is the JSON interchange format of design doc
//! §1 (ADR-0177): [`Law::to_data`]/[`Law::from_data`] round-trip the
//! `LAW_FORMAT = 2` record shape (`version`/`kind`/`name`/`law`/`lhs`/
//! `rhs`/`cond`/`dspec`/`tags`/`derivation`/`cert`/`error_bound`/
//! `bound_norm`/`serializable`/`missing_hooks`), mirroring
//! `mlopt_core.laws.serialize`. Patterns are spelled in the [`Pat`]
//! authoring view — never the compiled bridge encoding.
//!
//! Guards are **interpreted**: `cond`/`dspec` are the declarative
//! guard trees of [`crate::guard`] (the stage-1 `cond.py` port over
//! [`handl_egraph::BoundEnv`]) — [`Law::compile`] parses them into
//! `Rewrite::with_check`/`with_derive` hooks, so a guarded law fires
//! only when its side condition holds. A guard the stage-1
//! vocabulary does not cover — an unported guard op
//! or a malformed node — refuses with [`LawError::GuardUnsupported`]/
//! [`LawError::GuardMalformed`]: firing without the side condition
//! would be a silent weakening. Likewise `serializable: false` or a
//! non-empty `missing_hooks` refuses with
//! [`LawError::Unserializable`]: the record's honesty flags port
//! verbatim and are honoured, not dropped.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use handl_egraph::{AttrValue, BoundEnv, Derived, Rewrite};
use serde_json::Value;

use crate::pat::{
    self, AttrBinding, AttrField, Ctx, Pat, PatError, Side, metavar_names, root_dim_var,
};

/// The law-record format version this codec reads and writes —
/// `mlopt_core.laws.serialize.LAW_FORMAT`.
pub const LAW_FORMAT: u64 = 2;

/// A rewrite law: `lhs -> rhs` over kernel-`Term` patterns.
///
/// `name` is the rule's unique identity (budget key + provenance);
/// `law` is the optional certificate/law-bank name it discharges
/// (metadata, carried through to `Rewrite::law`).
#[derive(Clone, Debug, PartialEq)]
pub struct Law {
    /// Rule name — must be unique within a rule set.
    pub name: String,
    /// The law/lemma name this rule certifies (metadata).
    pub law: Option<String>,
    /// Left-hand pattern.
    pub lhs: Pat,
    /// Right-hand pattern.
    pub rhs: Pat,
    /// Intrinsic classification tags (symmetry / expansive / fusion /
    /// …) — domain metadata carried opaquely, like `Rewrite::tags`.
    pub tags: BTreeSet<String>,
    /// The `cond` side-condition tree — the declarative guard of
    /// [`crate::guard::Cond`], carried as the raw JSON the record
    /// stores and interpreted by [`Law::compile`].
    pub cond: Option<Value>,
    /// The `dspec` derive-spec tree — the declarative derive of
    /// [`crate::guard::Dspec`], interpreted like [`Law::cond`].
    pub dspec: Option<Value>,
    /// The recorded derivation chain (record-level provenance — the
    /// `Rewrite` carries no such field; it round-trips through the
    /// record only).
    pub derivation: Vec<String>,
    /// An optional replayable-certificate record, carried opaquely —
    /// record-level provenance, never folded into the 2-cell (the
    /// `cert_to_data` posture of `law_from_data`).
    pub cert: Option<Value>,
    /// Bounded-error axis (ε-laws): `‖lhs − rhs‖ ≤ error_bound` in
    /// the norm named by [`Law::bound_norm`]. `None` for exact rules.
    pub error_bound: Option<f64>,
    /// The norm `error_bound` is measured in (e.g. `"spectral"`).
    pub bound_norm: String,
    /// Whether the record round-trips behaviourally — `false` when
    /// procedural `check=`/`derive=` hooks could not be folded into
    /// `cond`/`dspec` (the `missing_hooks` honesty boundary). A
    /// non-serializable law may be inspected but must not fire
    /// silently: [`Law::compile`] refuses it.
    pub serializable: bool,
    /// Names of hooks the record cannot carry (`"check"`/`"derive"`).
    pub missing_hooks: BTreeSet<String>,
}

/// Why a [`Law`] failed to compile or decode.
#[derive(Clone, Debug, PartialEq)]
pub enum LawError {
    /// The RHS mentions a metavariable the LHS never binds — the
    /// rewrite could not instantiate it.
    UnboundRhsVar(String),
    /// The law's `cond`/`dspec` names an op the guard interpreter
    /// does not cover — a flat-map/repeat pairing guard or a
    /// tensor-backend channel. Firing would silently drop the side
    /// condition.
    GuardUnsupported {
        /// The law's name.
        law: String,
        /// What the interpreter refused (the op and why).
        detail: String,
    },
    /// The law's `cond`/`dspec` tree is malformed — a bad node
    /// shape, arity, literal type, or an undefined op name.
    GuardMalformed {
        /// The law's name.
        law: String,
        /// What was malformed.
        detail: String,
    },
    /// The record flagged itself non-serializable (`serializable:
    /// false` or non-empty `missing_hooks`) — procedural hooks the
    /// data cannot carry; firing would silently weaken the law.
    Unserializable {
        /// The law's name.
        law: String,
        /// The hooks the record names as missing.
        missing: Vec<String>,
    },
}

impl std::fmt::Display for LawError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LawError::UnboundRhsVar(name) => {
                write!(f, "RHS metavariable '{name}' is not bound by the LHS")
            }
            LawError::GuardUnsupported { law, detail } => {
                write!(f, "law '{law}' guard unsupported: {detail}")
            }
            LawError::GuardMalformed { law, detail } => {
                write!(f, "law '{law}' guard malformed: {detail}")
            }
            LawError::Unserializable { law, missing } => write!(
                f,
                "law '{law}' is not serializable (missing hooks: {})",
                missing.join(", ")
            ),
        }
    }
}

impl std::error::Error for LawError {}

/// Why a law **record** failed to decode — distinct from
/// [`LawError`], which covers compile-time refusals.
#[derive(Debug)]
pub enum RecordError {
    /// The input was not valid JSON.
    Json(serde_json::Error),
    /// The top-level value is not a law record (or, for set loaders,
    /// an array/object of records).
    Shape(String),
    /// `version` is absent or not [`LAW_FORMAT`].
    Version(Option<u64>),
    /// `kind` is present and not `"law"`.
    Kind(String),
    /// A required field is missing or has the wrong type.
    Malformed(String),
    /// A `lhs`/`rhs`/`dim` pattern failed to decode.
    Pattern {
        /// Which pattern position failed (`"lhs"`, `"rhs"`, `"dim"`).
        side: &'static str,
        /// The underlying pattern error.
        source: PatError,
    },
    /// A law record inside a [`crate::lawset::LawSet`] failed; `index`
    /// is its position in the array.
    SetMember {
        /// Zero-based position in the record array.
        index: usize,
        /// The record's `name`, when it decoded far enough to have one.
        name: Option<String>,
        /// The underlying record error.
        source: Box<RecordError>,
    },
}

impl std::fmt::Display for RecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecordError::Json(e) => write!(f, "invalid JSON: {e}"),
            RecordError::Shape(desc) => write!(f, "bad record shape: {desc}"),
            RecordError::Version(v) => {
                write!(f, "bad law record version: {v:?} (expected {LAW_FORMAT})")
            }
            RecordError::Kind(k) => write!(f, "unknown object kind: {k:?}"),
            RecordError::Malformed(msg) => write!(f, "malformed law record: {msg}"),
            RecordError::Pattern { side, source } => {
                write!(f, "bad {side} pattern: {source}")
            }
            RecordError::SetMember {
                index,
                name,
                source,
            } => match name {
                Some(n) => write!(f, "law record {index} ({n}): {source}"),
                None => write!(f, "law record {index}: {source}"),
            },
        }
    }
}

impl std::error::Error for RecordError {}

impl From<serde_json::Error> for RecordError {
    fn from(e: serde_json::Error) -> Self {
        RecordError::Json(e)
    }
}

/// Fetch a required string field.
fn req_str<'a>(
    obj: &'a serde_json::Map<String, Value>,
    field: &'static str,
) -> Result<&'a str, RecordError> {
    obj.get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| RecordError::Malformed(format!("\"{field}\" must be a string")))
}

/// Fetch an optional string-or-null field.
fn opt_str(obj: &serde_json::Map<String, Value>, field: &str) -> Option<String> {
    obj.get(field).and_then(Value::as_str).map(str::to_string)
}

/// Fetch an optional JSON array-of-strings field (`null`/absent →
/// `None`; a non-array or non-string element is a [`RecordError`]).
fn opt_str_list(
    obj: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<Vec<String>>, RecordError> {
    match obj.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| RecordError::Malformed(format!("\"{field}\" must be strings")))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(_) => Err(RecordError::Malformed(format!(
            "\"{field}\" must be an array of strings"
        ))),
    }
}

impl Law {
    /// A law `name: lhs -> rhs` with no law metadata or tags.
    pub fn new(name: impl Into<String>, lhs: Pat, rhs: Pat) -> Law {
        Law {
            name: name.into(),
            law: None,
            lhs,
            rhs,
            tags: BTreeSet::new(),
            cond: None,
            dspec: None,
            derivation: Vec::new(),
            cert: None,
            error_bound: None,
            bound_norm: "spectral".to_string(),
            serializable: true,
            missing_hooks: BTreeSet::new(),
        }
    }

    /// Set the certificate/law-bank name (`Rewrite::law`).
    pub fn with_law(mut self, law: impl Into<String>) -> Law {
        self.law = Some(law.into());
        self
    }

    /// Add a classification tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Law {
        self.tags.insert(tag.into());
        self
    }

    /// Carry a `cond` guard tree (raw record JSON — the
    /// [`crate::guard`] interpreter reads it at [`Law::compile`]).
    pub fn with_cond(mut self, cond: impl Into<Value>) -> Law {
        self.cond = Some(cond.into());
        self
    }

    /// Carry a `dspec` derive-spec tree (see [`Law::with_cond`]).
    pub fn with_dspec(mut self, dspec: impl Into<Value>) -> Law {
        self.dspec = Some(dspec.into());
        self
    }

    /// Append a derivation-chain entry (record-level provenance).
    pub fn with_derivation(mut self, step: impl Into<String>) -> Law {
        self.derivation.push(step.into());
        self
    }

    /// Attach an opaque certificate record (record-level provenance —
    /// never folded into the 2-cell).
    pub fn with_cert(mut self, cert: impl Into<Value>) -> Law {
        self.cert = Some(cert.into());
        self
    }

    /// Mark as an ε-law: `‖lhs − rhs‖ ≤ bound` in `bound_norm`.
    pub fn with_error_bound(mut self, bound: f64, norm: impl Into<String>) -> Law {
        self.error_bound = Some(bound);
        self.bound_norm = norm.into();
        self
    }

    /// Encode this law as its `LAW_FORMAT = 2` JSON record — the
    /// shape `mlopt_core.laws.serialize.law_to_data` emits, with the
    /// pattern pair in the [`Pat`] authoring view.
    ///
    /// `cond`/`dspec` round-trip as the raw guard trees;
    /// `serializable`/`missing_hooks` port verbatim.
    pub fn to_data(&self) -> Value {
        let mut rec = serde_json::Map::new();
        rec.insert("version".to_string(), Value::from(LAW_FORMAT));
        rec.insert("kind".to_string(), Value::from("law"));
        rec.insert("name".to_string(), Value::from(self.name.as_str()));
        rec.insert(
            "law".to_string(),
            match &self.law {
                Some(l) => Value::from(l.as_str()),
                None => Value::Null,
            },
        );
        rec.insert("lhs".to_string(), self.lhs.to_data());
        rec.insert("rhs".to_string(), self.rhs.to_data());
        rec.insert("cond".to_string(), self.cond.clone().unwrap_or(Value::Null));
        rec.insert(
            "dspec".to_string(),
            self.dspec.clone().unwrap_or(Value::Null),
        );
        rec.insert(
            "tags".to_string(),
            Value::Array(self.tags.iter().map(|t| Value::from(t.as_str())).collect()),
        );
        rec.insert(
            "derivation".to_string(),
            Value::Array(
                self.derivation
                    .iter()
                    .map(|d| Value::from(d.as_str()))
                    .collect(),
            ),
        );
        rec.insert("cert".to_string(), self.cert.clone().unwrap_or(Value::Null));
        rec.insert(
            "error_bound".to_string(),
            match self.error_bound {
                Some(b) => serde_json::Number::from_f64(b)
                    .map(Value::Number)
                    .unwrap_or(Value::Null),
                None => Value::Null,
            },
        );
        rec.insert(
            "bound_norm".to_string(),
            Value::from(self.bound_norm.as_str()),
        );
        rec.insert("serializable".to_string(), Value::from(self.serializable));
        rec.insert(
            "missing_hooks".to_string(),
            Value::Array(
                self.missing_hooks
                    .iter()
                    .map(|h| Value::from(h.as_str()))
                    .collect(),
            ),
        );
        Value::Object(rec)
    }

    /// Decode a [`Law`] from its record form (see [`Law::to_data`]).
    ///
    /// `cond`/`dspec` are stored verbatim — the
    /// [`crate::guard`] interpreter reads them at [`Law::compile`].
    /// `serializable`/`missing_hooks` are honoured: a flagged record
    /// loads (for inspection) but refuses to compile.
    ///
    /// # Errors
    ///
    /// [`RecordError`] on a bad version/kind, missing or mistyped
    /// field, or a malformed `lhs`/`rhs` pattern.
    pub fn from_data(value: &Value) -> Result<Law, RecordError> {
        let obj = value
            .as_object()
            .ok_or_else(|| RecordError::Shape(format!("{value}")))?;
        let version = obj.get("version").and_then(Value::as_u64);
        if version != Some(LAW_FORMAT) {
            return Err(RecordError::Version(version));
        }
        if let Some(kind) = obj.get("kind").and_then(Value::as_str)
            && kind != "law"
        {
            return Err(RecordError::Kind(kind.to_string()));
        }
        let name = req_str(obj, "name")?.to_string();
        let law = opt_str(obj, "law");
        let lhs = Pat::from_data(
            obj.get("lhs")
                .ok_or_else(|| RecordError::Malformed("\"lhs\" is required".into()))?,
        )
        .map_err(|source| RecordError::Pattern {
            side: "lhs",
            source,
        })?;
        let rhs = Pat::from_data(
            obj.get("rhs")
                .ok_or_else(|| RecordError::Malformed("\"rhs\" is required".into()))?,
        )
        .map_err(|source| RecordError::Pattern {
            side: "rhs",
            source,
        })?;
        let tags = opt_str_list(obj, "tags")?
            .unwrap_or_default()
            .into_iter()
            .collect();
        let derivation = opt_str_list(obj, "derivation")?.unwrap_or_default();
        let missing_hooks = opt_str_list(obj, "missing_hooks")?
            .unwrap_or_default()
            .into_iter()
            .collect();
        let error_bound = match obj.get("error_bound") {
            None | Some(Value::Null) => None,
            Some(v) => Some(v.as_f64().ok_or_else(|| {
                RecordError::Malformed("\"error_bound\" must be a number".into())
            })?),
        };
        Ok(Law {
            name,
            law,
            lhs,
            rhs,
            tags,
            cond: obj.get("cond").filter(|v| !v.is_null()).cloned(),
            dspec: obj.get("dspec").filter(|v| !v.is_null()).cloned(),
            derivation,
            cert: obj.get("cert").filter(|v| !v.is_null()).cloned(),
            error_bound,
            bound_norm: opt_str(obj, "bound_norm").unwrap_or_else(|| "spectral".to_string()),
            serializable: obj
                .get("serializable")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            missing_hooks,
        })
    }

    /// Decode a [`Law`] from a JSON record (see [`Law::from_data`]).
    ///
    /// # Errors
    ///
    /// [`RecordError::Json`] on unparsable input, else whatever
    /// [`Law::from_data`] reports.
    pub fn from_json(text: &str) -> Result<Law, RecordError> {
        let value: Value = serde_json::from_str(text)?;
        Law::from_data(&value)
    }

    /// Compile to the e-graph's internal [`Rewrite`].
    ///
    /// # Errors
    ///
    /// * [`LawError::Unserializable`] — the record flagged procedural
    ///   hooks the data cannot carry (`serializable: false` or
    ///   `missing_hooks` non-empty).
    /// * [`LawError::GuardUnsupported`] — `cond`/`dspec` names a
    ///   guard op outside the ported vocabulary (the flat-map/repeat
    ///   pairing guards and tensor-value channels).
    /// * [`LawError::GuardMalformed`] — `cond`/`dspec` is not a
    ///   well-formed guard tree.
    /// * [`LawError::UnboundRhsVar`] — the RHS names a metavariable
    ///   (term or attr) absent from the LHS.
    pub fn compile(&self) -> Result<Rewrite, LawError> {
        if !self.serializable || !self.missing_hooks.is_empty() {
            return Err(LawError::Unserializable {
                law: self.name.clone(),
                missing: self.missing_hooks.iter().cloned().collect(),
            });
        }
        // Parse the declarative guards once — a refusal here is a
        // compile error, never an eval-time surprise.
        let cond = self
            .cond
            .as_ref()
            .map(|v| {
                crate::guard::Cond::from_data(v).map_err(|e| match e {
                    crate::guard::GuardError::Malformed(d) => LawError::GuardMalformed {
                        law: self.name.clone(),
                        detail: format!("cond: {d}"),
                    },
                    crate::guard::GuardError::Unsupported(d) => LawError::GuardUnsupported {
                        law: self.name.clone(),
                        detail: format!("cond: {d}"),
                    },
                })
            })
            .transpose()?;
        let dspec = self
            .dspec
            .as_ref()
            .map(|v| {
                crate::guard::Dspec::from_data(v).map_err(|e| match e {
                    crate::guard::GuardError::Malformed(d) => LawError::GuardMalformed {
                        law: self.name.clone(),
                        detail: format!("dspec: {d}"),
                    },
                    crate::guard::GuardError::Unsupported(d) => LawError::GuardUnsupported {
                        law: self.name.clone(),
                        detail: format!("dspec: {d}"),
                    },
                })
            })
            .transpose()?;
        let mut lhs_vars = BTreeSet::new();
        metavar_names(&self.lhs, &mut lhs_vars);
        // `dspec` bindings count as bound — `$leaf:H` mints term
        // metavar `H`, `$attr:X`/bare `X` binds attr metavar `X`
        // (the Python `eval_derive` contract).
        if let Some(d) = &dspec {
            lhs_vars.extend(d.bound_vars());
        }
        let mut rhs_vars = BTreeSet::new();
        metavar_names(&self.rhs, &mut rhs_vars);
        for name in &rhs_vars {
            if !lhs_vars.contains(name) {
                return Err(LawError::UnboundRhsVar(name.clone()));
            }
        }
        let mut ctx = Ctx::new();
        // The LHS root op's dim metavariable — RHS `dim: None`
        // inherits it (dim-preserving rewrites; see `pat` docs).
        ctx.root_dim = root_dim_var(&self.lhs, &mut ctx);
        let mut rule = Rewrite::new(
            self.name.as_str(),
            pat::compile(&self.lhs, &mut ctx, Side::Lhs, true),
            pat::compile(&self.rhs, &mut ctx, Side::Rhs, true),
        );
        rule.law = self.law.clone().unwrap_or_else(|| self.name.clone());
        rule.tags = self.tags.clone();
        rule.error_bound = self.error_bound;
        rule.bound_norm = self.bound_norm.clone();
        let lhs_attrs: Vec<AttrBinding> = ctx
            .attr_bindings
            .iter()
            .filter(|b| b.side == Side::Lhs)
            .cloned()
            .collect();
        let rhs_attrs: Vec<AttrBinding> = ctx
            .attr_bindings
            .iter()
            .filter(|b| b.side == Side::Rhs)
            .cloned()
            .collect();
        // The declarative hooks — `check` before `derive`, the order
        // `EGraph::apply_rule` runs them (a vetoed check never pays
        // for the derive). `AttrsSpec::Fields` adds its exact-keyset /
        // literal / attr-metavar check to `check` and its RHS payload
        // synthesis to `derive`.
        if cond.is_some() || !lhs_attrs.is_empty() {
            let check_attrs = lhs_attrs.clone();
            rule = rule.with_check(move |env| {
                let expanded;
                let env = if check_attrs.is_empty() {
                    env
                } else {
                    match expand_attr_fields(env, &check_attrs) {
                        Some(e) => {
                            expanded = e;
                            &expanded
                        }
                        None => return false,
                    }
                };
                match &cond {
                    Some(c) => c.eval(env),
                    None => true,
                }
            });
        }
        if dspec.is_some() || !rhs_attrs.is_empty() {
            let derive_attrs = lhs_attrs;
            rule = rule.with_derive(move |env| {
                let expanded;
                let env = if derive_attrs.is_empty() {
                    env
                } else {
                    match expand_attr_fields(env, &derive_attrs) {
                        Some(e) => {
                            expanded = e;
                            &expanded
                        }
                        None => return None,
                    }
                };
                let mut derived = match &dspec {
                    Some(d) => d.eval(env)?,
                    None => Vec::new(),
                };
                let mut attrs = env.attrs.clone();
                for d in &derived {
                    if let Derived::Attr(name, value) = d {
                        attrs.insert(name.clone(), value.clone());
                    }
                }
                for binding in &rhs_attrs {
                    derived.push(synthesize_attr_payload(&attrs, binding)?);
                }
                Some(derived)
            });
        }
        Ok(rule)
    }
}

/// Expand every LHS [`AttrsSpec::Fields`] payload into the shared
/// attr-metavar namespace: exact keyset, literal pins, and bound
/// fields must agree wherever they repeat (the Python `Op.make`
/// per-key attr-metavariable contract the e-graph's top-level
/// `TermAttr::Meta` cannot express directly).
fn expand_attr_fields(env: &BoundEnv, bindings: &[AttrBinding]) -> Option<BoundEnv> {
    let mut expanded = env.clone();
    for binding in bindings {
        let AttrValue::Object(fields) = env.attrs.get(binding.whole.as_ref())? else {
            return None;
        };
        if fields.len() != binding.fields.len()
            || !binding
                .fields
                .keys()
                .all(|k| fields.contains_key(k.as_str()))
        {
            return None;
        }
        for (name, field) in &binding.fields {
            let actual = fields.get(name.as_str())?;
            match field {
                AttrField::Lit(v) => {
                    if !actual.match_eq(&AttrValue::from(v)) {
                        return None;
                    }
                }
                AttrField::Bind(mvar) => {
                    let key = Rc::from(mvar.as_str());
                    match expanded.attrs.get(&key) {
                        Some(bound) if !bound.match_eq(actual) => return None,
                        Some(_) => {}
                        None => {
                            expanded.attrs.insert(key, actual.clone());
                        }
                    }
                }
            }
        }
    }
    Some(expanded)
}

/// Synthesize one RHS [`AttrsSpec::Fields`] payload from the
/// post-`dspec` attr namespace — the `Derived::Attr` the compiled
/// `__attrs__` metavariable needs for RHS instantiation.
fn synthesize_attr_payload(
    attrs: &BTreeMap<Rc<str>, AttrValue>,
    binding: &AttrBinding,
) -> Option<Derived> {
    let mut object = BTreeMap::new();
    for (name, field) in &binding.fields {
        let value = match field {
            AttrField::Lit(v) => AttrValue::from(v),
            AttrField::Bind(mvar) => attrs.get(mvar.as_str())?.clone(),
        };
        object.insert(Rc::from(name.as_str()), value);
    }
    Some(Derived::Attr(
        binding.whole.clone(),
        AttrValue::Object(Rc::new(object)),
    ))
}

/// Decode a [`Law`] from its record form — module-level spelling of
/// [`Law::from_data`] (design doc §1's `law::from_data`).
///
/// # Errors
///
/// [`RecordError`] — see [`Law::from_data`].
pub fn from_data(value: &Value) -> Result<Law, RecordError> {
    Law::from_data(value)
}
