//! [`Law`] — a named rewrite rule in the [`crate::pat`] pattern
//! language, compiled to the e-graph's internal [`Rewrite`].
//!
//! v1 laws are **unconditional**: the Python `cond.py` guard
//! machinery (`check`/`derive`/`dspec`) has no kit-data spelling yet
//! (spike §6 open question), so [`Law`] carries no hooks. When the
//! guard-format decision lands, `check`/`derive` slots belong here.

use std::collections::BTreeSet;

use handl_egraph::Rewrite;

use crate::pat::{self, Ctx, Pat, Side, metavar_names, root_dim_var};

/// A rewrite law: `lhs -> rhs` over kernel-`Term` patterns.
///
/// `name` is the rule's unique identity (budget key + provenance);
/// `law` is the optional certificate/law-bank name it discharges
/// (metadata, carried through to `Rewrite::law`).
#[derive(Clone, Debug)]
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
}

/// Why a [`Law`] failed to compile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LawError {
    /// The RHS mentions a metavariable the LHS never binds — the
    /// rewrite could not instantiate it.
    UnboundRhsVar(String),
}

impl std::fmt::Display for LawError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LawError::UnboundRhsVar(name) => {
                write!(f, "RHS metavariable '{name}' is not bound by the LHS")
            }
        }
    }
}

impl std::error::Error for LawError {}

impl Law {
    /// A law `name: lhs -> rhs` with no law metadata or tags.
    pub fn new(name: impl Into<String>, lhs: Pat, rhs: Pat) -> Law {
        Law {
            name: name.into(),
            law: None,
            lhs,
            rhs,
            tags: BTreeSet::new(),
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

    /// Compile to the e-graph's internal [`Rewrite`].
    ///
    /// # Errors
    ///
    /// [`LawError::UnboundRhsVar`] when the RHS names a metavariable
    /// (term or attr) absent from the LHS.
    pub(crate) fn compile(&self) -> Result<Rewrite, LawError> {
        let mut lhs_vars = BTreeSet::new();
        metavar_names(&self.lhs, &mut lhs_vars);
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
        Ok(rule)
    }
}
