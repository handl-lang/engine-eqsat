//! [`LawSet`] — an ordered collection of [`Law`]s, loadable from the
//! JSON record corpus (design doc §1: `laws/*.json` is the reference
//! corpus until `kit-eqsat` exists).
//!
//! [`LawSet::from_json`] accepts three top-level shapes:
//!
//! * an array of law records — `[record, record, …]`;
//! * a single law record object — `{"version": 2, "kind": "law", …}`;
//! * an envelope — `{"laws": [record, …]}` (extra envelope keys are
//!   ignored; the envelope is a container, not a record).
//!
//! A record that fails to decode fails the whole load with
//! [`RecordError::SetMember`] naming its index — declines are named,
//! never partial-and-silent.

use std::collections::BTreeSet;

use handl_egraph::Rewrite;
use serde_json::Value;

use crate::law::{Law, LawError, RecordError};

/// An ordered set of laws — the `laws` argument of the
/// [`crate::SearchEngine`] contract.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LawSet {
    laws: Vec<Law>,
}

impl LawSet {
    /// An empty set.
    pub fn new() -> LawSet {
        LawSet { laws: Vec::new() }
    }

    /// A set from an explicit law list.
    pub fn from_laws(laws: Vec<Law>) -> LawSet {
        LawSet { laws }
    }

    /// The laws, in order.
    pub fn laws(&self) -> &[Law] {
        &self.laws
    }

    /// Number of laws.
    pub fn len(&self) -> usize {
        self.laws.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.laws.is_empty()
    }

    /// Iterate the laws.
    pub fn iter(&self) -> impl Iterator<Item = &Law> {
        self.laws.iter()
    }

    /// Compile every law to the e-graph's internal [`Rewrite`]s.
    ///
    /// # Errors
    ///
    /// The first [`LawError`] any law reports — including
    /// [`LawError::GuardUnsupported`]/[`LawError::GuardMalformed`]
    /// (a `cond`/`dspec` tree outside the stage-1 vocabulary) and
    /// [`LawError::Unserializable`] (the record's honesty flags).
    /// One bad law refuses the set: a partially-applied rule set is
    /// a silent weakening.
    pub fn compile(&self) -> Result<Vec<Rewrite>, LawError> {
        self.laws.iter().map(Law::compile).collect()
    }

    /// Encode the set as a JSON array of law records.
    pub fn to_data(&self) -> Value {
        Value::Array(self.laws.iter().map(Law::to_data).collect())
    }

    /// Decode a [`LawSet`] from its JSON text (see the module docs
    /// for the accepted top-level shapes).
    ///
    /// # Errors
    ///
    /// [`RecordError::Json`] on unparsable input,
    /// [`RecordError::Shape`] on an unrecognised top-level shape, and
    /// [`RecordError::SetMember`] wrapping the first law record that
    /// fails to decode.
    pub fn from_json(text: &str) -> Result<LawSet, RecordError> {
        let value: Value = serde_json::from_str(text)?;
        LawSet::from_data(&value)
    }

    /// Decode a [`LawSet`] from a JSON value.
    ///
    /// # Errors
    ///
    /// As [`LawSet::from_json`].
    pub fn from_data(value: &Value) -> Result<LawSet, RecordError> {
        let records: Vec<&Value> = match value {
            Value::Array(items) => items.iter().collect(),
            Value::Object(obj) if obj.contains_key("laws") => match obj.get("laws") {
                Some(Value::Array(items)) => items.iter().collect(),
                _ => {
                    return Err(RecordError::Shape(
                        "\"laws\" envelope must hold an array".to_string(),
                    ));
                }
            },
            Value::Object(_) => vec![value],
            _ => {
                return Err(RecordError::Shape(format!(
                    "expected an array or object of law records, got {value}"
                )));
            }
        };
        let mut laws = Vec::with_capacity(records.len());
        for (index, rec) in records.into_iter().enumerate() {
            let name = rec.get("name").and_then(Value::as_str).map(str::to_string);
            match Law::from_data(rec) {
                Ok(law) => laws.push(law),
                Err(source) => {
                    return Err(RecordError::SetMember {
                        index,
                        name,
                        source: Box::new(source),
                    });
                }
            }
        }
        Ok(LawSet { laws })
    }
}

impl From<Vec<Law>> for LawSet {
    fn from(laws: Vec<Law>) -> LawSet {
        LawSet::from_laws(laws)
    }
}

impl FromIterator<Law> for LawSet {
    fn from_iter<I: IntoIterator<Item = Law>>(iter: I) -> LawSet {
        LawSet::from_laws(iter.into_iter().collect())
    }
}

impl IntoIterator for LawSet {
    type Item = Law;
    type IntoIter = std::vec::IntoIter<Law>;

    fn into_iter(self) -> Self::IntoIter {
        self.laws.into_iter()
    }
}

impl<'a> IntoIterator for &'a LawSet {
    type Item = &'a Law;
    type IntoIter = std::slice::Iter<'a, Law>;

    fn into_iter(self) -> Self::IntoIter {
        self.laws.iter()
    }
}

/// Collect the union of every law's classification tags —
/// convenience for corpus inspection.
pub fn all_tags(set: &LawSet) -> BTreeSet<String> {
    set.laws
        .iter()
        .flat_map(|l| l.tags.iter().cloned())
        .collect()
}
