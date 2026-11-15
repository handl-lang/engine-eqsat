//! The first law pack — unconditional ports of `mlopt_core`'s
//! `SIMPLIFICATION_RULES` identities (`eqsat/packages/mlopt-core/
//! src/mlopt_core/laws/tensor.py`), spelled in [`crate::pat`].
//!
//! These are the Python bank's guard-free laws: no `cond.py`
//! side-conditions, so they are the honest first batch for the Rust
//! interpretation (spike §7 step 2). Names and `law` strings are
//! carried verbatim so provenance lines up with the Python bank.
//!
//! Scope note: the Python patterns live in the *typed* `Op` IR; here
//! they are kernel `Term::Op`s — `dim` is child slot 0 (auto-bound,
//! RHS inherits the root dim) and `Const` leaves are JSON-valued.

use crate::law::Law;
use crate::pat::{OpPat, Pat};

/// `id_add`: `add(a, 0) → a` — additive identity.
///
/// Python: `Op.make("add", "a", Const(0)) → "a"`,
/// law "Additive identity: a + 0 = a."
pub fn id_add() -> Law {
    Law::new(
        "id_add",
        Pat::op(
            "add",
            vec![Pat::var("a"), Pat::constant(serde_json::json!(0))],
        ),
        Pat::var("a"),
    )
    .with_law("Additive identity: a + 0 = a.")
    .with_tag("simplification")
}

/// `id_mul`: `mul(a, 1) → a` — multiplicative identity.
///
/// Python: `Op.make("mul", "a", Const(1)) → "a"`,
/// law "Multiplicative identity: a * 1 = a."
pub fn id_mul() -> Law {
    Law::new(
        "id_mul",
        Pat::op(
            "mul",
            vec![Pat::var("a"), Pat::constant(serde_json::json!(1))],
        ),
        Pat::var("a"),
    )
    .with_law("Multiplicative identity: a * 1 = a.")
    .with_tag("simplification")
}

/// `double_neg`: `neg(neg(a)) → a` — involution.
///
/// Python: `Op.make("neg", Op.make("neg", "a")) → "a"`,
/// law "Double negation: ¬¬a = a (involution)."
pub fn double_neg() -> Law {
    Law::new(
        "double_neg",
        Pat::op("neg", vec![Pat::op("neg", vec![Pat::var("a")])]),
        Pat::var("a"),
    )
    .with_law("Double negation: ¬¬a = a (involution).")
    .with_tag("simplification")
}

/// `sub_to_add`: `sub(a, b) → add(a, neg(b))` — subtraction as
/// addition of inverse.
///
/// Python: `Op.make("sub", "a", "b") → Op.make("add", "a",
/// Op.make("neg", "b"))`, law "Subtraction as addition of inverse:
/// a - b = a + (-b)."
///
/// This law exercises the RHS-op path: the instantiated `add`/`neg`
/// inherit the matched root's dim (see [`crate::pat`] docs).
pub fn sub_to_add() -> Law {
    Law::new(
        "sub_to_add",
        Pat::op("sub", vec![Pat::var("a"), Pat::var("b")]),
        Pat::op(
            "add",
            vec![Pat::var("a"), Pat::op("neg", vec![Pat::var("b")])],
        ),
    )
    .with_law("Subtraction as addition of inverse: a - b = a + (-b).")
    .with_tag("simplification")
}

/// The guard-free `SIMPLIFICATION_RULES` identities shipped as the
/// default v1 law set.
pub fn simplification() -> Vec<Law> {
    vec![id_add(), id_mul(), double_neg(), sub_to_add()]
}

/// A demonstration law for the attrs-bind path (the spike's
/// `reshape → flatten`): `tensor.reshape(x, attrs=?a) →
/// tensor.flatten(x, attrs=?a)` — the `__attrs__` payload is bound
/// whole and re-emitted verbatim. Not part of `simplification()`;
/// it exists to pin the attrs round-trip in tests.
pub fn reshape_flatten() -> Law {
    Law::new(
        "reshape_flatten",
        Pat::Op(
            OpPat::new("tensor.reshape")
                .arg(Pat::var("x"))
                .attrs_bind("a"),
        ),
        Pat::Op(
            OpPat::new("tensor.flatten")
                .arg(Pat::var("x"))
                .attrs_bind("a"),
        ),
    )
    .with_law("reshape(x, attrs) = flatten(x, attrs) — attrs payload preserved.")
    .with_tag("demo")
}
