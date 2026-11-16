//! The law packs — ports of `mlopt_core`'s tensor law banks
//! (`eqsat/packages/mlopt-core/src/mlopt_core/laws/tensor.py`),
//! spelled in [`crate::pat`].
//!
//! These are the Python bank's laws whose guards fit the ported
//! [`crate::guard`]/[`crate::shape`] vocabulary: unguarded ports plus
//! the `cond`/`dspec` members whose side conditions read bound terms,
//! attrs, and the declarative shape table — stage 4 completed the
//! vocabulary (`flat-map-unsq`/`repeat-*`/`axis-align-eq`/
//! `bcast-dim-inv` pairing guards; `shape`/`vshape`/`dim`/`leaf-dim`/
//! `bcast` dspec channels). Laws needing the remaining channels stay
//! out — `tval`/`tapp` tensor values, tensor `$leaf:` mints, and the
//! procedural `check`/`derive` hooks (design doc §2). Each pack's doc
//! comment carries its own census. Names, `law` strings,
//! `cond`/`dspec` trees, and tag bundles are carried verbatim so
//! provenance lines up with the Python bank (`mlopt_core.laws.tags`:
//! `simplification` / `categorical` / `prior_art` / `symmetry` /
//! `expansive` / `fusion` / `subsumed` / `attention`).
//!
//! Pack layout mirrors the Python collections:
//!
//! * [`simplification`] — the portable members of
//!   `SIMPLIFICATION_RULES` (monoid/group laws, the silu/square/pow
//!   decompositions and folds, the view-op strips/folds
//!   (`select_mul`, `transpose_noop`, `chunk_single`, the
//!   unsqueeze-pad family, `mul_unsq_nat`, `mul_reshape_inert_l`,
//!   `softmax_fold`, `glu_fold`, `rms_norm_fold`,
//!   `rms_norm_fold_nogain`), the rsqrt canonicalization,
//!   `mul_div_swap`, `matmul_div_distribute`, `div_add_factor`,
//!   `div_sqrt_to_rsqrt`, `pow_to_rsqrt`).
//! * [`categorical`] — the portable members of `CATEGORICAL_RULES`
//!   (bilinearity and the weight-merge family, the `linear`
//!   composition/scale laws, scalar naturality and `split_scalar_*`,
//!   the product-structure fuses `qkv_fuse`/`qkv_fuse_asym`,
//!   `gqa_absorb_repeat`, the `sdpa_fold*` family).
//! * [`attention`] — `ATTENTION_RULES` + `SDPA_EXPAND_RULES`: the
//!   36 portable `laws/attention.py` members (rope scale
//!   commutation, right-multiply absorb, score-scale migration,
//!   uniform view-scale) ahead of the full `SDPA_EXPAND_RULES`
//!   (all eight key-sets; the scale-free spellings mint
//!   `1/√(q[-1])` through the `dim` dspec channel).
//!   `rope_cat_compose` stays Python-side — its half-split guard
//!   needs attr-arithmetic the `cond` vocabulary cannot form.
//! * [`batch_norm`] — `BATCH_NORM_RULES`, the opt-in eval-mode
//!   `batch_norm` expand triple (`$leaf:` Const mint + the
//!   `concat`/`take`/`len` channel-view dspec).
//! * [`conv_folds`] — `CONV_FOLD_RULES`, the 60-law conv
//!   channel-scale/shift family (5 attr-prefixes × bias arity × 6
//!   rule shapes), opt-in like the Python collection.
//! * [`prior_art`] — the portable slice of `PRIOR_ART_RULES`: the
//!   const-algebra folds (`pow_*`, `*_const_*`, `neg_const`, the
//!   `*_const_assoc_*` families), the sigmoid/tanh/hardsigmoid
//!   folds whose guards are `const-cmp` conjunctions, the §2
//!   pointwise idempotence/parity family, the const-free scalar
//!   folds (`recip_recip`, `add/sub/neg` refolds, `id_sub`/`id_div`,
//!   `add_self`/`mul2_to_add`, `mul_neg1`), and the spelled-unit
//!   `*_zeroslike_r`/`*_oneslike_r` folds whose `shaped`/`shape-eq`/
//!   `bcast` grid guard is declarative data.
//!   Deliberately NOT in [`all`], mirroring the Python bank's
//!   opt-in `prior_art` preset.
//! * [`pointwise_factor`] — `POINTWISE_FACTOR_RULES`, the opt-in
//!   Hadamard distributivity pair the Python bank deliberately keeps
//!   out of `ALL_RULES` (a closure generator).
//! * [`scan`] — `SCAN_LAWS` + `SCAN_DIAG_LAWS`, the 22-law
//!   scan-monoid family (dense + diagonal-affine carriers), opt-in
//!   like the Python `CARRIER_SEARCH` preset.
//! * [`layout`] — `LAYOUT_RULES`, the 77-law transpose/layout
//!   migration family (pointwise commutation, involution, product
//!   transpose, the NT bridge), opt-in like the Python
//!   `ALL_RULES_WITH_LAYOUT`/`WITH_LAYOUT` preset — every
//!   procedural `check=` hook re-authored as a `cond` tree.
//! * [`all`] — `simplification` + `categorical`, the portable subset
//!   of `ALL_RULES` (`all_rules()`).
//!
//! Scope note: the Python patterns live in the *typed* `Op` IR; here
//! they are kernel `Term::Op`s — `dim` is child slot 0 (auto-bound,
//! RHS inherits the root dim) and `Const` leaves are JSON-valued.
//! Python attr *metavariables* have two spellings on this surface:
//! [`OpPat::attrs_bind`] binds the whole `__attrs__` payload (exact
//! when the payload is the single attr the Python pattern names),
//! and [`OpPat::attrs_fields`] spells the per-key contract —
//! exact-keyset match, literal pins, and per-key attr metavariables —
//! which `Law::compile` interprets through generated whole-payload
//! metavariables plus `check`/`derive` hooks. [`OpPat::attrs_lit`]
//! pins literal
//! payloads exactly.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::law::Law;
use crate::pat::{AttrField, OpPat, Pat};

/// An [`AttrsSpec::Fields`](crate::pat::AttrsSpec::Fields) entry
/// binding one attrs key to an attribute metavariable.
fn af_bind(name: &str) -> AttrField {
    AttrField::Bind(name.to_string())
}

/// An [`AttrsSpec::Fields`](crate::pat::AttrsSpec::Fields) entry
/// pinning one attrs key to a literal JSON value.
fn af_lit(value: serde_json::Value) -> AttrField {
    AttrField::Lit(value)
}

/// Collect `(key, field)` pairs into the `attrs_fields` map.
fn afields(entries: Vec<(&str, AttrField)>) -> BTreeMap<String, AttrField> {
    entries
        .into_iter()
        .map(|(k, f)| (k.to_string(), f))
        .collect()
}

// ---------------------------------------------------------------------------
//  Monoid laws: commutativity & associativity  (tags: symmetry+expansive)
// ---------------------------------------------------------------------------

/// `comm_add`: `add(a, b) → add(b, a)` — symmetry of the additive
/// monoid (SMC symmetry).
///
/// Python: `Op.make("add", "a", "b") → Op.make("add", "b", "a")`,
/// law "Commutativity in a symmetric monoidal category: σ ∘ (f ⊗ g)
/// = g ⊗ f."
pub fn comm_add() -> Law {
    Law::new(
        "comm_add",
        Pat::op("add", vec![Pat::var("a"), Pat::var("b")]),
        Pat::op("add", vec![Pat::var("b"), Pat::var("a")]),
    )
    .with_law("Commutativity in a symmetric monoidal category: σ ∘ (f ⊗ g) = g ⊗ f.")
    .with_tag("symmetry")
    .with_tag("expansive")
}

/// `comm_mul`: `mul(a, b) → mul(b, a)` — commutativity of the
/// Hadamard product.
///
/// Python: `Op.make("mul", "a", "b") → Op.make("mul", "b", "a")`,
/// law "Hadamard product is commutative (SMC symmetry)."
pub fn comm_mul() -> Law {
    Law::new(
        "comm_mul",
        Pat::op("mul", vec![Pat::var("a"), Pat::var("b")]),
        Pat::op("mul", vec![Pat::var("b"), Pat::var("a")]),
    )
    .with_law("Hadamard product is commutative (SMC symmetry).")
    .with_tag("symmetry")
    .with_tag("expansive")
}

/// `assoc_add`: `add(a, add(b, c)) → add(add(a, b), c)` — associativity
/// of sequential composition.
///
/// Python: `Op.make("add", "a", Op.make("add", "b", "c")) →
/// Op.make("add", Op.make("add", "a", "b"), "c")`, law
/// "Associativity of sequential composition in a category."
pub fn assoc_add() -> Law {
    Law::new(
        "assoc_add",
        Pat::op(
            "add",
            vec![
                Pat::var("a"),
                Pat::op("add", vec![Pat::var("b"), Pat::var("c")]),
            ],
        ),
        Pat::op(
            "add",
            vec![
                Pat::op("add", vec![Pat::var("a"), Pat::var("b")]),
                Pat::var("c"),
            ],
        ),
    )
    .with_law("Associativity of sequential composition in a category.")
    .with_tag("symmetry")
    .with_tag("expansive")
}

/// `assoc_mul`: `mul(a, mul(b, c)) → mul(mul(a, b), c)` — associativity
/// of parallel composition.
///
/// Python: `Op.make("mul", "a", Op.make("mul", "b", "c")) →
/// Op.make("mul", Op.make("mul", "a", "b"), "c")`, law
/// "Associativity of parallel composition in a monoidal category."
pub fn assoc_mul() -> Law {
    Law::new(
        "assoc_mul",
        Pat::op(
            "mul",
            vec![
                Pat::var("a"),
                Pat::op("mul", vec![Pat::var("b"), Pat::var("c")]),
            ],
        ),
        Pat::op(
            "mul",
            vec![
                Pat::op("mul", vec![Pat::var("a"), Pat::var("b")]),
                Pat::var("c"),
            ],
        ),
    )
    .with_law("Associativity of parallel composition in a monoidal category.")
    .with_tag("symmetry")
    .with_tag("expansive")
}

// ---------------------------------------------------------------------------
//  Group / simplification identities  (tags: simplification)
// ---------------------------------------------------------------------------

/// `id_add`: `add(a, 0) → a` — additive identity.
///
/// Python: `Op.make("add", "a", Const(0)) → "a"`,
/// law "Additive identity: a + 0 = a."
pub fn id_add() -> Law {
    Law::new(
        "id_add",
        Pat::op("add", vec![Pat::var("a"), Pat::constant(json!(0))]),
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
        Pat::op("mul", vec![Pat::var("a"), Pat::constant(json!(1))]),
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

// ---------------------------------------------------------------------------
//  Decompositions and kernel folds  (tags: simplification)
// ---------------------------------------------------------------------------

/// `silu_expand`: `silu(x) → mul(x, sigmoid(x))` — the SiLU definition.
///
/// Python: `Op.make("silu", "x") → Op.make("mul", "x",
/// Op.make("sigmoid", "x"))`, law "SiLU definition: silu(x) = x · σ(x)."
pub fn silu_expand() -> Law {
    Law::new(
        "silu_expand",
        Pat::op("silu", vec![Pat::var("x")]),
        Pat::op(
            "mul",
            vec![Pat::var("x"), Pat::op("sigmoid", vec![Pat::var("x")])],
        ),
    )
    .with_law("SiLU definition: silu(x) = x · σ(x).")
    .with_tag("simplification")
}

/// `silu_fold`: `mul(x, sigmoid(x)) → silu(x)` — the manual-silu fold,
/// definitional inverse of [`silu_expand`].
///
/// Python: `Op.make("mul", "x", Op.make("sigmoid", "x")) →
/// Op.make("silu", "x")`, law "The manual-silu fold: x · σ(x) =
/// silu(x) — the definitional inverse of silu_expand, and the
/// mediating 3-cell of the silu_expand/silu_mul_form × swiglu_fuse
/// critical pairs."
pub fn silu_fold() -> Law {
    Law::new(
        "silu_fold",
        Pat::op(
            "mul",
            vec![Pat::var("x"), Pat::op("sigmoid", vec![Pat::var("x")])],
        ),
        Pat::op("silu", vec![Pat::var("x")]),
    )
 .with_law(
 "The manual-silu fold: x · σ(x) = silu(x) — the definitional inverse of silu_expand, and the mediating 3-cell of the silu_expand/silu_mul_form × swiglu_fuse critical pairs.",
    )
    .with_tag("simplification")
}

/// `silu_mul_form`: `mul(silu(g), u) → mul(mul(g, sigmoid(g)), u)` —
/// the SwiGLU bridge exposing the shared `g·σ(g)` factor.
///
/// Python: `Op.make("mul", Op.make("silu", "g"), "u") →
/// Op.make("mul", Op.make("mul", "g", Op.make("sigmoid", "g")), "u")`,
/// law "SwiGLU: silu(g)*u = (g*sigmoid(g))*u (factor for prefix
/// sharing)."
pub fn silu_mul_form() -> Law {
    Law::new(
        "silu_mul_form",
        Pat::op(
            "mul",
            vec![Pat::op("silu", vec![Pat::var("g")]), Pat::var("u")],
        ),
        Pat::op(
            "mul",
            vec![
                Pat::op(
                    "mul",
                    vec![Pat::var("g"), Pat::op("sigmoid", vec![Pat::var("g")])],
                ),
                Pat::var("u"),
            ],
        ),
    )
    .with_law("SwiGLU: silu(g)*u = (g*sigmoid(g))*u (factor for prefix sharing).")
    .with_tag("simplification")
}

/// `softsign_fold`: `div(x, add(abs(x), 1)) → softsign(x)` — the
/// bounded rational map spelled in decomposed form folds to the
/// kernel.
///
/// Python: `Op.make("div", "x", Op.make("add", Op.make("abs", "x"),
/// Const(1))) → Op.make("softsign", "x")`, law "x / (|x| + 1) =
/// softsign(x) — the bounded rational map, written as a three-op
/// decomposed spelling, folds into the one kernel."
pub fn softsign_fold() -> Law {
    Law::new(
        "softsign_fold",
        Pat::op(
            "div",
            vec![
                Pat::var("x"),
                Pat::op(
                    "add",
                    vec![Pat::op("abs", vec![Pat::var("x")]), Pat::constant(json!(1))],
                ),
            ],
        ),
        Pat::op("softsign", vec![Pat::var("x")]),
    )
 .with_law(
 "x / (|x| + 1) = softsign(x) — the bounded rational map, written as a three-op decomposed spelling, folds into the one kernel.",
    )
    .with_tag("simplification")
}

/// `square_expand`: `square(x) → mul(x, x)` — self-composition.
///
/// Python: `Op.make("square", "x") → Op.make("mul", "x", "x")`,
/// law "Self-composition: x² = x · x."
pub fn square_expand() -> Law {
    Law::new(
        "square_expand",
        Pat::op("square", vec![Pat::var("x")]),
        Pat::op("mul", vec![Pat::var("x"), Pat::var("x")]),
    )
    .with_law("Self-composition: x² = x · x.")
    .with_tag("simplification")
}

/// `pow_to_square`: `pow(x, 2) → square(x)` — the SwiGLU/RMSNorm
/// bridge.
///
/// Python: `Op.make("pow", "x", Const(2)) → Op.make("square", "x")`,
/// law "pow(x, 2) ≡ square(x) ≡ x·x (SwiGLU/RMSNorm bridge)."
///
/// The exponent is a JSON `Const` leaf: `2` matches `2`, not `2.0`
/// (the bridge's self-describing leaf key is the serialized kernel
/// term — a `serde_json` number-kind distinction the Python `Const`
/// equality does not make).
pub fn pow_to_square() -> Law {
    Law::new(
        "pow_to_square",
        Pat::op("pow", vec![Pat::var("x"), Pat::constant(json!(2))]),
        Pat::op("square", vec![Pat::var("x")]),
    )
    .with_law("pow(x, 2) ≡ square(x) ≡ x·x (SwiGLU/RMSNorm bridge).")
    .with_tag("simplification")
}

/// `square_to_pow`: `square(x) → pow(x, 2)` — reverse direction for
/// shape/cost reasons.
///
/// Python: `Op.make("square", "x") → Op.make("pow", "x", Const(2))`,
/// law "Reverse: square(x) ≡ pow(x, 2) for shape/cost reasons."
pub fn square_to_pow() -> Law {
    Law::new(
        "square_to_pow",
        Pat::op("square", vec![Pat::var("x")]),
        Pat::op("pow", vec![Pat::var("x"), Pat::constant(json!(2))]),
    )
    .with_law("Reverse: square(x) ≡ pow(x, 2) for shape/cost reasons.")
    .with_tag("simplification")
}

/// `mul_square`: `mul(u, u) → square(u)` — the mul spelling of x²
/// folds to the unary kernel; definitional inverse of
/// [`square_expand`]. The shared `u` metavariable is the whole
/// precondition — the matcher binds both operands to the same
/// e-class.
///
/// Python: `Op.make("mul", "u", "u") → Op.make("square", "u")`,
/// law "x·x = square(x): the mul spelling of x² folds to the unary
/// kernel — the definitional inverse of square_expand."
pub fn mul_square() -> Law {
    Law::new(
        "mul_square",
        Pat::op("mul", vec![Pat::var("u"), Pat::var("u")]),
        Pat::op("square", vec![Pat::var("u")]),
    )
 .with_law(
 "x·x = square(x): the mul spelling of x² folds to the unary kernel — the definitional inverse of square_expand.",
    )
    .with_tag("simplification")
}

/// `recip_sqrt_to_rsqrt`: `reciprocal(sqrt(u)) → rsqrt(u)` — the
/// exported ``1/sqrt`` spelling folds to the canonical kernel.
///
/// Python: `Op.make("reciprocal", Op.make("sqrt", "u")) →
/// Op.make("rsqrt", "u")`, law "reciprocal(sqrt(x)) = rsqrt(x): the
/// exported ``1/sqrt`` spelling — aten lowers scalar-over-tensor div
/// to reciprocal — folds to the same canonical kernel."
pub fn recip_sqrt_to_rsqrt() -> Law {
    Law::new(
        "recip_sqrt_to_rsqrt",
        Pat::op("reciprocal", vec![Pat::op("sqrt", vec![Pat::var("u")])]),
        Pat::op("rsqrt", vec![Pat::var("u")]),
    )
 .with_law(
 "reciprocal(sqrt(x)) = rsqrt(x): the exported ``1/sqrt`` spelling — aten lowers scalar-over-tensor div to reciprocal — folds to the same canonical kernel.",
    )
    .with_tag("simplification")
}

/// `mul_div_swap`: `mul(div(a, b), c) → div(mul(a, c), b)` —
/// pointwise div/mul naturality.
///
/// Python: `Op.make("mul", Op.make("div", "a", "b"), "c") →
/// Op.make("div", Op.make("mul", "a", "c"), "b")`, law "(a/b)·c =
/// (a·c)/b — pointwise div/mul naturality, exact for any broadcast
/// (the decay-weighted score's scale slides out of the product)."
pub fn mul_div_swap() -> Law {
    Law::new(
        "mul_div_swap",
        Pat::op(
            "mul",
            vec![
                Pat::op("div", vec![Pat::var("a"), Pat::var("b")]),
                Pat::var("c"),
            ],
        ),
        Pat::op(
            "div",
            vec![
                Pat::op("mul", vec![Pat::var("a"), Pat::var("c")]),
                Pat::var("b"),
            ],
        ),
    )
 .with_law(
 "(a/b)·c = (a·c)/b — pointwise div/mul naturality, exact for any broadcast (the decay-weighted score's scale slides out of the product).",
    )
    .with_tag("simplification")
}

// ---------------------------------------------------------------------------
//  Categorical: bilinearity, associativity, products  (tags: categorical+expansive / fusion+subsumed)
// ---------------------------------------------------------------------------

/// `right_distribute_matmul`: `matmul(add(a, b), W) →
/// add(matmul(a, W), matmul(b, W))` — bilinearity in the left slot.
/// Guard-free in Python: the contraction axis is the last axis for
/// every rank, exactly the axis broadcast aligns.
///
/// Python: `Op.make("matmul", Op.make("add", "a", "b"), "W") →
/// Op.make("add", Op.make("matmul", "a", "W"),
/// Op.make("matmul", "b", "W"))`, law "Bilinearity: linear maps
/// distribute over addition in BOTH slots."
pub fn right_distribute_matmul() -> Law {
    Law::new(
        "right_distribute_matmul",
        Pat::op(
            "matmul",
            vec![
                Pat::op("add", vec![Pat::var("a"), Pat::var("b")]),
                Pat::var("W"),
            ],
        ),
        Pat::op(
            "add",
            vec![
                Pat::op("matmul", vec![Pat::var("a"), Pat::var("W")]),
                Pat::op("matmul", vec![Pat::var("b"), Pat::var("W")]),
            ],
        ),
    )
    .with_law("Bilinearity: linear maps distribute over addition in BOTH slots.")
    .with_tag("categorical")
    .with_tag("expansive")
}

/// `right_factor_matmul`: `add(matmul(a, W), matmul(b, W)) →
/// matmul(add(a, b), W)` — factoring a shared right weight.
///
/// Python: `Op.make("add", Op.make("matmul", "a", "W"),
/// Op.make("matmul", "b", "W")) → Op.make("matmul",
/// Op.make("add", "a", "b"), "W")`, law "Factor a shared right-weight
/// (the slot `x @ W` uses)."
pub fn right_factor_matmul() -> Law {
    Law::new(
        "right_factor_matmul",
        Pat::op(
            "add",
            vec![
                Pat::op("matmul", vec![Pat::var("a"), Pat::var("W")]),
                Pat::op("matmul", vec![Pat::var("b"), Pat::var("W")]),
            ],
        ),
        Pat::op(
            "matmul",
            vec![
                Pat::op("add", vec![Pat::var("a"), Pat::var("b")]),
                Pat::var("W"),
            ],
        ),
    )
    .with_law("Factor a shared right-weight (the slot `x @ W` uses).")
    .with_tag("categorical")
    .with_tag("expansive")
}

/// `assoc_matmul`: `matmul(A, matmul(B, C)) → matmul(matmul(A, B), C)`
/// — associativity of composition.
///
/// Python: `Op.make("matmul", "A", Op.make("matmul", "B", "C")) →
/// Op.make("matmul", Op.make("matmul", "A", "B"), "C")`, law
/// "Associativity of composition in a category: (f∘g)∘h =
/// f∘(g∘h)."
pub fn assoc_matmul() -> Law {
    Law::new(
        "assoc_matmul",
        Pat::op(
            "matmul",
            vec![
                Pat::var("A"),
                Pat::op("matmul", vec![Pat::var("B"), Pat::var("C")]),
            ],
        ),
        Pat::op(
            "matmul",
            vec![
                Pat::op("matmul", vec![Pat::var("A"), Pat::var("B")]),
                Pat::var("C"),
            ],
        ),
    )
    .with_law("Associativity of composition in a category: (f∘g)∘h = f∘(g∘h).")
    .with_tag("categorical")
    .with_tag("expansive")
}

/// `assoc_matmul_rev`: `matmul(matmul(A, B), C) →
/// matmul(A, matmul(B, C))` — reverse associativity.
///
/// Python: `Op.make("matmul", Op.make("matmul", "A", "B"), "C") →
/// Op.make("matmul", "A", Op.make("matmul", "B", "C"))`, law "Reverse
/// associativity: f∘(g∘h) = (f∘g)∘h."
pub fn assoc_matmul_rev() -> Law {
    Law::new(
        "assoc_matmul_rev",
        Pat::op(
            "matmul",
            vec![
                Pat::op("matmul", vec![Pat::var("A"), Pat::var("B")]),
                Pat::var("C"),
            ],
        ),
        Pat::op(
            "matmul",
            vec![
                Pat::var("A"),
                Pat::op("matmul", vec![Pat::var("B"), Pat::var("C")]),
            ],
        ),
    )
    .with_law("Reverse associativity: f∘(g∘h) = (f∘g)∘h.")
    .with_tag("categorical")
    .with_tag("expansive")
}

/// `assoc_linear`: `linear(linear(x, A), B) → linear(x, matmul(B, A))`
/// — stacked `nn.Linear`s compose; the fused weight is `B @ A`
/// (transposes flip the product order).
///
/// Python: `Op.make("linear", Op.make("linear", "x", "A"), "B") →
/// Op.make("linear", "x", Op.make("matmul", "B", "A"))`, law
/// "Compose stacked nn.Linears: fused weight is B @ A (transposes
/// flip the product order)."
pub fn assoc_linear() -> Law {
    Law::new(
        "assoc_linear",
        Pat::op(
            "linear",
            vec![
                Pat::op("linear", vec![Pat::var("x"), Pat::var("A")]),
                Pat::var("B"),
            ],
        ),
        Pat::op(
            "linear",
            vec![
                Pat::var("x"),
                Pat::op("matmul", vec![Pat::var("B"), Pat::var("A")]),
            ],
        ),
    )
    .with_law(
        "Compose stacked nn.Linears: fused weight is B @ A (transposes flip the product order).",
    )
    .with_tag("categorical")
    .with_tag("expansive")
}

/// `assoc_linear_rev`: `linear(x, matmul(B, A)) →
/// linear(linear(x, A), B)` — reverse linear composition.
///
/// Python: `Op.make("linear", "x", Op.make("matmul", "B", "A")) →
/// Op.make("linear", Op.make("linear", "x", "A"), "B")`, law
/// "Reverse linear composition: linear(x, B@A) = linear(linear(x, A),
/// B) (eqsat weighs fused vs split)."
pub fn assoc_linear_rev() -> Law {
    Law::new(
        "assoc_linear_rev",
        Pat::op(
            "linear",
            vec![
                Pat::var("x"),
                Pat::op("matmul", vec![Pat::var("B"), Pat::var("A")]),
            ],
        ),
        Pat::op(
            "linear",
            vec![
                Pat::op("linear", vec![Pat::var("x"), Pat::var("A")]),
                Pat::var("B"),
            ],
        ),
    )
 .with_law(
 "Reverse linear composition: linear(x, B@A) = linear(linear(x, A), B) (eqsat weighs fused vs split).",
    )
    .with_tag("categorical")
    .with_tag("expansive")
}

/// `right_factor_linear`: `add(linear(a, W), linear(b, W)) →
/// linear(add(a, b), W)` — factor a shared right-hand `nn.Linear`
/// weight.
///
/// Python: `Op.make("add", Op.make("linear", "a", "W"),
/// Op.make("linear", "b", "W")) → Op.make("linear",
/// Op.make("add", "a", "b"), "W")`, law "Factor a shared right-hand
/// nn.Linear weight."
pub fn right_factor_linear() -> Law {
    Law::new(
        "right_factor_linear",
        Pat::op(
            "add",
            vec![
                Pat::op("linear", vec![Pat::var("a"), Pat::var("W")]),
                Pat::op("linear", vec![Pat::var("b"), Pat::var("W")]),
            ],
        ),
        Pat::op(
            "linear",
            vec![
                Pat::op("add", vec![Pat::var("a"), Pat::var("b")]),
                Pat::var("W"),
            ],
        ),
    )
    .with_law("Factor a shared right-hand nn.Linear weight.")
    .with_tag("categorical")
    .with_tag("expansive")
}

/// The head-splitting view shared by [`qkv_fuse`]'s operands:
/// `transpose(reshape(t, shape=?S), dim0=1, dim1=2)`.
///
/// The `shape` attr is spelled per-key (`attrs_fields` with a `shape`
/// bind) — the Python `shape="S"` attr metavariable's exact-keyset
/// equivalent on this surface (see the module docs for the
/// whole-payload caveat).
fn head_view(t: Pat) -> Pat {
    Pat::Op(
        OpPat::new("transpose")
            .arg(Pat::Op(
                OpPat::new("reshape")
                    .arg(t)
                    .attrs_fields(afields(vec![("shape", af_bind("S"))])),
            ))
            .attrs_lit(json!({"dim0": 1, "dim1": 2})),
    )
}

/// `qkv_fuse`: three `linear` projections of the same input feeding
/// `sdpa` fuse into ONE GEMM over the concatenated weights plus
/// three zero-cost `chunk` projections.
///
/// Python: `Op.make("sdpa", head(linear(x,Q)), head(linear(x,K)),
/// head(linear(x,V)), scale="SC") → sdpa(head(chunk_i(linear(x,
/// cat(cat(Q,K),V)))), i=0..2, scale="SC")`, law "Triple pairing
/// <q,k,v> : X -> V^3 — three projections of the same input are ONE
/// GEMM into the product space, then three zero-cost chunk
/// projections.  (Fused QKV.)"
///
/// Attr spellings: `scale`/`shape` are per-key binds
/// (`attrs_fields`) — the Python `scale="SC"`/`shape="S"` attr
/// metavariables' exact-keyset equivalent on this surface;
/// `transpose`/`concat`/`chunk` attrs are literal payloads
/// (`attrs_lit`).
pub fn qkv_fuse() -> Law {
    // The fused weight: `concat(concat(Q, K, dim=0), V, dim=0)`.
    let fused_weight = Pat::Op(
        OpPat::new("concat")
            .arg(Pat::Op(
                OpPat::new("concat")
                    .arg(Pat::var("Q"))
                    .arg(Pat::var("K"))
                    .attrs_lit(json!({"dim": 0})),
            ))
            .arg(Pat::var("V"))
            .attrs_lit(json!({"dim": 0})),
    );
    // One chunk projection of the fused GEMM: `chunk(linear(x, W),
    // chunks=3, dim=-1, index=i)`.
    let chunk_proj = |index: i64| {
        Pat::Op(
            OpPat::new("chunk")
                .arg(Pat::op("linear", vec![Pat::var("x"), fused_weight.clone()]))
                .attrs_lit(json!({"chunks": 3, "dim": -1, "index": index})),
        )
    };
    let rhs_heads: Vec<Pat> = (0..3).map(|i| head_view(chunk_proj(i))).collect();
    Law::new(
        "qkv_fuse",
        Pat::Op(
            OpPat::new("sdpa")
                .arg(head_view(Pat::op(
                    "linear",
                    vec![Pat::var("x"), Pat::var("Q")],
                )))
                .arg(head_view(Pat::op(
                    "linear",
                    vec![Pat::var("x"), Pat::var("K")],
                )))
                .arg(head_view(Pat::op(
                    "linear",
                    vec![Pat::var("x"), Pat::var("V")],
                )))
                .attrs_fields(afields(vec![("scale", af_bind("SC"))])),
        ),
        Pat::Op(
            OpPat::new("sdpa")
                .args(rhs_heads)
                .attrs_fields(afields(vec![("scale", af_bind("SC"))])),
        ),
    )
 .with_law(
 "Triple pairing <q,k,v> : X -> V^3 — three projections of the same input are ONE GEMM into the product space, then three zero-cost chunk projections.  (Fused QKV.)",
    )
    .with_tag("fusion")
    .with_tag("subsumed")
}

// ---------------------------------------------------------------------------
//  Opt-in: pointwise (Hadamard) distributivity  (tags: categorical+expansive)
// ---------------------------------------------------------------------------

/// `distribute_mul_add`: `mul(c, add(x, y)) → add(mul(c, x),
/// mul(c, y))` — the elementwise product distributes over the
/// additive monoid.
///
/// Python: `Op.make("mul", "c", Op.make("add", "x", "y")) →
/// Op.make("add", Op.make("mul", "c", "x"), Op.make("mul", "c",
/// "y"))`, law "Hadamard distributivity: c ⊙ (x+y) = c⊙x + c⊙y — the
/// elementwise product distributes over the additive monoid."
///
/// Unguarded in Python and OPT-IN (`POINTWISE_FACTOR_RULES`): the
/// pair is a closure generator of the comm/assoc class, so it stays
/// out of `CATEGORICAL_RULES`/`ALL_RULES` — [`pointwise_factor`]
/// exposes it explicitly, the same convention as the Python bank.
pub fn distribute_mul_add() -> Law {
    Law::new(
        "distribute_mul_add",
        Pat::op(
            "mul",
            vec![
                Pat::var("c"),
                Pat::op("add", vec![Pat::var("x"), Pat::var("y")]),
            ],
        ),
        Pat::op(
            "add",
            vec![
                Pat::op("mul", vec![Pat::var("c"), Pat::var("x")]),
                Pat::op("mul", vec![Pat::var("c"), Pat::var("y")]),
            ],
        ),
    )
 .with_law(
 "Hadamard distributivity: c ⊙ (x+y) = c⊙x + c⊙y — the elementwise product distributes over the additive monoid.",
    )
    .with_tag("categorical")
    .with_tag("expansive")
}

/// `factor_mul_add`: `add(mul(c, x), mul(c, y)) → mul(c, add(x, y))`
/// — factoring a shared elementwise factor (reverse distributivity).
///
/// Python: `Op.make("add", Op.make("mul", "c", "x"), Op.make("mul",
/// "c", "y")) → Op.make("mul", "c", Op.make("add", "x", "y"))`, law
/// "Factor a shared elementwise factor (reverse distributivity) —
/// c⊙x + c⊙y = c⊙(x+y); what the carrier fold chains consume."
///
/// Same opt-in status as [`distribute_mul_add`] — see its docs.
pub fn factor_mul_add() -> Law {
    Law::new(
        "factor_mul_add",
        Pat::op(
            "add",
            vec![
                Pat::op("mul", vec![Pat::var("c"), Pat::var("x")]),
                Pat::op("mul", vec![Pat::var("c"), Pat::var("y")]),
            ],
        ),
        Pat::op(
            "mul",
            vec![
                Pat::var("c"),
                Pat::op("add", vec![Pat::var("x"), Pat::var("y")]),
            ],
        ),
    )
 .with_law(
 "Factor a shared elementwise factor (reverse distributivity) — c⊙x + c⊙y = c⊙(x+y); what the carrier fold chains consume.",
    )
    .with_tag("categorical")
    .with_tag("expansive")
}

// ---------------------------------------------------------------------------
//  Guarded laws — the stage-1 `cond`/`dspec` vocabulary (`guard.rs`).
//
//  `cond` trees are the `cond.py` predicate spelling carried as JSON;
//  `dspec` the `{NAME: expr}` derive spec. Both refuse at compile
//  when they name an unported op (flat-map/repeat pairing guards,
//  tensor channels) — see `crate::guard`.
// ---------------------------------------------------------------------------

/// `div_sqrt_to_rsqrt`: `div(ONE, sqrt(u)) → rsqrt(u)` guarded by
/// `ONE` being the Const 1 — the two-op reciprocal-root spelling
/// folds to the dispatched kernel.
///
/// Python: `tensor.py` `DIV_SQRT_TO_RSQRT`, tags `simplification`.
pub fn div_sqrt_to_rsqrt() -> Law {
    Law::new(
        "div_sqrt_to_rsqrt",
        Pat::op(
            "div",
            vec![
                Pat::var("ONE"),
                Pat::op("sqrt", vec![Pat::var("u")]),
            ],
        ),
        Pat::op("rsqrt", vec![Pat::var("u")]),
    )
 .with_law(
 "1 / sqrt(x) = rsqrt(x): the two-op reciprocal-root spelling folds to the dispatched kernel — the canonical form the RMSNorm fold's pattern pins.",
    )
    .with_cond(json!(["const-cmp", "ONE", "==", 1]))
    .with_tag("simplification")
}

/// `pow_to_rsqrt`: `pow(u, P) → rsqrt(u)` guarded by `P` being the
/// Const −0.5 — the negative-half-power spelling folds to the same
/// canonical kernel.
///
/// Python: `tensor.py` `POW_TO_RSQRT`, tags `simplification`.
pub fn pow_to_rsqrt() -> Law {
    Law::new(
        "pow_to_rsqrt",
        Pat::op("pow", vec![Pat::var("u"), Pat::var("P")]),
        Pat::op("rsqrt", vec![Pat::var("u")]),
    )
 .with_law(
 "pow(x, -0.5) = rsqrt(x): the negative-half-power spelling folds to the dispatched kernel — the same canonicalization as div_sqrt_to_rsqrt one op-family over.",
    )
    .with_cond(json!(["const-cmp", "P", "==", -0.5]))
    .with_tag("simplification")
}

// ---------------------------------------------------------------------------
//  prior_art §1 — TASO substitutions: concat structure (20 laws)
// ---------------------------------------------------------------------------

/// The shared `dim` attr-metavariable spec of the concat family —
/// `Op.make(…, dim="D")` binds one axis across every operand and the
/// result.
fn concat_dim() -> BTreeMap<String, AttrField> {
    afields(vec![("dim", af_bind("D"))])
}

/// The `unsqueeze(·, dim="D")` operand view of the stack bridge.
fn unsq_dim(t: Pat) -> Pat {
    Pat::Op(OpPat::new("unsqueeze").arg(t).attrs_fields(concat_dim()))
}

/// `concat_single`: `concat(u, dim=D) → u`.
///
/// Python: `priorart.py` `CONCAT_SINGLE`, tags
/// `prior_art`+`simplification`.
pub fn concat_single() -> Law {
    Law::new(
        "concat_single",
        Pat::Op(
            OpPat::new("concat")
                .arg(Pat::var("u"))
                .attrs_fields(concat_dim()),
        ),
        Pat::var("u"),
    )
    .with_law("concat of one tensor is the tensor (TASO cat-noop).")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `concat_flat_l`: `concat(concat(a,b,D),c,D) → concat(a,b,c,D)`.
///
/// Python: `priorart.py` `CONCAT_FLAT_L`, tags
/// `prior_art`+`expansive`.
pub fn concat_flat_l() -> Law {
    Law::new(
        "concat_flat_l",
        Pat::Op(
            OpPat::new("concat")
                .arg(Pat::Op(
                    OpPat::new("concat")
                        .arg(Pat::var("a"))
                        .arg(Pat::var("b"))
                        .attrs_fields(concat_dim()),
                ))
                .arg(Pat::var("c"))
                .attrs_fields(concat_dim()),
        ),
        Pat::Op(
            OpPat::new("concat")
                .arg(Pat::var("a"))
                .arg(Pat::var("b"))
                .arg(Pat::var("c"))
                .attrs_fields(concat_dim()),
        ),
    )
    .with_law(
        "concat is associative: cat(cat(a,b),c) = cat(a,b,c) — the n-ary flatten (TASO concat associativity; the KV-cache append chain).",
    )
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `concat_flat_r`: `concat(a,concat(b,c,D),D) → concat(a,b,c,D)`.
///
/// Python: `priorart.py` `CONCAT_FLAT_R`, tags
/// `prior_art`+`expansive`.
pub fn concat_flat_r() -> Law {
    Law::new(
        "concat_flat_r",
        Pat::Op(
            OpPat::new("concat")
                .arg(Pat::var("a"))
                .arg(Pat::Op(
                    OpPat::new("concat")
                        .arg(Pat::var("b"))
                        .arg(Pat::var("c"))
                        .attrs_fields(concat_dim()),
                ))
                .attrs_fields(concat_dim()),
        ),
        Pat::Op(
            OpPat::new("concat")
                .arg(Pat::var("a"))
                .arg(Pat::var("b"))
                .arg(Pat::var("c"))
                .attrs_fields(concat_dim()),
        ),
    )
    .with_law("concat(a, cat(b,c)) = cat(a,b,c) — right-nested flatten.")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `chunk(u, chunks, dim=D, index)` with the literal chunk count and
/// index pinned and the axis bound to the family's `D`.
fn chunk_attrs(chunks: i64, index: i64) -> BTreeMap<String, AttrField> {
    afields(vec![
        ("chunks", af_lit(json!(chunks))),
        ("dim", af_bind("D")),
        ("index", af_lit(json!(index))),
    ])
}

/// `concat_chunk2`: `cat(chunk0₂(u), chunk1₂(u)) → u`.
///
/// Python: `priorart.py` `CONCAT_CHUNK2`, tags
/// `prior_art`+`simplification`.
pub fn concat_chunk2() -> Law {
    let chunk = |index: i64| {
        Pat::Op(
            OpPat::new("chunk")
                .arg(Pat::var("u"))
                .attrs_fields(chunk_attrs(2, index)),
        )
    };
    Law::new(
        "concat_chunk2",
        Pat::Op(
            OpPat::new("concat")
                .arg(chunk(0))
                .arg(chunk(1))
                .attrs_fields(concat_dim()),
        ),
        Pat::var("u"),
    )
    .with_law(
        "cat(chunk0(u), chunk1(u)) = u — the split/merge round-trip (TASO); holds for odd extents too since chunk splits first-big.",
    )
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `concat_chunk3`: `cat(chunk0₃(u), chunk1₃(u), chunk2₃(u)) → u`.
///
/// Python: `priorart.py` `CONCAT_CHUNK3`, tags
/// `prior_art`+`simplification`.
pub fn concat_chunk3() -> Law {
    let chunk = |index: i64| {
        Pat::Op(
            OpPat::new("chunk")
                .arg(Pat::var("u"))
                .attrs_fields(chunk_attrs(3, index)),
        )
    };
    Law::new(
        "concat_chunk3",
        Pat::Op(
            OpPat::new("concat")
                .arg(chunk(0))
                .arg(chunk(1))
                .arg(chunk(2))
                .attrs_fields(concat_dim()),
        ),
        Pat::var("u"),
    )
    .with_law("The 3-way split/merge round-trip.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `concat_split2`: `cat(split0(u,SZ), split1(u,SZ)) → u` when `SZ`
/// holds exactly two sizes — the uneven split's round-trip.
///
/// Python: `priorart.py` `CONCAT_SPLIT2`, cond
/// `("attr-len", "SZ", "==", 2)` verbatim, tags
/// `prior_art`+`simplification`.
pub fn concat_split2() -> Law {
    let split = |index: i64| {
        Pat::Op(
            OpPat::new("split")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![
                    ("sizes", af_bind("SZ")),
                    ("dim", af_bind("D")),
                    ("index", af_lit(json!(index))),
                ])),
        )
    };
    Law::new(
        "concat_split2",
        Pat::Op(
            OpPat::new("concat")
                .arg(split(0))
                .arg(split(1))
                .attrs_fields(concat_dim()),
        ),
        Pat::var("u"),
    )
    .with_law(
        "cat(split0(u), split1(u)) = u — the uneven split's round-trip (two sizes cover the axis).",
    )
    .with_cond(json!(["attr-len", "SZ", "==", 2]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `stack_fold`: `cat(unsqueeze(a,D), unsqueeze(b,D), D) →
/// stack(a,b,D)`.
///
/// Python: `priorart.py` `STACK_FOLD`, tags
/// `prior_art`+`simplification`.
pub fn stack_fold() -> Law {
    Law::new(
        "stack_fold",
        Pat::Op(
            OpPat::new("concat")
                .arg(unsq_dim(Pat::var("a")))
                .arg(unsq_dim(Pat::var("b")))
                .attrs_fields(concat_dim()),
        ),
        Pat::Op(
            OpPat::new("stack")
                .arg(Pat::var("a"))
                .arg(Pat::var("b"))
                .attrs_fields(concat_dim()),
        ),
    )
    .with_law(
        "cat(unsqueeze(a,d), unsqueeze(b,d), d) = stack(a,b,d) — torch.stack IS the cat of the unsqueezed pair.",
    )
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `stack_expand`: `stack(a,b,D) → cat(unsqueeze(a,D),
/// unsqueeze(b,D), D)` — the decomposition direction, derivation
/// `stack_fold`.
///
/// Python: `priorart.py` `STACK_EXPAND`, tags
/// `prior_art`+`expansive`.
pub fn stack_expand() -> Law {
    Law::new(
        "stack_expand",
        Pat::Op(
            OpPat::new("stack")
                .arg(Pat::var("a"))
                .arg(Pat::var("b"))
                .attrs_fields(concat_dim()),
        ),
        Pat::Op(
            OpPat::new("concat")
                .arg(unsq_dim(Pat::var("a")))
                .arg(unsq_dim(Pat::var("b")))
                .attrs_fields(concat_dim()),
        ),
    )
    .with_law("stack(a,b,d) decomposes to the unsqueeze+concat pair.")
    .with_derivation("stack_fold")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `_concat_pull(op)` — `cat(op(a), op(b), D) → op(cat(a,b,D))` for
/// the unary pointwise ops.
fn concat_pull_unary_rules() -> Vec<Law> {
    [
        "neg", "abs", "relu", "sigmoid", "silu", "tanh", "gelu", "exp", "square",
    ]
    .into_iter()
    .map(|op| {
        Law::new(
            format!("concat_pull_{op}"),
            Pat::Op(
                OpPat::new("concat")
                    .arg(Pat::op(op, vec![Pat::var("a")]))
                    .arg(Pat::op(op, vec![Pat::var("b")]))
                    .attrs_fields(concat_dim()),
            ),
            Pat::op(
                op,
                vec![Pat::Op(
                    OpPat::new("concat")
                        .arg(Pat::var("a"))
                        .arg(Pat::var("b"))
                        .attrs_fields(concat_dim()),
                )],
            ),
        )
        .with_law(format!(
            "cat({op}(a), {op}(b)) = {op}(cat(a,b)) — the pointwise op commutes with the partition (TASO concat-pointwise)."
        ))
        .with_tag("prior_art")
        .with_tag("simplification")
    })
    .collect()
}

/// `_concat_pull2(op)` — `cat(op(a1,b1), op(a2,b2), D) →
/// op(cat(a1,a2,D), cat(b1,b2,D))` for the binary pointwise ops.
fn concat_pull_binary_rules() -> Vec<Law> {
    ["add", "mul", "sub"]
        .into_iter()
        .map(|op| {
            let cat = |x: &str, y: &str| {
                Pat::Op(
                    OpPat::new("concat")
                        .arg(Pat::var(x))
                        .arg(Pat::var(y))
                        .attrs_fields(concat_dim()),
                )
            };
            Law::new(
                format!("concat_pull_{op}"),
                Pat::Op(
                    OpPat::new("concat")
                        .arg(Pat::op(op, vec![Pat::var("a1"), Pat::var("b1")]))
                        .arg(Pat::op(op, vec![Pat::var("a2"), Pat::var("b2")]))
                        .attrs_fields(concat_dim()),
                ),
                Pat::op(op, vec![cat("a1", "a2"), cat("b1", "b2")]),
            )
            .with_law(format!(
                "cat({op}(a1,b1), {op}(a2,b2)) = {op}(cat(a1,a2), cat(b1,b2)) — the pointwise pair commutes with the partition."
            ))
            .with_tag("prior_art")
            .with_tag("simplification")
        })
        .collect()
}

// ---------------------------------------------------------------------------
//  prior_art §1 — TASO: matmul/linear concat merges and split-K
// ---------------------------------------------------------------------------

/// `dim` attr-metavariable `name` — the shared-axis spelling the
/// mm/linear merge family binds (`CD` on the output concat, `AD` on
/// the fused weight concat).
fn concat_dim_named(name: &str) -> BTreeMap<String, AttrField> {
    afields(vec![("dim", af_bind(name))])
}

/// `cat(a@W, b@W, CD) → cat(a,b,CD)@W` — the row/batch-block merge on
/// a non-contraction axis.
///
/// Python: `priorart.py` `MM_CONCAT_LEFT`, cond `_COND_MM_CL`
/// (`dim-eq a -1 b -1` + `not(axis a CD -1)`), tags
/// `prior_art`+`fusion`.
pub fn mm_concat_left() -> Law {
    Law::new(
        "mm_concat_left",
        Pat::Op(
            OpPat::new("concat")
                .arg(Pat::op("matmul", vec![Pat::var("a"), Pat::var("W")]))
                .arg(Pat::op("matmul", vec![Pat::var("b"), Pat::var("W")]))
                .attrs_fields(concat_dim_named("CD")),
        ),
        Pat::Op(
            OpPat::new("matmul")
                .arg(Pat::Op(
                    OpPat::new("concat")
                        .arg(Pat::var("a"))
                        .arg(Pat::var("b"))
                        .attrs_fields(concat_dim_named("CD")),
                ))
                .arg(Pat::var("W")),
        ),
    )
    .with_law(
        "cat(a@W, b@W) = cat(a,b)@W on a non-contraction axis — row/batch blocks of a shared contraction merge into one matmul (TASO matmul-concat).",
    )
    .with_cond(json!([
        "and",
        ["dim-eq", "a", -1, "b", -1],
        ["not", ["axis", "a", "CD", -1]]
    ]))
    .with_tag("prior_art")
    .with_tag("fusion")
}

/// `cat(a,b,CD)@W → cat(a@W, b@W, CD)` — the split direction, eqsat
/// weighs split vs merged; derivation `mm_concat_left`.
///
/// Python: `priorart.py` `MM_CONCAT_LEFT_REV`, cond `_COND_MM_CL`,
/// tags `prior_art`+`expansive`.
pub fn mm_concat_left_rev() -> Law {
    Law::new(
        "mm_concat_left_rev",
        Pat::Op(
            OpPat::new("matmul")
                .arg(Pat::Op(
                    OpPat::new("concat")
                        .arg(Pat::var("a"))
                        .arg(Pat::var("b"))
                        .attrs_fields(concat_dim_named("CD")),
                ))
                .arg(Pat::var("W")),
        ),
        Pat::Op(
            OpPat::new("concat")
                .arg(Pat::op("matmul", vec![Pat::var("a"), Pat::var("W")]))
                .arg(Pat::op("matmul", vec![Pat::var("b"), Pat::var("W")]))
                .attrs_fields(concat_dim_named("CD")),
        ),
    )
    .with_law(
        "Reverse: split a matmul's non-contraction axis into two products (eqsat weighs split vs merged).",
    )
    .with_cond(json!([
        "and",
        ["dim-eq", "a", -1, "b", -1],
        ["not", ["axis", "a", "CD", -1]]
    ]))
    .with_derivation("mm_concat_left")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// Python `_COND_MM_CR` — the column-block merge guard: `CD` pinned
/// to the output axis (`-1`), both right operands rank-2 with the
/// contraction dim (`-2`) agreeing and matching `x`'s last axis.
fn cond_mm_cr() -> Value {
    json!([
        "and",
        ["rank", "x", "<=", 2],
        ["rank", "b", "==", 2],
        ["rank", "c", "==", 2],
        ["attr-eq", "CD", -1],
        ["axis", "b", "CD", -1],
        ["dim-eq", "b", -2, "c", -2],
        ["dim-eq", "x", -1, "b", -2]
    ])
}

/// `cat(x@b, x@c, CD) → x@cat(b,c,CD)` — column blocks of a shared
/// input merge into one matmul.
///
/// Python: `priorart.py` `MM_CONCAT_RIGHT`, cond `_COND_MM_CR`, tags
/// `prior_art`+`fusion`.
pub fn mm_concat_right() -> Law {
    Law::new(
        "mm_concat_right",
        Pat::Op(
            OpPat::new("concat")
                .arg(Pat::op("matmul", vec![Pat::var("x"), Pat::var("b")]))
                .arg(Pat::op("matmul", vec![Pat::var("x"), Pat::var("c")]))
                .attrs_fields(concat_dim_named("CD")),
        ),
        Pat::Op(
            OpPat::new("matmul")
                .arg(Pat::var("x"))
                .arg(Pat::Op(
                    OpPat::new("concat")
                        .arg(Pat::var("b"))
                        .arg(Pat::var("c"))
                        .attrs_fields(concat_dim_named("CD")),
                )),
        ),
    )
    .with_law(
        "cat(x@b, x@c) = x@cat(b,c) along the output axis — column blocks of a shared input merge into one matmul.",
    )
    .with_cond(cond_mm_cr())
    .with_tag("prior_art")
    .with_tag("fusion")
}

/// `x@cat(b,c,CD) → cat(x@b, x@c, CD)` — the column-block split,
/// derivation `mm_concat_right`.
///
/// Python: `priorart.py` `MM_CONCAT_RIGHT_REV`, cond `_COND_MM_CR`,
/// tags `prior_art`+`expansive`.
pub fn mm_concat_right_rev() -> Law {
    Law::new(
        "mm_concat_right_rev",
        Pat::Op(
            OpPat::new("matmul").arg(Pat::var("x")).arg(Pat::Op(
                OpPat::new("concat")
                    .arg(Pat::var("b"))
                    .arg(Pat::var("c"))
                    .attrs_fields(concat_dim_named("CD")),
            )),
        ),
        Pat::Op(
            OpPat::new("concat")
                .arg(Pat::op("matmul", vec![Pat::var("x"), Pat::var("b")]))
                .arg(Pat::op("matmul", vec![Pat::var("x"), Pat::var("c")]))
                .attrs_fields(concat_dim_named("CD")),
        ),
    )
    .with_law("Reverse column-block split (eqsat weighs both forms).")
    .with_cond(cond_mm_cr())
    .with_derivation("mm_concat_right")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `split(W, sizes=SZ, dim=-2, index)` — the weight-split view of the
/// split-K pair; `SZ` is dspec-derived (forward) or bound by the LHS
/// pattern (reverse).
fn splitk_w(index: i64) -> Pat {
    Pat::Op(
        OpPat::new("split")
            .arg(Pat::var("W"))
            .attrs_fields(afields(vec![
                ("sizes", af_bind("SZ")),
                ("dim", af_lit(json!(-2))),
                ("index", af_lit(json!(index))),
            ])),
    )
}

/// Python `_COND_MM_SPLITK` — the rank-2 GEMM scope plus the
/// contraction-axis agreement (`a`'s `CD` axis IS its last) and the
/// weight `-2` extent equation, all read through the shape table.
fn cond_mm_splitk() -> Value {
    json!([
        "and",
        ["rank", "a", "==", 2],
        ["rank", "b", "==", 2],
        ["dim-eq", "a", -2, "b", -2],
        ["rank", "W", "==", 2],
        ["axis", "a", "CD", -1],
        [
            "attr-cmp",
            ["dim", "W", -2],
            "==",
            ["add", ["dim", "a", -1], ["dim", "b", -1]]
        ]
    ])
}

/// The split-K weight-split sizes: `SZ = (dim a -1, dim b -1)` — the
/// two operands' contraction extents.
fn dspec_splitk() -> Value {
    json!({"SZ": ["tuple", ["dim", "a", -1], ["dim", "b", -1]]})
}

/// `matmul(concat(a,b,CD), W) → a@W_top + b@W_bot` — the split-K
/// decomposition: the contraction axis partitions and the weight
/// splits on its `-2` axis.
///
/// Python: `priorart.py` `MM_SPLITK`, cond `_COND_MM_SPLITK`, dspec
/// `_DSPEC_SPLITK`, tags `prior_art`+`categorical`.
pub fn mm_splitk() -> Law {
    Law::new(
        "mm_splitk",
        Pat::Op(
            OpPat::new("matmul")
                .arg(Pat::Op(
                    OpPat::new("concat")
                        .arg(Pat::var("a"))
                        .arg(Pat::var("b"))
                        .attrs_fields(concat_dim_named("CD")),
                ))
                .arg(Pat::var("W")),
        ),
        Pat::op(
            "add",
            vec![
                Pat::op(
                    "matmul",
                    vec![Pat::var("a"), splitk_w(0)],
                ),
                Pat::op(
                    "matmul",
                    vec![Pat::var("b"), splitk_w(1)],
                ),
            ],
        ),
    )
    .with_law(
        "(cat(a,b,-1)) @ W = a@W_top + b@W_bot — the split-K decomposition: the contraction axis partitions, the weight splits on its -2 axis (TASO).",
    )
    .with_cond(cond_mm_splitk())
    .with_dspec(dspec_splitk())
    .with_tag("prior_art")
    .with_tag("categorical")
}

/// The split-K refold — `a@W_top + b@W_bot → matmul(concat(a,b,-1),
/// W)`; the bound `SZ` must equal the operands' contraction extents
/// (the unfold direction re-derives nothing — the guard pins it).
///
/// Python: `priorart.py` `MM_SPLITK_REV`, tags `prior_art`+`expansive`,
/// derivation `mm_splitk`.
pub fn mm_splitk_rev() -> Law {
    Law::new(
        "mm_splitk_rev",
        Pat::op(
            "add",
            vec![
                Pat::op("matmul", vec![Pat::var("a"), splitk_w(0)]),
                Pat::op("matmul", vec![Pat::var("b"), splitk_w(1)]),
            ],
        ),
        Pat::Op(
            OpPat::new("matmul")
                .arg(Pat::Op(
                    OpPat::new("concat")
                        .arg(Pat::var("a"))
                        .arg(Pat::var("b"))
                        .attrs_lit(json!({"dim": -1})),
                ))
                .arg(Pat::var("W")),
        ),
    )
    .with_law("The split-K refold — the two partial products merge into one contraction.")
    .with_cond(json!([
        "and",
        ["rank", "a", "==", 2],
        ["rank", "b", "==", 2],
        ["dim-eq", "a", -2, "b", -2],
        ["rank", "W", "==", 2],
        [
            "attr-cmp",
            "SZ",
            "==",
            ["tuple", ["dim", "a", -1], ["dim", "b", -1]]
        ],
        [
            "attr-cmp",
            ["dim", "W", -2],
            "==",
            ["add", ["dim", "a", -1], ["dim", "b", -1]]
        ]
    ]))
    .with_derivation("mm_splitk")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// Python `_COND_LIN_CAT` — the fused-projection guard: both weight
/// operands rank-2 with agreeing `dim 1` (the output axis `linear`
/// produces), and the output concat `CD` rides `x`'s last axis.
fn cond_lin_cat() -> Value {
    json!([
        "and",
        ["rank", "A", "==", 2],
        ["rank", "B", "==", 2],
        ["dim-eq", "A", 1, "B", 1],
        ["axis", "x", "CD", -1]
    ])
}

/// `cat(xAᵀ, xBᵀ, CD) → x[A;B]ᵀ` — the MergedColumnParallelLinear
/// weight-row merge (TASO; the qkv_fuse class).
///
/// Python: `priorart.py` `LINEAR_CONCAT_W`, cond `_COND_LIN_CAT`,
/// tags `prior_art`+`fusion`.
pub fn linear_concat_w() -> Law {
    Law::new(
        "linear_concat_w",
        Pat::Op(
            OpPat::new("concat")
                .arg(Pat::op("linear", vec![Pat::var("x"), Pat::var("A")]))
                .arg(Pat::op("linear", vec![Pat::var("x"), Pat::var("B")]))
                .attrs_fields(concat_dim_named("CD")),
        ),
        Pat::op(
            "linear",
            vec![
                Pat::var("x"),
                Pat::Op(
                    OpPat::new("concat")
                        .arg(Pat::var("A"))
                        .arg(Pat::var("B"))
                        .attrs_lit(json!({"dim": 0})),
                ),
            ],
        ),
    )
    .with_law(
        "cat(xAᵀ, xBᵀ) = x[A;B]ᵀ on the output axis — the MergedColumnParallelLinear merge (TASO).",
    )
    .with_cond(cond_lin_cat())
    .with_tag("prior_art")
    .with_tag("fusion")
}

/// `linear(x, cat(A,B,AD)) → cat(xAᵀ, xBᵀ, -1)` — the fused
/// projection split; the `AD` axis must be `A`'s dim 0.
///
/// Python: `priorart.py` `LINEAR_CONCAT_W_REV`, tags
/// `prior_art`+`expansive`, derivation `linear_concat_w`.
pub fn linear_concat_w_rev() -> Law {
    Law::new(
        "linear_concat_w_rev",
        Pat::op(
            "linear",
            vec![
                Pat::var("x"),
                Pat::Op(
                    OpPat::new("concat")
                        .arg(Pat::var("A"))
                        .arg(Pat::var("B"))
                        .attrs_fields(concat_dim_named("AD")),
                ),
            ],
        ),
        Pat::Op(
            OpPat::new("concat")
                .arg(Pat::op("linear", vec![Pat::var("x"), Pat::var("A")]))
                .arg(Pat::op("linear", vec![Pat::var("x"), Pat::var("B")]))
                .attrs_lit(json!({"dim": -1})),
        ),
    )
    .with_law("Reverse: split a fused projection into per-block linears.")
    .with_cond(json!([
        "and",
        ["rank", "A", "==", 2],
        ["rank", "B", "==", 2],
        ["dim-eq", "A", 1, "B", 1],
        ["axis", "A", "AD", 0]
    ]))
    .with_derivation("linear_concat_w")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// Python `_COND_LINEAR_BIAS` — the bias-epilogue guard: the
/// `mm-shape-ok`/`rank W == 2` well-typedness pair, `b` broadcasting
/// into `linear(x, W)`'s output grid (`_SPEC_LINEAR_OUT` =
/// `mm-out x (transpose-out W)`), and `rank b <= 1`.
fn cond_linear_bias() -> Value {
    json!([
        "and",
        ["mm-shape-ok", "x", spec_wt()],
        ["rank", "W", "==", 2],
        ["bcast-into", "b", spec_linear_out()],
        ["rank", "b", "<=", 1]
    ])
}

/// `add(linear(x,W), b) → linear(x,W,b)` — the bias epilogue folds
/// into the GEMM (Inductor's addmm; TASO's matmul+bias).
///
/// Python: `priorart.py` `LINEAR_BIAS_FOLD`, cond `_COND_LINEAR_BIAS`,
/// tags `prior_art`+`fusion`.
pub fn linear_bias_fold() -> Law {
    Law::new(
        "linear_bias_fold",
        Pat::op(
            "add",
            vec![
                Pat::op("linear", vec![Pat::var("x"), Pat::var("W")]),
                Pat::var("b"),
            ],
        ),
        Pat::op("linear", vec![Pat::var("x"), Pat::var("W"), Pat::var("b")]),
    )
    .with_law(
        "xWᵀ + b = linear(x, W, b) — the bias epilogue folds into the GEMM (Inductor's addmm; TASO's matmul+bias).",
    )
    .with_cond(cond_linear_bias())
    .with_tag("prior_art")
    .with_tag("fusion")
}

/// `add(b, linear(x,W)) → linear(x,W,b)` — the operand-swapped
/// bias-epilogue fold.
///
/// Python: `priorart.py` `LINEAR_BIAS_FOLD_L`, cond
/// `_COND_LINEAR_BIAS`, tags `prior_art`+`fusion`.
pub fn linear_bias_fold_l() -> Law {
    Law::new(
        "linear_bias_fold_l",
        Pat::op(
            "add",
            vec![
                Pat::var("b"),
                Pat::op("linear", vec![Pat::var("x"), Pat::var("W")]),
            ],
        ),
        Pat::op("linear", vec![Pat::var("x"), Pat::var("W"), Pat::var("b")]),
    )
    .with_law("The operand-swapped bias-epilogue fold.")
    .with_cond(cond_linear_bias())
    .with_tag("prior_art")
    .with_tag("fusion")
}

/// `linear(x,W,b) → add(linear(x,W), b)` — expose the fused bias as a
/// pointwise epilogue; derivation `linear_bias_fold`.
///
/// Python: `priorart.py` `LINEAR_BIAS_EXPAND`, cond
/// `_COND_LINEAR_BIAS`, tags `prior_art`+`expansive`.
pub fn linear_bias_expand() -> Law {
    Law::new(
        "linear_bias_expand",
        Pat::op("linear", vec![Pat::var("x"), Pat::var("W"), Pat::var("b")]),
        Pat::op(
            "add",
            vec![
                Pat::op("linear", vec![Pat::var("x"), Pat::var("W")]),
                Pat::var("b"),
            ],
        ),
    )
    .with_law(
        "Reverse: expose the fused bias as a pointwise epilogue (eqsat weighs fused vs spelled).",
    )
    .with_cond(cond_linear_bias())
    .with_derivation("linear_bias_fold")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `add(linear(x,A,b1), linear(x,B,b2)) → linear(x, A+B, b1+b2)` —
/// the affine-map weight merge (model-soup / LoRA-merge class).
///
/// Python: `priorart.py` `LINEAR_BIAS_MERGE`, cond
/// `_COND_LIN_BIAS_MERGE` (`rank A == 2`, `rank B == 2`,
/// `shape-compat A B`, `shape-compat b1 b2`), tags
/// `prior_art`+`fusion`.
pub fn linear_bias_merge() -> Law {
    Law::new(
        "linear_bias_merge",
        Pat::op(
            "add",
            vec![
                Pat::op(
                    "linear",
                    vec![Pat::var("x"), Pat::var("A"), Pat::var("b1")],
                ),
                Pat::op(
                    "linear",
                    vec![Pat::var("x"), Pat::var("B"), Pat::var("b2")],
                ),
            ],
        ),
        Pat::op(
            "linear",
            vec![
                Pat::var("x"),
                Pat::op("add", vec![Pat::var("A"), Pat::var("B")]),
                Pat::op("add", vec![Pat::var("b1"), Pat::var("b2")]),
            ],
        ),
    )
    .with_law(
        "linear(x,A,b1) + linear(x,B,b2) = linear(x, A+B, b1+b2) — the affine-map weight merge (model-soup / LoRA-merge class).",
    )
    .with_cond(json!([
        "and",
        ["rank", "A", "==", 2],
        ["rank", "B", "==", 2],
        ["shape-compat", "A", "B"],
        ["shape-compat", "b1", "b2"]
    ]))
    .with_tag("prior_art")
    .with_tag("fusion")
}

// ---------------------------------------------------------------------------
//  prior_art §1 — TASO: conv output merge (the Inception-style fold)
// ---------------------------------------------------------------------------

/// `_CONV_ATTR_MV`'s attr tails in Python order — the shared
/// metavar names (`ST`/`PD`/`DL`/`GR`) bind both convs' attrs AND the
/// merged conv's to the SAME spelling, so all three nodes carry the
/// identical `attrs_fields`.
const PA_CONV_TAILS: &[(&str, &[(&str, &str)])] = &[
    ("", &[]),
    ("_sp", &[("stride", "ST"), ("padding", "PD")]),
    (
        "_spdg",
        &[
            ("stride", "ST"),
            ("padding", "PD"),
            ("dilation", "DL"),
            ("groups", "GR"),
        ],
    ),
];

/// `conv2d(x, W[, b], attrs…)` — the conv-merge operand pattern; the
/// attr tail binds `ST`/`PD`/`DL`/`GR` per `_CONV_ATTR_MV`.
fn pa_conv_pat(x: Pat, w: Pat, b: Option<Pat>, tail: &[(&str, &str)]) -> Pat {
    let mut op = OpPat::new("conv2d").arg(x).arg(w);
    if let Some(b) = b {
        op = op.arg(b);
    }
    if tail.is_empty() {
        // Python `attrs {}` — no declared attrs (the `Pat::op`
        // convention: no `__attrs__` key).
        Pat::Op(op)
    } else {
        Pat::Op(op.attrs_fields(afields(
            tail.iter().map(|&(k, m)| (k, af_bind(m))).collect(),
        )))
    }
}

/// `_conv_merge_cond(attrs, bias)` — the same-geometry guard: `x`/`W1`
/// rank-4 concrete, the input-channel agreement (`dim-eq x 1 W1 1`,
/// plus `attr-eq GR 1` when `groups` is bound — groups>1 needs a
/// channel interleave, not a plain cat), the filter-bank channel
/// agreement, the output concat riding `x`'s channel axis (`CD` at
/// axis 1), and the bias pair when present.
fn pa_conv_merge_cond(has_groups: bool, bias: bool) -> Value {
    let inc = if has_groups {
        json!(["and", ["attr-eq", "GR", 1], ["dim-eq", "x", 1, "W1", 1]])
    } else {
        json!(["dim-eq", "x", 1, "W1", 1])
    };
    let mut items = vec![
        json!(["concrete", "x"]),
        json!(["rank", "x", "==", 4]),
        json!(["concrete", "W1"]),
        json!(["rank", "W1", "==", 4]),
        inc,
        json!(["dim-eq", "W1", 1, "W2", 1]),
        json!(["axis", "x", "CD", 1]),
    ];
    if bias {
        items.extend([
            json!(["rank", "b1", "==", 1]),
            json!(["dim-eq", "b1", 0, "W1", 0]),
            json!(["rank", "b2", "==", 1]),
            json!(["dim-eq", "b2", 0, "W2", 0]),
        ]);
    }
    let mut cond = vec![json!("and")];
    cond.extend(items);
    Value::Array(cond)
}

/// `_make_conv_merge_rules` — `cat(conv(x,W1), conv(x,W2), ch) =
/// conv(x, cat(W1,W2,0))`: two same-geometry convs over a shared
/// input are one conv over the output-concat'd filter bank (TASO's
/// conv-merge; the Inception parallel fold). Three attr tails
/// (`""`/`"_sp"`/`"_spdg"`) × bias arity = six rules.
fn pa_conv_merge_rules() -> Vec<Law> {
    let mut out = Vec::new();
    for &(suffix, tail) in PA_CONV_TAILS {
        let has_groups = tail.iter().any(|&(_, m)| m == "GR");
        for bias in [true, false] {
            let bs = if bias { "" } else { "_nobias" };
            let conv1 = pa_conv_pat(
                Pat::var("x"),
                Pat::var("W1"),
                bias.then(|| Pat::var("b1")),
                tail,
            );
            let conv2 = pa_conv_pat(
                Pat::var("x"),
                Pat::var("W2"),
                bias.then(|| Pat::var("b2")),
                tail,
            );
            let lhs = Pat::Op(
                OpPat::new("concat")
                    .arg(conv1)
                    .arg(conv2)
                    .attrs_fields(concat_dim_named("CD")),
            );
            // The merged filter bank: `concat(W1, W2, dim=0)` (and the
            // bias pair likewise when present).
            let cat0 = |p: Pat, q: Pat| {
                Pat::Op(
                    OpPat::new("concat")
                        .arg(p)
                        .arg(q)
                        .attrs_lit(json!({"dim": 0})),
                )
            };
            let w_cat = cat0(Pat::var("W1"), Pat::var("W2"));
            let rhs = if bias {
                pa_conv_pat(
                    Pat::var("x"),
                    w_cat,
                    Some(cat0(Pat::var("b1"), Pat::var("b2"))),
                    tail,
                )
            } else {
                pa_conv_pat(Pat::var("x"), w_cat, None, tail)
            };
            out.push(
                Law::new(
                    format!("conv_out_merge{suffix}{bs}"),
                    lhs,
                    rhs,
                )
                .with_law(
                    "cat(conv(x,W1), conv(x,W2)) = conv(x, cat(W1,W2,0)) on the channel axis — same-geometry convs merge into one (TASO's conv-merge; the Inception parallel fold).",
                )
                .with_cond(pa_conv_merge_cond(has_groups, bias))
                .with_tag("prior_art")
                .with_tag("fusion"),
            );
        }
    }
    out
}

/// `pow_one`: `pow(u, P) → u` guarded by `P` being Const 1.
///
/// Python: `priorart.py` `POW_ONE`, tags `prior_art`+`simplification`.
pub fn pow_one() -> Law {
    Law::new(
        "pow_one",
        Pat::op("pow", vec![Pat::var("u"), Pat::var("P")]),
        Pat::var("u"),
    )
    .with_law("pow(u, 1) = u.")
    .with_cond(json!(["const-cmp", "P", "==", 1]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `pow_zero`: `pow(u, 0) → ones_like(u)` — unguarded (the exponent
/// is a pinned Const leaf); ported with its `pow_*` family.
///
/// Python: `priorart.py` `POW_ZERO`, tags `prior_art`+`simplification`.
pub fn pow_zero() -> Law {
    Law::new(
        "pow_zero",
        Pat::op("pow", vec![Pat::var("u"), Pat::constant(json!(0))]),
        Pat::op("ones_like", vec![Pat::var("u")]),
    )
 .with_law(
 "pow(u, 0) = ones(u) — IEEE-total including NaN bases (pow(nan,0)=1); spelled ones_like so the mint carries u's shape (the eval compare is shape-strict, not broadcast).",
    )
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `pow_neg_one`: `pow(u, −1) → reciprocal(u)` — the exponent is a
/// pinned Const leaf; the guard excludes an int-typed (const-num)
/// base, where `pow(2, −1)` would be int exponentiation.
///
/// Python: `priorart.py` `POW_NEG_ONE`, tags `prior_art`+`simplification`.
pub fn pow_neg_one() -> Law {
    Law::new(
        "pow_neg_one",
        Pat::op("pow", vec![Pat::var("u"), Pat::constant(json!(-1))]),
        Pat::op("reciprocal", vec![Pat::var("u")]),
    )
 .with_law(
 "pow(u, −1) = 1/u — the reciprocal spelling.  The Const-operand corner is excluded: an int-typed base does int exponentiation (pow(2,−1) → 0), which the pattern language can't dtype-guard otherwise.",
    )
    .with_cond(json!(["not", ["const-num", "u"]]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `pow_half`: `pow(u, P) → sqrt(u)` guarded by `P` being Const 0.5.
///
/// Python: `priorart.py` `POW_HALF`, tags `prior_art`+`simplification`.
pub fn pow_half() -> Law {
    Law::new(
        "pow_half",
        Pat::op("pow", vec![Pat::var("u"), Pat::var("P")]),
        Pat::op("sqrt", vec![Pat::var("u")]),
    )
    .with_law("pow(u, 0.5) = sqrt(u) — the canonical square root.")
    .with_cond(json!(["const-cmp", "P", "==", 0.5]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `sqrt_to_pow`: `sqrt(u) → pow(u, 0.5)` — unguarded; the pow
/// spelling (bridge to the pow family).
///
/// Python: `priorart.py` `SQRT_TO_POW`, tags `prior_art`+`expansive`,
/// derivation `pow_half`.
pub fn sqrt_to_pow() -> Law {
    Law::new(
        "sqrt_to_pow",
        Pat::op("sqrt", vec![Pat::var("u")]),
        Pat::op("pow", vec![Pat::var("u"), Pat::constant(json!(0.5))]),
    )
    .with_law("sqrt(u) = u^0.5 — the pow spelling (bridge to the pow family).")
    .with_derivation("pow_half")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `div_one_recip`: `div(ONE, u) → reciprocal(u)` guarded by `ONE`
/// being Const 1.
///
/// Python: `priorart.py` `DIV_ONE_RECIP`, tags `prior_art`+`simplification`.
pub fn div_one_recip() -> Law {
    Law::new(
        "div_one_recip",
        Pat::op("div", vec![Pat::var("ONE"), Pat::var("u")]),
        Pat::op("reciprocal", vec![Pat::var("u")]),
    )
    .with_law("1/u = reciprocal(u) — the div-spelled reciprocal folds.")
    .with_cond(json!(["const-cmp", "ONE", "==", 1]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `recip_expand`: `reciprocal(u) → div(1, u)` — unguarded; the div
/// spelling.
///
/// Python: `priorart.py` `RECIP_EXPAND`, tags `prior_art`+`expansive`,
/// derivation `div_one_recip`.
pub fn recip_expand() -> Law {
    Law::new(
        "recip_expand",
        Pat::op("reciprocal", vec![Pat::var("u")]),
        Pat::op("div", vec![Pat::constant(json!(1)), Pat::var("u")]),
    )
    .with_law("reciprocal(u) = 1/u — the div spelling.")
    .with_derivation("div_one_recip")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `sigmoid_fold`: `reciprocal(add(ONE, exp(neg(x)))) → sigmoid(x)`
/// guarded by `ONE` being Const 1.
///
/// Python: `priorart.py` `SIGMOID_FOLD`, tags
/// `prior_art`+`simplification`, derivation `sigmoid_expand`.
pub fn sigmoid_fold() -> Law {
    Law::new(
        "sigmoid_fold",
        Pat::op(
            "reciprocal",
            vec![Pat::op(
                "add",
                vec![
                    Pat::var("ONE"),
                    Pat::op("exp", vec![Pat::op("neg", vec![Pat::var("x")])]),
                ],
            )],
        ),
        Pat::op("sigmoid", vec![Pat::var("x")]),
    )
 .with_law(
 "1/(1+e⁻ˣ) folds to σ(x) — the manual-spelling fold (softmax_fold's sibling one kernel over).",
    )
    .with_cond(json!(["const-cmp", "ONE", "==", 1]))
    .with_derivation("sigmoid_expand")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `sigmoid_fold_div`: the div-spelled `1/(1+e⁻ˣ)` folds to
/// `sigmoid(x)` — both unit constants pinned by the guard.
///
/// Python: `priorart.py` `SIGMOID_FOLD_DIV`, tags
/// `prior_art`+`simplification`, derivation `sigmoid_expand`.
pub fn sigmoid_fold_div() -> Law {
    Law::new(
        "sigmoid_fold_div",
        Pat::op(
            "div",
            vec![
                Pat::var("ONE"),
                Pat::op(
                    "add",
                    vec![
                        Pat::var("ONE2"),
                        Pat::op("exp", vec![Pat::op("neg", vec![Pat::var("x")])]),
                    ],
                ),
            ],
        ),
        Pat::op("sigmoid", vec![Pat::var("x")]),
    )
    .with_law("The div-spelled 1/(1+e⁻ˣ) folds to σ(x).")
    .with_cond(json!([
        "and",
        ["const-cmp", "ONE", "==", 1],
        ["const-cmp", "ONE2", "==", 1]
    ]))
    .with_derivation("sigmoid_expand")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `tanh_fold`: `sub(mul(TWO, sigmoid(mul(TWO2, x))), ONE) → tanh(x)`
/// — all three constants pinned by the guard.
///
/// Python: `priorart.py` `TANH_FOLD`, tags
/// `prior_art`+`simplification`, derivation `tanh_expand`.
pub fn tanh_fold() -> Law {
    Law::new(
        "tanh_fold",
        Pat::op(
            "sub",
            vec![
                Pat::op(
                    "mul",
                    vec![
                        Pat::var("TWO"),
                        Pat::op(
                            "sigmoid",
                            vec![Pat::op("mul", vec![Pat::var("TWO2"), Pat::var("x")])],
                        ),
                    ],
                ),
                Pat::var("ONE"),
            ],
        ),
        Pat::op("tanh", vec![Pat::var("x")]),
    )
    .with_law("2σ(2x) − 1 folds to tanh(x).")
    .with_cond(json!([
        "and",
        ["const-cmp", "TWO", "==", 2],
        ["const-cmp", "TWO2", "==", 2],
        ["const-cmp", "ONE", "==", 1]
    ]))
    .with_derivation("tanh_expand")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `hardsigmoid_fold`: `div(relu6(add(x, THREE)), SIX) →
/// hardsigmoid(x)` — both constants pinned by the guard.
///
/// Python: `priorart.py` `HARDSIGMOID_FOLD`, tags
/// `prior_art`+`simplification`, derivation `hardsigmoid_expand`.
pub fn hardsigmoid_fold() -> Law {
    Law::new(
        "hardsigmoid_fold",
        Pat::op(
            "div",
            vec![
                Pat::op(
                    "relu6",
                    vec![Pat::op("add", vec![Pat::var("x"), Pat::var("THREE")])],
                ),
                Pat::var("SIX"),
            ],
        ),
        Pat::op("hardsigmoid", vec![Pat::var("x")]),
    )
    .with_law("relu6(x+3)/6 folds to hardsigmoid(x).")
    .with_cond(json!([
        "and",
        ["const-cmp", "THREE", "==", 3],
        ["const-cmp", "SIX", "==", 6]
    ]))
    .with_derivation("hardsigmoid_expand")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `neg_const`: `neg(C) → NC` — the negated value minted as a Const
/// leaf through `dspec` (`$leaf:` channel).
///
/// Python: `priorart.py` `NEG_CONST`, tags `prior_art`+`simplification`.
pub fn neg_const() -> Law {
    Law::new(
        "neg_const",
        Pat::op("neg", vec![Pat::var("C")]),
        Pat::var("NC"),
    )
    .with_law("−c folds to the negated Const leaf.")
    .with_cond(json!(["const-num", "C"]))
    .with_dspec(json!({"$leaf:NC": ["neg", ["const", "C"]]}))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// The `mul_const_assoc_{0..3}` family: `mul(·c1,·c2) → mul(·,c1·c2)`
/// in all four operand orders — the scalar-factor fold (Inductor
/// const-fold), `C3` minted by `dspec`.
///
/// Python: `priorart.py` `_mul_const_assoc`, tags
/// `prior_art`+`simplification`.
fn mul_const_assoc_rules() -> Vec<Law> {
    let cond = json!(["and", ["const-num", "C1"], ["const-num", "C2"]]);
    let dspec = json!({"$leaf:C3": ["mul", ["const", "C1"], ["const", "C2"]]});
    let law = "(u·c1)·c2 = u·(c1c2) — the scalar-factor fold (Inductor const-fold).";
    let inner_l = || Pat::op("mul", vec![Pat::var("u"), Pat::var("C1")]);
    let inner_r = || Pat::op("mul", vec![Pat::var("C1"), Pat::var("u")]);
    let cases: [(Pat, Pat); 4] = [
        (
            Pat::op("mul", vec![inner_l(), Pat::var("C2")]),
            Pat::op("mul", vec![Pat::var("u"), Pat::var("C3")]),
        ),
        (
            Pat::op("mul", vec![inner_r(), Pat::var("C2")]),
            Pat::op("mul", vec![Pat::var("u"), Pat::var("C3")]),
        ),
        (
            Pat::op("mul", vec![Pat::var("C2"), inner_l()]),
            Pat::op("mul", vec![Pat::var("C3"), Pat::var("u")]),
        ),
        (
            Pat::op("mul", vec![Pat::var("C2"), inner_r()]),
            Pat::op("mul", vec![Pat::var("C3"), Pat::var("u")]),
        ),
    ];
    cases
        .into_iter()
        .enumerate()
        .map(|(i, (lhs, rhs))| {
            Law::new(format!("mul_const_assoc_{i}"), lhs, rhs)
                .with_law(law)
                .with_cond(cond.clone())
                .with_dspec(dspec.clone())
                .with_tag("prior_art")
                .with_tag("simplification")
        })
        .collect()
}

/// The `add_const_assoc_{0..3}` family: `add(·c1,·c2) → add(·,c1+c2)`
/// in all four operand orders — the offset fold.
///
/// Python: `priorart.py` `_add_const_assoc`, tags
/// `prior_art`+`simplification`.
fn add_const_assoc_rules() -> Vec<Law> {
    let cond = json!(["and", ["const-num", "C1"], ["const-num", "C2"]]);
    let dspec = json!({"$leaf:C3": ["add", ["const", "C1"], ["const", "C2"]]});
    let law = "(u+c1)+c2 = u+(c1+c2) — the offset fold.";
    let inner_l = || Pat::op("add", vec![Pat::var("u"), Pat::var("C1")]);
    let inner_r = || Pat::op("add", vec![Pat::var("C1"), Pat::var("u")]);
    let cases: [(Pat, Pat); 4] = [
        (
            Pat::op("add", vec![inner_l(), Pat::var("C2")]),
            Pat::op("add", vec![Pat::var("u"), Pat::var("C3")]),
        ),
        (
            Pat::op("add", vec![inner_r(), Pat::var("C2")]),
            Pat::op("add", vec![Pat::var("u"), Pat::var("C3")]),
        ),
        (
            Pat::op("add", vec![Pat::var("C2"), inner_l()]),
            Pat::op("add", vec![Pat::var("C3"), Pat::var("u")]),
        ),
        (
            Pat::op("add", vec![Pat::var("C2"), inner_r()]),
            Pat::op("add", vec![Pat::var("C3"), Pat::var("u")]),
        ),
    ];
    cases
        .into_iter()
        .enumerate()
        .map(|(i, (lhs, rhs))| {
            Law::new(format!("add_const_assoc_{i}"), lhs, rhs)
                .with_law(law)
                .with_cond(cond.clone())
                .with_dspec(dspec.clone())
                .with_tag("prior_art")
                .with_tag("simplification")
        })
        .collect()
}

/// `sub_const_sub`: `sub(sub(u, C1), C2) → sub(u, C3)` —
/// `(u−c1)−c2 = u−(c1+c2)`, `C3` minted by `dspec`.
///
/// Python: `priorart.py` `SUB_CONST_SUB`, tags
/// `prior_art`+`simplification`.
pub fn sub_const_sub() -> Law {
    Law::new(
        "sub_const_sub",
        Pat::op(
            "sub",
            vec![
                Pat::op("sub", vec![Pat::var("u"), Pat::var("C1")]),
                Pat::var("C2"),
            ],
        ),
        Pat::op("sub", vec![Pat::var("u"), Pat::var("C3")]),
    )
    .with_law("(u−c1)−c2 = u−(c1+c2).")
    .with_cond(json!(["and", ["const-num", "C1"], ["const-num", "C2"]]))
    .with_dspec(json!({"$leaf:C3": ["add", ["const", "C1"], ["const", "C2"]]}))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `sub_const_sub_l`: `sub(sub(C1, u), C2) → sub(C3, u)` —
/// `(c1−u)−c2 = (c1−c2)−u`.
///
/// Python: `priorart.py` `SUB_CONST_SUB_L`, tags
/// `prior_art`+`simplification`.
pub fn sub_const_sub_l() -> Law {
    Law::new(
        "sub_const_sub_l",
        Pat::op(
            "sub",
            vec![
                Pat::op("sub", vec![Pat::var("C1"), Pat::var("u")]),
                Pat::var("C2"),
            ],
        ),
        Pat::op("sub", vec![Pat::var("C3"), Pat::var("u")]),
    )
    .with_law("(c1−u)−c2 = (c1−c2)−u.")
    .with_cond(json!(["and", ["const-num", "C1"], ["const-num", "C2"]]))
    .with_dspec(json!({"$leaf:C3": ["sub", ["const", "C1"], ["const", "C2"]]}))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `div_const_div`: `div(div(u, C1), C2) → div(u, C3)` —
/// `(u/c1)/c2 = u/(c1·c2)`.
///
/// Python: `priorart.py` `DIV_CONST_DIV`, tags
/// `prior_art`+`simplification`.
pub fn div_const_div() -> Law {
    Law::new(
        "div_const_div",
        Pat::op(
            "div",
            vec![
                Pat::op("div", vec![Pat::var("u"), Pat::var("C1")]),
                Pat::var("C2"),
            ],
        ),
        Pat::op("div", vec![Pat::var("u"), Pat::var("C3")]),
    )
    .with_law("(u/c1)/c2 = u/(c1·c2).")
    .with_cond(json!(["and", ["const-num", "C1"], ["const-num", "C2"]]))
    .with_dspec(json!({"$leaf:C3": ["mul", ["const", "C1"], ["const", "C2"]]}))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `div_const_to_mul`: `div(u, C) → mul(u, CI)` guarded by `C` being
/// a nonzero Const — `u/c = u·(1/c)`, `CI` minted by `dspec`.
///
/// Python: `priorart.py` `DIV_CONST_TO_MUL`, tags
/// `prior_art`+`simplification`.
pub fn div_const_to_mul() -> Law {
    Law::new(
        "div_const_to_mul",
        Pat::op("div", vec![Pat::var("u"), Pat::var("C")]),
        Pat::op("mul", vec![Pat::var("u"), Pat::var("CI")]),
    )
 .with_law(
 "u/c = u·(1/c) — the div→mul strength reduction (the softcap/quant scale's canonical form).",
    )
    .with_cond(json!([
        "and",
        ["const-num", "C"],
        ["not", ["const-cmp", "C", "==", 0]]
    ]))
    .with_dspec(json!({"$leaf:CI": ["recip", ["const", "C"]]}))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `mul_const_div`: `mul(div(u, C1), C2) → mul(u, C3)` —
/// `(u/c1)·c2 = u·(c2/c1)`.
///
/// Python: `priorart.py` `MUL_CONST_DIV`, tags
/// `prior_art`+`simplification`.
pub fn mul_const_div() -> Law {
    Law::new(
        "mul_const_div",
        Pat::op(
            "mul",
            vec![
                Pat::op("div", vec![Pat::var("u"), Pat::var("C1")]),
                Pat::var("C2"),
            ],
        ),
        Pat::op("mul", vec![Pat::var("u"), Pat::var("C3")]),
    )
    .with_law("(u/c1)·c2 = u·(c2/c1).")
    .with_cond(json!(["and", ["const-num", "C1"], ["const-num", "C2"]]))
    .with_dspec(json!({"$leaf:C3": ["fdiv", ["const", "C2"], ["const", "C1"]]}))
    .with_tag("prior_art")
    .with_tag("simplification")
}

// ---------------------------------------------------------------------------
//  prior_art §2 head — `mul_unsq_rev` (the §1 index-view tail) and the
//  no-op elimination block.  The declarative `cond=` guards port
//  verbatim; the procedural `check=` hooks are re-authored as `cond`
//  trees over the ported vocabulary (`mod`/`len`/`shape` exprs,
//  `unique`/`sum`/`posmap` sequence reads, `attr-is` null arms) — the
//  same re-authoring posture as the layout pack.  `squeeze_all_noop`
//  and `expand_noop` stay Python-side: their all-dims guards need a
//  forall the vocabulary cannot form.
// ---------------------------------------------------------------------------

/// `mul_unsq_rev`: `unsq(u·v, d) = unsq(u, d)·unsq(v, d)` — the
/// broadcast-sink direction, guarded to the `flat-map-unsq` region
/// AND `UD` literal-in-range for both operand unsqs.
///
/// Python: `priorart.py` `MUL_UNSQ_REV`, cond verbatim, tags
/// `prior_art`+`categorical`.
pub fn mul_unsq_rev() -> Law {
    let unsq = |t: Pat| {
        Pat::Op(
            OpPat::new("unsqueeze")
                .arg(t)
                .attrs_fields(afields(vec![("dim", af_bind("UD"))])),
        )
    };
    Law::new(
        "mul_unsq_rev",
        Pat::Op(
            OpPat::new("unsqueeze")
                .arg(Pat::op("mul", vec![Pat::var("u"), Pat::var("v")]))
                .attrs_fields(afields(vec![("dim", af_bind("UD"))])),
        ),
        Pat::op("mul", vec![unsq(Pat::var("u")), unsq(Pat::var("v"))]),
    )
    .with_law(
        "unsq(u·v, d) = unsq(u, d)·unsq(v, d) — the size-1-axis insert distributes over elementwise mul: XLA's broadcast-sink direction, guarded to the flat-map-unsq region AND UD literal-in-range for both operand unsqs (the implicit-broadcast soundness envelope).",
    )
    .with_cond(json!([
        "and",
        ["flat-map-unsq", "u", "UD", "v", ["unsq-out", "v", "UD"]],
        ["attr-cmp", "UD", ">=", ["neg", ["add", ["len", ["shape", "u"]], 1]]],
        ["attr-cmp", "UD", "<=", ["len", ["shape", "u"]]],
        ["attr-cmp", "UD", ">=", ["neg", ["add", ["len", ["shape", "v"]], 1]]],
        ["attr-cmp", "UD", "<=", ["len", ["shape", "v"]]]
    ]))
    .with_tag("prior_art")
    .with_tag("categorical")
}

/// The value-identity relayout strips — `op(u) → u` for the ops the
/// IR treats as materialisation/relabel boundaries, not value
/// changes (`contiguous_id`/`clone_id`/`alias_id`/`detach_id`/
/// `detach_inplace_id`). Unguarded, tags `prior_art`+`simplification`.
fn strip_id(name: &str, op: &str, law: &str) -> Law {
    Law::new(name, Pat::op(op, vec![Pat::var("u")]), Pat::var("u"))
        .with_law(law)
        .with_tag("prior_art")
        .with_tag("simplification")
}

/// `contiguous_id`: `contiguous(u) → u`.
///
/// Python: `priorart.py` `CONTIGUOUS_ID`.
pub fn contiguous_id() -> Law {
    strip_id(
        "contiguous_id",
        "contiguous",
        "contiguous(x) is a value-identity relayout — the IR's semantics is value-level, so the strip is exact.",
    )
}

/// `clone_id`: `clone(u) → u`.
///
/// Python: `priorart.py` `CLONE_ID`.
pub fn clone_id() -> Law {
    strip_id(
        "clone_id",
        "clone",
        "clone(x) preserves values exactly — a materialization, not a value change.",
    )
}

/// `alias_id`: `alias(u) → u`.
///
/// Python: `priorart.py` `ALIAS_ID`.
pub fn alias_id() -> Law {
    strip_id(
        "alias_id",
        "alias",
        "alias(x) is the identity leaf-relabel.",
    )
}

/// `detach_id`: `detach(u) → u`.
///
/// Python: `priorart.py` `DETACH_ID`.
pub fn detach_id() -> Law {
    strip_id(
        "detach_id",
        "detach",
        "detach(x) is the value identity — the autograd boundary the IR does not model (Inductor strips it in inference).",
    )
}

/// `detach_inplace_id`: `detach_(u) → u`.
///
/// Python: `priorart.py` `DETACH_INPLACE_ID`.
pub fn detach_inplace_id() -> Law {
    strip_id(
        "detach_inplace_id",
        "detach_",
        "detach_(x) — the in-place spelling of the same value identity.",
    )
}

/// `CAST_IDEM_RULES` — `float`/`half`/`double`/`bfloat16` idempotence:
/// `op(op(u)) → op(u)`.
fn cast_idem_rules() -> Vec<Law> {
    ["float", "half", "double", "bfloat16"]
        .into_iter()
        .map(|op| {
            Law::new(
                format!("{op}_idem"),
                Pat::op(op, vec![Pat::op(op, vec![Pat::var("u")])]),
                Pat::op(op, vec![Pat::var("u")]),
            )
            .with_law(format!(
                "{op}({op}(x)) = {op}(x) — dtype casts are idempotent."
            ))
            .with_tag("prior_art")
            .with_tag("simplification")
        })
        .collect()
}

/// `unsqueeze_squeeze`: `squeeze(unsqueeze(u, UD), SD) → u` when the
/// squeezed axis IS the inserted one — `UD % (rank(u)+1) ==
/// SD % (rank(u)+1)`, re-authored from `_check_unsq_sq` as a `mod`
/// expr pair over `len(shape(u))+1`.
///
/// Python: `priorart.py` `UNSQUEEZE_SQUEEZE`, check `_check_unsq_sq`,
/// tags `prior_art`+`simplification`.
pub fn unsqueeze_squeeze() -> Law {
    Law::new(
        "unsqueeze_squeeze",
        Pat::Op(
            OpPat::new("squeeze")
                .arg(Pat::Op(
                    OpPat::new("unsqueeze")
                        .arg(Pat::var("u"))
                        .attrs_fields(afields(vec![("dim", af_bind("UD"))])),
                ))
                .attrs_fields(afields(vec![("dim", af_bind("SD"))])),
        ),
        Pat::var("u"),
    )
    .with_law(
        "squeeze(unsqueeze(u, d), d) = u — the insert/remove pair cancels on the same normalized axis.",
    )
    .with_cond(json!([
        "and",
        ["attr-type", "UD", "int"],
        ["attr-type", "SD", "int"],
        [
            "attr-cmp",
            ["mod", ["attr", "UD"], ["add", ["len", ["shape", "u"]], 1]],
            "==",
            ["mod", ["attr", "SD"], ["add", ["len", ["shape", "u"]], 1]]
        ]
    ]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `squeeze_unsqueeze`: `unsqueeze(squeeze(u, D), D) → u` when
/// `u[D] == 1` — the remove/insert pair cancels on a size-1 axis
/// only. `_check_sq_unsq` re-authored: `attr-cmp-dim` reads the
/// bound dim attr against `shape(u)` (its `_dim_at` range check is
/// the in-range half of the hook's `s[d % len(s)]` read — an
/// out-of-range spelled dim is not a valid squeeze axis anyway).
///
/// Python: `priorart.py` `SQUEEZE_UNSQUEEZE`, check `_check_sq_unsq`,
/// tags `prior_art`+`simplification`.
pub fn squeeze_unsqueeze() -> Law {
    Law::new(
        "squeeze_unsqueeze",
        Pat::Op(
            OpPat::new("unsqueeze")
                .arg(Pat::Op(
                    OpPat::new("squeeze")
                        .arg(Pat::var("u"))
                        .attrs_fields(afields(vec![("dim", af_bind("D"))])),
                ))
                .attrs_fields(afields(vec![("dim", af_bind("D"))])),
        ),
        Pat::var("u"),
    )
    .with_law(
        "unsqueeze(squeeze(u, d), d) = u when u[d] == 1 — the remove/insert pair cancels on a size-1 axis only.",
    )
    .with_cond(json!([
        "and",
        ["attr-type", "D", "int"],
        ["attr-cmp-dim", 1, "==", "u", "D"]
    ]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `_COND_AXIS` — the nonneg in-range axis spelling shared by the
/// slice no-op pair (Python `priorart.py` `_COND_AXIS`).
fn cond_axis() -> Value {
    json!([
        "and",
        ["attr-type", "D", "int"],
        ["attr-cmp", "D", ">=", 0],
        ["attr-cmp", ["len", ["shape", "u"]], ">", ["attr", "D"]]
    ])
}

/// `squeeze_noop`: `squeeze(u, D) → u` when `u[D] != 1`.
///
/// Python: `priorart.py` `SQUEEZE_NOOP`, cond verbatim, tags
/// `prior_art`+`simplification`.
pub fn squeeze_noop() -> Law {
    Law::new(
        "squeeze_noop",
        Pat::Op(
            OpPat::new("squeeze")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![("dim", af_bind("D"))])),
        ),
        Pat::var("u"),
    )
    .with_law(
        "squeeze(u, d) on a non-1 axis removes nothing — a no-op the exporter's belt-and-braces squeezes leave behind.",
    )
    .with_cond(json!(["attr-cmp-dim", 1, "!=", "u", "D"]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `reshape_noop`: `reshape(u, shape(u)) → u`.
///
/// Python: `priorart.py` `RESHAPE_NOOP`, cond verbatim, tags
/// `prior_art`+`simplification`.
pub fn reshape_noop() -> Law {
    Law::new(
        "reshape_noop",
        Pat::Op(
            OpPat::new("reshape")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![("shape", af_bind("S"))])),
        ),
        Pat::var("u"),
    )
    .with_law(
        "reshape(u, shape(u)) = u — the no-op reshape strip (Inductor view-chain simplification).",
    )
    .with_cond(json!(["attr-cmp", "S", "==", ["shape", "u"]]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `reshape_reshape`: `reshape(reshape(u, S0), S1) → reshape(u, S1)`.
///
/// Python: `priorart.py` `RESHAPE_RESHAPE`, tags
/// `prior_art`+`simplification`.
pub fn reshape_reshape() -> Law {
    Law::new(
        "reshape_reshape",
        Pat::Op(
            OpPat::new("reshape")
                .arg(Pat::Op(
                    OpPat::new("reshape")
                        .arg(Pat::var("u"))
                        .attrs_fields(afields(vec![("shape", af_bind("S0"))])),
                ))
                .attrs_fields(afields(vec![("shape", af_bind("S1"))])),
        ),
        Pat::Op(
            OpPat::new("reshape")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![("shape", af_bind("S1"))])),
        ),
    )
    .with_law(
        "reshape(reshape(u, s0), s1) = reshape(u, s1) — view composition keeps only the final target.",
    )
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `broadcast_to_noop`: `broadcast_to(u, shape(u)) → u`.
///
/// Python: `priorart.py` `BROADCAST_TO_NOOP`, cond verbatim, tags
/// `prior_art`+`simplification`.
pub fn broadcast_to_noop() -> Law {
    Law::new(
        "broadcast_to_noop",
        Pat::Op(
            OpPat::new("broadcast_to")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![("shape", af_bind("S"))])),
        ),
        Pat::var("u"),
    )
    .with_law("broadcast_to(u, shape(u)) = u — the same no-op for the materializing spelling.")
    .with_cond(json!(["attr-cmp", "S", "==", ["shape", "u"]]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `broadcast_to_expand`: `broadcast_to(u, S) → expand(u, S)` —
/// expansive, the strict spelling as a special case of the view.
///
/// Python: `priorart.py` `BROADCAST_TO_EXPAND`, tags
/// `prior_art`+`expansive`.
pub fn broadcast_to_expand() -> Law {
    Law::new(
        "broadcast_to_expand",
        Pat::Op(
            OpPat::new("broadcast_to")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![("shape", af_bind("S"))])),
        ),
        Pat::Op(
            OpPat::new("expand")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![("shape", af_bind("S"))])),
        ),
    )
    .with_law(
        "broadcast_to(u, s) = expand(u, s) — the strict spelling is a special case of the view.",
    )
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `slice_noop`: `slice(u, D, 0, E) → u` when `E >= u[D]` — the
/// full-cover slice strip.
///
/// Python: `priorart.py` `SLICE_NOOP`, cond verbatim, tags
/// `prior_art`+`simplification`.
pub fn slice_noop() -> Law {
    let mut cond = cond_axis();
    cond.as_array_mut()
        .expect("cond_axis is an and-node")
        .push(json!(["attr-cmp-dim", "E", ">=", "u", "D"]));
    Law::new(
        "slice_noop",
        Pat::Op(
            OpPat::new("slice")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![
                    ("dim", af_bind("D")),
                    ("start", af_lit(json!(0))),
                    ("end", af_bind("E")),
                ])),
        ),
        Pat::var("u"),
    )
    .with_law(
        "slice(u, d, 0, e) = u when e >= u[d] — the full-cover slice strip (the export's ``x[..., :e]`` no-op).",
    )
    .with_cond(cond)
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `slice_noop_nostart`: the start-elided `x[..., :e]` spelling.
///
/// Python: `priorart.py` `SLICE_NOOP_NOSTART`, cond verbatim, tags
/// `prior_art`+`simplification`.
pub fn slice_noop_nostart() -> Law {
    let mut cond = cond_axis();
    cond.as_array_mut()
        .expect("cond_axis is an and-node")
        .push(json!(["attr-cmp-dim", "E", ">=", "u", "D"]));
    Law::new(
        "slice_noop_nostart",
        Pat::Op(
            OpPat::new("slice")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![("dim", af_bind("D")), ("end", af_bind("E"))])),
        ),
        Pat::var("u"),
    )
    .with_law("The start-elided full-cover slice — ``x[..., :e]`` with e >= u[d].")
    .with_cond(cond)
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `index_select_id`: `index_select(u, D, range(u[D])) → u` — the
/// identity gather.
///
/// Python: `priorart.py` `INDEX_SELECT_ID`, cond verbatim, tags
/// `prior_art`+`simplification`.
pub fn index_select_id() -> Law {
    Law::new(
        "index_select_id",
        Pat::Op(
            OpPat::new("index_select")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![
                    ("dim", af_bind("D")),
                    ("index", af_bind("I")),
                ])),
        ),
        Pat::var("u"),
    )
    .with_law("index_select(u, d, range(u[d])) = u — the identity gather.")
    .with_cond(json!(["attr-range", "I", "u", "D"]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `roll_noop`: `roll(u, shifts=0) → u` — `_check_roll_noop`
/// re-authored. `SH` int spelling is `attr-eq 0`; the sequence
/// spelling is "at most one distinct value and it sums to 0" —
/// `len(unique(SH)) <= 1 ∧ sum(unique(SH)) == 0` iff every element
/// `== 0` (the unique/sum pair is the vocabulary's only elementwise
/// fold; non-sequence spellings decline inside `unique` exactly like
/// the hook's `isinstance` gate).
///
/// Python: `priorart.py` `ROLL_NOOP`, check `_check_roll_noop`,
/// tags `prior_art`+`simplification`.
pub fn roll_noop() -> Law {
    Law::new(
        "roll_noop",
        Pat::Op(
            OpPat::new("roll")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![
                    ("shifts", af_bind("SH")),
                    ("dims", af_bind("DM")),
                ])),
        ),
        Pat::var("u"),
    )
    .with_law("roll(u, shifts=0) = u — a zero shift on every axis.")
    .with_cond(json!([
        "or",
        ["and", ["attr-type", "SH", "int"], ["attr-eq", "SH", 0]],
        [
            "and",
            ["attr-cmp", ["len", ["unique", ["attr", "SH"]]], "<=", 1],
            ["attr-cmp", ["sum", ["unique", ["attr", "SH"]]], "==", 0]
        ]
    ]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `pad_noop`: `pad(u, zeros) → u` — `_check_pad_noop` re-authored
/// as the same unique/sum all-zero fold (the empty pad reads
/// `unique → ()`, `sum → 0`, matching `all(())`).
///
/// Python: `priorart.py` `PAD_NOOP`, check `_check_pad_noop`,
/// tags `prior_art`+`simplification`.
pub fn pad_noop() -> Law {
    Law::new(
        "pad_noop",
        Pat::Op(
            OpPat::new("pad")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![("pad", af_bind("P"))])),
        ),
        Pat::var("u"),
    )
    .with_law("pad(u, zeros) = u — an all-zero pad pad is a no-op.")
    .with_cond(json!([
        "and",
        ["attr-cmp", ["len", ["unique", ["attr", "P"]]], "<=", 1],
        ["attr-cmp", ["sum", ["unique", ["attr", "P"]]], "==", 0]
    ]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `repeat_noop`: `repeat(u, ones) → u` — `_check_repeat_noop`
/// re-authored: `len(S) == rank(shape(u))` (the full-cover half —
/// repeat left-pads the rank) AND every factor `== 1`, the latter as
/// "empty, or at most one distinct value summing to 1".
///
/// Python: `priorart.py` `REPEAT_NOOP`, check `_check_repeat_noop`,
/// tags `prior_art`+`simplification`.
pub fn repeat_noop() -> Law {
    Law::new(
        "repeat_noop",
        Pat::Op(
            OpPat::new("repeat")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![("shape", af_bind("S"))])),
        ),
        Pat::var("u"),
    )
    .with_law("repeat(u, ones) = u — tiling by 1 on every axis.")
    .with_cond(json!([
        "and",
        [
            "attr-cmp",
            ["len", ["attr", "S"]],
            "==",
            ["len", ["shape", "u"]]
        ],
        [
            "or",
            ["attr-cmp", ["len", ["unique", ["attr", "S"]]], "==", 0],
            [
                "and",
                ["attr-cmp", ["len", ["unique", ["attr", "S"]]], "==", 1],
                ["attr-cmp", ["sum", ["unique", ["attr", "S"]]], "==", 1]
            ]
        ]
    ]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `permute_noop`: `permute(u, identity) → u` — `_check_permute_id`
/// re-authored. `P == range(rank(u))` has no range expr, so the
/// identity is pinned three ways: `len(P) == rank`, all elements
/// distinct (`len(unique(P)) == len(P)` — kills `(0,0)`-style
/// non-permutations the `posmap` fixed-point alone would admit),
/// and `posmap(P, P) == P` — for a distinct sequence the posmap
/// yields the first-occurrence positions, so equality with `P`
/// forces `P[i] == i` elementwise.
///
/// Python: `priorart.py` `PERMUTE_NOOP`, check `_check_permute_id`,
/// tags `prior_art`+`simplification`.
pub fn permute_noop() -> Law {
    Law::new(
        "permute_noop",
        Pat::Op(
            OpPat::new("permute")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![("dims", af_bind("P"))])),
        ),
        Pat::var("u"),
    )
    .with_law("permute(u, identity) = u.")
    .with_cond(json!([
        "and",
        [
            "attr-cmp",
            ["len", ["attr", "P"]],
            "==",
            ["len", ["shape", "u"]]
        ],
        [
            "attr-cmp",
            ["len", ["unique", ["attr", "P"]]],
            "==",
            ["len", ["attr", "P"]]
        ],
        [
            "attr-cmp",
            ["posmap", ["attr", "P"], ["attr", "P"]],
            "==",
            ["attr", "P"]
        ]
    ]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `permute_compose`: `permute(permute(u,P),Q) → permute(u,C)` with
/// `C = gather(P, Q)` — permutation composition minted as the
/// index-of-index form (`C[j] = P[Q[j]]`).
///
/// Python: `priorart.py` `PERMUTE_COMPOSE`, cond + dspec verbatim,
/// tags `prior_art`+`simplification`.
pub fn permute_compose() -> Law {
    Law::new(
        "permute_compose",
        Pat::Op(
            OpPat::new("permute")
                .arg(Pat::Op(
                    OpPat::new("permute")
                        .arg(Pat::var("u"))
                        .attrs_fields(afields(vec![("dims", af_bind("P"))])),
                ))
                .attrs_fields(afields(vec![("dims", af_bind("Q"))])),
        ),
        Pat::Op(
            OpPat::new("permute")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![("dims", af_bind("C"))])),
        ),
    )
    .with_law(
        "permute(permute(u,P),Q) = permute(u, P∘Q) — permutation composition, minted as gather(P, Q).",
    )
    .with_cond(json!([
        "and",
        ["attr-ints", "P"],
        ["attr-ints", "Q"],
        ["attr-cmp", ["len", ["attr", "P"]], "==", ["len", ["shape", "u"]]],
        ["attr-cmp", ["len", ["attr", "Q"]], "==", ["len", ["attr", "P"]]]
    ]))
    .with_dspec(json!({"C": ["gather", ["attr", "P"], ["attr", "Q"]]}))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `movedim_involution`: `movedim(movedim(u,S,D),D,S) → u` — the
/// attr reuse across the two spellings pins source/destination
/// swapped; the cond keeps the int spelling the eval binding
/// iterates.
///
/// Python: `priorart.py` `MOVEDIM_INVOLUTION`, cond verbatim, tags
/// `prior_art`+`simplification`.
pub fn movedim_involution() -> Law {
    Law::new(
        "movedim_involution",
        Pat::Op(
            OpPat::new("movedim")
                .arg(Pat::Op(
                    OpPat::new("movedim")
                        .arg(Pat::var("u"))
                        .attrs_fields(afields(vec![
                            ("source", af_bind("S")),
                            ("destination", af_bind("D")),
                        ])),
                ))
                .attrs_fields(afields(vec![
                    ("source", af_bind("D")),
                    ("destination", af_bind("S")),
                ])),
        ),
        Pat::var("u"),
    )
    .with_law("movedim(movedim(u,s,d),d,s) = u — moving an axis forth and back is the identity.")
    .with_cond(json!([
        "and",
        ["attr-type", "S", "int"],
        ["attr-type", "D", "int"]
    ]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `flatten_noop`: `flatten(u, SD, ED) → u` when `SD % rank ==
/// ED % rank` — `_check_flatten_noop` re-authored. The hook's
/// `ed = -1 if ed is None` arm is the `attr-is ED null` branch (a
/// bound `null` `end_dim` reads as the -1 default; the fields
/// keyset still requires the attr present, matching `Op.make`'s
/// exact-keyset).
///
/// Python: `priorart.py` `FLATTEN_NOOP`, check `_check_flatten_noop`,
/// tags `prior_art`+`simplification`.
pub fn flatten_noop() -> Law {
    Law::new(
        "flatten_noop",
        Pat::Op(
            OpPat::new("flatten")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![
                    ("start_dim", af_bind("SD")),
                    ("end_dim", af_bind("ED")),
                ])),
        ),
        Pat::var("u"),
    )
    .with_law("flatten(u, i, i) = u — a one-axis merge is a no-op.")
    .with_cond(json!([
        "and",
        ["attr-type", "SD", "int"],
        [
            "attr-cmp",
            ["mod", ["attr", "SD"], ["len", ["shape", "u"]]],
            "==",
            [
                "mod",
                ["if", ["attr-is", "ED", null], -1, ["attr", "ED"]],
                ["len", ["shape", "u"]]
            ]
        ]
    ]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `unflatten_noop`: `unflatten(u, D, (u[D],)) → u` — a singleton
/// split restores the same axis.
///
/// Python: `priorart.py` `UNFLATTEN_NOOP`, cond verbatim, tags
/// `prior_art`+`simplification`.
pub fn unflatten_noop() -> Law {
    Law::new(
        "unflatten_noop",
        Pat::Op(
            OpPat::new("unflatten")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![
                    ("dim", af_bind("D")),
                    ("sizes", af_bind("SZ")),
                ])),
        ),
        Pat::var("u"),
    )
    .with_law("unflatten(u, d, (u[d],)) = u — a singleton split restores the same axis.")
    .with_cond(json!([
        "and",
        ["attr-len", "SZ", "==", 1],
        [
            "attr-cmp",
            ["nth", ["attr", "SZ"], 0],
            "==",
            ["nth", ["shape", "u"], ["attr", "D"]]
        ]
    ]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `split_single`: `split(u, (u[D],), D, 0) → u` — a one-section
/// split is the whole tensor.
///
/// Python: `priorart.py` `SPLIT_SINGLE`, cond verbatim, tags
/// `prior_art`+`simplification`.
pub fn split_single() -> Law {
    Law::new(
        "split_single",
        Pat::Op(
            OpPat::new("split")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![
                    ("sizes", af_bind("SZ")),
                    ("dim", af_bind("D")),
                    ("index", af_bind("I")),
                ])),
        ),
        Pat::var("u"),
    )
    .with_law(
        "split(u, (u[d],), d, 0) = u — a one-section split is the whole tensor (the chunk_single sibling).",
    )
    .with_cond(json!([
        "and",
        ["attr-eq", "I", 0],
        ["attr-len", "SZ", "==", 1],
        [
            "attr-cmp",
            ["nth", ["attr", "SZ"], 0],
            "==",
            ["nth", ["shape", "u"], ["attr", "D"]]
        ]
    ]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `flip_involution`: `flip(flip(u, D1), D2) → u` — `_check_flip_inv`
/// re-authored: the dims attrs must be int tuples and equal
/// (`attr-ints` + `attr-eq-attr`; the int-dims gate is the hook's
/// eval-binding note — an int `dims` can't spell flip).
///
/// Python: `priorart.py` `FLIP_INVOLUTION`, check `_check_flip_inv`,
/// tags `prior_art`+`simplification`.
pub fn flip_involution() -> Law {
    Law::new(
        "flip_involution",
        Pat::Op(
            OpPat::new("flip")
                .arg(Pat::Op(
                    OpPat::new("flip")
                        .arg(Pat::var("u"))
                        .attrs_fields(afields(vec![("dims", af_bind("D1"))])),
                ))
                .attrs_fields(afields(vec![("dims", af_bind("D2"))])),
        ),
        Pat::var("u"),
    )
    .with_law("flip(flip(u, d), d) = u — reversing the same axes twice.")
    .with_cond(json!([
        "and",
        ["attr-ints", "D1"],
        ["attr-eq-attr", "D1", "D2"]
    ]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `tril_idem`: `tril(tril(u, G), G) → tril(u, G)` — the repeated
/// `diagonal` bind pins the same bound.
///
/// Python: `priorart.py` `TRIL_IDEM`, tags `prior_art`+`simplification`.
pub fn tril_idem() -> Law {
    Law::new(
        "tril_idem",
        Pat::Op(
            OpPat::new("tril")
                .arg(Pat::Op(
                    OpPat::new("tril")
                        .arg(Pat::var("u"))
                        .attrs_fields(afields(vec![("diagonal", af_bind("G"))])),
                ))
                .attrs_fields(afields(vec![("diagonal", af_bind("G"))])),
        ),
        Pat::Op(
            OpPat::new("tril")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![("diagonal", af_bind("G"))])),
        ),
    )
    .with_law("tril(tril(u, g), g) = tril(u, g) — masking is idempotent.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `triu_idem`: `triu(triu(u, G), G) → triu(u, G)`.
///
/// Python: `priorart.py` `TRIU_IDEM`, tags `prior_art`+`simplification`.
pub fn triu_idem() -> Law {
    Law::new(
        "triu_idem",
        Pat::Op(
            OpPat::new("triu")
                .arg(Pat::Op(
                    OpPat::new("triu")
                        .arg(Pat::var("u"))
                        .attrs_fields(afields(vec![("diagonal", af_bind("G"))])),
                ))
                .attrs_fields(afields(vec![("diagonal", af_bind("G"))])),
        ),
        Pat::Op(
            OpPat::new("triu")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![("diagonal", af_bind("G"))])),
        ),
    )
    .with_law("triu(triu(u, g), g) = triu(u, g).")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `clamp_idem`: `clamp(clamp(u, LO, HI), LO, HI) → clamp(u, LO, HI)`.
///
/// Python: `priorart.py` `CLAMP_IDEM`, tags `prior_art`+`simplification`.
pub fn clamp_idem() -> Law {
    Law::new(
        "clamp_idem",
        Pat::Op(
            OpPat::new("clamp")
                .arg(Pat::Op(
                    OpPat::new("clamp")
                        .arg(Pat::var("u"))
                        .attrs_fields(afields(vec![
                            ("min", af_bind("LO")),
                            ("max", af_bind("HI")),
                        ])),
                ))
                .attrs_fields(afields(vec![
                    ("min", af_bind("LO")),
                    ("max", af_bind("HI")),
                ])),
        ),
        Pat::Op(
            OpPat::new("clamp")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![
                    ("min", af_bind("LO")),
                    ("max", af_bind("HI")),
                ])),
        ),
    )
    .with_law("clamp(clamp(u,l,h),l,h) = clamp(u,l,h) — same-bound nesting is a no-op.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

// ---------------------------------------------------------------------------
//  prior_art §2 middle — the reduction algebra (std/var/norm2/log-softmax)
//  and the activation decompositions.  `REDUCE_DIM1_RULES` stays
//  Python-side: `_COND_DIM1`'s singleton-tuple arm reads
//  `("attr0", "D")` inside `attr-cmp-dim`'s shape-attr position, and
//  inline derive exprs there are a named `GuardError::Unsupported` —
//  attr-arithmetic the grammar cannot form.  `mean_expand`/`mean_fold`
//  stay too: the reduced-extent divisor is a *product* over the bound
//  dims (no product/fold expr in the dspec algebra) and the fold's
//  check needs a Const-vs-computed-expr equality no cond forms.
// ---------------------------------------------------------------------------

/// `std_sqrt_var`: `std(u, d, c, k) → sqrt(var(u, d, c, k))` — the
/// same dims/correction/keepdim metavariables on both sides.
///
/// Python: `priorart.py` `STD_SQRT_VAR`, tags `prior_art`+`simplification`.
pub fn std_sqrt_var() -> Law {
    let red = |op: &str| {
        Pat::Op(OpPat::new(op).arg(Pat::var("u")).attrs_fields(afields(vec![
            ("dim", af_bind("D")),
            ("correction", af_bind("C")),
            ("keepdim", af_bind("K")),
        ])))
    };
    Law::new(
        "std_sqrt_var",
        red("std"),
        Pat::op("sqrt", vec![red("var")]),
    )
    .with_law(
        "std = sqrt(var) — the same dims/correction/keepdim on both sides (XLA's std decomposition).",
    )
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `var_square_std`: `var(u, d, c, k) → square(std(u, d, c, k))` — the
/// reverse spelling.
///
/// Python: `priorart.py` `VAR_SQUARE_STD`, tags
/// `prior_art`+`expansive`, derivation `std_sqrt_var`.
pub fn var_square_std() -> Law {
    let red = |op: &str| {
        Pat::Op(OpPat::new(op).arg(Pat::var("u")).attrs_fields(afields(vec![
            ("dim", af_bind("D")),
            ("correction", af_bind("C")),
            ("keepdim", af_bind("K")),
        ])))
    };
    Law::new(
        "var_square_std",
        red("var"),
        Pat::op("square", vec![red("std")]),
    )
    .with_law("var = std² — the reverse spelling.")
    .with_derivation("std_sqrt_var")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `std_sqrt_var_bare`: `std(u) → sqrt(var(u))` — the whole-tensor
/// spelling, no attrs.
///
/// Python: `priorart.py` `STD_SQRT_VAR_BARE`, tags
/// `prior_art`+`simplification`.
pub fn std_sqrt_var_bare() -> Law {
    Law::new(
        "std_sqrt_var_bare",
        Pat::op("std", vec![Pat::var("u")]),
        Pat::op("sqrt", vec![Pat::op("var", vec![Pat::var("u")])]),
    )
    .with_law("The whole-tensor spelling — std() = sqrt(var()).")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `var_square_std_bare`: `var(u) → square(std(u))` — the
/// whole-tensor var = std².
///
/// Python: `priorart.py` `VAR_SQUARE_STD_BARE`, tags
/// `prior_art`+`expansive`, derivation `std_sqrt_var_bare`.
pub fn var_square_std_bare() -> Law {
    Law::new(
        "var_square_std_bare",
        Pat::op("var", vec![Pat::var("u")]),
        Pat::op("square", vec![Pat::op("std", vec![Pat::var("u")])]),
    )
    .with_law("The whole-tensor var = std².")
    .with_derivation("std_sqrt_var_bare")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `norm2_expand`: `linalg_vector_norm(u, ord=2, d, k) →
/// sqrt(sum(square(u), d, k))` — the L2 norm's spelled work; `ord`
/// is a literal-2 pin, `dim`/`keepdim` share metavariables across the
/// pair.
///
/// Python: `priorart.py` `NORM2_EXPAND`, tags `prior_art`+`expansive`.
pub fn norm2_expand() -> Law {
    Law::new(
        "norm2_expand",
        Pat::Op(
            OpPat::new("linalg_vector_norm")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![
                    ("ord", af_lit(json!(2))),
                    ("dim", af_bind("D")),
                    ("keepdim", af_bind("K")),
                ])),
        ),
        Pat::op(
            "sqrt",
            vec![Pat::Op(
                OpPat::new("sum")
                    .arg(Pat::op("square", vec![Pat::var("u")]))
                    .attrs_fields(afields(vec![
                        ("dim", af_bind("D")),
                        ("keepdim", af_bind("K")),
                    ])),
            )],
        ),
    )
    .with_law("‖u‖₂ = sqrt(Σ u²) — the L2 norm's spelled work (XLA linalg decomposition).")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `norm2_fold`: `sqrt(sum(square(u), d, k)) → linalg_vector_norm(u,
/// ord=2, d, k)` — the spelled work folds to the L2 kernel.
///
/// Python: `priorart.py` `NORM2_FOLD`, tags `prior_art`+`simplification`,
/// derivation `norm2_expand`.
pub fn norm2_fold() -> Law {
    Law::new(
        "norm2_fold",
        Pat::op(
            "sqrt",
            vec![Pat::Op(
                OpPat::new("sum")
                    .arg(Pat::op("square", vec![Pat::var("u")]))
                    .attrs_fields(afields(vec![
                        ("dim", af_bind("D")),
                        ("keepdim", af_bind("K")),
                    ])),
            )],
        ),
        Pat::Op(
            OpPat::new("linalg_vector_norm")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![
                    ("ord", af_lit(json!(2))),
                    ("dim", af_bind("D")),
                    ("keepdim", af_bind("K")),
                ])),
        ),
    )
    .with_law("sqrt(Σ u²) folds to the L2 kernel.")
    .with_derivation("norm2_expand")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `log_softmax_fold`: `log(softmax(u, d)) → log_softmax(u, d)` — the
/// fused log-softmax fold.
///
/// Python: `priorart.py` `LOG_SOFTMAX_FOLD`, tags
/// `prior_art`+`simplification`.
pub fn log_softmax_fold() -> Law {
    let sm = |op: &str| {
        Pat::Op(
            OpPat::new(op)
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![("dim", af_bind("D"))])),
        )
    };
    Law::new(
        "log_softmax_fold",
        Pat::op("log", vec![sm("softmax")]),
        sm("log_softmax"),
    )
    .with_law(
        "log(softmax(u, d)) = log_softmax(u, d) — the fused log-softmax fold (softcap/logprob heads).",
    )
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `log_softmax_expand`: `exp(log_softmax(u, d)) → softmax(u, d)` —
/// the same kernel pair the other way.
///
/// Python: `priorart.py` `LOG_SOFTMAX_EXPAND`, tags
/// `prior_art`+`expansive`, derivation `log_softmax_fold`.
pub fn log_softmax_expand() -> Law {
    let sm = |op: &str| {
        Pat::Op(
            OpPat::new(op)
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![("dim", af_bind("D"))])),
        )
    };
    Law::new(
        "log_softmax_expand",
        Pat::op("exp", vec![sm("log_softmax")]),
        sm("softmax"),
    )
    .with_law("exp(log_softmax(u, d)) = softmax(u, d) — the same kernel pair the other way.")
    .with_derivation("log_softmax_fold")
    .with_tag("prior_art")
    .with_tag("expansive")
}

// -- activation decompositions (the epilogue spellings) --

/// `sigmoid_expand`: `sigmoid(x) → reciprocal(1 + exp(−x))` — the
/// definition, expanded so pointwise laws see the interior.
///
/// Python: `priorart.py` `SIGMOID_EXPAND`, tags `prior_art`+`expansive`.
pub fn sigmoid_expand() -> Law {
    Law::new(
        "sigmoid_expand",
        Pat::op("sigmoid", vec![Pat::var("x")]),
        Pat::op(
            "reciprocal",
            vec![Pat::op(
                "add",
                vec![
                    Pat::constant(json!(1)),
                    Pat::op("exp", vec![Pat::op("neg", vec![Pat::var("x")])]),
                ],
            )],
        ),
    )
    .with_law("σ(x) = 1/(1+e⁻ˣ) — the definition, expanded so pointwise laws see the interior (Inductor decomp).")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `tanh_expand`: `tanh(x) → 2σ(2x) − 1` — the sigmoid-based
/// decomposition.
///
/// Python: `priorart.py` `TANH_EXPAND`, tags `prior_art`+`expansive`.
pub fn tanh_expand() -> Law {
    Law::new(
        "tanh_expand",
        Pat::op("tanh", vec![Pat::var("x")]),
        Pat::op(
            "sub",
            vec![
                Pat::op(
                    "mul",
                    vec![
                        Pat::constant(json!(2)),
                        Pat::op(
                            "sigmoid",
                            vec![Pat::op("mul", vec![Pat::constant(json!(2)), Pat::var("x")])],
                        ),
                    ],
                ),
                Pat::constant(json!(1)),
            ],
        ),
    )
    .with_law(
        "tanh(x) = 2σ(2x) − 1 — the sigmoid-based decomposition (the TASO/XLA tanh spelling).",
    )
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `gelu_expand`: `gelu(x) → 0.5x(1 + erf(x/√2))` — the exact erf
/// definition (the tanh approximation is a different op).
///
/// Python: `priorart.py` `GELU_EXPAND`, tags `prior_art`+`expansive`.
pub fn gelu_expand() -> Law {
    Law::new(
        "gelu_expand",
        Pat::op("gelu", vec![Pat::var("x")]),
        Pat::op(
            "mul",
            vec![
                Pat::op(
                    "mul",
                    vec![Pat::var("x"), Pat::constant(json!(0.5))],
                ),
                Pat::op(
                    "add",
                    vec![
                        Pat::constant(json!(1)),
                        Pat::op(
                            "erf",
                            vec![Pat::op(
                                "div",
                                vec![
                                    Pat::var("x"),
                                    // √2 — the Python `Const(1.4142135623730951)`
                                    // spells the same f64.
                                    Pat::constant(json!(std::f64::consts::SQRT_2)),
                                ],
                            )],
                        ),
                    ],
                ),
            ],
        ),
    )
    .with_law("gelu(x) = 0.5x(1 + erf(x/√2)) — the exact erf definition (aten's default gelu; the tanh approximation is a DIFFERENT op under this bank and is not claimed).")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `mish_expand`: `mish(x) → x·tanh(softplus(x))` — the Mish
/// definition.
///
/// Python: `priorart.py` `MISH_EXPAND`, tags `prior_art`+`expansive`.
pub fn mish_expand() -> Law {
    Law::new(
        "mish_expand",
        Pat::op("mish", vec![Pat::var("x")]),
        Pat::op(
            "mul",
            vec![
                Pat::var("x"),
                Pat::op("tanh", vec![Pat::op("softplus", vec![Pat::var("x")])]),
            ],
        ),
    )
    .with_law("mish(x) = x·tanh(softplus(x)) — the Mish definition.")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `hardsigmoid_expand`: `hardsigmoid(x) → relu6(x+3)/6` — the
/// piecewise-linear approximation's definition.
///
/// Python: `priorart.py` `HARDSIGMOID_EXPAND`, tags
/// `prior_art`+`expansive`.
pub fn hardsigmoid_expand() -> Law {
    Law::new(
        "hardsigmoid_expand",
        Pat::op("hardsigmoid", vec![Pat::var("x")]),
        Pat::op(
            "div",
            vec![
                Pat::op(
                    "relu6",
                    vec![Pat::op("add", vec![Pat::var("x"), Pat::constant(json!(3))])],
                ),
                Pat::constant(json!(6)),
            ],
        ),
    )
    .with_law("hardsigmoid(x) = relu6(x+3)/6 — the piecewise-linear approximation's definition.")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `hardswish_expand`: `hardswish(x) → x·relu6(x+3)/6`.
///
/// Python: `priorart.py` `HARDSWISH_EXPAND`, tags
/// `prior_art`+`expansive`.
pub fn hardswish_expand() -> Law {
    Law::new(
        "hardswish_expand",
        Pat::op("hardswish", vec![Pat::var("x")]),
        Pat::op(
            "mul",
            vec![
                Pat::var("x"),
                Pat::op(
                    "div",
                    vec![
                        Pat::op(
                            "relu6",
                            vec![Pat::op("add", vec![Pat::var("x"), Pat::constant(json!(3))])],
                        ),
                        Pat::constant(json!(6)),
                    ],
                ),
            ],
        ),
    )
    .with_law("hardswish(x) = x·relu6(x+3)/6.")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `softplus_expand`: `softplus(x) → log1p(eˣ)` at the default
/// beta=1 — the exported bare spelling.
///
/// Python: `priorart.py` `SOFTPLUS_EXPAND`, tags `prior_art`+`expansive`.
pub fn softplus_expand() -> Law {
    Law::new(
        "softplus_expand",
        Pat::op("softplus", vec![Pat::var("x")]),
        Pat::op("log1p", vec![Pat::op("exp", vec![Pat::var("x")])]),
    )
    .with_law("softplus(x) = log1p(eˣ) at the default beta=1 — the exported bare spelling.")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `softplus_fold`: `log1p(eˣ) → softplus(x)`.
///
/// Python: `priorart.py` `SOFTPLUS_FOLD`, tags
/// `prior_art`+`simplification`, derivation `softplus_expand`.
pub fn softplus_fold() -> Law {
    Law::new(
        "softplus_fold",
        Pat::op("log1p", vec![Pat::op("exp", vec![Pat::var("x")])]),
        Pat::op("softplus", vec![Pat::var("x")]),
    )
    .with_law("log1p(eˣ) folds to softplus(x).")
    .with_derivation("softplus_expand")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `elu_expand`: `elu(x) → where(x > 0, x, eˣ−1)` at the default
/// α=1 — the where-spelled piecewise.
///
/// Python: `priorart.py` `ELU_EXPAND`, tags `prior_art`+`expansive`.
pub fn elu_expand() -> Law {
    Law::new(
        "elu_expand",
        Pat::op("elu", vec![Pat::var("x")]),
        Pat::op(
            "where",
            vec![
                Pat::op(
                    "gt",
                    vec![Pat::var("x"), Pat::constant(json!(0))],
                ),
                Pat::var("x"),
                Pat::op("expm1", vec![Pat::var("x")]),
            ],
        ),
    )
    .with_law("elu(x) = x·[x>0] + (eˣ−1)·[x≤0] at the default α=1 — the where-spelled piecewise (XLA elu).")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `leaky_relu_expand`: `leaky_relu(x) → max(x, 0.01x)` at the
/// default slope 0.01.
///
/// Python: `priorart.py` `LEAKY_RELU_EXPAND`, tags
/// `prior_art`+`expansive`.
pub fn leaky_relu_expand() -> Law {
    Law::new(
        "leaky_relu_expand",
        Pat::op("leaky_relu", vec![Pat::var("x")]),
        Pat::op(
            "maximum",
            vec![
                Pat::var("x"),
                Pat::op("mul", vec![Pat::var("x"), Pat::constant(json!(0.01))]),
            ],
        ),
    )
    .with_law("leaky_relu(x) = max(x, 0.01x) at the default slope 0.01.")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `softsign_expand`: `softsign(x) → x/(|x|+1)` — the bounded
/// rational map's definition; the fused ATen kernel's denotation
/// (composite-free — the kernel IS the manual spelling), hence the
/// `fusion` tag like its norm siblings.
///
/// Python: `priorart.py` `SOFTSIGN_EXPAND`, tags `prior_art`+`fusion`.
pub fn softsign_expand() -> Law {
    Law::new(
        "softsign_expand",
        Pat::op("softsign", vec![Pat::var("x")]),
        Pat::op(
            "div",
            vec![
                Pat::var("x"),
                Pat::op(
                    "add",
                    vec![
                        Pat::op("abs", vec![Pat::var("x")]),
                        Pat::constant(json!(1)),
                    ],
                ),
            ],
        ),
    )
    .with_law("softsign(x) = x/(|x|+1) — the bounded rational map's definition, expanded: the fused ATen kernel's denotation (composite-free — the kernel IS the manual spelling).")
    .with_tag("prior_art")
    .with_tag("fusion")
}

/// `relu_fold_max_r`: `max(u, 0) → relu(u)`.
///
/// Python: `priorart.py` `RELU_FOLD_MAX_R`, tags
/// `prior_art`+`simplification`.
pub fn relu_fold_max_r() -> Law {
    Law::new(
        "relu_fold_max_r",
        Pat::op("maximum", vec![Pat::var("u"), Pat::constant(json!(0))]),
        Pat::op("relu", vec![Pat::var("u")]),
    )
    .with_law("max(u, 0) = relu(u).")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `relu_fold_max_l`: `max(0, u) → relu(u)`.
///
/// Python: `priorart.py` `RELU_FOLD_MAX_L`, tags
/// `prior_art`+`simplification`.
pub fn relu_fold_max_l() -> Law {
    Law::new(
        "relu_fold_max_l",
        Pat::op("maximum", vec![Pat::constant(json!(0)), Pat::var("u")]),
        Pat::op("relu", vec![Pat::var("u")]),
    )
    .with_law("max(0, u) = relu(u).")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `relu_fold_clamp`: `clamp_min(u, 0) → relu(u)` — the exported
/// spelling folds.
///
/// Python: `priorart.py` `RELU_FOLD_CLAMP`, tags
/// `prior_art`+`simplification`.
pub fn relu_fold_clamp() -> Law {
    let clamp_min = |u: Pat| {
        Pat::Op(
            OpPat::new("clamp_min")
                .arg(u)
                .attrs_fields(afields(vec![("min", af_lit(json!(0)))])),
        )
    };
    Law::new(
        "relu_fold_clamp",
        clamp_min(Pat::var("u")),
        Pat::op("relu", vec![Pat::var("u")]),
    )
    .with_law("clamp_min(u, 0) = relu(u) — the exported spelling folds.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `relu_expand`: `relu(u) → clamp_min(u, 0)`.
///
/// Python: `priorart.py` `RELU_EXPAND`, tags `prior_art`+`expansive`,
/// derivation `relu_fold_clamp`.
pub fn relu_expand() -> Law {
    Law::new(
        "relu_expand",
        Pat::op("relu", vec![Pat::var("u")]),
        Pat::Op(
            OpPat::new("clamp_min")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![("min", af_lit(json!(0)))])),
        ),
    )
    .with_law("relu(u) = clamp_min(u, 0).")
    .with_derivation("relu_fold_clamp")
    .with_tag("prior_art")
    .with_tag("expansive")
}

/// `relu_idem`: `relu(relu(u)) → relu(u)` — the projection is
/// idempotent.
///
/// Python: `priorart.py` `RELU_IDEM`, tags `prior_art`+`simplification`.
pub fn relu_idem() -> Law {
    Law::new(
        "relu_idem",
        Pat::op("relu", vec![Pat::op("relu", vec![Pat::var("u")])]),
        Pat::op("relu", vec![Pat::var("u")]),
    )
    .with_law("relu(relu(u)) = relu(u) — the projection is idempotent.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `relu6_fold`: `clamp(u, 0, 6) → relu6(u)`.
///
/// Python: `priorart.py` `RELU6_FOLD`, tags `prior_art`+`simplification`.
pub fn relu6_fold() -> Law {
    let clamp = |u: Pat| {
        Pat::Op(OpPat::new("clamp").arg(u).attrs_fields(afields(vec![
            ("min", af_lit(json!(0))),
            ("max", af_lit(json!(6))),
        ])))
    };
    Law::new(
        "relu6_fold",
        clamp(Pat::var("u")),
        Pat::op("relu6", vec![Pat::var("u")]),
    )
    .with_law("clamp(u, 0, 6) = relu6(u).")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `relu6_expand`: `relu6(u) → clamp(u, 0, 6)`.
///
/// Python: `priorart.py` `RELU6_EXPAND`, tags `prior_art`+`expansive`,
/// derivation `relu6_fold`.
pub fn relu6_expand() -> Law {
    Law::new(
        "relu6_expand",
        Pat::op("relu6", vec![Pat::var("u")]),
        Pat::Op(
            OpPat::new("clamp")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![
                    ("min", af_lit(json!(0))),
                    ("max", af_lit(json!(6))),
                ])),
        ),
    )
    .with_law("relu6(u) = clamp(u, 0, 6).")
    .with_derivation("relu6_fold")
    .with_tag("prior_art")
    .with_tag("expansive")
}

// ---------------------------------------------------------------------------
//  prior_art §2 tail — pointwise idempotence/parity, the scalar/Const
//  remainder, and the spelled-unit folds.  Every member is unguarded or
//  rides the ported `shaped`/`shape-eq`/`bcast` vocabulary verbatim.
// ---------------------------------------------------------------------------

/// `abs_idem`: `abs(abs(u)) → abs(u)`.
///
/// Python: `priorart.py` `ABS_IDEM`, tags `prior_art`+`simplification`.
pub fn abs_idem() -> Law {
    Law::new(
        "abs_idem",
        Pat::op("abs", vec![Pat::op("abs", vec![Pat::var("u")])]),
        Pat::op("abs", vec![Pat::var("u")]),
    )
    .with_law("abs(abs(u)) = abs(u).")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `abs_square`: `abs(square(u)) → square(u)` — the square is already
/// nonneg.
///
/// Python: `priorart.py` `ABS_SQUARE`, tags `prior_art`+`simplification`.
pub fn abs_square() -> Law {
    Law::new(
        "abs_square",
        Pat::op("abs", vec![Pat::op("square", vec![Pat::var("u")])]),
        Pat::op("square", vec![Pat::var("u")]),
    )
    .with_law("|u²| = u² — the square is already nonneg.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `square_abs`: `square(abs(u)) → square(u)` — abs is invisible to
/// the square.
///
/// Python: `priorart.py` `SQUARE_ABS`, tags `prior_art`+`simplification`.
pub fn square_abs() -> Law {
    Law::new(
        "square_abs",
        Pat::op("square", vec![Pat::op("abs", vec![Pat::var("u")])]),
        Pat::op("square", vec![Pat::var("u")]),
    )
    .with_law("(|u|)² = u² — abs is invisible to the square.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `abs_neg`: `abs(neg(u)) → abs(u)`.
///
/// Python: `priorart.py` `ABS_NEG`, tags `prior_art`+`simplification`.
pub fn abs_neg() -> Law {
    Law::new(
        "abs_neg",
        Pat::op("abs", vec![Pat::op("neg", vec![Pat::var("u")])]),
        Pat::op("abs", vec![Pat::var("u")]),
    )
    .with_law("|−u| = |u|.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `sqrt_square`: `sqrt(square(u)) → abs(u)` — the canonical spelling
/// (NOT `u` — the negative branch keeps the abs).
///
/// Python: `priorart.py` `SQRT_SQUARE`, tags `prior_art`+`simplification`.
pub fn sqrt_square() -> Law {
    Law::new(
        "sqrt_square",
        Pat::op("sqrt", vec![Pat::op("square", vec![Pat::var("u")])]),
        Pat::op("abs", vec![Pat::var("u")]),
    )
    .with_law("√(u²) = |u| — the canonical spelling (NOT u — the negative branch keeps the abs).")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `rsqrt_square`: `rsqrt(square(u)) → reciprocal(abs(u))` — the
/// rsqrt twin.
///
/// Python: `priorart.py` `RSQRT_SQUARE`, tags `prior_art`+`simplification`.
pub fn rsqrt_square() -> Law {
    Law::new(
        "rsqrt_square",
        Pat::op("rsqrt", vec![Pat::op("square", vec![Pat::var("u")])]),
        Pat::op("reciprocal", vec![Pat::op("abs", vec![Pat::var("u")])]),
    )
    .with_law("1/√(u²) = 1/|u| — the rsqrt twin.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `maximum_idem`: `maximum(u, u) → u`.
///
/// Python: `priorart.py` `MAXIMUM_IDEM`, tags `prior_art`+`simplification`.
pub fn maximum_idem() -> Law {
    Law::new(
        "maximum_idem",
        Pat::op("maximum", vec![Pat::var("u"), Pat::var("u")]),
        Pat::var("u"),
    )
    .with_law("max(u, u) = u.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `minimum_idem`: `minimum(u, u) → u`.
///
/// Python: `priorart.py` `MINIMUM_IDEM`, tags `prior_art`+`simplification`.
pub fn minimum_idem() -> Law {
    Law::new(
        "minimum_idem",
        Pat::op("minimum", vec![Pat::var("u"), Pat::var("u")]),
        Pat::var("u"),
    )
    .with_law("min(u, u) = u.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `tanh_neg`: `tanh(neg(u)) → neg(tanh(u))` — tanh is odd.
///
/// Python: `priorart.py` `TANH_NEG`, tags `prior_art`+`simplification`.
pub fn tanh_neg() -> Law {
    Law::new(
        "tanh_neg",
        Pat::op("tanh", vec![Pat::op("neg", vec![Pat::var("u")])]),
        Pat::op("neg", vec![Pat::op("tanh", vec![Pat::var("u")])]),
    )
    .with_law("tanh is odd: tanh(−u) = −tanh(u).")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `sigmoid_neg`: `sigmoid(neg(u)) → sub(1, sigmoid(u))` — the
/// symmetric complement.
///
/// Python: `priorart.py` `SIGMOID_NEG`, tags `prior_art`+`simplification`.
pub fn sigmoid_neg() -> Law {
    Law::new(
        "sigmoid_neg",
        Pat::op("sigmoid", vec![Pat::op("neg", vec![Pat::var("u")])]),
        Pat::op(
            "sub",
            vec![
                Pat::constant(json!(1)),
                Pat::op("sigmoid", vec![Pat::var("u")]),
            ],
        ),
    )
    .with_law("σ(−u) = 1 − σ(u) — the symmetric complement.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `softsign_neg`: `softsign(neg(u)) → neg(softsign(u))` — odd.
///
/// Python: `priorart.py` `SOFTSIGN_NEG`, tags `prior_art`+`simplification`.
pub fn softsign_neg() -> Law {
    Law::new(
        "softsign_neg",
        Pat::op("softsign", vec![Pat::op("neg", vec![Pat::var("u")])]),
        Pat::op("neg", vec![Pat::op("softsign", vec![Pat::var("u")])]),
    )
    .with_law("softsign is odd.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `erf_neg`: `erf(neg(u)) → neg(erf(u))` — erf is odd.
///
/// Python: `priorart.py` `ERF_NEG`, tags `prior_art`+`simplification`.
pub fn erf_neg() -> Law {
    Law::new(
        "erf_neg",
        Pat::op("erf", vec![Pat::op("neg", vec![Pat::var("u")])]),
        Pat::op("neg", vec![Pat::op("erf", vec![Pat::var("u")])]),
    )
    .with_law("erf is odd.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `sin_neg`: `sin(neg(u)) → neg(sin(u))` — sin is odd.
///
/// Python: `priorart.py` `SIN_NEG`, tags `prior_art`+`simplification`.
pub fn sin_neg() -> Law {
    Law::new(
        "sin_neg",
        Pat::op("sin", vec![Pat::op("neg", vec![Pat::var("u")])]),
        Pat::op("neg", vec![Pat::op("sin", vec![Pat::var("u")])]),
    )
    .with_law("sin is odd.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `cos_neg`: `cos(neg(u)) → cos(u)` — cos is even.
///
/// Python: `priorart.py` `COS_NEG`, tags `prior_art`+`simplification`.
pub fn cos_neg() -> Law {
    Law::new(
        "cos_neg",
        Pat::op("cos", vec![Pat::op("neg", vec![Pat::var("u")])]),
        Pat::op("cos", vec![Pat::var("u")]),
    )
    .with_law("cos is even: cos(−u) = cos(u).")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `exp_neg_recip`: `exp(neg(u)) → reciprocal(exp(u))`.
///
/// Python: `priorart.py` `EXP_NEG_RECIP`, tags
/// `prior_art`+`simplification`.
pub fn exp_neg_recip() -> Law {
    Law::new(
        "exp_neg_recip",
        Pat::op("exp", vec![Pat::op("neg", vec![Pat::var("u")])]),
        Pat::op("reciprocal", vec![Pat::op("exp", vec![Pat::var("u")])]),
    )
    .with_law("e⁻ᵘ = 1/eᵘ.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `log_exp`: `log(exp(u)) → u` — eᵘ > 0 always, so the inverse is
/// total.
///
/// Python: `priorart.py` `LOG_EXP`, tags `prior_art`+`simplification`.
pub fn log_exp() -> Law {
    Law::new(
        "log_exp",
        Pat::op("log", vec![Pat::op("exp", vec![Pat::var("u")])]),
        Pat::var("u"),
    )
    .with_law("log(eᵘ) = u — eᵘ > 0 always, so the inverse is total.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `log_recip`: `log(reciprocal(u)) → neg(log(u))`.
///
/// Python: `priorart.py` `LOG_RECIP`, tags `prior_art`+`simplification`.
pub fn log_recip() -> Law {
    Law::new(
        "log_recip",
        Pat::op("log", vec![Pat::op("reciprocal", vec![Pat::var("u")])]),
        Pat::op("neg", vec![Pat::op("log", vec![Pat::var("u")])]),
    )
    .with_law("log(1/u) = −log(u).")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `recip_recip`: `reciprocal(reciprocal(u)) → u` — reciprocal
/// involution.
///
/// Python: `priorart.py` `RECIP_RECIP`, tags
/// `prior_art`+`simplification`.
pub fn recip_recip() -> Law {
    Law::new(
        "recip_recip",
        Pat::op(
            "reciprocal",
            vec![Pat::op("reciprocal", vec![Pat::var("u")])],
        ),
        Pat::var("u"),
    )
    .with_law("1/(1/u) = u — reciprocal involution.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `add_neg_to_sub`: `add(a, neg(b)) → sub(a, b)` — the refold
/// (`sub_to_add`'s inverse).
///
/// Python: `priorart.py` `ADD_NEG_TO_SUB`, tags
/// `prior_art`+`simplification`.
pub fn add_neg_to_sub() -> Law {
    Law::new(
        "add_neg_to_sub",
        Pat::op(
            "add",
            vec![Pat::var("a"), Pat::op("neg", vec![Pat::var("b")])],
        ),
        Pat::op("sub", vec![Pat::var("a"), Pat::var("b")]),
    )
    .with_law("a + (−b) = a − b — the refold (sub_to_add's inverse).")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `sub_neg_to_add`: `sub(a, neg(b)) → add(a, b)`.
///
/// Python: `priorart.py` `SUB_NEG_TO_ADD`, tags
/// `prior_art`+`simplification`.
pub fn sub_neg_to_add() -> Law {
    Law::new(
        "sub_neg_to_add",
        Pat::op(
            "sub",
            vec![Pat::var("a"), Pat::op("neg", vec![Pat::var("b")])],
        ),
        Pat::op("add", vec![Pat::var("a"), Pat::var("b")]),
    )
    .with_law("a − (−b) = a + b.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `neg_sub`: `neg(sub(a, b)) → sub(b, a)`.
///
/// Python: `priorart.py` `NEG_SUB`, tags `prior_art`+`simplification`.
pub fn neg_sub() -> Law {
    Law::new(
        "neg_sub",
        Pat::op(
            "neg",
            vec![Pat::op("sub", vec![Pat::var("a"), Pat::var("b")])],
        ),
        Pat::op("sub", vec![Pat::var("b"), Pat::var("a")]),
    )
    .with_law("−(a−b) = b−a.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `sub_zero_to_neg`: `sub(0, u) → neg(u)`.
///
/// Python: `priorart.py` `SUB_ZERO_TO_NEG`, tags
/// `prior_art`+`simplification`.
pub fn sub_zero_to_neg() -> Law {
    Law::new(
        "sub_zero_to_neg",
        Pat::op("sub", vec![Pat::constant(json!(0)), Pat::var("u")]),
        Pat::op("neg", vec![Pat::var("u")]),
    )
    .with_law("0 − u = −u.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `id_sub`: `sub(u, 0) → u`.
///
/// Python: `priorart.py` `ID_SUB`, tags `prior_art`+`simplification`.
pub fn id_sub() -> Law {
    Law::new(
        "id_sub",
        Pat::op("sub", vec![Pat::var("u"), Pat::constant(json!(0))]),
        Pat::var("u"),
    )
    .with_law("u − 0 = u.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `id_div`: `div(u, 1) → u`.
///
/// Python: `priorart.py` `ID_DIV`, tags `prior_art`+`simplification`.
pub fn id_div() -> Law {
    Law::new(
        "id_div",
        Pat::op("div", vec![Pat::var("u"), Pat::constant(json!(1))]),
        Pat::var("u"),
    )
    .with_law("u/1 = u.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `add_self`: `add(u, u) → mul(u, 2)` — the strength reduction.
///
/// Python: `priorart.py` `ADD_SELF`, tags `prior_art`+`simplification`.
pub fn add_self() -> Law {
    Law::new(
        "add_self",
        Pat::op("add", vec![Pat::var("u"), Pat::var("u")]),
        Pat::op("mul", vec![Pat::var("u"), Pat::constant(json!(2))]),
    )
    .with_law("u+u = 2u — the strength reduction.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `mul2_to_add`: `mul(u, 2) → add(u, u)` — the sharing direction,
/// derivation `add_self`.
///
/// Python: `priorart.py` `MUL2_TO_ADD`, tags
/// `prior_art`+`simplification`.
pub fn mul2_to_add() -> Law {
    Law::new(
        "mul2_to_add",
        Pat::op("mul", vec![Pat::var("u"), Pat::constant(json!(2))]),
        Pat::op("add", vec![Pat::var("u"), Pat::var("u")]),
    )
    .with_law("2u = u+u — the sharing direction.")
    .with_derivation("add_self")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `mul_neg1`: `mul(u, −1) → neg(u)`.
///
/// Python: `priorart.py` `MUL_NEG1`, tags `prior_art`+`simplification`.
pub fn mul_neg1() -> Law {
    Law::new(
        "mul_neg1",
        Pat::op("mul", vec![Pat::var("u"), Pat::constant(json!(-1))]),
        Pat::op("neg", vec![Pat::var("u")]),
    )
    .with_law("u·(−1) = −u.")
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// The `_zeros_like` family: `op(u, unit_like(v)) → u` guarded by the
/// grid-equality pair — `u` and `v` must broadcast AND the result
/// shape must BE `u`'s (a smaller `u` broadcasting up produces the
/// grid shape, not `u`).
///
/// Python: `priorart.py` `_zeros_like`, tags
/// `prior_art`+`simplification`.
fn unit_like_rule(op: &str, unit: &str) -> Law {
    let name = format!("{op}_{unit}like_r");
    let law = format!(
        "{op}(u, {unit}_like(v)) = u — the spelled additive/multiplicative unit (grid-equality guard keeps it honest)."
    );
    Law::new(
        name,
        Pat::op(
            op,
            vec![
                Pat::var("u"),
                Pat::op(format!("{unit}_like"), vec![Pat::var("v")]),
            ],
        ),
        Pat::var("u"),
    )
    .with_law(law)
    .with_cond(json!([
        "and",
        ["shaped", ["bcast", "u", "v"]],
        ["shape-eq", "u", ["bcast", "u", "v"]]
    ]))
    .with_tag("prior_art")
    .with_tag("simplification")
}

/// `add_zeroslike_r`: `add(u, zeros_like(v)) → u`.
///
/// Python: `priorart.py` `ADD_ZEROS_R`, tags
/// `prior_art`+`simplification`.
pub fn add_zeroslike_r() -> Law {
    unit_like_rule("add", "zeros")
}

/// `sub_zeroslike_r`: `sub(u, zeros_like(v)) → u`.
///
/// Python: `priorart.py` `SUB_ZEROS_R`, tags
/// `prior_art`+`simplification`.
pub fn sub_zeroslike_r() -> Law {
    unit_like_rule("sub", "zeros")
}

/// `mul_oneslike_r`: `mul(u, ones_like(v)) → u`.
///
/// Python: `priorart.py` `MUL_ONES_R`, tags
/// `prior_art`+`simplification`.
pub fn mul_oneslike_r() -> Law {
    unit_like_rule("mul", "ones")
}

/// `div_oneslike_r`: `div(u, ones_like(v)) → u`.
///
/// Python: `priorart.py` `DIV_ONES_R`, tags
/// `prior_art`+`simplification`.
pub fn div_oneslike_r() -> Law {
    unit_like_rule("div", "ones")
}

// ---------------------------------------------------------------------------
//  Guarded tensor laws — the `cond`/`dspec` members of
//  `SIMPLIFICATION_RULES`/`CATEGORICAL_RULES` whose guards fit the
//  ported [`crate::guard`] vocabulary (attr predicates, the scalar
//  expr algebra, and the [`crate::shape`] predicate/spec table).
//  Attr metavariables ride [`OpPat::attrs_fields`] — the Python
//  `Op.make(…, dim="D")` per-key spelling — which `Law::compile`
//  interprets as exact-keyset match + per-field binding.
// ---------------------------------------------------------------------------

/// `select_mul`: `sel(u,D,I) ⊙ sel(v,D,I) → sel(u ⊙ v, D, I)` —
/// elementwise mul commutes with select.
///
/// Python: `tensor.py` `SELECT_MUL`, cond
/// `("and", ("dim-eq-attr","u","D","v","D"), ("shaped",("bcast","u","v")))`,
/// tags `simplification`.
pub fn select_mul() -> Law {
    let sel = |t: Pat| {
        Pat::Op(OpPat::new("select").arg(t).attrs_fields(afields(vec![
            ("dim", af_bind("D")),
            ("index", af_bind("I")),
        ])))
    };
    Law::new(
        "select_mul",
        Pat::op("mul", vec![sel(Pat::var("u")), sel(Pat::var("v"))]),
        sel(Pat::op("mul", vec![Pat::var("u"), Pat::var("v")])),
    )
    .with_law(
        "mul commutes with select: sel(u) ⊙ sel(v) = sel(u ⊙ v) — the elementwise product of two slices is the slice of the product (naturality of the elementwise action over the select view).  Removes one dispatched op per site.",
    )
    .with_cond(json!([
        "and",
        ["dim-eq-attr", "u", "D", "v", "D"],
        ["shaped", ["bcast", "u", "v"]]
    ]))
    .with_tag("simplification")
}

/// `transpose_noop`: `transpose(u, D0, D1) → u` when the swap axes
/// coincide or both swapped extents are 1.
///
/// Python: `tensor.py` `TRANSPOSE_NOOP`, cond `("axes-noop","u","D0","D1")`,
/// tags `simplification`.
pub fn transpose_noop() -> Law {
    Law::new(
        "transpose_noop",
        Pat::Op(
            OpPat::new("transpose")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![
                    ("dim0", af_bind("D0")),
                    ("dim1", af_bind("D1")),
                ])),
        ),
        Pat::var("u"),
    )
    .with_law(
        "A transpose whose swap axes coincide (d0 == d1 mod rank) or whose swapped extents are both 1 is a semantic no-op — the stride permutation maps each position to itself; strip the view.",
    )
    .with_cond(json!(["axes-noop", "u", "D0", "D1"]))
    .with_tag("simplification")
}

/// `chunk_single`: `chunk(u, chunks=1, …) → u`.
///
/// Python: `tensor.py` `CHUNK_SINGLE`, cond `("attr-eq","CK",1)`,
/// tags `simplification`.
pub fn chunk_single() -> Law {
    Law::new(
        "chunk_single",
        Pat::Op(
            OpPat::new("chunk")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![
                    ("chunks", af_bind("CK")),
                    ("dim", af_bind("CD")),
                    ("index", af_bind("CI")),
                ])),
        ),
        Pat::var("u"),
    )
    .with_law(
        "chunk(u, chunks=1, dim, index) is u — a one-section partition returns the whole tensor unchanged.",
    )
    .with_cond(json!(["attr-eq", "CK", 1]))
    .with_tag("simplification")
}

/// The broadcast-pad strip region for `op(unsqueeze(u, K), v)` —
/// Python `_COND_UNSQ_PAD_L`.
fn cond_unsq_pad_l() -> serde_json::Value {
    json!([
        "and",
        ["bcast-eq", ["unsq-out", "u", "UD"], "v", "u", "v"],
        ["bcast-into", "u", ["unsq-out", "u", "UD"]]
    ])
}

/// The right-operand twin — Python `_COND_UNSQ_PAD_R`.
fn cond_unsq_pad_r() -> serde_json::Value {
    json!([
        "and",
        ["bcast-eq", ["unsq-out", "v", "UD"], "u", "v", "u"],
        ["bcast-into", "v", ["unsq-out", "v", "UD"]]
    ])
}

/// `unsqueeze(u, dim=UD)` — the view op with its axis attr bound.
fn unsqueeze_u(var: &str) -> Pat {
    Pat::Op(
        OpPat::new("unsqueeze")
            .arg(Pat::var(var))
            .attrs_fields(afields(vec![("dim", af_bind("UD"))])),
    )
}

/// `mul_unsq_pad_l`: `unsqueeze(u, d) · v → u · v` under the
/// broadcast-pad grid guard.
///
/// Python: `tensor.py` `MUL_UNSQ_PAD_L`, tags `simplification`.
pub fn mul_unsq_pad_l() -> Law {
    Law::new(
        "mul_unsq_pad_l",
        Pat::op("mul", vec![unsqueeze_u("u"), Pat::var("v")]),
        Pat::op("mul", vec![Pat::var("u"), Pat::var("v")]),
    )
    .with_law(
        "unsqueeze(u, d) · v = u · v when the inserted axis is a broadcast pad and the broadcast grids coincide — the lifted operand reads the same positions, so the pad view is dead.  Drops one dispatched op per site (the exported ALiBi spelling carries it).",
    )
    .with_cond(cond_unsq_pad_l())
    .with_tag("simplification")
}

/// `mul_unsq_pad_r`: `u · unsqueeze(v, d) → u · v` — right-operand
/// twin of [`mul_unsq_pad_l`].
///
/// Python: `tensor.py` `MUL_UNSQ_PAD_R`, tags `simplification`.
pub fn mul_unsq_pad_r() -> Law {
    Law::new(
        "mul_unsq_pad_r",
        Pat::op("mul", vec![Pat::var("u"), unsqueeze_u("v")]),
        Pat::op("mul", vec![Pat::var("u"), Pat::var("v")]),
    )
    .with_law(
        "Right-operand twin of mul_unsq_pad_l: u · unsqueeze(v, d) = u · v under the same broadcast-pad grid guard.",
    )
    .with_cond(cond_unsq_pad_r())
    .with_tag("simplification")
}

/// `sub_unsq_pad_l`: `unsqueeze(u, d) − v → u − v` under the same
/// guard (the guard, not commutativity, carries it).
///
/// Python: `tensor.py` `SUB_UNSQ_PAD_L`, tags `simplification`.
pub fn sub_unsq_pad_l() -> Law {
    Law::new(
        "sub_unsq_pad_l",
        Pat::op("sub", vec![unsqueeze_u("u"), Pat::var("v")]),
        Pat::op("sub", vec![Pat::var("u"), Pat::var("v")]),
    )
    .with_law(
        "unsqueeze(u, d) - v = u - v under the broadcast-pad guard — same strip for the asymmetric operand pair (the guard, not the op's commutativity, is what carries it).",
    )
    .with_cond(cond_unsq_pad_l())
    .with_tag("simplification")
}

/// `add_unsq_pad_r`: `u + unsqueeze(v, d) → u + v` — the add sibling
/// of [`mul_unsq_pad_r`] (the intake-minted
/// `mixed:add_unsqueeze_r_id` candidate).
///
/// Python: `tensor.py` `ADD_UNSQ_PAD_R`, tags `simplification`.
pub fn add_unsq_pad_r() -> Law {
    Law::new(
        "add_unsq_pad_r",
        Pat::op("add", vec![Pat::var("u"), unsqueeze_u("v")]),
        Pat::op("add", vec![Pat::var("u"), Pat::var("v")]),
    )
    .with_law(
        "u + unsqueeze(v, d) = u + v under the broadcast-pad grid guard — the add sibling of mul_unsq_pad_r: the lifted operand's pad view is dead when the broadcast grids coincide.  Minted by auto-cond from the intake census's ``mixed:add_unsqueeze_r_id`` candidate and admitted off a strictly-real paying site (``RelativePositionBias`` — see docs/results_realcorpus.md).",
    )
    .with_cond(cond_unsq_pad_r())
    .with_tag("simplification")
}

/// `mul_reshape_inert_l`: `reshape(u, S) · v → u · v` when the
/// reshape is broadcast-inert (equal broadcast grids).
///
/// Python: `tensor.py` `MUL_RESHAPE_INERT_L`, cond
/// `("bcast-eq",("reshape-out","u","RS"),"v","u","v")`,
/// tags `simplification`.
pub fn mul_reshape_inert_l() -> Law {
    Law::new(
        "mul_reshape_inert_l",
        Pat::op(
            "mul",
            vec![
                Pat::Op(
                    OpPat::new("reshape")
                        .arg(Pat::var("u"))
                        .attrs_fields(afields(vec![("shape", af_bind("RS"))])),
                ),
                Pat::var("v"),
            ],
        ),
        Pat::op("mul", vec![Pat::var("u"), Pat::var("v")]),
    )
    .with_law(
        "reshape(u, S) · v = u · v when the reshape is broadcast-inert — equal broadcast grids force the reshape to a 1-axis insert/remove (a permuting reshape changes an extent and the grids disagree).  Grid equality is the minted single-clause guard.",
    )
    .with_cond(json!([
        "bcast-eq",
        ["reshape-out", "u", "RS"],
        "v",
        "u",
        "v"
    ]))
    .with_tag("simplification")
}

/// `softmax_fold`: `exp(u) / Σ exp(u) → softmax(u)` — the
/// manual-softmax fold; `keepdim`/`dim` guards plus the `attr0`
/// dspec unwrapping `sum`'s `dim` tuple into `softmax`'s scalar dim.
///
/// Python: `tensor.py` `SOFTMAX_FOLD`, tags `simplification`.
pub fn softmax_fold() -> Law {
    Law::new(
        "softmax_fold",
        Pat::op(
            "div",
            vec![
                Pat::op("exp", vec![Pat::var("u")]),
                Pat::Op(
                    OpPat::new("sum")
                        .arg(Pat::op("exp", vec![Pat::var("u")]))
                        .attrs_fields(afields(vec![
                            ("dim", af_bind("RD")),
                            ("keepdim", af_bind("RK")),
                        ])),
                ),
            ],
        ),
        Pat::Op(
            OpPat::new("softmax")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![("dim", af_bind("SD"))])),
        ),
    )
    .with_law(
        "exp(u) / Σ exp(u) = softmax(u): the manual normalization fold — a composed-then-reduced chain IS the kernel's definition.  Folds div+exp+sum to one dispatched op.",
    )
    .with_cond(json!([
        "and",
        ["attr-is", "RK", true],
        [
            "or",
            ["attr-type", "RD", "int"],
            [
                "and",
                ["attr-type", "RD", "tuple"],
                ["attr-len", "RD", "==", 1]
            ]
        ]
    ]))
    .with_dspec(json!({"SD": ["attr0", "RD"]}))
    .with_tag("simplification")
}

/// `glu_fold`: `chunk-half · σ(chunk-half) → glu(u)` — the
/// manual-GLU fold; the parity guard is `dim-mod` on the attr-named
/// split axis.
///
/// Python: `tensor.py` `GLU_FOLD`, tags `simplification`.
pub fn glu_fold() -> Law {
    let chunk = |index: i64| {
        Pat::Op(
            OpPat::new("chunk")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![
                    ("chunks", af_lit(json!(2))),
                    ("dim", af_bind("D")),
                    ("index", af_lit(json!(index))),
                ])),
        )
    };
    Law::new(
        "glu_fold",
        Pat::op(
            "mul",
            vec![chunk(0), Pat::op("sigmoid", vec![chunk(1)])],
        ),
        Pat::Op(
            OpPat::new("glu")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![("dim", af_bind("D"))])),
        ),
    )
    .with_law(
        "chunk-half · σ(chunk-half) IS glu(u): the manual-GLU fold — a ⊗ σ(b) over the two equal halves of u IS the kernel's definition.  Folds mul+sigmoid+2 chunks to one dispatched op.",
    )
    .with_cond(json!([
        "and",
        ["rank", "u", ">=", 1],
        ["dim-mod", "u", "D", 2, 0]
    ]))
    .with_tag("simplification")
}

/// `matmul_div_distribute`: `(a/c) @ v → (a @ v)/c` for scalar `c`.
///
/// Python: `tensor.py` `MATMUL_DIV_DISTRIBUTE`, cond `("scalar","c")`,
/// tags `simplification`.
pub fn matmul_div_distribute() -> Law {
    Law::new(
        "matmul_div_distribute",
        Pat::op(
            "matmul",
            vec![Pat::op("div", vec![Pat::var("a"), Pat::var("c")]), Pat::var("v")],
        ),
        Pat::op(
            "div",
            vec![
                Pat::op("matmul", vec![Pat::var("a"), Pat::var("v")]),
                Pat::var("c"),
            ],
        ),
    )
    .with_law(
        "(a/c)@v = (a@v)/c when c is scalar — linear readout of a scalar-denominated operand; the score scale exits the contraction.",
    )
    .with_cond(json!(["scalar", "c"]))
    .with_tag("simplification")
}

/// `div_add_factor`: `x/c + y/c → (x+y)/c` for scalar `c`.
///
/// Python: `tensor.py` `DIV_ADD_FACTOR`, cond `("scalar","c")`,
/// tags `simplification`.
pub fn div_add_factor() -> Law {
    Law::new(
        "div_add_factor",
        Pat::op(
            "add",
            vec![
                Pat::op("div", vec![Pat::var("x"), Pat::var("c")]),
                Pat::op("div", vec![Pat::var("y"), Pat::var("c")]),
            ],
        ),
        Pat::op(
            "div",
            vec![Pat::op("add", vec![Pat::var("x"), Pat::var("y")]), Pat::var("c")],
        ),
    )
    .with_law(
        "x/c + y/c = (x+y)/c — shared-denominator factoring over the additive monoid; the score scale exits the block sum the way it exited the contraction.",
    )
    .with_cond(json!(["scalar", "c"]))
    .with_tag("simplification")
}

// ---------------------------------------------------------------------------
//  Categorical guarded laws — bilinearity guards, the biased/scale
//  naturality family, the product-structure fuses, and the sdpa fold.
// ---------------------------------------------------------------------------

/// Python `_COND_MM_ADDENDS` — the addend-alignment guard shared by
/// `distribute_matmul_over_add`/`factor_matmul`.
fn cond_mm_addends() -> serde_json::Value {
    json!([
        "or",
        [
            "and",
            ["rank-eq", "a", "b"],
            ["rank", "a", ">=", 1],
            ["shape-eq", "a", "b"]
        ],
        [
            "and",
            ["rank", "a", ">=", 2],
            ["rank", "b", ">=", 2],
            ["dim-eq", "a", -2, "b", -2]
        ]
    ])
}

/// Python `_COND_MM_WEIGHTS` — the `matmul` weight-pair guard.
fn cond_mm_weights() -> serde_json::Value {
    json!([
        "or",
        [
            "and",
            ["rank-eq", "W", "W2"],
            ["rank", "W", ">=", 1],
            ["shape-eq", "W", "W2"]
        ],
        [
            "and",
            ["rank", "W", ">=", 2],
            ["rank", "W2", ">=", 2],
            ["dim-eq", "W", -2, "W2", -2]
        ]
    ])
}

/// Python `_COND_MM_WEIGHTS_LINEAR` — the `linear` spelling of the
/// weight guard (contraction axis is `W[-1]`, the transpose's -2).
fn cond_mm_weights_linear() -> serde_json::Value {
    json!([
        "or",
        [
            "and",
            ["rank-eq", "W", "W2"],
            ["rank", "W", ">=", 1],
            ["shape-eq", "W", "W2"]
        ],
        [
            "and",
            ["rank", "W", ">=", 2],
            ["rank", "W2", ">=", 2],
            ["dim-eq", "W", -1, "W2", -1]
        ]
    ])
}

/// `distribute_matmul_over_add`: `matmul(W, a+b) → matmul(W,a) +
/// matmul(W,b)` — left-slot bilinearity.
///
/// Python: `tensor.py` `DISTRIBUTE_MUL`, cond `_COND_MM_ADDENDS`,
/// tags `categorical`+`expansive`.
pub fn distribute_matmul_over_add() -> Law {
    Law::new(
        "distribute_matmul_over_add",
        Pat::op(
            "matmul",
            vec![
                Pat::var("W"),
                Pat::op("add", vec![Pat::var("a"), Pat::var("b")]),
            ],
        ),
        Pat::op(
            "add",
            vec![
                Pat::op("matmul", vec![Pat::var("W"), Pat::var("a")]),
                Pat::op("matmul", vec![Pat::var("W"), Pat::var("b")]),
            ],
        ),
    )
    .with_law("Distributivity of linear maps over addition (bilinearity).")
    .with_cond(cond_mm_addends())
    .with_tag("categorical")
    .with_tag("expansive")
}

/// `factor_matmul`: `matmul(W,a) + matmul(W,b) → matmul(W, a+b)` —
/// factoring a shared left weight.
///
/// Python: `tensor.py` `FACTOR_MUL`, cond `_COND_MM_ADDENDS`,
/// tags `categorical`+`expansive`, derivation
/// `distribute_matmul_over_add`.
pub fn factor_matmul() -> Law {
    Law::new(
        "factor_matmul",
        Pat::op(
            "add",
            vec![
                Pat::op("matmul", vec![Pat::var("W"), Pat::var("a")]),
                Pat::op("matmul", vec![Pat::var("W"), Pat::var("b")]),
            ],
        ),
        Pat::op(
            "matmul",
            vec![
                Pat::var("W"),
                Pat::op("add", vec![Pat::var("a"), Pat::var("b")]),
            ],
        ),
    )
    .with_law("Factoring common linear maps (reverse distributivity).")
    .with_cond(cond_mm_addends())
    .with_derivation("distribute_matmul_over_add")
    .with_tag("categorical")
    .with_tag("expansive")
}

/// `weight_factor_matmul`: `x@W + x@W2 → x@(W+W2)` — the weight-merge
/// rule (REDUNDANT alpha-duplicate of [`factor_matmul`], annotated
/// not deleted, as in Python).
///
/// Python: `tensor.py` `WEIGHT_FACTOR`, cond `_COND_MM_WEIGHTS`,
/// tags `categorical`+`expansive`+`redundant`, derivation
/// `distribute_matmul_over_add`.
pub fn weight_factor_matmul() -> Law {
    Law::new(
        "weight_factor_matmul",
        Pat::op(
            "add",
            vec![
                Pat::op("matmul", vec![Pat::var("x"), Pat::var("W")]),
                Pat::op("matmul", vec![Pat::var("x"), Pat::var("W2")]),
            ],
        ),
        Pat::op(
            "matmul",
            vec![
                Pat::var("x"),
                Pat::op("add", vec![Pat::var("W"), Pat::var("W2")]),
            ],
        ),
    )
    .with_law("Merge shared-input projections: x@W1 + x@W2 = x@(W1+W2).")
    .with_cond(cond_mm_weights())
    .with_derivation("distribute_matmul_over_add")
    .with_tag("categorical")
    .with_tag("expansive")
    .with_tag("redundant")
}

/// `weight_distribute_matmul`: `x@(W+W2) → x@W + x@W2` — the reverse
/// weight merge.
///
/// Python: `tensor.py` `WEIGHT_DISTRIBUTE`, cond `_COND_MM_WEIGHTS`,
/// tags `categorical`+`expansive`+`redundant`, derivation
/// `distribute_matmul_over_add`.
pub fn weight_distribute_matmul() -> Law {
    Law::new(
        "weight_distribute_matmul",
        Pat::op(
            "matmul",
            vec![
                Pat::var("x"),
                Pat::op("add", vec![Pat::var("W"), Pat::var("W2")]),
            ],
        ),
        Pat::op(
            "add",
            vec![
                Pat::op("matmul", vec![Pat::var("x"), Pat::var("W")]),
                Pat::op("matmul", vec![Pat::var("x"), Pat::var("W2")]),
            ],
        ),
    )
    .with_law("Reverse weight merge (lets eqsat weigh fused vs split forms).")
    .with_cond(cond_mm_weights())
    .with_derivation("distribute_matmul_over_add")
    .with_tag("categorical")
    .with_tag("expansive")
    .with_tag("redundant")
}

/// `weight_factor_linear`: `linear(x,W) + linear(x,W2) →
/// linear(x, W+W2)` — the `nn.Linear` weight merge.
///
/// Python: `tensor.py` `WEIGHT_FACTOR_LINEAR`, cond
/// `_COND_MM_WEIGHTS_LINEAR`, tags `categorical`+`expansive`,
/// derivation `weight_distribute_linear`.
pub fn weight_factor_linear() -> Law {
    Law::new(
        "weight_factor_linear",
        Pat::op(
            "add",
            vec![
                Pat::op("linear", vec![Pat::var("x"), Pat::var("W")]),
                Pat::op("linear", vec![Pat::var("x"), Pat::var("W2")]),
            ],
        ),
        Pat::op(
            "linear",
            vec![Pat::var("x"), Pat::op("add", vec![Pat::var("W"), Pat::var("W2")])],
        ),
    )
    .with_law(
        "Merge shared-input nn.Linears: linear(x,W1)+linear(x,W2) = linear(x, W1+W2)  (transpose distributes over +).",
    )
    .with_cond(cond_mm_weights_linear())
    .with_derivation("weight_distribute_linear")
    .with_tag("categorical")
    .with_tag("expansive")
}

/// `weight_distribute_linear`: `linear(x, W+W2) → linear(x,W) +
/// linear(x,W2)` — expand a merged `nn.Linear`.
///
/// Python: `tensor.py` `WEIGHT_DISTRIBUTE_LINEAR`, cond
/// `_COND_MM_WEIGHTS_LINEAR`, tags `categorical`+`expansive`.
pub fn weight_distribute_linear() -> Law {
    Law::new(
        "weight_distribute_linear",
        Pat::op(
            "linear",
            vec![
                Pat::var("x"),
                Pat::op("add", vec![Pat::var("W"), Pat::var("W2")]),
            ],
        ),
        Pat::op(
            "add",
            vec![
                Pat::op("linear", vec![Pat::var("x"), Pat::var("W")]),
                Pat::op("linear", vec![Pat::var("x"), Pat::var("W2")]),
            ],
        ),
    )
    .with_law("Expand a merged nn.Linear so eqsat can compare both forms.")
    .with_cond(cond_mm_weights_linear())
    .with_tag("categorical")
    .with_tag("expansive")
}

/// Python `_COND_BIAS_COMPOSE` — the affine-compose shape guard.
fn cond_bias_compose() -> serde_json::Value {
    json!([
        "and",
        ["rank", "A", "==", 2],
        ["rank", "B", "==", 2],
        ["dim-eq", "B", 1, "A", 0],
        ["rank", "b1", "==", 1],
        ["dim-eq", "b1", 0, "A", 0],
        [
            "or",
            ["scalar", "b2"],
            ["and", ["rank", "b2", "==", 1], ["dim-eq", "b2", 0, "B", 0]]
        ],
        ["rank", "x", ">=", 1],
        ["dim-eq", "x", -1, "A", 1]
    ])
}

/// `assoc_linear_bias`: `linear(linear(x,A,b1),B,b2) →
/// add(linear(x, B@A, B·b1), b2)` — affine-map composition.
///
/// Python: `tensor.py` `ASSOC_LINEAR_BIAS`, cond
/// `_COND_BIAS_COMPOSE`, tags `categorical`+`expansive`.
pub fn assoc_linear_bias() -> Law {
    Law::new(
        "assoc_linear_bias",
        Pat::op(
            "linear",
            vec![
                Pat::op("linear", vec![Pat::var("x"), Pat::var("A"), Pat::var("b1")]),
                Pat::var("B"),
                Pat::var("b2"),
            ],
        ),
        Pat::op(
            "add",
            vec![
                Pat::op(
                    "linear",
                    vec![
                        Pat::var("x"),
                        Pat::op("matmul", vec![Pat::var("B"), Pat::var("A")]),
                        Pat::op("matmul", vec![Pat::var("B"), Pat::var("b1")]),
                    ],
                ),
                Pat::var("b2"),
            ],
        ),
    )
    .with_law(
        "Affine-map composition: (B,b2)∘(A,b1) = (BA, B·b1 + b2).  Fused weight is B @ A (same transpose flip as assoc_linear); the fused bias is spelled linear(x, BA, B·b1) + b2 so both B-products fold at compile time — one GEMM plus one broadcast add at runtime.",
    )
    .with_cond(cond_bias_compose())
    .with_tag("categorical")
    .with_tag("expansive")
}

/// `assoc_linear_bias_rev`: reverse affine composition.
///
/// Python: `tensor.py` `ASSOC_LINEAR_BIAS_REV`, cond
/// `_COND_BIAS_COMPOSE`, tags `categorical`+`expansive`, derivation
/// `assoc_linear_bias`.
pub fn assoc_linear_bias_rev() -> Law {
    Law::new(
        "assoc_linear_bias_rev",
        Pat::op(
            "add",
            vec![
                Pat::op(
                    "linear",
                    vec![
                        Pat::var("x"),
                        Pat::op("matmul", vec![Pat::var("B"), Pat::var("A")]),
                        Pat::op("matmul", vec![Pat::var("B"), Pat::var("b1")]),
                    ],
                ),
                Pat::var("b2"),
            ],
        ),
        Pat::op(
            "linear",
            vec![
                Pat::op("linear", vec![Pat::var("x"), Pat::var("A"), Pat::var("b1")]),
                Pat::var("B"),
                Pat::var("b2"),
            ],
        ),
    )
    .with_law("Reverse affine composition (eqsat weighs fused vs split).")
    .with_cond(cond_bias_compose())
    .with_derivation("assoc_linear_bias")
    .with_tag("categorical")
    .with_tag("expansive")
}

/// `naturality_scalar`: `matmul(W, x∘c) → (W@x)∘c` for scalar `c`.
///
/// Python: `tensor.py` `NATURALITY_SCALAR`, cond `("scalar","c")`,
/// tags `categorical`+`expansive`.
pub fn naturality_scalar() -> Law {
    Law::new(
        "naturality_scalar",
        Pat::op(
            "matmul",
            vec![
                Pat::var("W"),
                Pat::op("mul", vec![Pat::var("x"), Pat::var("c")]),
            ],
        ),
        Pat::op(
            "mul",
            vec![
                Pat::op("matmul", vec![Pat::var("W"), Pat::var("x")]),
                Pat::var("c"),
            ],
        ),
    )
    .with_law("Naturality: scalar multiplication commutes with linear maps.")
    .with_cond(json!(["scalar", "c"]))
    .with_tag("categorical")
    .with_tag("expansive")
}

/// `naturality_scalar_rev`: `(W@x)∘c → W@(x∘c)` — pull the scalar
/// into the matmul's input.
///
/// Python: `tensor.py` `NATURALITY_SCALAR_REV`, cond `("scalar","c")`,
/// tags `categorical`+`expansive`, derivation `naturality_scalar`.
pub fn naturality_scalar_rev() -> Law {
    Law::new(
        "naturality_scalar_rev",
        Pat::op(
            "mul",
            vec![
                Pat::op("matmul", vec![Pat::var("W"), Pat::var("x")]),
                Pat::var("c"),
            ],
        ),
        Pat::op(
            "matmul",
            vec![
                Pat::var("W"),
                Pat::op("mul", vec![Pat::var("x"), Pat::var("c")]),
            ],
        ),
    )
    .with_law("Reverse naturality: pull scalar into the matmul's input.")
    .with_cond(json!(["scalar", "c"]))
    .with_derivation("naturality_scalar")
    .with_tag("categorical")
    .with_tag("expansive")
}

/// The `split_scalar_{op}_{l,r}` family — `split_D(f(x,c)) =
/// f(split_D(x), c)` (and the left-scalar variant) for the four
/// `POINTWISE_BINARY_OPS` (`add`/`div`/`mul`/`sub`), guarded by
/// `("scalar","c")`.
///
/// Python: `tensor.py` `_split_scalar`/`SPLIT_SCALAR_RULES`, tags
/// `categorical`+`expansive`, derivation `naturality_scalar`.
fn split_scalar_rules() -> Vec<Law> {
    let mut laws = Vec::new();
    for op in ["add", "div", "mul", "sub"] {
        for scalar_left in [false, true] {
            let suf = if scalar_left { "l" } else { "r" };
            let split_attrs = || {
                afields(vec![
                    ("sizes", af_bind("SZ")),
                    ("dim", af_bind("D")),
                    ("index", af_bind("I")),
                ])
            };
            let split_of = |t: Pat| Pat::Op(OpPat::new("split").arg(t).attrs_fields(split_attrs()));
            let (lhs, rhs) = if scalar_left {
                (
                    split_of(Pat::op(op, vec![Pat::var("c"), Pat::var("x")])),
                    Pat::op(op, vec![Pat::var("c"), split_of(Pat::var("x"))]),
                )
            } else {
                (
                    split_of(Pat::op(op, vec![Pat::var("x"), Pat::var("c")])),
                    Pat::op(op, vec![split_of(Pat::var("x")), Pat::var("c")]),
                )
            };
            laws.push(
                Law::new(format!("split_scalar_{op}_{suf}"), lhs, rhs)
                    .with_law(format!(
                        "split commutes with pointwise {op} when the other operand is scalar: the partition reads the same elementwise grid; the scalar accompanies every block unchanged."
                    ))
                    .with_cond(json!(["scalar", "c"]))
                    .with_derivation("naturality_scalar")
                    .with_tag("categorical")
                    .with_tag("expansive"),
            );
        }
    }
    laws
}

/// Python `_COND_FUSE_PAIR` — the shared-weight-shape guard for the
/// product-structure fuses.
fn cond_fuse_pair() -> serde_json::Value {
    json!([
        "and",
        ["shape-compat", "A", "B"],
        ["rank", "A", ">=", 2],
        ["rank", "B", ">=", 2]
    ])
}

/// `concat(A, B, dim=0)` — the fused weight.
fn fused_weight_ab() -> Pat {
    Pat::Op(
        OpPat::new("concat")
            .arg(Pat::var("A"))
            .arg(Pat::var("B"))
            .attrs_lit(json!({"dim": 0})),
    )
}

/// `linear(x, concat(A,B,dim=0))` — the fused GEMM.
fn fused_linear() -> Pat {
    Pat::op("linear", vec![Pat::var("x"), fused_weight_ab()])
}

/// `chunk(fused_linear, chunks=2, dim=-1, index=i)` — one projection.
fn fused_chunk(index: i64) -> Pat {
    Pat::Op(
        OpPat::new("chunk")
            .arg(fused_linear())
            .attrs_lit(json!({"chunks": 2, "dim": -1, "index": index})),
    )
}

/// `swiglu_fuse`: `silu(linear(x,A)) · linear(x,B) →
/// silu(chunk0(fused)) · chunk1(fused)` — the product universal
/// property (fused SwiGLU gate/up).
///
/// Python: `tensor.py` `SWIGLU_FUSE`, cond `_COND_FUSE_PAIR`, tags
/// `fusion`+`subsumed`.
pub fn swiglu_fuse() -> Law {
    Law::new(
        "swiglu_fuse",
        Pat::op(
            "mul",
            vec![
                Pat::op("silu", vec![Pat::op("linear", vec![Pat::var("x"), Pat::var("A")])]),
                Pat::op("linear", vec![Pat::var("x"), Pat::var("B")]),
            ],
        ),
        Pat::op(
            "mul",
            vec![
                Pat::op("silu", vec![fused_chunk(0)]),
                fused_chunk(1),
            ],
        ),
    )
    .with_law(
        "Product universal property: <f,g> = (f x g) . Delta.  Two projections of the same input are ONE GEMM into V x V, then project.  (Fused SwiGLU gate/up — MergedColumnParallelLinear.)",
    )
    .with_cond(cond_fuse_pair())
    .with_tag("fusion")
    .with_tag("subsumed")
}

/// `parallel_mul_fuse`: `linear(x,A) · linear(x,B) →
/// chunk0(fused) · chunk1(fused)` — the gate-free pairing.
///
/// Python: `tensor.py` `PARALLEL_MUL_FUSE`, cond `_COND_FUSE_PAIR`,
/// tags `fusion`+`subsumed`.
pub fn parallel_mul_fuse() -> Law {
    Law::new(
        "parallel_mul_fuse",
        Pat::op(
            "mul",
            vec![
                Pat::op("linear", vec![Pat::var("x"), Pat::var("A")]),
                Pat::op("linear", vec![Pat::var("x"), Pat::var("B")]),
            ],
        ),
        Pat::op("mul", vec![fused_chunk(0), fused_chunk(1)]),
    )
    .with_law(
        "Pairing without a gate nonlinearity: mul(<pi1 f>, <pi2 g>) recovers the parallel-product form.",
    )
    .with_cond(cond_fuse_pair())
    .with_tag("fusion")
    .with_tag("subsumed")
}

// ---------------------------------------------------------------------------
//  Diagonal-scale naturality for `linear` — the RMSNorm-folding rules,
//  and the sdpa fold/expand attention family.
// ---------------------------------------------------------------------------

/// Python `_SPEC_WT` — `F.linear`'s weight transpose as a shape spec
/// (`transpose-out` with torch-default `None` axes).
fn spec_wt() -> serde_json::Value {
    json!(["transpose-out", "W", null, null])
}

/// Python `_COND_LINEAR_XW` — `linear(x, W)` is well-typed.
fn cond_linear_xw() -> serde_json::Value {
    json!(["mm-shape-ok", "x", spec_wt()])
}

/// Python `_COND_LINEAR_WT` — `F.linear`'s weight arity ceiling.
fn cond_linear_wt() -> serde_json::Value {
    json!(["rank", "W", "<=", 2])
}

/// Python `_SPEC_LINEAR_OUT` — `linear(x, W)`'s output shape.
fn spec_linear_out() -> serde_json::Value {
    json!(["mm-out", "x", spec_wt()])
}

/// Python `_COND_CHANNEL_SCALE` — the per-channel scale guard.
fn cond_channel_scale() -> serde_json::Value {
    json!([
        "and",
        cond_linear_xw(),
        cond_linear_wt(),
        ["shaped", "c"],
        ["rank", "W", ">=", 1],
        [
            "or",
            ["scalar", "c"],
            ["and", ["dim-eq", "c", -1, "W", -1], ["ones-but-last", "c"]]
        ]
    ])
}

/// Python `_COND_ROW_SCALE` (var `r`) / `_COND_ROW_SCALE_C` (var `c`)
/// — the per-row scale guard.
fn cond_row_scale(var: &str) -> serde_json::Value {
    json!([
        "and",
        ["or", ["scalar", var], ["dim-eq-const", var, -1, 1]],
        ["bcast-into", var, spec_linear_out()],
        cond_linear_wt()
    ])
}

/// Python `_COND_CHANNEL_TO_ROW` — the guarded composite's
/// transported precondition.
fn cond_channel_to_row() -> serde_json::Value {
    json!(["and", cond_channel_scale(), cond_row_scale("c")])
}

/// `linear_channel_scale`: `linear(x∘c, W) → linear(x, W∘c)` — a
/// channel scale is a right diagonal absorbed at compile time.
///
/// Python: `tensor.py` `LINEAR_CHANNEL_SCALE`, cond
/// `_COND_CHANNEL_SCALE`, tags `symmetry`+`expansive`.
pub fn linear_channel_scale() -> Law {
    Law::new(
        "linear_channel_scale",
        Pat::op(
            "linear",
            vec![Pat::op("mul", vec![Pat::var("x"), Pat::var("c")]), Pat::var("W")],
        ),
        Pat::op(
            "linear",
            vec![Pat::var("x"), Pat::op("mul", vec![Pat::var("W"), Pat::var("c")])],
        ),
    )
    .with_law(
        "Channel scale is a right diagonal: (xD)W = x(DW).  Folds the norm's affine gain into the weight at compile time.",
    )
    .with_cond(cond_channel_scale())
    .with_tag("symmetry")
    .with_tag("expansive")
}

/// `linear_channel_scale_rev`: `linear(x, W∘c) → linear(x∘c, W)` —
/// unfold the absorbed channel scale.
///
/// Python: `tensor.py` `LINEAR_CHANNEL_SCALE_REV`, cond
/// `_COND_CHANNEL_SCALE`, tags `symmetry`+`expansive`, derivation
/// `linear_channel_scale`.
pub fn linear_channel_scale_rev() -> Law {
    Law::new(
        "linear_channel_scale_rev",
        Pat::op(
            "linear",
            vec![
                Pat::var("x"),
                Pat::op("mul", vec![Pat::var("W"), Pat::var("c")]),
            ],
        ),
        Pat::op(
            "linear",
            vec![
                Pat::op("mul", vec![Pat::var("x"), Pat::var("c")]),
                Pat::var("W"),
            ],
        ),
    )
    .with_law("Reverse channel-scale fold (eqsat compares both forms).")
    .with_cond(cond_channel_scale())
    .with_derivation("linear_channel_scale")
    .with_tag("symmetry")
    .with_tag("expansive")
}

/// `linear_row_scale`: `linear(x∘r, W) → linear(x,W)∘r` — a row
/// scale is a left diagonal commuting through the linear map.
///
/// Python: `tensor.py` `LINEAR_ROW_SCALE`, cond `_COND_ROW_SCALE`,
/// tags `symmetry`+`expansive`.
pub fn linear_row_scale() -> Law {
    Law::new(
        "linear_row_scale",
        Pat::op(
            "linear",
            vec![Pat::op("mul", vec![Pat::var("x"), Pat::var("r")]), Pat::var("W")],
        ),
        Pat::op(
            "mul",
            vec![Pat::op("linear", vec![Pat::var("x"), Pat::var("W")]), Pat::var("r")],
        ),
    )
    .with_law(
        "Row scale is a left diagonal: commutes through the linear map to the output (naturality of scalar action).",
    )
    .with_cond(cond_row_scale("r"))
    .with_tag("symmetry")
    .with_tag("expansive")
}

/// `linear_row_scale_rev`: `linear(x,W)∘r → linear(x∘r, W)` —
/// reverse row-scale hoist.
///
/// Python: `tensor.py` `LINEAR_ROW_SCALE_REV`, cond
/// `_COND_ROW_SCALE`, tags `symmetry`+`expansive`, derivation
/// `linear_row_scale`.
pub fn linear_row_scale_rev() -> Law {
    Law::new(
        "linear_row_scale_rev",
        Pat::op(
            "mul",
            vec![
                Pat::op("linear", vec![Pat::var("x"), Pat::var("W")]),
                Pat::var("r"),
            ],
        ),
        Pat::op(
            "linear",
            vec![
                Pat::op("mul", vec![Pat::var("x"), Pat::var("r")]),
                Pat::var("W"),
            ],
        ),
    )
    .with_law("Reverse row-scale hoist (eqsat compares both forms).")
    .with_cond(cond_row_scale("r"))
    .with_derivation("linear_row_scale")
    .with_tag("symmetry")
    .with_tag("expansive")
}

/// `linear_channel_to_row_scale`: `linear(x, W∘c) → c·linear(x,W)`
/// — the guarded composite of `linear_channel_scale_rev` then
/// `linear_row_scale`, premise-guarded.
///
/// Python: `tensor.py` `LINEAR_CHANNEL_TO_ROW_SCALE`, cond
/// `_COND_CHANNEL_TO_ROW`, tags `categorical`, derivation
/// `(linear_channel_scale_rev, linear_row_scale)`.
pub fn linear_channel_to_row_scale() -> Law {
    Law::new(
        "linear_channel_to_row_scale",
        Pat::op(
            "linear",
            vec![Pat::var("x"), Pat::op("mul", vec![Pat::var("W"), Pat::var("c")])],
        ),
        Pat::op(
            "mul",
            vec![Pat::op("linear", vec![Pat::var("x"), Pat::var("W")]), Pat::var("c")],
        ),
    )
    .with_law(
        "linear(x, W∘c) = c·linear(x, W) — a channel scale folded into the weight migrates through the linear map and re-emerges as a row scale on the output (channel_scale_rev then row_scale, fused into one premise-guarded step).",
    )
    .with_cond(cond_channel_to_row())
    .with_derivation("linear_channel_scale_rev")
    .with_derivation("linear_row_scale")
    .with_tag("categorical")
}

// ---------------------------------------------------------------------------
//  The sdpa fold family — `softmax(qkᵀ·s [+mask])v → sdpa`, and the
//  opt-in `sdpa_expand*` decompose family.
// ---------------------------------------------------------------------------

/// Python `_QK_SCORES` — `matmul(Q, transpose(K, TD1, TD2))`.
fn qk_scores() -> Pat {
    Pat::op(
        "matmul",
        vec![
            Pat::var("Q"),
            Pat::Op(
                OpPat::new("transpose")
                    .arg(Pat::var("K"))
                    .attrs_fields(afields(vec![
                        ("dim0", af_bind("TD1")),
                        ("dim1", af_bind("TD2")),
                    ])),
            ),
        ],
    )
}

/// Python `_COND_SCORE_T` + `_COND_SM_DIM` + `_COND_SDPA_BASE`.
fn cond_sdpa_base() -> serde_json::Value {
    json!([
        "and",
        [
            "and",
            ["concrete", "K"],
            ["attr-type", "TD1", "int"],
            ["attr-type", "TD2", "int"],
            ["axes-last2", "K", "TD1", "TD2"]
        ],
        ["axis", "Q", "SD", -1]
    ])
}

/// Python `_COND_SDPA_SCALED` — `_COND_SDPA_BASE` + `("const-num","S")`.
fn cond_sdpa_scaled() -> serde_json::Value {
    json!(["and", cond_sdpa_base(), ["const-num", "S"]])
}

/// `softmax(<scores>, dim=SD)` — the spelled attention core.
fn spelled_softmax(inner: Pat) -> Pat {
    Pat::Op(
        OpPat::new("softmax")
            .arg(inner)
            .attrs_fields(afields(vec![("dim", af_bind("SD"))])),
    )
}

/// `dropout(<inner>, p=DP, train=DT)` — the eval-mode wrapper the
/// export may carry around the softmax.
fn spelled_dropout(inner: Pat) -> Pat {
    Pat::Op(OpPat::new("dropout").arg(inner).attrs_fields(afields(vec![
        ("p", af_bind("DP")),
        ("train", af_bind("DT")),
    ])))
}

/// One `sdpa_fold*` scaled spelling: name suffix, scores pattern,
/// add-cond, masked-fill-cond, dspec.
type SdpaScaledRow = (&'static str, fn() -> Pat, Value, Value, Value);

/// The `sdpa_fold_add*`/`sdpa_fold_masked_fill*` family: the six
/// mask/scale spellings × the optional dropout wrapper (12 laws),
/// plus the three mask-free `sdpa_fold_*_nomask` laws — Python's
/// `SDPA_FOLD_RULES`.
fn sdpa_fold_rules() -> Vec<Law> {
    let mut laws = Vec::new();
    // (name suffix, scores pat, add-cond, masked-fill-cond, dspec)
    let scaled: [SdpaScaledRow; 3] = [
        (
            "mul",
            || Pat::op("mul", vec![qk_scores(), Pat::var("S")]),
            cond_sdpa_scaled(),
            json!([
                "and",
                cond_sdpa_base(),
                ["const-cmp", "F", "<", -1e30],
                ["const-num", "S"]
            ]),
            json!({"SC": ["float", ["const", "S"]]}),
        ),
        (
            "div",
            || Pat::op("div", vec![qk_scores(), Pat::var("S")]),
            cond_sdpa_scaled(),
            json!([
                "and",
                cond_sdpa_base(),
                ["const-cmp", "F", "<", -1e30],
                ["const-num", "S"]
            ]),
            json!({"SC": ["recip", ["float", ["const", "S"]]]}),
        ),
        (
            "",
            qk_scores,
            cond_sdpa_base(),
            json!(["and", cond_sdpa_base(), ["const-cmp", "F", "<", -1e30]]),
            json!({"SC": 1.0}),
        ),
    ];
    for (sname, scores, cond_add, cond_mf, dspec) in &scaled {
        for (wname, wrap) in [
            ("", spelled_softmax as fn(Pat) -> Pat),
            ("_drop", |sm| spelled_dropout(spelled_softmax(sm))),
        ] {
            // add-mask form: matmul(softmax(add(scores, M)), V)
            let lhs_add = Pat::op(
                "matmul",
                vec![
                    wrap(Pat::op("add", vec![scores(), Pat::var("M")])),
                    Pat::var("V"),
                ],
            );
            let rhs_add = Pat::Op(
                OpPat::new("sdpa")
                    .arg(Pat::var("Q"))
                    .arg(Pat::var("K"))
                    .arg(Pat::var("V"))
                    .arg(Pat::var("M"))
                    .attrs_fields(afields(vec![("scale", af_bind("SC"))])),
            );
            laws.push(
                Law::new(
                    format!("sdpa_fold_add{sname}{wname}"),
                    lhs_add,
                    rhs_add,
                )
                .with_law(
                    "softmax(qk^T s + m) v is sdpa — the additive mask is the kernel's attn_mask argument.",
                )
                .with_cond(cond_add.clone())
                .with_dspec(dspec.clone())
                .with_tag("fusion"),
            );
            // masked_fill form: matmul(softmax(masked_fill(scores, MK, F)), V)
            let lhs_mf = Pat::op(
                "matmul",
                vec![
                    wrap(Pat::op(
                        "masked_fill",
                        vec![scores(), Pat::var("MK"), Pat::var("F")],
                    )),
                    Pat::var("V"),
                ],
            );
            let rhs_mf = Pat::Op(
                OpPat::new("sdpa")
                    .arg(Pat::var("Q"))
                    .arg(Pat::var("K"))
                    .arg(Pat::var("V"))
                    .arg(Pat::op("logical_not", vec![Pat::var("MK")]))
                    .attrs_fields(afields(vec![("scale", af_bind("SC"))])),
            );
            laws.push(
                Law::new(
                    format!("sdpa_fold_masked_fill{sname}{wname}"),
                    lhs_mf,
                    rhs_mf,
                )
                .with_law(
                    "masked_fill(m, -inf) before softmax is a boolean attn_mask — logical_not turns the fill-mask into SDPA's keep-mask.",
                )
                .with_cond(cond_mf.clone())
                .with_dspec(dspec.clone())
                .with_tag("fusion"),
            );
        }
    }
    // The mask-free forms — `nn.MultiheadAttention`'s default export.
    let qv_rank = json!(["rank", "Q", ">=", 2]);
    let v_rank = json!(["rank", "V", ">=", 2]);
    let nomask: [(&str, Pat, serde_json::Value, serde_json::Value, String); 3] = [
        (
            "sdpa_fold_nomask",
            Pat::op("matmul", vec![spelled_softmax(qk_scores()), Pat::var("V")]),
            json!(["and", cond_sdpa_base(), qv_rank.clone(), v_rank.clone()]),
            json!({"SC": 1.0}),
            "softmax(q kᵀ) v is sdpa(q, k, v, scale=1) — the mask-free spelled attention is one fused kernel (the add/masked_fill family's empty-mask row).".to_string(),
        ),
        (
            "sdpa_fold_div_nomask",
            Pat::op(
                "matmul",
                vec![
                    spelled_softmax(Pat::op("div", vec![qk_scores(), Pat::var("S")])),
                    Pat::var("V"),
                ],
            ),
            json!([
                "and",
                cond_sdpa_scaled(),
                qv_rank.clone(),
                v_rank.clone()
            ]),
            json!({"SC": ["recip", ["float", ["const", "S"]]]}),
            "softmax(q kᵀ / s) v is sdpa(q, k, v, scale=1/s) — the mask-free spelled form with the decomposed pre-softmax scale.".to_string(),
        ),
        (
            "sdpa_fold_mul_nomask",
            Pat::op(
                "matmul",
                vec![
                    spelled_softmax(Pat::op("mul", vec![qk_scores(), Pat::var("S")])),
                    Pat::var("V"),
                ],
            ),
            json!([
                "and",
                cond_sdpa_scaled(),
                qv_rank.clone(),
                v_rank.clone()
            ]),
            json!({"SC": ["float", ["const", "S"]]}),
            "softmax(q kᵀ · s) v is sdpa(q, k, v, scale=s) — the mul spelling of the mask-free scaled fold, and the re-fusion half of the sdpa_expand pair (the mediating inverse for the expand/fold critical pair, the silu_fold analogue).".to_string(),
        ),
    ];
    for (name, lhs, cond, dspec, law) in nomask {
        laws.push(
            Law::new(
                name,
                lhs,
                Pat::Op(
                    OpPat::new("sdpa")
                        .arg(Pat::var("Q"))
                        .arg(Pat::var("K"))
                        .arg(Pat::var("V"))
                        .attrs_fields(afields(vec![("scale", af_bind("SC"))])),
                ),
            )
            .with_law(law)
            .with_cond(cond)
            .with_dspec(dspec)
            .with_tag("fusion"),
        );
    }
    laws
}

/// The expanded interior every `sdpa_expand*` mints:
/// `matmul(softmax(mul(matmul(Q, Kᵀ₋₂₋₁), SC0), dim=-1), V)`.
fn sdpa_expanded() -> Pat {
    Pat::op(
        "matmul",
        vec![
            Pat::Op(
                OpPat::new("softmax")
                    .arg(Pat::op(
                        "mul",
                        vec![
                            Pat::op(
                                "matmul",
                                vec![
                                    Pat::var("Q"),
                                    Pat::Op(
                                        OpPat::new("transpose")
                                            .arg(Pat::var("K"))
                                            .attrs_lit(json!({"dim0": -2, "dim1": -1})),
                                    ),
                                ],
                            ),
                            Pat::var("SC0"),
                        ],
                    ))
                    .attrs_lit(json!({"dim": -1})),
            ),
            Pat::var("V"),
        ],
    )
}

/// The `sdpa_expand*` family — all eight key-sets.  The four
/// scale-free spellings (`sdpa_expand`, `_d`, `_dc`, `_dcg`) mint
/// the implicit scale `1/√(q[-1])` through the `("dim","Q",-1)`
/// shape channel — ported at stage 4 — and `_d`/`_dc`/`_dcg`'s
/// `dropout_p`/`is_causal`/`enable_gqa` defaults are literal pins
/// (no metavariable binds), so they needed no guard at all.
///
/// Python: `tensor.py` `_make_sdpa_expand_rules` /
/// `SDPA_EXPAND_RULES`, opt-in `attention` preset, tag `attention`,
/// derivation `sdpa_fold_mul_nomask`.
fn sdpa_expand_rules() -> Vec<Law> {
    // (suffix, attrs fields, law text, dspec)
    type SdpaKeyset = (
        &'static str,
        Vec<(&'static str, AttrField)>,
        &'static str,
        serde_json::Value,
    );
    // The implicit-scale mint every scale-free key-set shares:
    // `SC0 = rsqrt(dim(Q, -1))` — the `dim` shape channel reads
    // Q's last extent (declines → the derive vetoes, never a guess).
    let free_dspec = || json!({"$leaf:SC0": ["rsqrt", ["dim", "Q", -1]]});
    let free_law = "sdpa(q, k, v) IS softmax(q@kᵀ/√d)@v with the implicit scale minted as mul(q@kᵀ, 1/√d_head) — the fused op's decomposed form, exposed for the interior laws (refolds through sdpa_fold_mul_nomask / sdpa_fold_nomask).";
    // Python spells the same kernel-definition text on all four
    // scaled key-sets (verified against `SDPA_EXPAND_RULES`).
    let scaled_law = "sdpa(q, k, v, scale=s) IS matmul(softmax(mul(q @ kᵀ, s), dim=-1), v) — the kernel's definition, expanded so the interior becomes a redex for the scale-migration and spelled-attention laws (the decompose half of the expand/fold pair; refolds through sdpa_fold_mul_nomask / sdpa_fold_nomask).";
    let keysets: [SdpaKeyset; 8] = [
        // ---- scale-free key-sets (implicit 1/√d mint) ---------------
        ("", vec![], free_law, free_dspec()),
        (
            "_d",
            vec![("dropout_p", af_lit(json!(0.0)))],
            free_law,
            free_dspec(),
        ),
        (
            "_dc",
            vec![
                ("dropout_p", af_lit(json!(0.0))),
                ("is_causal", af_lit(json!(false))),
            ],
            free_law,
            free_dspec(),
        ),
        (
            "_dcg",
            vec![
                ("dropout_p", af_lit(json!(0.0))),
                ("is_causal", af_lit(json!(false))),
                ("enable_gqa", af_lit(json!(false))),
            ],
            free_law,
            free_dspec(),
        ),
        // ---- scaled key-sets (`scale` binds `SC`) -------------------
        (
            "_s",
            vec![("scale", af_bind("SC"))],
            scaled_law,
            json!({"$leaf:SC0": ["attr", "SC"]}),
        ),
        (
            "_ds",
            vec![("dropout_p", af_lit(json!(0.0))), ("scale", af_bind("SC"))],
            scaled_law,
            json!({"$leaf:SC0": ["attr", "SC"]}),
        ),
        (
            "_dcs",
            vec![
                ("dropout_p", af_lit(json!(0.0))),
                ("is_causal", af_lit(json!(false))),
                ("scale", af_bind("SC")),
            ],
            scaled_law,
            json!({"$leaf:SC0": ["attr", "SC"]}),
        ),
        (
            "_dcgs",
            vec![
                ("dropout_p", af_lit(json!(0.0))),
                ("is_causal", af_lit(json!(false))),
                ("enable_gqa", af_lit(json!(false))),
                ("scale", af_bind("SC")),
            ],
            scaled_law,
            json!({"$leaf:SC0": ["attr", "SC"]}),
        ),
    ];
    keysets
        .into_iter()
        .map(|(suffix, fields, law, dspec)| {
            // Python `attrs {}` spells the no-declared-attrs
            // convention (kernel `attrs: None`); the literal-keyed
            // sets go through `attrs_fields` either way.
            let lhs = if fields.is_empty() {
                Pat::Op(
                    OpPat::new("sdpa")
                        .arg(Pat::var("Q"))
                        .arg(Pat::var("K"))
                        .arg(Pat::var("V")),
                )
            } else {
                Pat::Op(
                    OpPat::new("sdpa")
                        .arg(Pat::var("Q"))
                        .arg(Pat::var("K"))
                        .arg(Pat::var("V"))
                        .attrs_fields(afields(fields)),
                )
            };
            Law::new(format!("sdpa_expand{suffix}"), lhs, sdpa_expanded())
                .with_law(law)
                .with_dspec(dspec)
                .with_derivation("sdpa_fold_mul_nomask")
                .with_tag("attention")
        })
        .collect()
}

// ---------------------------------------------------------------------------
//  Stage-4 ports — the flat-map/repeat pairing guards and the
//  shape-reading dspec channels (`shape`/`dim`/`leaf-dim`/`bcast`)
//  over `crate::shape`'s inference. What remains Python-side:
//  `tval`/`tapp` tensor values, procedural `check`/`derive` hooks
//  (`conv_bn_fold`'s `missing_hooks`), and the `prim:synop*` minted
//  records' backend-side synthetic-op denotations (their fold
//  records ship as the kit-eqsat `prim_folds` pack data instead).
// ---------------------------------------------------------------------------

/// `mul_unsq_nat`: `unsqueeze(u,d) · unsqueeze(v,d) →
/// unsqueeze(u · v, d)` under the `flat-map-unsq` pairing guard —
/// the broadcast grids must coincide AND both operands' flat-read
/// coefficient maps must agree across the two evaluation orders
/// (the guard decides the region — broadcast-incompatible pairs and
/// unproven shapes decline, never a guess).
///
/// Python: `tensor.py` `MUL_UNSQ_NAT`, cond
/// `("flat-map-unsq","u","UD","v",("unsq-out","v","UD"))`, tags
/// `simplification`.
pub fn mul_unsq_nat() -> Law {
    Law::new(
        "mul_unsq_nat",
        Pat::op("mul", vec![unsqueeze_u("u"), unsqueeze_u("v")]),
        Pat::Op(
            OpPat::new("unsqueeze")
                .arg(Pat::op("mul", vec![Pat::var("u"), Pat::var("v")]))
                .attrs_fields(afields(vec![("dim", af_bind("UD"))])),
        ),
    )
    .with_law(
        "unsqueeze(u, d) · unsqueeze(v, d) = unsqueeze(u · v, d) — elementwise mul commutes with a shared size-1 axis insertion exactly when the broadcast grids coincide AND both operands' flat-read maps agree across the two orders (the flat-map-unsq region — equal ranks always qualify; unequal ranks only when the insert lands in the pad).  Fuses two views into one and shrinks the pointwise domain a rank.",
    )
    .with_cond(json!([
        "flat-map-unsq",
        "u",
        "UD",
        "v",
        ["unsq-out", "v", "UD"]
    ]))
    .with_tag("simplification")
}

/// The manual-RMSNorm reduction both `rms_norm_fold*` laws share:
/// `mul(u, rsqrt(add(mean(pow(u, P), dim=MD, keepdim=MK), EPS)))` —
/// `P`/`EPS` are term metavars (Const leaves), `MD`/`MK` attr binds.
fn rms_reduce(u: &str) -> Pat {
    Pat::op(
        "mul",
        vec![
            Pat::var(u),
            Pat::Op(OpPat::new("rsqrt").arg(Pat::op(
                "add",
                vec![
                        Pat::Op(
                            OpPat::new("mean")
                                .arg(Pat::op("pow", vec![Pat::var(u), Pat::var("P")]))
                                .attrs_fields(afields(vec![
                                    ("dim", af_bind("MD")),
                                    ("keepdim", af_bind("MK")),
                                ])),
                        ),
                        Pat::var("EPS"),
                    ],
            ))),
        ],
    )
}

/// The shared `rms_norm_fold*` cond prefix: keepdim pinned true,
/// `EPS` a numeric Const, `P == 2`.
fn cond_rms_base() -> serde_json::Value {
    json!([
        "and",
        ["attr-is", "MK", true],
        ["const-num", "EPS"],
        ["const-cmp", "P", "==", 2]
    ])
}

/// The shared `rms_norm_fold*` dspec: the reduce dims name `u`'s
/// trailing block (the normalized shape) and `eps` mints a Const
/// leaf through `float(const(EPS))`.
fn dspec_rms() -> serde_json::Value {
    json!({
        "ND": ["shape", ["tail-block", "u", "MD"]],
        "EP": ["float", ["const", "EPS"]]
    })
}

/// `rms_norm(u, w)` spelled manually: `u·rsqrt(mean(u²)+eps)·w`.
///
/// Python: `tensor.py` `RMS_NORM_FOLD`, cond
/// `_COND_RMS_FOLD` + `("shape-eq","w",("tail-block","u","MD"))`,
/// tags `simplification`.
pub fn rms_norm_fold() -> Law {
    Law::new(
        "rms_norm_fold",
        Pat::op("mul", vec![rms_reduce("u"), Pat::var("w")]),
        Pat::Op(
            OpPat::new("rms_norm")
                .arg(Pat::var("u"))
                .arg(Pat::var("w"))
                .attrs_fields(afields(vec![
                    ("dim", af_bind("ND")),
                    ("eps", af_bind("EP")),
                ])),
        ),
    )
    .with_law(
        "x·rsqrt(mean(x²)+eps)·w IS rms_norm(x, w): the manual-RMSNorm fold — composed-then-reduced then scaled IS the kernel's definition.  Folds mul+mul+rsqrt+add+mean+pow to one dispatched op.",
    )
    .with_cond(json!([
        "and",
        cond_rms_base(),
        ["shape-eq", "w", ["tail-block", "u", "MD"]]
    ]))
    .with_dspec(dspec_rms())
    .with_tag("simplification")
}

/// `rms_norm(u)` spelled manually — the weight-free twin.
///
/// Python: `tensor.py` `RMS_NORM_FOLD_NOGAIN`, cond
/// `_COND_RMS_FOLD` + `("shaped",("tail-block","u","MD"))`, tags
/// `simplification`.
pub fn rms_norm_fold_nogain() -> Law {
    Law::new(
        "rms_norm_fold_nogain",
        rms_reduce("u"),
        Pat::Op(
            OpPat::new("rms_norm")
                .arg(Pat::var("u"))
                .attrs_fields(afields(vec![
                    ("dim", af_bind("ND")),
                    ("eps", af_bind("EP")),
                ])),
        ),
    )
    .with_law(
        "x·rsqrt(mean(x²)+eps) IS rms_norm(x): the weight-free manual-RMSNorm fold — the gained fold's inner ``mul`` and a real spelling of its own (gain-free RMSNorm blocks).",
    )
    .with_cond(json!([
        "and",
        cond_rms_base(),
        ["shaped", ["tail-block", "u", "MD"]]
    ]))
    .with_dspec(dspec_rms())
    .with_tag("simplification")
}

/// The `head(t, shape_bind)` view for [`qkv_fuse_asym`]'s uneven
/// splits — `transpose(reshape(t, shape=?S<i>), dim0=1, dim1=2)`
/// with a per-head shape metavar (unlike [`head_view`]'s shared `S`).
fn head_view_named(t: Pat, shape_bind: &str) -> Pat {
    Pat::Op(
        OpPat::new("transpose")
            .arg(Pat::Op(
                OpPat::new("reshape")
                    .arg(t)
                    .attrs_fields(afields(vec![("shape", af_bind(shape_bind))])),
            ))
            .attrs_lit(json!({"dim0": 1, "dim1": 2})),
    )
}

/// `qkv_fuse_asym`: the asymmetric triple pairing — three `linear`
/// projections of the same input with *different* output dims fuse
/// into ONE GEMM over the concatenated weights plus three uneven
/// `split` views (the GQA fused-QKV spelling; `SZ` mints the split
/// sizes from the weights' declared dim-0 extents via `leaf-dim`).
///
/// Python: `Op.make("sdpa", head_Si(linear(x,Q)), …,
/// scale="SC", enable_gqa="G") → sdpa(head_Si(split_i(linear(x,
/// cat(cat(Q,K),V)))), …)`, law "Asymmetric triple pairing: the
/// same product law as qkv_fuse, but the three projections have
/// different output dims — one GEMM, three uneven split views.
/// (GQA fused QKV.)", tags `fusion`+`subsumed`.
pub fn qkv_fuse_asym() -> Law {
    // The fused weight: `concat(concat(Q, K, dim=0), V, dim=0)` —
    // same product law as `qkv_fuse`.
    let fused_weight = Pat::Op(
        OpPat::new("concat")
            .arg(Pat::Op(
                OpPat::new("concat")
                    .arg(Pat::var("Q"))
                    .arg(Pat::var("K"))
                    .attrs_lit(json!({"dim": 0})),
            ))
            .arg(Pat::var("V"))
            .attrs_lit(json!({"dim": 0})),
    );
    // One uneven split projection: `split(linear(x, W), sizes=SZ,
    // dim=-1, index=i)`.
    let split_proj = |index: i64| {
        Pat::Op(
            OpPat::new("split")
                .arg(Pat::op("linear", vec![Pat::var("x"), fused_weight.clone()]))
                .attrs_fields(afields(vec![
                    ("sizes", af_bind("SZ")),
                    ("dim", af_lit(json!(-1))),
                    ("index", af_lit(json!(index))),
                ])),
        )
    };
    let sdpa_attrs = || afields(vec![("scale", af_bind("SC")), ("enable_gqa", af_bind("G"))]);
    let lhs_heads: Vec<Pat> = [
        ("S1", Pat::var("Q")),
        ("S2", Pat::var("K")),
        ("S3", Pat::var("V")),
    ]
    .into_iter()
    .map(|(s, w)| head_view_named(Pat::op("linear", vec![Pat::var("x"), w]), s))
    .collect();
    let rhs_heads: Vec<Pat> = ["S1", "S2", "S3"]
        .into_iter()
        .enumerate()
        .map(|(i, s)| head_view_named(split_proj(i as i64), s))
        .collect();
    Law::new(
        "qkv_fuse_asym",
        Pat::Op(
            OpPat::new("sdpa")
                .args(lhs_heads)
                .attrs_fields(sdpa_attrs()),
        ),
        Pat::Op(OpPat::new("sdpa").args(rhs_heads).attrs_fields(sdpa_attrs())),
    )
    .with_law(
        "Asymmetric triple pairing: the same product law as qkv_fuse, but the three projections have different output dims — one GEMM, three uneven split views.  (GQA fused QKV.)",
    )
    .with_dspec(json!({
        "SZ": ["tuple",
            ["leaf-dim", "Q", 0],
            ["leaf-dim", "K", 0],
            ["leaf-dim", "V", 0]
        ]
    }))
    .with_tag("fusion")
    .with_tag("subsumed")
}

/// `gqa_absorb_repeat`: the repeat-absorbed SDPA — the
/// `unsqueeze → expand → reshape` copies of K/V (each kv head
/// repeated `r` times, `repeat_kv`) push into the consumer as
/// `enable_gqa`, deleting the materialisation. The `repeat-chain`
/// guards verify both copy maps and `repeat-heads` the head-count
/// equation `q[-2] == k[-2] * r`.
///
/// Python: `tensor.py` `GQA_ABSORB_REPEAT`, cond
/// `_COND_GQA_ABSORB` (`rank==4` ×3, two `repeat-chain`s,
/// `attr-eq-attr` on the expand shapes, `repeat-heads`), tags
/// `fusion`.
pub fn gqa_absorb_repeat() -> Law {
    // `transpose(t, dim0=1, dim1=2)` — the head view.
    let head_t = |t: Pat| {
        Pat::Op(
            OpPat::new("transpose")
                .arg(t)
                .attrs_lit(json!({"dim0": 1, "dim1": 2})),
        )
    };
    // `transpose(reshape(expand(unsqueeze(k, dim=UDk), shape=ESk),
    // shape=RSk), 1, 2)` — the repeat_kv copy chain for one operand.
    let repeat_head = |var: &str, ud: &str, es: &str, rs: &str| {
        head_t(Pat::Op(
            OpPat::new("reshape")
                .arg(Pat::Op(
                    OpPat::new("expand")
                        .arg(Pat::Op(
                            OpPat::new("unsqueeze")
                                .arg(Pat::var(var))
                                .attrs_fields(afields(vec![("dim", af_bind(ud))])),
                        ))
                        .attrs_fields(afields(vec![("shape", af_bind(es))])),
                ))
                .attrs_fields(afields(vec![("shape", af_bind(rs))])),
        ))
    };
    // LHS carries only the shared `{dropout_p, is_causal}` binds;
    // the RHS adds the literal `enable_gqa: true` the absorb
    // installs (Python's exact keysets on both sides).
    let shared_attrs = || {
        afields(vec![
            ("dropout_p", af_bind("D")),
            ("is_causal", af_bind("C")),
        ])
    };
    let mut rhs_attrs = shared_attrs();
    rhs_attrs.insert("enable_gqa".to_string(), af_lit(json!(true)));
    Law::new(
        "gqa_absorb_repeat",
        Pat::Op(
            OpPat::new("sdpa")
                .arg(head_t(Pat::var("q")))
                .arg(repeat_head("k", "UDk", "ESk", "RSk"))
                .arg(repeat_head("v", "UDv", "ESv", "RSv"))
                .attrs_fields(shared_attrs()),
        ),
        Pat::Op(
            OpPat::new("sdpa")
                .arg(head_t(Pat::var("q")))
                .arg(head_t(Pat::var("k")))
                .arg(head_t(Pat::var("v")))
                .attrs_fields(rhs_attrs),
        ),
    )
    .with_law(
        "The diagonal is natural: unsqueeze->expand->reshape copies each kv head r times (repeat_kv).  SDPA implements that copy inside the kernel via enable_gqa — pushing Delta into the consumer deletes the materialisation entirely.",
    )
    .with_cond(json!([
        "and",
        ["rank", "q", "==", 4],
        ["rank", "k", "==", 4],
        ["rank", "v", "==", 4],
        ["repeat-chain", "k", "UDk", "ESk", "RSk"],
        ["repeat-chain", "v", "UDv", "ESv", "RSv"],
        ["attr-eq-attr", "ESk", "ESv"],
        ["repeat-heads", "q", "k", "UDk", "ESk"]
    ]))
    .with_tag("fusion")
}

/// The `batch_norm` attrs-fields shared by the three expand laws:
/// `training`/`momentum`/`eps`/`cudnn_enabled` all bind.
fn bn_attrs() -> BTreeMap<String, AttrField> {
    afields(vec![
        ("training", af_bind("TR")),
        ("momentum", af_bind("M")),
        ("eps", af_bind("EP")),
        ("cudnn_enabled", af_bind("CU")),
    ])
}

/// `reshape(t, shape=CH)` — the channel-view reshape every
/// `batch_norm_expand*` RHS spells (`CH` is dspec-derived).
fn bn_view(t: Pat) -> Pat {
    Pat::Op(
        OpPat::new("reshape")
            .arg(t)
            .attrs_fields(afields(vec![("shape", af_bind("CH"))])),
    )
}

/// `rsqrt(add(reshape(rv, CH), EPS))` — the eval-mode inverse-std
/// factor (`EPS` is the `$leaf:`-minted Const).
fn bn_rsqrt(rv: &str) -> Pat {
    Pat::Op(OpPat::new("rsqrt").arg(Pat::op("add", vec![bn_view(Pat::var(rv)), Pat::var("EPS")])))
}

/// The shared `batch_norm_expand*` cond/dspec — eval mode only
/// (`training=False`), `x` rank 2–6 concrete, channel stats `rm`
/// rank-1 concrete aligned on `x`'s dim 1, `rv` shaped like `rm`.
fn bn_expand_guards(extra: &[serde_json::Value]) -> (serde_json::Value, serde_json::Value) {
    let mut items = vec![
        json!(["attr-is", "TR", false]),
        json!(["concrete", "x"]),
        json!(["rank", "x", ">=", 2]),
        json!(["rank", "x", "<=", 6]),
        json!(["concrete", "rm"]),
        json!(["rank", "rm", "==", 1]),
        json!(["dim-eq", "x", 1, "rm", 0]),
        json!(["shape-eq", "rv", "rm"]),
    ];
    items.extend(extra.iter().cloned());
    let mut cond = vec![json!("and")];
    cond.extend(items);
    let dspec = json!({
        "$leaf:EPS": ["attr", "EP"],
        "CH": ["concat",
            ["tuple", 1],
            ["shape", "rm"],
            ["take", ["tuple", 1, 1, 1, 1], ["sub", ["len", ["shape", "x"]], 2]]
        ]
    });
    (Value::Array(cond), dspec)
}

/// `batch_norm_expand`: eval-mode `batch_norm(x, γ, β, μ, σ²)` IS
/// `γ·((x−μ)·rsqrt(σ²+ε)) + β` — the fused kernel's spelled work,
/// exposed so weight-space laws can fold the per-channel affine into
/// a neighbouring conv.
///
/// Python: `tensor.py` `BATCH_NORM_EXPAND`, tags `simplification`.
/// Opt-in pack [`batch_norm`] — `BATCH_NORM_RULES` in Python.
pub fn batch_norm_expand() -> Law {
    let rhs = Pat::op(
        "add",
        vec![
            Pat::op(
                "mul",
                vec![
                    bn_view(Pat::var("w")),
                    Pat::op(
                        "mul",
                        vec![
                            Pat::op("sub", vec![Pat::var("x"), bn_view(Pat::var("rm"))]),
                            bn_rsqrt("rv"),
                        ],
                    ),
                ],
            ),
            bn_view(Pat::var("b")),
        ],
    );
    let (cond, dspec) = bn_expand_guards(&[
        json!(["shape-eq", "w", "rm"]),
        json!(["shape-eq", "b", "rm"]),
    ]);
    Law::new(
        "batch_norm_expand",
        Pat::Op(
            OpPat::new("batch_norm")
                .arg(Pat::var("x"))
                .arg(Pat::var("w"))
                .arg(Pat::var("b"))
                .arg(Pat::var("rm"))
                .arg(Pat::var("rv"))
                .attrs_fields(bn_attrs()),
        ),
        rhs,
    )
    .with_law(
        "batch_norm(x, γ, β, μ, σ²) IS γ·((x−μ)·rsqrt(σ²+ε)) + β in eval mode — the fused kernel's spelled work, exposed so weight-space laws can fold the per-channel affine into a neighbouring conv.  Fused member stays cheapest under extraction (priced 4 < ~7 expanded passes).",
    )
    .with_cond(cond)
    .with_dspec(dspec)
    .with_tag("simplification")
}

/// `batch_norm_expand_nobias`: the bias-free export arity — the lone
/// mid operand reads as weight.
///
/// Python: `tensor.py` `BATCH_NORM_EXPAND_NOBIAS`, tags
/// `simplification`.
pub fn batch_norm_expand_nobias() -> Law {
    let rhs = Pat::op(
        "mul",
        vec![
            bn_view(Pat::var("w")),
            Pat::op(
                "mul",
                vec![
                    Pat::op("sub", vec![Pat::var("x"), bn_view(Pat::var("rm"))]),
                    bn_rsqrt("rv"),
                ],
            ),
        ],
    );
    let (cond, dspec) = bn_expand_guards(&[json!(["shape-eq", "w", "rm"])]);
    Law::new(
        "batch_norm_expand_nobias",
        Pat::Op(
            OpPat::new("batch_norm")
                .arg(Pat::var("x"))
                .arg(Pat::var("w"))
                .arg(Pat::var("rm"))
                .arg(Pat::var("rv"))
                .attrs_fields(bn_attrs()),
        ),
        rhs,
    )
    .with_law(
        "batch_norm(x, γ, μ, σ²) IS γ·((x−μ)·rsqrt(σ²+ε)) in eval mode — the bias-free export arity (the lone mid operand reads as weight).",
    )
    .with_cond(cond)
    .with_dspec(dspec)
    .with_tag("simplification")
}

/// `batch_norm_expand_plain`: the affine-free export arity.
///
/// Python: `tensor.py` `BATCH_NORM_EXPAND_PLAIN`, tags
/// `simplification`.
pub fn batch_norm_expand_plain() -> Law {
    let rhs = Pat::op(
        "mul",
        vec![
            Pat::op("sub", vec![Pat::var("x"), bn_view(Pat::var("rm"))]),
            bn_rsqrt("rv"),
        ],
    );
    let (cond, dspec) = bn_expand_guards(&[]);
    Law::new(
        "batch_norm_expand_plain",
        Pat::Op(
            OpPat::new("batch_norm")
                .arg(Pat::var("x"))
                .arg(Pat::var("rm"))
                .arg(Pat::var("rv"))
                .attrs_fields(bn_attrs()),
        ),
        rhs,
    )
    .with_law(
        "batch_norm(x, μ, σ²) IS (x−μ)·rsqrt(σ²+ε) in eval mode — the affine-free export arity.",
    )
    .with_cond(cond)
    .with_dspec(dspec)
    .with_tag("simplification")
}

/// `BATCH_NORM_RULES` — the opt-in batch-norm expand pack (Python
/// keeps it out of `ALL_RULES`/`DEFAULT`; the same convention here).
pub fn batch_norm() -> Vec<Law> {
    vec![
        batch_norm_expand(),
        batch_norm_expand_nobias(),
        batch_norm_expand_plain(),
    ]
}

// ---------------------------------------------------------------------------
//  The conv channel-scale/shift family — `CONV_FOLD_RULES`, the
//  `_make_conv_fold_rules` port: per-output-channel scale/shift
//  folds into `conv2d`'s weight/bias under the channel-grid guards.
// ---------------------------------------------------------------------------

/// The conv attr tails in Python order: `("", 0)`, `("_s", 1)`,
/// `("_sp", 2)`, `("_spd", 3)`, `("_spdg", 4)` — each prefix binds
/// the leading attrs `stride`/`padding`/`dilation`/`groups` to the
/// `ST`/`PD`/`DL`/`GR` attr metavars.
const CONV_ATTR_TAIL: &[(&str, &str, &str)] = &[
    ("stride", "ST", ""),
    ("padding", "PD", "_s"),
    ("dilation", "DL", "_sp"),
    ("groups", "GR", "_spd"),
];

/// The conv family suffixes — `na` attrs bound, `""` for none.
const CONV_SUFFIXES: &[&str] = &["", "_s", "_sp", "_spd", "_spdg"];

/// The conv2d attrs-fields for the first `na` attr tails — each
/// binds its attr metavar (`ST`/`PD`/`DL`/`GR`).
fn conv_attrs(na: usize) -> BTreeMap<String, AttrField> {
    CONV_ATTR_TAIL[..na]
        .iter()
        .map(|(key, mvar, _)| (key.to_string(), af_bind(mvar)))
        .collect()
}

/// `conv2d(x, W[, b], attrs…)` — the conv node pattern.
fn conv_pat(x: Pat, w: Pat, b: Option<Pat>, na: usize) -> Pat {
    let mut op = OpPat::new("conv2d").arg(x).arg(w);
    if let Some(b) = b {
        op = op.arg(b);
    }
    if na == 0 {
        // Python `attrs {}` — the no-declared-attrs spelling (the
        // `Pat::op` convention: no `__attrs__` key).
        Pat::Op(op)
    } else {
        Pat::Op(op.attrs_fields(conv_attrs(na)))
    }
}

/// `reshape(sv, shape=WV)` / `reshape(sv, shape=SV)` — the channel
/// view reshapes the conv RHS spells (`WV`/`SV` are dspec-derived).
fn conv_view(var: &Pat, shape_bind: &str) -> Pat {
    Pat::Op(
        OpPat::new("reshape")
            .arg(var.clone())
            .attrs_fields(afields(vec![("shape", af_bind(shape_bind))])),
    )
}

/// The shared conv channel guard for operand `var` (`"sv"` scale /
/// `"cv"` shift): x/W rank-4 concrete, channel alignment (`dim-eq`
/// or the groups-aware `attr-cmp` when `groups` is bound), `var`
/// rank-equal concrete `(1, C, 1, 1)`-shaped, plus the bias triple
/// when present.
fn conv_channel_cond(var: &str, na: usize, bias: bool) -> serde_json::Value {
    let in_ch = if na == 4 {
        // `x.shape[1] == W.shape[1] * groups` — the grouped-conv
        // channel equation as an inline `dim`/`attr` expr compare.
        json!([
            "attr-cmp",
            ["dim", "x", 1],
            "==",
            ["mul", ["dim", "W", 1], ["attr", "GR"]]
        ])
    } else {
        json!(["dim-eq", "x", 1, "W", 1])
    };
    let mut items = vec![
        json!(["concrete", "x"]),
        json!(["rank", "x", "==", 4]),
        json!(["concrete", "W"]),
        json!(["rank", "W", "==", 4]),
        in_ch,
        json!(["concrete", var]),
        json!(["rank-eq", var, "x"]),
        json!(["dim-eq-const", var, 0, 1]),
        json!(["dim-eq", var, 1, "W", 0]),
        json!(["dim-eq-const", var, 2, 1]),
        json!(["dim-eq-const", var, 3, 1]),
    ];
    if bias {
        items.extend([
            json!(["concrete", "b"]),
            json!(["rank", "b", "==", 1]),
            json!(["dim-eq", "b", 0, "W", 0]),
        ]);
    }
    let mut cond = vec![json!("and")];
    cond.extend(items);
    Value::Array(cond)
}

/// The `*_rev` laws' extra clauses: the derived `WV`/`SV` views must
/// equal the freshly-computed channel shapes (the unfold direction
/// re-derives them, so the guard pins the attr payloads it emits).
fn conv_rev_extra(bias: bool) -> Vec<serde_json::Value> {
    let mut extra = vec![json!([
        "attr-cmp",
        "WV",
        "==",
        ["concat", ["take", ["shape", "W"], 1], ["tuple", 1, 1, 1]]
    ])];
    if bias {
        extra.push(json!(["attr-cmp", "SV", "==", ["take", ["shape", "W"], 1]]));
    }
    extra
}

/// The conv channel-view dspec — `SV = take(shape(W), 1)` (the
/// `(C,)` channel vector), `WV = concat(SV, (1,1,1))` (the
/// `(C,1,1,1)` weight broadcast).
fn conv_dspec() -> serde_json::Value {
    json!({
        "SV": ["take", ["shape", "W"], 1],
        "WV": ["concat", ["take", ["shape", "W"], 1], ["tuple", 1, 1, 1]]
    })
}

/// The conv scale RHS: `conv(x, W·reshape(sv,WV), b·reshape(sv,SV))`
/// (bias) or `conv(x, W·reshape(sv,WV))` (nobias).
fn conv_scale_rhs(na: usize, bias: bool) -> Pat {
    let w_fold = Pat::op("mul", vec![Pat::var("W"), conv_view(&Pat::var("sv"), "WV")]);
    if !bias {
        return conv_pat(Pat::var("x"), w_fold, None, na);
    }
    let b_fold = Pat::op("mul", vec![Pat::var("b"), conv_view(&Pat::var("sv"), "SV")]);
    conv_pat(Pat::var("x"), w_fold, Some(b_fold), na)
}

/// The conv shift RHS bias operand: `b ± reshape(cv, SV)` for
/// `add`/`sub` (bias), `reshape(cv, SV)` alone (nobias add),
/// `neg(reshape)` (nobias sub).
fn conv_shift_bias(cv_op: &str, bias: bool) -> Pat {
    let view = conv_view(&Pat::var("cv"), "SV");
    if bias {
        Pat::op(cv_op, vec![Pat::var("b"), view])
    } else if cv_op == "sub" {
        Pat::op("neg", vec![view])
    } else {
        view
    }
}

/// Build the six conv-fold laws for one `(na, bias)` cell — names,
/// conds, dspecs, derivations and law texts carried verbatim from
/// `tensor.py`'s `_make_conv_fold_rules`.
fn conv_fold_group(na: usize, bias: bool) -> Vec<Law> {
    let suffix = CONV_SUFFIXES[na];
    let arity = if bias { "" } else { "_nobias" };
    let name = |stem: &str| format!("{stem}{suffix}{arity}");
    // The scale/shift operand metavars differ per family.
    let scale_var = "sv";
    let shift_var = "cv";

    // ---- scale laws -------------------------------------------------
    let scale_law_text = "conv2d's per-output-channel scale is a right action on the output: (conv·ŝ)[o] = s[o]·out[o] = conv(x, W∘ŝ′, b∘s)[o] — convolution is linear in its filter.  The conv twin of linear_channel_scale: the fold half of the conv+BN pair.";
    let scale_l_law_text = "Scale-left twin of conv_channel_scale — the operand order the batch_norm expansion emits (``mul(γ̂, xhat)``).";
    let rev_law_text =
        "Unfold the absorbed channel scale back out of the conv (eqsat compares both forms).";
    let shift_law_text = "conv2d's per-output-channel shift folds into the bias: (conv ± ĉ)[o] = out[o] ± c[o] = conv(x, W, b ± c)[o] — the shift half of the conv+BN weight-space fold.";

    let conv_plain = |na: usize| {
        conv_pat(
            Pat::var("x"),
            Pat::var("W"),
            bias.then(|| Pat::var("b")),
            na,
        )
    };
    let mul_pair = |a: Pat, b: Pat| Pat::op("mul", vec![a, b]);
    let add_pair = |a: Pat, b: Pat| Pat::op("add", vec![a, b]);
    let sub_pair = |a: Pat, b: Pat| Pat::op("sub", vec![a, b]);

    let mut laws = Vec::with_capacity(6);

    // conv_channel_scale{,_l,_rev}{suffix}{arity}
    for (stem, left, is_rev) in [
        ("conv_channel_scale", false, false),
        ("conv_channel_scale_l", true, false),
        ("conv_channel_scale_rev", false, true),
    ] {
        let lname = name(stem);
        let (lhs, rhs) = if is_rev {
            (
                conv_scale_rhs(na, bias),
                mul_pair(conv_plain(na), Pat::var(scale_var)),
            )
        } else {
            let conv = conv_plain(na);
            let outer = if left {
                mul_pair(Pat::var(scale_var), conv)
            } else {
                mul_pair(conv, Pat::var(scale_var))
            };
            (outer, conv_scale_rhs(na, bias))
        };
        // `conv_channel_scale_rev` is the only unfold — there is no
        // `_rev_l`/`_rev_sub` in `_make_conv_fold_rules`.
        let mut cond_items = conv_channel_cond(scale_var, na, bias);
        if is_rev && let Value::Array(items) = &mut cond_items {
            items.extend(conv_rev_extra(bias));
        }
        let law_text = if is_rev {
            rev_law_text
        } else if stem.ends_with("_l") {
            scale_l_law_text
        } else {
            scale_law_text
        };
        let mut law = Law::new(&lname, lhs, rhs)
            .with_law(law_text)
            .with_cond(cond_items)
            .with_dspec(conv_dspec())
            .with_tag("symmetry")
            .with_tag("expansive");
        if is_rev {
            law = law.with_derivation(name("conv_channel_scale"));
        }
        laws.push(law);
    }

    // conv_channel_shift{,_l,_sub}{suffix}{arity}
    for (stem, op, left) in [
        ("conv_channel_shift", "add", false),
        ("conv_channel_shift_l", "add", true),
        ("conv_channel_shift_sub", "sub", false),
    ] {
        let conv = conv_plain(na);
        let bias_arg = conv_shift_bias(op, bias);
        let folded = conv_pat(Pat::var("x"), Pat::var("W"), Some(bias_arg), na);
        let lhs = if left {
            add_pair(Pat::var(shift_var), conv)
        } else {
            match op {
                "sub" => sub_pair(conv, Pat::var(shift_var)),
                _ => add_pair(conv, Pat::var(shift_var)),
            }
        };
        laws.push(
            Law::new(name(stem), lhs, folded)
                .with_law(shift_law_text)
                .with_cond(conv_channel_cond(shift_var, na, bias))
                .with_dspec(conv_dspec())
                .with_tag("symmetry")
                .with_tag("expansive"),
        );
    }
    laws
}

/// `CONV_FOLD_RULES` — the 60-law conv channel-scale/shift family
/// (5 attr-prefixes × bias arity × 6 rule shapes), opt-in like the
/// Python collection. Each law's `cond`/`dspec` trees are the
/// `cond.py`/`_make_conv_fold_rules` spelling carried as record data;
/// the `shape`/`dim`/`attr`/`take`/`concat`/`tuple`/`sub`/`mul` dspec
/// and `attr-cmp` inline-expr channels are ported.
///
/// Note: the `na == 0` variants spell `attrs {}` as the no-`__attrs__`
/// convention (kernel `attrs: None`) — a conv2d carrying an empty
/// attrs *object* is a different e-node and will not match, the same
/// representational caveat every `Pat::op` law carries.
pub fn conv_folds() -> Vec<Law> {
    let mut laws = Vec::with_capacity(60);
    for na in 0..CONV_SUFFIXES.len() {
        for bias in [true, false] {
            laws.extend(conv_fold_group(na, bias));
        }
    }
    laws
}

// ---------------------------------------------------------------------------
//  Attention-path laws — `ATTENTION_RULES` (laws/attention.py)
// ---------------------------------------------------------------------------
//
// Rotary (RoPE) and score-scale algebra over the exported elementwise
// spellings: the half-split cat and rotate-half rotary patterns, the
// rank-2 right-multiply absorb into projection weights, the
// score-scale migration (left-matmul naturality, `linear` out-scale,
// uniform-factor commutation through every view op in the spellings).
// Opt-in like the Python collection — the commutation pairs are
// closure-generating — and carried in the `attention` pack ahead of
// `SDPA_EXPAND_RULES`, mirroring the Python `ATTENTION` ruleset's
// `ATTENTION_RULES + SDPA_EXPAND_RULES` order.
//
// `rope_cat_compose` stays Python-side: its `_check_rope_cat_compose`
// hook relates six slice-attr bounds to the feature-axis extent by
// arithmetic the guard vocabulary cannot form — a common `dim % rank`
// axis across six attr metavars, `width == 2·(end−start)` against a
// *shape dim*, and `end ≥ width` sentinel reads. `attr-cmp` compares
// attr exprs and `attr-cmp-dim`/`dim-eq-attr` compare attr-vs-dim or
// dim-vs-dim, but no predicate equates a shape dim to an attr
// expression, so the half-split guard has no faithful `cond` spelling.
// Declined by name, not weakened.

/// `_slice(t, pre)` — the unstrided `slice` spelling with
/// `{pre}D`/`{pre}A`/`{pre}E` attr metavars. Strided slices carry
/// `step` and do not match — exact-keyset, like the Python pattern.
fn att_slice(t: Pat, pre: &str) -> Pat {
    let (d, a, e) = (format!("{pre}D"), format!("{pre}A"), format!("{pre}E"));
    Pat::Op(OpPat::new("slice").arg(t).attrs_fields(afields(vec![
        ("dim", af_bind(d.as_str())),
        ("start", af_bind(a.as_str())),
        ("end", af_bind(e.as_str())),
    ])))
}

/// `_rope_cat(x, C, S, p1, p2, cd)` — the half-split rotary:
/// `concat(sub(x1·C, x2·S), add(x2·C, x1·S), dim=cd)` over the two
/// contiguous slices `p1`/`p2`.
fn rope_cat(x: Pat, c: &str, s: &str, p1: &str, p2: &str, cd: &str) -> Pat {
    let x1 = att_slice(x.clone(), p1);
    let x2 = att_slice(x, p2);
    Pat::Op(
        OpPat::new("concat")
            .arg(Pat::op(
                "sub",
                vec![
                    Pat::op("mul", vec![x1.clone(), Pat::var(c)]),
                    Pat::op("mul", vec![x2.clone(), Pat::var(s)]),
                ],
            ))
            .arg(Pat::op(
                "add",
                vec![
                    Pat::op("mul", vec![x2, Pat::var(c)]),
                    Pat::op("mul", vec![x1, Pat::var(s)]),
                ],
            ))
            .attrs_fields(afields(vec![("dim", af_bind(cd))])),
    )
}

/// `_rope_rh(x, C, S, p1, p2, cd)` — the rotate-half rotary:
/// `x·C + concat(−x2, x1)·S` (`p2` namespaces the negated slice,
/// `p1` the plain one).
fn rope_rh(x: Pat, c: &str, s: &str, p1: &str, p2: &str, cd: &str) -> Pat {
    Pat::op(
        "add",
        vec![
            Pat::op("mul", vec![x.clone(), Pat::var(c)]),
            Pat::op(
                "mul",
                vec![
                    Pat::Op(
                        OpPat::new("concat")
                            .arg(Pat::op("neg", vec![att_slice(x.clone(), p2)]))
                            .arg(att_slice(x, p1))
                            .attrs_fields(afields(vec![("dim", af_bind(cd))])),
                    ),
                    Pat::var(s),
                ],
            ),
        ],
    )
}

/// `_check_rope_scale` — `a` may slide through a rotary: scalar, or
/// a last-dim-1 factor broadcasting into `x` (per-row pair-constant).
/// The Python hook's `_shape_of`/`_broadcast` reads are the
/// `scalar`/`dim-eq-const`/`bcast-into` shape channels; a `None`
/// last dim declines exactly like Python's `sa[-1] != 1`.
fn cond_rope_scale() -> Value {
    json!([
        "or",
        ["scalar", "a"],
        [
            "and",
            ["dim-eq-const", "a", -1, 1],
            ["bcast-into", "a", "x"]
        ]
    ])
}

/// `_check_left_scale` — `s` slides between `mul(A@B, s)` and
/// `matmul(A·s, B)`: scalar or last-dim-1, broadcasting into the
/// matmul output, which must be well-formed (`mm-shape-ok` — the
/// Python hook's `_matmul_shape(...) is None` veto, kept as a
/// top-level conjunct because the scalar arm skips the broadcast
/// read but not the shape check).
fn cond_left_scale() -> Value {
    json!([
        "and",
        ["shaped", "s"],
        ["mm-shape-ok", "A", "B"],
        [
            "or",
            ["scalar", "s"],
            [
                "and",
                ["dim-eq-const", "s", -1, 1],
                ["bcast-into", "s", ["mm-out", "A", "B"]]
            ]
        ]
    ])
}

/// `_check_linear_mm_absorb` — `linear(x,W)@R` folds only for
/// rank-2 `W`/`R` whose contraction dims agree (`None` dims are
/// wildcards — `dim-compat`, not `dim-eq`).
fn cond_mm_absorb() -> Value {
    json!([
        "and",
        ["rank", "W", "==", 2],
        ["rank", "R", "==", 2],
        ["dim-compat", "R", 0, "W", 0]
    ])
}

/// `_check_linear_mm_absorb_bias` — the biased absorb adds the
/// `b@R` matvec: `b` rank-1 with `b[0]` `None`-wildcard-equal to
/// `W[0]`.
fn cond_mm_absorb_bias() -> Value {
    json!([
        "and",
        ["rank", "W", "==", 2],
        ["rank", "R", "==", 2],
        ["dim-compat", "R", 0, "W", 0],
        ["rank", "b", "==", 1],
        ["dim-compat", "b", 0, "W", 0]
    ])
}

/// `_is_uniform` — scalar or all-dims-1 broadcast, an
/// elementwise-constant factor (`uniform` reads the same tuple
/// check: `None` dims are not `1`).
fn cond_uniform(var: &str) -> Value {
    json!(["uniform", var])
}

/// The 24 `_VIEW_SCALE_RULES` — `mul(view(t), a) ↔ view(mul(t), a)`
/// for every view op in the attention spellings, both directions,
/// plus the `concat`/`stack` two-operand distribution. Uniform
/// factors only — the Python `_check_uniform` hook.
fn view_scale_rules() -> Vec<Law> {
    /// One unary view op: `(rule-name infix, op name, attrs fields)`.
    type ViewSpec = (&'static str, &'static str, Vec<(&'static str, AttrField)>);
    let views: Vec<ViewSpec> = vec![
        (
            "transpose",
            "transpose",
            vec![("dim0", af_bind("D0")), ("dim1", af_bind("D1"))],
        ),
        ("reshape", "reshape", vec![("shape", af_bind("SH"))]),
        (
            "slice",
            "slice",
            vec![
                ("dim", af_bind("D")),
                ("start", af_bind("A")),
                ("end", af_bind("E")),
            ],
        ),
        (
            "slice_step",
            "slice",
            vec![
                ("dim", af_bind("D")),
                ("start", af_bind("A")),
                ("end", af_bind("E")),
                ("step", af_bind("P")),
            ],
        ),
        (
            "unbind",
            "unbind",
            vec![("dim", af_bind("D")), ("index", af_bind("I"))],
        ),
        (
            "chunk",
            "chunk",
            vec![
                ("chunks", af_bind("N")),
                ("dim", af_bind("D")),
                ("index", af_bind("I")),
            ],
        ),
        ("flatten", "flatten", vec![("start_dim", af_bind("F0"))]),
        (
            "flatten2",
            "flatten",
            vec![("start_dim", af_bind("F0")), ("end_dim", af_bind("F1"))],
        ),
        ("unsqueeze", "unsqueeze", vec![("dim", af_bind("D"))]),
        ("squeeze", "squeeze", vec![("dim", af_bind("D"))]),
    ];
    let view = |op: &str, fields: &[(&'static str, AttrField)], t: Pat| {
        Pat::Op(OpPat::new(op).arg(t).attrs_fields(afields(fields.to_vec())))
    };
    let mut rules = Vec::with_capacity(24);
    for &(name, op, ref fields) in &views {
        let cond = cond_uniform("a");
        rules.push(
            Law::new(
                format!("scale_in_{name}"),
                Pat::op(
                    "mul",
                    vec![view(op, fields, Pat::var("t")), Pat::var("a")],
                ),
                view(
                    op,
                    fields,
                    Pat::op("mul", vec![Pat::var("t"), Pat::var("a")]),
                ),
            )
            .with_law(format!(
                "Uniform factor commutes into {name}: a·view(t) = view(a·t) — the elementwise-constant scale rides through any relayout."
            ))
            .with_cond(cond.clone())
            .with_tag("attention")
            .with_tag("expansive"),
        );
        rules.push(
            Law::new(
                format!("scale_out_{name}"),
                view(
                    op,
                    fields,
                    Pat::op("mul", vec![Pat::var("t"), Pat::var("a")]),
                ),
                Pat::op(
                    "mul",
                    vec![view(op, fields, Pat::var("t")), Pat::var("a")],
                ),
            )
            .with_law(format!(
                "Reverse uniform commutation out of {name} — factors a shared scale back to the view's output."
            ))
            .with_cond(cond)
            .with_tag("attention")
            .with_tag("expansive"),
        );
    }
    for name in ["concat", "stack"] {
        let shared = Pat::Op(
            OpPat::new(name)
                .arg(Pat::op("mul", vec![Pat::var("t1"), Pat::var("a")]))
                .arg(Pat::op("mul", vec![Pat::var("t2"), Pat::var("a")]))
                .attrs_fields(afields(vec![("dim", af_bind("D"))])),
        );
        let join = Pat::Op(
            OpPat::new(name)
                .arg(Pat::var("t1"))
                .arg(Pat::var("t2"))
                .attrs_fields(afields(vec![("dim", af_bind("D"))])),
        );
        let cond = cond_uniform("a");
        rules.push(
            Law::new(
                format!("scale_in_{name}"),
                Pat::op("mul", vec![join.clone(), Pat::var("a")]),
                shared.clone(),
            )
            .with_law(format!(
                "Uniform factor distributes into {name}: a·cat(u,v) = cat(a·u, a·v)."
            ))
            .with_cond(cond.clone())
            .with_tag("attention")
            .with_tag("expansive"),
        );
        rules.push(
            Law::new(
                format!("scale_out_{name}"),
                shared,
                Pat::op("mul", vec![join, Pat::var("a")]),
            )
            .with_law(format!("Reverse: shared uniform factors out of {name}."))
            .with_cond(cond)
            .with_tag("attention")
            .with_tag("expansive"),
        );
    }
    rules
}

/// `ATTENTION_RULES` minus `rope_cat_compose` — the 36 portable
/// members in Python collection order: the four rotary scale
/// commutations, the four right-multiply absorbs, the four
/// score-scale migrations, then the 24 view-scale rules.
/// `rope_cat_compose` is declined by name (see the section header).
fn attention_path_rules() -> Vec<Law> {
    let fus = |l: Law| l.with_tag("attention").with_tag("fusion");
    let exp = |l: Law| l.with_tag("attention").with_tag("expansive");
    let mut laws = vec![
        // Rotary is x-linear — both spellings × both directions.
        exp(
            Law::new(
                "rope_cat_scale_out",
                rope_cat(
                    Pat::op("mul", vec![Pat::var("x"), Pat::var("a")]),
                    "C",
                    "S",
                    "I1",
                    "I2",
                    "CD",
                ),
                Pat::op(
                    "mul",
                    vec![rope_cat(Pat::var("x"), "C", "S", "I1", "I2", "CD"), Pat::var("a")],
                ),
            )
            .with_law(
                "Rotary is linear in x: rope(a·x) = a·rope(x) for a pair-constant factor — a scalar or per-row broadcast rides outside the whole elementwise chain.",
            )
            .with_cond(cond_rope_scale()),
        ),
        exp(
            Law::new(
                "rope_cat_scale_in",
                Pat::op(
                    "mul",
                    vec![rope_cat(Pat::var("x"), "C", "S", "I1", "I2", "CD"), Pat::var("a")],
                ),
                rope_cat(
                    Pat::op("mul", vec![Pat::var("x"), Pat::var("a")]),
                    "C",
                    "S",
                    "I1",
                    "I2",
                    "CD",
                ),
            )
            .with_law(
                "Reverse: push a uniform factor inside the rotary so the view/weight laws can route it into a projection.",
            )
            .with_cond(cond_rope_scale()),
        ),
        exp(
            Law::new(
                "rope_rh_scale_out",
                rope_rh(
                    Pat::op("mul", vec![Pat::var("x"), Pat::var("a")]),
                    "C",
                    "S",
                    "I1",
                    "I2",
                    "CD",
                ),
                Pat::op(
                    "mul",
                    vec![rope_rh(Pat::var("x"), "C", "S", "I1", "I2", "CD"), Pat::var("a")],
                ),
            )
            .with_law(
                "x·cos + rotate_half(x)·sin is also x-linear: a uniform factor rides out of the HF spelling.",
            )
            .with_cond(cond_rope_scale()),
        ),
        exp(
            Law::new(
                "rope_rh_scale_in",
                Pat::op(
                    "mul",
                    vec![rope_rh(Pat::var("x"), "C", "S", "I1", "I2", "CD"), Pat::var("a")],
                ),
                rope_rh(
                    Pat::op("mul", vec![Pat::var("x"), Pat::var("a")]),
                    "C",
                    "S",
                    "I1",
                    "I2",
                    "CD",
                ),
            )
            .with_law(
                "Reverse: push a uniform factor inside the rotate-half spelling.",
            )
            .with_cond(cond_rope_scale()),
        ),
        // Right-multiply absorb — `(x@Wᵀ)@R = x@(RᵀW)ᵀ = linear(x, RᵀW)`.
        fus(
            Law::new(
                "linear_mm_absorb",
                Pat::op(
                    "matmul",
                    vec![Pat::op("linear", vec![Pat::var("x"), Pat::var("W")]), Pat::var("R")],
                ),
                Pat::op(
                    "linear",
                    vec![
                        Pat::var("x"),
                        Pat::op(
                            "matmul",
                            vec![
                                Pat::Op(
                                    OpPat::new("transpose")
                                        .arg(Pat::var("R"))
                                        .attrs_lit(json!({"dim0": -2, "dim1": -1})),
                                ),
                                Pat::var("W"),
                            ],
                        ),
                    ],
                ),
            )
            .with_law(
                "(xWᵀ)R = x(RᵀW)ᵀ — a constant right-multiply absorbs into the projection weight (rotary-by-matmul included).",
            )
            .with_cond(cond_mm_absorb()),
        ),
        exp(
            Law::new(
                "linear_mm_absorb_rev",
                Pat::op(
                    "linear",
                    vec![
                        Pat::var("x"),
                        Pat::op(
                            "matmul",
                            vec![
                                Pat::Op(
                                    OpPat::new("transpose")
                                        .arg(Pat::var("R"))
                                        .attrs_lit(json!({"dim0": -2, "dim1": -1})),
                                ),
                                Pat::var("W"),
                            ],
                        ),
                    ],
                ),
                Pat::op(
                    "matmul",
                    vec![Pat::op("linear", vec![Pat::var("x"), Pat::var("W")]), Pat::var("R")],
                ),
            )
            .with_law(
                "Reverse absorb — eqsat weighs the folded weight against a runtime right-multiply.",
            )
            .with_cond(cond_mm_absorb()),
        ),
        fus(
            Law::new(
                "linear_mm_absorb_bias",
                Pat::op(
                    "matmul",
                    vec![
                        Pat::op("linear", vec![Pat::var("x"), Pat::var("W"), Pat::var("b")]),
                        Pat::var("R"),
                    ],
                ),
                Pat::op(
                    "linear",
                    vec![
                        Pat::var("x"),
                        Pat::op(
                            "matmul",
                            vec![
                                Pat::Op(
                                    OpPat::new("transpose")
                                        .arg(Pat::var("R"))
                                        .attrs_lit(json!({"dim0": -2, "dim1": -1})),
                                ),
                                Pat::var("W"),
                            ],
                        ),
                        Pat::op("matmul", vec![Pat::var("b"), Pat::var("R")]),
                    ],
                ),
            )
            .with_law(
                "Affine right-multiply absorb: (xWᵀ + b)R = x(RᵀW)ᵀ + bR — the fused bias is the matvec b@R, both products compile-time.",
            )
            .with_cond(cond_mm_absorb_bias()),
        ),
        exp(
            Law::new(
                "linear_mm_absorb_bias_rev",
                Pat::op(
                    "linear",
                    vec![
                        Pat::var("x"),
                        Pat::op(
                            "matmul",
                            vec![
                                Pat::Op(
                                    OpPat::new("transpose")
                                        .arg(Pat::var("R"))
                                        .attrs_lit(json!({"dim0": -2, "dim1": -1})),
                                ),
                                Pat::var("W"),
                            ],
                        ),
                        Pat::op("matmul", vec![Pat::var("b"), Pat::var("R")]),
                    ],
                ),
                Pat::op(
                    "matmul",
                    vec![
                        Pat::op("linear", vec![Pat::var("x"), Pat::var("W"), Pat::var("b")]),
                        Pat::var("R"),
                    ],
                ),
            )
            .with_law("Reverse biased absorb (eqsat weighs fused vs split).")
            .with_cond(cond_mm_absorb_bias()),
        ),
        // Score-scale migration.
        exp(
            Law::new(
                "naturality_scalar_left",
                Pat::op(
                    "matmul",
                    vec![Pat::op("mul", vec![Pat::var("A"), Pat::var("s")]), Pat::var("B")],
                ),
                Pat::op(
                    "mul",
                    vec![Pat::op("matmul", vec![Pat::var("A"), Pat::var("B")]), Pat::var("s")],
                ),
            )
            .with_law(
                "Row-uniform scale slides out of a matmul's left operand: (s·A)@B = s·(A@B) — the left mirror of naturality_scalar.",
            )
            .with_cond(cond_left_scale()),
        ),
        exp(
            Law::new(
                "naturality_scalar_left_rev",
                Pat::op(
                    "mul",
                    vec![Pat::op("matmul", vec![Pat::var("A"), Pat::var("B")]), Pat::var("s")],
                ),
                Pat::op(
                    "matmul",
                    vec![Pat::op("mul", vec![Pat::var("A"), Pat::var("s")]), Pat::var("B")],
                ),
            )
            .with_law(
                "Reverse: s·(A@B) = (s·A)@B — pushes a score scale into the left factor, where the view/weight laws route it to Wq.",
            )
            .with_cond(cond_left_scale()),
        ),
        fus(
            Law::new(
                "linear_out_scale",
                Pat::op(
                    "mul",
                    vec![Pat::op("linear", vec![Pat::var("x"), Pat::var("W")]), Pat::var("s")],
                ),
                Pat::op(
                    "linear",
                    vec![Pat::var("x"), Pat::op("mul", vec![Pat::var("W"), Pat::var("s")])],
                ),
            )
            .with_law(
                "Uniform scale folds into the weight: s·(xWᵀ) = x(sW)ᵀ — the runtime mul becomes a compile-time weight mul.",
            )
            .with_cond(cond_uniform("s")),
        ),
        exp(
            Law::new(
                "linear_out_scale_rev",
                Pat::op(
                    "linear",
                    vec![Pat::var("x"), Pat::op("mul", vec![Pat::var("W"), Pat::var("s")])],
                ),
                Pat::op(
                    "mul",
                    vec![Pat::op("linear", vec![Pat::var("x"), Pat::var("W")]), Pat::var("s")],
                ),
            )
            .with_law(
                "Reverse uniform fold — eqsat compares weight-folded vs runtime-scaled forms.",
            )
            .with_cond(cond_uniform("s")),
        ),
    ];
    laws.extend(view_scale_rules());
    laws
}

// ---------------------------------------------------------------------------
//  Scan monoids — `SCAN_LAWS` / `SCAN_DIAG_LAWS` (laws/scan.py)
// ---------------------------------------------------------------------------
//
// The affine-map and diagonal-affine scan carriers: a recurrence step
// `h ↦ A·h + x` (dense) or `h ↦ a ⊙ h + x` (diagonal) is a monoid
// element; `aff`/`aff_compose`/`apply` and `aff_diag`/`affd_compose`/
// `applyd` are the carrier-domain spelling the balanced-tree (Blelloch)
// reassociation reaches through associativity alone. Opt-in like the
// Python collections — they sit in `CARRIER_SEARCH`, never `DEFAULT`.
// Every guard the family carries fits the ported vocabulary:
// `op-in`/`leaf`/`not`/`const` conds and the `bcast` dspec channel.

/// `_COND_AFFD_STATE` — the `h`-is-state side condition shared by the
/// four `affd_lift*` operand-position variants: `h` must be a previous
/// step's `add`/`sub` spine, an already lifted application, or a leaf
/// (per-step vectors are NOT states — an economy guard, not a
/// soundness one: the rewrite is valid for any `h`).
fn affd_state_cond() -> Value {
    json!([
        "or",
        ["op-in", "h", ["add", "sub", "apply", "applyd"]],
        ["leaf", "h"]
    ])
}

/// `_COND_AFFD_UNIT_STATE` — the unit-lift variant: same state shape,
/// plus a `Const` exclusion (a scalar offset is not an accumulating
/// state).
fn affd_unit_cond() -> Value {
    json!([
        "or",
        ["op-in", "h", ["add", "sub", "apply", "applyd"]],
        ["and", ["leaf", "h"], ["not", ["const", "h"]]]
    ])
}

/// `_DSPEC_AFFD_UNIT` — `US` := broadcast(shape(h), shape(x)): the
/// unit diagonal materialises at the add's output shape (all ones);
/// `bcast` declines when the result is not concrete, the old hook's
/// veto verbatim.
fn affd_unit_dspec() -> Value {
    json!({"US": ["bcast", "h", "x"]})
}

/// `_AFFD_UNIT` — `expand(Const(1.0), shape=US)`: the unit diagonal as
/// a shared pattern fragment, spelled as a Const broadcast so the
/// carrier leaf reports the state's `(d,)` shape like every other
/// `aff_diag` leaf (the batched executor stacks leaf a-parts).
fn affd_unit() -> Pat {
    Pat::Op(
        OpPat::new("expand")
            .arg(Pat::constant(json!(1.0)))
            .attrs_fields(afields(vec![("shape", af_bind("US"))])),
    )
}

/// `applyd(aff_diag(a, x), h)` — the diagonal-affine application every
/// `affd_lift*` variant rewrites to (`a` is the step's diagonal part).
fn affd_apply(a: Pat) -> Pat {
    Pat::op(
        "applyd",
        vec![Pat::op("aff_diag", vec![a, Pat::var("x")]), Pat::var("h")],
    )
}

/// `applyd(affd_compose(aff_diag(a, x), f), h)` — the composed-step
/// RHS every `affd_lift*_step*` variant rewrites to.
fn affd_compose_apply(a: Pat) -> Pat {
    Pat::op(
        "applyd",
        vec![
            Pat::op(
                "affd_compose",
                vec![Pat::op("aff_diag", vec![a, Pat::var("x")]), Pat::var("f")],
            ),
            Pat::var("h"),
        ],
    )
}

/// The six `SCAN_LAWS` — the dense affine monoid, Python order.
fn scan_laws() -> Vec<Law> {
    let tag = |l: Law| l.with_tag("scan");
    vec![
        // aff_lift: add(matmul(A, h), x) → apply(aff(A, x), h)
        tag(Law::new(
            "aff_lift",
            Pat::op(
                "add",
                vec![
                    Pat::op("matmul", vec![Pat::var("A"), Pat::var("h")]),
                    Pat::var("x"),
                ],
            ),
            Pat::op(
                "apply",
                vec![
                    Pat::op("aff", vec![Pat::var("A"), Pat::var("x")]),
                    Pat::var("h"),
                ],
            ),
        )
        .with_law("recurrence step is affine-map application")),
        // aff_lift_step: add(matmul(A, apply(f, h)), x) →
        //   apply(aff_compose(aff(A, x), f), h)
        tag(Law::new(
            "aff_lift_step",
            Pat::op(
                "add",
                vec![
                    Pat::op(
                        "matmul",
                        vec![
                            Pat::var("A"),
                            Pat::op("apply", vec![Pat::var("f"), Pat::var("h")]),
                        ],
                    ),
                    Pat::var("x"),
                ],
            ),
            Pat::op(
                "apply",
                vec![
                    Pat::op(
                        "aff_compose",
                        vec![
                            Pat::op("aff", vec![Pat::var("A"), Pat::var("x")]),
                            Pat::var("f"),
                        ],
                    ),
                    Pat::var("h"),
                ],
            ),
        )
        .with_law("compose step with the preceding map")),
        // aff_unlift: apply(aff(A, x), h) → add(matmul(A, h), x)
        tag(Law::new(
            "aff_unlift",
            Pat::op(
                "apply",
                vec![
                    Pat::op("aff", vec![Pat::var("A"), Pat::var("x")]),
                    Pat::var("h"),
                ],
            ),
            Pat::op(
                "add",
                vec![
                    Pat::op("matmul", vec![Pat::var("A"), Pat::var("h")]),
                    Pat::var("x"),
                ],
            ),
        )
        .with_law("affine application unfolds")),
        // aff_compose_unfold: apply(aff_compose(f, g), h) →
        //   apply(f, apply(g, h))
        tag(Law::new(
            "aff_compose_unfold",
            Pat::op(
                "apply",
                vec![
                    Pat::op("aff_compose", vec![Pat::var("f"), Pat::var("g")]),
                    Pat::var("h"),
                ],
            ),
            Pat::op(
                "apply",
                vec![
                    Pat::var("f"),
                    Pat::op("apply", vec![Pat::var("g"), Pat::var("h")]),
                ],
            ),
        )
        .with_law("composition is sequential application")),
        // aff_assoc: aff_compose(aff_compose(f, g), h) →
        //   aff_compose(f, aff_compose(g, h))
        tag(Law::new(
            "aff_assoc",
            Pat::op(
                "aff_compose",
                vec![
                    Pat::op("aff_compose", vec![Pat::var("f"), Pat::var("g")]),
                    Pat::var("h"),
                ],
            ),
            Pat::op(
                "aff_compose",
                vec![
                    Pat::var("f"),
                    Pat::op("aff_compose", vec![Pat::var("g"), Pat::var("h")]),
                ],
            ),
        )
        .with_law("affine composition is associative")),
        // aff_assoc_rev: the reverse bracketing.
        tag(Law::new(
            "aff_assoc_rev",
            Pat::op(
                "aff_compose",
                vec![
                    Pat::var("f"),
                    Pat::op("aff_compose", vec![Pat::var("g"), Pat::var("h")]),
                ],
            ),
            Pat::op(
                "aff_compose",
                vec![
                    Pat::op("aff_compose", vec![Pat::var("f"), Pat::var("g")]),
                    Pat::var("h"),
                ],
            ),
        )
        .with_law("affine composition is associative")),
    ]
}

/// The sixteen `SCAN_DIAG_LAWS` — the diagonal-affine monoid for
/// elementwise / Mamba-faithful SSM steps, Python order: the four
/// `affd_lift` operand positions, the four `affd_lift_step`
/// positions, the two unit lifts, the two unit steps, then the
/// unfold/compose/assoc tail.
fn scan_diag_laws() -> Vec<Law> {
    let tag = |l: Law| l.with_tag("scan");
    let state = affd_state_cond();
    let unit = affd_unit_cond();
    let unit_dspec = affd_unit_dspec();
    let lift_law =
        "diagonal recurrence step is diagonal-affine application: a⊙h + x = (aff_diag(a,x))(h)";
    let lift_swap_law =
        "mul-order variant of affd_lift (canonicalised terms put the state operand first)";
    let lift_post_law = "add-order variant of affd_lift (state-mul in the second add slot)";
    let lift_post_swap_law = "remaining operand position of affd_lift";
    vec![
        // -- the four operand positions of the mul-form step --------
        tag(
            Law::new(
                "affd_lift",
                Pat::op(
                    "add",
                    vec![
                        Pat::op("mul", vec![Pat::var("a"), Pat::var("h")]),
                        Pat::var("x"),
                    ],
                ),
                affd_apply(Pat::var("a")),
            )
            .with_law(lift_law)
            .with_cond(state.clone()),
        ),
        tag(
            Law::new(
                "affd_lift_swap",
                Pat::op(
                    "add",
                    vec![
                        Pat::op("mul", vec![Pat::var("h"), Pat::var("a")]),
                        Pat::var("x"),
                    ],
                ),
                affd_apply(Pat::var("a")),
            )
            .with_law(lift_swap_law)
            .with_cond(state.clone()),
        ),
        tag(
            Law::new(
                "affd_lift_post",
                Pat::op(
                    "add",
                    vec![
                        Pat::var("x"),
                        Pat::op("mul", vec![Pat::var("a"), Pat::var("h")]),
                    ],
                ),
                affd_apply(Pat::var("a")),
            )
            .with_law(lift_post_law)
            .with_cond(state.clone()),
        ),
        tag(
            Law::new(
                "affd_lift_post_swap",
                Pat::op(
                    "add",
                    vec![
                        Pat::var("x"),
                        Pat::op("mul", vec![Pat::var("h"), Pat::var("a")]),
                    ],
                ),
                affd_apply(Pat::var("a")),
            )
            .with_law(lift_post_swap_law)
            .with_cond(state),
        ),
        // -- the four operand positions of the composed step --------
        tag(
            Law::new(
                "affd_lift_step",
                Pat::op(
                    "add",
                    vec![
                        Pat::op(
                            "mul",
                            vec![
                                Pat::var("a"),
                                Pat::op("applyd", vec![Pat::var("f"), Pat::var("h")]),
                            ],
                        ),
                        Pat::var("x"),
                    ],
                ),
                affd_compose_apply(Pat::var("a")),
            )
            .with_law("compose step with the preceding map"),
        ),
        tag(
            Law::new(
                "affd_lift_step_swap",
                Pat::op(
                    "add",
                    vec![
                        Pat::op(
                            "mul",
                            vec![
                                Pat::op("applyd", vec![Pat::var("f"), Pat::var("h")]),
                                Pat::var("a"),
                            ],
                        ),
                        Pat::var("x"),
                    ],
                ),
                affd_compose_apply(Pat::var("a")),
            )
            .with_law("mul-order variant of affd_lift_step"),
        ),
        tag(
            Law::new(
                "affd_lift_step_post",
                Pat::op(
                    "add",
                    vec![
                        Pat::var("x"),
                        Pat::op(
                            "mul",
                            vec![
                                Pat::var("a"),
                                Pat::op("applyd", vec![Pat::var("f"), Pat::var("h")]),
                            ],
                        ),
                    ],
                ),
                affd_compose_apply(Pat::var("a")),
            )
            .with_law("add-order variant of affd_lift_step"),
        ),
        tag(
            Law::new(
                "affd_lift_step_post_swap",
                Pat::op(
                    "add",
                    vec![
                        Pat::var("x"),
                        Pat::op(
                            "mul",
                            vec![
                                Pat::op("applyd", vec![Pat::var("f"), Pat::var("h")]),
                                Pat::var("a"),
                            ],
                        ),
                    ],
                ),
                affd_compose_apply(Pat::var("a")),
            )
            .with_law("remaining operand position of affd_lift_step"),
        ),
        // -- the unit lift: pure accumulation h + x ≡ a ≡ 1 ---------
        tag(
            Law::new(
                "affd_lift_unit",
                Pat::op("add", vec![Pat::var("h"), Pat::var("x")]),
                affd_apply(affd_unit()),
            )
            .with_law(
                "Unit introduction: pure accumulation h + x IS the diagonal affine map with a ≡ 1 — applyd(aff_diag(1, x), h).",
            )
            .with_cond(unit.clone())
            .with_dspec(unit_dspec.clone()),
        ),
        tag(
            Law::new(
                "affd_lift_unit_post",
                Pat::op("add", vec![Pat::var("x"), Pat::var("h")]),
                affd_apply(affd_unit()),
            )
            .with_law(
                "add-order variant of affd_lift_unit (state operand in the second add slot)",
            )
            .with_cond(unit.clone())
            .with_dspec(unit_dspec.clone()),
        ),
        tag(
            Law::new(
                "affd_lift_unit_step",
                Pat::op(
                    "add",
                    vec![
                        Pat::op("applyd", vec![Pat::var("f"), Pat::var("h")]),
                        Pat::var("x"),
                    ],
                ),
                affd_compose_apply(affd_unit()),
            )
            .with_law(
                "compose a unit (pure-accumulation) step with the preceding map — the h ↦ h + x analogue of affd_lift_step",
            )
            .with_dspec(unit_dspec.clone()),
        ),
        tag(
            Law::new(
                "affd_lift_unit_step_post",
                Pat::op(
                    "add",
                    vec![
                        Pat::var("x"),
                        Pat::op("applyd", vec![Pat::var("f"), Pat::var("h")]),
                    ],
                ),
                affd_compose_apply(affd_unit()),
            )
            .with_law("add-order variant of affd_lift_unit_step")
            .with_dspec(unit_dspec),
        ),
        // -- the unfold / compose / assoc tail ----------------------
        // affd_unlift: applyd(aff_diag(a, x), h) → add(mul(a, h), x)
        tag(Law::new(
            "affd_unlift",
            Pat::op(
                "applyd",
                vec![
                    Pat::op("aff_diag", vec![Pat::var("a"), Pat::var("x")]),
                    Pat::var("h"),
                ],
            ),
            Pat::op(
                "add",
                vec![
                    Pat::op("mul", vec![Pat::var("a"), Pat::var("h")]),
                    Pat::var("x"),
                ],
            ),
        )
        .with_law("diagonal-affine application unfolds")),
        // affd_compose_unfold: applyd(affd_compose(f, g), h) →
        //   applyd(f, applyd(g, h))
        tag(Law::new(
            "affd_compose_unfold",
            Pat::op(
                "applyd",
                vec![
                    Pat::op("affd_compose", vec![Pat::var("f"), Pat::var("g")]),
                    Pat::var("h"),
                ],
            ),
            Pat::op(
                "applyd",
                vec![
                    Pat::var("f"),
                    Pat::op("applyd", vec![Pat::var("g"), Pat::var("h")]),
                ],
            ),
        )
        .with_law("composition is sequential application")),
        // affd_assoc / affd_assoc_rev: diagonal-affine composition is
        // associative.
        tag(Law::new(
            "affd_assoc",
            Pat::op(
                "affd_compose",
                vec![
                    Pat::op("affd_compose", vec![Pat::var("f"), Pat::var("g")]),
                    Pat::var("h"),
                ],
            ),
            Pat::op(
                "affd_compose",
                vec![
                    Pat::var("f"),
                    Pat::op("affd_compose", vec![Pat::var("g"), Pat::var("h")]),
                ],
            ),
        )
        .with_law("diagonal-affine composition is associative")),
        tag(Law::new(
            "affd_assoc_rev",
            Pat::op(
                "affd_compose",
                vec![
                    Pat::var("f"),
                    Pat::op("affd_compose", vec![Pat::var("g"), Pat::var("h")]),
                ],
            ),
            Pat::op(
                "affd_compose",
                vec![
                    Pat::op("affd_compose", vec![Pat::var("f"), Pat::var("g")]),
                    Pat::var("h"),
                ],
            ),
        )
        .with_law("diagonal-affine composition is associative")),
    ]
}

// ---------------------------------------------------------------------------
//  Layout laws — `LAYOUT_RULES` (laws/layout.py)
// ---------------------------------------------------------------------------
//
// The transpose/layout migration family: `transpose` is a pure
// re-layout (a stride permutation lowering to a view), so these laws
// let the e-graph hold both the materialised-transpose and the
// migrated/fused form — pointwise naturality (push/pull over every
// unary and same-rank binary pointwise op, both the canonical
// `dim0`/`dim1` spelling and the bare `t()` form), the involution
// `T∘T = id`, the product transpose `(A@B).mT = B.mT @ A.mT` and its
// reverse `P.mT @ Q.mT = (Q@P).mT`, and the NT bridge
// `matmul(x, W.mT) ≡ linear(x, W)`. Opt-in like the Python surface —
// `ALL_RULES_WITH_LAYOUT`/`WITH_LAYOUT`, never `DEFAULT` (measured
// closure-cost blowup, tensor.py's ALL_RULES note). All 77 members
// carry the `layout` tag; every procedural `check=` hook translates
// to the ported `cond` vocabulary (`axes-*`/`rank`/`rank-eq`/
// `dim-compat`/`mm-out` shape specs) — no declines in this pack.

/// The twelve `POINTWISE_UNARY_OPS` (Python `sorted` order).
const LAYOUT_UNARY_OPS: &[&str] = &[
    "abs", "exp", "gelu", "log", "neg", "relu", "rsqrt", "sigmoid", "silu", "sqrt", "square",
    "tanh",
];

/// The four `POINTWISE_BINARY_OPS` (Python `sorted` order).
const LAYOUT_BINARY_OPS: &[&str] = &["add", "div", "mul", "sub"];

/// `_t(term, dimmed, k0, k1)` — a `transpose` pattern over *term*.
/// The dimmed spelling binds `dim0`/`dim1` to the attr metavariables
/// *k0*/*k1* (the names every transpose in one rule shares); the bare
/// spelling is the attr-less `t()` form — the implicit last-two swap.
fn layout_t(term: Pat, dimmed: bool, k0: &str, k1: &str) -> Pat {
    if dimmed {
        Pat::Op(
            OpPat::new("transpose")
                .arg(term)
                .attrs_fields(afields(vec![("dim0", af_bind(k0)), ("dim1", af_bind(k1))])),
        )
    } else {
        Pat::Op(OpPat::new("transpose").arg(term))
    }
}

/// `_mt(term)` — the canonical last-two-axis swap mint: the `x.mT`
/// transpose with literal `dim0=-2, dim1=-1` attrs.
fn layout_mt(term: Pat) -> Pat {
    Pat::Op(
        OpPat::new("transpose")
            .arg(term)
            .attrs_lit(json!({"dim0": -2, "dim1": -1})),
    )
}

/// `_check_commute_unary` — `T(f(x)) = f(T(x))`: the bound `x` must
/// carry a tuple shape whose transpose axis pair is real and
/// distinct (rank ≥ 2, `d0 != d1` after normalisation; a self-pair
/// is a no-op view, not a layout worth holding).
fn cond_commute_unary() -> Value {
    json!(["axes-distinct", "x", "D0", "D1"])
}

/// `_check_commute_binary` — `T(g(a, b)) = g(T(a), T(b))`: equal-rank
/// pointwise operands only (broadcasting is positional — a common
/// permutation preserves the broadcast dim-for-dim).
fn cond_commute_binary() -> Value {
    json!([
        "and",
        ["rank-eq", "a", "b"],
        ["axes-distinct", "a", "D0", "D1"]
    ])
}

/// `_check_involution` — `T(T(x)) = x`: both transposes swap the SAME
/// axis pair (`axes-eq` requires both pairs distinct and equal as
/// sets).
fn cond_involution() -> Value {
    json!(["axes-eq", "x", "I0", "I1", "O0", "O1"])
}

/// `_check_transpose_matmul` — `(A@B).mT = B.mT @ A.mT`: rank-≥2
/// operands whose `mm-out` spec resolves (the batch broadcast and
/// contraction checks `_matmul_shape` folds in), with the outer
/// transpose the output's last-two swap.
fn cond_transpose_matmul() -> Value {
    json!([
        "and",
        ["rank", "A", ">=", 2],
        ["rank", "B", ">=", 2],
        ["axes-last2", ["mm-out", "A", "B"], "D0", "D1"]
    ])
}

/// `_check_mm_transposes` — `P.mT @ Q.mT = (Q@P).mT`: both inner
/// transposes are the last-two swap of their own operand (rank < 2
/// declines inside `axes-pair`).
fn cond_mm_transposes() -> Value {
    json!([
        "and",
        ["axes-last2", "P", "P0", "P1"],
        ["axes-last2", "Q", "Q0", "Q1"]
    ])
}

/// `_linear_shapes_ok` — the `linear` contract shared by the NT
/// bridge pair: `W` rank-2 `(out, in)`, `x` rank ≥ 1, and `x`'s last
/// dim feeds `in` (`dim-compat` — unknown dims wildcard, matching
/// the hook's int-only comparison).
fn cond_linear_shapes() -> Value {
    json!([
        "and",
        ["rank", "W", "==", 2],
        ["rank", "x", ">=", 1],
        ["dim-compat", "x", -1, "W", 1]
    ])
}

/// `_check_mm_t_is_linear` — `matmul(x, W.mT) ≡ linear(x, W)`: the
/// linear contract plus the bound transpose being W's full swap
/// (rank-2 makes the last-two pair the only real one).
fn cond_mm_t_is_linear() -> Value {
    json!([
        "and",
        ["rank", "W", "==", 2],
        ["rank", "x", ">=", 1],
        ["dim-compat", "x", -1, "W", 1],
        ["axes-last2", "W", "D0", "D1"]
    ])
}

/// `_commute_rules()` — the 64 pointwise commutation laws: per op,
/// per spelling (dimmed then bare), push then pull — Python order
/// (48 unary over `POINTWISE_UNARY_OPS`, 16 binary over
/// `POINTWISE_BINARY_OPS`).
fn layout_commute_rules() -> Vec<Law> {
    let mut laws = Vec::with_capacity(64);
    for &op in LAYOUT_UNARY_OPS {
        for bare in [false, true] {
            let tag = if bare { "_bare" } else { "" };
            let fx = Pat::op(op, vec![Pat::var("x")]);
            let tx = layout_t(Pat::var("x"), !bare, "D0", "D1");
            let outer = layout_t(fx.clone(), !bare, "D0", "D1");
            let inner_op = Pat::op(op, vec![tx]);
            let push_law = format!(
                "Transpose commutes with the pointwise map {op}: T·f(x) = f(T·x) — naturality of the elementwise action."
            );
            let pull_law =
                format!("Inverse: f(T·x) = T·f(x) for {op} — hoist the layout change back out.");
            laws.push(
                Law::new(
                    format!("transpose_push_{op}{tag}"),
                    outer.clone(),
                    inner_op.clone(),
                )
                .with_law(push_law)
                .with_cond(cond_commute_unary())
                .with_tag("layout"),
            );
            laws.push(
                Law::new(format!("transpose_pull_{op}{tag}"), inner_op, outer)
                    .with_law(pull_law)
                    .with_cond(cond_commute_unary())
                    .with_tag("layout"),
            );
        }
    }
    for &op in LAYOUT_BINARY_OPS {
        for bare in [false, true] {
            let tag = if bare { "_bare" } else { "" };
            let fab = Pat::op(op, vec![Pat::var("a"), Pat::var("b")]);
            // Shared attr metavariables on both operand transposes
            // force an identical swap — positional broadcasting only
            // commutes with a common permutation.
            let tab = Pat::op(
                op,
                vec![
                    layout_t(Pat::var("a"), !bare, "D0", "D1"),
                    layout_t(Pat::var("b"), !bare, "D0", "D1"),
                ],
            );
            let outer = layout_t(fab, !bare, "D0", "D1");
            laws.push(
                Law::new(
                    format!("transpose_push_{op}{tag}"),
                    outer.clone(),
                    tab.clone(),
                )
                .with_law(format!(
                    "Transpose distributes over {op}: T·(a ⊙ b) = T·a ⊙ T·b — same-rank broadcast commutes with relayout."
                ))
                .with_cond(cond_commute_binary())
                .with_tag("layout"),
            );
            laws.push(
                Law::new(format!("transpose_pull_{op}{tag}"), tab, outer)
                    .with_law(format!("Inverse: T·a ⊙ T·b = T·(a ⊙ b) for {op}."))
                    .with_cond(cond_commute_binary())
                    .with_tag("layout"),
            );
        }
    }
    laws
}

/// `_involution_rules()` — `T(T(x)) = x` over every bare/dimmed
/// spelling combination (inner then outer, Python order dd/bd/db/bb).
fn layout_involution_rules() -> Vec<Law> {
    let mut out = Vec::with_capacity(4);
    for inner in [true, false] {
        for outer in [true, false] {
            let tag = format!(
                "{}{}",
                if outer { "d" } else { "b" },
                if inner { "d" } else { "b" }
            );
            out.push(
                Law::new(
                    format!("transpose_transpose_{tag}"),
                    layout_t(
                        layout_t(Pat::var("x"), inner, "I0", "I1"),
                        outer,
                        "O0",
                        "O1",
                    ),
                    Pat::var("x"),
                )
                .with_law(
                    "Transpose is an involution per axis pair: T₂∘T₁ = id when both swap the same axes.",
                )
                .with_cond(cond_involution())
                .with_tag("layout"),
            );
        }
    }
    out
}

/// `_matmul_rules()` — the product-transpose law both spellings and
/// directions, then the four `matmul_transpose_rev_*` operand-spelling
/// combinations (Python order dd/db/bd/bb).
fn layout_matmul_rules() -> Vec<Law> {
    let mm = || Pat::op("matmul", vec![Pat::var("A"), Pat::var("B")]);
    let mut out = vec![
        Law::new(
            "transpose_matmul",
            layout_t(mm(), true, "D0", "D1"),
            Pat::op(
                "matmul",
                vec![layout_mt(Pat::var("B")), layout_mt(Pat::var("A"))],
            ),
        )
        .with_law(
            "(A@B).mT = B.mT @ A.mT — the output's last-two swap folds into per-operand transposes (the BLAS NT/TN flag read, batched).",
        )
        .with_cond(cond_transpose_matmul())
        .with_tag("layout"),
        Law::new(
            "transpose_matmul_bare",
            layout_t(mm(), false, "D0", "D1"),
            Pat::op(
                "matmul",
                vec![layout_mt(Pat::var("B")), layout_mt(Pat::var("A"))],
            ),
        )
        .with_law(
            "Bare ``t()`` spelling of the product-transpose law — an implicit last-two swap.",
        )
        .with_cond(cond_transpose_matmul())
        .with_tag("layout"),
    ];
    for pd in [true, false] {
        for qd in [true, false] {
            let tag = format!(
                "{}{}",
                if pd { "d" } else { "b" },
                if qd { "d" } else { "b" }
            );
            out.push(
                Law::new(
                    format!("matmul_transpose_rev_{tag}"),
                    Pat::op(
                        "matmul",
                        vec![
                            layout_t(Pat::var("P"), pd, "P0", "P1"),
                            layout_t(Pat::var("Q"), qd, "Q0", "Q1"),
                        ],
                    ),
                    layout_mt(Pat::op("matmul", vec![Pat::var("Q"), Pat::var("P")])),
                )
                .with_law(
                    "P.mT @ Q.mT = (Q@P).mT — hoist the product's last-two swap outward, dropping both operand transposes.",
                )
                .with_cond(cond_mm_transposes())
                .with_tag("layout"),
            );
        }
    }
    out
}

/// The NT bridge trio — `LINEAR_FROM_MM_T`,
/// `LINEAR_FROM_MM_T_BARE`, `LINEAR_TO_MM_T`.
fn layout_linear_rules() -> Vec<Law> {
    vec![
        Law::new(
            "linear_from_matmul_t",
            Pat::op(
                "matmul",
                vec![
                    Pat::var("x"),
                    layout_t(Pat::var("W"), true, "D0", "D1"),
                ],
            ),
            Pat::op("linear", vec![Pat::var("x"), Pat::var("W")]),
        )
        .with_law(
            "matmul(x, W.mT) ≡ linear(x, W): F.linear IS the BLAS-NT call — W stays in (out, in) storage and no transpose materialises.",
        )
        .with_cond(cond_mm_t_is_linear())
        .with_tag("layout"),
        Law::new(
            "linear_from_matmul_t_bare",
            Pat::op(
                "matmul",
                vec![Pat::var("x"), layout_t(Pat::var("W"), false, "D0", "D1")],
            ),
            Pat::op("linear", vec![Pat::var("x"), Pat::var("W")]),
        )
        .with_law(
            "Bare-t() spelling of the NT bridge: x @ W.t() = linear(x, W) — the exported form of ``x @ W.t()``.",
        )
        .with_cond(cond_mm_t_is_linear())
        .with_tag("layout"),
        Law::new(
            "linear_to_matmul_t",
            Pat::op("linear", vec![Pat::var("x"), Pat::var("W")]),
            Pat::op(
                "matmul",
                vec![Pat::var("x"), layout_mt(Pat::var("W"))],
            ),
        )
        .with_law(
            "Reverse NT bridge — exposes the weight's implicit transpose so the migration laws can move it.",
        )
        .with_cond(cond_linear_shapes())
        .with_tag("layout"),
    ]
}

/// `LAYOUT_RULES` — the 77-law transpose/layout migration family
/// (64 pointwise commutation — 48 unary + 16 binary — + 4
/// involution + 6 matmul transpose + 3 NT-bridge, in Python order:
/// commute, involution, matmul, linear), opt-in like the Python
/// `WITH_LAYOUT` preset (never `DEFAULT`/`all`). Every member's
/// procedural `check=` hook is re-authored as a `cond` tree over
/// the ported `axes-*`/`rank`/`dim-compat`/`mm-out` vocabulary —
/// the `_CHECK_TRANSLATED` divergence family on the Python side.
pub fn layout() -> Vec<Law> {
    let mut laws = layout_commute_rules();
    laws.extend(layout_involution_rules());
    laws.extend(layout_matmul_rules());
    laws.extend(layout_linear_rules());
    laws
}

// ---------------------------------------------------------------------------
//  Law packs — mirror the Python bank collections.
// ---------------------------------------------------------------------------

/// The members of `SIMPLIFICATION_RULES` (Python order): the
/// monoid laws, the group identities, the decomposition /
/// kernel-fold / rsqrt-canonicalization / div-mul-natural laws, the
/// view-op strips and folds whose guards fit the ported vocabulary
/// (`select_mul`, `transpose_noop`, `chunk_single`, the
/// unsqueeze-pad family, `mul_unsq_nat`, `mul_reshape_inert_l`,
/// `softmax_fold`, `glu_fold`, `rms_norm_fold`,
/// `rms_norm_fold_nogain`, `matmul_div_distribute`,
/// `div_add_factor`), and the two const-guarded folds
/// (`div_sqrt_to_rsqrt`, `pow_to_rsqrt`). `batch_norm_expand*` stays
/// out — Python keeps `BATCH_NORM_RULES` opt-in too ([`batch_norm`]).
pub fn simplification() -> Vec<Law> {
    vec![
        comm_add(),
        comm_mul(),
        assoc_add(),
        assoc_mul(),
        id_add(),
        id_mul(),
        double_neg(),
        sub_to_add(),
        silu_expand(),
        silu_fold(),
        silu_mul_form(),
        softsign_fold(),
        square_expand(),
        pow_to_square(),
        square_to_pow(),
        mul_square(),
        select_mul(),
        transpose_noop(),
        chunk_single(),
        mul_unsq_pad_l(),
        mul_unsq_pad_r(),
        sub_unsq_pad_l(),
        add_unsq_pad_r(),
        mul_unsq_nat(),
        mul_reshape_inert_l(),
        softmax_fold(),
        glu_fold(),
        rms_norm_fold(),
        rms_norm_fold_nogain(),
        div_sqrt_to_rsqrt(),
        pow_to_rsqrt(),
        recip_sqrt_to_rsqrt(),
        mul_div_swap(),
        matmul_div_distribute(),
        div_add_factor(),
    ]
}

/// The members of `CATEGORICAL_RULES` (Python order): left/right
/// bilinearity and the weight-merge family, the `linear` merge /
/// composition / biased-compose laws, scalar naturality and the
/// `split_scalar_*` lemmas, the product-structure fuses
/// (`swiglu_fuse`, `parallel_mul_fuse`, `qkv_fuse`,
/// `qkv_fuse_asym`), the diagonal-scale naturality family plus
/// `gqa_absorb_repeat`, and the `sdpa_fold*` family. `conv_bn_fold`
/// stays out — its procedural `check`/`derive` hooks are the
/// transported `missing_hooks` — and the `prim:synop*` minted
/// records stay out of the pack: their fold patterns are portable
/// LAW_FORMAT=2 data but the synthetic `synop_*` operator
/// denotations are domain-side, so they ship as the
/// kit-eqsat `prim_folds` corpus instead (never in `all`).
pub fn categorical() -> Vec<Law> {
    let mut laws = vec![
        distribute_matmul_over_add(),
        factor_matmul(),
        weight_factor_matmul(),
        weight_distribute_matmul(),
        weight_factor_linear(),
        weight_distribute_linear(),
    ];
    laws.extend([
        right_distribute_matmul(),
        right_factor_matmul(),
        right_factor_linear(),
        assoc_linear(),
        assoc_linear_rev(),
        assoc_linear_bias(),
        assoc_linear_bias_rev(),
        naturality_scalar(),
        naturality_scalar_rev(),
        assoc_matmul(),
        assoc_matmul_rev(),
    ]);
    laws.extend(split_scalar_rules());
    laws.extend([
        swiglu_fuse(),
        parallel_mul_fuse(),
        qkv_fuse(),
        qkv_fuse_asym(),
        linear_channel_scale(),
        linear_channel_scale_rev(),
        linear_row_scale(),
        linear_row_scale_rev(),
        linear_channel_to_row_scale(),
        gqa_absorb_repeat(),
    ]);
    laws.extend(sdpa_fold_rules());
    laws
}

/// The `ATTENTION_RULES` + `SDPA_EXPAND_RULES` opt-in pack — the
/// Python `ATTENTION` ruleset's order: the 36 portable
/// `laws/attention.py` members (rope scale commutation, the
/// right-multiply absorbs, score-scale migration, the 24 uniform
/// view-scale rules; `rope_cat_compose` declined — see the section
/// header) ahead of the full `SDPA_EXPAND_RULES` (all eight
/// key-sets; the scale-free spellings mint `1/√(q[-1])` through the
/// `dim` dspec channel). `batch_norm_expand*` and the 60-law
/// `conv_channel_*` family live in their own opt-in packs
/// ([`batch_norm`], [`conv_folds`]) — Python keeps them out of
/// `ALL_RULES`/`DEFAULT` the same way.
pub fn attention() -> Vec<Law> {
    let mut laws = attention_path_rules();
    laws.extend(sdpa_expand_rules());
    laws
}

/// The portable slice of `PRIOR_ART_RULES` (Python order):
/// the §1 TASO concat-structure block (`concat_single`, the
/// `concat_flat_l`/`concat_flat_r` n-ary flattens,
/// `concat_chunk2`/`concat_chunk3`/`concat_split2` split-merge
/// round-trips — `concat_split2`'s `attr-len` guard rides the
/// ported attr vocabulary verbatim — the `stack_fold`/
/// `stack_expand` bridge pair, and the `concat_pull_*`
/// pointwise-commutation families), then the §1 TASO
/// matmul/linear/conv merges (`mm_concat_left`/`_rev`,
/// `mm_concat_right`/`_rev`, the `mm_splitk`/`_rev` pair whose
/// `dim`-expr `attr-cmp`s and `SZ` dspec ride the ported shape
/// vocabulary verbatim, the `linear_concat_w`/`_rev` projection
/// merges, the `linear_bias_*` epilogue fold/expand/merge family
/// over the `mm-shape-ok`/`transpose-out`/`mm-out`/`bcast-into`
/// specs, and the six `conv_out_merge*` same-geometry conv folds —
/// the shared `ST`/`PD`/`DL`/`GR` attr metavars bind both convs and
/// the merged conv identically), then the §1 index-view tail
/// `mul_unsq_rev` and the §2 no-op
/// elimination block (`contiguous_id`/`clone_id`/`alias_id`/
/// `detach_id`/`detach_inplace_id`, the `*_idem` cast family,
/// the squeeze/unsqueeze/reshape/broadcast/slice/index_select/
/// roll/pad/repeat/permute/movedim/flatten/unflatten/split/flip/
/// tril/triu/clamp folds — the procedural `check=` hooks
/// re-authored as `cond` trees over the ported `mod`/`len`/
/// `shape`/`unique`/`sum`/`posmap`/`attr-is` vocabulary;
/// `squeeze_all_noop` and `expand_noop` stay Python-side — the
/// all-dims forall and the `-1` keep-dim read need expr forms
/// the vocabulary cannot shape), then the §2 reduction algebra
/// (`std_sqrt_var`/`var_square_std` and the bare pair,
/// `norm2_expand`/`norm2_fold`, `log_softmax_fold`/
/// `log_softmax_expand` — `reduce_dim1`'s `_COND_DIM1` needs
/// `attr0` inside `attr-cmp-dim`'s shape-attr position and the
/// `mean_*` pair needs a product-over-dims derive, all named
/// refusals) and the §2 activation decompositions
/// (`sigmoid_expand`, `tanh_expand`, `gelu_expand`, `mish_expand`,
/// `hardsigmoid_expand`, `hardswish_expand`, `softplus_expand`/
/// `softplus_fold`, `elu_expand`, `leaky_relu_expand`,
/// `softsign_expand`, the `relu_*`/`relu6_*` fold-expand-idempotence
/// family) ahead of the activation folds
/// (`sigmoid_fold`, `sigmoid_fold_div`,
/// `tanh_fold`, `hardsigmoid_fold`), the §2 pointwise
/// idempotence/parity family (`abs_*`, `sqrt_square`,
/// `rsqrt_square`, `maximum_idem`/`minimum_idem`, the odd/even
/// `*_neg` slides, `exp_neg_recip`, `log_exp`, `log_recip`), the
/// const-algebra family (`pow_one`/`pow_zero`/`pow_neg_one`/
/// `pow_half`, `sqrt_to_pow`, `recip_recip`, `div_one_recip`,
/// `recip_expand`, the `add/sub/neg` const-free folds, `id_sub`,
/// `id_div`, `add_self`/`mul2_to_add`, `mul_neg1`, `neg_const`,
/// the `mul_const_assoc_*`/`add_const_assoc_*` families,
/// `sub_const_sub`, `sub_const_sub_l`, `div_const_div`,
/// `div_const_to_mul`, `mul_const_div`), and the spelled-unit
/// folds (`add_zeroslike_r`/`sub_zeroslike_r`/`mul_oneslike_r`/
/// `div_oneslike_r`) whose `shaped`/`shape-eq`/`bcast` grid guard
/// rides the ported shape vocabulary. Every other member reads
/// shapes (`shape`/`dim` specs, `attr-range`, `rank`, broadcast
/// predicates) or tensor channels — unported — carries procedural
/// `check`/`derive` hooks, or is unguarded but outside this
/// slice's region. Deliberately NOT in [`all`], mirroring the
/// Python bank's opt-in `prior_art` preset.
pub fn prior_art() -> Vec<Law> {
    let mut laws = vec![
        // -- §1 TASO concat structure
        concat_single(),
        concat_flat_l(),
        concat_flat_r(),
        concat_chunk2(),
        concat_chunk3(),
        concat_split2(),
        stack_fold(),
        stack_expand(),
    ];
    laws.extend(concat_pull_unary_rules());
    laws.extend(concat_pull_binary_rules());
    laws.extend([
        // -- §1 TASO matmul/linear concat merges and split-K
        mm_concat_left(),
        mm_concat_left_rev(),
        mm_concat_right(),
        mm_concat_right_rev(),
        mm_splitk(),
        mm_splitk_rev(),
        linear_concat_w(),
        linear_concat_w_rev(),
        linear_bias_fold(),
        linear_bias_fold_l(),
        linear_bias_expand(),
        linear_bias_merge(),
    ]);
    laws.extend(pa_conv_merge_rules());
    laws.extend([
        // -- §1 index-view tail + §2 no-op elimination
        mul_unsq_rev(),
        contiguous_id(),
        clone_id(),
        alias_id(),
        detach_id(),
        detach_inplace_id(),
    ]);
    laws.extend(cast_idem_rules());
    laws.extend([
        unsqueeze_squeeze(),
        squeeze_unsqueeze(),
        squeeze_noop(),
        reshape_noop(),
        reshape_reshape(),
        broadcast_to_noop(),
        broadcast_to_expand(),
        slice_noop(),
        slice_noop_nostart(),
        index_select_id(),
        roll_noop(),
        pad_noop(),
        repeat_noop(),
        permute_noop(),
        permute_compose(),
        movedim_involution(),
        flatten_noop(),
        unflatten_noop(),
        split_single(),
        flip_involution(),
        tril_idem(),
        triu_idem(),
        clamp_idem(),
        // -- §2 reduction algebra (REDUCE_DIM1_RULES and the mean
        //    pair stay Python-side — see the section header)
        std_sqrt_var(),
        var_square_std(),
        std_sqrt_var_bare(),
        var_square_std_bare(),
        norm2_expand(),
        norm2_fold(),
        log_softmax_fold(),
        log_softmax_expand(),
        // -- §2 activation decompositions + folds
        sigmoid_expand(),
        sigmoid_fold(),
        sigmoid_fold_div(),
        tanh_expand(),
        tanh_fold(),
        gelu_expand(),
        mish_expand(),
        hardsigmoid_expand(),
        hardsigmoid_fold(),
        hardswish_expand(),
        softplus_expand(),
        softplus_fold(),
        elu_expand(),
        leaky_relu_expand(),
        softsign_expand(),
        relu_fold_max_r(),
        relu_fold_max_l(),
        relu_fold_clamp(),
        relu_expand(),
        relu_idem(),
        relu6_fold(),
        relu6_expand(),
        // -- §2 pointwise idempotence / parity
        abs_idem(),
        abs_square(),
        square_abs(),
        abs_neg(),
        sqrt_square(),
        rsqrt_square(),
        maximum_idem(),
        minimum_idem(),
        tanh_neg(),
        sigmoid_neg(),
        softsign_neg(),
        erf_neg(),
        sin_neg(),
        cos_neg(),
        exp_neg_recip(),
        log_exp(),
        log_recip(),
        // -- §2 scalar/Const algebra
        pow_one(),
        pow_zero(),
        pow_neg_one(),
        pow_half(),
        sqrt_to_pow(),
        recip_recip(),
        div_one_recip(),
        recip_expand(),
        add_neg_to_sub(),
        sub_neg_to_add(),
        neg_sub(),
        sub_zero_to_neg(),
        id_sub(),
        id_div(),
        add_self(),
        mul2_to_add(),
        mul_neg1(),
        neg_const(),
    ]);
    laws.extend(mul_const_assoc_rules());
    laws.extend(add_const_assoc_rules());
    laws.extend([
        sub_const_sub(),
        sub_const_sub_l(),
        div_const_div(),
        div_const_to_mul(),
        mul_const_div(),
        // -- §2 spelled-unit folds
        add_zeroslike_r(),
        sub_zeroslike_r(),
        mul_oneslike_r(),
        div_oneslike_r(),
    ]);
    laws
}

/// `SCAN_LAWS` + `SCAN_DIAG_LAWS` — the 22-law scan-monoid family
/// (dense affine + diagonal-affine carriers), opt-in like the Python
/// collections (`CARRIER_SEARCH`, never `DEFAULT`). The `affd_lift*`
/// state conds (`op-in`/`leaf`/`not`/`const`) and the unit lift's
/// `bcast` dspec are carried as record data verbatim.
pub fn scan() -> Vec<Law> {
    let mut laws = scan_laws();
    laws.extend(scan_diag_laws());
    laws
}

/// `POINTWISE_FACTOR_RULES` — the opt-in unguarded Hadamard
/// distributivity pair. Deliberately NOT in [`all`], mirroring the
/// Python bank's exclusion from `ALL_RULES`/`DEFAULT`.
pub fn pointwise_factor() -> Vec<Law> {
    vec![distribute_mul_add(), factor_mul_add()]
}

/// The stage-1 subset of `ALL_RULES` (`all_rules()`):
/// [`simplification`] + [`categorical`]. [`prior_art`] stays opt-in,
/// like the Python bank.
pub fn all() -> Vec<Law> {
    let mut laws = simplification();
    laws.extend(categorical());
    laws
}

/// **UNSOUND DEMO — not a law; never ship it.** The spike's
/// `reshape → flatten` attrs-bind demonstration: `tensor.reshape(x,
/// attrs=?a) → tensor.flatten(x, attrs=?a)` — the `__attrs__`
/// payload is bound whole and re-emitted verbatim.
///
/// reshape→flatten is NOT value- or shape-preserving in general
/// (reshape admits arbitrary re-layouts; flatten commits to one), so
/// this rewrite is a non-equivalence kept ONLY to pin the
/// `attrs_bind` round-trip in tests. The `_demo` suffix is the
/// warning: it is in NO pack ([`simplification`]/[`categorical`]/
/// [`all`] exclude it), and passing it to `saturate`/`optimize`
/// asserts an equivalence that does not hold. Use it for matcher
/// tests only.
pub fn reshape_flatten_demo() -> Law {
    Law::new(
        "reshape_flatten_demo",
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
