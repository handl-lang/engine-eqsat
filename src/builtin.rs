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
//! * [`attention`] — the full `SDPA_EXPAND_RULES` (all eight
//!   key-sets; the scale-free spellings mint `1/√(q[-1])` through
//!   the `dim` dspec channel).
//! * [`batch_norm`] — `BATCH_NORM_RULES`, the opt-in eval-mode
//!   `batch_norm` expand triple (`$leaf:` Const mint + the
//!   `concat`/`take`/`len` channel-view dspec).
//! * [`conv_folds`] — `CONV_FOLD_RULES`, the 60-law conv
//!   channel-scale/shift family (5 attr-prefixes × bias arity × 6
//!   rule shapes), opt-in like the Python collection.
//! * [`prior_art`] — the portable slice of `PRIOR_ART_RULES`: the
//!   const-algebra folds (`pow_*`, `*_const_*`, `neg_const`, the
//!   `*_const_assoc_*` families) and the sigmoid/tanh/hardsigmoid
//!   folds whose guards are `const-cmp` conjunctions.
//!   Deliberately NOT in [`all`], mirroring the Python bank's
//!   opt-in `prior_art` preset.
//! * [`pointwise_factor`] — `POINTWISE_FACTOR_RULES`, the opt-in
//!   Hadamard distributivity pair the Python bank deliberately keeps
//!   out of `ALL_RULES` (a closure generator).
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
/// The `shape` attr payload is bound whole (`attrs_bind("S")`) —
/// the Python `shape="S"` attr metavariable's equivalent on this
/// surface (see the module docs for the granularity caveat).
fn head_view(t: Pat) -> Pat {
    Pat::Op(
        OpPat::new("transpose")
            .arg(Pat::Op(OpPat::new("reshape").arg(t).attrs_bind("S")))
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
/// Attr spellings: `scale`/`shape` are bound whole-payload
/// (`attrs_bind`) — equivalent to the Python attr metavars when the
/// payload carries only that key, more permissive otherwise (see
/// module docs); `transpose`/`concat`/`chunk` attrs are literal
/// payloads (`attrs_lit`).
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
                .attrs_bind("SC"),
        ),
        Pat::Op(OpPat::new("sdpa").args(rhs_heads).attrs_bind("SC")),
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
//  records ship as `laws/prim_folds.json` data instead).
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
/// `laws/prim_folds.json` corpus instead (never in `all`).
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

/// The full `SDPA_EXPAND_RULES` — all eight key-sets (the four
/// scale-free spellings mint `1/√(q[-1])` through the `dim` shape
/// channel; ported at stage 4). Opt-in `attention` preset, tag
/// `attention`. `batch_norm_expand*` and the 60-law
/// `conv_channel_*` family live in their own opt-in packs
/// ([`batch_norm`], [`conv_folds`]) — Python keeps them out of
/// `ALL_RULES`/`DEFAULT` the same way.
pub fn attention() -> Vec<Law> {
    sdpa_expand_rules()
}

/// The stage-1-portable slice of `PRIOR_ART_RULES` (Python order):
/// the activation folds (`sigmoid_fold`, `sigmoid_fold_div`,
/// `tanh_fold`, `hardsigmoid_fold`), the const-algebra family
/// (`pow_one`/`pow_zero`/`pow_neg_one`/`pow_half`, `sqrt_to_pow`,
/// `div_one_recip`, `recip_expand`, `neg_const`, the
/// `mul_const_assoc_*`/`add_const_assoc_*` families,
/// `sub_const_sub`, `sub_const_sub_l`, `div_const_div`,
/// `div_const_to_mul`, `mul_const_div`). Every other member reads
/// shapes (`shape`/`dim` specs, `attr-range`, `rank`, broadcast
/// predicates) or tensor channels — unported — or is unguarded but
/// outside this slice's region. Deliberately NOT in [`all`],
/// mirroring the Python bank's opt-in `prior_art` preset.
pub fn prior_art() -> Vec<Law> {
    let mut laws = vec![
        sigmoid_fold(),
        sigmoid_fold_div(),
        tanh_fold(),
        hardsigmoid_fold(),
        pow_one(),
        pow_zero(),
        pow_neg_one(),
        pow_half(),
        sqrt_to_pow(),
        div_one_recip(),
        recip_expand(),
        neg_const(),
    ];
    laws.extend(mul_const_assoc_rules());
    laws.extend(add_const_assoc_rules());
    laws.extend([
        sub_const_sub(),
        sub_const_sub_l(),
        div_const_div(),
        div_const_to_mul(),
        mul_const_div(),
    ]);
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
