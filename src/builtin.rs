//! The law packs — unconditional ports of `mlopt_core`'s tensor law
//! banks (`eqsat/packages/mlopt-core/src/mlopt_core/laws/tensor.py`),
//! spelled in [`crate::pat`].
//!
//! These are the Python bank's **guard-free** laws: no `cond.py`
//! side-conditions and no `dspec` derives, so they are the honest
//! batch for the Rust interpretation (spike §7 step 2; `Law` carries
//! no hooks — see [`crate::law`]). Names, `law` strings, and tag
//! bundles are carried verbatim so provenance lines up with the
//! Python bank (`mlopt_core.laws.tags`: `simplification` /
//! `categorical` / `symmetry` / `expansive` / `fusion` / `subsumed`).
//!
//! Pack layout mirrors the Python collections:
//!
//! * [`simplification`] — the guard-free members of
//!   `SIMPLIFICATION_RULES` (monoid/group laws, the silu/square/pow
//!   decompositions and folds, the rsqrt canonicalization,
//!   `mul_div_swap`).
//! * [`categorical`] — the guard-free members of
//!   `CATEGORICAL_RULES` (right-slot bilinearity, `assoc_matmul`,
//!   the `linear` composition laws, `qkv_fuse`).
//! * [`pointwise_factor`] — `POINTWISE_FACTOR_RULES`, the opt-in
//!   Hadamard distributivity pair the Python bank deliberately keeps
//!   out of `ALL_RULES` (a closure generator).
//! * [`all`] — `simplification` + `categorical`, the guard-free
//!   subset of `ALL_RULES` (`all_rules()`).
//!
//! Scope note: the Python patterns live in the *typed* `Op` IR; here
//! they are kernel `Term::Op`s — `dim` is child slot 0 (auto-bound,
//! RHS inherits the root dim) and `Const` leaves are JSON-valued.
//! Python attr *metavariables* (`scale="SC"`, `shape="S"`) have no
//! per-key spelling on this surface: [`OpPat::attrs_bind`] binds the
//! whole `__attrs__` payload, which is equivalent when the payload
//! is the single attr the Python pattern names, and strictly more
//! permissive when a node carries extra keys (the Python exact-
//! keyset match would decline). [`OpPat::attrs_lit`] pins literal
//! payloads exactly.

use serde_json::json;

use crate::law::Law;
use crate::pat::{OpPat, Pat};

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
        "The manual-silu fold: x · σ(x) = silu(x) — the definitional \
         inverse of silu_expand, and the mediating 3-cell of the \
         silu_expand/silu_mul_form × swiglu_fuse critical pairs.",
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
        "x / (|x| + 1) = softsign(x) — the bounded rational map, \
         written as a three-op decomposed spelling, folds into the one \
         kernel.",
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
        "x·x = square(x): the mul spelling of x² folds to the unary \
         kernel — the definitional inverse of square_expand.",
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
        "reciprocal(sqrt(x)) = rsqrt(x): the exported ``1/sqrt`` \
         spelling — aten lowers scalar-over-tensor div to reciprocal — \
         folds to the same canonical kernel.",
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
        "(a/b)·c = (a·c)/b — pointwise div/mul naturality, exact \
         for any broadcast (the decay-weighted score's scale slides \
         out of the product).",
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
        "Compose stacked nn.Linears: fused weight is B @ A \
         (transposes flip the product order).",
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
        "Reverse linear composition: linear(x, B@A) = \
         linear(linear(x, A), B) (eqsat weighs fused vs split).",
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
        "Triple pairing <q,k,v> : X -> V^3 — three projections of the \
         same input are ONE GEMM into the product space, then three \
         zero-cost chunk projections.  (Fused QKV.)",
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
        "Hadamard distributivity: c ⊙ (x+y) = c⊙x + c⊙y — the \
         elementwise product distributes over the additive monoid.",
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
        "Factor a shared elementwise factor (reverse distributivity) \
         — c⊙x + c⊙y = c⊙(x+y); what the carrier fold chains consume.",
    )
    .with_tag("categorical")
    .with_tag("expansive")
}

// ---------------------------------------------------------------------------
//  Law packs — mirror the Python bank collections.
// ---------------------------------------------------------------------------

/// The guard-free members of `SIMPLIFICATION_RULES` (Python order):
/// the monoid laws, the group identities, and the decomposition /
/// kernel-fold / rsqrt-canonicalization / div-mul-natural laws. The
/// guarded members (`select_mul`, `transpose_noop`, `chunk_single`,
/// the unsqueeze-pad strips, `mul_reshape_inert_l`, `mul_unsq_nat`,
/// `softmax_fold`, `glu_fold`, `rms_norm_fold*`, `div_sqrt_to_rsqrt`,
/// `pow_to_rsqrt`, `matmul_div_distribute`, `div_add_factor`) need
/// `cond`/`dspec` hooks [`crate::law::Law`] does not carry yet.
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
        recip_sqrt_to_rsqrt(),
        mul_div_swap(),
    ]
}

/// The guard-free members of `CATEGORICAL_RULES` (Python order):
/// right-slot bilinearity, the `linear` composition laws, matmul /
/// linear associativity, and `qkv_fuse`. The guarded members
/// (`distribute_matmul_over_add`, `factor_matmul`, the weight-merge
/// family, the scale naturality laws, the fuse family, `gqa_absorb`,
/// the sdpa folds, and the minted composites) need `cond`/`dspec`
/// hooks [`crate::law::Law`] does not carry yet.
pub fn categorical() -> Vec<Law> {
    vec![
        right_distribute_matmul(),
        right_factor_matmul(),
        right_factor_linear(),
        assoc_linear(),
        assoc_linear_rev(),
        assoc_matmul(),
        assoc_matmul_rev(),
        qkv_fuse(),
    ]
}

/// `POINTWISE_FACTOR_RULES` — the opt-in unguarded Hadamard
/// distributivity pair. Deliberately NOT in [`all`], mirroring the
/// Python bank's exclusion from `ALL_RULES`/`DEFAULT`.
pub fn pointwise_factor() -> Vec<Law> {
    vec![distribute_mul_add(), factor_mul_add()]
}

/// The guard-free subset of `ALL_RULES` (`all_rules()`):
/// [`simplification`] + [`categorical`].
pub fn all() -> Vec<Law> {
    let mut laws = simplification();
    laws.extend(categorical());
    laws
}

/// A demonstration law for the attrs-bind path (the spike's
/// `reshape → flatten`): `tensor.reshape(x, attrs=?a) →
/// tensor.flatten(x, attrs=?a)` — the `__attrs__` payload is bound
/// whole and re-emitted verbatim. Not part of any pack; it exists
/// to pin the attrs round-trip in tests.
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
