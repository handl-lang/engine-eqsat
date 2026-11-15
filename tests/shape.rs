//! Stage-2/3 tests: shape inference ([`eqsat::shape`]), the ported
//! shape predicates through `guard.rs`, and `flops` formula pricing
//! in `cost.rs` — all over kernel `Term`s bridged with
//! `handl_egraph::from_kernel`.

use handl_egraph::{BoundEnv, from_kernel};
use hnl_engine::Term as KTerm;
use hnl_engine_eqsat::shape::Shape;
use hnl_engine_eqsat::{self as eqsat, CostSpec, Law, shape};
use serde_json::json;
use std::rc::Rc;

// ---------------------------------------------------------------------------
//  helpers
// ---------------------------------------------------------------------------

/// A tensor leaf declaring its shape through the `__tensor__` wire
/// convention (`{"__tensor__": {"shape": […], "dtype": …}}`).
fn tensor(shape: serde_json::Value) -> KTerm {
    KTerm::constant(json!({"__tensor__": {"shape": shape}}))
}

/// A shaped tensor leaf with concrete dims.
fn shaped(dims: &[i64]) -> KTerm {
    tensor(json!(dims))
}

/// A provably-unknown leaf — an `Opaque` record carries no shape
/// payload, so inference must decline rather than guess. (A `Const`
/// scalar is *not* unknown: numbers are `()`-shaped.)
fn leaf() -> KTerm {
    KTerm::Opaque {
        data: json!({"tag": "mystery"}),
    }
}

fn op(name: &str, args: Vec<KTerm>) -> KTerm {
    KTerm::op_native(name, args)
}

fn op_attrs(name: &str, args: Vec<KTerm>, attrs: serde_json::Value) -> KTerm {
    KTerm::op_native_attrs(name, args, attrs)
}

/// Infer the shape of a kernel term (the public boundary: kernel
/// `Term` → bridge → [`shape::shape_of`]).
fn shape_of_k(t: &KTerm) -> Shape {
    shape::shape_of(&from_kernel(t))
}

fn tuple(dims: &[i64]) -> Shape {
    Shape::Tuple(dims.iter().map(|d| Some(*d)).collect())
}

/// A tuple with explicit `None` (unknown) extents.
fn ptup(dims: &[Option<i64>]) -> Shape {
    Shape::Tuple(dims.to_vec())
}

fn assert_shape(t: &KTerm, expected: Shape) {
    assert_eq!(shape_of_k(t), expected, "shape of {t:?}");
}

// ---------------------------------------------------------------------------
//  shape inference — known shapes
// ---------------------------------------------------------------------------

#[test]
fn tensor_leaf_declares_shape() {
    assert_shape(&shaped(&[2, 3]), tuple(&[2, 3]));
    assert_shape(&shaped(&[]), Shape::Tuple(vec![]));
    // An unknown-extent dim stays `None` inside the tuple.
    let t = KTerm::constant(json!({"__tensor__": {"shape": [2, null]}}));
    assert_eq!(shape_of_k(&t), Shape::Tuple(vec![Some(2), None]));
}

#[test]
fn plain_leaves_are_unknown_not_guessed() {
    assert_eq!(shape_of_k(&leaf()), Shape::Unknown);
    // A bare `{"shape": …}` payload on an Opaque leaf also reads
    // (a `Const` scalar stays `()` — only `__tensor__` wire values
    // declare tensor shapes on consts).
    let t = KTerm::Opaque {
        data: json!({"shape": [4, 5]}),
    };
    assert_eq!(shape_of_k(&t), tuple(&[4, 5]));
}

#[test]
fn elementwise_broadcast() {
    let a = shaped(&[2, 3]);
    let b = shaped(&[3]);
    let c = shaped(&[2, 1]);
    let d = shaped(&[4]); // incompatible trailing dim
    assert_shape(&op("add", vec![a.clone(), b]), tuple(&[2, 3]));
    assert_shape(&op("mul", vec![a.clone(), c]), tuple(&[2, 3]));
    assert_shape(
        &op("tensor.add", vec![a.clone(), shaped(&[1])]),
        tuple(&[2, 3]),
    );
    // (2,3) + (4,) cannot broadcast — provably Invalid, never a guess.
    assert_eq!(shape_of_k(&op("add", vec![a, d])), Shape::Invalid);
}

#[test]
fn broadcast_helper() {
    assert_eq!(
        shape::broadcast(&tuple(&[2, 3]), &tuple(&[3])),
        tuple(&[2, 3])
    );
    assert_eq!(
        shape::broadcast(&ptup(&[Some(2), None]), &tuple(&[3])),
        Shape::Tuple(vec![Some(2), None])
    );
    // `_broadcast(None, b) = b` — an unknown operand shape is the
    // wildcard that yields the other side (Python verbatim).
    assert_eq!(shape::broadcast(&Shape::Unknown, &tuple(&[3])), tuple(&[3]));
    // (2,) aligns trailing: 2 vs 3 mismatches → Invalid; (3,) works.
    assert_eq!(
        shape::broadcast(&tuple(&[2]), &tuple(&[2, 3])),
        Shape::Invalid
    );
    assert_eq!(
        shape::broadcast(&tuple(&[3]), &tuple(&[2, 3])),
        tuple(&[2, 3])
    );
}

#[test]
fn matmul_shapes() {
    // (2,3) @ (3,4) → (2,4)
    assert_shape(
        &op("matmul", vec![shaped(&[2, 3]), shaped(&[3, 4])]),
        tuple(&[2, 4]),
    );
    // batched: (5,2,3) @ (5,3,4) → (5,2,4)
    assert_shape(
        &op("matmul", vec![shaped(&[5, 2, 3]), shaped(&[5, 3, 4])]),
        tuple(&[5, 2, 4]),
    );
    // contraction mismatch: inference is the simple read
    // `(*a[:-1], b[-1])` — validity is `mm-shape-ok`'s job, not the
    // shape's (Python `_infer_op_shape` verbatim).
    assert_shape(
        &op("matmul", vec![shaped(&[2, 3]), shaped(&[4, 5])]),
        tuple(&[2, 5]),
    );
    // unknown operand → Unknown
    assert_eq!(
        shape_of_k(&op("matmul", vec![leaf(), shaped(&[3, 4])])),
        Shape::Unknown
    );
}

#[test]
fn matmul_shape_helper() {
    assert_eq!(
        shape::matmul_shape(&tuple(&[2, 3]), &tuple(&[3, 4])),
        Some(vec![Some(2), Some(4)])
    );
    // rank-1 rhs: (2,3) @ (3,) → (2,)
    assert_eq!(
        shape::matmul_shape(&tuple(&[2, 3]), &tuple(&[3])),
        Some(vec![Some(2)])
    );
    // mismatch → None
    assert_eq!(shape::matmul_shape(&tuple(&[2, 3]), &tuple(&[4, 5])), None);
    assert_eq!(shape::matmul_shape(&Shape::Unknown, &tuple(&[3, 4])), None);
}

#[test]
fn reshape_folds_negative_one() {
    let x = shaped(&[2, 3, 4]);
    // reshape(x, [6, 4]) — positional Const shape-list arg.
    let shape_arg = KTerm::constant(json!([6, 4]));
    assert_shape(&op("reshape", vec![x.clone(), shape_arg]), tuple(&[6, 4]));
    // -1 folds: [2, 3, 4] → [-1, 4] gives [6, 4].
    let neg = KTerm::constant(json!([-1, 4]));
    assert_shape(&op("reshape", vec![x.clone(), neg]), tuple(&[6, 4]));
    // explicit -1 via "shape" attr.
    let attr = op_attrs("reshape", vec![x.clone()], json!({"shape": [-1, 2, 2]}));
    assert_shape(&attr, tuple(&[6, 2, 2]));
    // numel mismatch is provably Invalid.
    let bad = KTerm::constant(json!([5, 4]));
    assert_eq!(
        shape_of_k(&op("reshape", vec![x.clone(), bad])),
        Shape::Invalid
    );
}

#[test]
fn transpose_and_views() {
    let x = shaped(&[2, 3, 4]);
    let t = op_attrs("transpose", vec![x.clone()], json!({"dim0": 0, "dim1": 2}));
    assert_shape(&t, tuple(&[4, 3, 2]));
    // permute-style attr spelling.
    let p = op_attrs("permute", vec![x.clone()], json!({"dims": [2, 0, 1]}));
    assert_shape(&p, tuple(&[4, 2, 3]));
    // unsqueeze/squeeze.
    let u = op_attrs("unsqueeze", vec![shaped(&[2, 3])], json!({"dim": 1}));
    assert_shape(&u, tuple(&[2, 1, 3]));
    let s = op_attrs("squeeze", vec![shaped(&[2, 1, 3])], json!({"dim": 1}));
    assert_shape(&s, tuple(&[2, 3]));
}

#[test]
fn reductions_drop_last_dim() {
    let x = shaped(&[2, 3, 4]);
    // sum over dim 1 → (2, 4)
    let s = op_attrs("sum", vec![x.clone()], json!({"dim": 1}));
    assert_shape(&s, tuple(&[2, 4]));
    // keepdim → (2, 1, 4)
    let k = op_attrs("sum", vec![x.clone()], json!({"dim": 1, "keepdim": true}));
    assert_shape(&k, tuple(&[2, 1, 4]));
    // full reduction → scalar tuple
    let all = op_attrs("sum", vec![x], json!({"dim": [0, 1, 2]}));
    assert_shape(&all, tuple(&[]));
}

#[test]
fn concat_slice_chunk() {
    // cat along dim 0: (2,3)+(4,3) → (6,3); dim 1: (2,3)+(2,5) → (2,8).
    assert_shape(
        &op("concat", vec![shaped(&[2, 3]), shaped(&[4, 3])]),
        tuple(&[6, 3]),
    );
    assert_shape(
        &op_attrs(
            "concat",
            vec![shaped(&[2, 3]), shaped(&[2, 5])],
            json!({"dim": 1}),
        ),
        tuple(&[2, 8]),
    );
    // slice: (2,3)[:, 1:3] → (2,2)
    let sl = op_attrs(
        "slice",
        vec![shaped(&[2, 3])],
        json!({"dim": 1, "start": 1, "end": 3}),
    );
    assert_shape(&sl, tuple(&[2, 2]));
    // chunk(2) along dim 0 of (4,3) → two (2,3)s — the head shape.
    let ch = op_attrs(
        "chunk",
        vec![shaped(&[4, 3])],
        json!({"chunks": 2, "dim": 0}),
    );
    let s = shape_of_k(&ch);
    assert!(matches!(s, Shape::Tuple(_)), "chunk shape {s:?}");
}

// ---------------------------------------------------------------------------
//  conservative unknowns
// ---------------------------------------------------------------------------

#[test]
fn unknowns_never_guess() {
    // reshape of an unknown input → Unknown (not a fabricated shape).
    let shape_arg = KTerm::constant(json!([2, 2]));
    assert_eq!(
        shape_of_k(&op("reshape", vec![leaf(), shape_arg])),
        Shape::Unknown
    );
    // elementwise with one unknown operand → Unknown: a `None`
    // *shape* sibling is genuinely unknown for broadcast binops
    // (the `_broadcast` wildcard covers `None` *dims*, not shapes).
    assert_eq!(
        shape_of_k(&op("add", vec![leaf(), shaped(&[2, 3])])),
        Shape::Unknown
    );
    // matmul needs both operand tuples — unknown → Unknown.
    assert_eq!(
        shape_of_k(&op("matmul", vec![leaf(), shaped(&[3, 4])])),
        Shape::Unknown
    );
    // a partially-known tuple still propagates None dims.
    let partial = KTerm::constant(json!({"__tensor__": {"shape": [2, null]}}));
    let out = shape_of_k(&op("add", vec![partial, shaped(&[3])]));
    assert_eq!(out, Shape::Tuple(vec![Some(2), None]));
}

#[test]
fn unsupported_ops_are_unknown() {
    // Python's `case _`: unknown ops pass operand 0's shape through.
    assert_shape(&op("frobnicate", vec![shaped(&[2])]), tuple(&[2]));
    assert_eq!(shape_of_k(&op("frobnicate", vec![leaf()])), Shape::Unknown);
}

// ---------------------------------------------------------------------------
//  numel helpers
// ---------------------------------------------------------------------------

#[test]
fn numel_helpers() {
    assert_eq!(shape::numel(&tuple(&[2, 3])), 6);
    assert_eq!(shape::numel(&ptup(&[Some(2), None])), 2); // unknown dims count 1
    assert_eq!(shape::numel(&Shape::Unknown), 1);
    // concrete-only numel — the honest flops gate.
    assert_eq!(shape::numel_concrete(&tuple(&[2, 3])), Some(6));
    assert_eq!(shape::numel_concrete(&ptup(&[Some(2), None])), None);
    assert_eq!(shape::numel_concrete(&Shape::Unknown), None);
    assert_eq!(shape::numel_concrete(&Shape::Invalid), None);
}

// ---------------------------------------------------------------------------
//  shape predicates (the `pred` table) over BoundEnv
// ---------------------------------------------------------------------------

fn env_with(t: KTerm) -> BoundEnv {
    let mut env = BoundEnv::default();
    env.terms.insert(Rc::from("x"), from_kernel(&t));
    env
}

#[test]
fn shape_predicates_evaluate() {
    use shape::pred;
    let env = env_with(op("add", vec![shaped(&[2, 3]), shaped(&[3])]));
    let spec = shape::ShapeSpec::Var("x".to_string());
    assert!(pred::shaped(&env, &spec));
    assert!(pred::concrete(&env, &spec));
    assert!(!pred::scalar(&env, &spec));
    assert!(pred::rank(&env, &spec, ">=", 2));
    assert!(!pred::rank(&env, &spec, "=", 3));

    // scalar tensor
    let scalar_env = env_with(shaped(&[]));
    assert!(pred::scalar(&scalar_env, &spec));

    // unknown term → nothing is provable
    let unk_env = env_with(leaf());
    assert!(!pred::shaped(&unk_env, &spec));
    assert!(!pred::concrete(&unk_env, &spec));
}

// ---------------------------------------------------------------------------
//  guards end-to-end: a `cond` shape predicate gates firing
// ---------------------------------------------------------------------------

/// A law `add(x, y) → mul(x, y)` gated on `["shaped","x"]` — fires
/// only when the bound `x` resolves to a provably-shaped tensor.
fn guarded_add_mul() -> Law {
    let mvar = |n: &str| json!({"mvar": n});
    let op_pat = |name: &str, args: Vec<serde_json::Value>| {
        json!({
            "op": name, "dim": null, "args": args, "source": "native",
            "lit": null, "attrs": null,
        })
    };
    Law::from_data(&json!({
        "version": 2,
        "kind": "law",
        "name": "add_to_mul_shaped",
        "lhs": op_pat("add", vec![mvar("x"), mvar("y")]),
        "rhs": op_pat("mul", vec![mvar("x"), mvar("y")]),
        "cond": ["shaped", "x"],
    }))
    .expect("law record")
}

#[test]
fn shape_guard_gates_firing() {
    let law = guarded_add_mul();
    // Shaped operands: the guard holds, the rewrite fires.
    let input = op("add", vec![shaped(&[2, 3]), shaped(&[3])]);
    let mut sat = eqsat::saturate(
        std::slice::from_ref(&input),
        std::slice::from_ref(&law),
        &eqsat::SaturateOptions::spec(),
    )
    .expect("saturate");
    let expected = op("mul", vec![shaped(&[2, 3]), shaped(&[3])]);
    let class = sat
        .graph
        .class_of_term(&from_kernel(&expected))
        .expect("rhs instantiated");
    assert_eq!(sat.graph.find_ref(class), sat.graph.find_ref(sat.roots[0]));

    // Unshaped operands: `x` binds an unknown leaf — the guard vetoes.
    let input2 = op("add", vec![leaf(), leaf()]);
    let sat2 = eqsat::saturate(
        std::slice::from_ref(&input2),
        std::slice::from_ref(&law),
        &eqsat::SaturateOptions::spec(),
    )
    .expect("saturate");
    let fired: Vec<&str> = sat2
        .graph
        .merge_log()
        .iter()
        .filter_map(|e| e.rule.as_deref())
        .collect();
    assert!(
        !fired.contains(&"add_to_mul_shaped"),
        "guarded law fired on unshaped input: {fired:?}"
    );
}

// ---------------------------------------------------------------------------
//  flops formulas
// ---------------------------------------------------------------------------

fn flops_spec(ops: serde_json::Value) -> CostSpec {
    CostSpec::from_data(&json!({
        "kind": "op_table",
        "default": 1.0,
        "leaf": 0.0,
        "ops": ops,
    }))
    .expect("cost spec")
}

#[test]
fn flops_matmul_formula_prices_known_shapes() {
    // 2*m*n*k over (2,3)@(3,4): m=2, n=4, k=3 → 48.
    let spec = flops_spec(json!({"matmul": {"flops": "2*m*n*k"}}));
    let t = op("matmul", vec![shaped(&[2, 3]), shaped(&[3, 4])]);
    assert_eq!(spec.price(&from_kernel(&t)), 48.0);
    // batched (5,2,3)@(5,3,4): out = (5,2,4) → numel 40, k = 3 →
    // 2*out*k = 240 — `out` sees the batch dims, m/n/k do not.
    let t2 = op("matmul", vec![shaped(&[5, 2, 3]), shaped(&[5, 3, 4])]);
    let spec2 = flops_spec(json!({"matmul": {"flops": "2*out*k"}}));
    assert_eq!(spec2.price(&from_kernel(&t2)), 240.0);
}

#[test]
fn flops_elementwise_uses_out() {
    let spec = flops_spec(json!({"add": {"flops": "out"}}));
    let t = op("add", vec![shaped(&[2, 3]), shaped(&[3])]);
    assert_eq!(spec.price(&from_kernel(&t)), 6.0);
}

#[test]
fn flops_unknown_shapes_price_infinite() {
    let spec = flops_spec(json!({"matmul": {"flops": "2*m*n*k"}}));
    // Unknown operand → unbound names → +∞.
    let t = op("matmul", vec![leaf(), shaped(&[3, 4])]);
    assert_eq!(spec.price(&from_kernel(&t)), f64::INFINITY);
    // A provably-Invalid shape (broadcast mismatch) → +∞.
    let bad = op("add", vec![shaped(&[2, 3]), shaped(&[4])]);
    let bad_spec = flops_spec(json!({"add": {"flops": "out"}}));
    assert_eq!(bad_spec.price(&from_kernel(&bad)), f64::INFINITY);
    // Unbound name in the formula → +∞ (never a guess).
    let spec2 = flops_spec(json!({"add": {"flops": "2*out*zzz"}}));
    assert_eq!(
        spec2.price(&from_kernel(&op("add", vec![shaped(&[2]), shaped(&[2])]))),
        f64::INFINITY
    );
    // Malformed expression → +∞.
    let spec3 = flops_spec(json!({"add": {"flops": "2**out"}}));
    assert_eq!(
        spec3.price(&from_kernel(&op("add", vec![shaped(&[2]), shaped(&[2])]))),
        f64::INFINITY
    );
}

#[test]
fn flops_count_and_size_unchanged() {
    // count/size semantics untouched by formula support.
    let t = op("matmul", vec![shaped(&[2, 3]), shaped(&[3, 4])]);
    assert_eq!(
        CostSpec::Count.price(&from_kernel(&t)),
        CostSpec::Count.price(&from_kernel(&t))
    );
    assert!(CostSpec::Count.price(&from_kernel(&t)) > 0.0);
    assert!(CostSpec::Size.price(&from_kernel(&t)) > 0.0);
    // op_table without formulas still prices flat weights.
    let spec = flops_spec(json!({"add": 1.0}));
    let c = spec.price(&from_kernel(&op("add", vec![leaf(), leaf()])));
    assert!(c.is_finite() && c > 0.0);
}

#[test]
fn flops_conv_formula() {
    // conv2d: 2*out*ic*kh*kw/groups over N=1,O=8,C=3,kh=kw=3 → out=1*8*H'*W'.
    let spec = flops_spec(json!({
        "conv2d": {"flops": "2*out*ic*kh*kw/groups"}
    }));
    let x = shaped(&[1, 3, 32, 32]);
    let w = shaped(&[8, 3, 3, 3]);
    let t = op_attrs(
        "conv2d",
        vec![x, w],
        json!({
            "stride": [1, 1], "padding": [0, 0], "dilation": [1, 1],
            "groups": 1,
        }),
    );
    let price = spec.price(&from_kernel(&t));
    assert!(price.is_finite() && price > 0.0, "conv2d price {price}");
    // Without groups attr, `groups` is unbound → +∞ (formulas must
    // name only bound dims).
    // Missing window attrs → output dims unknown → `out` unbound → ∞.
    let t2 = op_attrs(
        "conv2d",
        vec![shaped(&[1, 3, 32, 32]), shaped(&[8, 3, 3, 3])],
        json!({"groups": 1}),
    );
    assert_eq!(spec.price(&from_kernel(&t2)), f64::INFINITY);
}
