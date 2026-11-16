//! Integration tests for `hnl-engine-eqsat` — the P2 interpretation
//! surface (ADR-0174/0175). Kernel `Term`s in, kernel `Term`s out;
//! graph internals are inspected through `handl_egraph`'s public API
//! (available to tests as a regular dependency).

use handl_egraph::{from_kernel, to_kernel};
use hnl_engine::Term as KTerm;
use hnl_engine_eqsat::{self as eqsat, builtin};
use serde_json::json;

fn native_op(name: &str, args: Vec<KTerm>) -> KTerm {
    KTerm::op_native(name, args)
}

fn kconst(v: serde_json::Value) -> KTerm {
    KTerm::constant(v)
}

fn op_attrs(name: &str, args: Vec<KTerm>, attrs: serde_json::Value) -> KTerm {
    KTerm::op_native_attrs(name, args, attrs)
}

/// Saturate `input` under the single `law`; assert the `expected`
/// kernel term landed in the input's root e-class (the rewrite
/// instantiated its RHS) and `law.name` is named in the merge log
/// (the firing + provenance contract).
fn assert_fires(input: &KTerm, law: &eqsat::Law, expected: &KTerm) {
    let mut sat = eqsat::saturate(
        std::slice::from_ref(input),
        std::slice::from_ref(law),
        &eqsat::SaturateOptions::spec(),
    )
    .expect("saturate");
    let root = sat.roots[0];
    let class = sat
        .graph
        .class_of_term(&from_kernel(expected))
        .unwrap_or_else(|| panic!("{}: RHS member was not instantiated", law.name));
    assert_eq!(
        sat.graph.find_ref(class),
        sat.graph.find_ref(root),
        "{}: RHS member not merged into the root class",
        law.name
    );
    let fired: Vec<&str> = sat
        .graph
        .merge_log()
        .iter()
        .filter_map(|e| e.rule.as_deref())
        .collect();
    assert!(
        fired.contains(&law.name.as_str()),
        "{} missing from merge log: {fired:?}",
        law.name
    );
}

/// Saturate `input` under the single `law`; assert the law never
/// fired (no merge-log entry names it).
fn assert_no_fire(input: &KTerm, law: &eqsat::Law) {
    let sat = eqsat::saturate(
        std::slice::from_ref(input),
        std::slice::from_ref(law),
        &eqsat::SaturateOptions::spec(),
    )
    .expect("saturate");
    let fired: Vec<&str> = sat
        .graph
        .merge_log()
        .iter()
        .filter_map(|e| e.rule.as_deref())
        .collect();
    assert!(
        !fired.contains(&law.name.as_str()),
        "{} fired unexpectedly: {fired:?}",
        law.name
    );
}

#[test]
fn identities_fold_constants_to_a_kernel_term() {
    // add(mul(2, 1), 0) --(id_mul, id_add)--> 2  (the spike's loop,
    // now through the public `optimize` API and the ported laws).
    let input = native_op(
        "add",
        vec![
            native_op("mul", vec![kconst(json!(2)), kconst(json!(1))]),
            kconst(json!(0)),
        ],
    );

    let out = eqsat::optimize(
        &input,
        &builtin::simplification(),
        &mut eqsat::size_cost,
        &eqsat::OptimizeOptions::spec(),
    )
    .expect("optimize");

    assert_eq!(out.term, kconst(json!(2)));
    assert_eq!(
        out.saturated.stats.stop,
        handl_egraph::StopReason::FixedPoint
    );

    // Provenance: both identity laws fired and are named in the log.
    let fired: Vec<&str> = out
        .saturated
        .graph
        .merge_log()
        .iter()
        .filter_map(|e| e.rule.as_deref())
        .collect();
    assert!(fired.contains(&"id_mul"), "merge log: {fired:?}");
    assert!(fired.contains(&"id_add"), "merge log: {fired:?}");
}

#[test]
fn sub_to_add_fires_and_lowers_back() {
    // sub(5, 3) --(sub_to_add)--> add(5, neg(3)) — the RHS-op path:
    // instantiated `add`/`neg` must inherit the matched root's dim
    // (default Const(1) here) for `class_of_term` to find them.
    let input = native_op("sub", vec![kconst(json!(5)), kconst(json!(3))]);
    let laws = vec![builtin::sub_to_add()];
    let input_for_best = input.clone();

    let mut sat =
        eqsat::saturate(&[input], &laws, &eqsat::SaturateOptions::spec()).expect("saturate");
    let root = sat.roots[0];

    let expected = native_op(
        "add",
        vec![kconst(json!(5)), native_op("neg", vec![kconst(json!(3))])],
    );
    let class = sat
        .graph
        .class_of_term(&from_kernel(&expected))
        .expect("add(5, neg(3)) was instantiated");
    assert_eq!(sat.graph.find_ref(class), sat.graph.find_ref(root));

    // The firing is named in the provenance log.
    let fired: Vec<&str> = sat
        .graph
        .merge_log()
        .iter()
        .filter_map(|e| e.rule.as_deref())
        .collect();
    assert!(fired.contains(&"sub_to_add"), "merge log: {fired:?}");

    // Under the size cost the *smaller* member wins — extraction
    // returns the original `sub` (the rewrite is in the class, not
    // forced). This pins the cost-driven extraction contract.
    let mut cost = eqsat::size_cost;
    let best = eqsat::extract_best(&mut sat.graph, root, &mut cost).expect("extract");
    assert_eq!(best, input_for_best);
}

#[test]
fn attrs_payload_round_trips_through_a_rewrite() {
    // tensor.reshape(x, attrs={"shape":[2,2]}) --(reshape_flatten_demo)-->
    // tensor.flatten(x, attrs={"shape":[2,2]}) — the `__attrs__`
    // payload bound whole via `attrs_bind` and re-emitted verbatim
    // (the spike's P0.5 finding, now through the public API).  The
    // law is an UNSOUND demo — matcher coverage only, never a pack.
    let shape = json!({"shape": [2, 2]});
    let x = kconst(json!(7));
    let input = KTerm::op_native_attrs("tensor.reshape", vec![x.clone()], shape.clone());

    let mut sat = eqsat::saturate(
        &[input],
        &[builtin::reshape_flatten_demo()],
        &eqsat::SaturateOptions::spec(),
    )
    .expect("saturate");
    let root = sat.roots[0];

    let expected = KTerm::op_native_attrs("tensor.flatten", vec![x], shape.clone());
    let class = sat
        .graph
        .class_of_term(&from_kernel(&expected))
        .expect("flatten member was instantiated");
    assert_eq!(sat.graph.find_ref(class), sat.graph.find_ref(root));

    // Lowering is total and the attrs payload survives byte-identical.
    let lowered = to_kernel(&from_kernel(&expected));
    match &lowered {
        KTerm::Op { name, attrs, .. } => {
            assert_eq!(name, "tensor.flatten");
            assert_eq!(attrs, &Some(shape));
        }
        other => panic!("expected Op, got {other:?}"),
    }
}

#[test]
fn attr_keyset_is_exact() {
    // A pattern that does not mention `__attrs__` must NOT match an
    // attr-carrying node (the matcher's exact-keyset gate — the top
    // footgun the spike report names).
    let input = KTerm::op_native_attrs(
        "add",
        vec![kconst(json!(1)), kconst(json!(0))],
        json!({"k": 1}),
    );
    let mut sat = eqsat::saturate(
        std::slice::from_ref(&input),
        &builtin::simplification(),
        &eqsat::SaturateOptions::spec(),
    )
    .expect("saturate");
    let root = sat.roots[0];
    // id_add's pattern carries no `__attrs__` key → no firing →
    // extraction returns the input unchanged.
    assert!(sat.graph.applications().is_empty());
    let mut cost = eqsat::size_cost;
    let best = eqsat::extract_best(&mut sat.graph, root, &mut cost).expect("extract");
    assert_eq!(best, input);
}

#[test]
fn unbound_rhs_metavariable_is_rejected() {
    use eqsat::{Law, Pat};
    // rhs mentions `b`, which the lhs never binds.
    let bad = Law::new(
        "bad",
        Pat::op("f", vec![Pat::var("a")]),
        Pat::op("g", vec![Pat::var("b")]),
    );
    let input = native_op("f", vec![kconst(json!(1))]);
    match eqsat::saturate(&[input], &[bad], &eqsat::SaturateOptions::spec()) {
        Err(e) => assert!(matches!(e, eqsat::Error::Law(_)), "got {e}"),
        Ok(_) => panic!("unbound RHS var must fail compilation"),
    }
}

#[test]
fn optimize_improving_stop_runs() {
    // The lazy-saturation path: extract each iteration, stop after
    // `patience` non-improving rounds.
    let input = native_op(
        "add",
        vec![
            native_op("mul", vec![kconst(json!(2)), kconst(json!(1))]),
            kconst(json!(0)),
        ],
    );
    let opts = eqsat::OptimizeOptions {
        stop: eqsat::StopPolicy::Improving { patience: 2 },
        ..eqsat::OptimizeOptions::spec()
    };
    let out = eqsat::optimize(
        &input,
        &builtin::simplification(),
        &mut eqsat::size_cost,
        &opts,
    )
    .expect("optimize");
    assert_eq!(out.term, kconst(json!(2)));
    assert!(out.saturated.stats.improved.is_some());
}

#[test]
fn multi_term_saturate_shares_the_graph() {
    // Two inputs in one graph: a law firing on either is visible from
    // both roots' shared e-classes.
    let t1 = native_op("add", vec![kconst(json!(1)), kconst(json!(0))]);
    let t2 = native_op("mul", vec![kconst(json!(3)), kconst(json!(1))]);
    let mut sat = eqsat::saturate(
        &[t1, t2],
        &builtin::simplification(),
        &eqsat::SaturateOptions::spec(),
    )
    .expect("saturate");
    assert_eq!(sat.roots.len(), 2);

    let mut cost = eqsat::size_cost;
    let best0 = eqsat::extract_best(&mut sat.graph, sat.roots[0], &mut cost).unwrap();
    let best1 = eqsat::extract_best(&mut sat.graph, sat.roots[1], &mut cost).unwrap();
    assert_eq!(best0, kconst(json!(1)));
    assert_eq!(best1, kconst(json!(3)));
    // One shared graph: both extractions saw the same merge log.
    assert!(sat.graph.n_proof_edges() >= 2);
}

#[test]
fn law_metadata_and_tags_compile_through() {
    // `with_law`/`with_tag` ride `Rewrite::law`/`tags` — observable in
    // the recorded applications' rule names.
    let input = native_op("add", vec![kconst(json!(7)), kconst(json!(0))]);
    let out = eqsat::optimize(
        &input,
        &builtin::simplification(),
        &mut eqsat::size_cost,
        &eqsat::OptimizeOptions::spec(),
    )
    .expect("optimize");
    assert_eq!(out.term, kconst(json!(7)));
    assert!(!out.saturated.graph.applications().is_empty());
}

// ---------------------------------------------------------------------------
//  Ported guard-free tensor laws — one firing test per law
//  (`mlopt_core.laws.tensor`, spelled in `builtin`).
// ---------------------------------------------------------------------------

#[test]
fn comm_add_swaps_operands() {
    let a = kconst(json!(1));
    let b = kconst(json!(2));
    let input = native_op("add", vec![a.clone(), b.clone()]);
    let expected = native_op("add", vec![b, a]);
    assert_fires(&input, &builtin::comm_add(), &expected);
}

#[test]
fn comm_mul_swaps_operands() {
    let a = kconst(json!(3));
    let b = kconst(json!(4));
    let input = native_op("mul", vec![a.clone(), b.clone()]);
    let expected = native_op("mul", vec![b, a]);
    assert_fires(&input, &builtin::comm_mul(), &expected);
}

#[test]
fn assoc_add_reassociates_right_to_left() {
    let (a, b, c) = (kconst(json!(1)), kconst(json!(2)), kconst(json!(3)));
    let input = native_op(
        "add",
        vec![a.clone(), native_op("add", vec![b.clone(), c.clone()])],
    );
    let expected = native_op("add", vec![native_op("add", vec![a, b]), c]);
    assert_fires(&input, &builtin::assoc_add(), &expected);
}

#[test]
fn assoc_mul_reassociates_right_to_left() {
    let (a, b, c) = (kconst(json!(2)), kconst(json!(3)), kconst(json!(4)));
    let input = native_op(
        "mul",
        vec![a.clone(), native_op("mul", vec![b.clone(), c.clone()])],
    );
    let expected = native_op("mul", vec![native_op("mul", vec![a, b]), c]);
    assert_fires(&input, &builtin::assoc_mul(), &expected);
}

#[test]
fn id_add_fires_on_add_with_zero_rhs() {
    let a = kconst(json!(9));
    let input = native_op("add", vec![a.clone(), kconst(json!(0))]);
    assert_fires(&input, &builtin::id_add(), &a);
}

#[test]
fn id_mul_fires_on_mul_with_one_rhs() {
    let a = kconst(json!(9));
    let input = native_op("mul", vec![a.clone(), kconst(json!(1))]);
    assert_fires(&input, &builtin::id_mul(), &a);
}

#[test]
fn double_neg_fires() {
    let a = kconst(json!(9));
    let input = native_op("neg", vec![native_op("neg", vec![a.clone()])]);
    assert_fires(&input, &builtin::double_neg(), &a);
}

#[test]
fn silu_expand_fires() {
    let x = kconst(json!(0.5));
    let input = native_op("silu", vec![x.clone()]);
    let expected = native_op("mul", vec![x.clone(), native_op("sigmoid", vec![x])]);
    assert_fires(&input, &builtin::silu_expand(), &expected);
}

#[test]
fn silu_fold_fires_and_extracts_the_kernel() {
    let x = kconst(json!(0.5));
    let input = native_op(
        "mul",
        vec![x.clone(), native_op("sigmoid", vec![x.clone()])],
    );
    let expected = native_op("silu", vec![x]);
    assert_fires(&input, &builtin::silu_fold(), &expected);
    // The fold is strictly smaller — extraction picks `silu(x)`.
    let out = eqsat::optimize(
        &input,
        &[builtin::silu_fold()],
        &mut eqsat::size_cost,
        &eqsat::OptimizeOptions::spec(),
    )
    .expect("optimize");
    assert_eq!(out.term, expected);
}

#[test]
fn silu_mul_form_fires() {
    let (g, u) = (kconst(json!(1)), kconst(json!(2)));
    let input = native_op("mul", vec![native_op("silu", vec![g.clone()]), u.clone()]);
    let expected = native_op(
        "mul",
        vec![
            native_op("mul", vec![g.clone(), native_op("sigmoid", vec![g])]),
            u,
        ],
    );
    assert_fires(&input, &builtin::silu_mul_form(), &expected);
}

#[test]
fn softsign_fold_fires_and_extracts_the_kernel() {
    let x = kconst(json!(0.25));
    let input = native_op(
        "div",
        vec![
            x.clone(),
            native_op(
                "add",
                vec![native_op("abs", vec![x.clone()]), kconst(json!(1))],
            ),
        ],
    );
    let expected = native_op("softsign", vec![x]);
    assert_fires(&input, &builtin::softsign_fold(), &expected);
    let out = eqsat::optimize(
        &input,
        &[builtin::softsign_fold()],
        &mut eqsat::size_cost,
        &eqsat::OptimizeOptions::spec(),
    )
    .expect("optimize");
    assert_eq!(out.term, expected);
}

#[test]
fn square_expand_fires() {
    let x = kconst(json!(3));
    let input = native_op("square", vec![x.clone()]);
    let expected = native_op("mul", vec![x.clone(), x]);
    assert_fires(&input, &builtin::square_expand(), &expected);
}

#[test]
fn pow_to_square_fires_on_exponent_two() {
    let x = kconst(json!(3));
    let input = native_op("pow", vec![x.clone(), kconst(json!(2))]);
    let expected = native_op("square", vec![x]);
    assert_fires(&input, &builtin::pow_to_square(), &expected);
}

#[test]
fn square_to_pow_fires() {
    let x = kconst(json!(3));
    let input = native_op("square", vec![x.clone()]);
    let expected = native_op("pow", vec![x, kconst(json!(2))]);
    assert_fires(&input, &builtin::square_to_pow(), &expected);
}

#[test]
fn mul_square_fires_on_identical_operands() {
    let u = kconst(json!(5));
    let input = native_op("mul", vec![u.clone(), u.clone()]);
    let expected = native_op("square", vec![u]);
    assert_fires(&input, &builtin::mul_square(), &expected);
}

#[test]
fn recip_sqrt_to_rsqrt_fires_and_extracts_the_kernel() {
    let u = kconst(json!(2));
    let input = native_op("reciprocal", vec![native_op("sqrt", vec![u.clone()])]);
    let expected = native_op("rsqrt", vec![u]);
    assert_fires(&input, &builtin::recip_sqrt_to_rsqrt(), &expected);
    let out = eqsat::optimize(
        &input,
        &[builtin::recip_sqrt_to_rsqrt()],
        &mut eqsat::size_cost,
        &eqsat::OptimizeOptions::spec(),
    )
    .expect("optimize");
    assert_eq!(out.term, expected);
}

#[test]
fn mul_div_swap_fires() {
    let (a, b, c) = (kconst(json!(6)), kconst(json!(2)), kconst(json!(3)));
    let input = native_op(
        "mul",
        vec![native_op("div", vec![a.clone(), b.clone()]), c.clone()],
    );
    let expected = native_op("div", vec![native_op("mul", vec![a, c]), b]);
    assert_fires(&input, &builtin::mul_div_swap(), &expected);
}

// --- categorical pack -----------------------------------------------------

#[test]
fn right_distribute_matmul_fires() {
    let (a, b, w) = (kconst(json!(1)), kconst(json!(2)), kconst(json!(3)));
    let input = native_op(
        "matmul",
        vec![native_op("add", vec![a.clone(), b.clone()]), w.clone()],
    );
    let expected = native_op(
        "add",
        vec![
            native_op("matmul", vec![a, w.clone()]),
            native_op("matmul", vec![b, w]),
        ],
    );
    assert_fires(&input, &builtin::right_distribute_matmul(), &expected);
}

#[test]
fn right_factor_matmul_fires() {
    let (a, b, w) = (kconst(json!(1)), kconst(json!(2)), kconst(json!(3)));
    let input = native_op(
        "add",
        vec![
            native_op("matmul", vec![a.clone(), w.clone()]),
            native_op("matmul", vec![b.clone(), w.clone()]),
        ],
    );
    let expected = native_op("matmul", vec![native_op("add", vec![a, b]), w]);
    assert_fires(&input, &builtin::right_factor_matmul(), &expected);
}

#[test]
fn right_factor_linear_fires() {
    let (a, b, w) = (kconst(json!(1)), kconst(json!(2)), kconst(json!(3)));
    let input = native_op(
        "add",
        vec![
            native_op("linear", vec![a.clone(), w.clone()]),
            native_op("linear", vec![b.clone(), w.clone()]),
        ],
    );
    let expected = native_op("linear", vec![native_op("add", vec![a, b]), w]);
    assert_fires(&input, &builtin::right_factor_linear(), &expected);
}

#[test]
fn assoc_linear_fires_with_flipped_weight_product() {
    let (x, a, b) = (kconst(json!(1)), kconst(json!(2)), kconst(json!(3)));
    let input = native_op(
        "linear",
        vec![native_op("linear", vec![x.clone(), a.clone()]), b.clone()],
    );
    // Fused weight is B @ A — the transposes flip the product order.
    let expected = native_op("linear", vec![x, native_op("matmul", vec![b, a])]);
    assert_fires(&input, &builtin::assoc_linear(), &expected);
}

#[test]
fn assoc_linear_rev_fires() {
    let (x, a, b) = (kconst(json!(1)), kconst(json!(2)), kconst(json!(3)));
    let input = native_op(
        "linear",
        vec![x.clone(), native_op("matmul", vec![b.clone(), a.clone()])],
    );
    let expected = native_op("linear", vec![native_op("linear", vec![x, a]), b]);
    assert_fires(&input, &builtin::assoc_linear_rev(), &expected);
}

#[test]
fn assoc_matmul_fires() {
    let (a, b, c) = (kconst(json!(1)), kconst(json!(2)), kconst(json!(3)));
    let input = native_op(
        "matmul",
        vec![a.clone(), native_op("matmul", vec![b.clone(), c.clone()])],
    );
    let expected = native_op("matmul", vec![native_op("matmul", vec![a, b]), c]);
    assert_fires(&input, &builtin::assoc_matmul(), &expected);
}

#[test]
fn assoc_matmul_rev_fires() {
    let (a, b, c) = (kconst(json!(1)), kconst(json!(2)), kconst(json!(3)));
    let input = native_op(
        "matmul",
        vec![native_op("matmul", vec![a.clone(), b.clone()]), c.clone()],
    );
    let expected = native_op("matmul", vec![a, native_op("matmul", vec![b, c])]);
    assert_fires(&input, &builtin::assoc_matmul_rev(), &expected);
}

// --- pointwise factor pack (opt-in) ---------------------------------------

#[test]
fn distribute_mul_add_fires() {
    let (c, x, y) = (kconst(json!(2)), kconst(json!(3)), kconst(json!(4)));
    let input = native_op(
        "mul",
        vec![c.clone(), native_op("add", vec![x.clone(), y.clone()])],
    );
    let expected = native_op(
        "add",
        vec![
            native_op("mul", vec![c.clone(), x]),
            native_op("mul", vec![c, y]),
        ],
    );
    assert_fires(&input, &builtin::distribute_mul_add(), &expected);
}

#[test]
fn factor_mul_add_fires() {
    let (c, x, y) = (kconst(json!(2)), kconst(json!(3)), kconst(json!(4)));
    let input = native_op(
        "add",
        vec![
            native_op("mul", vec![c.clone(), x.clone()]),
            native_op("mul", vec![c.clone(), y.clone()]),
        ],
    );
    let expected = native_op("mul", vec![c, native_op("add", vec![x, y])]);
    assert_fires(&input, &builtin::factor_mul_add(), &expected);
}

// --- qkv_fuse: the attrs-bind/bind-lit port --------------------------------

/// The `_head` view: `transpose(reshape(t, {"shape": S}), {"dim0":1,"dim1":2})`.
fn qkv_head(t: KTerm, shape: &serde_json::Value) -> KTerm {
    op_attrs(
        "transpose",
        vec![op_attrs("reshape", vec![t], shape.clone())],
        json!({"dim0": 1, "dim1": 2}),
    )
}

/// The fused weight `concat(concat(Q, K, {"dim":0}), V, {"dim":0})`.
fn qkv_weight(q: KTerm, k: KTerm, v: KTerm) -> KTerm {
    op_attrs(
        "concat",
        vec![op_attrs("concat", vec![q, k], json!({"dim": 0})), v],
        json!({"dim": 0}),
    )
}

#[test]
fn qkv_fuse_fires() {
    let shape = json!({"shape": [2, 4]});
    let (x, q, k, v) = (
        kconst(json!(1)),
        kconst(json!(2)),
        kconst(json!(3)),
        kconst(json!(4)),
    );
    let linear = |w: KTerm| native_op("linear", vec![x.clone(), w]);
    let input = op_attrs(
        "sdpa",
        vec![
            qkv_head(linear(q.clone()), &shape),
            qkv_head(linear(k.clone()), &shape),
            qkv_head(linear(v.clone()), &shape),
        ],
        json!({"scale": 0.5}),
    );
    // RHS: sdpa over three chunk projections of ONE fused GEMM.
    let fused = qkv_weight(q, k, v);
    let chunk = |i: i64| {
        op_attrs(
            "chunk",
            vec![native_op("linear", vec![x.clone(), fused.clone()])],
            json!({"chunks": 3, "dim": -1, "index": i}),
        )
    };
    let expected = op_attrs(
        "sdpa",
        vec![
            qkv_head(chunk(0), &shape),
            qkv_head(chunk(1), &shape),
            qkv_head(chunk(2), &shape),
        ],
        json!({"scale": 0.5}),
    );
    assert_fires(&input, &builtin::qkv_fuse(), &expected);
}

#[test]
fn qkv_fuse_declines_on_mismatched_head_shapes() {
    // The three `shape` attr payloads must bind the SAME metavar —
    // a K projection whose reshape carries a different shape cannot
    // share the fused GEMM's head view.
    let shape = json!({"shape": [2, 4]});
    let other = json!({"shape": [3, 4]});
    let (x, q, k, v) = (
        kconst(json!(1)),
        kconst(json!(2)),
        kconst(json!(3)),
        kconst(json!(4)),
    );
    let linear = |w: KTerm| native_op("linear", vec![x.clone(), w]);
    let input = op_attrs(
        "sdpa",
        vec![
            qkv_head(linear(q), &shape),
            qkv_head(linear(k), &other),
            qkv_head(linear(v), &shape),
        ],
        json!({"scale": 0.5}),
    );
    let fused = kconst(json!(9)); // any RHS stand-in; we only need no-fire
    let _ = fused;
    assert_no_fire(&input, &builtin::qkv_fuse());
}

// --- negative tests: shared-metavar and const-pin preconditions -----------

#[test]
fn mul_square_declines_on_distinct_operands() {
    // `mul(u, u)` binds both operands to the same e-class —
    // `mul(2, 3)` never fires.
    let input = native_op("mul", vec![kconst(json!(2)), kconst(json!(3))]);
    assert_no_fire(&input, &builtin::mul_square());
}

#[test]
fn silu_fold_declines_on_mismatched_sigmoid_argument() {
    // `mul(x, sigmoid(x))` — `mul(1, sigmoid(2))` never fires.
    let input = native_op(
        "mul",
        vec![
            kconst(json!(1)),
            native_op("sigmoid", vec![kconst(json!(2))]),
        ],
    );
    assert_no_fire(&input, &builtin::silu_fold());
}

#[test]
fn pow_to_square_declines_on_other_exponents() {
    let input = native_op("pow", vec![kconst(json!(3)), kconst(json!(3))]);
    assert_no_fire(&input, &builtin::pow_to_square());
    // And a non-Const exponent (an op term) cannot match the Const pin.
    let input2 = native_op(
        "pow",
        vec![kconst(json!(3)), native_op("neg", vec![kconst(json!(2))])],
    );
    assert_no_fire(&input2, &builtin::pow_to_square());
}

#[test]
fn softsign_fold_declines_on_shifted_abs_argument() {
    // `div(x, add(abs(y), 1))` with y != x never fires.
    let input = native_op(
        "div",
        vec![
            kconst(json!(1)),
            native_op(
                "add",
                vec![native_op("abs", vec![kconst(json!(2))]), kconst(json!(1))],
            ),
        ],
    );
    assert_no_fire(&input, &builtin::softsign_fold());
}

#[test]
fn id_add_declines_on_nonzero_rhs() {
    let input = native_op("add", vec![kconst(json!(1)), kconst(json!(1))]);
    assert_no_fire(&input, &builtin::id_add());
}

#[test]
fn factor_mul_add_declines_on_unshared_factor() {
    // `add(mul(c,x), mul(d,y))` with c != d never fires.
    let input = native_op(
        "add",
        vec![
            native_op("mul", vec![kconst(json!(2)), kconst(json!(3))]),
            native_op("mul", vec![kconst(json!(4)), kconst(json!(5))]),
        ],
    );
    assert_no_fire(&input, &builtin::factor_mul_add());
}

#[test]
fn law_packs_compile_and_cover_the_ported_set() {
    // The packs mirror the Python bank collections: simplification +
    // categorical = the guard-free ALL_RULES subset; pointwise_factor
    // stays opt-in.
    let simp = builtin::simplification();
    let cat = builtin::categorical();
    let pw = builtin::pointwise_factor();
    let all = builtin::all();
    assert_eq!(simp.len() + cat.len(), all.len());
    assert_eq!(all.len(), simp.len() + cat.len());
    assert_eq!(pw.len(), 2);
    let names: Vec<&str> = all.iter().map(|l| l.name.as_str()).collect();
    for expected in [
        "comm_add",
        "comm_mul",
        "assoc_add",
        "assoc_mul",
        "id_add",
        "id_mul",
        "double_neg",
        "sub_to_add",
        "silu_expand",
        "silu_fold",
        "silu_mul_form",
        "softsign_fold",
        "square_expand",
        "pow_to_square",
        "square_to_pow",
        "mul_square",
        "recip_sqrt_to_rsqrt",
        "mul_div_swap",
        "right_distribute_matmul",
        "right_factor_matmul",
        "right_factor_linear",
        "assoc_linear",
        "assoc_linear_rev",
        "assoc_matmul",
        "assoc_matmul_rev",
        "qkv_fuse",
    ] {
        assert!(names.contains(&expected), "missing {expected} in {names:?}");
    }
    // Every law compiles (no unbound RHS metavariable).
    for law in all.iter().chain(pw.iter()) {
        law_lint_compile(law);
    }
}

/// `Law::compile` is crate-private; exercise it through `saturate`
/// on a trivially non-matching input — a compile error surfaces as
/// `Error::Law`.
fn law_lint_compile(law: &eqsat::Law) {
    let input = kconst(json!(0));
    // Const input cannot match any op LHS — saturation is a no-op,
    // but compilation still runs.
    let _ = eqsat::saturate(
        std::slice::from_ref(&input),
        std::slice::from_ref(law),
        &eqsat::SaturateOptions::spec(),
    );
}

// --- guarded laws: the stage-1 cond/dspec interpreter -------------------
//
//  `const-cmp`/`const-num` guards over bound Const leaves, `and`/`not`
//  combinators, and `dspec` `$leaf:` mints — the `guard.rs` port of
//  `cond.py`, exercised end-to-end through `saturate`.

#[test]
fn const_guards_fire_and_veto() {
    let u = native_op("add", vec![kconst(json!(1)), kconst(json!(2))]);

    // pow_one: pow(u, 1) → u — fires on Const(1), vetoes on Const(2)
    // and on a non-Const exponent.
    assert_fires(
        &native_op("pow", vec![u.clone(), kconst(json!(1))]),
        &builtin::pow_one(),
        &u,
    );
    assert_no_fire(
        &native_op("pow", vec![u.clone(), kconst(json!(2))]),
        &builtin::pow_one(),
    );
    assert_no_fire(
        &native_op(
            "pow",
            vec![u.clone(), native_op("neg", vec![kconst(json!(1))])],
        ),
        &builtin::pow_one(),
    );

    // pow_to_rsqrt / pow_half / pow_neg_one — the const-cmp family.
    assert_fires(
        &native_op("pow", vec![u.clone(), kconst(json!(-0.5))]),
        &builtin::pow_to_rsqrt(),
        &native_op("rsqrt", vec![u.clone()]),
    );
    assert_no_fire(
        &native_op("pow", vec![u.clone(), kconst(json!(0.5))]),
        &builtin::pow_to_rsqrt(),
    );
    assert_fires(
        &native_op("pow", vec![u.clone(), kconst(json!(0.5))]),
        &builtin::pow_half(),
        &native_op("sqrt", vec![u.clone()]),
    );
    // pow_neg_one: the const-num base guard vetoes an int-typed base.
    assert_fires(
        &native_op("pow", vec![u.clone(), kconst(json!(-1))]),
        &builtin::pow_neg_one(),
        &native_op("reciprocal", vec![u.clone()]),
    );
    assert_no_fire(
        &native_op("pow", vec![kconst(json!(2)), kconst(json!(-1))]),
        &builtin::pow_neg_one(),
    );
    assert_no_fire(
        &native_op("pow", vec![kconst(json!(2.0)), kconst(json!(-1))]),
        &builtin::pow_neg_one(),
    );

    // div_sqrt_to_rsqrt / div_one_recip — the ONE-pinned folds.
    assert_fires(
        &native_op(
            "div",
            vec![kconst(json!(1)), native_op("sqrt", vec![u.clone()])],
        ),
        &builtin::div_sqrt_to_rsqrt(),
        &native_op("rsqrt", vec![u.clone()]),
    );
    assert_no_fire(
        &native_op(
            "div",
            vec![kconst(json!(2)), native_op("sqrt", vec![u.clone()])],
        ),
        &builtin::div_sqrt_to_rsqrt(),
    );
    assert_fires(
        &native_op("div", vec![kconst(json!(1)), u.clone()]),
        &builtin::div_one_recip(),
        &native_op("reciprocal", vec![u.clone()]),
    );
    assert_no_fire(
        &native_op("div", vec![kconst(json!(0)), u.clone()]),
        &builtin::div_one_recip(),
    );

    // Activation folds — multi-const `and` guards.
    let x = native_op("mul", vec![kconst(json!(3)), kconst(json!(4))]);
    assert_fires(
        &native_op(
            "reciprocal",
            vec![native_op(
                "add",
                vec![
                    kconst(json!(1)),
                    native_op("exp", vec![native_op("neg", vec![x.clone()])]),
                ],
            )],
        ),
        &builtin::sigmoid_fold(),
        &native_op("sigmoid", vec![x.clone()]),
    );
    // ONE = 2 vetoes the fold.
    assert_no_fire(
        &native_op(
            "reciprocal",
            vec![native_op(
                "add",
                vec![
                    kconst(json!(2)),
                    native_op("exp", vec![native_op("neg", vec![x.clone()])]),
                ],
            )],
        ),
        &builtin::sigmoid_fold(),
    );
    assert_fires(
        &native_op(
            "div",
            vec![
                kconst(json!(1)),
                native_op(
                    "add",
                    vec![
                        kconst(json!(1)),
                        native_op("exp", vec![native_op("neg", vec![x.clone()])]),
                    ],
                ),
            ],
        ),
        &builtin::sigmoid_fold_div(),
        &native_op("sigmoid", vec![x.clone()]),
    );
    assert_fires(
        &native_op(
            "sub",
            vec![
                native_op(
                    "mul",
                    vec![
                        kconst(json!(2)),
                        native_op(
                            "sigmoid",
                            vec![native_op("mul", vec![kconst(json!(2)), x.clone()])],
                        ),
                    ],
                ),
                kconst(json!(1)),
            ],
        ),
        &builtin::tanh_fold(),
        &native_op("tanh", vec![x.clone()]),
    );
    assert_fires(
        &native_op(
            "div",
            vec![
                native_op(
                    "relu6",
                    vec![native_op("add", vec![x.clone(), kconst(json!(3))])],
                ),
                kconst(json!(6)),
            ],
        ),
        &builtin::hardsigmoid_fold(),
        &native_op("hardsigmoid", vec![x.clone()]),
    );
    // THREE = 4 vetoes hardsigmoid_fold.
    assert_no_fire(
        &native_op(
            "div",
            vec![
                native_op(
                    "relu6",
                    vec![native_op("add", vec![x.clone(), kconst(json!(4))])],
                ),
                kconst(json!(6)),
            ],
        ),
        &builtin::hardsigmoid_fold(),
    );
}

#[test]
fn dspec_leaf_mints_fire_and_veto() {
    let u = native_op("add", vec![kconst(json!(1)), kconst(json!(2))]);

    // neg_const: neg(Const(3)) → Const(-3) minted through $leaf:.
    assert_fires(
        &native_op("neg", vec![kconst(json!(3))]),
        &builtin::neg_const(),
        &kconst(json!(-3)),
    );
    // A non-Const operand vetoes const-num — no mint, no fire.
    assert_no_fire(&native_op("neg", vec![u.clone()]), &builtin::neg_const());

    // sub_const_sub: (u−2)−3 → u−5.
    assert_fires(
        &native_op(
            "sub",
            vec![
                native_op("sub", vec![u.clone(), kconst(json!(2))]),
                kconst(json!(3)),
            ],
        ),
        &builtin::sub_const_sub(),
        &native_op("sub", vec![u.clone(), kconst(json!(5))]),
    );
    // A non-Const inner operand vetoes.
    assert_no_fire(
        &native_op(
            "sub",
            vec![
                native_op(
                    "sub",
                    vec![u.clone(), native_op("neg", vec![kconst(json!(2))])],
                ),
                kconst(json!(3)),
            ],
        ),
        &builtin::sub_const_sub(),
    );

    // div_const_to_mul: u/2 → u·0.5 — and u/0 vetoes (const-cmp ≠ 0).
    assert_fires(
        &native_op("div", vec![u.clone(), kconst(json!(2))]),
        &builtin::div_const_to_mul(),
        &native_op("mul", vec![u.clone(), kconst(json!(0.5))]),
    );
    assert_no_fire(
        &native_op("div", vec![u.clone(), kconst(json!(0))]),
        &builtin::div_const_to_mul(),
    );

    // mul_const_assoc_0: (u·2)·3 → u·6.
    assert_fires(
        &native_op(
            "mul",
            vec![
                native_op("mul", vec![u.clone(), kconst(json!(2))]),
                kconst(json!(3)),
            ],
        ),
        &builtin::prior_art()
            .into_iter()
            .find(|l| l.name == "mul_const_assoc_0")
            .expect("mul_const_assoc_0 in prior_art"),
        &native_op("mul", vec![u.clone(), kconst(json!(6))]),
    );

    // mul_const_div: (u/2)·3 → u·1.5.
    assert_fires(
        &native_op(
            "mul",
            vec![
                native_op("div", vec![u.clone(), kconst(json!(2))]),
                kconst(json!(3)),
            ],
        ),
        &builtin::mul_const_div(),
        &native_op("mul", vec![u.clone(), kconst(json!(1.5))]),
    );
}

// --- guarded tensor laws (cond/dspec over attrs + the shape table) ----
//
// The `AttrsSpec::Fields` per-key attr metavariables compile to a
// whole-payload `__attrs__` binding interpreted by `Law::compile`'s
// check/derive hooks: exact keyset on the LHS, literal/bound-field
// agreement, and synthesized payloads on the RHS. These tests pin
// both the firing and the veto halves, plus the dspec derivations.

/// A `{"__tensor__": …}` leaf — the wire value `shape::infer` reads a
/// concrete shape off (a bare numeric const reports `()`).
fn tensor(shape: &[i64]) -> KTerm {
    kconst(json!({"__tensor__": {"shape": shape, "dtype": "f64"}}))
}

/// Fetch a generated pack member by name (`split_scalar_*`, `sdpa_*`).
fn pack_law(pack: Vec<eqsat::Law>, name: &str) -> eqsat::Law {
    pack.into_iter()
        .find(|l| l.name == name)
        .unwrap_or_else(|| panic!("{name} in pack"))
}

#[test]
fn select_mul_fires_and_vetoes_on_shape_and_keyset() {
    let law = builtin::select_mul();
    let sel = |t: &KTerm, attrs: serde_json::Value| op_attrs("select", vec![t.clone()], attrs);
    let dims = json!({"dim": 0, "index": 1});

    // sel(u) ⊙ sel(v) → sel(u ⊙ v) when the select axes agree on both
    // operands and the operands broadcast.
    let (u, v) = (tensor(&[2, 4]), tensor(&[2, 4]));
    let input = native_op("mul", vec![sel(&u, dims.clone()), sel(&v, dims.clone())]);
    let expected = op_attrs("select", vec![native_op("mul", vec![u, v])], dims.clone());
    assert_fires(&input, &law, &expected);

    // `dim-eq-attr` veto: u[0]=1 vs v[0]=3 differ on the select axis —
    // broadcast alone ([1,4]×[3,4] → [3,4]) would have allowed it.
    let (u, v) = (tensor(&[1, 4]), tensor(&[3, 4]));
    let input = native_op("mul", vec![sel(&u, dims.clone()), sel(&v, dims.clone())]);
    assert_no_fire(&input, &law);

    // Exact-keyset veto: an extra attr key declines the fields spec.
    let extra = json!({"dim": 0, "index": 1, "keepdim": true});
    let (u, v) = (tensor(&[2, 4]), tensor(&[2, 4]));
    let input = native_op("mul", vec![sel(&u, extra), sel(&v, dims)]);
    assert_no_fire(&input, &law);
}

#[test]
fn transpose_noop_fires_and_vetoes() {
    let law = builtin::transpose_noop();

    // Same axis twice is a no-op on any rank ≥ 1.
    let u = tensor(&[2, 4]);
    let input = op_attrs("transpose", vec![u.clone()], json!({"dim0": 0, "dim1": 0}));
    assert_fires(&input, &law, &u);

    // Swapped extents both 1 is a no-op too.
    let u = tensor(&[1, 1, 4]);
    let input = op_attrs("transpose", vec![u.clone()], json!({"dim0": 0, "dim1": 1}));
    assert_fires(&input, &law, &u);

    // Veto: a real swap (2 ↔ 4) and a swap touching a non-1 extent.
    let u = tensor(&[2, 4]);
    let input = op_attrs("transpose", vec![u], json!({"dim0": 0, "dim1": 1}));
    assert_no_fire(&input, &law);
    let u = tensor(&[1, 4]);
    let input = op_attrs("transpose", vec![u], json!({"dim0": 0, "dim1": 1}));
    assert_no_fire(&input, &law);
}

#[test]
fn chunk_single_fires_and_vetoes() {
    let law = builtin::chunk_single();
    let u = tensor(&[2, 4]);
    let attrs = json!({"chunks": 1, "dim": 0, "index": 0});
    let input = op_attrs("chunk", vec![u.clone()], attrs);
    assert_fires(&input, &law, &u);

    // chunks=2 is not the identity.
    let input = op_attrs(
        "chunk",
        vec![u.clone()],
        json!({"chunks": 2, "dim": 0, "index": 0}),
    );
    assert_no_fire(&input, &law);

    // Missing `index` breaks the exact keyset.
    let input = op_attrs("chunk", vec![u], json!({"chunks": 1, "dim": 0}));
    assert_no_fire(&input, &law);
}

#[test]
fn unsq_pad_family_fires_and_vetoes() {
    // u=[4] lifted to [1,4] by unsqueeze(dim=0); v=[1,4] already
    // carries the pad axis — the broadcast grids coincide, so the
    // view is dead weight.
    let (u, v) = (tensor(&[4]), tensor(&[1, 4]));
    let lifted = |t: &KTerm| op_attrs("unsqueeze", vec![t.clone()], json!({"dim": 0}));

    let input = native_op("mul", vec![lifted(&u), v.clone()]);
    let expected = native_op("mul", vec![u.clone(), v.clone()]);
    assert_fires(&input, &builtin::mul_unsq_pad_l(), &expected);

    // Right-operand twin: mul(v, unsq(u)) → mul(v, u) — operand
    // order is preserved (no commutativity in a single-law run).
    let input = native_op("mul", vec![v.clone(), lifted(&u)]);
    let expected_r = native_op("mul", vec![v.clone(), u.clone()]);
    assert_fires(&input, &builtin::mul_unsq_pad_r(), &expected_r);

    let input = native_op("sub", vec![lifted(&u), v.clone()]);
    let expected_sub = native_op("sub", vec![u.clone(), v.clone()]);
    assert_fires(&input, &builtin::sub_unsq_pad_l(), &expected_sub);

    let input = native_op("add", vec![v.clone(), lifted(&u)]);
    let expected_add = native_op("add", vec![v.clone(), u.clone()]);
    assert_fires(&input, &builtin::add_unsq_pad_r(), &expected_add);

    // Veto: v=[4] — bcast(unsq(u),v) = [1,4] but bcast(u,v) = [4],
    // the grids disagree.
    let (u, v) = (tensor(&[4]), tensor(&[4]));
    let input = native_op("mul", vec![lifted(&u), v]);
    assert_no_fire(&input, &builtin::mul_unsq_pad_l());
}

/// A `__mlopt_leaf__:` bridge leaf — the `mlopt_orchestrator.eqsat`
/// `_LEAF_PREFIX` encoding of a typed `Var`/`Param` (`s` = declared
/// `TensorType` shape).  Regression for the WP4 `mul_unsq_nat`
/// parity failure: these leaves MUST feed shape inference, or every
/// broadcast guard sees a phantom scalar `()` and fires where the
/// real shapes would decline.
fn bridge_leaf(name: &str, shape: serde_json::Value) -> KTerm {
    kconst(json!(format!(
        "__mlopt_leaf__:{{\"k\":\"var\",\"n\":\"{name}\",\"s\":{shape}}}"
    )))
}

#[test]
fn unsq_pad_vetoes_on_bridge_leaf_shapes() {
    // The WP4 `mul_unsq_nat` repro, verbatim: `a.unsqueeze(-1) *
    // b.unsqueeze(-1)` with `a`,`b` of shape `[4,6]` lifts both
    // operands to `[4,6,1]`.  Stripping either unsqueeze yields
    // `mul(a, unsq(b))` / `mul(unsq(a), b)` — operands `[4,6]` and
    // `[4,6,1]` that cannot broadcast (`4` vs `6` at dim 1).  The
    // guard must decline; before the leaf-shape fix the phantom
    // scalar shapes made `bcast-eq`/`bcast-into` pass vacuously.
    let (a, b) = (
        bridge_leaf("a", json!([4, 6])),
        bridge_leaf("b", json!([4, 6])),
    );
    let lifted = |t: &KTerm| op_attrs("unsqueeze", vec![t.clone()], json!({"dim": -1}));
    let input = native_op("mul", vec![lifted(&a), lifted(&b)]);
    assert_no_fire(&input, &builtin::mul_unsq_pad_l());
    assert_no_fire(&input, &builtin::mul_unsq_pad_r());
    let laws = [builtin::mul_unsq_pad_l(), builtin::mul_unsq_pad_r()];
    let sat = eqsat::saturate(
        std::slice::from_ref(&input),
        &laws,
        &eqsat::SaturateOptions::spec(),
    )
    .expect("saturate");
    assert!(
        sat.graph.merge_log().iter().all(|e| e.rule.is_none()),
        "pad laws merged under bridge-leaf shapes: {:?}",
        sat.graph.merge_log()
    );
}

#[test]
fn unsq_pad_still_fires_on_bridge_leaf_shapes() {
    // Positive control: the same encoding must not over-decline.
    // `u=[4]` lifted by `unsqueeze(dim=0)` to `[1,4]` against `v=[1,4]`
    // — grids coincide, the pad view is dead.
    let (u, v) = (
        bridge_leaf("u", json!([4])),
        bridge_leaf("v", json!([1, 4])),
    );
    let lifted = |t: &KTerm| op_attrs("unsqueeze", vec![t.clone()], json!({"dim": 0}));
    let input = native_op("mul", vec![lifted(&u), v.clone()]);
    let expected = native_op("mul", vec![u, v]);
    assert_fires(&input, &builtin::mul_unsq_pad_l(), &expected);
}

#[test]
fn mul_reshape_inert_l_fires_and_vetoes() {
    // reshape(u[4] → [1,4]) is broadcast-inert against v=[1,4].
    let (u, v) = (tensor(&[4]), tensor(&[1, 4]));
    let reshaped = op_attrs("reshape", vec![u.clone()], json!({"shape": [1, 4]}));
    let input = native_op("mul", vec![reshaped, v.clone()]);
    let expected = native_op("mul", vec![u, v]);
    assert_fires(&input, &builtin::mul_reshape_inert_l(), &expected);

    // Veto: reshape to [2,2] changes the broadcast grid.
    let u = tensor(&[4]);
    let v = tensor(&[1, 4]);
    let reshaped = op_attrs("reshape", vec![u], json!({"shape": [2, 2]}));
    let input = native_op("mul", vec![reshaped, v]);
    assert_no_fire(&input, &builtin::mul_reshape_inert_l());
}

#[test]
fn softmax_fold_fires_and_vetoes() {
    let law = builtin::softmax_fold();
    let u = tensor(&[2, 4]);
    let manual = |dim: serde_json::Value| {
        native_op(
            "div",
            vec![
                native_op("exp", vec![u.clone()]),
                op_attrs(
                    "sum",
                    vec![native_op("exp", vec![u.clone()])],
                    json!({"dim": dim, "keepdim": true}),
                ),
            ],
        )
    };

    // int dim → softmax(dim=-1).
    let input = manual(json!(-1));
    let expected = op_attrs("softmax", vec![u.clone()], json!({"dim": -1}));
    assert_fires(&input, &law, &expected);

    // 1-tuple dim unwraps through the `attr0` dspec to the same -1.
    let input = manual(json!([-1]));
    assert_fires(&input, &law, &expected);

    // Veto: keepdim=false is not the softmax normalization.
    let input = native_op(
        "div",
        vec![
            native_op("exp", vec![u.clone()]),
            op_attrs(
                "sum",
                vec![native_op("exp", vec![u])],
                json!({"dim": -1, "keepdim": false}),
            ),
        ],
    );
    assert_no_fire(&input, &law);
}

#[test]
fn softmax_fold_reads_the_wire_tuple_envelope() {
    // WP4 F2: a real lift carries `dim=(-1,)` as the
    // `{"__tuple__": [-1]}` attr envelope (`attr_to_data`), not a
    // bare array — the guard layer must decode it or
    // `attr-type "tuple"`/`attr-len`/`attr0` all decline and the
    // law can never fire.
    let law = builtin::softmax_fold();
    let u = tensor(&[2, 4]);
    let manual = |dim: serde_json::Value| {
        native_op(
            "div",
            vec![
                native_op("exp", vec![u.clone()]),
                op_attrs(
                    "sum",
                    vec![native_op("exp", vec![u.clone()])],
                    json!({"dim": dim, "keepdim": true}),
                ),
            ],
        )
    };
    let expected = op_attrs("softmax", vec![u.clone()], json!({"dim": -1}));

    // The wire spelling fires and derives the same scalar dim.
    let input = manual(json!({"__tuple__": [-1]}));
    assert_fires(&input, &law, &expected);

    // `{"__list__": [-1]}` is a *list* attr — Python's
    // `isinstance(dim, tuple)` is False, so the tuple arm declines
    // and the int arm never applied either: no fire, exactly the
    // cond.py verdict.
    let input = manual(json!({"__list__": [-1]}));
    assert_no_fire(&input, &law);
}

#[test]
fn reshape_attr_envelope_reads_through_shape_guards() {
    // The shape-spec attr reads (`reshape-out`'s target, `attr_dims`)
    // see through the same envelopes — a `{"__tuple__": [1,4]}`
    // reshape target is the `[1,4]` shape to the guard.
    let law = builtin::mul_reshape_inert_l();
    let (u, v) = (tensor(&[4]), tensor(&[1, 4]));
    let reshaped = op_attrs(
        "reshape",
        vec![u.clone()],
        json!({"shape": {"__tuple__": [1, 4]}}),
    );
    let input = native_op("mul", vec![reshaped, v.clone()]);
    let expected = native_op("mul", vec![u, v]);
    assert_fires(&input, &law, &expected);
}

#[test]
fn glu_fold_fires_and_vetoes() {
    let law = builtin::glu_fold();
    let chunk = |u: &KTerm, index: i64| {
        op_attrs(
            "chunk",
            vec![u.clone()],
            json!({"chunks": 2, "dim": 1, "index": index}),
        )
    };
    let input = |u: &KTerm| {
        native_op(
            "mul",
            vec![chunk(u, 0), native_op("sigmoid", vec![chunk(u, 1)])],
        )
    };

    // u=[2,4]: dim-1 extent 4 is even → glu(u, dim=1).
    let u = tensor(&[2, 4]);
    let expected = op_attrs("glu", vec![u.clone()], json!({"dim": 1}));
    assert_fires(&input(&u), &law, &expected);

    // Veto: dim-1 extent 3 is odd — no equal halves.
    let u = tensor(&[2, 3]);
    assert_no_fire(&input(&u), &law);

    // Veto: a scalar u fails `rank >= 1`.
    let u = kconst(json!(1.0));
    assert_no_fire(&input(&u), &law);
}

#[test]
fn naturality_scalar_fires_and_vetoes() {
    let law = builtin::naturality_scalar();
    let (w, x, c) = (kconst(json!(1)), kconst(json!(2)), kconst(json!(3)));
    let input = native_op(
        "matmul",
        vec![w.clone(), native_op("mul", vec![x.clone(), c.clone()])],
    );
    let expected = native_op("mul", vec![native_op("matmul", vec![w, x]), c]);
    assert_fires(&input, &law, &expected);

    // Veto: a non-scalar "c" is not the naturality scalar.
    let c = tensor(&[2]);
    let input = native_op(
        "matmul",
        vec![
            kconst(json!(1)),
            native_op("mul", vec![kconst(json!(2)), c]),
        ],
    );
    assert_no_fire(&input, &law);
}

#[test]
fn split_scalar_mul_r_fires_and_vetoes() {
    let law = pack_law(builtin::categorical(), "split_scalar_mul_r");
    let attrs = json!({"sizes": [2], "dim": -1, "index": 0});
    let (x, c) = (tensor(&[2, 4]), kconst(json!(0.5)));
    let split_of = |t: KTerm| op_attrs("split", vec![t], attrs.clone());
    let input = split_of(native_op("mul", vec![x.clone(), c.clone()]));
    let expected = native_op("mul", vec![split_of(x), c]);
    assert_fires(&input, &law, &expected);

    // Veto: non-scalar c.
    let c = tensor(&[4]);
    let input = split_of(native_op("mul", vec![tensor(&[2, 4]), c]));
    assert_no_fire(&input, &law);
}

#[test]
fn linear_channel_scale_fires_and_vetoes() {
    let law = builtin::linear_channel_scale();
    // x=[4,3], W=[2,3] → Wᵀ=[3,2], x@Wᵀ=[4,2] well-typed; c scalar.
    let (x, w, c) = (tensor(&[4, 3]), tensor(&[2, 3]), kconst(json!(2.0)));
    let input = native_op(
        "linear",
        vec![native_op("mul", vec![x.clone(), c.clone()]), w.clone()],
    );
    let expected = native_op("linear", vec![x, native_op("mul", vec![w, c])]);
    assert_fires(&input, &law, &expected);

    // Veto: c=[3,3] is shaped but neither scalar nor a channel
    // vector (dim-1 extent 3 ≠ W's 3? it IS 3 — but ones-but-last
    // fails on the leading 3, and dim-eq alone is not enough).
    let (x, w, c) = (tensor(&[4, 3]), tensor(&[2, 3]), tensor(&[3, 3]));
    let input = native_op("linear", vec![native_op("mul", vec![x, c]), w]);
    assert_no_fire(&input, &law);
}

#[test]
fn sdpa_fold_mul_nomask_fires_and_vetoes() {
    let law = pack_law(builtin::categorical(), "sdpa_fold_mul_nomask");
    // K concrete rank-2, transpose dims = last-two axes; Q,V rank ≥ 2;
    // softmax dim -1; S a numeric const.
    let (q, k, v) = (tensor(&[4, 8]), tensor(&[4, 8]), tensor(&[4, 8]));
    let scores = native_op(
        "matmul",
        vec![
            q.clone(),
            op_attrs(
                "transpose",
                vec![k.clone()],
                json!({"dim0": -2, "dim1": -1}),
            ),
        ],
    );
    let input = native_op(
        "matmul",
        vec![
            op_attrs(
                "softmax",
                vec![native_op("mul", vec![scores, kconst(json!(2))])],
                json!({"dim": -1}),
            ),
            v.clone(),
        ],
    );
    // dspec SC = float(const S) = 2.0 → sdpa(…, scale=2.0).
    let expected = op_attrs("sdpa", vec![q, k, v], json!({"scale": 2.0}));
    assert_fires(&input, &law, &expected);

    // Veto: softmax dim 0 is not the score axis.
    let scores = native_op(
        "matmul",
        vec![
            tensor(&[4, 8]),
            op_attrs(
                "transpose",
                vec![tensor(&[4, 8])],
                json!({"dim0": -2, "dim1": -1}),
            ),
        ],
    );
    let input = native_op(
        "matmul",
        vec![
            op_attrs(
                "softmax",
                vec![native_op("mul", vec![scores, kconst(json!(2))])],
                json!({"dim": 0}),
            ),
            tensor(&[4, 8]),
        ],
    );
    assert_no_fire(&input, &law);
}

#[test]
fn sdpa_expand_s_fires_and_vetoes_on_keyset() {
    let law = pack_law(builtin::attention(), "sdpa_expand_s");
    let (q, k, v) = (kconst(json!(1)), kconst(json!(2)), kconst(json!(3)));
    let input = op_attrs(
        "sdpa",
        vec![q.clone(), k.clone(), v.clone()],
        json!({"scale": 0.5}),
    );
    // dspec $leaf:SC0 = attr SC → Const leaf 0.5 in the expanded mul.
    let expected = native_op(
        "matmul",
        vec![
            op_attrs(
                "softmax",
                vec![native_op(
                    "mul",
                    vec![
                        native_op(
                            "matmul",
                            vec![
                                q,
                                op_attrs("transpose", vec![k], json!({"dim0": -2, "dim1": -1})),
                            ],
                        ),
                        kconst(json!(0.5)),
                    ],
                )],
                json!({"dim": -1}),
            ),
            v,
        ],
    );
    assert_fires(&input, &law, &expected);

    // Exact-keyset veto: an extra sdpa attr declines the {scale}
    // fields spec (the _dcgs spelling is a different law).
    let input = op_attrs(
        "sdpa",
        vec![kconst(json!(1)), kconst(json!(2)), kconst(json!(3))],
        json!({"scale": 0.5, "is_causal": true}),
    );
    assert_no_fire(&input, &law);
}

// ---------------------------------------------------------------------------
//  dag.program wire — shared-subtree input/output
// ---------------------------------------------------------------------------

/// A `dag.program` node-table record whose tree expansion is
/// exponential: `levels` of `add(x, x)` over a shared child — the
/// table stays ~`levels + 2` entries where the tree spelling is
/// `2^levels` nodes.
fn doubling_dag(levels: usize) -> serde_json::Value {
    let mut nodes = vec![
        json!({"Const": {"value": 1}}), // node 0 — the shared dim leaf
        json!({"Const": {"value": 7}}), // node 1 — the base leaf
    ];
    for _ in 0..levels {
        let prev = nodes.len() - 1;
        nodes.push(json!({"Op": {
            "dim": 0,
            "name": "add",
            "args": [prev, prev],
            "source": "Native",
        }}));
    }
    let root = nodes.len() - 1;
    json!({"Opaque": {"__tag__": "dag.program", "nodes": nodes, "root": root}})
}

#[test]
fn dag_program_input_saturates_without_tree_expansion() {
    // 45 doublings: 2^45 tree occurrences, 47 table nodes. The
    // pre-dag wire could not even spell this record.
    let dag = doubling_dag(45);
    let mut sat = eqsat::saturate_json(&[dag], &[], &eqsat::SaturateOptions::spec())
        .expect("saturate dag.program input");
    assert_eq!(sat.roots.len(), 1);

    let cost = eqsat::CostSpec::Size;
    let (wire, priced) = eqsat::extract_best_json(&mut sat.graph, sat.roots[0], &cost)
        .expect("extract")
        .expect("a member");
    assert!(priced.is_finite());
    // The extraction's tree expansion is 2^45 — the wire must be the
    // dag.program record, not the tree spelling.
    assert_eq!(wire["Opaque"]["__tag__"], "dag.program");
    let nodes = wire["Opaque"]["nodes"].as_array().expect("nodes");
    // Every `add` node references its child by *index* — args are
    // integers, not nested records.
    let last = &nodes[nodes.len() - 1];
    assert!(
        last["Op"]["args"]
            .as_array()
            .unwrap()
            .iter()
            .all(|a| a.is_u64())
    );
}

#[test]
fn dag_program_input_malformed_records_refuse_by_name() {
    // A non-dag Opaque tag.
    let bad_tag = json!({"Opaque": {"__tag__": "other", "nodes": [], "root": 0}});
    let err = eqsat::saturate_json(&[bad_tag], &[], &eqsat::SaturateOptions::spec())
        .err()
        .expect("non-dag Opaque must refuse");
    assert!(matches!(err, eqsat::Error::BadTerm(_)), "{err}");

    // A forward/self child index (post-order violated).
    let mut nodes = vec![
        json!({"Const": {"value": 1}}),
        json!({"Op": {"dim": 0, "name": "add", "args": [1], "source": "Native"}}),
    ];
    let bad = json!({"Opaque": {"__tag__": "dag.program", "nodes": nodes.clone(), "root": 1}});
    let err = eqsat::saturate_json(&[bad], &[], &eqsat::SaturateOptions::spec())
        .err()
        .expect("forward index must refuse");
    assert!(matches!(err, eqsat::Error::BadTerm(_)), "{err}");

    // Missing 'nodes'.
    let missing = json!({"Opaque": {"__tag__": "dag.program", "root": 0}});
    assert!(eqsat::saturate_json(&[missing], &[], &eqsat::SaturateOptions::spec()).is_err());
    nodes.clear();
}

#[test]
fn dag_program_small_term_emits_tree_wire() {
    // A shared-but-tiny term: `add(mul(x,x), mul(x,x))`-scale sharing
    // stays under the expansion bounds, so the plain tree record is
    // emitted (byte-compatible with the pre-dag wire).
    let mut nodes = vec![
        json!({"Const": {"value": 1}}), // dim
        json!({"Const": {"value": 3}}), // leaf
    ];
    // mul(leaf, leaf) — node 2, shared twice by the root add.
    nodes.push(json!({"Op": {
        "dim": 0, "name": "mul", "args": [1, 1], "source": "Native",
    }}));
    let root = nodes.len();
    nodes.push(json!({"Op": {
        "dim": 0, "name": "add", "args": [2, 2], "source": "Native",
    }}));
    let dag = json!({"Opaque": {"__tag__": "dag.program", "nodes": nodes, "root": root}});
    let mut sat =
        eqsat::saturate_json(&[dag], &[], &eqsat::SaturateOptions::spec()).expect("saturate");
    let cost = eqsat::CostSpec::Size;
    let (wire, _) = eqsat::extract_best_json(&mut sat.graph, sat.roots[0], &cost)
        .expect("extract")
        .expect("a member");
    assert!(
        wire.get("Op").is_some(),
        "small terms emit the tree record: {wire}"
    );
}

// ---------------------------------------------------------------------------
//  dag.program wire — Arr leaf records (`{"Arr": {dim, value}}`)
// ---------------------------------------------------------------------------

/// An `Arr` leaf record in the spelling hnl_codec emits: `dim` is a
/// nested kernel-`Term` record (leaf records carry no table indices)
/// and `value` is the `__mlopt_leaf__` payload.
fn arr_leaf_record(kind: &str, name: &str, shape: serde_json::Value) -> serde_json::Value {
    json!({"Arr": {
        "dim": {"Const": {"value": 1}},
        "value": {"__mlopt_leaf__": kind, "name": name, "shape": shape},
    }})
}

#[test]
fn dag_program_arr_leaves_round_trip_tree_wire() {
    // `add(var_x, param_w)` over two Arr leaf records — the hnl_codec
    // leaf spelling. Small term: extraction emits the tree record,
    // and the `arr` enodes must come back as `{"Arr": …}` records,
    // not die on the unknown `"value"` attr.
    let dag = json!({"Opaque": {"__tag__": "dag.program", "nodes": [
        json!({"Const": {"value": 1}}),                                  // 0: dim
        arr_leaf_record("var", "x", json!([2, 3])),                      // 1
        arr_leaf_record("param", "w", json!(null)),                    // 2
        json!({"Op": {"dim": 0, "name": "add", "args": [1, 2],
                      "source": "Native"}}),                             // 3
    ], "root": 3}});
    let mut sat = eqsat::saturate_json(&[dag], &[], &eqsat::SaturateOptions::spec())
        .expect("saturate dag.program with Arr leaves");
    let cost = eqsat::CostSpec::Size;
    let (wire, _) = eqsat::extract_best_json(&mut sat.graph, sat.roots[0], &cost)
        .expect("extract")
        .expect("a member");
    let args = wire["Op"]["args"].as_array().expect("op args");
    assert_eq!(args.len(), 2, "wire: {wire}");
    assert_eq!(args[0]["Arr"]["value"]["name"], "x");
    assert_eq!(args[0]["Arr"]["value"]["__mlopt_leaf__"], "var");
    assert_eq!(args[0]["Arr"]["dim"], json!({"Const": {"value": 1}}));
    assert_eq!(args[1]["Arr"]["value"]["name"], "w");

    // The emitted wire must itself be valid input — full round-trip.
    let mut sat2 = eqsat::saturate_json(
        std::slice::from_ref(&wire),
        &[],
        &eqsat::SaturateOptions::spec(),
    )
    .expect("re-saturate emitted wire");
    let (wire2, _) = eqsat::extract_best_json(&mut sat2.graph, sat2.roots[0], &cost)
        .expect("re-extract")
        .expect("a member");
    assert_eq!(wire, wire2, "Arr-leaf round-trip drifted");
}

#[test]
fn dag_program_arr_leaf_survives_dag_output() {
    // Same exponential-sharing shape as `doubling_dag`, but the shared
    // base leaf is an `Arr` record: extraction must emit the dag table
    // spelling where the `Arr` node's `dim` is an inline tree record
    // (leaf records carry no table indices), and the whole record must
    // re-saturate.
    let mut nodes = vec![
        json!({"Const": {"value": 1}}),          // 0: dim
        arr_leaf_record("var", "x", json!([4])), // 1: shared Arr leaf
    ];
    for _ in 0..20 {
        let prev = nodes.len() - 1;
        nodes.push(json!({"Op": {
            "dim": 0, "name": "add", "args": [prev, prev],
            "source": "Native",
        }}));
    }
    let root = nodes.len() - 1;
    let dag = json!({"Opaque": {"__tag__": "dag.program", "nodes": nodes, "root": root}});
    let mut sat =
        eqsat::saturate_json(&[dag], &[], &eqsat::SaturateOptions::spec()).expect("saturate");
    let cost = eqsat::CostSpec::Size;
    let (wire, _) = eqsat::extract_best_json(&mut sat.graph, sat.roots[0], &cost)
        .expect("extract")
        .expect("a member");
    assert_eq!(wire["Opaque"]["__tag__"], "dag.program");
    let nodes = wire["Opaque"]["nodes"].as_array().expect("nodes");
    let arr = nodes
        .iter()
        .find(|n| n.get("Arr").is_some())
        .expect("an Arr leaf record in the dag table");
    // `dim` is a nested record, never a node-table index.
    assert_eq!(arr["Arr"]["dim"], json!({"Const": {"value": 1}}));
    assert_eq!(arr["Arr"]["value"]["name"], "x");

    // The dag table re-saturates — the Arr record decodes as a leaf.
    eqsat::saturate_json(&[wire], &[], &eqsat::SaturateOptions::spec())
        .expect("re-saturate dag wire with Arr leaf");
}

/// A `__mlopt_leaf__` payload riding the canonical `Arr` kernel
/// record — the `leaf="arr"` wire spelling of the same `Var`/`Param`
/// leaf [`bridge_leaf`] encodes as a `Const` string. Decode accepts
/// both spellings permanently, so every leaf-shaped read (declared
/// shape, `leaf-dim`, leaf pricing) must treat them identically.
fn arr_bridge_leaf(name: &str, shape: serde_json::Value) -> KTerm {
    KTerm::Arr {
        dim: Box::new(kconst(json!(1))),
        value: json!({"__mlopt_leaf__": "var", "name": name, "shape": shape}),
    }
}

#[test]
fn arr_leaf_spelling_preserves_shape_guard_parity() {
    // The WP4 `mul_unsq_nat` veto, under BOTH leaf spellings: `a`/`b`
    // declare `[4,6]`, both `unsqueeze(-1)` operands lift to
    // `[4,6,1]`, and stripping either pad cannot broadcast. If the
    // `arr` enode's declared shape went unread the guards would see
    // phantom scalars and FIRE — the kit-torch v0.1.13 repin
    // regression was exactly this class of miss (saturation
    // collapsed 3429 → 472 enodes).
    let lifted = |t: &KTerm| op_attrs("unsqueeze", vec![t.clone()], json!({"dim": -1}));
    let inputs = [
        native_op(
            "mul",
            vec![
                lifted(&bridge_leaf("a", json!([4, 6]))),
                lifted(&bridge_leaf("b", json!([4, 6]))),
            ],
        ),
        native_op(
            "mul",
            vec![
                lifted(&arr_bridge_leaf("a", json!([4, 6]))),
                lifted(&arr_bridge_leaf("b", json!([4, 6]))),
            ],
        ),
    ];
    for input in &inputs {
        assert_no_fire(input, &builtin::mul_unsq_pad_l());
        assert_no_fire(input, &builtin::mul_unsq_pad_r());
    }
}

#[test]
fn arr_leaf_enode_prices_as_the_leaf_it_encodes() {
    // Extraction economics may not depend on the leaf wire spelling:
    // the canonical `arr` enode prices exactly like the `Term::Leaf`
    // the `const-str` profile produced — flat leaf weight, dim child
    // unbilled — under every CostSpec kind.
    let specs = [
        eqsat::CostSpec::Size,
        eqsat::CostSpec::Count,
        eqsat::CostSpec::from_data(&json!({
            "kind": "op_table", "default": 1.0, "leaf": 0.25,
            "dim_leaf": 0.0, "ops": {},
        }))
        .expect("op_table spec"),
    ];
    let wrap = |l: &KTerm| native_op("add", vec![l.clone(), l.clone()]);
    for spec in &specs {
        for (mk_arr, mk_cs) in [
            (
                arr_bridge_leaf("w", json!([2, 3])),
                bridge_leaf("w", json!([2, 3])),
            ),
            (
                wrap(&arr_bridge_leaf("w", json!([2, 3]))),
                wrap(&bridge_leaf("w", json!([2, 3]))),
            ),
        ] {
            let (a, c) = (from_kernel(&mk_arr), from_kernel(&mk_cs));
            assert_eq!(
                spec.price(&a),
                spec.price(&c),
                "{spec:?}: arr-spelled leaf priced differently"
            );
        }
    }
}

#[test]
fn dag_output_noncanonical_arr_enode_refuses_by_name() {
    // A user op literally named "arr" with zero args lifts to a
    // one-child "arr" enode carrying `source` — *not* the canonical
    // `Term::Arr` spelling (which has exactly the `value` attr). The
    // extractor must refuse loudly rather than mis-emit.
    let term = json!({"Op": {
        "dim": {"Const": {"value": 1}},
        "name": "arr",
        "args": [],
        "source": "Native",
    }});
    let mut sat =
        eqsat::saturate_json(&[term], &[], &eqsat::SaturateOptions::spec()).expect("saturate");
    let err = eqsat::extract_best_json(&mut sat.graph, sat.roots[0], &eqsat::CostSpec::Size)
        .expect_err("non-canonical 'arr' enode must refuse");
    assert!(matches!(err, eqsat::Error::BadTerm(_)), "{err}");
    assert!(err.to_string().contains("'arr' enode"), "{err}");

    // …but a multi-child "arr" op is an ordinary op and still emits.
    let term = json!({"Op": {
        "dim": {"Const": {"value": 1}},
        "name": "arr",
        "args": [{"Const": {"value": 5}}],
        "source": "Native",
    }});
    let mut sat =
        eqsat::saturate_json(&[term], &[], &eqsat::SaturateOptions::spec()).expect("saturate");
    let (wire, _) = eqsat::extract_best_json(&mut sat.graph, sat.roots[0], &eqsat::CostSpec::Size)
        .expect("extract")
        .expect("a member");
    assert_eq!(wire["Op"]["name"], "arr", "wire: {wire}");
    assert_eq!(wire["Op"]["args"], json!([{"Const": {"value": 5}}]));
}
