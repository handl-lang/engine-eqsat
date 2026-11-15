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
    // tensor.reshape(x, attrs={"shape":[2,2]}) --(reshape_flatten)-->
    // tensor.flatten(x, attrs={"shape":[2,2]}) — the `__attrs__`
    // payload bound whole via `attrs_bind` and re-emitted verbatim
    // (the spike's P0.5 finding, now through the public API).
    let shape = json!({"shape": [2, 2]});
    let x = kconst(json!(7));
    let input = KTerm::op_native_attrs("tensor.reshape", vec![x.clone()], shape.clone());

    let mut sat = eqsat::saturate(
        &[input],
        &[builtin::reshape_flatten()],
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
