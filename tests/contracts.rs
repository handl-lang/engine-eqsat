//! Tests for the P2 search contracts, stage 1 (design doc
//! `docs/design/p2-search-contracts.md`, ADR-0177): the law JSON
//! record codec, `CostSpec`, `LawSet`, the `SearchEngine` trait, and
//! the `LAW_FORMAT = 2` corpus — canonical in `handl-kit-eqsat`
//! (ADR-0177 §1), consumed here via `LAW_PACKS`.

use handl_egraph::from_kernel;
use hnl_engine::Term as KTerm;
use hnl_engine_eqsat::cost::{CostError, CostSpec, OpEntry};
use hnl_engine_eqsat::law::{self, Law, LawError, RecordError};
use hnl_engine_eqsat::lawset::LawSet;
use hnl_engine_eqsat::pat::{AttrsSpec, OpPat, Pat, PatError, PatSource};
use hnl_engine_eqsat::{self as eqsat, builtin};
use serde_json::{Value, json};

fn kconst(v: Value) -> KTerm {
    KTerm::constant(v)
}

/// The canonical `LAW_FORMAT = 2` corpus lives in `handl-kit-eqsat`
/// (`data/laws/*.json`, ADR-0177 §1); `laws/` in this repo keeps only
/// the `reshape_flatten_demo` matcher demo.
fn pack(name: &str) -> &'static str {
    handl_kit_eqsat::law_pack(name)
        .unwrap_or_else(|| panic!("kit-eqsat has no pack {name:?}"))
        .json
}

fn native_op(name: &str, args: Vec<KTerm>) -> KTerm {
    KTerm::op_native(name, args)
}

// ---------------------------------------------------------------------------
//  Pat codec
// ---------------------------------------------------------------------------

#[test]
fn pat_record_round_trips_all_node_kinds() {
    // mvar / const / op with dim, args, source, foreign, lit, attrs.
    let pat = Pat::Op(
        OpPat::new("tensor.reshape")
            .dim(Pat::var("d"))
            .arg(Pat::var("x"))
            .arg(Pat::constant(json!([2, 2])))
            .source(PatSource::Foreign("host.fn".to_string()))
            .lit(json!({"k": 1}))
            .attrs_lit(json!({"shape": [2, 2]})),
    );
    let data = pat.to_data();
    assert_eq!(Pat::from_data(&data).unwrap(), pat);

    // The record spelling is the authoring view, spelled out.
    assert_eq!(data["op"], "tensor.reshape");
    assert_eq!(data["dim"], json!({"mvar": "d"}));
    assert_eq!(data["source"], "foreign");
    assert_eq!(data["foreign"], "host.fn");
    assert_eq!(data["lit"], json!({"k": 1}));
    assert_eq!(data["attrs"], json!({"lit": {"shape": [2, 2]}}));

    // Leaf forms.
    assert_eq!(Pat::Var("a".into()).to_data(), json!({"mvar": "a"}));
    assert_eq!(Pat::constant(json!(0)).to_data(), json!({"const": 0}));
    assert_eq!(
        Pat::from_data(&json!({"const": null})).unwrap(),
        Pat::Const(Value::Null)
    );
}

#[test]
fn pat_attrs_bind_and_source_variants_round_trip() {
    let pat = Pat::Op(
        OpPat::new("reshape")
            .arg(Pat::var("t"))
            .attrs_bind("S")
            .source(PatSource::Judgment),
    );
    let back = Pat::from_data(&pat.to_data()).unwrap();
    assert_eq!(back, pat);
    match &back {
        Pat::Op(op) => {
            assert_eq!(op.attrs, Some(AttrsSpec::Bind("S".to_string())));
            assert_eq!(op.source, PatSource::Judgment);
        }
        _ => panic!("expected op"),
    }

    // Dynamic source.
    let dyn_pat = Pat::Op(OpPat::new("f").source(PatSource::Dynamic));
    assert_eq!(Pat::from_data(&dyn_pat.to_data()).unwrap(), dyn_pat);
}

#[test]
fn pat_decode_refuses_malformed_records() {
    // Not an object / unknown node.
    assert!(matches!(
        Pat::from_data(&json!([1, 2])),
        Err(PatError::UnknownNode(_))
    ));
    assert!(matches!(
        Pat::from_data(&json!({"bogus": 1})),
        Err(PatError::UnknownNode(_))
    ));
    // Bad source tag.
    assert!(matches!(
        Pat::from_data(&json!({"op": "f", "source": "warp"})),
        Err(PatError::UnknownSource(_))
    ));
    // foreign without a name.
    assert!(matches!(
        Pat::from_data(&json!({"op": "f", "source": "foreign"})),
        Err(PatError::MalformedOp(_))
    ));
    // Bad attrs spec.
    assert!(matches!(
        Pat::from_data(&json!({"op": "f", "attrs": {"other": 1}})),
        Err(PatError::UnknownAttrs(_))
    ));
    // args not an array.
    assert!(matches!(
        Pat::from_data(&json!({"op": "f", "args": {}})),
        Err(PatError::MalformedOp(_))
    ));
}

// ---------------------------------------------------------------------------
//  Law record codec
// ---------------------------------------------------------------------------

#[test]
fn law_record_round_trips_through_json() {
    for law in builtin::simplification()
        .into_iter()
        .chain([builtin::reshape_flatten_demo()])
    {
        let data = law.to_data();
        // LAW_FORMAT = 2 field set, spelled like serialize.py.
        for key in [
            "version",
            "kind",
            "name",
            "law",
            "lhs",
            "rhs",
            "cond",
            "dspec",
            "tags",
            "derivation",
            "cert",
            "error_bound",
            "bound_norm",
            "serializable",
            "missing_hooks",
        ] {
            assert!(data.get(key).is_some(), "{} missing {key}", law.name);
        }
        assert_eq!(data["version"], 2);
        assert_eq!(data["kind"], "law");
        let back = Law::from_data(&data).unwrap_or_else(|e| panic!("{}: {e}", law.name));
        assert_eq!(back, law, "{}", law.name);
        // Text round-trip too.
        let text = serde_json::to_string(&data).unwrap();
        assert_eq!(Law::from_json(&text).unwrap(), law);
    }
}

#[test]
fn law_record_carries_every_metadata_field() {
    let law = Law::new("eps_law", Pat::var("a"), Pat::var("a"))
        .with_law("approximate identity")
        .with_tag("epsilon")
        .with_derivation("step-1")
        .with_derivation("step-2")
        .with_cert(json!({"steps": 2}))
        .with_error_bound(1e-3, "spectral");
    let back = Law::from_data(&law.to_data()).unwrap();
    assert_eq!(back, law);
    assert_eq!(back.error_bound, Some(1e-3));
    assert_eq!(back.bound_norm, "spectral");
    assert_eq!(back.derivation, vec!["step-1", "step-2"]);
    assert_eq!(back.cert, Some(json!({"steps": 2})));
}

#[test]
fn guarded_records_round_trip_and_interpret() {
    // A record carrying guard trees loads with the data preserved —
    // never silently dropped — and stage-1 guards compile into the
    // check/derive hooks.
    let mut law = builtin::id_add();
    law.cond = Some(json!(["and", ["attr-eq", "dim", 1]]));
    law.dspec = Some(json!({"$attr:out": ["attr", "dim"]}));
    let back = Law::from_data(&law.to_data()).unwrap();
    assert_eq!(back.cond, law.cond);
    assert_eq!(back.dspec, law.dspec);
    assert!(back.compile().is_ok());

    // The tensor-value dspec channels still refuse by name — never approximated.
    let mut law3 = builtin::id_add();
    law3.dspec = Some(json!({"X": ["tval", "u"]}));
    assert!(matches!(
        law3.compile(),
        Err(LawError::GuardUnsupported { .. })
    ));

    // A malformed node refuses as malformed, not unsupported —
    // including a wrong-arity flat-pair-unsq (now a ported guard).
    let mut law4 = builtin::id_add();
    law4.cond = Some(json!(["leaf"])); // missing the term arg
    assert!(matches!(
        law4.compile(),
        Err(LawError::GuardMalformed { .. })
    ));
    let mut law4b = builtin::id_add();
    law4b.cond = Some(json!(["flat-pair-unsq", "x"]));
    assert!(matches!(
        law4b.compile(),
        Err(LawError::GuardMalformed { .. })
    ));
    let mut law5 = builtin::id_add();
    law5.dspec = Some(json!({"X": ["tapp", "f", 1]})); // tensor apply
    assert!(matches!(
        law5.compile(),
        Err(LawError::GuardUnsupported { .. })
    ));
}

#[test]
fn unserializable_records_refuse_at_fire_time() {
    // serializable:false — procedural hooks the data cannot carry.
    let mut law = builtin::id_add();
    law.serializable = false;
    law.missing_hooks.insert("check".to_string());
    let back = Law::from_data(&law.to_data()).unwrap();
    assert!(!back.serializable);
    assert!(matches!(
        back.compile(),
        Err(LawError::Unserializable { .. })
    ));

    // missing_hooks alone (serializable still true in the record)
    // also refuses — a named hook is a dropped hook.
    let mut law2 = builtin::id_add();
    law2.missing_hooks.insert("derive".to_string());
    assert!(matches!(
        law2.compile(),
        Err(LawError::Unserializable { .. })
    ));
}

#[test]
fn engine_names_the_record_refusals() {
    // saturate/optimize surface the named Error variants, not a
    // wrapped LawError.
    let guarded = {
        let mut l = builtin::id_add();
        l.dspec = Some(json!({"X": ["tval", "u"]}));
        l
    };
    let input = native_op("add", vec![kconst(json!(1)), kconst(json!(0))]);
    match eqsat::saturate(
        std::slice::from_ref(&input),
        &[guarded],
        &eqsat::SaturateOptions::spec(),
    ) {
        Err(e) => assert!(
            matches!(e, eqsat::Error::GuardUnsupported(ref n) if n == "id_add"),
            "got {e}"
        ),
        Ok(_) => panic!("guarded law must not fire"),
    }

    let mut flagged = builtin::id_add();
    flagged.serializable = false;
    flagged.missing_hooks.insert("check".to_string());
    match eqsat::optimize(
        &input,
        &[flagged],
        &mut eqsat::size_cost,
        &eqsat::OptimizeOptions::spec(),
    ) {
        Err(e) => assert!(
            matches!(e, eqsat::Error::UnserializableLaw(ref n) if n == "id_add"),
            "got {e}"
        ),
        Ok(_) => panic!("non-serializable law must not fire"),
    }
}

#[test]
fn law_record_decode_refuses_bad_records() {
    // Wrong version.
    let mut data = builtin::id_add().to_data();
    data["version"] = json!(1);
    assert!(matches!(
        Law::from_data(&data),
        Err(RecordError::Version(_))
    ));
    // Unknown kind.
    let mut data = builtin::id_add().to_data();
    data["kind"] = json!("abstraction");
    assert!(matches!(Law::from_data(&data), Err(RecordError::Kind(_))));
    // Missing name.
    let mut data = builtin::id_add().to_data();
    data.as_object_mut().unwrap().remove("name");
    assert!(matches!(
        Law::from_data(&data),
        Err(RecordError::Malformed(_))
    ));
    // Malformed lhs pattern surfaces as Pattern{side:"lhs"}.
    let mut data = builtin::id_add().to_data();
    data["lhs"] = json!({"bogus": 1});
    assert!(matches!(
        Law::from_data(&data),
        Err(RecordError::Pattern { side: "lhs", .. })
    ));
    // kind absent defaults to "law" (the Python object_kind posture).
    let mut data = builtin::id_add().to_data();
    data.as_object_mut().unwrap().remove("kind");
    assert!(Law::from_data(&data).is_ok());
}

#[test]
fn law_from_data_free_function_and_json_entry() {
    let data = builtin::double_neg().to_data();
    assert_eq!(law::from_data(&data).unwrap().name, "double_neg");
    let text = serde_json::to_string(&data).unwrap();
    assert_eq!(law::Law::from_json(&text).unwrap().name, "double_neg");
    assert!(matches!(
        Law::from_json("not json"),
        Err(RecordError::Json(_))
    ));
}

// ---------------------------------------------------------------------------
//  LawSet
// ---------------------------------------------------------------------------

#[test]
fn lawset_from_json_accepts_array_record_and_envelope() {
    let set = LawSet::from_laws(builtin::simplification());
    let text = serde_json::to_string_pretty(&set.to_data()).unwrap();

    // Array of records.
    let a = LawSet::from_json(&text).unwrap();
    assert_eq!(a, set);
    // {"laws": [...]} envelope.
    let env = format!("{{\"laws\": {text}, \"extra\": 1}}");
    assert_eq!(LawSet::from_json(&env).unwrap(), set);
    // Single record object.
    let one = serde_json::to_string(&set.laws()[0].to_data()).unwrap();
    let single = LawSet::from_json(&one).unwrap();
    assert_eq!(single.len(), 1);
    assert_eq!(single.laws()[0].name, set.laws()[0].name);
}

#[test]
fn lawset_reports_the_failing_member() {
    let mut good = builtin::id_add().to_data();
    good["name"] = json!("good");
    let mut bad = builtin::id_mul().to_data();
    bad["lhs"] = json!({"nope": true});
    bad["name"] = json!("bad_law");
    let text = serde_json::to_string(&json!([good, bad])).unwrap();
    match LawSet::from_json(&text) {
        Err(RecordError::SetMember { index, name, .. }) => {
            assert_eq!(index, 1);
            assert_eq!(name.as_deref(), Some("bad_law"));
        }
        other => panic!("expected SetMember, got {other:?}"),
    }
}

#[test]
fn lawset_compile_collects_rules_and_refuses_sets() {
    let set = LawSet::from_laws(vec![builtin::id_add(), builtin::id_mul()]);
    let rules = set.compile().unwrap();
    assert_eq!(rules.len(), 2);
    assert_eq!(rules[0].name.as_ref(), "id_add");

    let mut guarded = builtin::id_mul();
    guarded.dspec = Some(json!({"X": ["tapp", "f", 1]}));
    let bad = LawSet::from_laws(vec![guarded]);
    assert!(matches!(
        bad.compile(),
        Err(LawError::GuardUnsupported { .. })
    ));
}

// ---------------------------------------------------------------------------
//  CostSpec
// ---------------------------------------------------------------------------

#[test]
fn cost_count_and_size_price_internal_terms() {
    // add(mul(2, 1), 0): kernel -> e-graph term.
    let kernel = native_op(
        "add",
        vec![
            native_op("mul", vec![kconst(json!(2)), kconst(json!(1))]),
            kconst(json!(0)),
        ],
    );
    let eterm = from_kernel(&kernel);

    // size mirrors engine::size_cost: Const leaves 0, ops 1+children,
    // dim slot counted (its Const(1) dim is a Leaf -> 0).
    assert_eq!(CostSpec::Size.price(&eterm), eqsat::size_cost(&kernel));
    assert_eq!(CostSpec::Size.price(&eterm), 2.0);

    // count: every node 1 — add, mul, 3 const leaves, 2 dim leaves.
    assert_eq!(CostSpec::Count.price(&eterm), 7.0);
}

#[test]
fn cost_op_table_weights_views_and_dim_leaf() {
    let table = eqsat::OpTable {
        name: Some("t".to_string()),
        default: 1.0,
        leaf: 0.0,
        dim_leaf: Some(5.0),
        ops: [
            ("add".to_string(), OpEntry::Weight(10.0)),
            ("tensor.reshape".to_string(), OpEntry::View),
        ]
        .into_iter()
        .collect(),
    };
    let spec = CostSpec::OpTable(table);

    // add(1, 2): head 10 + dim leaf 5 + two const leaves 0 = 15.
    let add = from_kernel(&native_op("add", vec![kconst(json!(1)), kconst(json!(2))]));
    assert_eq!(spec.price(&add), 15.0);

    // mul is unlisted -> default 1 + dim_leaf 5 = 6.
    let mul = from_kernel(&native_op("mul", vec![kconst(json!(1)), kconst(json!(2))]));
    assert_eq!(spec.price(&mul), 6.0);

    // A view op is free but its children still price.
    let view = from_kernel(&KTerm::op_native_attrs(
        "tensor.reshape",
        vec![kconst(json!(1))],
        json!({"s": 1}),
    ));
    // head 0 (view) + dim_leaf 5 + arg leaf 0 + __attrs__ rides attrs
    // not children -> 5.
    assert_eq!(spec.price(&view), 5.0);

    // dim_leaf None falls back to leaf.
    let mut t2 = spec.clone();
    if let CostSpec::OpTable(ref mut tab) = t2 {
        tab.dim_leaf = None;
    }
    assert_eq!(t2.price(&add), 10.0);
}

#[test]
fn cost_flops_formula_defers_to_infinity() {
    // A "flops" entry is kept as data (deferred until shape
    // inference) and prices +inf — never under-priced.
    let spec = CostSpec::from_data(&json!({
        "kind": "op_table",
        "default": 1.0,
        "leaf": 0.0,
        "ops": {"matmul": {"flops": "2*m*n*k"}, "add": 2.0}
    }))
    .unwrap();
    let mm = from_kernel(&native_op(
        "matmul",
        vec![kconst(json!(1)), kconst(json!(2))],
    ));
    assert!(spec.price(&mm).is_infinite());

    // Round-trip preserves the formula.
    let back = CostSpec::from_data(&spec.to_data()).unwrap();
    assert_eq!(back, spec);
}

#[test]
fn cost_record_round_trips_and_refuses_unknown_kinds() {
    for spec in [
        CostSpec::Count,
        CostSpec::Size,
        CostSpec::OpTable(eqsat::OpTable {
            name: None,
            default: 0.5,
            leaf: 0.25,
            dim_leaf: None,
            ops: [("add".to_string(), OpEntry::Weight(3.0))]
                .into_iter()
                .collect(),
        }),
    ] {
        assert_eq!(CostSpec::from_data(&spec.to_data()).unwrap(), spec);
    }

    // Unknown kinds refused at load — named, never approximated.
    for kind in ["flops", "roofline", "fusion", "executor"] {
        let data = json!({"kind": kind});
        assert!(
            matches!(CostSpec::from_data(&data), Err(CostError::UnknownKind(_))),
            "{kind} must be refused"
        );
    }
    // Malformed records.
    assert!(matches!(
        CostSpec::from_data(&json!({"kind": 3})),
        Err(CostError::Malformed(_))
    ));
    assert!(matches!(
        CostSpec::from_data(&json!(["count"])),
        Err(CostError::Malformed(_))
    ));
    assert!(matches!(
        CostSpec::from_data(&json!({"kind": "op_table", "ops": {"f": "x"}})),
        Err(CostError::BadEntry { .. })
    ));
    // from_json text path.
    assert_eq!(
        CostSpec::from_json("{\"kind\": \"count\"}").unwrap(),
        CostSpec::Count
    );
    assert!(matches!(
        CostSpec::from_json("nope"),
        Err(CostError::Malformed(_))
    ));
}

#[test]
fn optimize_spec_prices_engine_side() {
    // The doc's quick-start term through the CostSpec path.
    let input = native_op(
        "add",
        vec![
            native_op("mul", vec![kconst(json!(2)), kconst(json!(1))]),
            kconst(json!(0)),
        ],
    );
    let set = LawSet::from_json(pack("simplification")).unwrap();
    let out = eqsat::optimize_spec(
        &input,
        set.laws(),
        &CostSpec::Size,
        &eqsat::OptimizeOptions::spec(),
    )
    .expect("optimize_spec");
    assert_eq!(out.term, kconst(json!(2)));

    // extract_best_spec on the saturated graph agrees.
    let mut sat = eqsat::saturate(
        std::slice::from_ref(&input),
        set.laws(),
        &eqsat::SaturateOptions::spec(),
    )
    .unwrap();
    let best = eqsat::extract_best_spec(&mut sat.graph, sat.roots[0], &CostSpec::Size)
        .expect("extract_best_spec");
    assert_eq!(best, kconst(json!(2)));
}

// ---------------------------------------------------------------------------
//  SearchEngine trait + the kit-eqsat LAW_FORMAT=2 corpus
// ---------------------------------------------------------------------------

#[test]
fn search_engine_trait_drives_the_corpus() {
    use eqsat::SearchEngine as _;
    let engine = eqsat::EGraphEngine;
    assert_eq!(engine.name(), "eqsat");

    let set = LawSet::from_json(pack("simplification")).unwrap();
    assert_eq!(set.len(), builtin::simplification().len());
    assert!(set.compile().is_ok());

    let input = native_op(
        "add",
        vec![
            native_op("mul", vec![kconst(json!(2)), kconst(json!(1))]),
            kconst(json!(0)),
        ],
    );
    let out = engine
        .optimize(
            &input,
            &set,
            &CostSpec::Size,
            &eqsat::OptimizeOptions::spec(),
        )
        .expect("engine.optimize");
    assert_eq!(out.term, kconst(json!(2)));

    let sat = engine
        .saturate(&[input], &set, &eqsat::SaturateOptions::spec())
        .expect("engine.saturate");
    assert_eq!(sat.roots.len(), 1);
}

#[test]
fn laws_corpus_files_load_and_match_the_builtin_packs() {
    // kit-eqsat pack "simplification" — the guard-free
    // SIMPLIFICATION_RULES ports, as records.
    let simpl = LawSet::from_json(pack("simplification")).unwrap();
    let expected = builtin::simplification();
    assert_eq!(simpl.laws(), expected.as_slice());

    // kit-eqsat pack "prior_art" — the stage-1-portable PRIOR_ART_RULES
    // slice: const-guarded folds whose `cond`/`dspec` trees are the
    // `cond.py` spelling carried as record data.
    let pa = LawSet::from_json(pack("prior_art")).unwrap();
    assert_eq!(pa.laws(), builtin::prior_art().as_slice());
    // Every guarded record compiles — the corpus exercises the whole
    // stage-1 vocabulary end to end.
    assert_eq!(pa.compile().unwrap().len(), pa.laws().len());

    // kit-eqsat pack "categorical" — the guard-free CATEGORICAL_RULES
    // slice (matmul/linear algebra + qkv_fuse).
    let cat = LawSet::from_json(pack("categorical")).unwrap();
    assert_eq!(cat.laws(), builtin::categorical().as_slice());
    assert_eq!(cat.compile().unwrap().len(), cat.laws().len());

    // kit-eqsat pack "pointwise_factor" — the opt-in unguarded
    // POINTWISE_FACTOR_RULES pair.
    let pf = LawSet::from_json(pack("pointwise_factor")).unwrap();
    assert_eq!(pf.laws(), builtin::pointwise_factor().as_slice());
    assert_eq!(pf.compile().unwrap().len(), pf.laws().len());

    // kit-eqsat pack "attention" — the `ATTENTION_RULES` +
    // `SDPA_EXPAND_RULES` opt-in pack (Python `ATTENTION` ruleset
    // order): the 36 portable `laws/attention.py` members whose
    // procedural `check` hooks translate to `uniform`/`scalar`/
    // `dim-eq-const`/`bcast-into`/`mm-shape-ok`/`rank`/`dim-compat`
    // conds (`rope_cat_compose` declined — its half-split guard
    // needs attr-arithmetic vs shape-extent comparisons the
    // vocabulary cannot form), ahead of the full SDPA_EXPAND_RULES
    // (stage 4 ported the `dim` dspec channel the scale-free
    // key-sets mint `1/√(q[-1])` through).
    let att = LawSet::from_json(pack("attention")).unwrap();
    assert_eq!(att.laws(), builtin::attention().as_slice());
    assert_eq!(att.laws().len(), 44);
    assert_eq!(att.compile().unwrap().len(), att.laws().len());

    // kit-eqsat pack "scan" — the opt-in SCAN_LAWS + SCAN_DIAG_LAWS
    // monoids (CARRIER_SEARCH, never DEFAULT): the `affd_lift*` state
    // conds and the unit lift's `bcast` dspec compile through the
    // ported guard vocabulary.
    let scan = LawSet::from_json(pack("scan")).unwrap();
    assert_eq!(scan.laws(), builtin::scan().as_slice());
    assert_eq!(scan.laws().len(), 22);
    assert_eq!(scan.compile().unwrap().len(), scan.laws().len());

    // laws/reshape_flatten_demo.json — the UNSOUND attrs-bind demo
    // (engine-local; kit-eqsat excludes it by design)
    // (matcher coverage only; the `_demo` name is the warning).
    let flat = LawSet::from_json(include_str!("../laws/reshape_flatten_demo.json")).unwrap();
    assert_eq!(flat.laws(), &[builtin::reshape_flatten_demo()]);

    // And the records re-serialize byte-stably through the codec.
    for set in [simpl, pa, cat, pf, att, scan, flat] {
        let reloaded = LawSet::from_data(&set.to_data()).unwrap();
        assert_eq!(reloaded, set);
    }
}
