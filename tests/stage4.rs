//! Stage-4 tests: the flat-map/repeat pairing guards, the
//! shape-reading dspec channels (`shape`/`dim`/`leaf-dim`/`bcast`),
//! and the law families they unlock — `mul_unsq_nat`,
//! `rms_norm_fold*`, `qkv_fuse_asym`, `gqa_absorb_repeat`, the full
//! `sdpa_expand*` family, `batch_norm_expand*`, and the 60-law
//! `conv_channel_*` family.

use hnl_engine::Term as KTerm;
use hnl_engine_eqsat::{self as eqsat, Law, LawSet, builtin};
use serde_json::json;

fn op(name: &str, args: Vec<KTerm>) -> KTerm {
    KTerm::op_native(name, args)
}

fn op_attrs(name: &str, args: Vec<KTerm>, attrs: serde_json::Value) -> KTerm {
    KTerm::op_native_attrs(name, args, attrs)
}

fn konst(v: serde_json::Value) -> KTerm {
    KTerm::constant(v)
}

/// A tensor leaf declaring its shape through the `__tensor__` wire
/// convention (same helper shape as `tests/shape.rs`).
fn tensor(shape: serde_json::Value) -> KTerm {
    KTerm::constant(json!({"__tensor__": {"shape": shape}}))
}

fn shaped(dims: &[i64]) -> KTerm {
    tensor(json!(dims))
}

/// Saturate `input` under `laws`; return the rule names the merge
/// log recorded (a name appears iff its rewrite instantiated an RHS
/// member — the same signal `tests/eqsat.rs`'s `assert_fires` uses).
fn fired(input: &KTerm, laws: &[Law]) -> Vec<String> {
    let sat = eqsat::saturate(
        std::slice::from_ref(input),
        laws,
        &eqsat::SaturateOptions::spec(),
    )
    .expect("saturate");
    sat.graph
        .merge_log()
        .iter()
        .filter_map(|e| e.rule.as_deref().map(str::to_string))
        .collect()
}

fn assert_fired(input: &KTerm, law: &Law) {
    let names = fired(input, std::slice::from_ref(law));
    assert!(
        names.iter().any(|n| n == &law.name),
        "{} did not fire: {names:?}",
        law.name
    );
}

fn assert_no_fire(input: &KTerm, law: &Law) {
    let names = fired(input, std::slice::from_ref(law));
    assert!(
        !names.iter().any(|n| n == &law.name),
        "{} fired unexpectedly: {names:?}",
        law.name
    );
}

fn law_named(laws: Vec<Law>, name: &str) -> Law {
    laws.into_iter()
        .find(|l| l.name == name)
        .unwrap_or_else(|| panic!("no law named {name}"))
}

// ---------------------------------------------------------------------------
//  mul_unsq_nat — the flat-map-unsq pairing guard
// ---------------------------------------------------------------------------

#[test]
fn mul_unsq_nat_fires_on_equal_rank_pair() {
    // u,v rank-2 broadcast-compatible shapes — the flat-map-unsq
    // region qualifies; the shared `dim` attr binds `UD`.
    let u = shaped(&[2, 3]);
    let v = shaped(&[2, 3]);
    let input = op(
        "mul",
        vec![
            op_attrs("unsqueeze", vec![u], json!({"dim": 1})),
            op_attrs("unsqueeze", vec![v], json!({"dim": 1})),
        ],
    );
    assert_fired(&input, &builtin::mul_unsq_nat());
}

#[test]
fn mul_unsq_nat_declines_on_incompatible_broadcast() {
    // [2,3] and [4,5] cannot broadcast — the grids never coincide,
    // so the pairing guard declines.
    let u = shaped(&[2, 3]);
    let v = shaped(&[4, 5]);
    let input = op(
        "mul",
        vec![
            op_attrs("unsqueeze", vec![u], json!({"dim": 1})),
            op_attrs("unsqueeze", vec![v], json!({"dim": 1})),
        ],
    );
    assert_no_fire(&input, &builtin::mul_unsq_nat());
}

#[test]
fn mul_unsq_nat_declines_on_unknown_shapes() {
    // Shape-less Opaque leaves → the guard cannot prove the pairing
    // → decline, never a guess.
    let leaf = || KTerm::Opaque {
        data: json!({"tag": "mystery"}),
    };
    let input = op(
        "mul",
        vec![
            op_attrs("unsqueeze", vec![leaf()], json!({"dim": 0})),
            op_attrs("unsqueeze", vec![leaf()], json!({"dim": 0})),
        ],
    );
    assert_no_fire(&input, &builtin::mul_unsq_nat());
}

// ---------------------------------------------------------------------------
//  rms_norm_fold* — the shape/dim dspec channels
// ---------------------------------------------------------------------------

/// The manual-RMSNorm spelling: `mul(mul(u, rsqrt(add(mean(pow(u,
/// P), dim=MD, keepdim=MK), EPS))), w)`.
fn manual_rms(u: &KTerm, w: Option<&KTerm>) -> KTerm {
    let inner = op(
        "mul",
        vec![
            u.clone(),
            op(
                "rsqrt",
                vec![op(
                    "add",
                    vec![
                        op_attrs(
                            "mean",
                            vec![op("pow", vec![u.clone(), konst(json!(2))])],
                            json!({"dim": 1, "keepdim": true}),
                        ),
                        konst(json!(1e-5)),
                    ],
                )],
            ),
        ],
    );
    match w {
        Some(w) => op("mul", vec![inner, w.clone()]),
        None => inner,
    }
}

#[test]
fn rms_norm_fold_fires_with_derived_dim_and_eps() {
    // u=[2,3], reduce dim=1 keepdim → the trailing block is (3,);
    // w=[3] matches `shape-eq w (tail-block u MD)`; `ND` derives
    // `shape(tail-block)` and `EP` derives `float(const(EPS))`.
    let u = shaped(&[2, 3]);
    let w = shaped(&[3]);
    let input = manual_rms(&u, Some(&w));
    let law = builtin::rms_norm_fold();
    assert_fired(&input, &law);
}

#[test]
fn rms_norm_fold_nogain_fires_without_weight() {
    let u = shaped(&[2, 3]);
    let input = manual_rms(&u, None);
    let law = builtin::rms_norm_fold_nogain();
    assert_fired(&input, &law);
}

#[test]
fn rms_norm_fold_declines_on_non_unit_exponent() {
    // P=3 violates `const-cmp P == 2` — decline.
    let u = shaped(&[2, 3]);
    let w = shaped(&[3]);
    let input = op(
        "mul",
        vec![
            op(
                "mul",
                vec![
                    u.clone(),
                    op(
                        "rsqrt",
                        vec![op(
                            "add",
                            vec![
                                op_attrs(
                                    "mean",
                                    vec![op("pow", vec![u, konst(json!(3))])],
                                    json!({"dim": 1, "keepdim": true}),
                                ),
                                konst(json!(1e-5)),
                            ],
                        )],
                    ),
                ],
            ),
            w,
        ],
    );
    assert_no_fire(&input, &builtin::rms_norm_fold());
}

// ---------------------------------------------------------------------------
//  qkv_fuse_asym — leaf-dim dspec
// ---------------------------------------------------------------------------

/// `transpose(reshape(linear(x, W), shape=S), dim0=1, dim1=2)` — one
/// head view with its own shape attr.
fn head_of(w: &KTerm, x: &KTerm, shape: serde_json::Value) -> KTerm {
    op_attrs(
        "transpose",
        vec![op_attrs(
            "reshape",
            vec![op("linear", vec![x.clone(), w.clone()])],
            json!({"shape": shape}),
        )],
        json!({"dim0": 1, "dim1": 2}),
    )
}

#[test]
fn qkv_fuse_asym_fires_with_declared_weight_shapes() {
    // Q/K/V are tensor leaves with declared shapes — `leaf-dim Q 0`
    // etc. mint the uneven `SZ` split sizes.
    let x = shaped(&[2, 8]);
    let q = shaped(&[4, 8]);
    let k = shaped(&[2, 8]);
    let v = shaped(&[6, 8]);
    let input = op_attrs(
        "sdpa",
        vec![
            head_of(&q, &x, json!([2, 4, 2])),
            head_of(&k, &x, json!([2, 2, 2])),
            head_of(&v, &x, json!([2, 6, 2])),
        ],
        json!({"scale": 0.5, "enable_gqa": false}),
    );
    let laws = builtin::categorical();
    assert_fired(&input, &law_named(laws, "qkv_fuse_asym"));
}

#[test]
fn qkv_fuse_asym_declines_without_declared_shapes() {
    // Shape-less Const weights → `leaf-dim` declines → the derive
    // vetoes (a `None` dspec value refuses the rewrite, never a guess).
    let x = konst(json!("x"));
    let q = konst(json!("q"));
    let k = konst(json!("k"));
    let v = konst(json!("v"));
    let input = op_attrs(
        "sdpa",
        vec![
            head_of(&q, &x, json!([2, 4, 2])),
            head_of(&k, &x, json!([2, 2, 2])),
            head_of(&v, &x, json!([2, 6, 2])),
        ],
        json!({"scale": 0.5, "enable_gqa": false}),
    );
    let laws = builtin::categorical();
    assert_no_fire(&input, &law_named(laws, "qkv_fuse_asym"));
}

// ---------------------------------------------------------------------------
//  gqa_absorb_repeat — repeat-chain/repeat-heads
// ---------------------------------------------------------------------------

/// `transpose(reshape(expand(unsqueeze(t, dim=UD), shape=ES),
/// shape=RS), dim0=1, dim1=2)` — the repeat_kv copy chain.
fn repeat_head(t: &KTerm, ud: i64, es: serde_json::Value, rs: serde_json::Value) -> KTerm {
    op_attrs(
        "transpose",
        vec![op_attrs(
            "reshape",
            vec![op_attrs(
                "expand",
                vec![op_attrs("unsqueeze", vec![t.clone()], json!({"dim": ud}))],
                json!({"shape": es}),
            )],
            json!({"shape": rs}),
        )],
        json!({"dim0": 1, "dim1": 2}),
    )
}

#[test]
fn gqa_absorb_repeat_fires_on_repeat_kv_chain() {
    // q=[2,4,6,8] (B,T,Hq,D), k/v=[2,4,3,8] (B,T,Hkv,D), r=2:
    // unsqueeze dim=3 → [2,4,3,1,8]; expand → [2,4,3,2,8]; reshape
    // → [2,4,6,8] — the merge repeats each kv head twice, and
    // `repeat-heads` reads 6 == 3*2.  ESk == ESv pins v's chain to
    // the same expand shape.
    let q = shaped(&[2, 4, 6, 8]);
    let k = shaped(&[2, 4, 3, 8]);
    let v = shaped(&[2, 4, 3, 8]);
    let input = op_attrs(
        "sdpa",
        vec![
            op_attrs("transpose", vec![q], json!({"dim0": 1, "dim1": 2})),
            repeat_head(&k, 3, json!([2, 4, 3, 2, 8]), json!([2, 4, 6, 8])),
            repeat_head(&v, 3, json!([2, 4, 3, 2, 8]), json!([2, 4, 6, 8])),
        ],
        json!({"dropout_p": 0.0, "is_causal": false}),
    );
    let laws = builtin::categorical();
    assert_fired(&input, &law_named(laws, "gqa_absorb_repeat"));
}

#[test]
fn gqa_absorb_repeat_declines_on_head_mismatch() {
    // Same chain but q has 5 query heads — 5 != 3*2 → repeat-heads
    // declines.
    let q = shaped(&[2, 4, 5, 8]);
    let k = shaped(&[2, 4, 3, 8]);
    let v = shaped(&[2, 4, 3, 8]);
    let input = op_attrs(
        "sdpa",
        vec![
            op_attrs("transpose", vec![q], json!({"dim0": 1, "dim1": 2})),
            repeat_head(&k, 3, json!([2, 4, 3, 2, 8]), json!([2, 4, 6, 8])),
            repeat_head(&v, 3, json!([2, 4, 3, 2, 8]), json!([2, 4, 6, 8])),
        ],
        json!({"dropout_p": 0.0, "is_causal": false}),
    );
    let laws = builtin::categorical();
    assert_no_fire(&input, &law_named(laws, "gqa_absorb_repeat"));
}

// ---------------------------------------------------------------------------
//  sdpa_expand — the scale-free implicit-scale mint
// ---------------------------------------------------------------------------

#[test]
fn sdpa_expand_mints_implicit_scale_from_q_dim() {
    // attrs-free sdpa → `SC0 = rsqrt(dim(Q, -1))` derives through
    // the `dim` shape channel; Q=[2,4,8,16] gives 1/√16 = 0.25.
    let q = shaped(&[2, 4, 8, 16]);
    let k = shaped(&[2, 4, 8, 16]);
    let v = shaped(&[2, 4, 16, 16]);
    let input = op("sdpa", vec![q, k, v]);
    let laws = builtin::attention();
    assert_fired(&input, &law_named(laws, "sdpa_expand"));
}

#[test]
fn sdpa_expand_keysets_are_exact() {
    // `sdpa_expand_d` requires exactly `{dropout_p: 0.0}` — a bare
    // sdpa does not match it, and a dropout-carrying sdpa does not
    // match the bare law.
    let q = shaped(&[2, 4, 8, 16]);
    let k = shaped(&[2, 4, 8, 16]);
    let v = shaped(&[2, 4, 16, 16]);
    let bare = op("sdpa", vec![q.clone(), k.clone(), v.clone()]);
    let laws = builtin::attention();
    assert_no_fire(&bare, &law_named(laws.clone(), "sdpa_expand_d"));

    let with_d = op_attrs("sdpa", vec![q, k, v], json!({"dropout_p": 0.0}));
    assert_fired(&with_d, &law_named(laws, "sdpa_expand_d"));
}

// ---------------------------------------------------------------------------
//  batch_norm_expand* — $leaf minting + concat/take/len dspec
// ---------------------------------------------------------------------------

#[test]
fn batch_norm_expand_fires_in_eval_mode() {
    // x=[2,3,4] (channel dim 1 = 3), stats rm/rv/w/b all [3];
    // `CH` derives (1,3,1) through concat/shape/take/len/sub.
    let x = shaped(&[2, 3, 4]);
    let w = shaped(&[3]);
    let b = shaped(&[3]);
    let rm = shaped(&[3]);
    let rv = shaped(&[3]);
    let input = op_attrs(
        "batch_norm",
        vec![x, w, b, rm, rv],
        json!({"training": false, "momentum": 0.1, "eps": 1e-5, "cudnn_enabled": true}),
    );
    let laws = builtin::batch_norm();
    assert_fired(&input, &law_named(laws, "batch_norm_expand"));
}

#[test]
fn batch_norm_expand_declines_in_training_mode() {
    let x = shaped(&[2, 3, 4]);
    let w = shaped(&[3]);
    let b = shaped(&[3]);
    let rm = shaped(&[3]);
    let rv = shaped(&[3]);
    let input = op_attrs(
        "batch_norm",
        vec![x, w, b, rm, rv],
        json!({"training": true, "momentum": 0.1, "eps": 1e-5, "cudnn_enabled": true}),
    );
    let laws = builtin::batch_norm();
    assert_no_fire(&input, &law_named(laws, "batch_norm_expand"));
}

// ---------------------------------------------------------------------------
//  conv_channel_* — the 60-law generated family
// ---------------------------------------------------------------------------

/// `mul(conv2d(x, W, b), sv)` with the channel-grid shapes the
/// guard wants: x=[1,3,8,8], W=[4,3,3,3], b=[4], sv=[1,4,1,1].
fn conv_scale_input() -> KTerm {
    let x = shaped(&[1, 3, 8, 8]);
    let w = shaped(&[4, 3, 3, 3]);
    let b = shaped(&[4]);
    let sv = shaped(&[1, 4, 1, 1]);
    op("mul", vec![op("conv2d", vec![x, w, b]), sv])
}

#[test]
fn conv_channel_scale_fires() {
    let laws = builtin::conv_folds();
    assert_eq!(laws.len(), 60);
    assert_fired(&conv_scale_input(), &law_named(laws, "conv_channel_scale"));
}

#[test]
fn conv_channel_scale_l_fires_on_flipped_operands() {
    // `mul(sv, conv)` — the batch_norm expansion's operand order.
    let x = shaped(&[1, 3, 8, 8]);
    let w = shaped(&[4, 3, 3, 3]);
    let b = shaped(&[4]);
    let sv = shaped(&[1, 4, 1, 1]);
    let input = op("mul", vec![sv, op("conv2d", vec![x, w, b])]);
    let laws = builtin::conv_folds();
    assert_fired(&input, &law_named(laws, "conv_channel_scale_l"));
}

#[test]
fn conv_channel_scale_spdg_fires_with_grouped_attrs() {
    // groups=1 → the attr-cmp channel equation x1 == W1*GR holds
    // (3 == 3*1); the conv2d carries the full stride/padding/
    // dilation/groups keyset.
    let x = shaped(&[1, 3, 8, 8]);
    let w = shaped(&[4, 3, 3, 3]);
    let b = shaped(&[4]);
    let sv = shaped(&[1, 4, 1, 1]);
    let conv = op_attrs(
        "conv2d",
        vec![x, w, b],
        json!({"stride": 1, "padding": 0, "dilation": 1, "groups": 1}),
    );
    let input = op("mul", vec![conv, sv]);
    let laws = builtin::conv_folds();
    assert_fired(&input, &law_named(laws, "conv_channel_scale_spdg"));
}

#[test]
fn conv_channel_shift_fires() {
    let x = shaped(&[1, 3, 8, 8]);
    let w = shaped(&[4, 3, 3, 3]);
    let b = shaped(&[4]);
    let cv = shaped(&[1, 4, 1, 1]);
    let input = op("add", vec![op("conv2d", vec![x, w, b]), cv]);
    let laws = builtin::conv_folds();
    assert_fired(&input, &law_named(laws, "conv_channel_shift"));
}

#[test]
fn conv_channel_scale_declines_on_channel_mismatch() {
    // sv=[1,5,1,1] vs W out-channels 4 → dim-eq sv1 W0 fails.
    let x = shaped(&[1, 3, 8, 8]);
    let w = shaped(&[4, 3, 3, 3]);
    let b = shaped(&[4]);
    let sv = shaped(&[1, 5, 1, 1]);
    let input = op("mul", vec![op("conv2d", vec![x, w, b]), sv]);
    let laws = builtin::conv_folds();
    assert_no_fire(&input, &law_named(laws, "conv_channel_scale"));
}

#[test]
fn conv_channel_scale_rev_fires_on_absorbed_form() {
    // The unfold direction: conv(x, W·reshape(sv,WV),
    // b·reshape(sv,SV)) → mul(conv(x,W,b), sv).  The RHS reshape
    // attrs are dspec-derived (`WV`/`SV`), so build the absorbed
    // conv with literal shapes matching `take(shape W,1)` ++
    // `(1,1,1)` = [4,1,1,1] and `SV` = [4].
    let x = shaped(&[1, 3, 8, 8]);
    let w = shaped(&[4, 3, 3, 3]);
    let b = shaped(&[4]);
    let sv = shaped(&[1, 4, 1, 1]);
    let conv = op(
        "conv2d",
        vec![
            x,
            op(
                "mul",
                vec![
                    w,
                    op_attrs("reshape", vec![sv.clone()], json!({"shape": [4, 1, 1, 1]})),
                ],
            ),
            op(
                "mul",
                vec![b, op_attrs("reshape", vec![sv], json!({"shape": [4]}))],
            ),
        ],
    );
    let laws = builtin::conv_folds();
    assert_fired(&conv, &law_named(laws, "conv_channel_scale_rev"));
}

// ---------------------------------------------------------------------------
//  Corpus files — batch_norm / conv_folds / prim_folds
// ---------------------------------------------------------------------------

#[test]
fn stage4_corpus_files_load_and_match_builtin_packs() {
    let bn = LawSet::from_json(include_str!("../laws/batch_norm.json")).unwrap();
    assert_eq!(bn.laws(), builtin::batch_norm().as_slice());
    assert_eq!(bn.compile().unwrap().len(), bn.laws().len());

    let conv = LawSet::from_json(include_str!("../laws/conv_folds.json")).unwrap();
    assert_eq!(conv.laws(), builtin::conv_folds().as_slice());
    assert_eq!(conv.laws().len(), 60);
    assert_eq!(conv.compile().unwrap().len(), conv.laws().len());

    // laws/prim_folds.json — the 60 minted `prim:synop*` fold records
    // as LAW_FORMAT=2 data.  They compile (patterns + record fields
    // parse) but are NOT in any builtin pack: the synthetic
    // `synop_*` operator denotations are domain-side and require a
    // consumer that registered them — matching Python, where the
    // folds only fire on terms containing those ops.
    let prim = LawSet::from_json(include_str!("../laws/prim_folds.json")).unwrap();
    assert_eq!(prim.laws().len(), 60);
    assert!(prim.laws().iter().all(|l| l.name.starts_with("prim:")));
    assert!(prim.laws().iter().all(|l| l.serializable));
    assert!(prim.laws().iter().all(|l| l.missing_hooks.is_empty()));
    assert_eq!(prim.compile().unwrap().len(), 60);
    let builtin_all = builtin::all();
    let all_names: std::collections::BTreeSet<&str> =
        builtin_all.iter().map(|l| l.name.as_str()).collect();
    assert!(
        prim.laws()
            .iter()
            .all(|l| !all_names.contains(l.name.as_str()))
    );

    // And the records re-serialize byte-stably.
    for set in [bn, conv, prim] {
        let reloaded = LawSet::from_data(&set.to_data()).unwrap();
        assert_eq!(reloaded, set);
    }
}
