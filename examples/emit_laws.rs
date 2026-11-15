//! Emit the `LAW_FORMAT = 2` reference corpus — the builtin packs
//! serialized to record form (design doc §1).
//!
//! **Canonical home: `handl-kit-eqsat`** (`data/laws/*.json`,
//! ADR-0177 §1 / ADR-0178). This example is the *regeneration path*:
//! after changing [`hnl_engine_eqsat::builtin`] or the record codec,
//! emit each pack and sync the output into kit-eqsat — the bytes are
//! never vendored back into this repo (`laws/` keeps only the
//! `reshape_flatten_demo` matcher demo).
//!
//! Usage:
//!
//! ```sh
//! cargo run --example emit_laws -- simplification > /tmp/simplification.json
//! cargo run --example emit_laws -- reshape_flatten_demo > laws/reshape_flatten_demo.json
//! cargo run --example emit_laws -- prior_art > /tmp/prior_art.json
//! cargo run --example emit_laws -- categorical > /tmp/categorical.json
//! cargo run --example emit_laws -- pointwise_factor > /tmp/pointwise_factor.json
//! cargo run --example emit_laws -- attention > /tmp/attention.json
//! cargo run --example emit_laws -- batch_norm > /tmp/batch_norm.json
//! cargo run --example emit_laws -- conv_folds > /tmp/conv_folds.json
//! ```
//!
//! Then copy the `/tmp/*.json` outputs over
//! `../kit-eqsat/crates/handl-kit-eqsat/data/laws/<name>.json` (except
//! `reshape_flatten_demo`, which stays engine-local).
//!
//! `prim_folds` has **no emit arm by design**: it is the 60 minted
//! `prim:synop*` fold records — data with no `builtin` constructor —
//! maintained directly in kit-eqsat. The contract tests load every
//! pack through `handl_kit_eqsat::law_pack(name).json`, so a stale
//! kit corpus fails the suite.

use hnl_engine_eqsat::{LawSet, builtin};

fn main() {
    let pack = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    let laws = match pack.as_str() {
        "simplification" => builtin::simplification(),
        "categorical" => builtin::categorical(),
        "pointwise_factor" => builtin::pointwise_factor(),
        "attention" => builtin::attention(),
        "batch_norm" => builtin::batch_norm(),
        "conv_folds" => builtin::conv_folds(),
        "prior_art" => builtin::prior_art(),
        "reshape_flatten_demo" => vec![builtin::reshape_flatten_demo()],
        "all" => {
            let mut v = builtin::simplification();
            v.push(builtin::reshape_flatten_demo());
            v
        }
        other => {
            eprintln!(
                "unknown pack {other:?} — expected simplification|categorical|pointwise_factor|attention|batch_norm|conv_folds|prior_art|reshape_flatten_demo|all"
            );
            std::process::exit(2);
        }
    };
    let set = LawSet::from_laws(laws);
    println!(
        "{}",
        serde_json::to_string_pretty(&set.to_data()).expect("records are JSON-safe")
    );
}
