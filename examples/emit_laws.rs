//! Emit the `laws/*.json` reference corpus — the builtin packs
//! serialized to the `LAW_FORMAT = 2` record form (design doc §1).
//!
//! Usage:
//!
//! ```sh
//! cargo run --example emit_laws -- simplification > laws/simplification.json
//! cargo run --example emit_laws -- reshape_flatten_demo > laws/reshape_flatten_demo.json
//! cargo run --example emit_laws -- prior_art > laws/prior_art.json
//! ```
//!
//! Regenerate after changing [`hnl_engine_eqsat::builtin`] or the
//! record codec; `tests/contracts.rs` loads these files, so a stale
//! corpus fails the suite.

use hnl_engine_eqsat::{LawSet, builtin};

fn main() {
    let pack = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    let laws = match pack.as_str() {
        "simplification" => builtin::simplification(),
        "prior_art" => builtin::prior_art(),
        "reshape_flatten_demo" => vec![builtin::reshape_flatten_demo()],
        "all" => {
            let mut v = builtin::simplification();
            v.push(builtin::reshape_flatten_demo());
            v
        }
        other => {
            eprintln!(
                "unknown pack {other:?} — expected simplification|prior_art|reshape_flatten_demo|all"
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
