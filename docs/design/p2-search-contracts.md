# P2 design: engine-eqsat search contracts

Date: 2026-11-2x · Status: proposed (decisions pending user ratification
via ADR-0177) · Serves ADR-0174 phase P2 / ADR-0175 dissolution row
`eqsat → engine-eqsat-* + kit-eqsat`.

This document fixes the five contracts that mass law porting depends
on **before** any law moves: the serialized law format, the guard
port, cost models, certificates, and the engine trait surface. It
answers spike §6 questions 1–5 (`host/project/reports/
engine-eqsat-spike.md`). Spike §6.6 (the differential harness) is
acknowledged but scoped as migration work, not a contract.

Current state: `hnl-engine-eqsat` ships `pat`/`law`/`builtin`/`engine`
— unconditional laws only, Rust-valued `Law`s, `cost_fn` closures over
kernel `Term`s, and `ProofEdge`/`Application` provenance inspectable
on `Saturated::graph`. Nothing below breaks that API; every decision
extends it.

---

## 1. Law serialization format

**Decision: one canonical format — a JSON law record mirroring
`mlopt_core.laws.serialize.LAW_FORMAT = 2`, with the pattern pair
spelled in the `Pat` authoring view (not the compiled bridge
encoding). `.hnl` is rejected.**

### Rationale

- `serialize.py` already proved the record shape: `version`, `kind`,
  `name`, `law`, `lhs`, `rhs`, `cond`, `dspec`, `tags`,
  `derivation`, `error_bound`/`bound_norm`, plus the honesty fields
  `serializable`/`missing_hooks` and optional `cert`. Guards and
  derive specs are *already* JSON-safe tuple trees (`cond_to_data`/
  `derive_to_data`) — the format exists; it just needs a Rust reader
  and a pattern encoding that matches what `pat.rs` compiles.
- `.hnl` is the wrong tool: a law is not an executable program; it is
  a 2-cell (term pair + side condition + provenance). Putting patterns
  in `.hnl` surface syntax buys nothing — the codec already exists
  (`term_to_data`), and ADR-0174 invariant 4 ("one codec") argues
  against a second term↔surface spelling. `.hnl` may *embed* law
  records later (a kit manifest referencing `laws/*.json`); the record
  itself stays JSON.
- The compiled bridge encoding (`handl_egraph::Term` with `dim` as
  child slot 0, exact-keyset attrs, minted `Const` leaf keys) must
  **never** be serialized — `handl_egraph::Term` stays
  engine-internal (ADR-0174 non-goal, spike §6.7). The record stores
  the `Pat`-level pattern; `Law::compile` lowers it exactly as it
  does for Rust-authored laws today, so serialized and in-code laws
  share one compilation path — no second semantics to drift.

### The record

```json
{
  "version": 2,
  "kind": "law",
  "name": "id_add",
  "law": "Additive identity: a + 0 = a.",
  "lhs": {
    "op": "add",
    "dim": null,
    "args": [
      {"mvar": "a"},
      {"const": 0}
    ],
    "source": "native",
    "lit": null,
    "attrs": null
  },
  "rhs": {"mvar": "a"},
  "cond": null,
  "dspec": null,
  "tags": ["simplification"],
  "derivation": [],
  "cert": null,
  "error_bound": null,
  "bound_norm": "spectral",
  "serializable": true,
  "missing_hooks": []
}
```

### Pattern encoding (`pat` as data)

Deliberately *not* `term_to_data`'s `{"op","args","attrs"}` — that
scheme describes typed-view `Op`s where `dim` is implicit and attrs
are flat. The bridge encoding's rules (dim slot, exact-keyset
matching, JSON-valued const leaves, source tags) belong to the
`Pat` layer, so the record spells them explicitly:

| `Pat` node | JSON |
|---|---|
| `Pat::Var("a")` | `{"mvar": "a"}` |
| `Pat::Const(v)` | `{"const": <json value>}` — minted via `from_kernel` at load, never a raw string |
| `Pat::Op` | `{"op": name, "dim": <pat>\|null, "args": [...], "source": "native"\|"foreign"\|"judgment"\|"dynamic", "foreign": name?, "lit": <json>\|null, "attrs": {"lit": v} \| {"bind": name} \| null}` |

`"dim": null` carries the existing `OpPat::dim: None` semantics:
don't-care on the LHS (fresh metavariable; the LHS *root* gets the
law's root dim var), "inherit the LHS root's dim" on the RHS. Pinning
a dim is `"dim": {"mvar": "d"}` or `{"const": ...}`. This is the one
place the record is *more* explicit than the Python `Op` scheme —
honestly, because dim semantics over the bridge are the footgun spike
§5 flagged.

`source`/`foreign` spell `PatSource`; `attrs` distinguishes
`AttrsSpec::Lit` (`{"lit": v}`) from `AttrsSpec::Bind`
(`{"bind": name}` — re-emitted verbatim, the `reshape_flatten_demo`
attrs-bind demo).
Attr metavariable names in `cond`/`dspec` keep the `"$attr:NAME"`
spelling; term metavars are bare names, matching `BoundEnv`.

### Where it lives

`kit-eqsat` (the KIT-layer repo ADR-0175 names — it does not exist
yet). Until then, records live in this repo under `laws/*.json` as
the reference corpus, loaded by a `law::from_data` /
`LawSet::from_json` API added to `hnl-engine-eqsat`. The minted store
(`minted.db`) keeps its own serialization; `kit-eqsat` is where its
rows are *projected* into this record format at admission time —
the store stays the ledger, the record is the interchange.

### Honesty fields

`serializable`/`missing_hooks` port verbatim. A record whose
procedural `check`/`derive` could not be folded into `cond`/`dspec`
loads with `missing_hooks: ["check"]` and the loader's contract is:
**a non-serializable record may be inspected but must not be fired
silently** — `LawSet::from_json` returns it flagged; `saturate`/
`optimize` callers choose to refuse (`Error::UnserializableLaw`) or
run it with the guard *absent*, which is a named, visible weakening —
never a silent one. This is the `missing_hooks` posture ported, not
invented.

---

## 2. Guard porting (`cond.py` → Rust)

**Decision: port the `cond`/`dspec` DSL to Rust as a guards-as-data
interpreter over `BoundEnv`. Procedural `check=`/`derive=` Python
hooks stay Python-side and are flagged `missing_hooks` — they are
the minority and the honesty fields already name them.**

### What the census says

Rough `cond=`/`dspec=`/`check=`/`derive=` site counts across
`mlopt_core/laws/` (each `Rewrite(`-family site, not exact law count):

| file | cond | dspec | check | derive |
|---|---|---|---|---|
| tensor.py | 48 | 17 | 0 | 0 |
| priorart.py | 52 | 13 | 16 | 5 |
| scan.py | 6 | 4 | 0 | 0 |
| minted.py | 3 | 3 | 1 | 1 |
| attention.py | 0 | 0 | 17 | 0 |
| layout.py | 0 | 0 | 11 | 0 |
| base.py | 1 | 1 | 2 | 3 |

Read honestly: the bulk of the shipped + machine-shipped banks
express their side conditions in the **declarative** `cond`/`dspec`
DSL already — that is precisely why `cond.py` exists (its docstring:
"the one thing keeping laws from being serializable data"). The
procedural remainder concentrates in `attention.py`/`layout.py`
(`check=` lambdas) and a scattering of priorart/minted hooks — call
it **~20–25% of guarded sites**, and those laws already serialize
with `serializable: false` today.

### Why not the alternatives

- **Compiled packs** (per-kit Rust rule packs, `builtin.rs`-style):
  reintroduces code where data belongs. Every ported guarded law
  becomes a hand-maintained Rust function — the exact
  representation-coupling the dissolution is removing, and it cannot
  carry minted laws at all (minted records are *generated*; a
  compiled pack cannot be generated into Rust source honestly).
- **Python callback seam**: keeps a Python host alive inside the
  search loop — spike §3's conclusion stands (Rust-over-Term is the
  only viable production path). A `BoundEnv`-crossing FFI per firing
  would also be the slowest possible predicate call. Rejected for
  the guarded majority; kept only as the escape hatch for the
  procedural minority until those laws are either re-expressed in
  `cond` or their consumers stay Python-side.

### The port, staged

`cond.py` is a finite predicate/expression tree — total, never calls
user code, strictness contract = decline (`false`/`None`). The Rust
port is an interpreter of the same trees over
`handl_egraph::rewrite::BoundEnv` (`terms`/`attrs` maps), wired into
`Rewrite::with_check`/`with_derive`. Two dependency layers, landed in
order:

1. **Attr/scalar predicates + the full `dspec` expr algebra** —
   `and`/`or`/`not`, `attr-*`, `const-*`, `term-eq`, `op-in`, `leaf`,
   `const`, `const-num`, `const-cmp`, plus every `expr` node
   (`attr`, `shape`-free arithmetic, `tuple`/`concat`/`gather`/
   `unique`/`posmap`, `guard`/`if`). Covers the attr-only guards and
   all non-shape derives. Small, total, honest.
2. **Shape-reading predicates + shape specs** — `shaped`, `scalar`,
   `uniform`, `rank*`, `shape-*`, `dim-*`, `bcast-*`, `mm-shape-ok`,
   `axes-*`, `axis`, `dim-mod`, `ones-*`, and the shape-spec forms
   (`mm-out`, `bcast`, `unsq-out`, `reshape-out`, `getitem-out`,
   `select-out`, `slice-out`, `chunk-out`, `transpose-out`,
   `tail-block`, `isel-out`, `vshape`). This requires a **kernel-side
   shape inference** — `_shape_of`/`_broadcast`/`_matmul_shape` over
   kernel `Term::Op`s reading the `dim` slot and `attrs`. That module
   (`engine-eqsat::shape` or kit data once stable) is the real work
   item; it must reproduce the Python typing conventions
   (`mlopt_core.typing`) over the bridge encoding, including `None`
   dims as wildcards exactly where `*-compat` predicates allow.

`dspec`'s tensor-value channels (`tval`, `tapp`, `mint`, `key` — the
`TensorAlgebra` push) stay **out of scope for P2**: they need a
registered tensor backend, which is kit/backend territory. Records
using them carry `missing_hooks`-equivalent flags until a backend
plugs in — same honesty posture, named not silent.

`Derived` maps cleanly: `{"$attr:NAME": v}` → `Derived::Attr`,
`{"$leaf:NAME": ...}` → `Derived::Term`/`Class` per the expr result
type. The `$leaf:` minting rule (deterministic content-keyed names —
certificate replay re-runs derives) ports as a constraint on the
expr evaluator's leaf construction.

---

## 3. Cost models

**Decision: a serializable `CostSpec` — named cost *kinds* plus a
per-op table — interpreted engine-side. `count`/`size` ship in P2;
`op_table` ports `_OP_FLOPS`-style data; `flops`/roofline/fusion/
executor models arrive later as additional named specs once shape
inference (§2 stage 2) exists. Python keeps the full `cost/` algebra
until then.**

### Why not a closure or a DSL-y rewrite

`cost_fn: FnMut(&KernelTerm) -> f64` is a closure today — kit data
cannot ship one. But the *shape* of every shipped cost model is the
same: local op pricing + a sum over the DAG. `count_cost`,
`flops_cost`, `param_bytes_cost`, `launch_aware_cost` are all
"per-op table + parametric formula over inferred shapes." That is
data.

### The `CostSpec` record

```json
{
  "kind": "op_table",
  "name": "flops-v1",
  "default": 1.0,
  "leaf": 0.0,
  "ops": {
    "add": 1.0, "mul": 1.0, "matmul": {"flops": "2*m*n*k"},
    "tensor.reshape": {"view": true, "cost": 0.0}
  },
  "dim_leaf": 0.0
}
```

Kinds, in landing order:

- **`count`** — node count (already expressible; `size_cost`'s
  honest bridge-adjusted variant stays as `size`).
- **`size`** — the shipped `size_cost` semantics (Const/Opaque = 0,
  node = 1 + children, dim slot counted — documented honest cost).
- **`op_table`** — per-op-name weights, `default`/`leaf` fallbacks,
  view-op zeroing (`_VIEW_OPS` port). Covers `count_cost`,
  `param_bytes_cost` (leaf = param bytes), and the launch-penalty
  variants once a `per_term` constant exists.
- **`flops`** (P2-late/P3) — parametric formulas over inferred
  shapes; *requires* §2 stage-2 shape inference. Formula names, not
  arbitrary expressions: `"2*m*n*k"` resolves against the bound
  shape tuple; unknown op → `default`. Roofline (`roofline_cost`),
  fusion (`fusion_regions`/`fused_cost_for`), and executor costs
  (`executor_cost_for`, `lowering_aware_cost_for`) are **backend
  profiles + algorithms**, not tables — they plug in as named specs
  with a `backend` field (peak flops/BW, launch overhead constants —
  `_profile_constants` is already JSON-shaped data) when a consumer
  needs them. Until then they stay Python-side; a `CostSpec` kind
  that can't be interpreted must be **refused at load**, not
  approximated — declines are named.

### Engine-side interpretation

The current `extract_best`/`optimize` adapters lower every candidate
to `KernelTerm` then price — `to_kernel` per e-class member per
extraction round is honest but not free. `CostSpec` interprets over
the internal `ETerm` directly (leaf de-minting via the existing
`from_kernel`/`to_kernel` leaf convention, attrs read in place), so
built-in and table costs never cross the bridge per node. The
`FnMut(&KernelTerm)` overload stays for callers with genuinely
procedural costs (the same `missing_hooks` posture: named, flagged,
never silent).

---

## 4. Certificates / proof

**Decision: do not port `proof.py` in P2. `ProofEdge`/`Application`/
`enode_app`/`enode_origin`/`merge_log` are the level-2 provenance and
they suffice; level-2 certificate reconstruction stays Python-side
until a consumer (P3 verified delivery, gauntlet replay) demands it.**

Rationale: the Rust crate deliberately ships level-2 provenance —
the spec's `_applications`/`_merge_log` records — and `proof.py`'s
709 lines exist to *replay* that provenance into replayable
certificates (`Certificate`/`CertStep`, `_CERT_MAX_*` budgets). That
replay is a consumer concern: today the only cert consumers are the
Python gauntlet/lemma store, which keep running on the Python engine
through P3 anyway. Premature porting would freeze a certificate
format before its consumer exists.

What P2 *does* need: provenance must be **exportable** — a
`Saturated`/`OptimizeResult` consumer must be able to serialize
`merge_log` + `applications` to the `cert_to_data`-compatible shape
when the Python side wants to materialize a cert. That is a serde
concern on the provenance records (add when the first consumer
lands), not a proof-engine port. Honesty note: an e-graph that ran
with `missing_hooks`-flagged laws produces provenance that names the
*fired* rule — the weakened guard is visible in the record, so a
cert built on it inherits the flag. That composition already holds
in Python; keep it.

---

## 5. Engine trait surface

**Decision: mint a minimal `SearchEngine` trait inside
`hnl-engine-eqsat` — not in `hnl-engine`, not a new contract crate
yet. Promote only when a second search interpretation exists.**

### Why here

- `hnl-engine` (`engine/`) holds the *kernel* contracts —
  `LanguageView`/`AstView`/`Model`/`OpHandler` — deliberately free of
  domain concepts. Laws, guards, cost specs, and saturation budgets
  are search-domain; putting `SearchEngine` there would leak the
  interpretation's vocabulary into the kernel crate.
- A separate contract crate (`engine-eqsat-contract` or similar) is
  speculative: ADR-0175 names `engine-eqsat-*` as the interpret-layer
  family, and the only inhabitant today is this one. The dissolution
  map's own rule — names are scaffolding, layers are structure —
  says: define the trait where the first implementation lives;
  extract when a second interpretation (beam search, A*/ILP, a
  learned-policy searcher) actually lands. That is the same posture
  `play.register()` took in graphopt/kernelopt: concrete first,
  seam on evidence.
- The trait is also the natural home for the *kit-facing* request
  shape spike §6.5 sketched: `optimize(term, laws, cost, opts)` where
  `laws` is a `LawSet` (records + compiled `Rewrite`s) and `cost` a
  `CostSpec`/closure — i.e. the trait's signature **is** the P2
  contract surface, so `hnl optimize model.hnl --kit kit-eqsat` can
  dispatch on it later without the CLI knowing about e-graphs.

### Shape (sketch)

```rust
pub trait SearchEngine {
    /// Saturate `terms` under `laws`; return roots + provenance.
    fn saturate(&self, terms: &[KernelTerm], laws: &LawSet,
                opts: &SaturateOptions) -> Result<Saturated, Error>;
    /// One-shot: lift, saturate, extract min-cost, lower.
    fn optimize(&self, term: &KernelTerm, laws: &LawSet,
                cost: &CostSpec, opts: &OptimizeOptions)
        -> Result<OptimizeResult, Error>;
    /// Name for provenance/diagnostics ("eqsat").
    fn name(&self) -> &'static str;
}
```

`EGraphEngine` (the current free functions, wrapped) is the sole
impl. `LawSet`/`CostSpec` are the §1/§3 types — the trait takes data,
not closures, which is what makes a second interpretation
implementable without re-deriving the contract.

---

## Migration path (P2 → P3)

1. **Land the codecs**: `law::from_data`/`LawSet`, `cost::CostSpec`
   interpreter, `cond`/`dspec` stage-1 guards. Reference corpus:
   the guard-free `builtin.rs` laws re-expressed as records (dogfood
   the format before porting).
2. **Port batch 1**: `SIMPLIFICATION_RULES` + the hand-promoted laws
   whose guards are unconditional or `cond`/`dspec`-expressible —
   serialize from Python via `law_to_data` → transcode patterns to
   the `Pat` JSON → differential-test extracted terms against
   `mlopt_core.egraph` on the discovery corpus (spike §6.6 harness).
3. **Stage-2 guards**: shape inference lands; the `tensor.py`/
   `priorart.py`/`scan.py` guarded majority ports. Procedural
   `check=`/`derive=` laws stay flagged; their Python consumers
   keep the Python engine until re-expressed.
4. **Costs**: `count`/`size`/`op_table` first; `flops` after shape
   inference; roofline/fusion/executor as backend-profiled named
   specs when a consumer demands them.
5. **P3**: `mlopt_core.egraph` demotes to spec/reference; kit-eqsat
   holds the law/cost corpus; `hnl optimize` dispatches on
   `SearchEngine`.

## What stays Python (through P3, honestly)

- The full `cost/` algebra beyond `op_table` (roofline, fusion,
  executor, lowering-aware) — backend algorithms, not tables.
- `proof.py` certificate reconstruction — consumer-driven port.
- Procedural `check=`/`derive=` laws and the `TensorAlgebra`
  (`tval`/`tapp`/`mint`) dspec channels — flagged, named, never
  silently weakened.
- `mlopt_core.egraph` itself — the spec the Rust port mirrors.

## Honesty invariants preserved

- **No unverified extraction**: unchanged — extraction is
  deterministic min-cost; the verify gate is a consumer concern and
  `missing_hooks`-flagged laws keep their flags in provenance.
- **Declines are named**: `Error::Extract` exists; new
  `Error::UnserializableLaw`, `Error::UnknownCostKind`,
  `Error::GuardUnsupported` join it — a guard atom the interpreter
  doesn't know *refuses the law at load*, never approximates.
- **Guards decline, never fabricate**: the `cond`/`dspec` strictness
  contract ports verbatim — unproven ⇒ `false`/veto.
- **One codec**: law records are JSON data; `.hnl` embeds, never
  re-spells. `handl_egraph::Term` never appears in a signature or a
  record.
- **Provenance stays inspectable**: `Saturated::graph` keeps
  `ProofEdge`/`Application`; law names in records match
  `Rewrite::name` so `merge_log` lines up with kit data.
