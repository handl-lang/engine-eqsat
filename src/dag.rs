//! `dag.program` wire records — shared-subtree term input.
//!
//! The kernel [`hnl_engine::Term`] is a *tree* (`Box`/`Vec` children,
//! no sharing), so a hash-consed DAG input tree-expands on the wire —
//! exponentially, and past `serde_json`'s recursion limit outright.
//! The ADR-0173 envelope solves this at the codec layer: a top-level
//! `{"Opaque": {"__tag__": "dag.program", "nodes": [...], "root": n}}`
//! record carries a **node table** — every distinct node once, an
//! `Op`'s `dim`/`args` as *indices* into `nodes` (children preceding
//! parents), leaves as complete records. The JSON itself is shallow
//! (table entries are flat objects), so term depth never reaches the
//! deserializer.
//!
//! This module lifts such a record **straight into e-class ids**,
//! bottom-up over the table: each node's children resolve to the
//! classes minted for earlier indices, so sharing survives as shared
//! e-classes instead of tree expansion. The `Op` encoding deliberately
//! mirrors [`handl_egraph::from_kernel`]'s `Term::Op` arm — `dim` at
//! child slot 0, `source`/`foreign`/`lit`/`__attrs__` attrs — so laws,
//! guards, and shape inference see byte-identical enodes whichever
//! wire spelling the term arrived in. Leaf records decode through the
//! ordinary kernel-`Term` path, keeping the self-describing leaf-key
//! convention (`Const` payloads — including the `__mlopt_leaf__:`
//! bridge leaves — round-trip verbatim).
//!
//! Node-record spellings this module accepts:
//!
//! * `{"Op": {"dim": <idx>, "name": <op>, "args": [<idx>…],
//!   "source": <ArrSource>, "lit"?: <value>, "attrs"?: <object>}}`
//!   — `dim`/`args` are node-table indices, everything else the
//!   kernel `Term::Op` serde fields;
//! * `{"Const": {"value": <value>}}` / `{"Arr": {"dim": <term>,
//!   "value": <value>}}` — leaves, decoded as kernel `Term`s (the
//!   `__mlopt_leaf__:`-style payload leaves included).
//!
//! Anything else — a nested `Opaque`, an unknown node tag, a non-index
//! `args` entry, an out-of-range index — is a named refusal, never a
//! silent approximation.

use std::rc::Rc;

use handl_egraph::{AttrValue, ClassId, EGraph, from_kernel};
use hnl_engine::ArrSource;
use serde_json::Value;

use crate::engine::Error;

/// The `dag.program` record's `__tag__` value (ADR-0173 / hnl_codec).
const DAG_TAG: &str = "dag.program";

/// If `v` is a `{"Opaque": …}` record, return its payload — refusing
/// non-`dag.program` tags by name. `None` means *v* is an ordinary
/// (tree-position) term JSON.
fn opaque_payload(v: &Value) -> Result<Option<&Value>, Error> {
    let Some(obj) = v.as_object() else {
        return Ok(None);
    };
    let Some(payload) = obj.get("Opaque") else {
        return Ok(None);
    };
    let tag = payload
        .as_object()
        .and_then(|o| o.get("__tag__"))
        .and_then(Value::as_str);
    if tag != Some(DAG_TAG) {
        return Err(Error::BadTerm(format!(
            "Opaque term (tag {tag:?}) is not a dag.program record"
        )));
    }
    Ok(Some(payload))
}

/// Add one wire term JSON value to `graph`; return its root e-class.
///
/// Tree-position `Term` JSON goes through the ordinary kernel decode +
/// [`from_kernel`] lift; a top-level `dag.program` record decodes
/// node-table-first (see module docs).
pub(crate) fn add_json(graph: &mut EGraph, v: &Value, provenance: &str) -> Result<ClassId, Error> {
    match opaque_payload(v)? {
        Some(payload) => add_dag(graph, payload, provenance),
        None => {
            let kt = serde_json::from_value::<hnl_engine::Term>(v.clone())
                .map_err(|e| Error::BadTerm(format!("term record: {e}")))?;
            graph
                .add_term(&from_kernel(&kt), provenance)
                .map_err(Error::AddTerm)
        }
    }
}

/// Decode the `dag.program` payload's node table bottom-up.
fn add_dag(graph: &mut EGraph, rec: &Value, provenance: &str) -> Result<ClassId, Error> {
    let obj = rec
        .as_object()
        .ok_or_else(|| Error::BadTerm("dag.program: payload must be an object".into()))?;
    let nodes = obj
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::BadTerm("dag.program: 'nodes' must be an array".into()))?;
    let root = obj
        .get("root")
        .and_then(Value::as_u64)
        .ok_or_else(|| Error::BadTerm("dag.program: 'root' must be an index".into()))?;
    if root as usize >= nodes.len() {
        return Err(Error::BadTerm(format!(
            "dag.program: bad root index {root} ({} nodes)",
            nodes.len()
        )));
    }
    // Post-order table: every index a node references is earlier, so
    // `built[i]` is the e-class of `nodes[i]` by the time it is read.
    let mut built: Vec<ClassId> = Vec::with_capacity(nodes.len());
    for (i, n) in nodes.iter().enumerate() {
        let cid = add_node(graph, n, i, &built, provenance)?;
        built.push(cid);
    }
    Ok(built[root as usize])
}

/// Resolve a node-table index reference — must point at an *earlier*
/// node (post-order: children precede parents).
fn node_ref(v: &Value, i: usize, field: &str) -> Result<usize, Error> {
    let idx = v.as_u64().ok_or_else(|| {
        Error::BadTerm(format!(
            "dag.program: node {i} {field} must be a node index"
        ))
    })? as usize;
    if idx >= i {
        return Err(Error::BadTerm(format!(
            "dag.program: node {i} {field} index {idx} is not an \
             earlier node (0..{})",
            i.saturating_sub(1)
        )));
    }
    Ok(idx)
}

/// Add one table entry; `built` holds the e-classes of nodes `0..i`.
fn add_node(
    graph: &mut EGraph,
    n: &Value,
    i: usize,
    built: &[ClassId],
    provenance: &str,
) -> Result<ClassId, Error> {
    let Some(obj) = n.as_object() else {
        return Err(Error::BadTerm(format!(
            "dag.program: node {i} must be a tagged record"
        )));
    };
    if obj.contains_key("Op") {
        return add_op_node(graph, &obj["Op"], i, built, provenance);
    }
    if obj.contains_key("Const") || obj.contains_key("Arr") {
        // A leaf record — the ordinary kernel decode keeps the
        // self-describing leaf-key convention (mlopt bridge leaves
        // included). No table indices inside a leaf record.
        let kt = serde_json::from_value::<hnl_engine::Term>(n.clone())
            .map_err(|e| Error::BadTerm(format!("dag.program node {i}: {e}")))?;
        return graph
            .add_term(&from_kernel(&kt), provenance)
            .map_err(Error::AddTerm);
    }
    Err(Error::BadTerm(format!(
        "dag.program: node {i} has unsupported tag {:?}",
        obj.keys().collect::<Vec<_>>()
    )))
}

/// The `Term::Op` arm of [`from_kernel`], node-table edition: child
/// slot 0 is `dim`, then `args`; `source`/`foreign`/`lit`/`__attrs__`
/// ride attrs exactly as the tree bridge spells them.
fn add_op_node(
    graph: &mut EGraph,
    node: &Value,
    i: usize,
    built: &[ClassId],
    provenance: &str,
) -> Result<ClassId, Error> {
    let bad = |m: &str| Error::BadTerm(format!("dag.program node {i}: {m}"));
    let obj = node
        .as_object()
        .ok_or_else(|| bad("'Op' must be an object"))?;
    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("'name' must be a string"))?;
    let source_v = obj
        .get("source")
        .ok_or_else(|| bad("'source' is required"))?;
    let source: ArrSource =
        serde_json::from_value(source_v.clone()).map_err(|e| bad(&format!("bad 'source': {e}")))?;

    let dim_idx = node_ref(
        obj.get("dim")
            .ok_or_else(|| bad("'dim' index is required"))?,
        i,
        "dim",
    )?;
    let arg_vals = obj
        .get("args")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("'args' must be an index list"))?;

    let mut children = Vec::with_capacity(arg_vals.len() + 1);
    children.push(built[dim_idx]);
    for a in arg_vals {
        children.push(built[node_ref(a, i, "arg")?]);
    }

    // -- attrs, mirroring from_kernel's Term::Op arm ---------------
    // (`add_enode` takes bare `AttrValue`s — `TermAttr::Meta` cannot
    // appear in a concrete input term.)
    let mut attrs: Vec<(Rc<str>, AttrValue)> =
        vec![(Rc::from("source"), source_tag(&source).into())];
    if let ArrSource::Foreign(n) = &source {
        attrs.push((Rc::from("foreign"), n.as_str().into()));
    }
    if let Some(l) = obj.get("lit") {
        attrs.push((Rc::from("lit"), AttrValue::from(l)));
    }
    if let Some(a) = obj.get("attrs") {
        attrs.push((Rc::from("__attrs__"), AttrValue::from(a)));
    }

    Ok(graph.add_enode(name, &children, attrs, Some(provenance)))
}

/// `ArrSource` → the `"source"` attr spelling `from_kernel` emits.
fn source_tag(s: &ArrSource) -> &'static str {
    match s {
        ArrSource::Native => "native",
        ArrSource::Foreign(_) => "foreign",
        ArrSource::Judgment => "judgment",
        ArrSource::DynamicForeign => "dynamic",
    }
}

// ---------------------------------------------------------------------------
//  The return direction — e-graph `Term` -> wire JSON (tree or dag.program)
// ---------------------------------------------------------------------------
//
// Extraction hands back a *shared* `handl_egraph::Term` (Rc-cloned
// subtrees). The kernel `Term` — and therefore `to_kernel` — is a
// tree, so serialising it naively re-expands the DAG exponentially,
// exactly the failure the input envelope fixes. This half mirrors it:
// one identity-keyed walk (the same pointer trick `priced` uses —
// structural `Hash`/`Eq` on a DAG is the exponential walk) counts
// occurrences, distinct nodes, and depth, then emits either the plain
// tree record (small, shallow terms — byte-compatible with the
// pre-dag wire) or a `dag.program` node table.

use handl_egraph::Term as ETerm;
use std::collections::HashMap;

/// Identity key for shared-node memoisation — pointer-equal `Rc`
/// allocations are the same node (see `handl_egraph::extract`'s
/// `TermKey`, kept crate-local so this module stands alone).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct NodeKey(u8, usize, usize, usize);

pub(crate) fn node_key(t: &ETerm) -> NodeKey {
    match t {
        ETerm::Var(n) => NodeKey(0, n.as_bytes().as_ptr() as usize, n.len(), 0),
        ETerm::Leaf(k) => NodeKey(1, k.as_bytes().as_ptr() as usize, k.len(), 0),
        ETerm::Op(o) => NodeKey(
            2,
            o.op.as_bytes().as_ptr() as usize,
            o.args.as_ptr() as usize,
            o.attrs.as_ptr() as usize,
        ),
    }
}

/// Tree-expansion guards for the plain-tree spelling: past either
/// bound the `dag.program` record is emitted instead — occurrences
/// bound the JSON size, depth stays under `serde_json`'s recursion
/// limit (128) with margin.
const TREE_MAX_OCCURRENCES: u64 = 20_000;
const TREE_MAX_DEPTH: u32 = 96;

/// Walk `t` once per *distinct* node and compute, by DP over the
/// node table:
///
/// * `order`/`index` — the post-order table (children before
///   parents) the `dag.program` record emits;
/// * `occurrences` — the tree-expansion size `Σ_v paths(root → v)`.
///   Counting it by walking edges is exponential on a DAG, so it is
///   computed instead: `occ[root] = 1`, and every node's count flows
///   to its children — parents all precede children in reverse
///   post-order, so one reverse pass settles every count;
/// * `depth_max` — `1 + max(child depths)`, likewise DP.
///
/// Iterative throughout — term depth is unbounded.
fn collect_nodes(t: &ETerm) -> (Vec<ETerm>, u64, u32, HashMap<NodeKey, usize>) {
    // Pass 1: iterative post-order over *distinct* nodes.
    let mut order: Vec<ETerm> = Vec::new();
    let mut index: HashMap<NodeKey, usize> = HashMap::default();
    // (node, next-child cursor); children are Rc-shared clones.
    let mut work: Vec<(ETerm, usize)> = vec![(t.clone(), 0)];
    let mut visited: HashMap<NodeKey, ()> = HashMap::default();
    while let Some((node, ci)) = work.last_mut() {
        let children: &[ETerm] = match &*node {
            ETerm::Op(o) => &o.args,
            _ => &[],
        };
        if *ci < children.len() {
            let child = children[*ci].clone();
            *ci += 1;
            let key = node_key(&child);
            if visited.insert(key, ()).is_none() {
                work.push((child, 0));
            }
        } else {
            let done = work.pop().expect("non-empty").0;
            let key = node_key(&done);
            index.insert(key, order.len());
            order.push(done);
        }
    }
    let n = order.len();

    // Pass 2 (reverse post-order — every parent before its
    // children): propagate occurrence counts root → leaves.
    let root_idx = index[&node_key(t)];
    let mut occ = vec![0u64; n];
    occ[root_idx] = 1;
    let mut occurrences = 0u64;
    for i in (0..n).rev() {
        let c = occ[i];
        occurrences = occurrences.saturating_add(c);
        if let ETerm::Op(o) = &order[i] {
            for child in o.args.iter() {
                let ci = index[&node_key(child)];
                occ[ci] = occ[ci].saturating_add(c);
            }
        }
    }

    // Pass 3 (post-order — children before parents): depth DP.
    let mut depth = vec![0u32; n];
    let mut depth_max = 0u32;
    for i in 0..n {
        let d = match &order[i] {
            ETerm::Op(o) => {
                1 + o
                    .args
                    .iter()
                    .map(|c| depth[index[&node_key(c)]])
                    .max()
                    .unwrap_or(0)
            }
            _ => 1,
        };
        depth[i] = d;
        depth_max = depth_max.max(d);
    }

    (order, occurrences, depth_max, index)
}

/// One node-table / tree record for an e-graph `Term`, in the kernel
/// `Term::Op` serde spelling this wire uses (`dim` at child slot 0,
/// `source`/`foreign`/`lit`/`attrs` channels — the inverse of
/// [`add_op_node`]). `kids` are already-resolved wire positions
/// (tree: nested records are built by the caller; dag: node indices).
fn op_record(o: &handl_egraph::Op, kids: &[Value]) -> Result<Value, Error> {
    // -- attrs -> kernel fields (inverse of from_kernel's Op arm) --
    // Enode attrs are `(key, TermAttr)` pairs; only `Lit` survives —
    // a `Meta` attr is pattern-position data with no kernel spelling.
    let mut source: Option<ArrSource> = None;
    let mut foreign: Option<String> = None;
    let mut lit: Option<Value> = None;
    let mut attrs: Option<Value> = None;
    for (k, v) in o.attrs.iter() {
        let handl_egraph::TermAttr::Lit(av) = v else {
            return Err(Error::BadTerm(format!(
                "dag output: attr {k:?} is a pattern metavariable"
            )));
        };
        match k.as_ref() {
            "source" => {
                let s = match av {
                    AttrValue::Str(s) => s.as_ref(),
                    other => {
                        return Err(Error::BadTerm(format!(
                            "dag output: 'source' attr is not a string: {}",
                            other.to_json()
                        )));
                    }
                };
                source = Some(match s {
                    "native" => ArrSource::Native,
                    "foreign" => ArrSource::Foreign(String::new()),
                    "judgment" => ArrSource::Judgment,
                    "dynamic" => ArrSource::DynamicForeign,
                    other => {
                        return Err(Error::BadTerm(format!(
                            "dag output: unknown 'source' attr {other:?}"
                        )));
                    }
                });
            }
            "foreign" => {
                foreign = Some(match av {
                    AttrValue::Str(s) => s.to_string(),
                    other => other.to_json().to_string(),
                });
            }
            "lit" => lit = Some(av.to_json()),
            "__attrs__" => attrs = Some(av.to_json()),
            other => {
                return Err(Error::BadTerm(format!(
                    "dag output: enode attr {other:?} has no kernel Term spelling"
                )));
            }
        }
    }
    let source = match source {
        Some(ArrSource::Foreign(_)) => ArrSource::Foreign(foreign.unwrap_or_default()),
        Some(s) => s,
        None => {
            return Err(Error::BadTerm(
                "dag output: enode has no 'source' attr — not a \
                 kernel Term::Op spelling"
                    .into(),
            ));
        }
    };
    let source_json = serde_json::to_value(&source)
        .map_err(|e| Error::BadTerm(format!("dag output: source encode: {e}")))?;

    // children[0] is the dim slot for op-encoded enodes.
    let (dim_kid, arg_kids) = kids
        .split_first()
        .ok_or_else(|| Error::BadTerm("dag output: op-encoded enode has no dim child".into()))?;
    let name: &str = &o.op;
    let mut node = serde_json::json!({
        "dim": dim_kid,
        "name": name,
        "args": arg_kids,
        "source": source_json,
    });
    if let Some(l) = lit {
        node["lit"] = l;
    }
    if let Some(a) = attrs {
        node["attrs"] = a;
    }
    Ok(serde_json::json!({"Op": node}))
}

/// The `Term::Arr` leaf inverse of `from_kernel`'s `"arr"` arm
/// (`handl_egraph::term_bridge`): an `"arr"` enode with exactly one
/// child (the dim slot) and exactly one attr — `("value", Lit)` — is
/// the e-graph spelling of `{"Arr": {"dim": …, "value": …}}`. `dim`
/// is emitted as a *tree* record, not a table index: leaf records
/// carry no node-table references (the `add_node` decode runs them
/// through the plain kernel-`Term` path), matching the
/// `__mlopt_leaf__` records hnl_codec emits.
///
/// Only the canonical spelling decodes: an `"arr"` enode with one
/// child but any other attr set is a named refusal — `to_kernel`'s
/// forward arm reads *any* one-child `"arr"` enode as `Term::Arr`
/// (defaulting/dropping attrs), so an `Op` record would misreport
/// what the term actually is. `"arr"` enodes with ≠1 child are
/// ordinary ops and fall through to [`op_record`].
fn arr_leaf_record(o: &handl_egraph::Op) -> Result<Option<Value>, Error> {
    if &*o.op != "arr" || o.args.len() != 1 {
        return Ok(None);
    }
    let bad = |m: &str| {
        Error::BadTerm(format!(
            "dag output: 'arr' enode {m} — not the canonical Term::Arr spelling"
        ))
    };
    if o.attrs.len() != 1 {
        return Err(bad("must carry exactly the 'value' attr"));
    }
    let (k, v) = &o.attrs[0];
    if k.as_ref() != "value" {
        return Err(bad("attr must be 'value'"));
    }
    let handl_egraph::TermAttr::Lit(av) = v else {
        return Err(bad("'value' attr is a pattern metavariable"));
    };
    let dim = tree_json(&o.args[0])?;
    Ok(Some(serde_json::json!({
        "Arr": {"dim": dim, "value": av.to_json()}
    })))
}

/// Leaf record: the e-graph leaf key IS a serialized kernel `Term` —
/// parse it back into the record it names (`{"Const": …}` etc.).
fn leaf_record(key: &str) -> Result<Value, Error> {
    serde_json::from_str::<Value>(key)
        .map_err(|e| Error::BadTerm(format!("dag output: leaf key is not term JSON: {e}")))
}

/// Serialise an extracted e-graph `Term` to wire JSON — the plain tree
/// record for small shallow terms, the `dag.program` node table
/// otherwise.
pub(crate) fn term_to_wire(t: &ETerm) -> Result<Value, Error> {
    let (order, occurrences, depth, index) = collect_nodes(t);
    // Tree whenever the expansion is bounded — sharing alone does not
    // force the record (a shared `add(x, x)` is three nodes either
    // way); past either bound the table is the only honest spelling.
    let use_dag = occurrences > TREE_MAX_OCCURRENCES || depth > TREE_MAX_DEPTH;
    if !use_dag {
        // Tree spelling: occurrences == distinct, so recursion is
        // bounded by the (small) node count.
        return tree_json(t);
    }
    // dag.program: `order` is post-order (children before parents);
    // Op args/dim are indices, leaves are complete records.
    let mut nodes: Vec<Value> = Vec::with_capacity(order.len());
    for node in &order {
        nodes.push(node_record(node, &index)?);
    }
    let root = index[&node_key(t)];
    Ok(serde_json::json!({
        "Opaque": {
            "__tag__": DAG_TAG,
            "nodes": nodes,
            "root": root,
        }
    }))
}

/// `index`-position of each collected node — children always precede
/// parents, so references are valid table indices.
fn node_record(node: &ETerm, index: &HashMap<NodeKey, usize>) -> Result<Value, Error> {
    match node {
        ETerm::Leaf(key) => leaf_record(key),
        ETerm::Var(name) => Err(Error::BadTerm(format!(
            "dag output: metavariable {name:?} reached extraction"
        ))),
        ETerm::Op(o) => {
            if let Some(rec) = arr_leaf_record(o)? {
                return Ok(rec);
            }
            // children[0] = dim, rest = args — indices into the table.
            let kids: Vec<Value> = o
                .args
                .iter()
                .map(|c| serde_json::json!(index[&node_key(c)] as u64))
                .collect();
            op_record(o, &kids)
        }
    }
}

/// The plain-tree spelling — only reached when occurrences ==
/// distinct and depth is bounded, so plain recursion is safe.
fn tree_json(t: &ETerm) -> Result<Value, Error> {
    match t {
        ETerm::Leaf(key) => leaf_record(key),
        ETerm::Var(name) => Err(Error::BadTerm(format!(
            "dag output: metavariable {name:?} reached extraction"
        ))),
        ETerm::Op(o) => {
            if let Some(rec) = arr_leaf_record(o)? {
                return Ok(rec);
            }
            let kids: Vec<Value> = o.args.iter().map(tree_json).collect::<Result<_, _>>()?;
            op_record(o, &kids)
        }
    }
}
