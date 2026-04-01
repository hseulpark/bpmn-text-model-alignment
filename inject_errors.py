import os
import csv
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Set, Dict
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

@dataclass
class AppliedError:
    error_type: str
    details: str
    payload: Dict[str, Any] = field(default_factory=dict)  

# -----------------------------
# Namespaces
# -----------------------------
NS = {
    "c": "http://cpee.org/ns/description/1.0",
    "a": "http://cpee.org/ns/annotation/1.0",
}
ET.register_namespace("", NS["c"])
ET.register_namespace("a", NS["a"])

def q(tag: str) -> str:
    return f"{{{NS['c']}}}{tag}"

A_MUTATED = f"{{{NS['a']}}}mutated"   # annotation-safe marker

DUMMY_TASK_LABELS = ["Review request", "Update database", "Validate data"]

# -----------------------------
# XML helpers
# -----------------------------
def parent_map(root: ET.Element) -> Dict[ET.Element, ET.Element]:
    pm = {}
    for p in root.iter():
        for ch in list(p):
            pm[ch] = p
    return pm

def find_calls(root: ET.Element) -> List[ET.Element]:
    return list(root.findall(".//c:call", NS))

def is_task_call(call_el: ET.Element) -> bool:
    t = call_el.find("./c:parameters/c:type", NS)
    return (t is not None and (t.text or "").strip() == ":task")

def get_task_label(call_el: ET.Element) -> str:
    lab = call_el.find("./c:parameters/c:label", NS)
    return (lab.text or "").strip() if lab is not None else ""

def set_task_label(call_el: ET.Element, new_label: str) -> None:
    params = call_el.find("./c:parameters", NS)
    if params is None:
        params = ET.SubElement(call_el, q("parameters"))
    lab = params.find("./c:label", NS)
    if lab is None:
        lab = ET.SubElement(params, q("label"))
    lab.text = new_label

def collect_existing_labels(root: ET.Element) -> set:
    out = set()
    for c in find_calls(root):
        lab = get_task_label(c).strip().lower()
        if lab:
            out.add(lab)
    return out

def next_call_id(root: ET.Element) -> str:
    ids = []
    for c in find_calls(root):
        cid = c.get("id", "")
        m = re.match(r"a(\d+)$", cid)
        if m:
            ids.append(int(m.group(1)))
    n = (max(ids) + 1) if ids else 1
    return f"a{n}"

def call_key(call_el: ET.Element) -> str:
    """Stable lock key for a call: prefer @id, else a:alt_id, else label."""
    cid = call_el.get("id")
    if cid:
        return f"id:{cid}"
    # sometimes calls have a:alt_id
    for k, v in call_el.attrib.items():
        if k.endswith("alt_id"):
            return f"alt_id:{v}"
    lab = get_task_label(call_el).strip().lower()
    return f"label:{lab}"

def gateway_key(gw_el: ET.Element) -> str:
    """
    Stable lock key for a gateway node:
    prefer a:alt_id attribute if present, else use in-run object id.
    Also if already marked mutated, still lock it.
    """
    if gw_el.get(A_MUTATED) == "true":
        # still want a stable key if possible
        for k, v in gw_el.attrib.items():
            if k.endswith("alt_id"):
                return f"alt_id:{v}"
        return f"py:{id(gw_el)}"
    for k, v in gw_el.attrib.items():
        if k.endswith("alt_id"):
            return f"alt_id:{v}"
    return f"py:{id(gw_el)}"

def remove_child(parent: ET.Element, child: ET.Element) -> None:
    kids = list(parent)
    if child in kids:
        kids.remove(child)
        parent[:] = kids

def insert_children_at(parent: ET.Element, idx: int, new_children: List[ET.Element]) -> None:
    kids = list(parent)
    parent[:] = kids[:idx] + new_children + kids[idx:]

def replace_child(parent: ET.Element, old: ET.Element, new: ET.Element) -> None:
    kids = list(parent)
    idx = kids.index(old)
    kids[idx] = new
    parent[:] = kids

def swap_children(parent: ET.Element, a: ET.Element, b: ET.Element) -> None:
    kids = list(parent)
    ia, ib = kids.index(a), kids.index(b)
    kids[ia], kids[ib] = kids[ib], kids[ia]
    parent[:] = kids

def build_dummy_call(root: ET.Element, label: str) -> ET.Element:
    call_el = ET.Element(q("call"), {"id": next_call_id(root), "endpoint": ""})
    params = ET.SubElement(call_el, q("parameters"))
    ET.SubElement(params, q("label")).text = label
    ET.SubElement(params, q("method")).text = ""
    ET.SubElement(params, q("type")).text = ":task"
    ET.SubElement(params, q("arguments")).text = ""
    return call_el

# -----------------------------
# Strict “sequence” pairs for merge/swap:
# - MUST be consecutive <call> siblings
# - both :task
# - both unlocked
# - parent must be a “flow container” (NOT gateway container)
# -----------------------------
FLOW_CONTAINER_TAGS = {
    "description",
    "alternative",
    "parallel_branch",
}

def strict_consecutive_task_pairs_unlocked(
    root: ET.Element,
    locked_calls: Set[str],
    locked_gateways: Set[str],
) -> List[Tuple[ET.Element, ET.Element, ET.Element]]:
    """
    returns [(parent, callA, callB), ...]
    where A and B are consecutive sibling calls and both are unlocked :task.
    Also avoids parents that are under a locked gateway (optional but safer).
    """
    pm = parent_map(root)
    out = []

    # Precompute “ancestor locked gateway” check
    def has_locked_gateway_ancestor(el: ET.Element) -> bool:
        cur = el
        while cur in pm:
            cur = pm[cur]
            local = cur.tag.split("}")[-1]
            if local in {"parallel", "choose"}:
                if gateway_key(cur) in locked_gateways:
                    return True
        return False

    for parent in root.iter():
        p_local = parent.tag.split("}")[-1]
        if p_local not in FLOW_CONTAINER_TAGS:
            continue

        kids = list(parent)
        for i in range(len(kids) - 1):
            a, b = kids[i], kids[i + 1]
            if a.tag != q("call") or b.tag != q("call"):
                continue
            if not is_task_call(a) or not is_task_call(b):
                continue
            if call_key(a) in locked_calls or call_key(b) in locked_calls:
                continue
            # extra safety: don't mess inside already mutated gateways
            if has_locked_gateway_ancestor(a) or has_locked_gateway_ancestor(b):
                continue
            out.append((parent, a, b))

    return out

def nonconsecutive_task_pairs_unlocked_same_parent(
    root: ET.Element,
    locked_calls: Set[str],
    locked_gateways: Set[str],
) -> List[Tuple[ET.Element, ET.Element, ET.Element]]:
    """
    random_sequences:
    - pick two calls under the same FLOW_CONTAINER parent but NOT adjacent
    - both :task
    - both unlocked
    """
    pm = parent_map(root)

    def has_locked_gateway_ancestor(el: ET.Element) -> bool:
        cur = el
        while cur in pm:
            cur = pm[cur]
            local = cur.tag.split("}")[-1]
            if local in {"parallel", "choose"}:
                if gateway_key(cur) in locked_gateways:
                    return True
        return False

    out = []
    for parent in root.iter():
        p_local = parent.tag.split("}")[-1]
        if p_local not in FLOW_CONTAINER_TAGS:
            continue

        kids = list(parent)
        call_idxs = [i for i, k in enumerate(kids) if k.tag == q("call") and is_task_call(k)]
        for ai in range(len(call_idxs)):
            for bi in range(ai + 1, len(call_idxs)):
                i, j = call_idxs[ai], call_idxs[bi]
                if j == i + 1:
                    continue
                a, b = kids[i], kids[j]
                if call_key(a) in locked_calls or call_key(b) in locked_calls:
                    continue
                if has_locked_gateway_ancestor(a) or has_locked_gateway_ancestor(b):
                    continue
                out.append((parent, a, b))
    return out

# -----------------------------
# Error ops (calls)
# -----------------------------
def err_missing_task(root: ET.Element, rng: random.Random, locked_calls: Set[str]) -> Optional[AppliedError]:
    pm = parent_map(root)
    candidates = [c for c in find_calls(root) if is_task_call(c) and call_key(c) not in locked_calls]
    if not candidates:
        return None

    victim = rng.choice(candidates)
    parent = pm.get(victim)
    if parent is None:
        return None

    lbl = get_task_label(victim)
    locked_calls.add(call_key(victim))  # lock BEFORE removing
    remove_child(parent, victim)
    return AppliedError(
        error_type="missing_task",
        details=f"removed call(label='{lbl}')",
        payload={
            "call_id": victim.get("id"),
            "label": lbl},
    )

def err_additional_task(root: ET.Element, rng: random.Random, locked_calls: Set[str]) -> Optional[AppliedError]:
    existing = collect_existing_labels(root)

    chosen = None
    for base in rng.sample(DUMMY_TASK_LABELS, k=len(DUMMY_TASK_LABELS)):
        if base.lower() not in existing:
            chosen = base
            break
    if chosen is None:
        suffix = 1
        while f"dummy_{suffix}".lower() in existing:
            suffix += 1
        chosen = f"dummy_{suffix}"

    new_call = build_dummy_call(root, chosen)
    # lock the inserted call so later errors cannot touch it
    locked_calls.add(call_key(new_call))

    kids = list(root)
    idx = rng.randrange(0, len(kids) + 1)
    insert_children_at(root, idx, [new_call])

    return AppliedError(
        error_type="additional_task",
        details=f"inserted call(label='{chosen}') at root index {idx}",
        payload={
            "inserted_call_id": new_call.get("id"),
            "label": chosen,
            "root_index": idx,
        },
    )

def merge_labels(label1: str, label2: str) -> str:
    l1 = (label1 or "").strip()
    l2 = (label2 or "").strip()
    if not l1 and not l2:
        return ""
    if not l1:
        return l2
    if not l2:
        return l1
    if l1.lower() == l2.lower():
        return l1
    return f"{l1} and {l2}"

def err_merge_tasks_strict(
    root: ET.Element,
    rng: random.Random,
    locked_calls: Set[str],
    locked_gateways: Set[str],
) -> Optional[AppliedError]:
    pairs = strict_consecutive_task_pairs_unlocked(root, locked_calls, locked_gateways)
    if not pairs:
        return None

    parent, a, b = rng.choice(pairs)
    la, lb = get_task_label(a), get_task_label(b)
    new_label = merge_labels(la, lb)

    # lock both calls BEFORE modifying tree
    locked_calls.add(call_key(a))
    locked_calls.add(call_key(b))

    set_task_label(a, new_label)
    remove_child(parent, b)
    return AppliedError(
        error_type="merged",
        details=f"merged '{la}' + '{lb}' -> '{new_label}' (removed second call)",
        payload={
            "label": new_label,
            "first_call_id": call_key(a),
            "second_call_id": call_key(b),
        },
    )

def err_swap_consecutive_strict(
    root: ET.Element,
    rng: random.Random,
    locked_calls: Set[str],
    locked_gateways: Set[str],
) -> Optional[AppliedError]:
    pairs = strict_consecutive_task_pairs_unlocked(root, locked_calls, locked_gateways)
    if not pairs:
        return None

    parent, a, b = rng.choice(pairs)
    la, lb = get_task_label(a), get_task_label(b)

    locked_calls.add(call_key(a))
    locked_calls.add(call_key(b))

    swap_children(parent, a, b)
    return AppliedError(
        error_type="2_wrong_sequences",
        details=f"swapped consecutive tasks '{la}' <-> '{lb}'",
        payload={
            "first_call_id": call_key(a),
            "second_call_id": call_key(b),
        },
    )

def err_swap_random_nonconsecutive(
    root: ET.Element,
    rng: random.Random,
    locked_calls: Set[str],
    locked_gateways: Set[str],
) -> Optional[AppliedError]:
    pairs = nonconsecutive_task_pairs_unlocked_same_parent(root, locked_calls, locked_gateways)
    if not pairs:
        return None

    parent, a, b = rng.choice(pairs)
    la, lb = get_task_label(a), get_task_label(b)

    locked_calls.add(call_key(a))
    locked_calls.add(call_key(b))

    swap_children(parent, a, b)
    return AppliedError(
        error_type="random_sequences",
        details=f"swapped non-consecutive sibling tasks '{la}' <-> '{lb}'",
        payload={
            "first_call_id": call_key(a),
            "second_call_id": call_key(b),
        },
    )

# -----------------------------
# Gateway transform (annotation-safe)
# -----------------------------
def _copy_non_branch_children(src_gateway_el: ET.Element, branch_tag_names: Set[str]) -> List[ET.Element]:
    out = []
    for ch in list(src_gateway_el):
        local = ch.tag.split("}")[-1]
        if local in branch_tag_names:
            continue
        out.append(deepcopy(ch))
    return out

def find_parallel_nodes_unlocked(root: ET.Element, locked_gateways: set[str]) -> List[ET.Element]:
    out = []
    for par in root.findall(".//c:parallel", NS):
        if gateway_key(par) not in locked_gateways:
            out.append(par)
    return out

def find_xor_nodes_unlocked(root: ET.Element, locked_gateways: Set[str]) -> List[ET.Element]:
    out = []
    for ch in root.findall(".//c:choose", NS):
        if (ch.get("mode", "").strip().lower() == "exclusive"):
            if gateway_key(ch) not in locked_gateways:
                out.append(ch)
    return out

A_NS = "http://cpee.org/ns/annotation/1.0"

def a_attr(name: str) -> str:
    """namespaced annotation attribute key, e.g., a:alt_id"""
    return f"{{{A_NS}}}{name}"

def convert_xor_to_parallel(ch: ET.Element) -> ET.Element:
    """
    <choose mode='exclusive'> ... </choose>
      -> <parallel wait='-1' cancel='last' ...> <parallel_branch>...</parallel_branch> ... </parallel>
    """
    new_par = ET.Element(q("parallel"), {"wait": "-1", "cancel": "last"})

    for k, v in ch.attrib.items():
        if k == a_attr("alt_id") or k.endswith("}alt_id"):
            new_par.set(k, v)

    new_par.set(a_attr("mutated"), "true")

    for non in _copy_non_branch_children(ch, branch_tag_names={"alternative"}):
        new_par.append(non)
        
    for alt in ch.findall("./c:alternative", NS):
        pb = ET.SubElement(new_par, q("parallel_branch"))
        for child in list(alt):
            pb.append(deepcopy(child))

    return new_par

def convert_parallel_to_xor(par: ET.Element) -> ET.Element:
    """
    <parallel wait='-1' cancel='last' ...>
      <parallel_branch>...</parallel_branch>
      <parallel_branch>...</parallel_branch>
    </parallel>

    ->

    <choose mode="exclusive" a:alt_id="...">
      <alternative condition="">...</alternative>
      <alternative condition="">...</alternative>
    </choose>
    """
    choose = ET.Element(q("choose"), {"mode": "exclusive"})

    for k, v in par.attrib.items():
        if k == a_attr("alt_id") or k.endswith("}alt_id"):
            choose.set(k, v)

    choose.set(a_attr("mutated"), "true")

    for non in _copy_non_branch_children(par, branch_tag_names={"parallel_branch"}):
        choose.append(non)

    for pb in par.findall("./c:parallel_branch", NS):
        alt = ET.SubElement(choose, q("alternative"), {"condition": ""})
        for child in list(pb):
            alt.append(deepcopy(child))
    return choose

def flatten_parallel_to_sequence_safe(par: ET.Element) -> List[ET.Element]:
    flattened = []
    flattened.extend(_copy_non_branch_children(par, branch_tag_names={"parallel_branch"}))
    for br in par.findall(q("parallel_branch")):
        for ch in list(br):
            flattened.append(deepcopy(ch))
    return flattened

def flatten_xor_to_sequence_safe(ch: ET.Element) -> List[ET.Element]:
    flattened = []
    flattened.extend(_copy_non_branch_children(ch, branch_tag_names={"alternative"}))
    for alt in ch.findall(q("alternative")):
        for sub in list(alt):
            flattened.append(deepcopy(sub))
    return flattened

def err_and_to_xor(
    root: ET.Element,
    rng: random.Random,
    locked_gateways: Set[str],
) -> Optional[AppliedError]:
    pm = parent_map(root)
    pars = find_parallel_nodes_unlocked(root, locked_gateways)
    if not pars:
        return None

    par = rng.choice(pars)
    parent = pm.get(par)
    if parent is None:
        return None

    locked_gateways.add(gateway_key(par))  # lock old gateway before replacing
    new_node = convert_parallel_to_xor(par)
    locked_gateways.add(gateway_key(new_node))  # lock new gateway too

    replace_child(parent, par, new_node)
    return AppliedError(
        error_type="and_to_xor",
        details="converted <parallel> to <choose mode='exclusive'> (annotation-safe)",
        payload={
            "from": {
                "tag": "parallel",
                "alt_id": par.get(a_attr("alt_id")) or par.get("{http://cpee.org/ns/annotation/1.0}alt_id"),
                "wait": par.get("wait"),
                "cancel": par.get("cancel"),
                "n_branches": len(par.findall("./c:parallel_branch", NS)),
            },
            "to": {
                "tag": "choose",
                "mode": "exclusive",
                "alt_id": new_node.get(a_attr("alt_id")) or new_node.get("{http://cpee.org/ns/annotation/1.0}alt_id"),
                "n_alternatives": len(new_node.findall("./c:alternative", NS)),
            },
        },
    )

def err_xor_to_and(
    root: ET.Element,
    rng: random.Random,
    locked_gateways: Set[str],
) -> Optional[AppliedError]:
    pm = parent_map(root)
    xors = find_xor_nodes_unlocked(root, locked_gateways)
    if not xors:
        return None

    ch = rng.choice(xors)
    parent = pm.get(ch)
    if parent is None:
        return None

    locked_gateways.add(gateway_key(ch))
    new_node = convert_xor_to_parallel(ch)
    locked_gateways.add(gateway_key(new_node))

    replace_child(parent, ch, new_node)
    return AppliedError(
        error_type="xor_to_and",
        details="converted <choose mode='exclusive'> to <parallel> (annotation-safe)",
        payload={
            "from": {
                "tag": "choose",
                "mode": ch.get("mode"),
                "alt_id": ch.get(a_attr("alt_id")) or ch.get("{http://cpee.org/ns/annotation/1.0}alt_id"),
            },
            "to": {
                "tag": "parallel",
                "alt_id": new_node.get(a_attr("alt_id")) or new_node.get("{http://cpee.org/ns/annotation/1.0}alt_id"),
                "n_branches": len(new_node.findall("./c:parallel_branch", NS)),
            },
        },
    )

def err_and_to_seq(
    root: ET.Element,
    rng: random.Random,
    locked_gateways: Set[str],
) -> Optional[AppliedError]:
    pm = parent_map(root)
    pars = find_parallel_nodes_unlocked(root, locked_gateways)
    if not pars:
        return None

    par = rng.choice(pars)
    parent = pm.get(par)
    if parent is None:
        return None

    kids = list(parent)
    idx = kids.index(par)

    locked_gateways.add(gateway_key(par))
    seq = flatten_parallel_to_sequence_safe(par)

    remove_child(parent, par)
    insert_children_at(parent, idx, seq)
    return AppliedError(
        error_type="and_to_seq",
        details="flattened <parallel> into sequential children (annotation-safe)",
        payload={
            "from": {
                "tag": "parallel",
                "alt_id": par.get(a_attr("alt_id")) or par.get("{http://cpee.org/ns/annotation/1.0}alt_id"),
                "n_branches": len(par.findall("./c:parallel_branch", NS)),
            },
            "to": {
                "tag": "sequence",
                "n_children": len(seq),
            },
        },
    )

def err_xor_to_seq(
    root: ET.Element,
    rng: random.Random,
    locked_gateways: Set[str],
) -> Optional[AppliedError]:
    pm = parent_map(root)
    xors = find_xor_nodes_unlocked(root, locked_gateways)
    if not xors:
        return None

    ch = rng.choice(xors)
    parent = pm.get(ch)
    if parent is None:
        return None

    kids = list(parent)
    idx = kids.index(ch)

    locked_gateways.add(gateway_key(ch))
    seq = flatten_xor_to_sequence_safe(ch)

    remove_child(parent, ch)
    insert_children_at(parent, idx, seq)
    return AppliedError(
        error_type="xor_to_seq",
        details="flattened <choose mode='exclusive'> into sequential children (annotation-safe)",
        payload={
            "from": {
                "tag": "choose",
                "mode": ch.get("mode"),
                "alt_id": ch.get(a_attr("alt_id")) or ch.get("{http://cpee.org/ns/annotation/1.0}alt_id"),
            },
            "to": {
                "tag": "sequence",
                "n_children": len(seq),
            },
        },
    )

# -----------------------------
# Feasibility + dispatcher
# -----------------------------
def list_calls_unlocked(root: ET.Element, locked_calls: Set[str]) -> List[ET.Element]:
    return [c for c in find_calls(root) if is_task_call(c) and call_key(c) not in locked_calls]

def feasible_error_types(root: ET.Element, locked_calls: Set[str], locked_gateways: Set[str]) -> List[str]:
    ok = []

    # 1) missing / additional
    if list_calls_unlocked(root, locked_calls):
        ok.append("missing_task")
    ok.append("additional_task") 
    
    # 2) merge / 2wrongseq: strict consecutive sequence pair
    if strict_consecutive_task_pairs_unlocked(root, locked_calls, locked_gateways):
        ok.append("merged")
        ok.append("2 wrong sequences")

    # 3) random_sequences: nonconsecutive pair
    if nonconsecutive_task_pairs_unlocked_same_parent(root, locked_calls, locked_gateways):
        ok.append("random sequences")

    # 4) AND gateway
    if find_parallel_nodes_unlocked(root, locked_gateways):
        ok.append("AND -> XOR")
        ok.append("AND -> SEQ")

    # 5) XOR gateway 
    if find_xor_nodes_unlocked(root, locked_gateways):
        ok.append("XOR -> AND")
        ok.append("XOR -> SEQ")

    return ok

FORCE_ERROR_MAP = {
    # task-level
    "missing_task": "missing_task",
    "additional_task": "additional_task",
    "merged": "merged",
    "two_wrong_sequences": "2 wrong sequences",  
    "random_sequences": "random sequences",

    # gateways
    "and_to_xor": "AND -> XOR",
    "and_to_seq": "AND -> SEQ",
    "xor_to_and": "XOR -> AND",
    "xor_to_seq": "XOR -> SEQ",
}

def normalize_force_error(s: str) -> str:
    """
    user CLI input -> standardized snake key in FORCE_ERROR_MAP
    examples:
      "AND -> SEQ" -> "and_to_seq"
      "2 wrong sequences" -> "two_wrong_sequences"
      "random sequences" -> "random_sequences"
    """
    t = (s or "").strip().lower()
    t = t.replace("-", "_").replace(" ", "_").replace("->", "_to_")
    t = re.sub(r"_+", "_", t).strip("_")

    # common aliases
    aliases = {
        "2_wrong_sequences": "two_wrong_sequences",
        "two_wrong_sequence": "two_wrong_sequences",
        "random_sequence": "random_sequences",
        "and_to_xor": "and_to_xor",
        "and_to_seq": "and_to_seq",
        "xor_to_and": "xor_to_and",
        "xor_to_seq": "xor_to_seq",
    }
    return aliases.get(t, t)

def apply_one(error_type: str, root: ET.Element, rng: random.Random,
              locked_calls: Set[str], locked_gateways: Set[str]) -> Optional[AppliedError]:

    if error_type == "missing_task":
        return err_missing_task(root, rng, locked_calls)

    if error_type == "additional_task":
        return err_additional_task(root, rng, locked_calls)

    if error_type == "merged":
        return err_merge_tasks_strict(root, rng, locked_calls, locked_gateways)

    if error_type == "2 wrong sequences":
        return err_swap_consecutive_strict(root, rng, locked_calls, locked_gateways)

    if error_type == "random sequences":
        return err_swap_random_nonconsecutive(root, rng, locked_calls, locked_gateways)

    # gateway
    if error_type == "AND -> XOR":
        return err_and_to_xor(root, rng, locked_gateways)

    if error_type == "AND -> SEQ":
        return err_and_to_seq(root, rng, locked_gateways)

    if error_type == "XOR -> AND":
        return err_xor_to_and(root, rng, locked_gateways)

    if error_type == "XOR -> SEQ":
        return err_xor_to_seq(root, rng, locked_gateways)

    return None

# -----------------------------
# Output path helpers
# -----------------------------
def make_out_path_unique(in_path: str, out_dir: str, suffix: str = "_err") -> str:
    stem, ext = os.path.splitext(os.path.basename(in_path))
    if not ext:
        ext = ".xml"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"{stem}{suffix}_{ts}{ext}")

# -----------------------------
# Main: apply N random errors
# -----------------------------
def apply_random_errors(
    xml_path: str,
    out_path: str,
    n_errors: int = 3,
    seed: Optional[int] = None,
    force_error: Optional[str] = None,   
) -> List[AppliedError]:
    rng = random.Random(seed)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    applied: List[AppliedError] = []
    used_error_types: set[str] = set()

    locked_calls: set[str] = set()
    locked_gateways: set[str] = set()
    used_error_types: set[str] = set() 
    
    if force_error:
        if n_errors != 1:
            n_errors = 1

        locked_calls: set[str] = set()
        locked_gateways: set[str] = set()

        candidates = feasible_error_types(root, locked_calls, locked_gateways)
        if force_error not in candidates:
            raise RuntimeError(f"force_error '{force_error}' not feasible for {xml_path}. feasible={candidates}")

        e = apply_one(force_error, root, rng, locked_calls, locked_gateways)
        if e is None:
            raise RuntimeError(f"force_error '{force_error}' returned None for {xml_path}")

        applied = [e]
        tree.write(out_path, encoding="utf-8", xml_declaration=True)
        return applied

    has_and = len(find_parallel_nodes_unlocked(root, locked_gateways)) > 0

    if has_and and n_errors >= 1:
        forced = rng.choice(["AND -> XOR", "AND -> SEQ"])
        if forced == "AND -> XOR":
            e = err_and_to_xor(root, rng, locked_gateways)
        else:
            e = err_and_to_seq(root, rng, locked_gateways)

        if e is None:
            raise RuntimeError("AND gateway exists but could not apply AND transformation error.")
        applied.append(e)
        used_error_types.add(forced)

    remaining = n_errors - len(applied)
    max_attempts = 500
    attempts = 0

    while remaining > 0 and attempts < max_attempts:
        attempts += 1

        candidates = feasible_error_types(root, locked_calls, locked_gateways)
        candidates = [c for c in candidates if c not in used_error_types]

        if has_and:
            candidates = [c for c in candidates if c not in ("AND -> XOR", "AND -> SEQ")]
        
        candidates = [c for c in candidates if c not in used_error_types]  

        if not candidates:
            break

        et = rng.choice(candidates)
        e = apply_one(et, root, rng, locked_calls, locked_gateways)
        if e is None:
            continue

        applied.append(e)
        remaining -= 1
        used_error_types.add(et)

    if remaining > 0:
        raise RuntimeError(f"Could only apply {len(applied)}/{n_errors} errors (not enough feasible errors without reusing locked items).")

    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return applied

# -----------------------------
# CSV log
# -----------------------------

ERROR_COLUMNS = [
    "missing_task",
    "additional_task",
    "merged_tasks",
    "swapped_sequence",
    "random_sequences",
    "and_to_xor",
    "xor_to_and",
    "and_to_seq",
    "xor_to_seq",
]

ERROR_TYPE_NORMALIZE = {
    "missing_task": "missing_task",
    "additional_task": "additional_task",
    "merged_tasks": "merged_tasks",
    "swapped_sequence": "swapped_sequence",
    "random_sequences": "random_sequences",

    "gateway_mismatch_AND_to_XOR": "and_to_xor",
    "gateway_mismatch_XOR_to_AND": "xor_to_and",
    "gateway_removal_AND_to_SEQ": "and_to_seq",
    "gateway_removal_XOR_to_SEQ": "xor_to_seq",

    "and_to_xor": "and_to_xor",
    "xor_to_and": "xor_to_and",
    "and_to_seq": "and_to_seq",
    "xor_to_seq": "xor_to_seq",
}

HEADER_COL_MAP = {
    "missing_task": "missing",
    "missing": "missing",

    "additional_task": "additional",
    "additional": "additional",

    "merged_tasks": "merged",
    "merge": "merged",
    "merged": "merged",
    "merged_task": "merged",

    "split_tasks": "splitted",
    "split": "splitted",
    "splitted": "splitted",

    "swapped_sequence": "2 wrong sequences",
    "2_wrong_sequences": "2 wrong sequences",
    "two_wrong_sequences": "2 wrong sequences",
    "2_wrong_sequences": "2 wrong sequences",
    "2_wrong_sequences".replace(" ", "_"): "2 wrong sequences",
    "2_wrong_sequences".lower(): "2 wrong sequences",
    "2_wrong_sequences".replace("-", "_"): "2 wrong sequences",

    "random_sequences": "random sequences",
    "random_sequence": "random sequences",

    "and_to_xor": "AND -> XOR",
    "xor_to_and": "XOR -> AND",
    "xor_to_seq": "XOR -> SEQ",
    "and_to_seq": "AND -> SEQ",
}

def normalize_error_type_key(s: str) -> str:
    t = (s or "").strip()
    t = t.replace("-", "_")
    t = t.replace(" ", "_")
    return t.lower()

def _clean_header_cell(s: str) -> str:
    return (s or "").replace("\ufeff", "").strip()

def normalize_csv_log_inplace(log_path: str) -> None:
    if not os.path.exists(log_path):
        return

    with open(log_path, "r", newline="", encoding="utf-8") as f:
        raw = f.read()

    raw = raw.lstrip("\ufeff")

    first_line = raw.splitlines()[0] if raw.splitlines() else ""
    delim = ";" if first_line.count(";") >= first_line.count(",") else ","

    rows = list(csv.reader(raw.splitlines(), delimiter=delim))
    if not rows:
        return

    header = [h.strip().lstrip("\ufeff") for h in rows[0]]
    header_join = ";".join(header)

    cleaned = [header]
    for r in rows[1:]:
        rr = [x.strip().lstrip("\ufeff") for x in r]
        if ";".join(rr) == header_join:
            continue
        if all(x == "" for x in rr):
            continue
        cleaned.append(rr)

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerows(cleaned)

def append_log_csv(log_path: str, input_xml: str, applied_errors: List[AppliedError]) -> None:
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"CSV log not found: {log_path}")

    with open(log_path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f, delimiter=";")
        header_raw = next(r, None)
    if not header_raw:
        raise RuntimeError("CSV header is missing/empty")

    header = [_clean_header_cell(h) for h in header_raw]

    row = {h: "" for h in header}

    row[header[0]] = os.path.basename(input_xml)

    for e in applied_errors:
        et_raw = getattr(e, "error_type", "")
        et_norm = normalize_error_type_key(et_raw)
        
        col = HEADER_COL_MAP.get(et_raw)
        if col is None:
            col = HEADER_COL_MAP.get(et_norm)

        if col is None:
            continue

        for h in header:
            if _clean_header_cell(h) == col:
                row[h] = "x"
                break

    with open(log_path, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow([row[h] for h in header])
        
def append_history_json(
    json_path: str,
    input_xml: str,
    output_xml: str,
    seed: Optional[int],
    n_errors: int,
    applied: List[AppliedError],
) -> None:
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_xml": os.path.basename(input_xml),   
        "output_xml": os.path.basename(output_xml), 
        "seed": seed,
        "n_errors": n_errors,
        "errors": [
            {
                "type": e.error_type,
                "details": e.details,
                "payload": getattr(e, "payload", {}) or {},
            }
            for e in applied
        ],
    }

    # load existing
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
        if not isinstance(data, list):
            data = [data]
    else:
        data = []

    data.append(record)

    # save back (pretty)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        

# -----------------------------
# CLI usage example
# -----------------------------
if __name__ == "__main__":
    import argparse
    import os
    import glob
    import re

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_xml", required=True)
    ap.add_argument("--out_dir", default="models_with_error") 
    ap.add_argument("--n_errors", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log_csv", default="error_injection_log.csv")
    ap.add_argument("--log_json", default="error_injection_history.json")
    ap.add_argument("--force_error", default=None,
                help="Apply exactly ONE specified error type and exit. "
                     "Examples: and_to_seq, and_to_xor, xor_to_seq, xor_to_and, "
                     "missing_task, additional_task, merged, two_wrong_sequences, random_sequences")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    def next_counter(out_dir: str, base: str, n_errors: int) -> int:

        name_tag = f"err{int(n_errors)}" 
        pattern = os.path.join(out_dir, f"{base}.{name_tag}.*.xml")
        files = glob.glob(pattern)

        max_c = 0
        rx = re.compile(rf"^{re.escape(base)}\.{re.escape(name_tag)}\.(\d{{3}})\.xml$", re.I)

        for fp in files:
            name = os.path.basename(fp)
            m = rx.match(name)
            if not m:
                continue
            c = int(m.group(1))
            if c > max_c:
                max_c = c

        return max_c + 1

    base = os.path.splitext(os.path.basename(args.in_xml))[0]
    force_key = normalize_force_error(args.force_error) if args.force_error else None
    name_tag = f"err{int(args.n_errors)}"
    ctr = next_counter(args.out_dir, base, args.n_errors)
    out_xml = os.path.join(args.out_dir, f"{base}.err{args.n_errors}.{ctr:03d}.xml")

    applied = apply_random_errors(
        xml_path=args.in_xml,
        out_path=out_xml,
        n_errors=args.n_errors,
        seed=args.seed,
        force_error=args.force_error
    )

    print("Wrote:", out_xml)
    for e in applied:
        print("-", e.error_type, "=>", e.details)
    print("Logged to:", args.log_csv)

    append_log_csv(args.log_csv, args.in_xml, applied)

    REPO_ROOT = Path(__file__).resolve().parent.parent
    
    append_history_json(
        json_path=str(REPO_ROOT / "history.json"),
        input_xml=args.in_xml,
        output_xml=out_xml,
        seed=args.seed,
        n_errors=args.n_errors,
        applied=applied,
    )
    print("History appended to:", os.path.join(args.out_dir, "history.json"))