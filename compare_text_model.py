from __future__ import annotations
from copy import deepcopy

import os
import re
import json
import argparse
import xml.etree.ElementTree as ET
import yaml
import spacy
import requests
import numpy as np
import tempfile

from typing import List, Dict, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher
from functools import lru_cache


# ---------------------------------------------------------------------
# Initialize spaCy and Sentence-BERT
# ---------------------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    raise RuntimeError("Please run: python -m spacy download en_core_web_lg")

FILLER_WORDS = {"first", "then", "next", "finally", "after", "before"}
PROCESS_META_SENT_RE = re.compile(
    r"^\s*(the\s+process\s+)?(starts?|begins?|ends?|finishes?|terminates?)\s*[\.\!]*\s*$",
    re.I
)

sbert = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------------------
# Read user input text
# ---------------------------------------------------------------------
def read_user_text_from_yaml(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    exposition = None
    def walk(obj):
        nonlocal exposition
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
        elif isinstance(obj, str) and exposition is None:
            if len(obj.split()) > 2:
                exposition = obj

    walk(data)
    return exposition.strip() if exposition else ""

def read_user_text(path: str) -> str:
    if path.lower().endswith((".yaml", ".yml")):
        return read_user_text_from_yaml(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def drop_filler_phrases(text: str) -> str:
    text = re.sub(
        r"(?i)\b("
        r"first|second|third|next|then|after|before|after that|finally|"
        r"at first|in the beginning|later|afterwards|"
        r"subsequently|eventually|prior to|once|when|while"
        r")\b[\s,]*",
        "",
        text,
    )
    return text.strip()

def normalize_task(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"\b(the|a|an)\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _is_subphrase_of_existing(phrase: str, existing: list[str]) -> bool:
    p = normalize_task(phrase)
    p_tokens = set(re.findall(r"[a-z0-9]+", p))
    if not p_tokens:
        return False

    for e in existing:
        e_norm = normalize_task(e)
        e_tokens = set(re.findall(r"[a-z0-9]+", e_norm))

        # Return True if the phrase is a token subset of an existing task and is strictly shorter.
        if p_tokens.issubset(e_tokens) and p != e_norm:
            return True

    return False

# ---------------------------------------------------------------------
# Extract tasks from user input
# ---------------------------------------------------------------------

OUTPUT_META_LEMMAS = {
    "start", "begin", "end", "finish", "terminate",
    "split", "branch", "complete", "process"
}

OUTPUT_META_PHRASES = {
     "parallel activities", "complete branches", "both branches", "process ends", "process starts", "end process", "complete either", "follow this"
}

def is_output_meta_task(task: str) -> bool:
    t = task.lower().strip()
    if t in OUTPUT_META_PHRASES:
        return True
    toks = re.findall(r"[a-z]+", t)
    if len(toks) == 1 and toks[0] in OUTPUT_META_LEMMAS:
        return True
    return False

SUBJECT_DEPS = {"nsubj", "nsubjpass"}   
OBJECT_DEPS  = {"dobj", "pobj", "attr", "dative", "oprd"}  
SUBJECT_POS  = {"NOUN", "PROPN"}        
OBJ_POS      = {"NOUN", "PROPN"}        

def _np_span(tok) -> str:
    parts = []
    for left in tok.lefts:
        if left.dep_ in {"compound", "amod", "poss"} and left.is_alpha:
            parts.append(left.text.lower())
    parts.append(tok.text.lower())
    return " ".join(parts)

def looks_narrative(text: str) -> bool:
    cues = ["department", "storehouse", "company", "customer", "member of"]
    t = text.lower()
    return sum(c in t for c in cues) >= 2

SUBJ_CUE_RE = re.compile(r"\b(company|department|storehouse|engineering|sales|customer|member)\b", re.I)

def has_subject_like_phrase(task: str) -> bool:
    # Patterns such as "sales department ..." or "storehouse ..."
    return bool(SUBJ_CUE_RE.search(task)) and len(task.split()) >= 3

def extract_svo_tasks(text: str, ordered: list[str]) -> list[str]:
    doc = nlp(text)

    AUX_LEMMAS = {"be", "have", "do"}
    SKIP_VERBS = AUX_LEMMAS | {"start", "begin", "end", "finish", "terminate", "split", "branch", "complete", "perform", "follow", "please"}

    candidates = []  # (start_char, phrase)

    for v in doc:
        if v.pos_ != "VERB":
            continue
        if v.lemma_.lower() in SKIP_VERBS:
            continue

        # ---------- subject ----------
        subj = None
        for ch in v.children:
            if ch.dep_ in SUBJECT_DEPS and ch.pos_ in SUBJECT_POS:
                subj = ch
                break

        if subj is None:
            for ch in v.children:
                if ch.dep_ == "agent":
                    for gch in ch.children:
                        if gch.dep_ == "pobj" and gch.pos_ in SUBJECT_POS:
                            subj = gch
                            break
                if subj is not None:
                    break

        if subj is None:
            continue

        subj_np = _np_span(subj)

        # ---------- verb (+ particle) ----------
        verb_tokens = [v.lemma_.lower()]
        prt = None
        for ch in v.children:
            if ch.dep_ == "prt":
                prt = ch.text.lower()
                break
        if prt:
            verb_tokens.append(prt)
        verb_part = " ".join(verb_tokens)

        # ---------- object ----------
        obj = None

        # 1) Direct object or attribute
        for ch in v.children:
            if ch.dep_ in {"dobj", "attr", "oprd", "dative"} and ch.pos_ in OBJ_POS:
                obj = ch
                break

        # 2) Prepositional object: verb -> prep -> pobj
        if obj is None:
            for prep in v.children:
                if prep.dep_ == "prep":
                    for pobj in prep.children:
                        if pobj.dep_ == "pobj" and pobj.pos_ in OBJ_POS:
                            obj = pobj
                            break
                if obj is not None:
                    break

        # Build the phrase
        if obj is not None:
            obj_np = _np_span(obj)
            phrase = f"{subj_np} {verb_part} {obj_np}"
        else:
            phrase = f"{subj_np} {verb_part}"

        phrase = re.sub(r"[^a-z0-9\s]", "", phrase.lower())
        phrase = re.sub(r"\s+", " ", phrase).strip()

        if not phrase:
            continue
        if phrase.split()[0] in FILLER_WORDS:
            continue

        if _is_subphrase_of_existing(phrase, ordered):
            continue

        candidates.append((v.idx, phrase))

    candidates.sort(key=lambda x: x[0])

    out = []
    seen = set()
    for _, p in candidates:
        if p not in seen and p not in ordered:
            out.append(p)
            seen.add(p)
    return out

def fallback_spacy_verb_object(text: str, ordered: list[str]) -> list[str]:
    doc = nlp(text)

    AUX_LEMMAS = {"be", "have", "do"}
    SKIP_VERBS = AUX_LEMMAS | {"start", "begin", "end", "finish", "terminate", "split", "branch", "complete", "perform", "follow", "please"}
    
    OBJ_DEPS = {"dobj", "pobj", "attr", "nsubjpass"} 

    candidates = []  

    for t in doc:
        if t.pos_ != "VERB":
            continue
        if t.lemma_.lower() in SKIP_VERBS:
            continue

        # (A) Verb (+ particle)
        verb_tokens = [t.lemma_.lower()]
        prt = None
        for ch in t.children:
            if ch.dep_ == "prt":  # e.g. "set up"
                prt = ch.text.lower()
                break
        if prt:
            verb_tokens.append(prt)

        # (B) Object
        obj = None
        for child in t.children:
            if child.dep_ in OBJ_DEPS and child.pos_ in {"NOUN","PROPN"}:
                obj = child
                break
        if obj is None:
            continue
        
        # Expand the object noun phrase
        noun_tokens = []
        for left in obj.lefts:
            if left.dep_ in {"compound", "amod"} and left.is_alpha:
                noun_tokens.append(left.text.lower())
        noun_tokens.append(obj.text.lower())

        phrase = " ".join(verb_tokens + noun_tokens)
        phrase = re.sub(r"[^a-z0-9\s]", "", phrase)
        phrase = re.sub(r"\s+", " ", phrase).strip()

        if not phrase:
            continue
        if phrase.split()[0] in FILLER_WORDS:
            continue

        if _is_subphrase_of_existing(phrase, ordered):
            continue

        candidates.append((t.idx, phrase))

    candidates.sort(key=lambda x: x[0])
    out = []
    seen = set()
    for _, p in candidates:
        if p not in seen and p not in ordered:
            out.append(p)
            seen.add(p)
    return out

# ---------------------------------------------------------------------
# Robust XOR narrative parser and precedence-edge extraction
# ---------------------------------------------------------------------

SPLIT_CUE_RE = re.compile(
    r"\b(two|2)\s+exclusive\s+paths\b"
    r"|\bone\s+of\s+two\s+exclusive\b"
    r"|\bfollows\s+one\s+of\b"
    r"|\b(one of two|one of 2)\s+(paths|branches)\b"
    r"|\b(two|2)\s+branches\b"
    r"|\bbased\s+on\s+the\s+outcome\b.*\b(exclusive\s+(decision|gateway)|two\s+exclusive\s+paths|two\s+paths)\b"
    r"|\bexclusive\s+(decision|gateway)\b"
    r"|\bbranches?\s+occurs\b"
    r"|\bwith\s+\d+\s+branches\b",
    re.I
)

IF_CUE_RE = re.compile(r"\bif\b", re.I)
ALT_CUE_RE = re.compile(r"\b(alternatively|otherwise|else)\b", re.I)
JOIN_CUE_RE = re.compile(r"\b(finally|in\s+the\s+end|process\s+ends?|ends?\s+after)\b", re.I)
CHECK_REF_RE = re.compile(r"\b(this|the)\s+check\b|\boutcome\s+of\s+(this|the)\s+check\b", re.I)

def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def _strip_light_meta(sents: List[str]) -> List[str]:
    out = []
    for s in sents:
        low = s.lower().strip()
        if low in {"the process starts.", "the process starts"}:
            continue
        if low in {"the process ends.", "the process ends"}:
            continue
        out.append(s)
    return out

def _find_split_index(sents: List[str]) -> Optional[int]:
    for i, s in enumerate(sents):
        if SPLIT_CUE_RE.search(s):
            return i
    whole = " ".join(sents)
    if ALT_CUE_RE.search(whole):
        for i, s in enumerate(sents):
            if IF_CUE_RE.search(s):
                return i
    return None

def _split_join_tail(rest_text: str) -> Tuple[str, str]:
    m = JOIN_CUE_RE.search(rest_text)
    if not m:
        return rest_text.strip(), ""
    return rest_text[:m.start()].strip(), rest_text[m.start():].strip()

def _extract_branch_chunks(rest_core: str) -> List[str]:
    # Branch start markers such as "if" or "alternatively/otherwise/else"
    markers = []
    for m in re.finditer(r"(?i)\bif\b|\b(alternatively|otherwise|else)\b", rest_core):
        markers.append(m.start())
    markers = sorted(set(markers))

    if not markers:
        return [rest_core.strip()] if rest_core.strip() else []

    chunks = []
    for k in range(len(markers)):
        a = markers[k]
        b = markers[k + 1] if (k + 1) < len(markers) else len(rest_core)
        chunk = rest_core[a:b].strip()
        if len(chunk.split()) >= 3:
            chunks.append(chunk)
    return chunks

def _split_condition_and_body(branch_chunk: str) -> Tuple[str, str]:
    """
    Robust condition/body split:
      1) "If COND, BODY"
      2) "Alternatively, if COND, BODY"
      3) "Alternatively BODY" 
      4) "If COND then BODY" 
    Returns: (condition_text, body_text)
    """
    s = branch_chunk.strip()

    # 1) Remove alternatively/otherwise/else prefixes from the main text
    alt_prefix = ""
    m_alt = re.match(r"(?is)^\s*(alternatively|otherwise|else)\s*,?\s*(.*)$", s)
    if m_alt:
        alt_prefix = m_alt.group(1).strip().lower()
        s = m_alt.group(2).strip()

    # 2) "if ..., ..."
    m_if_comma = re.match(r"(?is)^\s*if\s+(.*?),(.*)$", s)
    if m_if_comma:
        cond = m_if_comma.group(1).strip()
        body = m_if_comma.group(2).strip()
        if alt_prefix:
            cond = f"{alt_prefix}: if {cond}"
        else:
            cond = f"if {cond}"
        return cond, body

    # 3) "if ... then ..."
    m_if_then = re.match(r"(?is)^\s*if\s+(.*?)\s+then\s+(.*)$", s)
    if m_if_then:
        cond = m_if_then.group(1).strip()
        body = m_if_then.group(2).strip()
        if alt_prefix:
            cond = f"{alt_prefix}: if {cond}"
        else:
            cond = f"if {cond}"
        return cond, body

    # 4) If the chunk starts with "if" but the body is ambiguous, keep an empty body (conservative choice).
    if re.match(r"(?is)^\s*if\b", s):
        # Without a comma or "then", the split is more likely to be unreliable.
        # Keep the full condition and leave the body empty to avoid extracting tasks from condition text.
        cond = s.strip()
        if alt_prefix:
            cond = f"{alt_prefix}: {cond}"
        return cond, ""

    # 5) If only an alternative prefix is present, keep an empty condition and use the remainder as the body.
    if alt_prefix:
        return alt_prefix, s.strip()

    # Default
    return "", branch_chunk.strip()

def pick_split_anchor_task(pre_tasks: List[str], split_sentence: str) -> Optional[str]:
    """
    If the split-cue sentence refers to "this check",
    prefer a pre-split task containing check/determine/verify semantics as the split source.
    Otherwise, use the last pre-split task.
    """
    if not pre_tasks:
        return None

    if CHECK_REF_RE.search(split_sentence):
        # Heuristic: prefer tasks containing check-related verbs or expressions
        candidates = []
        for t in pre_tasks:
            low = t.lower()
            score = 0
            if "check" in low: score += 3
            if "determin" in low: score += 2
            if "relevant" in low: score += 2
            if "perform" in low: score += 1
            if score > 0:
                candidates.append((score, t))
        if candidates:
            candidates.sort(reverse=True, key=lambda x: x[0])
            return candidates[0][1]

    return pre_tasks[-1]

def parse_user_xor_and_edges(
    user_text: str,
    extract_tasks_fn,
) -> Dict[str, Any]:
    """
    Full pipeline:
      - split narrative into pre / branches / join
      - extract tasks for each part (using your extractor)
      - choose split anchor (robust)
      - build precedence edges
    """
    sents = _strip_light_meta(_sentences(user_text))
    split_idx = _find_split_index(sents)

    # Fallback: if no XOR cue is found, return only linear precedence edges
    if split_idx is None:
        tasks = extract_tasks_fn(user_text)
        edges = [(tasks[i], tasks[i+1]) for i in range(len(tasks)-1)]
        return {
            "note": "No clear XOR split cue found; returned linear precedence only.",
            "tasks": tasks,
            "xor": None,
            "edges": edges,
        }

    pre_text = " ".join(sents[:split_idx]).strip()
    rest_text = " ".join(sents[split_idx:]).strip()

    rest_core, join_text = _split_join_tail(rest_text)
    branch_chunks = _extract_branch_chunks(rest_core)

    # Tasks before the split
    pre_tasks = extract_tasks_fn(pre_text) if pre_text else []

    # Select the split anchor task; if the text refers to "this check", prefer a check-related task.
    split_sentence = sents[split_idx] if split_idx < len(sents) else rest_text
    split_src = pick_split_anchor_task(pre_tasks, split_sentence)

    branches = []
    for ch in branch_chunks:
        cond, body = _split_condition_and_body(ch)
        # If the body is empty, do not pass the condition into the task extractor.
        # This avoids false positives from conditional text.
        body_tasks = extract_tasks_fn(body) if body else []
        branches.append({
            "condition_text": cond,
            "body_text": body,
            "tasks": body_tasks,
            "raw": ch,
        })

    # For XOR patterns, keep the first two branches by default.
    branches = branches[:2]

    # Extract join tasks from the join segment. This segment may still contain mixed text.
    join_tasks = extract_tasks_fn(join_text) if join_text else []

    # ---- Build precedence edges ----
    edges = set()

    # Sequential edges in the pre-split part
    for i in range(len(pre_tasks) - 1):
        a, b = pre_tasks[i], pre_tasks[i+1]
        if a and b and a != b:
            edges.add((a, b))

    # Split source -> branch start
    join_target = join_tasks[0] if join_tasks else "END"

    if split_src and branches:
        for br in branches:
            ts = br.get("tasks", [])
            if not ts:
                continue
            edges.add((split_src, ts[0]))
            for i in range(len(ts) - 1):
                edges.add((ts[i], ts[i+1]))
            edges.add((ts[-1], join_target))

    # Join sequence + end
    if join_tasks:
        for i in range(len(join_tasks) - 1):
            edges.add((join_tasks[i], join_tasks[i+1]))
        edges.add((join_tasks[-1], "END"))

    return {
        "tasks": extract_tasks_fn(user_text),  # Also return the full task list
        "xor": {
            "pre_text": pre_text,
            "split_sentence": split_sentence,
            "split_src_task": split_src,
            "pre_tasks": pre_tasks,
            "branches": branches,
            "join_text": join_text,
            "join_tasks": join_tasks,
        },
        "edges": sorted(edges),
    }

def extract_verbal_tasks(text: str):
    text = re.sub(r"(?im)^\s*#\s*user\s*input\s*:\s*$", "", text)  # "# User Input:"
    text = re.sub(r"(?im)^\s*user\s*input\s*:\s*$", "", text)     # "User Input:"
    text = re.sub(r"(?im)^\s*#.*$", "", text)                     # any full-line comments
    text = re.sub(r"\n+", "\n", text).strip()
    # 1. Split into sentences and remove filler phrases
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for s in sentences:
        s = drop_filler_phrases(s.strip())
        if s and s[0].islower():
            s = s[0].upper() + s[1:]
        if not s:
            continue
        if PROCESS_META_SENT_RE.match(s):
            continue
        if not s.endswith("."):
            s += "."
        cleaned.append(s)

    tasks = []

    # 2. Run spaCy-based analysis for each sentence
    for sent_text in cleaned:

        # Split clauses by commas and conjunctions and parse them separately.
        clauses = re.split(r",|\band\b", sent_text, flags=re.I)

        for cl in clauses:
            cl = cl.strip()
            if not cl:
                continue

            doc = nlp(cl if cl.endswith(".") else cl + ".")

            AUX_LEMMAS = {"be", "have", "do"}

            verbs = []
            for t in doc:
                if t.lemma_ in AUX_LEMMAS:
                    continue
                if t.pos_ == "VERB":
                    verbs.append(t)
                    continue
                if t.pos_ == "ADJ" and t.tag_ == "VBN":
                    verbs.append(t)

            for verb in verbs:
                phrase_tokens = [verb.lemma_]

                # particle (set up)
                for child in verb.children:
                    if child.dep_ == "prt":
                        phrase_tokens.append(child.text)

                # Keep only objects or passive subjects
                for child in verb.children:
                    if child.dep_ in {"dobj", "pobj", "attr", "nsubjpass"}:   # Intentionally exclude active subjects here
                        # Add left-side modifiers first so phrases such as "data subject" remain intact.        
                        noun_tokens = []
                        for left in child.lefts:
                            if left.dep_ in {"compound", "amod", "poss"} and left.is_alpha:
                                noun_tokens.append(left.text)
                        noun_tokens.append(child.text)
                        
                        
                        phrase_tokens.extend(noun_tokens)

                phrase_str = " ".join(dict.fromkeys(phrase_tokens))
                phrase_str = re.sub(r"[^a-zA-Z0-9\s]", "", phrase_str)
                phrase_str = re.sub(r"\s+", " ", phrase_str).strip().lower()
                # Do not remove "perform" blindly.
                # Expressions such as "perform check" are meaningful tasks.
                # Remove it only when the object is overly generic.
                if phrase_str.startswith("perform "):
                    rest = phrase_str[len("perform "):].strip()
                    # "perform" can be removed for overly generic objects.
                    GENERIC_PERFORM_OBJECTS = {"task", "tasks", "step", "steps", "activity", "activities", "action", "actions"}
                    if rest in GENERIC_PERFORM_OBJECTS:
                        phrase_str = rest
                    # Otherwise, keep "perform" (e.g. "perform check").

                ALLOW_SINGLE_VERB_TASKS = {"serve"}  # Extend if additional single-verb tasks are needed.

                if len(phrase_str.split()) < 2:
                    # Allow selected single-verb tasks
                    if phrase_str in ALLOW_SINGLE_VERB_TASKS:
                        tasks.append(phrase_str)
                    continue
                if phrase_str.split()[0] not in FILLER_WORDS:
                    tasks.append(phrase_str)

            # Fallback 1: recover VERB + NOUN or compound-VERB + NOUN patterns
            for i in range(len(doc) - 1):
                token = doc[i]
                next_token = doc[i + 1]

                # Standard VERB + NOUN pattern (only when the noun is the local phrase head)
                if (
                    token.pos_ == "VERB"
                    and next_token.pos_ == "NOUN"
                    and next_token.dep_ in {"dobj", "pobj", "attr", "ROOT"}  # Only when the noun is the local phrase head
                ):
                    phrase = f"{token.text} {next_token.text}".lower()
                    if phrase not in tasks:
                        tasks.append(phrase)

                # Recover cases where a verb was misclassified as a compound token
                elif token.dep_ == "compound" and token.pos_ == "VERB" and next_token.pos_ == "NOUN":
                    phrase = f"{token.text} {next_token.text}".lower()
                    if phrase not in tasks:
                        tasks.append(phrase)

                # Recover imperative verbs at sentence start that were misclassified as PROPN (e.g. "Boil water").
                elif (
                    i == 0
                    and token.pos_ == "PROPN"
                    and next_token.pos_ == "NOUN"
                    and token.text[0].isupper()
                ):
                    phrase = f"{token.text.lower()} {next_token.text.lower()}"
                    if phrase not in tasks:
                        tasks.append(phrase)
                        
    # ---------------------------------------------------------------------
    # Extra rescue for clause-initial "X Y" imperative patterns
    # - Recover cases such as "Boil water" even if spaCy misses the verb tag
    # - Use a structural heuristic instead of a hardcoded verb list
    # ---------------------------------------------------------------------
    rescue_candidates = []  # (start_char, phrase)

    for sent_text in cleaned:
        clauses = re.split(r",|\band\b", sent_text, flags=re.I)

        for cl in clauses:
            cl = cl.strip()
            if not cl:
                continue

            doc = nlp(cl if cl.endswith(".") else cl + ".")
            toks = [t for t in doc if not t.is_space and not t.is_punct]
            if len(toks) < 2:
                continue

            t0, t1 = toks[0], toks[1]

            # The second token only needs to look nominal (NOUN/PROPN) to be accepted.
            if t1.pos_ not in {"NOUN", "PROPN"} or not t1.is_alpha:
                continue

            # Recover the first token even if it was misclassified as PROPN/NOUN.
            # Exclude pronouns, determiners, adpositions, conjunctions, and numbers to reduce false positives.
            if t0.pos_ in {"PRON", "DET", "ADP", "CCONJ", "SCONJ", "NUM"}:
                continue

            # Exclude filler or meta words
            v = t0.text.lower()
            n = t1.text.lower()
            if v in FILLER_WORDS or n in FILLER_WORDS:
                continue

            # Remove one-character noise tokens
            if len(v) <= 1 or len(n) <= 1:
                continue

            phrase = f"{v} {n}".strip()

            # Skip the phrase if it is already covered by a longer extracted task.
            if phrase in tasks:
                continue

            rescue_candidates.append((t0.idx, phrase))
    
    # 3. Remove duplicates while preserving order
    seen, ordered = set(), []
    
    BAD_TASKS = {
    "please user", "please input", "user input",
    "follow one", "follow one of", "follow one of two", "follow one of two exclusive paths",
    "follow this", "follow path", "follow paths",
    }
    
    for t in tasks:
        if t.startswith("follow one"):
            continue
        if t not in seen:
            ordered.append(t)
            seen.add(t)
            
    ordered = [t for t in ordered if t not in BAD_TASKS and not t.startswith("please ")]
    
    # Add rescue-based tasks in original text order.
    rescue_candidates.sort(key=lambda x: x[0])
    for _, phrase in rescue_candidates:
        if phrase in seen:
            continue
        if _is_subphrase_of_existing(phrase, ordered):
            continue
        ordered.append(phrase)
        seen.add(phrase)

    # === Narrative mode: SVO-first extraction with VO-noise filtering ===
    if looks_narrative(text):
        # 1) Use SVO extraction as the primary strategy
        ordered_svo = []
        ordered_svo.extend(extract_svo_tasks(text, ordered_svo))

        # 2) Add fallback verb-object tasks for missing cases
        extra_vo = fallback_spacy_verb_object(text, ordered_svo)
        ordered_svo.extend(extra_vo)

        # 3) Remove short verb-object tasks without explicit subjects
        filtered = []
        seen2 = set()
        for t in ordered_svo:
            # Keep phrases that still look like subject-based actions
            if has_subject_like_phrase(t):
                if t not in seen2:
                    filtered.append(t); seen2.add(t)

        # 4) Fallback safeguard: if too many tasks were removed, keep the broader set
        if len(filtered) >= 5:
            ordered = filtered
        else:
            ordered = ordered_svo

    else:
        # Keep the default extraction logic for imperative or short-clause text
        extra_svo = extract_svo_tasks(text, ordered)
        for p in extra_svo:
            ordered.append(p)

        extra = fallback_spacy_verb_object(text, ordered)
        for p in extra:
            ordered.append(p)
    # Apply an additional fallback only when too few tasks were extracted
    if len(ordered) < 5:
        extra = fallback_spacy_verb_object(text, ordered)
        for p in extra:
            ordered.append(p)
    return ordered

# ---------------------------------------------------------------------
# Extract task labels from BPMN/CPEE XML
# ---------------------------------------------------------------------
def parse_bpmn_tasks(path: str):
    tree = ET.parse(path)
    root = tree.getroot()
    tasks = []
    for label in root.findall(".//{*}label"):
        text = label.text or ""
        if text.strip():
            tasks.append(text.strip().lower())
    return tasks

def build_user_to_model_mapping(user_tasks, model_tasks, sim_threshold=0.25, return_debug=False):
    """
    Allow many-to-one mappings from user tasks to model tasks.
    If similarity is low but important keywords overlap (e.g. check/explain),
    still allow the mapping.
    """
    if not user_tasks or not model_tasks:
        return ({}, []) if return_debug else {}

    KEYWORDS = ["check", "explain", "notification", "notify", "receive", "retrieve", "eliminate"]

    def token_set(s: str) -> set:
        return set(re.findall(r"[a-z]+", normalize_task(s)))

    def kw_overlap(u: str, m: str) -> int:
        us = token_set(u)
        ms = token_set(m)
        return sum(1 for k in KEYWORDS if (k in us and k in ms))

    u2m = {}
    drops = []

    for u in user_tasks:
        best = None  # (total_score, sim, keyword_overlap, best_model_task)
        for m in model_tasks:
            sim = blended_sim(u, m, alpha=0.75)
            kw = kw_overlap(u, m)

            # Give a strong bonus to keyword overlap to avoid dropping valid mappings.
            total = sim + 0.20 * kw

            if best is None or total > best[0]:
                best = (total, sim, kw, m)

        total, sim, kw, best_m = best

        # If there is at least one overlapping keyword, allow the mapping even below the similarity threshold.
        if sim >= sim_threshold or kw >= 1:
            u2m[normalize_task(u)] = normalize_task(best_m)
        else:
            if return_debug:
                drops.append({
                    "user": normalize_task(u),
                    "best_model": normalize_task(best_m),
                    "sim": round(float(sim), 3),
                    "kw": int(kw),
                })

    return (u2m, drops) if return_debug else u2m


def translate_user_edges_to_model_space(user_edges, u2m: dict):
    """
    user_edges: set[(u_a,u_b)] in user label space (normalized already)
    u2m: dict { user_lbl -> model_lbl } many-to-one
    return: (expected_edges, collapsed_edges)
      - expected_edges: set[(m_a,m_b)] with m_a != m_b
      - collapsed_edges: list of user edges that collapsed to same model node (merge symptom)
    """
    expected = set()
    collapsed = []

    for ua, ub in user_edges:
        ma = u2m.get(ua)
        mb = u2m.get(ub)
        if not ma or not mb:
            continue
        if ma == mb:
            collapsed.append((ua, ub, ma))
            continue
        expected.add((ma, mb))

    return expected, collapsed


def count_exclusive_choose_in_cpee(xml_path: str) -> int:
    """
    Count the number of <choose mode="exclusive"> nodes in a CPEE XML file.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cnt = 0
    for ch in root.findall(".//c:choose", CPEE_NS):
        mode = (ch.get("mode") or "").strip().lower()
        if mode == "exclusive":
            cnt += 1
    return cnt

def count_parallel_in_cpee(xml_path: str) -> int:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return sum(1 for _ in root.findall(".//c:parallel", CPEE_NS))

def user_has_parallel_cue(user_text: str) -> bool:
    if not user_text:
        return False
    t = user_text.lower()
    return bool(re.search(r"\b(simultaneously|in parallel|at the same time|concurrently)\b", t)) \
        or bool(re.search(r"\bonce\s+both\b|\bboth\s+.*completed\b", t))

# ---------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------
def token_set(text: str) -> set:
    # Optional: stopword filtering could also be added here if needed.
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def jaccard_sim(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def sbert_cos_sim(a: str, b: str) -> float:
    ea = sbert.encode(a, convert_to_tensor=True)
    eb = sbert.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(ea, eb))

def blended_sim(a: str, b: str, alpha: float = 0.75) -> float:
    """
    alpha controls the SBERT weight (0 to 1). Values around 0.7 to 0.85 usually work well.
    """
    a = normalize_task(a)
    b = normalize_task(b)
    return alpha * sbert_cos_sim(a, b) + (1 - alpha) * jaccard_sim(a, b)

def find_task_index(canonical_tasks: list[str], task_str: str) -> int | None:
    t = normalize_task(task_str)
    for i, ct in enumerate(canonical_tasks):
        if normalize_task(ct) == t:
            return i
    return None

# ---------------------------------------------------------------------
# Greedy 1:1 matching
# ---------------------------------------------------------------------
def greedy_match(user_tasks: List[str], model_tasks: List[str], sim_threshold: float = 0.55) -> Dict[str, Any]:
    """
    return:
      pairs: list of (ui, mj, sim)
      matched_user: set(ui)
      matched_model: set(mj)
      sim_matrix: np array [U, M]
    """
    U, M = len(user_tasks), len(model_tasks)
    sim_matrix = np.zeros((U, M), dtype=float)

    for i in range(U):
        for j in range(M):
            sim_matrix[i, j] = blended_sim(user_tasks[i], model_tasks[j])

    # Sort all candidates by similarity and perform greedy 1:1 matching.
    candidates = []
    for i in range(U):
        for j in range(M):
            if sim_matrix[i, j] >= sim_threshold:
                candidates.append((sim_matrix[i, j], i, j))
    candidates.sort(reverse=True, key=lambda x: x[0])

    matched_user, matched_model = set(), set()
    pairs = []
    for sim, i, j in candidates:
        if i in matched_user or j in matched_model:
            continue
        matched_user.add(i)
        matched_model.add(j)
        pairs.append((i, j, float(sim)))

    return {
        "pairs": pairs,
        "matched_user": matched_user,
        "matched_model": matched_model,
        "sim_matrix": sim_matrix,
    }

def lex_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_anchors(user_tasks, model_tasks, sim_um, strong_exact=0.90, lex_exact=0.85, debug=False):
    # sim_um has shape (U, M)
    U, M = len(user_tasks), len(model_tasks)

    # Best model per user, and best user per model
    user_best = [(int(sim_um[i].argmax()), float(sim_um[i].max())) for i in range(U)]
    model_best = [(int(sim_um[:, j].argmax()), float(sim_um[:, j].max())) for j in range(M)]

    anchors = []  # list of (u_idx, m_idx, semantic_similarity, lexical_similarity)
    anchored_user = set()
    anchored_model = set()

    for u in range(U):
        m, s = user_best[u]
        # Keep only mutual-best candidates
        u2, s2 = model_best[m]
        if u2 != u:
            continue

        l = lex_sim(user_tasks[u], model_tasks[m])

        if s >= strong_exact and l >= lex_exact:
            anchors.append((u, m, s, l))
            anchored_user.add(u)
            anchored_model.add(m)

    return anchors, anchored_user, anchored_model

def detect_merge_split(
    user_tasks,
    model_tasks,
    strong_exact=0.90,
    merge_user_min=0.40,
    merge_combo_min=0.55,
    merge_top2_gap=0.10,
    split_model_min=0.55,
    split_pair_min=0.60,
    prefer_adjacent=True,
    max_split_candidates=8,
):
    """
    MERGE (2 user -> 1 model)
    SPLIT (1 user -> 2+ model)   e.g., "boil water" -> ["start boiling water", "finish boiling water"]

    Notes:
    - Strong 1:1 anchor matches are excluded from merge/split candidates.
    - If prefer_adjacent=True, contiguous model-task groups are preferred for split detection.
    """

    # --- Similarity matrix (U x M) ---
    user_embs = sbert.encode(user_tasks, convert_to_tensor=True)
    model_embs = sbert.encode(model_tasks, convert_to_tensor=True)
    sim_um = util.cos_sim(user_embs, model_embs).cpu().numpy()  # Shape: (U, M)

    # --- Strong 1:1 anchor matches ---
    anchors, anchored_user, anchored_model = find_anchors(
        user_tasks,
        model_tasks,
        sim_um,
        strong_exact=strong_exact,
        lex_exact=0.85,
        debug=False,
    )

    merged = []
    split = []

    # Prevent the same task from being reused excessively across merge/split detections.
    used_users = set()
    used_models = set()

    # ======================
    # MERGE (model -> top two user tasks)
    # ======================
    for m_idx, m_task in enumerate(model_tasks):
        if m_idx in anchored_model:
            continue
        if m_idx in used_models:
            continue

        candidates = [
            (u_idx, float(sim_um[u_idx, m_idx]))
            for u_idx in range(len(user_tasks))
            if (u_idx not in anchored_user) and (u_idx not in used_users)
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        if len(candidates) < 2:
            continue

        (u1, s1), (u2, s2) = candidates[0], candidates[1]
        s3 = candidates[2][1] if len(candidates) >= 3 else 0.0

        if s2 < merge_user_min:
            continue
        if (s2 - s3) < merge_top2_gap:
            continue

        combo_text = f"{user_tasks[u1]} then {user_tasks[u2]}"
        combo_emb = sbert.encode(combo_text, convert_to_tensor=True)
        combo_sim = float(util.cos_sim(combo_emb, model_embs[m_idx]))

        if combo_sim < merge_combo_min:
            continue
        
        # Sort the user tasks by original order for readability.
        if u1 > u2:
            u1, u2 = u2, u1
            s1, s2 = s2, s1

        merged.append({
            "model": m_task,
            "model_index": m_idx,
            "user": [user_tasks[u1], user_tasks[u2]],
            "user_indices": [u1, u2]
        })

        used_models.add(m_idx)
        used_users.add(u1)
        used_users.add(u2)

    # ======================
    # SPLIT (user -> two or more model tasks)
    # ======================
    for u_idx, u_task in enumerate(user_tasks):
        if u_idx in anchored_user:
            continue
        if u_idx in used_users:
            continue

        # Split candidates: model tasks above the threshold,
        # excluding anchored or already used model tasks.
        cand = [
            (m_idx, float(sim_um[u_idx, m_idx]))
            for m_idx in range(len(model_tasks))
            if (m_idx not in anchored_model) and (m_idx not in used_models) and (sim_um[u_idx, m_idx] >= split_model_min)
        ]

        if len(cand) < 2:
            continue

        # Limit the number of candidates to avoid combinatorial explosion.
        cand.sort(key=lambda x: x[1], reverse=True)
        cand = cand[:max_split_candidates]

        # Sort by model index and search for contiguous groups.
        cand_by_idx = sorted(cand, key=lambda x: x[0])

        # Build contiguous groups
        groups = []
        cur = [cand_by_idx[0]]
        for (m_i, s_i) in cand_by_idx[1:]:
            prev_m, _ = cur[-1]
            if m_i == prev_m + 1:
                cur.append((m_i, s_i))
            else:
                groups.append(cur)
                cur = [(m_i, s_i)]
        groups.append(cur)

        best_choice = None  # (score, model_indices, pair_sims, combo_sim, adjacent)

        # 1) Evaluate contiguous groups first
        for g in groups:
            if len(g) < 2:
                continue

            m_indices = [m for m, _ in g]
            sims = [s for _, s in g]

            combo_text = " then ".join([model_tasks[m] for m in m_indices])
            combo_emb = sbert.encode(combo_text, convert_to_tensor=True)
            combo_sim = float(util.cos_sim(combo_emb, user_embs[u_idx]))

            if combo_sim < split_pair_min:
                continue

            # Score is primarily based on combo similarity, with a small bonus for longer groups.
            score = combo_sim + 0.01 * (len(g) - 2)
            adjacent = True  # The group itself is contiguous

            if (best_choice is None) or (score > best_choice[0]):
                best_choice = (score, m_indices, sims, combo_sim, adjacent)

        # 2) If contiguous groups are unavailable and prefer_adjacent is False,
        #    optionally evaluate non-contiguous fallback pairs.
        if best_choice is None and (not prefer_adjacent):
            # Evaluate all remaining pairs and keep the best one.
            cand_pairs = []
            for i in range(len(cand)):
                for j in range(i + 1, len(cand)):
                    (m1, s1) = cand[i]
                    (m2, s2) = cand[j]
                    m_indices = sorted([m1, m2])
                    sims = [s1, s2]

                    combo_text = " then ".join([model_tasks[m] for m in m_indices])
                    combo_emb = sbert.encode(combo_text, convert_to_tensor=True)
                    combo_sim = float(util.cos_sim(combo_emb, user_embs[u_idx]))

                    if combo_sim >= split_pair_min:
                        adjacent = (m_indices[1] == m_indices[0] + 1)
                        score = combo_sim + (0.02 if adjacent else 0.0)
                        cand_pairs.append((score, m_indices, sims, combo_sim, adjacent))

            if cand_pairs:
                cand_pairs.sort(key=lambda x: x[0], reverse=True)
                best_choice = cand_pairs[0]

        # 3) If prefer_adjacent is True and no contiguous group was found,
        #    skip the split detection under the current policy.
        if best_choice is None:
            continue

        _, m_indices, sims, combo_sim, adjacent = best_choice

        split.append({
            "user": u_task,
            "user_index": u_idx,
            "model": [model_tasks[m] for m in m_indices],
            "model_indices": m_indices
        })

        # Do not reuse model tasks already assigned to a split detection.
        used_users.add(u_idx)
        for m in m_indices:
            used_models.add(m)

    return merged, split

# ---------------------------------------------------------------------
# Pair-based order analysis using precedence edges
# ---------------------------------------------------------------------

# CPEE namespace for <call>, <choose>, and <parallel> (matching the XML injection scripts)
CPEE_NS = {"c": "http://cpee.org/ns/description/1.0"}

def cq(tag: str) -> str:
    return f"{{{CPEE_NS['c']}}}{tag}"

def _is_cpee_task_call(call_el: ET.Element) -> bool:
    t = call_el.find("./c:parameters/c:type", CPEE_NS)
    return (t is not None and (t.text or "").strip() == ":task")

def _get_cpee_label(call_el: ET.Element) -> str:
    lab = call_el.find("./c:parameters/c:label", CPEE_NS)
    return (lab.text or "").strip().lower() if lab is not None else ""

FLOW_CONTAINER_TAGS = {"description", "alternative", "parallel_branch"}
# choose and parallel are treated as gateway structures

class CFResult:
    def __init__(self, start, end, edges):
        self.start = start      # set[str]
        self.end = end          # set[str]
        self.edges = edges      # set[tuple[str, str]]

def _build_precedence_from_node(node: ET.Element) -> CFResult:
    local = node.tag.split("}")[-1]

    # Task call
    if node.tag == cq("call") and _is_cpee_task_call(node):
        lbl = _get_cpee_label(node)
        if not lbl:
            return CFResult(set(), set(), set())
        return CFResult({lbl}, {lbl}, set())

    # Sequential containers
    if local in FLOW_CONTAINER_TAGS:
        child_results = []
        for ch in list(node):
            r = _build_precedence_from_node(ch)
            if r.start or r.end or r.edges:
                child_results.append(r)

        if not child_results:
            return CFResult(set(), set(), set())

        edges = set()
        for r in child_results:
            edges |= r.edges

        # Connect the end nodes of the previous child to the start nodes of the next child.
        for i in range(len(child_results) - 1):
            prev, nxt = child_results[i], child_results[i + 1]
            for a in prev.end:
                for b in nxt.start:
                    edges.add((a, b))

        return CFResult(child_results[0].start, child_results[-1].end, edges)

    # XOR choose
    if local == "choose":
        alts = node.findall("./c:alternative", CPEE_NS)
        branch_results = []
        for alt in alts:
            r = _build_precedence_from_node(alt)
            if r.start or r.end or r.edges:
                branch_results.append(r)

        if not branch_results:
            return CFResult(set(), set(), set())

        edges = set()
        for r in branch_results:
            edges |= r.edges

        start = set().union(*[r.start for r in branch_results])
        end   = set().union(*[r.end   for r in branch_results])
        return CFResult(start, end, edges)

    # AND parallel
    if local == "parallel":
        brs = node.findall("./c:parallel_branch", CPEE_NS)
        branch_results = []
        for br in brs:
            r = _build_precedence_from_node(br)
            if r.start or r.end or r.edges:
                branch_results.append(r)

        if not branch_results:
            return CFResult(set(), set(), set())

        edges = set()
        for r in branch_results:
            edges |= r.edges

        start = set().union(*[r.start for r in branch_results])
        end   = set().union(*[r.end   for r in branch_results])
        return CFResult(start, end, edges)

    # Fallback: treat unknown nodes as sequential containers.
    child_results = []
    for ch in list(node):
        r = _build_precedence_from_node(ch)
        if r.start or r.end or r.edges:
            child_results.append(r)

    if not child_results:
        return CFResult(set(), set(), set())

    edges = set()
    for r in child_results:
        edges |= r.edges
    for i in range(len(child_results) - 1):
        prev, nxt = child_results[i], child_results[i + 1]
        for a in prev.end:
            for b in nxt.start:
                edges.add((a, b))

    return CFResult(child_results[0].start, child_results[-1].end, edges)

def extract_precedence_edges_from_cpee_xml(xml_path: str) -> set[tuple[str, str]]:
    """
    Returns precedence edges between TASK LABELS (lowercased).
    Example:
      a->XOR->(b|c)->join->d  => (a,b),(a,c),(b,d),(c,d)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    res = _build_precedence_from_node(root)
    return {(a, b) for (a, b) in res.edges if a and b and a != b}


SEQ_PARENTS = {"description", "alternative", "parallel_branch"}  # Matches the structure used by the injection scripts.

def _local(tag: str) -> str:
    return tag.split("}")[-1]

def _iter_seq_parents(root: ET.Element):
    for p in root.iter():
        if _local(p.tag) in SEQ_PARENTS:
            yield p

def _is_task_call_cpee(el: ET.Element) -> bool:
    return el.tag == cq("call") and _is_cpee_task_call(el)

def _label_of_call(el: ET.Element) -> str:
    return _get_cpee_label(el)  # Already lowercased

def _swap_two_children(parent: ET.Element, ia: int, ib: int):
    kids = list(parent)
    kids[ia], kids[ib] = kids[ib], kids[ia]
    parent[:] = kids

def infer_best_adjacent_swap_pair_file(
    xml_path: str,
    expected_model_edges: set[tuple[str,str]],
    model_edges_lbl: set[tuple[str,str]],
    max_pairs_per_parent: int = 400,
) -> dict | None:

    tree = ET.parse(xml_path)
    root = tree.getroot()

    base_score = _mismatch_score(expected_model_edges, model_edges_lbl)
    best = None
    best_improvement = None

    for parent in _iter_seq_parents(root):
        kids = list(parent)
        call_idxs = [i for i, ch in enumerate(kids) if _is_task_call_cpee(ch)]
        if len(call_idxs) < 2:
            continue

        pairs = []
        for k in range(len(call_idxs) - 1):
            ia, ib = call_idxs[k], call_idxs[k+1]
            pairs.append((ia, ib))

        for ia, ib in pairs:
            tree2 = ET.ElementTree(deepcopy(root))
            root2 = tree2.getroot()

            target_parent = None
            parent_sig = [(_local(ch.tag), _label_of_call(ch) if _is_task_call_cpee(ch) else "") for ch in list(parent)]
            for p2 in _iter_seq_parents(root2):
                sig2 = [(_local(ch.tag), _label_of_call(ch) if _is_task_call_cpee(ch) else "") for ch in list(p2)]
                if sig2 == parent_sig:
                    target_parent = p2
                    break
            if target_parent is None:
                continue

            _swap_two_children(target_parent, ia, ib)

            with tempfile.NamedTemporaryFile(suffix=".xml", delete=True) as tmp:
                tree2.write(tmp.name, encoding="utf-8", xml_declaration=True)
                swapped_edges = extract_precedence_edges_from_cpee_xml(tmp.name)
                swapped_edges = {(normalize_task(a), normalize_task(b)) for (a,b) in swapped_edges}

            after_score = _mismatch_score(expected_model_edges, swapped_edges)
            improvement = base_score - after_score
            if improvement <= 0:
                continue

            la = _label_of_call(list(parent)[ia])
            lb = _label_of_call(list(parent)[ib])

            cand = {
                "task_a": la,
                "task_b": lb,
            }
            if (best is None) or (best_improvement is None) or (improvement > best_improvement):
                best = cand
                best_improvement = improvement

    return best

def _mismatch_score(expected_edges: set[tuple[str,str]], model_edges: set[tuple[str,str]]) -> int:
    # Simple mismatch score: missing + extra edges
    missing = len(expected_edges - model_edges)
    extra   = len(model_edges - expected_edges)
    return missing + extra

def infer_best_random_swap_pair_file(
    xml_path: str,
    expected_model_edges: set[tuple[str,str]],
    model_edges_lbl: set[tuple[str,str]],
    max_pairs_per_parent: int = 400,
) -> dict | None:

    tree = ET.parse(xml_path)
    root = tree.getroot()

    base_score = _mismatch_score(expected_model_edges, model_edges_lbl)
    best = None
    best_improvement = None

    for parent in _iter_seq_parents(root):
        kids = list(parent)
        call_idxs = [i for i, ch in enumerate(kids) if _is_task_call_cpee(ch)]
        if len(call_idxs) < 3:
            continue

        pairs = []
        for ai in range(len(call_idxs)):
            for bi in range(ai + 1, len(call_idxs)):
                ia, ib = call_idxs[ai], call_idxs[bi]
                if abs(ia - ib) == 1:
                    continue
                pairs.append((ia, ib))
        if len(pairs) > max_pairs_per_parent:
            pairs = pairs[:max_pairs_per_parent]

        for ia, ib in pairs:
            tree2 = ET.ElementTree(deepcopy(root))
            root2 = tree2.getroot()

            target_parent = None
            parent_sig = [(_local(ch.tag), _label_of_call(ch) if _is_task_call_cpee(ch) else "") for ch in list(parent)]

            for p2 in _iter_seq_parents(root2):
                sig2 = [(_local(ch.tag), _label_of_call(ch) if _is_task_call_cpee(ch) else "") for ch in list(p2)]
                if sig2 == parent_sig:
                    target_parent = p2
                    break
            if target_parent is None:
                continue

            _swap_two_children(target_parent, ia, ib)

            with tempfile.NamedTemporaryFile(suffix=".xml", delete=True) as tmp:
                tree2.write(tmp.name, encoding="utf-8", xml_declaration=True)
                swapped_edges = extract_precedence_edges_from_cpee_xml(tmp.name)
                swapped_edges = {(normalize_task(a), normalize_task(b)) for (a,b) in swapped_edges}

            after_score = _mismatch_score(expected_model_edges, swapped_edges)
            improvement = base_score - after_score

            if improvement <= 0:
                continue

            la = _label_of_call(list(parent)[ia])
            lb = _label_of_call(list(parent)[ib])

            cand = {
                "task_a": la,
                "task_b": lb,
            }
            if (best is None) or (best_improvement is None) or (improvement > best_improvement):
                best = cand
                best_improvement = improvement

    return best

# ---------------------------------------------------------------------
# Gateway logic analysis
# ---------------------------------------------------------------------
def generate_text_from_bpmn(bpmn_path: str, llm: str = "gpt-4o", timeout_sec: int = 60) -> tuple[str, str]:
    url = "https://autobpmn.ai/llm/text/llm/"  
    try:
        with open(bpmn_path, "rb") as f:
            files = {
                "rpst_xml": (os.path.basename(bpmn_path), f, "text/xml"),
                "llm": (None, llm),  
            }

            headers = {
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (autobpmn-client)", 
            }

            resp = requests.post(url, files=files, headers=headers, timeout=timeout_sec)

        if resp.status_code != 200:
            note = (
                f"AutoBPMN API failed: status={resp.status_code}, "
                f"content_type={resp.headers.get('Content-Type')}, "
                f"server={resp.headers.get('Server')}, "
                f"body_head={(resp.text or '')[:500]!r}"
            )
            return "", note

        payload = resp.json()
        out = (payload.get("output_text", "") or "").strip()
        if not out:
            return "", "AutoBPMN API 200 but returned empty output_text"
        return out, "AutoBPMN API OK"

    except Exception as e:
        return "", f"AutoBPMN API call failed: {e}"


@lru_cache(maxsize=4096)
def _lemma_token_set(text: str) -> set:
    doc = nlp(text.lower())
    return {
        t.lemma_
        for t in doc
        if t.is_alpha and not t.is_stop
    }

PARTICLES = {"up", "down", "off", "out", "in", "on", "over"}

def task_in_sentence(task: str, sent: str, min_cover: float = 0.80) -> bool:
    # Fast substring check
    t_norm = normalize_task(task)
    s_norm = normalize_task(sent)
    if t_norm and t_norm in s_norm:
        return True

    # Lemma sets
    t_lem = _lemma_token_set(task)
    s_lem = _lemma_token_set(sent)
    if not t_lem or not s_lem:
        return False

    # Ignore particles during matching because they are often omitted in passive constructions.
    t_lem2 = {x for x in t_lem if x not in PARTICLES}

    # Coverage: how much of the task lemma set is present in the sentence
    inter = t_lem2 & s_lem
    cover = len(inter) / max(1, len(t_lem2))

    return cover >= min_cover

def align_user_output_tasks(user_tasks, out_tasks, sim_threshold=0.50):
    res = greedy_match(user_tasks, out_tasks, sim_threshold=sim_threshold)

    out_to_user = {}
    user_to_out = {}

    for (ui, oj, sim) in res["pairs"]:
        out_to_user[oj] = ui
        user_to_out[ui] = oj

    transitive_single_verbs = {"prepare", "use", "pour", "boil", "set"}
    for oj, ot in enumerate(out_tasks):
        if oj in out_to_user:
            continue
        toks = ot.split()
        if len(toks) != 1:
            continue
        v = toks[0]
        if v not in transitive_single_verbs:
            continue

        cand = [ui for ui, ut in enumerate(user_tasks) if ut.startswith(v + " ")]
        if len(cand) == 1 and cand[0] not in user_to_out:
            out_to_user[oj] = cand[0]
            user_to_out[cand[0]] = oj

    return {"out_to_user": out_to_user, "user_to_out": user_to_out}

PARALLEL_CUES = re.compile(r"\b(simultaneously|in parallel|at the same time|concurrently)\b", re.I)
EXCLUSIVE_CUES = re.compile(r"\b(either\s+.+\s+or|either|or\b|alternatively|instead)\b", re.I)
OR_SENT_START = re.compile(r"^\s*(or|alternatively|otherwise)\b", re.I)

JOIN_AND_CUES = re.compile(r"\b(after|once)\s+both\b|\bboth\s+branches\b", re.I)
JOIN_XOR_CUES = re.compile(r"\b(after|once)\s+either\b|\bafter\s+either\b", re.I)

# Cues tailored to output verbalization
OUT_PARALLEL_VERB = re.compile(r"\bsplits?\s+into\s+two\s+parallel\b|\bparallel\s+activities\b|\bboth\s+branches\b", re.I)
OUT_BRANCH_1 = re.compile(r"\bin\s+the\s+first\s+branch\b", re.I)
OUT_BRANCH_2 = re.compile(r"\bin\s+the\s+second\s+branch\b", re.I)
OUT_XOR_VERB = re.compile(r"\bbased\s+on\s+a\s+condition\b|\beither\b.+\bor\b", re.I)

def make_tid_user(i: int) -> str:
    return f"U{i}"

def make_tid_out(j: int) -> str:
    return f"O{j}"

def find_tasks_in_sentence(tasks: List[str], sent: str) -> List[int]:
    sent_norm = normalize_task(sent)

    # 1) Collect candidate hits and compute a simple score
    cand = []
    for i, t in enumerate(tasks):
        t_norm = normalize_task(t)
        if not t_norm:
            continue

        # Fast substring check
        if t_norm in sent_norm:
            score = 1.0 + 0.01 * len(t_norm)   # Small bonus for longer matches
            cand.append((i, score))
            continue
        
        # Lemma coverage
        if task_in_sentence(t, sent, min_cover=0.60):
            score = 0.9 + 0.001 * len(t_norm)
            cand.append((i, score))
            continue

        # Token-level Jaccard similarity
        jt = jaccard_sim(t_norm, sent_norm)
        if jt >= 0.30:
            score = jt + 0.001 * len(t_norm)
            cand.append((i, score))

    if not cand:
        return []

    # 2) Sort by descending score
    cand.sort(key=lambda x: x[1], reverse=True)

    # 3) Remove subset/substring duplicates when a longer task already covers them.
    picked = []
    picked_token_sets = []
    for idx, _score in cand:
        t_norm = normalize_task(tasks[idx])
        t_set = set(re.findall(r"[a-z0-9]+", t_norm))

        dominated = False
        for j, t_set2 in enumerate(picked_token_sets):
            t_norm2 = normalize_task(tasks[picked[j]])
            # A shorter task is dominated when its tokens are a subset and its text is a substring of a longer task.
            if t_set and t_set.issubset(t_set2) and t_norm in t_norm2:
                dominated = True
                break

        if not dominated:
            picked.append(idx)
            picked_token_sets.append(t_set)

    # 4) Sort by textual order within the sentence
    picked.sort(
        key=lambda i: sent_norm.find(normalize_task(tasks[i]))
        if normalize_task(tasks[i]) in sent_norm else 10**9
    )
    return picked

def extract_gateway_relations(
    text: str,
    tasks: List[str],
    side: str,
    out_to_user: Dict[int, int] = None,  # Used to map output-side task indices into shared IDs
) -> Dict[str, Any]:
    """
    side: "user" or "output"
    out_to_user: maps an output-task index to a user-task index after alignment

    return:
      {
        "relations": [ ... ],
        "join_relations": [ ... ],
        "evidence": [ ... ]  # Sentence-level evidence for each relation
      }
    """
    out_to_user = out_to_user or {}
    sents = _sentences(text)

    relations = []      # Pairwise relations: [("PARALLEL"/"EXCLUSIVE"/"SEQUENTIAL", tidA, tidB), ...]
    join_relations = [] # Set-based relations: [("JOIN_AND"/"JOIN_XOR", [tid...]), ...]
    evidence = []

    def tid_of_task_idx(idx: int) -> str:
        if side == "user":
            return make_tid_user(idx)
        # On the output side, lift matched tasks into user-level task IDs when possible.
        if idx in out_to_user:
            return make_tid_user(out_to_user[idx])
        return make_tid_out(idx)

    # ---- Branch parsing for output text only ----
    if side == "output":
        lower = text.lower()
        if OUT_BRANCH_1.search(lower) and OUT_BRANCH_2.search(lower):

            # First determine whether the branch structure is parallel (AND) or exclusive (XOR).
            is_xor = bool(OUT_XOR_VERB.search(lower)) or bool(EXCLUSIVE_CUES.search(lower))
            is_parallel = bool(OUT_PARALLEL_VERB.search(lower)) or bool(PARALLEL_CUES.search(lower))

            # If XOR and parallel cues both occur, the current policy prefers EXCLUSIVE unless parallel evidence is stronger.
            branch_relation_type = "EXCLUSIVE" if (is_xor and not is_parallel) else "PARALLEL"

            parts = re.split(r"(?i)\bin\s+the\s+second\s+branch\b", text, maxsplit=1)
            first_part = parts[0]
            second_part = parts[1] if len(parts) > 1 else ""
            first_part = re.split(r"(?i)\bin\s+the\s+first\s+branch\b", first_part, maxsplit=1)[-1]

            first_hits, second_hits = set(), set()

            def _clauses(s: str) -> list[str]:
                # Reuse the clause splitting logic here as well.
                parts = re.split(r",|\band\b", s, flags=re.I)
                return [p.strip() for p in parts if p.strip()]

            for sent in _sentences(first_part):
                for cl in _clauses(sent):
                    for idx in find_tasks_in_sentence(tasks, cl):
                        first_hits.add(idx)

            for sent in _sentences(second_part):
                for cl in _clauses(sent):
                    for idx in find_tasks_in_sentence(tasks, cl):
                        second_hits.add(idx)

            # Emit PARALLEL or EXCLUSIVE relations between the two branches.
            for a in first_hits:
                for b in second_hits:
                    if a == b:
                        continue
                    ta, tb = tid_of_task_idx(a), tid_of_task_idx(b)
                    x, y = sorted([ta, tb])
                    relations.append((branch_relation_type, x, y))
                    evidence.append({
                        "type": branch_relation_type,
                        "tasks": [x, y],
                        "evidence": f"output: branch structure (first/second) + cue={'XOR' if branch_relation_type=='EXCLUSIVE' else 'PARALLEL'}"
                    })

            # Add consistent join relations as well.
            tids = sorted(set([tid_of_task_idx(i) for i in (first_hits | second_hits)]))
            if len(tids) >= 2:
                if branch_relation_type == "PARALLEL":
                    join_relations.append(("JOIN_AND", tids))
                else:
                    join_relations.append(("JOIN_XOR", tids))

    # ---- Sentence-level cues shared by user and output text ----
    prev_tids_unique: List[str] = []
    prev_sent: str | None = None

    for sent in sents:
        hits = find_tasks_in_sentence(tasks, sent)

        # If the sentence contains no task, do not update the previous-sentence state.
        if not hits:
            continue

        tids = [tid_of_task_idx(i) for i in hits]
        tids_unique = []
        for t in tids:
            if t not in tids_unique:
                tids_unique.append(t)

        sent_l = sent.lower()
        is_parallel = bool(PARALLEL_CUES.search(sent_l)) or (side == "output" and bool(OUT_PARALLEL_VERB.search(sent_l)))
        is_excl = bool(EXCLUSIVE_CUES.search(sent_l)) or (side == "output" and bool(OUT_XOR_VERB.search(sent_l)))

        # (A) Intra-sentence relations: only when at least two tasks are present
        if len(tids_unique) >= 2:
            if JOIN_AND_CUES.search(sent_l):
                join_relations.append(("JOIN_AND", sorted(tids_unique)))
                evidence.append({"type": "JOIN_AND", "tasks": sorted(tids_unique), "evidence": sent})

            if JOIN_XOR_CUES.search(sent_l):
                join_relations.append(("JOIN_XOR", sorted(tids_unique)))
                evidence.append({"type": "JOIN_XOR", "tasks": sorted(tids_unique), "evidence": sent})

            if is_parallel:
                for i in range(len(tids_unique)):
                    for j in range(i + 1, len(tids_unique)):
                        a, b = sorted([tids_unique[i], tids_unique[j]])
                        relations.append(("PARALLEL", a, b))
                        evidence.append({"type": "PARALLEL", "tasks": [a, b], "evidence": sent})

            elif is_excl:
                for i in range(len(tids_unique)):
                    for j in range(i + 1, len(tids_unique)):
                        a, b = sorted([tids_unique[i], tids_unique[j]])
                        relations.append(("EXCLUSIVE", a, b))
                        evidence.append({"type": "EXCLUSIVE", "tasks": [a, b], "evidence": sent})

            else:
                # Without stronger cues, extract only weak sequential relations based on local text order.
                for i in range(len(tids_unique) - 1):
                    a, b = tids_unique[i], tids_unique[i + 1]
                    if a != b:
                        relations.append(("SEQUENTIAL", a, b))
                        evidence.append({"type": "SEQUENTIAL", "tasks": [a, b], "evidence": sent})

        # (B) Cross-sentence EXCLUSIVE relation: e.g. "A. Or B."
        if is_excl and prev_sent is not None and prev_tids_unique:
            cur_rep = tids_unique[0]

            # If the sentence starts with "Or", connect it to the first task of the previous sentence.
            if OR_SENT_START.search(sent):
                prev_rep = prev_tids_unique[0]
            else:
                prev_rep = prev_tids_unique[-1]  # Default behavior

            if prev_rep != cur_rep:
                x, y = sorted([prev_rep, cur_rep])
                relations.append(("EXCLUSIVE", x, y))
                evidence.append({
                    "type": "EXCLUSIVE",
                    "tasks": [x, y],
                    "evidence": f"(prev) {prev_sent}  ||  (cur) {sent}"
                })
        
        # (C) Default cross-sentence SEQUENTIAL relation, e.g. "A. Then B."
        # Apply only when no EXCLUSIVE or PARALLEL cue is present.
        if (not is_excl) and (not is_parallel) and prev_sent is not None and prev_tids_unique:
            prev_last = prev_tids_unique[-1]
            cur_first = tids_unique[0]
            if prev_last != cur_first:
                relations.append(("SEQUENTIAL", prev_last, cur_first))
                evidence.append({
                    "type": "SEQUENTIAL",
                    "tasks": [prev_last, cur_first],
                    "evidence": f"(prev) {prev_sent}  ||  (cur) {sent}"
                })

        # Always update the previous-sentence state when the current sentence contains at least one task.
        prev_tids_unique = tids_unique
        prev_sent = sent

    # Remove duplicates once after the loop
    rel_set = list(dict.fromkeys(relations))

    join_set = []
    seen_join = set()
    for jt_type, jt_list in join_relations:
        # Keep the JSON output list-based; use a tuple key only for deduplication.
        key = (jt_type, tuple(sorted(jt_list)))
        if key in seen_join:
            continue
        seen_join.add(key)
        join_set.append((jt_type, sorted(jt_list)))

    return {"relations": rel_set, "join_relations": join_set, "evidence": evidence}


# --- Cues ---
SIMUL_RE = re.compile(r"\b(simultaneously|in parallel|at the same time|concurrently)\b", re.I)
BOTH_RE  = re.compile(r"\b(after|once)\s+both\b|\bboth\s+.*completed\b|\bboth\s+branches\b", re.I)
FIRST_ACT_RE = re.compile(r"\bfirst\s+activity\b", re.I)
SECOND_ACT_RE = re.compile(r"\bsecond\s+activity\b", re.I)

def _types_for_pair_from_rel(rel: dict, a: str, b: str) -> list[str]:
    """Extract relation types for the pair (a, b) from rel["relations"]."""
    if rel is None:
        return []
    out = []
    for t, x, y in rel.get("relations", []):
        if sorted([x, y]) == sorted([a, b]):
            out.append(t)
    # Deduplicate while preserving order
    return sorted(list(dict.fromkeys(out)))

def pick_parallel_pair_from_user_rel(user_rel: dict) -> tuple[str, str] | None:
    """Return any PARALLEL pair from user_rel."""
    if user_rel is None:
        return None
    for t, a, b in user_rel.get("relations", []):
        if t == "PARALLEL" and a.startswith("U") and b.startswith("U") and a != b:
            return (a, b)
    return None

def pick_exclusive_pair_from_user_xor(user_text: str, user_tasks: list[str]) -> tuple[str, str] | None:
    """
    Return the first task of each XOR branch as U-indices based on the narrative XOR parser.
    This representative pair is used for xor->seq and xor->and mismatch detection.
    """
    if not user_text:
        return None
    parsed_x = parse_user_xor_and_edges(user_text, extract_tasks_fn=extract_verbal_tasks).get("xor")
    if not parsed_x:
        return None
    brs = parsed_x.get("branches", [])
    if len(brs) < 2:
        return None
    b1 = brs[0].get("tasks", [])
    b2 = brs[1].get("tasks", [])
    if not b1 or not b2:
        return None

    u1 = find_task_index(user_tasks, b1[0])
    u2 = find_task_index(user_tasks, b2[0])
    if u1 is None or u2 is None or u1 == u2:
        return None
    return (f"U{u1}", f"U{u2}")

def pick_parallel_pair_from_user_text(user_text: str, user_tasks: list[str]) -> tuple[str, str] | None:
    """
    robust parallel pair picker:
    1) "first activity ...", "second activity ..." patterns
    2) task pairs inside sentences such as "once both X and Y ..." or "after both ..."
    """
    if not user_text:
        return None

    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", user_text) if s.strip()]
    if not any(SIMUL_RE.search(s) or BOTH_RE.search(s) for s in sents):
        return None

    # (1) "first activity" / "second activity" pattern
    first_sent = next((s for s in sents if FIRST_ACT_RE.search(s)), None)
    second_sent = next((s for s in sents if SECOND_ACT_RE.search(s)), None)
    if first_sent and second_sent:
        t1s = extract_verbal_tasks(first_sent)
        t2s = extract_verbal_tasks(second_sent)
        if t1s and t2s:
            u1 = find_task_index(user_tasks, t1s[0])
            u2 = find_task_index(user_tasks, t2s[0])
            if u1 is not None and u2 is not None and u1 != u2:
                return (f"U{u1}", f"U{u2}")

    # (2) "once/after both ..." sentence: use the first two task hits when available.
    both_sent = next((s for s in sents if BOTH_RE.search(s)), None)
    if both_sent:
        hits = find_tasks_in_sentence(user_tasks, both_sent)
        if len(hits) >= 2:
            return (f"U{hits[0]}", f"U{hits[1]}")

    return None


def compare_gateway_relations(user_rel: Dict[str, Any], out_rel: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare only on the intersection of shared task IDs (U*).
    Mismatch types:
      - branching_semantics_inconsistency (PARALLEL <-> EXCLUSIVE)
      - parallel_to_sequential_inconsistency
      - exclusive_flattened_to_sequence
      - join_semantics_inconsistency (JOIN_AND <-> JOIN_XOR)
    """
    def is_common_tid(tid: str) -> bool:
        return tid.startswith("U")

    # Map task pairs to relation-type sets
    def build_pair_map(rels):
        mp = {}
        for t, a, b in rels:
            if not (is_common_tid(a) and is_common_tid(b)):
                continue
            key = (a, b)
            mp.setdefault(key, set()).add(t)
        return mp

    user_map = build_pair_map(user_rel["relations"])
    out_map  = build_pair_map(out_rel["relations"])

    mismatches = []

    # Compare pair relations over the union of both key sets
    all_keys = set(user_map.keys()) | set(out_map.keys())
    for key in all_keys:
        ut = user_map.get(key, set())
        ot = out_map.get(key, set())

        # AND <-> XOR/OR mismatch
        if ("PARALLEL" in ut and "EXCLUSIVE" in ot) or ("EXCLUSIVE" in ut and "PARALLEL" in ot):
            mismatches.append({
                "mismatch_type": "branching_semantics_inconsistency",
                "task_pair": list(key),
                "user_types": sorted(list(ut)),
                "output_types": sorted(list(ot)),
            })
            continue

        # Parallel -> sequential mismatch
        if "PARALLEL" in ut and "SEQUENTIAL" in ot:
            mismatches.append({
                "mismatch_type": "parallel_to_sequential_inconsistency",
                "task_pair": list(key),
                "user_types": sorted(list(ut)),
                "output_types": sorted(list(ot)),
            })
            continue

        # Exclusive branch flattened into sequence
        if "EXCLUSIVE" in ut and "SEQUENTIAL" in ot:
            mismatches.append({
                "mismatch_type": "exclusive_flattened_to_sequence",
                "task_pair": list(key),
                "user_types": sorted(list(ut)),
                "output_types": sorted(list(ot)),
            })
            continue

    # Compare join relations for the same task set
    def build_join_map(jrels):
        mp = {}
        for jt, s in jrels:
            tids = sorted([x for x in list(s) if is_common_tid(x)])
            if len(tids) < 2:
                continue
            mp[tuple(tids)] = jt
        return mp

    uj = build_join_map(user_rel["join_relations"])
    oj = build_join_map(out_rel["join_relations"])

    for k in set(uj.keys()) | set(oj.keys()):
        if k in uj and k in oj and uj[k] != oj[k]:
            mismatches.append({
                "mismatch_type": "join_semantics_inconsistency",
                "task_set": list(k),
                "user_join": uj[k],
                "output_join": oj[k],
            })

    return {
        "gateway_mismatches": mismatches,
        "mismatch_count": len(mismatches),
    }

# ---------------------------------------------------------------------
# Compare task lists and derive issue categories
# ---------------------------------------------------------------------
def compare_tasks(user_tasks, model_tasks, user_text="",output_text="",bpmn_path=""):
    # -----------------------------------------------------------------
    # 1) Detect merge and split errors first
    # -----------------------------------------------------------------
    merged, split = detect_merge_split(
        user_tasks,
        model_tasks,
        strong_exact=0.90,
        merge_user_min=0.40,
        merge_combo_min=0.55,
        merge_top2_gap=0.10,
        split_model_min=0.55,
        split_pair_min=0.60,
        prefer_adjacent=True,
    )

    # Mark task indices involved in merge/split detections as already explained.
    explained_user = set()
    explained_model = set()

    for m in merged:
        explained_model.add(m["model_index"])
        for ui in m["user_indices"]:
            explained_user.add(ui)

    for s in split:
        explained_user.add(s["user_index"])
        for mj in s["model_indices"]:
            explained_model.add(mj)

    # -----------------------------------------------------------------
    # 2) Run 1:1 matching only on tasks not already explained by merge/split
    # -----------------------------------------------------------------
    # Instead of masking indices inside greedy_match, create filtered sublists
    # and map the matched local indices back to the original task lists.

    remaining_user_idxs = [i for i in range(len(user_tasks)) if i not in explained_user]
    remaining_model_idxs = [j for j in range(len(model_tasks)) if j not in explained_model]

    remaining_user_tasks = [user_tasks[i] for i in remaining_user_idxs]
    remaining_model_tasks = [model_tasks[j] for j in remaining_model_idxs]

    match_res = greedy_match(remaining_user_tasks, remaining_model_tasks, sim_threshold=0.55)

    matched_user = set()
    matched_model = set()

    # Map the local indices returned by greedy_match back to the original indices.
    for u_local in match_res["matched_user"]:
        matched_user.add(remaining_user_idxs[u_local])
    for m_local in match_res["matched_model"]:
        matched_model.add(remaining_model_idxs[m_local])

    # -----------------------------------------------------------------
    # 3) Also treat merge/split indices as matched
    # -----------------------------------------------------------------
    matched_user |= explained_user
    matched_model |= explained_model

    # -----------------------------------------------------------------
    # 4) Compute missing and additional tasks
    # -----------------------------------------------------------------
    missing = [user_tasks[i] for i in range(len(user_tasks)) if i not in matched_user]
    additional = [model_tasks[j] for j in range(len(model_tasks)) if j not in matched_model]
    
    # -----------------------------------------------------------------
    # 5) Order analysis (swap candidates only)
    # -----------------------------------------------------------------
    order_analysis = {
        "note": "",
        "random_sequences_best_pair": None,
        "two_wrong_sequences_best_pair": None
    }

    if not bpmn_path:
        order_analysis["note"] = "Order analysis skipped: bpmn_path not provided."
    else:
        # (A) User-side precedence edges
        parsed = parse_user_xor_and_edges(user_text, extract_tasks_fn=extract_verbal_tasks)
        user_edges_lbl = {
            (normalize_task(a), normalize_task(b))
            for (a, b) in parsed.get("edges", [])
            if a not in {"END", ""} and b not in {"END", ""} and a != b
        }

        # (B) Model-side precedence edges
        model_edges_lbl = extract_precedence_edges_from_cpee_xml(bpmn_path)
        model_edges_lbl = {
            (normalize_task(a), normalize_task(b))
            for (a, b) in model_edges_lbl
            if a and b and a != b
        }

        # (C) Map user tasks into model-task space
        u2m = build_user_to_model_mapping(
            user_tasks,
            model_tasks,
            sim_threshold=0.25,
            return_debug=False,
        )

        # (D) Expected model-space edges
        expected_model_edges, _collapsed = translate_user_edges_to_model_space(user_edges_lbl, u2m)

        # (E) Infer the best swap candidates
        best_rand = infer_best_random_swap_pair_file(
            xml_path=bpmn_path,
            expected_model_edges=expected_model_edges,
            model_edges_lbl=model_edges_lbl,
        )
        best_adj = infer_best_adjacent_swap_pair_file(
            xml_path=bpmn_path,
            expected_model_edges=expected_model_edges,
            model_edges_lbl=model_edges_lbl,
        )

        order_analysis["random_sequences_best_pair"] = best_rand
        order_analysis["two_wrong_sequences_best_pair"] = best_adj

    # -----------------------------------------------------------------
    # 6) Gateway analysis (structural + text-based)
    # -----------------------------------------------------------------
    gateway_analysis = {
        "note": "",
        "gateway_mismatches": [],
        "mismatch_count": 0,
        "structural": {}
    }

    if not bpmn_path:
        gateway_analysis["note"] = "Gateway analysis skipped: bpmn_path not provided."
    else:
        # -------------------------------------------------------------
        # (A) Structural signals (counts + user cues)
        # -------------------------------------------------------------
        parallel_cnt = count_parallel_in_cpee(bpmn_path)
        choose_cnt = count_exclusive_choose_in_cpee(bpmn_path)

        user_parallel = user_has_parallel_cue(user_text)
        user_has_xor = (
            parse_user_xor_and_edges(user_text, extract_tasks_fn=extract_verbal_tasks).get("xor") is not None
            if user_text else False
        )

        gateway_analysis["structural"].update({
            "model_parallel_count": parallel_cnt,
            "model_exclusive_choose_count": choose_cnt
        })

        xor_to_seq_suspected = bool(user_has_xor and choose_cnt == 0)
        and_to_xor_suspected = bool(user_parallel and choose_cnt > 0 and parallel_cnt == 0)

        gateway_analysis["structural"]["xor_to_seq_suspected"] = xor_to_seq_suspected
        gateway_analysis["structural"]["and_to_xor_suspected"] = and_to_xor_suspected

        # Keep a structural-only mismatch entry when useful for debugging.
        if xor_to_seq_suspected:
            gateway_analysis["gateway_mismatches"].append({
                "mismatch_type": "xor_to_seq_suspected_structural",
                "detail": "User text contains XOR cue, but BPMN has no <choose mode='exclusive'>."
            })

        # -------------------------------------------------------------
        # (B) Text-based gateway relations (user_rel / out_rel)
        # -------------------------------------------------------------
        user_rel = None
        out_rel = None

        if user_text:
            canonical = user_tasks
            user_rel = extract_gateway_relations(text=user_text, tasks=canonical, side="user")

            if output_text:
                out_tasks = extract_verbal_tasks(output_text)
                out_tasks = [t for t in out_tasks if not is_output_meta_task(t)]

                align = align_user_output_tasks(user_tasks, out_tasks, sim_threshold=0.50)
                out_to_user = align["out_to_user"]

                out_rel = extract_gateway_relations(
                    text=output_text,
                    tasks=out_tasks,
                    side="output",
                    out_to_user=out_to_user,
                )

                gateway_cmp = compare_gateway_relations(user_rel, out_rel)
                gateway_analysis["gateway_mismatches"].extend(gateway_cmp["gateway_mismatches"])
                gateway_analysis["note"] = "Gateway analysis computed (structural + AutoBPMN output_text)."
            else:
                gateway_analysis["note"] = "Gateway analysis ON, but output_text is empty (AutoBPMN API failed)."
        else:
            gateway_analysis["note"] = "Gateway analysis ON, but user_text is empty."

        # -------------------------
        # (C) Robust pair-style gateway mismatches (structural fallback)
        # -------------------------
        # Helper: append one mismatch safely
        def _append_pair_mismatch(mismatch_type: str, pair: tuple[str, str], user_types: list[str], out_types_base: list[str], detail: str):
            a, b = sorted(pair)

            out_types = list(out_types_base)

            # Merge relation types inferred from output_text when available.
            if out_rel is not None:
                out_types += _types_for_pair_from_rel(out_rel, a, b)

            # Deduplicate
            out_types = sorted(list(set(out_types)))

            already = any(
                (m.get("mismatch_type") == mismatch_type and m.get("task_pair") == [a, b])
                for m in gateway_analysis["gateway_mismatches"]
            )
            if not already:
                gateway_analysis["gateway_mismatches"].append({
                    "mismatch_type": mismatch_type,
                    "task_pair": [a, b],
                    "user_types": user_types,
                    "output_types": out_types,
                    "detail": detail,
                })

        # ---- AND -> XOR (parallel expected, exclusive in the model) ----
        if and_to_xor_suspected and user_rel is not None:
            pair = pick_parallel_pair_from_user_rel(user_rel) or pick_parallel_pair_from_user_text(user_text, user_tasks)
            if pair is not None:
                _append_pair_mismatch(
                    mismatch_type="parallel_to_exclusive_inconsistency",
                    pair=pair,
                    user_types=["PARALLEL"],
                    out_types_base=["EXCLUSIVE"],  # Structural choose
                    detail="Parallel semantics in the user text are represented as exclusive branching in the model."
                )

        # ---- AND -> SEQ (parallel expected, but the model has neither parallel nor choose) ----
        and_to_seq_suspected = bool(user_parallel and parallel_cnt == 0 and choose_cnt == 0)
        gateway_analysis["structural"]["and_to_seq_suspected"] = and_to_seq_suspected

        if and_to_seq_suspected and user_rel is not None:
            pair = pick_parallel_pair_from_user_rel(user_rel) or pick_parallel_pair_from_user_text(user_text, user_tasks)
            if pair is not None:
                _append_pair_mismatch(
                    mismatch_type="parallel_to_sequential_inconsistency",
                    pair=pair,
                    user_types=["PARALLEL"],
                    out_types_base=["SEQUENTIAL"],  # Structural fallback
                    detail="Parallel semantics in the user text are flattened into sequential flow in the model."
                )

        # ---- XOR -> SEQ (exclusive expected, but the model has no choose) ----
        if xor_to_seq_suspected and user_text:
            pair = pick_exclusive_pair_from_user_xor(user_text, user_tasks)
            if pair is not None:
                _append_pair_mismatch(
                    mismatch_type="exclusive_flattened_to_sequence",
                    pair=pair,
                    user_types=["EXCLUSIVE"],
                    out_types_base=["SEQUENTIAL"],  # Structural fallback
                    detail="Exclusive branching described in the user text is represented as sequential flow in the model."
                )

        # ---- XOR -> AND (exclusive expected, but the model uses parallel) ----
        # Structurally, suspect XOR -> AND when the user text contains XOR cues and the model contains parallel structures.
        xor_to_and_suspected = bool(user_has_xor and parallel_cnt > 0 and choose_cnt == 0)
        # This is a conservative version of the rule.

        # A more aggressive alternative would be:
        # xor_to_and_suspected = bool(user_has_xor and parallel_cnt > 0)
        gateway_analysis["structural"]["xor_to_and_suspected"] = xor_to_and_suspected

        if xor_to_and_suspected and user_text:
            pair = pick_exclusive_pair_from_user_xor(user_text, user_tasks)
            if pair is not None:
                _append_pair_mismatch(
                    mismatch_type="exclusive_to_parallel_inconsistency",
                    pair=pair,
                    user_types=["EXCLUSIVE"],
                    out_types_base=["PARALLEL"],   # Structural parallel
                    detail="Exclusive branching described in the user text is represented as parallel flow in the model."
                )


    # Always refresh the mismatch count
    gateway_analysis["mismatch_count"] = len(gateway_analysis["gateway_mismatches"])
            
    return {
        "issues": {
            "missing_tasks": missing,
            "additional_tasks": additional,
            "merged_tasks": merged,
            "split_tasks": split,
            "order_analysis": order_analysis,
            "gateway_analysis": gateway_analysis,
        }
    }

# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--bpmn", required=True)
    ap.add_argument("--report_root", default=None)
    args = ap.parse_args()

    # By default, read the user text from the provided log file.
    user_text = read_user_text(args.log)
    
    output_text, output_note = generate_text_from_bpmn(args.bpmn, llm="gpt-4o")
    # Generate an LLM-based textual description from the BPMN model for gateway analysis.
#   output_text = generate_text_from_bpmn(args.bpmn, llm="gpt-4o")
    user_tasks = extract_verbal_tasks(user_text)
    model_tasks = parse_bpmn_tasks(args.bpmn)

    report = compare_tasks(
        user_tasks,
        model_tasks,
        user_text,
        output_text,
        bpmn_path=args.bpmn
    )

    if args.report_root:
        report_dir = args.report_root
    else:
        report_dir = os.path.join(os.path.dirname(__file__), "report")

    os.makedirs(report_dir, exist_ok=True)

    base_name = os.path.basename(re.sub(r"\.xml$", ".spacy_report.json", args.bpmn))
    output_path = os.path.join(report_dir, base_name)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "log_path": args.log,
            "bpmn_path": args.bpmn,
            "user_tasks": user_tasks,
            "model_tasks": model_tasks,
            "comparison": report,
            "user_text": user_text,
            "output_text": output_text,
        }, f, ensure_ascii=False, indent=2)

    
if __name__ == "__main__":
    main()