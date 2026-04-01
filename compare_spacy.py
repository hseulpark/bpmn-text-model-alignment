import os
import re
import json
import argparse
import sys
import xml.etree.ElementTree as ET
from annotated_types import doc
import yaml
import spacy
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher

# -------------------------
# spaCy & SBERT 초기화
# -------------------------
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

# -------------------------
# YAML에서 유저 입력 추출
# -------------------------
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

def _is_subphrase_of_existing(phrase: str, existing: list[str]) -> bool:
    """
    phrase가 existing 안의 어떤 태스크의 '부분(토큰 subset)'이면 True.
    예) "use coffee" 는 "use coffee machine"의 subset -> True (추가하면 안 됨)
    """
    p = normalize_task(phrase)
    p_tokens = set(re.findall(r"[a-z0-9]+", p))
    if not p_tokens:
        return False

    for e in existing:
        e_norm = normalize_task(e)
        e_tokens = set(re.findall(r"[a-z0-9]+", e_norm))

        # phrase가 existing의 토큰 subset이고, 완전히 동일한 문장은 아니면(=더 짧으면)
        if p_tokens.issubset(e_tokens) and p != e_norm:
            return True

    return False

# -------------------------
# 유저 입력에서 태스크 추출
# -------------------------

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
    # lemma 기반으로 단어 1개짜리 메타 제거
    toks = re.findall(r"[a-z]+", t)
    if len(toks) == 1 and toks[0] in OUTPUT_META_LEMMAS:
        return True
    return False

def _fallback2_spacy_verb_object(text: str, ordered: list[str]) -> list[str]:
    """
    fallback2 개선:
    - spaCy 토큰 기반으로 VERB(+PRT) + NOUN-phrase 패턴만 잡음
    - 원문 등장 순서대로 ordered에 merge
    """
    doc = nlp(text)

    AUX_LEMMAS = {"be", "have", "do"}
    SKIP_VERBS = AUX_LEMMAS | {"start", "begin", "end", "finish", "terminate", "split", "branch", "complete", "perform", "follow", "please"}
    
    OBJ_DEPS = {"dobj", "pobj", "attr", "nsubjpass"}  # ✅ nsubj 제거

    candidates = []  # (start_char, phrase)

    for t in doc:
        if t.pos_ != "VERB":
            continue
        if t.lemma_.lower() in SKIP_VERBS:
            continue

        # (A) verb(+particle)
        verb_tokens = [t.lemma_.lower()]
        prt = None
        for ch in t.children:
            if ch.dep_ == "prt":  # set up
                prt = ch.text.lower()
                break
        if prt:
            verb_tokens.append(prt)

        # (B) object 찾기: dobj/pobj/attr/nsubjpass 중 하나
        obj = None
        for child in t.children:
            if child.dep_ in OBJ_DEPS and child.pos_ in {"NOUN","PROPN"}:
                obj = child
                break
        if obj is None:
            continue
        # obj로 noun phrase 만들기

        # object noun phrase 확장 (compound/amod 포함)
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

        # 이미 더 긴 태스크에 포함되는 짧은 태스크면 skip
        if _is_subphrase_of_existing(phrase, ordered):
            continue

        candidates.append((t.idx, phrase))

    # 등장 순서대로 정렬 + 중복 제거
    candidates.sort(key=lambda x: x[0])
    out = []
    seen = set()
    for _, p in candidates:
        if p not in seen and p not in ordered:
            out.append(p)
            seen.add(p)
    return out

def extract_verbal_tasks(text: str):
    text = re.sub(r"(?im)^\s*#\s*user\s*input\s*:\s*$", "", text)  # "# User Input:"
    text = re.sub(r"(?im)^\s*user\s*input\s*:\s*$", "", text)     # "User Input:"
    text = re.sub(r"(?im)^\s*#.*$", "", text)                     # any full-line comments
    text = re.sub(r"\n+", "\n", text).strip()
    # 1. 문장 단위로 분리 및 filler 제거
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

    # 2. 각 문장별 spaCy 분석
    for sent_text in cleaned:

        # ✅ NEW: comma / and 로 clause 분리해서 각각 따로 파싱
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

                # 목적어 / 수동태 주어만
                for child in verb.children:
                    if child.dep_ in {"dobj", "pobj", "attr", "nsubjpass"}:   # ✅ nsubj는 빼는 걸 추천
                        noun_tokens = [child.text]
                        for sub in child.children:
                            if sub.dep_ in {"compound", "amod"}:
                                noun_tokens.insert(0, sub.text)
                        phrase_tokens.extend(noun_tokens)

                phrase_str = " ".join(dict.fromkeys(phrase_tokens))
                phrase_str = re.sub(r"[^a-zA-Z0-9\s]", "", phrase_str)
                phrase_str = re.sub(r"\s+", " ", phrase_str).strip().lower()
                if phrase_str.startswith("perform "):
                    phrase_str = phrase_str[len("perform "):].strip()

                ALLOW_SINGLE_VERB_TASKS = {"serve"}  # 필요하면 "wait", "rest" 등 추가

                if len(phrase_str.split()) < 2:
                    # ✅ 단일 동사 task 허용
                    if phrase_str in ALLOW_SINGLE_VERB_TASKS:
                        tasks.append(phrase_str)
                    continue
                if phrase_str.split()[0] not in FILLER_WORDS:
                    tasks.append(phrase_str)


            # fallback 1: VERB + NOUN 또는 compound VERB + NOUN 보정 (정확도 개선)
            for i in range(len(doc) - 1):
                token = doc[i]
                next_token = doc[i + 1]

                # ✅ 일반 VERB + NOUN 구조 (명사구의 head가 바로 다음일 때만)
                if (
                    token.pos_ == "VERB"
                    and next_token.pos_ == "NOUN"
                    and next_token.dep_ in {"dobj", "pobj", "attr", "ROOT"}  # 명사구의 핵심일 때만
                ):
                    phrase = f"{token.text} {next_token.text}".lower()
                    if phrase not in tasks:
                        tasks.append(phrase)

                # ✅ VERB가 compound로 오인된 경우 (boil compound water)
                elif token.dep_ == "compound" and token.pos_ == "VERB" and next_token.pos_ == "NOUN":
                    phrase = f"{token.text} {next_token.text}".lower()
                    if phrase not in tasks:
                        print(f"⚙️ Fixed compound verb: {phrase}")  # optional debug
                        tasks.append(phrase)

                # ✅ NEW: 문장 맨 앞 대문자 때문에 PROPN으로 잡힌 동사 rescue (Boil water)
                elif (
                    i == 0
                    and token.pos_ == "PROPN"
                    and next_token.pos_ == "NOUN"
                    and token.text[0].isupper()
                ):
                    phrase = f"{token.text.lower()} {next_token.text.lower()}"
                    if phrase not in tasks:
                        print(f"⚙️ Fixed PROPN-as-verb: {phrase}")
                        tasks.append(phrase)
                        
        # -------------------------
    # EXTRA RESCUE: clause 시작의 "X Y" (imperative likely)
    # - spaCy가 X를 VERB로 못 잡아도, "Boil water" 같은 케이스 복구
    # - verb 하드코딩 없이: "첫 토큰이 동사로 쓰였을 가능성" + "둘째 토큰이 명사류" 조건
    # -------------------------
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

            # 목적어는 명사류면 OK (NOUN/PROPN 흔들림 대응)
            if t1.pos_ not in {"NOUN", "PROPN"} or not t1.is_alpha:
                continue

            # 첫 토큰이 명백히 "동사"가 아니어도(=PROPN/NOUN로 오인) 복구하고 싶으니
            # 단, 대명사/관사/전치사/접속사/숫자는 제외 (오탐 방지)
            if t0.pos_ in {"PRON", "DET", "ADP", "CCONJ", "SCONJ", "NUM"}:
                continue

            # filler/meta word 제외
            v = t0.text.lower()
            n = t1.text.lower()
            if v in FILLER_WORDS or n in FILLER_WORDS:
                continue

            # 한 글자 같은 잡음 제외
            if len(v) <= 1 or len(n) <= 1:
                continue

            phrase = f"{v} {n}".strip()

            # 이미 있는 더 긴 task의 subphrase면 skip
            # (ordered는 아직 없으니 tasks 대상으로 검사하거나, 아래 단계에서 필터해도 됨)
            if phrase in tasks:
                continue

            rescue_candidates.append((t0.idx, phrase))
    
    # 3. 중복 제거 + 순서 유지
    seen, ordered = set(), []
    
    BAD_TASKS = {"please user", "please input", "user input"}
    ordered = [t for t in ordered if t not in BAD_TASKS and not t.startswith("please ")]
    
    for t in tasks:
        if t not in seen:
            ordered.append(t)
            seen.add(t)
    
    # ✅ Regex-added(=rescue) tasks를 "원문 등장 순"으로 추가 (insert(0) 금지)
    rescue_candidates.sort(key=lambda x: x[0])
    for _, phrase in rescue_candidates:
        if phrase in seen:
            continue
        if _is_subphrase_of_existing(phrase, ordered):
            continue
        print(f"⚙️ Regex-added task: {phrase}")
        ordered.append(phrase)
        seen.add(phrase)

    # ✅ fallback2 교체
    extra = _fallback2_spacy_verb_object(text, ordered)
    # 등장 순서대로 뒤에 붙이기 (ordered.insert(0, ...) 금지)
    for p in extra:
        ordered.append(p)

    print("[DBG] FINAL TASKS:", ordered)
    print("[DBG] SINGLE-TOKEN TASKS:", [t for t in ordered if len(t.split()) == 1])
    return ordered

# -------------------------
# BPMN XML에서 태스크 추출
# -------------------------
def parse_bpmn_tasks(path: str):
    tree = ET.parse(path)
    root = tree.getroot()
    tasks = []
    for label in root.findall(".//{*}label"):
        text = label.text or ""
        if text.strip():
            tasks.append(text.strip().lower())
    return tasks

import numpy as np
from typing import List, Dict, Tuple, Any

# -------------------------
# Similarity helpers
# -------------------------
def _token_set(text: str) -> set:
    # stopwords 제거까지 하고 싶으면 여기서 spaCy stopword 필터링 가능
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def normalize_task(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"\b(the|a|an)\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def jaccard_sim(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def sbert_cos_sim(a: str, b: str) -> float:
    ea = sbert.encode(a, convert_to_tensor=True)
    eb = sbert.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(ea, eb))

def blended_sim(a: str, b: str, alpha: float = 0.75) -> float:
    """
    alpha: SBERT 비중 (0~1). 보통 0.7~0.85 추천
    """
    a = normalize_task(a)
    b = normalize_task(b)
    return alpha * sbert_cos_sim(a, b) + (1 - alpha) * jaccard_sim(a, b)


# -------------------------
# 1:1 greedy matching (conflict-free)
# -------------------------
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

    # 모든 후보를 sim 내림차순으로 정렬 후 greedy로 1:1 매칭
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

from difflib import SequenceMatcher

def lex_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_anchors(user_tasks, model_tasks, sim_um, strong_exact=0.90, lex_exact=0.85, debug=False):
    # sim_um: shape (U, M)
    U, M = len(user_tasks), len(model_tasks)

    # 각 user의 best model, 각 model의 best user
    user_best = [(int(sim_um[i].argmax()), float(sim_um[i].max())) for i in range(U)]
    model_best = [(int(sim_um[:, j].argmax()), float(sim_um[:, j].max())) for j in range(M)]

    anchors = []  # list of (u_idx, m_idx, sim, lex)
    anchored_user = set()
    anchored_model = set()

    for u in range(U):
        m, s = user_best[u]
        # mutual best?
        u2, s2 = model_best[m]
        if u2 != u:
            continue

        l = lex_sim(user_tasks[u], model_tasks[m])

        if s >= strong_exact and l >= lex_exact:
            anchors.append((u, m, s, l))
            anchored_user.add(u)
            anchored_model.add(m)

    if debug:
        print("\n--- ANCHORS (strong 1:1 matches) ---")
        for u, m, s, l in anchors:
            print(f"ANCHOR user[{u}]='{user_tasks[u]}' <-> model[{m}]='{model_tasks[m]}' sim={s:.3f} lex={l:.3f}")

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
    - Anchors(1:1 강한 매칭)는 merge/split 후보에서 제외 (오탐 방지)
    - prefer_adjacent=True면, split에서 연속 모델 태스크를 우선 채택
    """

    # --- similarity matrix (U x M) ---
    user_embs = sbert.encode(user_tasks, convert_to_tensor=True)
    model_embs = sbert.encode(model_tasks, convert_to_tensor=True)
    sim_um = util.cos_sim(user_embs, model_embs).cpu().numpy()  # (U, M)

    # --- anchors: 1:1 확정 매칭 ---
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

    # (optional) 중복 방지: 하나의 태스크가 여러 merge/split에 과도하게 엮이는 것 방지
    used_users = set()
    used_models = set()

    # ======================
    # MERGE (model -> top2 users)
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

        merged.append({
            "model": m_task,
            "model_index": m_idx,
            "user": [user_tasks[u1], user_tasks[u2]],
            "user_indices": [u1, u2],
            "pair_sims": [round(s1, 3), round(s2, 3)],
            "combo_sim": round(combo_sim, 3),
        })

        used_models.add(m_idx)
        used_users.add(u1)
        used_users.add(u2)

    # ======================
    # SPLIT (user -> 2+ model)
    # ======================
    for u_idx, u_task in enumerate(user_tasks):
        if u_idx in anchored_user:
            continue
        if u_idx in used_users:
            continue

        # split 후보 model: sim >= split_model_min
        # anchored_model/used_models 제외
        cand = [
            (m_idx, float(sim_um[u_idx, m_idx]))
            for m_idx in range(len(model_tasks))
            if (m_idx not in anchored_model) and (m_idx not in used_models) and (sim_um[u_idx, m_idx] >= split_model_min)
        ]

        if len(cand) < 2:
            continue

        # 너무 많은 후보가 생기면 상위 후보만 사용(조합 폭발 방지)
        cand.sort(key=lambda x: x[1], reverse=True)
        cand = cand[:max_split_candidates]

        # 후보를 model index 기준으로 정렬해서 "연속 그룹" 찾기
        cand_by_idx = sorted(cand, key=lambda x: x[0])

        # 연속 그룹 만들기
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

        best_choice = None  # (score, models_list, sims_list, combo_sim, adjacent_bool)

        # 1) 연속 그룹 우선 평가
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

            # score: combo_sim 중심 + 그룹 길이 보너스(조금)
            score = combo_sim + 0.01 * (len(g) - 2)
            adjacent = True  # 그룹 자체가 연속

            if (best_choice is None) or (score > best_choice[0]):
                best_choice = (score, m_indices, sims, combo_sim, adjacent)

        # 2) prefer_adjacent인데 연속 그룹이 없으면 -> fallback pair(비연속) 고려할지 결정
        if best_choice is None and (not prefer_adjacent):
            # 모든 pair 중 best를 찾기(간단히)
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

        # 3) prefer_adjacent=True인데 연속 그룹이 없으면:
        #    "연속이 아니면 split으로 보지 않는다"가 정책이면 그냥 continue
        #    하지만 너가 원하면 여기서도 pair fallback을 허용할 수 있음.
        if best_choice is None:
            continue

        _, m_indices, sims, combo_sim, adjacent = best_choice

        split.append({
            "user": u_task,
            "user_index": u_idx,
            "model": [model_tasks[m] for m in m_indices],
            "model_indices": m_indices,
            "pair_sims": [round(float(s), 3) for s in sims],
            "combo_sim": round(float(combo_sim), 3),
            "adjacent": bool(adjacent),
        })

        # split로 사용된 model task들은 다른 merge/split에 또 쓰지 않게 막음(오탐 방지)
        used_users.add(u_idx)
        for m in m_indices:
            used_models.add(m)

    return merged, split


# -------------------------
# Compare the task elements
# -------------------------
def combined_similarity(a, b):
    a_vec = nlp(a).vector
    b_vec = nlp(b).vector
    cosine = 0
    if a_vec.any() and b_vec.any():
        cosine = float(a_vec @ b_vec) / ((a_vec**2).sum()**0.5 * (b_vec**2).sum()**0.5 + 1e-9)
    seq = SequenceMatcher(None, a, b).ratio()
    return max(cosine, seq)

# -------------------------
# Wrong Sequence Detection
# -------------------------
from typing import List, Dict, Any, Tuple

def task_similarity(u: str, m: str, u_emb=None, m_emb=None) -> float:
    """
    텍스트가 조금 달라도 매칭되도록:
    - 기존 combined_similarity(스파시 벡터/문자열 유사도)
    - SBERT cosine
    둘 중 더 큰 값 사용
    """
    sim1 = combined_similarity(u, m)

    sim2 = 0.0
    if u_emb is not None and m_emb is not None:
        sim2 = float(util.cos_sim(u_emb, m_emb))
    return max(sim1, sim2)


def extract_matched_tasks(
    user_tasks: List[str],
    model_tasks: List[str],
    # ADJUSTABLE
    threshold: float = 0.6
) -> List[Dict[str, Any]]:
    """
    공통 task 교집합을 '텍스트 달라도' 매칭되게 만들기.
    - 1:1 매칭 (중복 방지)
    - 전역 greedy: 가능한 모든 pair 중 similarity 큰 것부터 채택
    반환: match_pairs = [{user_i, model_j, sim}, ...]
    """
    if not user_tasks or not model_tasks:
        return []

    user_embs = sbert.encode(user_tasks, convert_to_tensor=True)
    model_embs = sbert.encode(model_tasks, convert_to_tensor=True)

    candidates = []
    for i, u in enumerate(user_tasks):
        for j, m in enumerate(model_tasks):
            sim = task_similarity(u, m, user_embs[i], model_embs[j])
            if sim >= threshold:
                candidates.append((sim, i, j))

    # similarity 높은 것부터 1:1로 채택
    candidates.sort(reverse=True, key=lambda x: x[0])

    used_u, used_m = set(), set()
    match_pairs = []
    for sim, i, j in candidates:
        if i in used_u or j in used_m:
            continue
        used_u.add(i)
        used_m.add(j)
        match_pairs.append({"user_i": i, "model_j": j, "sim": float(sim)})

    return match_pairs


def inversion_count_and_involvement(perm: List[int]) -> Tuple[int, List[int]]:
    """
    perm: model 순서로 정렬된 공통 태스크들을 user 순서 index로 바꾼 배열
    - inversion count = Kendall tau distance
    - involvement[k] = k번째 원소가 inversion에 얼마나 관여했는지(대략적인 blame)
      (정확한 per-item involvement는 O(n^2)이 가장 간단; n이 작을 거라 가정하고 O(n^2)로 제공)
    """
    n = len(perm)
    if n <= 1:
        return 0, [0] * n

    # Kendall tau distance = inversion count (Fenwick BIT)
    # 좌표압축
    vals = sorted(set(perm))
    rank = {v: idx + 1 for idx, v in enumerate(vals)}
    bit = [0] * (len(vals) + 2)

    def bit_add(i: int, v: int):
        while i < len(bit):
            bit[i] += v
            i += i & -i

    def bit_sum(i: int) -> int:
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s

    inv = 0
    seen = 0
    for x in perm:
        rx = rank[x]
        inv += seen - bit_sum(rx)
        bit_add(rx, 1)
        seen += 1

    involvement = [0] * n
    if n <= 2000:
        for a in range(n):
            for b in range(a + 1, n):
                if perm[a] > perm[b]:
                    involvement[a] += 1
                    involvement[b] += 1
                    
    return inv, involvement


def compute_kendall_tau(
    user_tasks: List[str],
    model_tasks: List[str],
    match_pairs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    1) 교집합(common) 시퀀스 생성: common_model_tasks / common_user_tasks
    2) Kendall tau distance 계산
    3) BPMN에서 어떤 태스크가 순서 오류에 관여했는지 추출
    """
    if len(match_pairs) <= 1:
        return {
            "common_user_tasks": [],
            "common_model_tasks": [],
            "kendall_tau_distance": 0,
            "kendall_tau_normalized": 0.0,
            "out_of_order_bpmn_tasks": [],
        }

    # common 시퀀스 만들기
    pairs_by_model = sorted(match_pairs, key=lambda x: x["model_j"])
    pairs_by_user  = sorted(match_pairs, key=lambda x: x["user_i"])

    common_model_tasks = [model_tasks[p["model_j"]] for p in pairs_by_model]
    common_user_tasks  = [user_tasks[p["user_i"]] for p in pairs_by_user]

    # Kendall tau 계산용 permutation 생성:
    # model 순서에서의 각 원소가 user 순서에서 몇 번째인지로 치환
    user_rank_of_pair = {}
    for rank, p in enumerate(pairs_by_user):
        user_rank_of_pair[(p["user_i"], p["model_j"])] = rank

    perm = [user_rank_of_pair[(p["user_i"], p["model_j"])] for p in pairs_by_model]

    inv, involvement = inversion_count_and_involvement(perm)
    n = len(perm)
    max_inv = n * (n - 1) // 2
    norm = (inv / max_inv) if max_inv > 0 else 0.0

    # “어떤 BPMN 태스크가 순서 틀림에 관여했는지” 리포트
    out_of_order = []
    for k, p in enumerate(pairs_by_model):
        inv = int(involvement[k])  # k번째(=common rank 기준) 관여도
        if inv > 0:
            out_of_order.append({
                "bpmn_task": model_tasks[p["model_j"]],
                "bpmn_index": p["model_j"],
                "matched_user_task": user_tasks[p["user_i"]],
                "user_index": p["user_i"],
                "inversions_involving_this_task": inv,  # ✅ 여기서 넣기
            })

    # 보기 좋게: 관여도 높은 순으로 정렬
    out_of_order.sort(key=lambda x: x.get("inversions_involving_this_task", 0), reverse=True)
    
    return {
        "common_user_tasks": common_user_tasks,
        "common_model_tasks": common_model_tasks,
        "kendall_tau_distance": int(inv),
        "out_of_order_bpmn_tasks": out_of_order,
        "common_count": n,
    }

def greedy_match_pairs_for_order(user_tasks, model_tasks, sim_threshold=0.6,
                                 exclude_user=None, exclude_model=None):
    exclude_user = exclude_user or set()
    exclude_model = exclude_model or set()

    user_embs = sbert.encode(user_tasks, convert_to_tensor=True)
    model_embs = sbert.encode(model_tasks, convert_to_tensor=True)
    sim_um = util.cos_sim(user_embs, model_embs).cpu().numpy()  # (U, M)

    edges = []
    for i in range(len(user_tasks)):
        if i in exclude_user:
            continue
        for j in range(len(model_tasks)):
            if j in exclude_model:
                continue
            s = float(sim_um[i, j])
            if s >= sim_threshold:
                edges.append((s, i, j))

    edges.sort(reverse=True, key=lambda x: x[0])

    matched_user = set()
    matched_model = set()
    pairs = []

    for s, i, j in edges:
        if i in matched_user or j in matched_model:
            continue
        matched_user.add(i)
        matched_model.add(j)

        pairs.append({
            # ✅ compute_kendall_tau 호환 키
            "user_i": i,
            "model_j": j,

            # (원하면 유지) 디버깅/리포트용
            "user_index": i,
            "model_index": j,
            "similarity": round(s, 3),
            "user_task": user_tasks[i],
            "model_task": model_tasks[j],
        })

    return pairs

# ============================================================
# Pair-based Wrong Sequence Detection (precedence edges)
# ============================================================

# ---------- (A) MODEL side: parse CPEE XML into precedence edges ----------
CPEE_NS = {"c": "http://cpee.org/ns/description/1.0"}

def cq(tag: str) -> str:
    return f"{{{CPEE_NS['c']}}}{tag}"

def _is_cpee_task_call(call_el: ET.Element) -> bool:
    t = call_el.find("./c:parameters/c:type", CPEE_NS)
    return (t is not None and (t.text or "").strip() == ":task")

def _cpee_label(call_el: ET.Element) -> str:
    lab = call_el.find("./c:parameters/c:label", CPEE_NS)
    return (lab.text or "").strip().lower() if lab is not None else ""

class CFRes:
    def __init__(self, start: set[str], end: set[str], edges: set[tuple[str,str]]):
        self.start = start
        self.end = end
        self.edges = edges

FLOW_CONTAINER_TAGS_CPEE = {"description", "alternative", "parallel_branch"}

def _build_edges_from_node(node: ET.Element) -> CFRes:
    local = node.tag.split("}")[-1]

    # task
    if node.tag == cq("call") and _is_cpee_task_call(node):
        lbl = _cpee_label(node)
        if not lbl:
            return CFRes(set(), set(), set())
        return CFRes({lbl}, {lbl}, set())

    # sequential container
    if local in FLOW_CONTAINER_TAGS_CPEE:
        child_res = []
        for ch in list(node):
            r = _build_edges_from_node(ch)
            if r.start or r.end or r.edges:
                child_res.append(r)

        if not child_res:
            return CFRes(set(), set(), set())

        edges = set()
        for r in child_res:
            edges |= r.edges

        # connect prev.end -> next.start
        for i in range(len(child_res) - 1):
            prev, nxt = child_res[i], child_res[i+1]
            for a in prev.end:
                for b in nxt.start:
                    if a != b:
                        edges.add((a,b))

        return CFRes(child_res[0].start, child_res[-1].end, edges)

    # XOR choose
    if local == "choose":
        branch_res = []
        for alt in node.findall("./c:alternative", CPEE_NS):
            r = _build_edges_from_node(alt)
            if r.start or r.end or r.edges:
                branch_res.append(r)
        if not branch_res:
            return CFRes(set(), set(), set())

        edges = set()
        for r in branch_res:
            edges |= r.edges
        start = set().union(*[r.start for r in branch_res])
        end   = set().union(*[r.end   for r in branch_res])
        return CFRes(start, end, edges)

    # AND parallel
    if local == "parallel":
        branch_res = []
        for br in node.findall("./c:parallel_branch", CPEE_NS):
            r = _build_edges_from_node(br)
            if r.start or r.end or r.edges:
                branch_res.append(r)
        if not branch_res:
            return CFRes(set(), set(), set())

        edges = set()
        for r in branch_res:
            edges |= r.edges
        start = set().union(*[r.start for r in branch_res])
        end   = set().union(*[r.end   for r in branch_res])
        return CFRes(start, end, edges)

    # fallback: treat unknown as sequential container
    child_res = []
    for ch in list(node):
        r = _build_edges_from_node(ch)
        if r.start or r.end or r.edges:
            child_res.append(r)
    if not child_res:
        return CFRes(set(), set(), set())

    edges = set()
    for r in child_res:
        edges |= r.edges
    for i in range(len(child_res) - 1):
        prev, nxt = child_res[i], child_res[i+1]
        for a in prev.end:
            for b in nxt.start:
                if a != b:
                    edges.add((a,b))
    return CFRes(child_res[0].start, child_res[-1].end, edges)

def extract_model_precedence_edges_cpee(xml_path: str) -> set[tuple[str,str]]:
    """
    Precedence edges between TASK LABELS (lowercase).
    Example: a->XOR->(b|c)->join->d  => (a,b),(a,c),(b,d),(c,d)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    res = _build_edges_from_node(root)
    return {(a,b) for (a,b) in res.edges if a and b and a != b}


# ---------- (B) USER side: build precedence edges from extracted relations ----------
def build_user_precedence_edges_from_rel(user_rel: Dict[str, Any]) -> set[tuple[str,str]]:
    """
    Uses your extract_gateway_relations output to create precedence edges between U-ids:
      - SEQUENTIAL(A,B) => (A,B)
      - When we see a JOIN_* over a set S, we treat S as "a branch segment"
        and when later a SEQUENTIAL from any s in S to some t appears, we add (all S -> t).
    This approximates join behavior and yields (a,b),(a,c),(b,d),(c,d) patterns.
    """
    edges: set[tuple[str,str]] = set()

    # 1) base sequential edges
    for (typ, a, b) in user_rel.get("relations", []):
        if typ == "SEQUENTIAL" and a.startswith("U") and b.startswith("U") and a != b:
            edges.add((a,b))

    # 2) track join sets and "lift" edges to represent join semantics
    join_sets = []
    for (jt, tids) in user_rel.get("join_relations", []):
        tids_u = [t for t in tids if t.startswith("U")]
        if len(tids_u) >= 2:
            join_sets.append(set(tids_u))

    if not join_sets:
        return edges

    # If any member of join set precedes X, assume all members precede X (join effect)
    extra = set()
    for S in join_sets:
        for (a,b) in list(edges):
            if a in S and b.startswith("U"):
                for s in S:
                    if s != b:
                        extra.add((s,b))
    edges |= extra

    return edges


def map_model_edges_to_user_ids(
    user_tasks: List[str],
    model_tasks: List[str],
    model_edges_labels: set[tuple[str,str]],
    sim_threshold: float = 0.60,
) -> tuple[set[tuple[str,str]], Dict[str,str]]:
    """
    Convert model edges (label,label) -> (U_i, U_j) using greedy matching.
    Returns:
      - mapped_edges: set[(U?,U?)] only when both endpoints could be mapped
      - label_to_uid: mapping from model label -> Uid
    """
    # Build a mapping label->uid by matching model_tasks to user_tasks (1:1)
    res = greedy_match(user_tasks, model_tasks, sim_threshold=sim_threshold)
    label_to_uid: Dict[str,str] = {}
    for (ui, mj, _sim) in res["pairs"]:
        label_to_uid[normalize_task(model_tasks[mj])] = make_tid_user(ui)

    mapped = set()
    for (a_lbl, b_lbl) in model_edges_labels:
        a = label_to_uid.get(normalize_task(a_lbl))
        b = label_to_uid.get(normalize_task(b_lbl))
        if a and b and a != b:
            mapped.add((a,b))

    return mapped, label_to_uid


def compare_precedence_edges(
    user_edges_uid: set[tuple[str,str]],
    model_edges_uid: set[tuple[str,str]],
) -> Dict[str, Any]:
    """
    Compare edge sets on the SAME id space (U*).
    """
    missing_in_model = sorted(list(user_edges_uid - model_edges_uid))
    extra_in_model   = sorted(list(model_edges_uid - user_edges_uid))

    swap_suspicions = []
    model_set = set(model_edges_uid)
    for (a,b) in user_edges_uid:
        if (a,b) not in model_set and (b,a) in model_set:
            swap_suspicions.append((a,b))

    return {
        "missing_precedence_in_model": missing_in_model,  # user expects but model doesn't have
        "extra_precedence_in_model": extra_in_model,      # model has but user doesn't expect
        "swap_suspicions": swap_suspicions,
        "missing_count": len(missing_in_model),
        "extra_count": len(extra_in_model),
        "swap_count": len(swap_suspicions),
    }

# -------------------------
# Gateway logic check
# -------------------------
def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

from functools import lru_cache

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
    # 빠른 substring (유지)
    t_norm = normalize_task(task)
    s_norm = normalize_task(sent)
    if t_norm and t_norm in s_norm:
        return True

    # lemma sets
    t_lem = _lemma_token_set(task)
    s_lem = _lemma_token_set(sent)
    if not t_lem or not s_lem:
        return False

    # particle은 매칭에서 제외(수동태에서 'up'이 빠지는 경우가 많음)
    t_lem2 = {x for x in t_lem if x not in PARTICLES}

    # ✅ coverage(= recall): task lemma가 문장에 얼마나 포함되는지
    inter = t_lem2 & s_lem
    cover = len(inter) / max(1, len(t_lem2))

    return cover >= min_cover

def align_user_output_tasks(user_tasks, out_tasks, sim_threshold=0.60):
    res = greedy_match(user_tasks, out_tasks, sim_threshold=sim_threshold)

    pairs = []
    out_to_user = {}
    user_to_out = {}

    for (ui, oj, sim) in res["pairs"]:
        pairs.append({"user_i": ui, "out_j": oj, "sim": float(sim)})
        out_to_user[oj] = ui
        user_to_out[ui] = oj

    # ✅ NEW: out_task가 'prepare' 같은 1-token이고 아직 매핑이 없으면
    transitive_single_verbs = {"prepare", "use", "pour", "boil", "set"}  # 목적어 있어야 정상인 동사만
    for oj, ot in enumerate(out_tasks):
        if oj in out_to_user:
            continue
        toks = ot.split()
        if len(toks) != 1:
            continue
        v = toks[0]
        if v not in transitive_single_verbs:
            continue

        # user_tasks 중 "prepare ..." 처럼 같은 동사로 시작하는 후보 찾기
        cand = [ui for ui, ut in enumerate(user_tasks) if ut.startswith(v + " ")]
        if len(cand) == 1 and cand[0] not in user_to_out:
            out_to_user[oj] = cand[0]
            user_to_out[cand[0]] = oj
            pairs.append({"user_i": cand[0], "out_j": oj, "sim": 0.0})  # fallback이라 sim=0 표시해도 됨

    return {"pairs": pairs, "out_to_user": out_to_user, "user_to_out": user_to_out}

PARALLEL_CUES = re.compile(r"\b(simultaneously|in parallel|at the same time|concurrently)\b", re.I)
EXCLUSIVE_CUES = re.compile(r"\b(either\s+.+\s+or|either|or\b|alternatively|instead)\b", re.I)
OR_SENT_START = re.compile(r"^\s*(or|alternatively|otherwise)\b", re.I)

JOIN_AND_CUES = re.compile(r"\b(after|once)\s+both\b|\bboth\s+branches\b", re.I)
JOIN_XOR_CUES = re.compile(r"\b(after|once)\s+either\b|\bafter\s+either\b", re.I)

# output verbalization에 특화된 cue들
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

    # 1) 후보 hit 수집 + 간단 점수(길이/자카드) 계산
    cand = []
    for i, t in enumerate(tasks):
        t_norm = normalize_task(t)
        if not t_norm:
            continue

        # 빠른 substring
        if t_norm in sent_norm:
            score = 1.0 + 0.01 * len(t_norm)   # 길이 가산
            cand.append((i, score))
            continue
        
        # lemma coverage
        if task_in_sentence(t, sent, min_cover=0.60):
            score = 0.9 + 0.001 * len(t_norm)
            cand.append((i, score))
            continue

        # token jaccard
        jt = jaccard_sim(t_norm, sent_norm)
        if jt >= 0.30:
            score = jt + 0.001 * len(t_norm)
            cand.append((i, score))

    if not cand:
        return []

    # 2) 점수 높은 순 정렬
    cand.sort(key=lambda x: x[1], reverse=True)

    # 3) subset/substring 제거: 이미 선택된 더 긴 task가 있으면 짧은 task는 버림
    picked = []
    picked_token_sets = []
    for idx, _score in cand:
        t_norm = normalize_task(tasks[idx])
        t_set = set(re.findall(r"[a-z0-9]+", t_norm))

        dominated = False
        for j, t_set2 in enumerate(picked_token_sets):
            t_norm2 = normalize_task(tasks[picked[j]])
            # (짧은게 긴 것에 포함) 조건: 토큰 subset AND 문자열도 substring
            if t_set and t_set.issubset(t_set2) and t_norm in t_norm2:
                dominated = True
                break

        if not dominated:
            picked.append(idx)
            picked_token_sets.append(t_set)

    # 4) 문장 내 등장 순서로 정렬
    picked.sort(
        key=lambda i: sent_norm.find(normalize_task(tasks[i]))
        if normalize_task(tasks[i]) in sent_norm else 10**9
    )
    return picked

def extract_gateway_relations(
    text: str,
    tasks: List[str],
    side: str,
    out_to_user: Dict[int, int] = None,  # output side에서 공통 id로 끌어올릴 때 사용
) -> Dict[str, Any]:
    """
    side: "user" or "output"
    out_to_user: output task index -> user task index (align 결과)

    return:
      {
        "relations": [ ... ],
        "join_relations": [ ... ],
        "evidence": [ ... ]  # relation별 문장 evidence
      }
    """
    out_to_user = out_to_user or {}
    sents = _sentences(text)

    relations = []      # pairwise: [("PARALLEL"/"EXCLUSIVE"/"SEQUENTIAL", tidA, tidB), ...]
    join_relations = [] # set-based: [("JOIN_AND"/"JOIN_XOR", [tid...]), ...]
    evidence = []

    def tid_of_task_idx(idx: int) -> str:
        if side == "user":
            return make_tid_user(idx)
        # output side: 매칭되면 user tid로 올림
        if idx in out_to_user:
            return make_tid_user(out_to_user[idx])
        return make_tid_out(idx)

    # ---- branch parsing (output 전용) ----
    if side == "output":
        lower = text.lower()
        if OUT_BRANCH_1.search(lower) and OUT_BRANCH_2.search(lower):

            # ✅ 먼저: 이 branch가 AND(Parallel)인지 OR(XOR)인지 판정
            is_xor = bool(OUT_XOR_VERB.search(lower)) or bool(EXCLUSIVE_CUES.search(lower))
            is_parallel = bool(OUT_PARALLEL_VERB.search(lower)) or bool(PARALLEL_CUES.search(lower))

            # XOR cue가 있으면 EXCLUSIVE 우선 (둘 다 뜨면 parallel 쪽으로 두고 싶으면 여기 정책 바꿔도 됨)
            branch_relation_type = "EXCLUSIVE" if (is_xor and not is_parallel) else "PARALLEL"
            
            print("[DBG] BRANCH TYPE =", branch_relation_type, "| is_xor=", is_xor, "| is_parallel=", is_parallel)

            parts = re.split(r"(?i)\bin\s+the\s+second\s+branch\b", text, maxsplit=1)
            first_part = parts[0]
            second_part = parts[1] if len(parts) > 1 else ""
            first_part = re.split(r"(?i)\bin\s+the\s+first\s+branch\b", first_part, maxsplit=1)[-1]

            first_hits, second_hits = set(), set()

            def _clauses(s: str) -> list[str]:
                # ✅ 네가 넣었던 clause split 그대로 사용 가능
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

            # ✅ branch 간 관계를 PARALLEL/EXCLUSIVE로 찍기
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

            # ✅ join도 일관되게
            tids = sorted(set([tid_of_task_idx(i) for i in (first_hits | second_hits)]))
            if len(tids) >= 2:
                if branch_relation_type == "PARALLEL":
                    join_relations.append(("JOIN_AND", tids))
                else:
                    join_relations.append(("JOIN_XOR", tids))

    # ---- sentence-level cues (user/output 공통) ----
    prev_tids_unique: List[str] = []
    prev_sent: str | None = None

    for sent in sents:
        hits = find_tasks_in_sentence(tasks, sent)

        # 문장에 task가 1개도 없으면 prev 업데이트도 하지 않음
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

        # ✅ (A) 문장 내부 관계: task 2개 이상일 때만
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
                # cue가 없으면 순차로만 약하게 추출(문장 내 등장 순서 기준)
                for i in range(len(tids_unique) - 1):
                    a, b = tids_unique[i], tids_unique[i + 1]
                    if a != b:
                        relations.append(("SEQUENTIAL", a, b))
                        evidence.append({"type": "SEQUENTIAL", "tasks": [a, b], "evidence": sent})

        # ✅ (B) 문장 사이 EXCLUSIVE: "A. Or B."
        if is_excl and prev_sent is not None and prev_tids_unique:
            cur_rep = tids_unique[0]

            # 🔥 "Or,"로 시작하면: 이전 문장 '첫 task'(branch start)와 연결
            if OR_SENT_START.search(sent):
                prev_rep = prev_tids_unique[0]   # <- 여기 중요(기존은 -1)
            else:
                prev_rep = prev_tids_unique[-1]  # 기존 로직 유지

            if prev_rep != cur_rep:
                x, y = sorted([prev_rep, cur_rep])
                relations.append(("EXCLUSIVE", x, y))
                evidence.append({
                    "type": "EXCLUSIVE",
                    "tasks": [x, y],
                    "evidence": f"(prev) {prev_sent}  ||  (cur) {sent}"
                })
        
        # ✅ (C) 문장 사이 기본 SEQUENTIAL: "A. Then B." 같은 서술을 SEQ로 연결
        # (EXCLUSIVE / PARALLEL cue가 없는 경우에만)
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

        # prev 업데이트는 항상(현재 문장에 task가 1개 이상이면)
        prev_tids_unique = tids_unique
        prev_sent = sent

    # ✅ 중복 제거는 루프 밖에서 1번만
    rel_set = list(dict.fromkeys(relations))

    join_set = []
    seen_join = set()
    for jt_type, jt_list in join_relations:
        # jt_list는 list이므로 JSON-safe. key는 tuple로 만들어 dedup에만 사용
        key = (jt_type, tuple(sorted(jt_list)))
        if key in seen_join:
            continue
        seen_join.add(key)
        join_set.append((jt_type, sorted(jt_list)))

    return {"relations": rel_set, "join_relations": join_set, "evidence": evidence}

def compare_gateway_relations(user_rel: Dict[str, Any], out_rel: Dict[str, Any]) -> Dict[str, Any]:
    """
    교집합 task_id(U*)에 대해서만 비교.
    mismatch types:
      - branching_semantics_inconsistency (PARALLEL <-> EXCLUSIVE)
      - parallel_to_sequential_inconsistency
      - exclusive_flattened_to_sequence
      - join_semantics_inconsistency (JOIN_AND <-> JOIN_XOR)
    """
    def is_common_tid(tid: str) -> bool:
        return tid.startswith("U")

    # pair -> type set
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

    # compare pair relations on union keys
    all_keys = set(user_map.keys()) | set(out_map.keys())
    for key in all_keys:
        ut = user_map.get(key, set())
        ot = out_map.get(key, set())

        # AND <-> XOR/OR
        if ("PARALLEL" in ut and "EXCLUSIVE" in ot) or ("EXCLUSIVE" in ut and "PARALLEL" in ot):
            mismatches.append({
                "mismatch_type": "branching_semantics_inconsistency",
                "task_pair": list(key),
                "user_types": sorted(list(ut)),
                "output_types": sorted(list(ot)),
            })
            continue

        # parallel -> sequential
        if "PARALLEL" in ut and "SEQUENTIAL" in ot:
            mismatches.append({
                "mismatch_type": "parallel_to_sequential_inconsistency",
                "task_pair": list(key),
                "user_types": sorted(list(ut)),
                "output_types": sorted(list(ot)),
            })
            continue

        # exclusive flattened to sequential
        if "EXCLUSIVE" in ut and "SEQUENTIAL" in ot:
            mismatches.append({
                "mismatch_type": "exclusive_flattened_to_sequence",
                "task_pair": list(key),
                "user_types": sorted(list(ut)),
                "output_types": sorted(list(ot)),
            })
            continue

    # join 비교 (같은 task-set에 대해)
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
        "user_relations": user_rel,
        "output_relations": out_rel,
        "gateway_mismatches": mismatches,
        "mismatch_count": len(mismatches),
    }

# -------------------------
# Compare the task lists
# -------------------------
def compare_tasks(user_tasks, model_tasks, user_text="",output_text="", bpmn_path=""):
    # -------------------------
    # 1) MERGE / SPLIT 먼저 탐지
    # -------------------------
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

    # merge/split에 사용된 task index는 "설명됨(explained)"으로 표시
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

    # -------------------------
    # 2) 1:1 매칭은 merge/split 제외한 것만 대상으로
    # -------------------------
    # greedy_match가 전체 리스트를 받도록 되어있다면,
    # 내부에서 제외할 수 있게 "masking" 하거나, 여기서 서브리스트 만들어서 매칭 후 원래 index로 복원
    # → 여기서는 "서브리스트로 매칭 후 index 복원" 방식 사용

    remaining_user_idxs = [i for i in range(len(user_tasks)) if i not in explained_user]
    remaining_model_idxs = [j for j in range(len(model_tasks)) if j not in explained_model]

    remaining_user_tasks = [user_tasks[i] for i in remaining_user_idxs]
    remaining_model_tasks = [model_tasks[j] for j in remaining_model_idxs]

    match_res = greedy_match(remaining_user_tasks, remaining_model_tasks, sim_threshold=0.55)

    matched_user = set()
    matched_model = set()

    # greedy_match가 "index set"을 주는 형태라고 가정하고 원래 index로 복원
    for u_local in match_res["matched_user"]:
        matched_user.add(remaining_user_idxs[u_local])
    for m_local in match_res["matched_model"]:
        matched_model.add(remaining_model_idxs[m_local])

    # -------------------------
    # 3) merge/split에 포함된 index는 matched로도 포함 (missing/additional 방지)
    # -------------------------
    matched_user |= explained_user
    matched_model |= explained_model

    # -------------------------
    # 4) missing / additional 계산
    # -------------------------
    missing = [user_tasks[i] for i in range(len(user_tasks)) if i not in matched_user]
    additional = [model_tasks[j] for j in range(len(model_tasks)) if j not in matched_model]
 
    # -------------------------
    # 5) order_analysis (PAIR-BASED precedence)
    # -------------------------
    order_analysis = {
        "note": "",
        "user_edges_uid": [],
        "model_edges_uid": [],
        "edge_mismatch": {},
    }

    if not bpmn_path:
        order_analysis["note"] = "Pair-based order analysis skipped: bpmn_path not provided."
    else:
        # (1) user precedence edges from your gateway relations (U-id space)
        user_rel = extract_gateway_relations(
            text=user_text,
            tasks=user_tasks,
            side="user",
        )
        user_edges_uid = build_user_precedence_edges_from_rel(user_rel)

        # (2) model precedence edges from XML (label-label)
        model_edges_labels = extract_model_precedence_edges_cpee(bpmn_path)

        # (3) map model label edges -> U-id edges using your task matching
        model_edges_uid, label_to_uid = map_model_edges_to_user_ids(
            user_tasks=user_tasks,
            model_tasks=model_tasks,
            model_edges_labels=model_edges_labels,
            sim_threshold=0.60,
        )

        # (4) OPTIONAL: compare only edges where both endpoints are common U-ids
        common_uids = set([make_tid_user(i) for i in range(len(user_tasks))])
        user_edges_uid  = {(a,b) for (a,b) in user_edges_uid  if a in common_uids and b in common_uids}
        model_edges_uid = {(a,b) for (a,b) in model_edges_uid if a in common_uids and b in common_uids}

        edge_mismatch = compare_precedence_edges(user_edges_uid, model_edges_uid)

        order_analysis = {
            "note": "Pair-based precedence comparison (U-id space).",
            "user_edge_count": len(user_edges_uid),
            "model_edge_count": len(model_edges_uid),
            "edge_mismatch": edge_mismatch,
        }

    # ✅ gateway_analysis는 무조건 기본값으로 먼저 정의
    gateway_analysis = {
        "note": "No output_text provided; gateway analysis skipped.",
        "gateway_mismatches": [],
        "mismatch_count": 0
    }

    # ✅ output_text 있을 때만 gateway 분석 수행
    if user_text and output_text:
        canonical = user_tasks

        out_tasks = extract_verbal_tasks(output_text)
        out_tasks = [t for t in out_tasks if not is_output_meta_task(t)]
        print("OUT_TASKS =", out_tasks)

        align = align_user_output_tasks(user_tasks, out_tasks, sim_threshold=0.60)
        out_to_user = align["out_to_user"]
        print("ALIGN PAIRS =", align["pairs"])
        print("OUT_TO_USER =", out_to_user)

        user_rel = extract_gateway_relations(
            text=user_text,
            tasks=canonical,
            side="user",
        )

        out_rel = extract_gateway_relations(
            text=output_text,
            tasks=out_tasks,
            side="output",
            out_to_user=out_to_user,
        )

        print("USER_REL CNT:", len(user_rel["relations"]), "JOIN:", len(user_rel["join_relations"]))
        print("OUT_REL  CNT:", len(out_rel["relations"]), "JOIN:", len(out_rel["join_relations"]))
        
        gateway_analysis = compare_gateway_relations(user_rel, out_rel)

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


# -------------------------
# Main Runner
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--bpmn", required=True)
    ap.add_argument("--user-text-stdin",
        action="store_true",
        help="If set, read user text from stdin (paste text, then Ctrl-D).")
    ap.add_argument("--output-text-stdin",
        action="store_true",
        help="If set, read output text from stdin (paste text, then Ctrl-D)."
    )
    args = ap.parse_args()

    # 기본값: user_text는 YAML에서
    user_text = read_user_text_from_yaml(args.log)

    output_text = ""

    # ✅ gateway 모드: stdin으로 user/output 받기 (구분자 사용)
    # 규칙: stdin에
    #   USER:
    #   ...
    #   ---OUTPUT---
    #   ...
    if args.user_text_stdin or args.output_text_stdin:
        raw = sys.stdin.read()
        if "---OUTPUT---" in raw:
            user_part, out_part = raw.split("---OUTPUT---", 1)
            if args.user_text_stdin:
                user_text = user_part.strip()
            if args.output_text_stdin:
                output_text = out_part.strip()
        else:
            # 구분자 없으면: user_text만 넣은 걸로 간주(원하면 반대로도 가능)
            if args.user_text_stdin:
                user_text = raw.strip()
            if args.output_text_stdin:
                output_text = raw.strip()

    user_tasks = extract_verbal_tasks(user_text)
    model_tasks = parse_bpmn_tasks(args.bpmn)
    
    report = compare_tasks(
        user_tasks,
        model_tasks,
        user_text,
        output_text,
        bpmn_path=args.bpmn
    )

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

    print(f"\nUser tasks: {user_tasks}")
    print(f"Model tasks: {model_tasks}")
    print(f"Gateway analysis: {'ON' if output_text else 'OFF'}")
    print(f"Report written to {output_path}")
    
    print("[DBG] user_text(from yaml) =", user_text[:200])
    user_tasks = extract_verbal_tasks(user_text)
    print("[DBG] user_tasks =", user_tasks)
    print(nlp("boil water."))
    
if __name__ == "__main__":
    main()