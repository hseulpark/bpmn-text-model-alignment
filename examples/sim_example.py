import sys
from pathlib import Path
import argparse
import json
import re

"""
Example:
  python3 examples/sim_example.py \
    --xml examples/models/gdpr_7_right_to_be_forgotten.json.xml \
    --txt examples/text/gdpr_7_right_to_be_forgotten.json.autobpmn.txt \
    --report_dir examples/reports/comparison_results
"""

# --- make repo root importable ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compare_text_model import (
    read_user_text,
    extract_verbal_tasks,
    parse_bpmn_tasks,
    blended_sim,
    greedy_match,
    detect_merge_split,
    compare_tasks,  # ✅ comparator core (import + call directly)
)

def safe_name(p: Path) -> str:
    # safe filename stem for json
    s = p.name
    s = re.sub(r"\.xml$", "", s, flags=re.I)
    s = re.sub(r"\.txt$", "", s, flags=re.I)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="examples/models/example_1.xml")
    ap.add_argument("--txt", default="examples/text/example_1.txt")
    ap.add_argument("--sim_threshold", type=float, default=0.55)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument(
        "--report_dir",
        default="examples/reports/comparison_results",
        help="Directory to write the comparator JSON report",
    )
    ap.add_argument(
        "--no_gateway_api",
        action="store_true",
        help="If set: do NOT call external API for output_text (offline-safe).",
    )
    args = ap.parse_args()

    xml_path = (ROOT / args.xml).resolve()
    txt_path = (ROOT / args.txt).resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")
    if not txt_path.exists():
        raise FileNotFoundError(f"TXT not found: {txt_path}")

    # 1) load input
    user_text = read_user_text(str(txt_path))
    user_tasks = extract_verbal_tasks(user_text)
    model_tasks = parse_bpmn_tasks(str(xml_path))

    print("=" * 80)
    print("REAL DEMO: text-model comparison (minimal)")
    print(f"XML : {xml_path.relative_to(ROOT)}")
    print(f"TXT : {txt_path.relative_to(ROOT)}")
    print("=" * 80)

    # 2) show extracted tasks
    print("\n[User tasks extracted]")
    for i, t in enumerate(user_tasks, 1):
        print(f"  U{i:02d}: {t}")

    print("\n[Model tasks extracted]")
    for i, t in enumerate(model_tasks, 1):
        print(f"  M{i:02d}: {t}")

    # 3) merge/split
    merged, split = detect_merge_split(user_tasks, model_tasks)
    print("\n[Merge / Split signals]")
    print(f"  merged candidates: {len(merged)}")
    for m in merged[: args.topk]:
        print(f"   - MERGE: model='{m.get('model')}'  user={m.get('user')}")

    print(f"  split candidates: {len(split)}")
    for s in split[: args.topk]:
        print(f"   - SPLIT: user='{s.get('user')}'  model={s.get('model')}")

    # 4) 1:1 greedy matching
    match = greedy_match(user_tasks, model_tasks, sim_threshold=args.sim_threshold)

    print("\n[Greedy 1:1 matches]")
    if not match["pairs"]:
        print(f"  (no pairs >= {args.sim_threshold})")
    else:
        for (ui, mj, sim) in match["pairs"]:
            print(f"  U{ui+1:02d} <-> M{mj+1:02d}   sim={sim:.3f}")
            print(f"     user : {user_tasks[ui]}")
            print(f"     model: {model_tasks[mj]}")

    # 5) missing/additional based on 1:1 only
    matched_user = set(match["matched_user"])
    matched_model = set(match["matched_model"])

    missing = [user_tasks[i] for i in range(len(user_tasks)) if i not in matched_user]
    additional = [model_tasks[j] for j in range(len(model_tasks)) if j not in matched_model]

    print("\n[Simple missing/additional based on 1:1 matches only]")
    print(f"  missing tasks   : {len(missing)}")
    for t in missing[: args.topk]:
        print("   -", t)

    print(f"  additional tasks: {len(additional)}")
    for t in additional[: args.topk]:
        print("   -", t)

    # 6) quick similarity sanity check (top-k similar for first user task)
    if user_tasks and model_tasks:
        u0 = user_tasks[0]
        scored = [(blended_sim(u0, m), m) for m in model_tasks]
        scored.sort(reverse=True, key=lambda x: x[0])
        print("\n[Sanity: top similarities for first user task]")
        print("  user:", u0)
        for sim, m in scored[: args.topk]:
            print(f"   sim={sim:.3f}  model='{m}'")

    # ------------------------------------------------------------------
    # ✅ 핵심: CLI를 돌리지 않고, comparator 함수(compare_tasks)를 직접 호출해서
    #    JSON 리포트를 생성 + 저장
    # ------------------------------------------------------------------
    # offline-safe: output_text는 빈 문자열로 두면, 외부 API 없이도 report 생성 가능
    output_text = ""  # keep empty for offline demo
    output_note = "output_text skipped (sim_example.py offline mode)."

    # IMPORTANT:
    # - compare_tasks가 bpmn_path를 받아야 order/gateway(구조 기반) 부분이 동작함
    # - output_text가 비어있으면 gateway의 "text-based" 일부는 스킵되고 note에 남음
    comparison = compare_tasks(
        user_tasks=user_tasks,
        model_tasks=model_tasks,
        user_text=user_text,
        output_text=output_text,
        bpmn_path=str(xml_path),
    )

    # filename
    out_name = f"{safe_name(xml_path)}__{safe_name(txt_path)}.sim_report.json"
    out_path = report_dir / out_name

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "log_path": str(txt_path),
                "bpmn_path": str(xml_path),
                "user_tasks": user_tasks,
                "model_tasks": model_tasks,
                "comparison": comparison,
                "user_text": user_text,
                "output_text": output_text,
                "output_note": output_note,
                "note": "Generated by sim_example.py via direct function calls (no comparator CLI).",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n[Comparator JSON report saved]")
    print(" ", out_path.relative_to(ROOT))
    print("\nDONE.")

if __name__ == "__main__":
    main()