"""
Example:
  python run_compare_eval_balanced.py \
    --models_dir models_with_error/eval_balanced \
    --desc_root process_description \
    --script compare_spacy_eval.py \
    --report_root report_batch_eval
"""

import argparse
import subprocess
from pathlib import Path

def base_name_from_err_xml(xml_path: Path) -> str:
    name = xml_path.name
    if ".err" in name:
        return name.split(".err", 1)[0]
    return xml_path.stem

def index_descriptions(desc_root: Path) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = {}
    for p in desc_root.rglob("*.txt"):
        idx.setdefault(p.name, []).append(p)
    return idx

def choose_best_desc(paths: list[Path]) -> Path:
    priority = ["sapsam", "domain", "pet"]

    def score(p: Path) -> int:
        s = str(p).replace("\\", "/")
        sc = 0
        
        if p.name.endswith(".autobpmn.txt"):
            sc += 1000

        for i, key in enumerate(priority):
            if f"/{key}/" in s:
                sc += (100 - i)
                break

        return sc

    return sorted(paths, key=score, reverse=True)[0]

def resolve_txt_for_base(desc_idx: dict[str, list[Path]], base: str) -> Path | None:
    cand1 = desc_idx.get(f"{base}.autobpmn.txt", [])
    cand2 = desc_idx.get(f"{base}.txt", [])
    cands = cand1 + cand2
    if not cands:
        return None
    return choose_best_desc(cands)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--desc_root", required=True)
    ap.add_argument("--script", default="compare_spacy_eval.py")
    ap.add_argument("--report_root", default="report_batch_eval")
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    desc_root = Path(args.desc_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    if not models_dir.exists():
        raise FileNotFoundError(f"models_dir not found: {models_dir}")
    if not desc_root.exists():
        raise FileNotFoundError(f"desc_root not found: {desc_root}")

    desc_idx = index_descriptions(desc_root)

    groups = [p.name for p in sorted(models_dir.iterdir()) if p.is_dir()]
    if args.groups:
        groups = [g for g in groups if g in set(args.groups)]

    if not groups:
        print("No groups found under:", models_dir)
        return

    pairs = []
    for g in groups:
        xml_dir = models_dir / g
        for xml_path in sorted(xml_dir.rglob("*.xml")):
            base = base_name_from_err_xml(xml_path)
            txt_path = resolve_txt_for_base(desc_idx, base)
            if txt_path is None:
                print(f"WARNING: no txt for {xml_path.name} (tried {base}.autobpmn.txt and {base}.txt)")
                continue
            pairs.append((xml_path, txt_path, g))

    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    if not pairs:
        print("No (xml, txt) pairs found.")
        return

    ok = 0
    fail = 0

    for i, (xml_path, txt_path, g) in enumerate(pairs, start=1):
        run_dir = report_root / g
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", str(Path(args.script).resolve()),
            "--log", str(txt_path.resolve()),
            "--bpmn", str(xml_path.resolve()),
            "--report_root", str(run_dir.resolve()),
        ]

        print(f"[{i}/{len(pairs)}] {g}: {xml_path.name}  <->  {txt_path.name}")
        res = subprocess.run(cmd, cwd=str(run_dir), capture_output=True, text=True)

        if res.returncode != 0:
            fail += 1
            print("  FAIL")
            print("  STDOUT:", (res.stdout or "")[-1200:])
            print("  STDERR:", (res.stderr or "")[-1200:])
        else:
            ok += 1
            print("  OK")

    print("\nDONE")
    print(" OK  :", ok)
    print(" FAIL:", fail)
    print(" reports under:", report_root)

if __name__ == "__main__":
    main()