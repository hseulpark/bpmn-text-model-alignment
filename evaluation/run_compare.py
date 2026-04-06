"""
    python run_compare.py \
        --models_dir models_with_error \
        --desc_dir process_description \
        --script compare_spacy_eval.py \
        --report_root report_batch 
"""
import argparse
import subprocess
from pathlib import Path

def base_name_from_err_xml(xml_path: Path) -> str:
    # "name.err3.001.xml" -> "name"
    name = xml_path.name
    if ".err" in name:
        return name.split(".err", 1)[0]
    # fallback
    return xml_path.stem.replace(".xml", "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", default="models_with_error")
    ap.add_argument("--desc_dir", default="process_description")
    ap.add_argument("--script", default="compare_spacy_eval.py")
    ap.add_argument("--report_root", default="report_batch")
    ap.add_argument("--domains", nargs="*", default=["domain", "pet"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    desc_dir = Path(args.desc_dir)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    pairs = []
    for d in args.domains:
        xml_dir = models_dir / d
        txt_dir = desc_dir / d
        if not xml_dir.exists():
            print("WARNING: missing models dir:", xml_dir)
            continue
        if not txt_dir.exists():
            print("WARNING: missing desc dir:", txt_dir)
            continue

        for xml_path in sorted(xml_dir.rglob("*.xml")):
            base = base_name_from_err_xml(xml_path)
            txt_path = txt_dir / f"{base}.autobpmn.txt"

            if not txt_path.exists():
                print(f"WARNING: no txt for {xml_path.name} (expected {txt_path})")
                continue

            pairs.append((xml_path, txt_path, d))

    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    if not pairs:
        print("No (xml, txt) pairs found.")
        return

    ok = 0
    fail = 0

    for i, (xml_path, txt_path, d) in enumerate(pairs, start=1):
        run_dir = report_root / d
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", str(Path(args.script).resolve()),
            "--log", str(txt_path.resolve()),
            "--bpmn", str(xml_path.resolve()),
            "--report_root", str(run_dir.resolve()),
        ]

        print(f"[{i}/{len(pairs)}] {d}: {xml_path.name}  <->  {txt_path.name}")
        res = subprocess.run(cmd, cwd=str(run_dir), capture_output=True, text=True)

        if res.returncode != 0:
            fail += 1
            print("  FAIL")
            print("  STDOUT:", (res.stdout or "")[-800:])
            print("  STDERR:", (res.stderr or "")[-800:])
        else:
            ok += 1
            print("  OK")

    print("\nDONE")
    print(" OK  :", ok)
    print(" FAIL:", fail)
    print(" reports under:", report_root)

if __name__ == "__main__":
    main()