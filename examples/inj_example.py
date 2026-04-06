"""
python3 examples/inj_example.py \
  --xml examples/models/healthcare_bpmai_2.xml \
  --force_error additional_task \
  --seed 42 \
  --out_dir examples/reports/injected
  
python3 examples/inj_example.py \
  --xml examples/models/gdpr_7_right_to_be_forgotten.json.xml \
  --n_errors 3 \
  --seed 42 \
  --out_dir examples/reports/injected
"""

import sys
import argparse
from pathlib import Path

# --- make repo root importable ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ✅ import from your injection implementation
from inject_errors import (
    apply_random_errors,
    normalize_force_error,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Path to input XML (e.g., examples/models/xxx.xml)")
    ap.add_argument("--out_dir", default="examples/reports/injected", help="Where to write injected XMLs")
    ap.add_argument("--n_errors", type=int, default=3, help="Number of errors to inject (ignored when --force_error is used)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--force_error", default=None, help="Force exactly one error (e.g., and_to_seq, missing_task, xor_to_and, ...)")
    args = ap.parse_args()

    in_xml = (ROOT / args.xml).resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_xml.exists():
        raise FileNotFoundError(f"XML not found: {in_xml}")

    # output file name
    base = in_xml.stem
    if args.force_error:
        force_key = normalize_force_error(args.force_error)
        out_xml = out_dir / f"{base}.forced_{force_key}.seed{args.seed}.xml"
        n_errors = 1
        force_error = args.force_error  # apply_random_errors expects the human-readable form used in feasible_error_types
    else:
        out_xml = out_dir / f"{base}.err{args.n_errors}.seed{args.seed}.xml"
        n_errors = args.n_errors
        force_error = None

    # run injection
    applied = apply_random_errors(
        xml_path=str(in_xml),
        out_path=str(out_xml),
        n_errors=n_errors,
        seed=args.seed,
        force_error=force_error,
    )

    # print summary
    print("=" * 80)
    print("INJECTION DEMO (programmatic usage)")
    print(f"INPUT : {in_xml.relative_to(ROOT)}")
    print(f"OUTPUT: {out_xml.relative_to(ROOT)}")
    print(f"seed={args.seed}  n_errors={n_errors}  force_error={args.force_error or 'None'}")
    print("-" * 80)
    for e in applied:
        print(f"- {e.error_type} => {e.details}")
    print("=" * 80)
    print("DONE.")

if __name__ == "__main__":
    main()