import sys
import glob
import secrets
import subprocess
from pathlib import Path

SELECTED_FILES = [
    "healthcare_pufahl_4.xml",
    "logistics_cimino_3.xml",
    "logistics_ferreira_4.xml",
    "tourism_milanovic_4.xml",
    "1-1.xml",
    "1-2.xml",
    "1-3.xml",
    "2-1.xml",
    "3-2.xml",
    "11_ground_truth.xml",
    "17_ground_truth.xml",
    "21_ground_truth.xml",
    "29_ground_truth.xml",
    "30_ground_truth.xml",
    "32_ground_truth.xml",
    "33_ground_truth.xml",
    "34_ground_truth.xml",
    "40_ground_truth.xml",
    "47_ground_truth.xml",
    "53_ground_truth.xml",
    "59_ground_truth.xml",
    "61_ground_truth.xml",
    "90_ground_truth.xml",
    "healthcare_redeker_5.xml",
    "logistics_signavio_2.xml",
    "manufacturing_lodhi_5.xml",
    "1-4.xml",
    "2-2.xml",
    "3-5.xml",
    "19_ground_truth.xml",
    "22_ground_truth.xml",
    "25_ground_truth.xml",
    "37_ground_truth.xml",
    "39_ground_truth.xml",
    "51_ground_truth.xml",
    "79_ground_truth.xml",
    "80_ground_truth.xml",
]

def main():
    ground_root = Path("ground_truth")
    out_dir = Path("models_with_error") / "selected"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_errors = 3
    log_csv = "error_injection_log.csv"

    all_xml = glob.glob(str(ground_root / "**" / "*.xml"), recursive=True)
    by_name = {}
    for p in all_xml:
        name = Path(p).name
        by_name.setdefault(name, []).append(p)

    resolved = []
    for name in SELECTED_FILES:
        hits = by_name.get(name, [])
        if not hits:
            print(f"WARNING: NOT FOUND: {name} (under {ground_root})")
            continue
        if len(hits) > 1:
            print(f"WARNING: MULTIPLE FOUND for {name}:")
            for h in hits:
                print("   -", h)
        for h in hits:
            resolved.append(Path(h))

    if not resolved:
        print("WARNING: No files resolved. Check SELECTED_FILES and ground_truth path.")
        return

    ok, fail = 0, 0

    for i, xml_path in enumerate(resolved, start=1):
        seed = secrets.randbelow(2**31 - 1)

        cmd = [
            "python", "inject_errors.py",
            "--in_xml", str(xml_path),
            "--out_dir", str(out_dir),
            "--n_errors", str(n_errors),
            "--seed", str(seed),
            "--log_csv", log_csv,
        ]

        print(f"\n[{i}/{len(resolved)}] Running: {' '.join(cmd)}")
        res = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True)

        if res.returncode != 0:
            fail += 1
            print(f" FAIL: {xml_path}")
        else:
            ok += 1
            print(f" OK: {xml_path.name}")

    print("\nDONE")
    print(" OK  :", ok)
    print(" FAIL:", fail)
    print(" Output dir:", out_dir)

if __name__ == "__main__":
    main()