import csv
import random
import subprocess
from pathlib import Path

CSV_MODEL_COL = "Model name"

ERR_MAP = {
    "missing": "missing_task",
    "additional": "additional_task",
    "merged": "merged",
    "splitted": "split",  
    "2 wrong sequences": "2 wrong sequences",
    "random sequences": "random sequences",
    "AND -> XOR": "AND -> XOR",
    "AND -> SEQ": "AND -> SEQ",
    "XOR -> AND": "XOR -> AND",
    "XOR -> SEQ": "XOR -> SEQ",
}

PRIORITY_DIRS = ["sapsam", "domain", "pet"] 

def index_ground_truth(gt_root: Path) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = {}
    for p in gt_root.rglob("*.xml"):
        idx.setdefault(p.name, []).append(p)
    return idx

def choose_best_path(paths: list[Path]) -> Path:
    # Select the preferred path based on the directory priority.
    def score(p: Path) -> int:
        s = str(p).replace("\\", "/")
        for i, key in enumerate(PRIORITY_DIRS):
            if f"/{key}/" in s:
                return 100 - i  
        return 0
    return sorted(paths, key=score, reverse=True)[0]

def read_eval_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        return list(r)

def main():
    csv_path = Path("error_injection_log_eval.csv")        
    gt_root  = Path("ground_truth")                       
    out_root = Path("models_with_error") / "eval_balanced"
    out_root.mkdir(parents=True, exist_ok=True)

    out_log_csv = Path("error_injection_log_eval_balanced.csv") 
    
    def ensure_log_csv_exists(path: Path):
        if path.exists():
            return
        src = Path("error_injection_log_eval.csv")
        with open(src, "r", encoding="utf-8-sig", newline="") as fsrc:
            header_line = fsrc.readline().lstrip("\ufeff")
        with open(path, "w", encoding="utf-8-sig", newline="") as fdst:
            fdst.write(header_line)

    ensure_log_csv_exists(out_log_csv)

    inject_script = Path("inject_errors.py") 
    samples_per_error = 30
    seed_base = random.randint(0, 10**9)
    # ==================

    rows = read_eval_csv(csv_path)
    gt_index = index_ground_truth(gt_root)

    # Select the preferred path based on the directory priority.
    by_err: dict[str, list[str]] = {k: [] for k in ERR_MAP.keys()}

    for row in rows:
        model = (row.get(CSV_MODEL_COL) or "").strip()
        if not model:
            continue

        for csv_col in ERR_MAP.keys():
            v = (row.get(csv_col) or "").strip().lower()
            if v == "x":
                by_err[csv_col].append(model)

    # Remove duplicates while preserving order, then keep up to the target sample size.
    def dedup_keep_order(xs: list[str]) -> list[str]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    selected: dict[str, list[str]] = {}
    for csv_col, models in by_err.items():
        uniq = dedup_keep_order(models)
        if len(uniq) < samples_per_error:
            print(f"WARNING: {csv_col}: only {len(uniq)} candidates in eval csv (need {samples_per_error})")
        selected[csv_col] = uniq[:samples_per_error]
        print(f"OK: {csv_col}: using {len(selected[csv_col])} models")

    # injection 
    for csv_col, model_names in selected.items():
        force_err = ERR_MAP[csv_col]
        err_out_dir = out_root / force_err.replace(" ", "_").replace("->", "to").lower()
        err_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {csv_col} -> force_error='{force_err}' ===")

        for i, model_name in enumerate(model_names, start=1):
            candidates = gt_index.get(model_name, [])
            if not candidates:
                print(f"FAIL: ground_truth not found: {model_name}")
                continue

            in_xml = choose_best_path(candidates)
            if len(candidates) > 1:
                print(f"WARNING: duplicate name: {model_name} -> chosen {in_xml}")

            seed = (seed_base + hash((force_err, model_name, i))) % (2**31 - 1)

            cmd = [
                "python", str(inject_script),
                "--in_xml", str(in_xml),
                "--out_dir", str(err_out_dir),
                "--n_errors", "1",                 
                "--seed", str(seed),
                "--log_csv", str(out_log_csv),
                "--force_error", force_err,        
            ]

            print(f"[{i}/{len(model_names)}] {model_name}  ({in_xml})")
            res = subprocess.run(cmd, capture_output=True, text=True)

            if res.returncode != 0:
                print("  FAIL")
                print("  STDOUT:", (res.stdout or "")[-1200:])
                print("  STDERR:", (res.stderr or "")[-1200:])
                continue
            else:
                out = (res.stdout or "").strip()
                if out:
                    print(out)

    print("\nDONE.")
    print("Outputs under:", out_root)
    print("Log CSV:", out_log_csv)

if __name__ == "__main__":
    main()