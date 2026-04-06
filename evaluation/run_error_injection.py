from pathlib import Path
import secrets, subprocess

def main():
    in_dir = Path("ground_truth/sapsam")
    out_dir = Path("models_with_error/sapsam")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_errors = 3
    log_csv = "error_injection_log.csv"

    xml_files = []
    for k in range(41, 91):  # 41..90
        p = in_dir / f"{k}_ground_truth.xml"
        if p.exists():
            xml_files.append(p)
        else:
            print("⚠️ missing:", p)

    if not xml_files:
        print("No matching XML files found.")
        return

    for i, xml_path in enumerate(xml_files, start=1):
        file_seed = secrets.randbelow(2**31 - 1)

        cmd = [
            "python", "inject_errors.py",
            "--in_xml", str(xml_path),
            "--out_dir", str(out_dir),
            "--n_errors", str(n_errors),
            "--seed", str(file_seed),
            "--log_csv", log_csv,
        ]

        print(f"[{i}/{len(xml_files)}] Running:", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)

        if res.returncode != 0:
            print("❌ Failed for:", xml_path.name)
            print("STDOUT:\n", res.stdout[-800:])
            print("STDERR:\n", res.stderr[-800:])
        else:
            print(res.stdout.strip())

    print("\nDone. Output dir:", out_dir)

if __name__ == "__main__":
    main()