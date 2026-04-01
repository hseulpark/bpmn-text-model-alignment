import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import requests


API_URL = "https://autobpmn.ai/llm/text/llm/"  # trailing slash OK


def call_autobpmn(xml_path: Path, llm: str, timeout_sec: int = 60) -> Tuple[Optional[str], str]:
    """
    Returns (output_text_or_none, note).
    note is a short status/debug message.
    """
    try:
        with xml_path.open("rb") as f:
            files = {
                "rpst_xml": (xml_path.name, f, "text/xml"),
                "llm": (None, llm),  # multipart field
            }
            headers = {
                "Accept": "application/json",
                "User-Agent": "autobpmn-batch-client",
            }
            resp = requests.post(API_URL, files=files, headers=headers, timeout=timeout_sec)

        if resp.status_code != 200:
            return None, (
                f"HTTP {resp.status_code} "
                f"content_type={resp.headers.get('Content-Type')} "
                f"body_head={(resp.text or '')[:300]!r}"
            )

        try:
            payload = resp.json()
        except Exception:
            return None, f"HTTP 200 but JSON parse failed. body_head={(resp.text or '')[:300]!r}"

        out = (payload.get("output_text") or "").strip()
        if not out:
            return None, "HTTP 200 but empty output_text"
        return out, "OK"

    except Exception as e:
        return None, f"request failed: {e}"


def mirror_out_path(in_dir: Path, xml_path: Path, out_dir: Path, suffix: str) -> Path:
    """
    Keep relative structure from in_dir to out_dir.
    Example:
      python generate_text.py \
        --in_dir ./ground_truth/pet \
        --out_dir ./process_description \
        --sleep 0.25
      -> repo/texts/a/b/c.xml.autobpmn.txt (or c.autobpmn.txt)
    """
    rel = xml_path.relative_to(in_dir)
    # choose output filename
    # option A: keep xml extension in name
    # out_name = rel.name + suffix
    # option B: replace xml with suffix
    out_name = rel.with_suffix("").name + suffix
    return out_dir / rel.parent / out_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory to scan recursively for *.xml")
    ap.add_argument("--out_dir", required=True, help="Directory to write *.autobpmn.txt outputs")
    ap.add_argument("--llm", default="gpt-4o")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between requests")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    ap.add_argument("--suffix", default=".autobpmn.txt", help="Output file suffix")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N xml files (0 = no limit)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(in_dir.rglob("*.xml"))
    if args.limit and args.limit > 0:
        xml_files = xml_files[: args.limit]

    if not xml_files:
        print("No .xml files found under:", in_dir)
        return

    ok = 0
    skip = 0
    fail = 0

    for i, xml_path in enumerate(xml_files, start=1):
        out_path = mirror_out_path(in_dir, xml_path, out_dir, args.suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.overwrite:
            print(f"[{i}/{len(xml_files)}] SKIP (exists): {xml_path} -> {out_path}")
            skip += 1
            continue

        print(f"[{i}/{len(xml_files)}] CALL: {xml_path}")
        text, note = call_autobpmn(xml_path, llm=args.llm, timeout_sec=args.timeout)

        if text is None:
            print(f"  ❌ FAIL: {note}")
            fail += 1
            # write a .err.json next to output for debugging
            err_path = out_path.with_suffix(out_path.suffix + ".err.json")
            err_payload = {
                "xml": str(xml_path),
                "note": note,
                "llm": args.llm,
                "api_url": API_URL,
            }
            err_path.write_text(json.dumps(err_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            out_path.write_text(text, encoding="utf-8")
            print(f"  ✅ OK -> {out_path}")
            ok += 1

        if args.sleep > 0:
            time.sleep(args.sleep)

    print("\nDONE")
    print("  OK  :", ok)
    print("  SKIP:", skip)
    print("  FAIL:", fail)
    print("  OUT :", out_dir)


if __name__ == "__main__":
    main()