# tools/mailabs_to_pairs_v2.py
import argparse, pathlib, csv, re
from typing import Optional

def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t.strip())
    return t

def find_wav(wavroot: pathlib.Path, name: str) -> Optional[pathlib.Path]:
    # name may come without extension
    p = wavroot / name
    if p.suffix.lower() != ".wav":
        p = p.with_suffix(".wav")
    if p.exists():
        return p
    # fallback: search recursively
    hits = list(wavroot.rglob(p.name))
    return hits[0] if hits else None

def parse_metadata_lines(meta_path: pathlib.Path):
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # M-AILABS is typically pipe-delimited
            if "|" in line:
                parts = line.split("|")
                if len(parts) >= 3:
                    fid = parts[0]
                    # parts[1] is speaker (e.g., delgado_f); keep it if you ever need mspeaker
                    text = "|".join(parts[2:])  # text may contain pipes
                elif len(parts) == 2:
                    fid, text = parts
                else:
                    # single column? skip
                    continue
                yield fid, clean_text(text)
            else:
                # Very rare: CSV with comma/tsv — try to parse as "path,text"
                try:
                    # crude split on first comma
                    fid, text = line.split(",", 1)
                    yield fid, clean_text(text)
                except ValueError:
                    continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="Path to M-AILABS metadata.csv")
    ap.add_argument("--wav-root24k", required=True, help="Root to RESAMPLED wavs (wavs24k)")
    ap.add_argument("--out", default="data/angelina/pairs.csv")
    args = ap.parse_args()

    wavroot = pathlib.Path(args.wav_root24k)
    meta = pathlib.Path(args.meta)
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    total = found = skipped_missing = skipped_empty = 0
    rows = []

    for fid, text in parse_metadata_lines(meta):
        total += 1
        if not text:
            skipped_empty += 1
            continue
        name = pathlib.Path(fid).name  # keep just filename-ish
        wav = find_wav(wavroot, name)
        if not wav:
            skipped_missing += 1
            continue
        rows.append((wav.as_posix(), text))
        found += 1

    with outp.open("w", encoding="utf-8", newline="") as g:
        w = csv.writer(g)
        w.writerow(["path", "text"])
        w.writerows(rows)

    print(f"Scanned: {total} | wrote: {found} | missing_wav: {skipped_missing} | empty_text: {skipped_empty}")
    print(f"Wrote {found} rows -> {outp}")

if __name__ == "__main__":
    main()
