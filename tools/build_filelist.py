# tools/build_filelist_strict.py
import csv, argparse, pathlib, random, soundfile as sf

p = argparse.ArgumentParser()
p.add_argument("--pairs", required=True)     # CSV with columns: path,text
p.add_argument("--out", default="filelists")
p.add_argument("--val_ratio", type=float, default=0.02)
p.add_argument("--speaker", default="spk1")
p.add_argument("--min_sec", type=float, default=1.0)   # optional duration filters
p.add_argument("--max_sec", type=float, default=14.0)
args = p.parse_args()

pairs = []
with open(args.pairs, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        wav = pathlib.Path(row["path"])
        text = (row["text"] or "").strip()
        if not wav.exists() or not text:
            continue
        try:
            info = sf.info(str(wav))
            dur = info.frames / info.samplerate
        except Exception:
            continue
        if not (args.min_sec <= dur <= args.max_sec):
            continue
        pairs.append(f"{wav.as_posix()}|{text}|{args.speaker}")

random.shuffle(pairs)
n_val = max(1, int(len(pairs) * args.val_ratio))
val = pairs[:n_val]
train = pairs[n_val:]

out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
(out / "train.txt").write_text("\n".join(train), encoding="utf-8")
(out / "val.txt").write_text("\n".join(val), encoding="utf-8")

print(f"Wrote {len(train)} train, {len(val)} val to {out}")
