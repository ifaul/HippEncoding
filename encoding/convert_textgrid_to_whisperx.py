#!/usr/bin/env python3
"""
Convert Praat TextGrid files to WhisperX-style JSON:
{
  "word_segments": [
    {"word": "Hello", "start": 0.0, "end": 0.12},
    ...
  ]
}

Usage:
  pip install praatio
  python convert_textgrid_to_whisperx.py /path/to/file.TextGrid --outdir data/stimuli/whisperx

You may pass a directory; all .TextGrid files inside will be processed.
"""
import argparse
import json
from pathlib import Path

try:
    from praatio import textgrid as praatio_textgrid
    from praatio.utilities.errors import TextgridStateError as PraatIOTextgridStateError
except Exception:
    praatio_textgrid = None
    PraatIOTextgridStateError = (UnicodeDecodeError, ValueError, TypeError)

from ridge_utils.textgrid import TextGrid

EPSILON = 0.01  # for point tiers -> small interval

def extract_intervals_from_tier(tier):
    """
    Return a list of (start, end, label) from a tier. Handles both interval
    and point tiers; point tiers are converted to small intervals.
    """
    entries = []

    if hasattr(tier, "entryList"):
        for entry in tier.entryList:
            if len(entry) == 3:
                start, end, label = entry
            elif len(entry) == 2:
                time, label = entry
                start = max(0.0, time - EPSILON / 2)
                end = time + EPSILON / 2
            else:
                continue
            label = "" if label is None else str(label).strip()
            entries.append((float(start), float(end), label))
        return entries

    if hasattr(tier, "simple_transcript"):
        for start, end, label in tier.simple_transcript:
            label = "" if label is None else str(label).strip()
            entries.append((float(start), float(end), label))
        return entries

    return entries


def choose_tier(tg_obj):
    if hasattr(tg_obj, "tierNameList"):
        tier_names = list(tg_obj.tierNameList)
        def tier_getter(name):
            return tg_obj.getTier(name)
    elif hasattr(tg_obj, "tierNames"):
        tier_names = list(tg_obj.tierNames)
        def tier_getter(name):
            return tg_obj.getTier(name)
    elif hasattr(tg_obj, "tiers"):
        tier_names = [getattr(tier, "nameid", getattr(tier, "name", "")) for tier in tg_obj.tiers]
        def tier_getter(name):
            return next((tier for tier in tg_obj.tiers if getattr(tier, "nameid", getattr(tier, "name", "")) == name), None)
    else:
        return None

    for name in tier_names:
        if "word" in str(name).lower():
            tier = tier_getter(name)
            if tier is not None:
                return tier

    for name in tier_names:
        tier = tier_getter(name)
        if tier is None:
            continue
        if getattr(tier, "entryList", None) and any(len(e) == 3 for e in tier.entryList):
            return tier
        if getattr(tier, "simple_transcript", None):
            return tier

    return None


def load_textgrid(in_path: Path):
    if praatio_textgrid is not None:
        try:
            return praatio_textgrid.openTextgrid(str(in_path), includeEmptyIntervals=True)
        except Exception as exc:
            print(f"PraatIO could not load {in_path}: {exc}. Trying fallback parser.")

    return TextGrid.load(str(in_path))


def convert_file(in_path: Path, out_dir: Path, narrative_override: str | None = None):
    try:
        tg_obj = load_textgrid(in_path)
    except Exception as exc:
        print(f"Skipping malformed TextGrid {in_path}: {exc}")
        return None

    tier = choose_tier(tg_obj)
    if tier is None:
        print(f"Skipping {in_path}: no usable interval tier found")
        return None

    entries = extract_intervals_from_tier(tier)
    # filter out empty labels
    entries = [(s, e, lab) for (s, e, lab) in entries if lab != ""]
    # ensure end > start
    cleaned = []
    for s, e, lab in entries:
        if e <= s:
            e = s + max(EPSILON, 1e-3)
        cleaned.append((s, e, lab))
    # sort by start time
    cleaned.sort(key=lambda x: x[0])

    word_segments = [{"word": lab, "start": s, "end": e} for (s, e, lab) in cleaned]

    if narrative_override:
        narrative = narrative_override
    else:
        # try to infer narrative: strip extension and optional "_audio"
        stem = in_path.stem
        narrative = stem
        if stem.endswith("_audio"):
            narrative = stem[: -len("_audio")]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{narrative}_audio.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"word_segments": word_segments}, f, indent=2, ensure_ascii=False)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="TextGrid file or directory containing TextGrid files")
    ap.add_argument("--outdir", "-o", "--o", default="derivatives/TextGrids", help="Output directory")
    ap.add_argument("--story", "-n", default=None, help="Optional story name override")
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.outdir)
    if inp.is_dir():
        tg_files = sorted(p for p in inp.iterdir() if p.suffix.lower() in (".textgrid",))
        if not tg_files:
            print(f"No .TextGrid files found in {inp}")
            return
        for tg_file in tg_files:
            out = convert_file(tg_file, out_dir)
            if out is not None:
                print("Wrote", out)
    elif inp.is_file():
        out = convert_file(inp, out_dir, narrative_override=args.story)
        if out is not None:
            print("Wrote", out)
    else:
        print("Input path not found:", inp)

if __name__ == "__main__":
    main()