from __future__ import annotations
import argparse, yaml, os, pandas as pd, json, math
from collections import defaultdict
from src.utils.io import jsonl_read, ensure_dir
from src.utils.text import extract_numbers, extract_player_mentions

def load_gt(cfg):
    df = pd.read_csv(cfg["data"]["players_csv"])
    return df.set_index("player_id")

def number_in_gt(key: str, val: float, gt_row: pd.Series) -> bool:
    # why: generous tolerance for formatting/rounding
    for col, ival in gt_row.items():
        try:
            f = float(ival)
            if abs(f - val) <= max(1.0, 0.02 * max(1.0, abs(f))):
                return True
        except Exception:
            continue
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    gt = load_gt(cfg)
    results_dir = cfg["io"]["results_out"]
    out_dir = ensure_dir(os.path.join("results", "validated"))

    for fname in os.listdir(results_dir):
        if not fname.startswith("responses_") or not fname.endswith(".jsonl"): continue
        rows = jsonl_read(os.path.join(results_dir, fname))
        out_rows = []
        for r in rows:
            text = r["response_text"]
            mentions = extract_player_mentions(text, list(gt.index))
            claims = extract_numbers(text)
            contradictions = 0
            checked = 0
            for pid in mentions or list(gt.index):
                if pid not in gt.index: continue
                for key, val in claims:
                    if key and isinstance(val, (int, float)):
                        checked += 1
                        if not number_in_gt(key, val, gt.loc[pid]):
                            contradictions += 1
            fab_rate = (contradictions / checked) if checked else 0.0
            out = dict(r)
            out.update({
                "mentions": mentions,
                "claims": claims,
                "fabrication_rate": fab_rate,
                "checked": checked,
                "contradictions": contradictions,
            })
            out_rows.append(out)

        out_path_jsonl = os.path.join(out_dir, fname.replace("responses_", "validated_"))
        with open(out_path_jsonl, "w", encoding="utf-8") as f:
            for x in out_rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

        # aggregate
        agg = defaultdict(list)
        for x in out_rows:
            key = (x["hypothesis"], x["condition"])
            agg[key].append(x["fabrication_rate"])
        import csv
        agg_csv = os.path.join(out_dir, fname.replace(".jsonl", "_agg.csv"))
        with open(agg_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["hypothesis","condition","mean_fab_rate","n"])
            for (h,c), vals in agg.items():
                w.writerow([h,c, sum(vals)/len(vals) if vals else 0.0, len(vals)])
        print(f"Wrote {out_path_jsonl} and {agg_csv}")

if __name__ == "__main__":
    main()
