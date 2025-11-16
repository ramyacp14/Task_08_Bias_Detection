from __future__ import annotations
import argparse, yaml, pandas as pd, json, os
from jinja2 import Template
from src.utils.io import ensure_dir, now_iso
from typing import Dict, Any

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "player_id" in df.columns, "CSV must have player_id"
    return df

def stats_blob(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c != "player_id"]
    lines = ["Player statistics (anonymized):"]
    for _, r in df.iterrows():
        parts = [f"{c} {int(r[c]) if float(r[c]).is_integer() else round(float(r[c]),2)}" for c in cols]
        lines.append(f"- {r['player_id']}: " + ", ".join(parts))
    return "\n".join(lines)

def demographics_blob(demo: pd.DataFrame | None) -> str:
    if demo is None: return ""
    cols = [c for c in demo.columns if c != "player_id"]
    lines = []
    for _, r in demo.iterrows():
        parts = [f"{c} {r[c]}" for c in cols]
        lines.append(f"{r['player_id']} ({', '.join(parts)})")
    return ", ".join(lines)

def render_template(path: str, **kwargs) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return Template(f.read()).render(**kwargs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    df = load_csv(cfg["data"]["players_csv"])
    demo = None
    if os.path.exists(cfg["data"].get("demographics_csv", "")):
      demo = pd.read_csv(cfg["data"]["demographics_csv"])

    stats = stats_blob(df)
    demo_blob = demographics_blob(demo)

    prompt_defs = [
        ("H1_positive", "prompts/framing_positive.md", dict(stats_blob=stats)),
        ("H1_negative", "prompts/framing_negative.md", dict(stats_blob=stats)),
        ("H2_neutral",  "prompts/base_neutral.md",    dict(stats_blob=stats)),
        ("H2_demo",     "prompts/base_demographic.md",dict(stats_blob=stats, demographics_blob=demo_blob)),
        ("H3_confirm",  "prompts/confirmation_bias.md", dict(stats_blob=stats, weak_hypothesis="Player_A underperformed due to turnovers")),
    ]

    out_dir = ensure_dir(cfg["io"]["prompts_out"])
    rows = []
    for pid, tpath, ctx in prompt_defs:
        ptxt = render_template(tpath, **ctx)
        rows.append({
            "timestamp": now_iso(),
            "prompt_id": pid,
            "hypothesis": pid.split("_")[0],
            "condition": pid.split("_")[1],
            "prompt_text": ptxt,
            "input_blob": {"stats_blob": stats},
        })
    out_path = os.path.join(out_dir, "prompts.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path} ({len(rows)} prompts)")

if __name__ == "__main__":
    main()
