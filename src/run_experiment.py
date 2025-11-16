.py
from __future__ import annotations
import argparse, yaml, os, json, random
from tqdm import tqdm
from src.utils.io import jsonl_read, ensure_dir, now_iso, safe_filename
from src.providers import LLMClient

def iter_enabled_providers(cfg):
    for prov, meta in cfg["providers"].items():
        if meta.get("enabled"):
            yield prov, meta["model"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompts", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    prompts_path = args.prompts or os.path.join(cfg["io"]["prompts_out"], "prompts.jsonl")
    prompts = jsonl_read(prompts_path)
    out_dir = ensure_dir(cfg["io"]["results_out"])
    samples = int(cfg["samples_per_prompt"])
    temperature = float(cfg["temperature"])
    top_p = float(cfg["top_p"])
    max_tokens = int(cfg["max_tokens"])
    base_seed = int(cfg["seed"])

    for prov, model in iter_enabled_providers(cfg):
        client = LLMClient(prov, model)
        outfile = os.path.join(out_dir, f"responses_{safe_filename(prov)}_{safe_filename(model)}.jsonl")
        with open(outfile, "a", encoding="utf-8") as f:
            for pr in tqdm(prompts, desc=f"{prov}/{model}"):
                for k in range(samples):
                    seed = base_seed + k
                    res = client.completions(pr["prompt_text"], temperature, top_p, max_tokens, seed)
                    row = {
                        "timestamp": now_iso(),
                        "provider": prov,
                        "model": model,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens,
                        "seed": seed,
                        "prompt_id": pr["prompt_id"],
                        "hypothesis": pr["hypothesis"],
                        "condition": pr["condition"],
                        "prompt_text": pr["prompt_text"],
                        "input_blob": pr["input_blob"],
                        "response_text": res["text"],
                        "usage": res.get("usage", {}),
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {outfile}")

if __name__ == "__main__":
    main()
