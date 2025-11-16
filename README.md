````markdown
# Task_08_Bias_Detection — LLM Bias in Data Narratives

---

## 🚀 Quick Start

```bash
# 1) Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon

# 2) Configure providers/models
cp config.yaml config.local.yaml 
# Edit model names, enable providers, set seeds/samples

# 3) Generate prompts
python src/experiment_design.py --config config.yaml

# 4) Run models (set API keys first)
export OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY / GOOGLE_API_KEY
python src/run_experiment.py --config config.yaml

# 5) Validate numeric claims vs ground truth
python src/validate_claims.py --config config.yaml

# 6) Analyze bias patterns (tables + figures + summary)
python src/analyze_bias.py --config config.yaml
````

---

## 📁 Project Structure

```
Task_08_Bias_Detection/
├─ data/                    # anonymized CSVs (no PII)
│  ├─ players_2024.csv
│  └─ demographics_anon.csv 
├─ prompts/
│  ├─ *.md                  # templates (minimal edits across conditions)
│  └─ generated/prompts.jsonl
├─ results/
│  ├─ raw/responses_*.jsonl # prompts+responses (large; .gitignored)
│  └─ validated/            # fabrication checks & aggregates
├─ analysis/
│  ├─ tables/*.csv          # mentions/sentiment/recs/tests/enriched
│  ├─ figures/*.png         # plots
│  └─ summary.md            # auto-written brief summary
├─ src/
│  ├─ experiment_design.py  # builds prompt variants
│  ├─ run_experiment.py     # executes LLMs & logs JSONL
│  ├─ validate_claims.py    # unsupported numeric claim detection
│  ├─ analyze_bias.py       # quant+qual analysis + stats + plots
│  └─ utils/                # helpers (io, text)
├─ REPORT.md                # final report (fill after running)
├─ README.md
├─ config.yaml              # seeds, models, paths, params
├─ requirements.txt
└─ .gitignore               # excludes large results
```

---

## 🧪 Hypotheses & Prompt Conditions

* **H1 (Framing):** *developing* vs *struggling* changes recommendations.
* **H2 (Demographics):** demographic mentions change coaching targets.
* **H3 (Opportunity vs Post-mortem):** “what opportunities exist” vs “what went wrong.”
* **H4 (Confirmation):** priming with a weak hypothesis increases agreement.
* **H5 (Selection):** which players/stats get emphasized varies by condition.

Templates: `prompts/*.md` → rendered to `prompts/generated/prompts.jsonl`.

---

## 📊 Datasets & Ground Truth

* Use anonymized stats (e.g., `Player_A`). **No PII.**
* Example schema: `player_id, goals, assists, turnovers, minutes, ...`
* Optional synthetic/public demographics: `player_id, class, ...`
* Ground truth = direct numeric computations from `data/*.csv`.
* If starting from a PDF, transcribe the relevant totals into CSV (kept in `data/`).

---

## ⚙️ Configuration & API Keys

* `config.yaml` controls:

  * `providers.openai|anthropic|gemini.enabled`
  * `model` names (e.g., `gpt-4o-mini`, `claude-3-5-sonnet-latest`, `gemini-1.5-pro`)
  * `seed`, `samples_per_prompt`, `temperature`, `top_p`, `max_tokens`
  * paths under `io.*`
* Export keys (only those you use): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`.

---

## 🧵 Pipeline (Design → Run → Validate → Analyze)

1. **Design** → `experiment_design.py`

   * Outputs: `prompts/generated/prompts.jsonl` with `prompt_id, hypothesis, condition, prompt_text`.

2. **Run** → `run_experiment.py`

   * Outputs: `results/raw/responses_<provider>_<model>.jsonl` (3–5 samples/condition).

3. **Validate** → `validate_claims.py`

   * Extracts numeric claims; compares with ground truth (±2% tolerance).
   * Outputs: `results/validated/validated_*.jsonl`, `*_agg.csv` with **fabrication_rate**.

4. **Analyze** → `analyze_bias.py`

   * Computes **mentions** (per player), **sentiment** (VADER), **recommendation types** (rule-based), **stats tests**:

     * Chi-square (mentions), t-test/Mann-Whitney (sentiment), simple logistic (optional).
   * Outputs:

     * Tables → `analysis/tables/*`
     * Figures → `analysis/figures/*`
     * Summary → `analysis/summary.md`

---

## 📦 Outputs:

* `analysis/tables/ - contains period related tables
* `analysis/figures/*.png` — per-player bars, differentials, efficiencies.
* `analysis/summary.md` — concise findings.

---

## 📥 Current ChatGPT-Session Artifacts (Ready to Download Here)

* **Bundle:** [analysis_bundle.zip](sandbox:/mnt/data/analysis/analysis_bundle.zip)
* **Tables:**

  * [period_goals.csv](sandbox:/mnt/data/analysis/tables/period_goals.csv)
  * [period_saves.csv](sandbox:/mnt/data/analysis/tables/period_saves.csv)
  * [period_shots.csv](sandbox:/mnt/data/analysis/tables/period_shots.csv)
  * [period_sog.csv](sandbox:/mnt/data/analysis/tables/period_sog.csv)
  * [team_efficiency.csv](sandbox:/mnt/data/analysis/tables/team_efficiency.csv)
* **Figures:**

  * [efficiency_comparison.png](sandbox:/mnt/data/analysis/figures/efficiency_comparison.png)
  * [goal_diff_by_period.png](sandbox:/mnt/data/analysis/figures/goal_diff_by_period.png)
  * [goals_by_period.png](sandbox:/mnt/data/analysis/figures/goals_by_period.png)
* **Summary:**

  * [summary.md](sandbox:/mnt/data/analysis/summary.md)

> To regenerate locally: place your CSVs under `data/`, then run **Validate** and **Analyze**.

---

## 🧯 Ethics & Privacy

* **No PII** anywhere (reports, code, repo).
* Demographics must be synthetic/public or permissioned.
* Avoid stereotype amplification; report **null** results.
