from __future__ import annotations
import argparse, yaml, os, json, pandas as pd, numpy as np, matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import statsmodels.api as sm
from src.utils.io import jsonl_read, ensure_dir
from src.utils.text import extract_player_mentions, sentiment_scores, recommendation_types

def load_all(results_dir: str):
    frames = []
    for fname in os.listdir(results_dir):
        if fname.startswith("responses_") and fname.endswith(".jsonl"):
            rows = jsonl_read(os.path.join(results_dir, fname))
            frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def sentiment_df(df: pd.DataFrame) -> pd.DataFrame:
    s = df["response_text"].apply(sentiment_scores).apply(pd.Series)
    s.columns = [f"sent_{c}" for c in s.columns]
    return pd.concat([df, s], axis=1)

def mentions_df(df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["mentions"] = df["response_text"].apply(lambda t: extract_player_mentions(t, ids))
    for pid in ids:
        df[f"m_{pid}"] = df["mentions"].apply(lambda ms: int(pid in ms))
    return df

def rec_types_df(df: pd.DataFrame) -> pd.DataFrame:
    kcols = ["defensive","offensive","individual","team"]
    df = df.copy()
    for k in kcols:
        df[f"rec_{k}"] = df["response_text"].apply(lambda t: recommendation_types(t)[k])
    return df

def plot_bar(series: pd.Series, title: str, outpath: str):
    plt.figure()
    series.plot(kind="bar")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    results_dir = cfg["io"]["results_out"]
    analysis_dir = ensure_dir(os.path.join(cfg["io"]["analysis_out"]))
    figs_dir = ensure_dir(os.path.join(analysis_dir, "figures"))
    tabs_dir = ensure_dir(os.path.join(analysis_dir, "tables"))

    raw = load_all(results_dir)
    if raw.empty:
        print("No responses found.")
        return

    # IDs from data
    ids = pd.read_csv(cfg["data"]["players_csv"])["player_id"].tolist()

    df = rec_types_df(mentions_df(sentiment_df(raw), ids))

    # 1) Which entities are mentioned per condition?
    mention_cols = [c for c in df.columns if c.startswith("m_")]
    mention_summary = df.groupby(["hypothesis","condition"])[mention_cols].mean()
    mention_summary.to_csv(os.path.join(tabs_dir, "mentions_rate.csv"))

    for col in mention_cols:
        ser = mention_summary[col]
        plot_bar(ser, f"Mention rate: {col[2:]}", os.path.join(figs_dir, f"mention_{col[2:]}.png"))

    # 2) Sentiment per condition
    sent_summary = df.groupby(["hypothesis","condition"])[["sent_pos","sent_neg","sent_neu","sent_compound"]].mean()
    sent_summary.to_csv(os.path.join(tabs_dir, "sentiment_by_condition.csv"))

    # 3) Recommendation types counts
    rec_cols = [c for c in df.columns if c.startswith("rec_")]
    rec_summary = df.groupby(["hypothesis","condition"])[rec_cols].mean()
    rec_summary.to_csv(os.path.join(tabs_dir, "recommendation_types.csv"))

    # 4) Statistical tests: example chi-square on Player_A mentions across H1 pos/neg
    tests = []
    for pid in ids:
        sub = df[df["hypothesis"]=="H1"]
        if sub.empty: continue
        pos = sub[sub["condition"]=="positive"][f"m_{pid}"].sum()
        neg = sub[sub["condition"]=="negative"][f"m_{pid}"].sum()
        pos_n = len(sub[sub["condition"]=="positive"])
        neg_n = len(sub[sub["condition"]=="negative"])
        table = np.array([[pos, pos_n-pos],[neg, neg_n-neg]])
        if table.min() >= 0:
            chi2, p, *_ = chi2_contingency(table)
            tests.append({"test":"chi2_mentions","player_id":pid,"chi2":chi2,"p":p})
    pd.DataFrame(tests).to_csv(os.path.join(tabs_dir, "tests_chi2_mentions.csv"), index=False)

    # 5) t-test on compound sentiment between pos/neg for H1
    sub = df[df["hypothesis"]=="H1"]
    if not sub.empty and "sent_compound" in sub:
        pos_vals = sub[sub["condition"]=="positive"]["sent_compound"].tolist()
        neg_vals = sub[sub["condition"]=="negative"]["sent_compound"].tolist()
        if len(pos_vals) > 1 and len(neg_vals) > 1:
            t,p = ttest_ind(pos_vals, neg_vals, equal_var=False)
            pd.DataFrame([{"test":"t_sentiment","t":t,"p":p,"n_pos":len(pos_vals),"n_neg":len(neg_vals)}]).to_csv(
                os.path.join(tabs_dir,"tests_t_sentiment.csv"), index=False
            )

    # 6) Simple logistic regression: mention Player_A ~ framing (pos=1/neg=0)
    try:
        sub = df[df["hypothesis"]=="H1"].copy()
        sub = sub[sub["condition"].isin(["positive","negative"])]
        sub["pos"] = (sub["condition"]=="positive").astype(int)
        sub["y"] = sub["m_Player_A"]
        X = sm.add_constant(sub[["pos"]])
        model = sm.Logit(sub["y"], X).fit(disp=False)
        with open(os.path.join(tabs_dir,"logit_player_A.txt"),"w") as f:
            f.write(model.summary().as_text())
    except Exception as e:
        pass  # why: optional if sample too small

    # 7) Aggregate CSV for dashboard/report
    df.to_csv(os.path.join(tabs_dir, "responses_enriched.csv"), index=False)

    # 8) Mini summary
    with open(os.path.join(analysis_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Auto Summary\n\n")
        f.write("- Mention rates, sentiment, recommendation types, and statistical tests saved in tables/.\n")
        f.write("- Figures in figures/.\n")

    print("Analysis complete.")

if __name__ == "__main__":
    main()
