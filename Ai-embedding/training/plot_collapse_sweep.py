"""
plot_collapse_sweep.py — collapse 스윕 런 비교 플로터/요약.

result_cs_collapse* 디렉토리들의 history.json 을 읽어:
  - 4패널 겹쳐그리기: PR(effective rank) / LOO Top-3 / cov / nt(NT-Xent)  — epoch 축
  - 런별 best 요약표: best Top-3 & 그 시점 PR/ep, 최종 ep, sweet-spot

순수 CPU/matplotlib — 진행 중인 학습(MPS)과 비경합.

Usage:
    python plot_collapse_sweep.py
    python plot_collapse_sweep.py --glob 'result_cs_collapse*' --out diagnostics/collapse_sweep.png
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_runs(pattern: str):
    runs = {}
    for d in sorted(glob.glob(pattern)):
        hp = Path(d) / "history.json"
        if not hp.is_file():
            continue
        try:
            h = json.load(open(hp))
        except Exception:
            continue
        if not h:
            continue
        # 런 이름: result_cs_collapse → 'baseline', _cov010 → 'cov010'
        name = Path(d).name.replace("result_cs_collapse", "").lstrip("_") or "baseline"
        runs[name] = h
    return runs


def summarize(runs):
    rows = []
    for name, h in runs.items():
        best = max(h, key=lambda e: e.get("rec_top3", 0))
        last = h[-1]
        rows.append({
            "run": name,
            "best_top3": best.get("rec_top3", 0),
            "best_ep": best.get("ep", 0),
            "pr_at_best": best.get("pr", 0),
            "last_ep": last.get("ep", 0),
            "last_pr": last.get("pr", 0),
            "last_top3": last.get("rec_top3", 0),
        })
    rows.sort(key=lambda r: -r["best_top3"])
    return rows


def print_table(rows):
    print(f"\n{'run':<16}{'best Top-3':>11}{'@ep':>6}{'PR@best':>9}"
          f"{'last ep':>9}{'last PR':>9}{'last Top3':>11}")
    print("─" * 71)
    for r in rows:
        star = "  ★" if r is rows[0] else ""
        print(f"{r['run']:<16}{r['best_top3']:>10.1%}{r['best_ep']:>6}"
              f"{r['pr_at_best']:>9.1f}{r['last_ep']:>9}{r['last_pr']:>9.1f}"
              f"{r['last_top3']:>10.1%}{star}")
    print()


def plot(runs, out: Path):
    panels = [
        ("pr",            "effective rank PR / 256", None),
        ("rec_top3",      "LOO Top-3",               "{:.0%}"),
        ("cov",           "covariance loss",         None),
        ("nt",            "NT-Xent loss",            None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    cmap = plt.cm.tab10
    for ai, (key, title, yfmt) in enumerate(panels):
        ax = axes[ai // 2][ai % 2]
        for ci, (name, h) in enumerate(runs.items()):
            xs = [e["ep"] for e in h if key in e]
            ys = [e[key] for e in h if key in e]
            if not xs:
                continue
            ax.plot(xs, ys, marker=".", ms=4, label=name, color=cmap(ci % 10))
        ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(alpha=0.25)
        if yfmt:
            ax.yaxis.set_major_formatter(lambda v, _, f=yfmt: f.format(v))
        ax.legend(fontsize=8)
    fig.suptitle("collapse 하이퍼파라미터 스윕 비교", fontsize=13)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="result_cs_collapse*")
    p.add_argument("--out",  default="diagnostics/collapse_sweep.png")
    args = p.parse_args()

    runs = load_runs(args.glob)
    if not runs:
        print(f"[!] '{args.glob}' 패턴에서 history.json 가진 런 없음")
        return
    print(f"[runs] {', '.join(f'{n}({len(h)}pt)' for n, h in runs.items())}")
    rows = summarize(runs)
    print_table(rows)
    plot(runs, Path(args.out))


if __name__ == "__main__":
    main()
