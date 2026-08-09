#!/usr/bin/env python3
"""
compare_sweeps.py - Aggregate MOE Pareto insights across multiple benchmark sweeps.

Usage:
  python3 compare_sweeps.py --out report.html SWEEP_LABEL=SweepDir [SWEEP_LABEL2=Dir ...]

Reads profile_export_aiperf.json from each <dir>/<dataset>_c<conc> and writes a
single HTML report overlay of throughput/efficiency vs concurrency plus a full
metrics table.
"""

import html
import json
import os
import re
import sys


DATASETS = ["text", "random", "reasoning"]
DS_COLORS = {
    "text": "#4e79a7",
    "random": "#f28e2b",
    "reasoning": "#e15759",
}


def load(sweep_dir, ds, conc):
    p = os.path.join(sweep_dir, f"{ds}_c{conc}", "profile_export_aiperf.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def avg(profile, key):
    if profile is None:
        return None
    v = profile.get(key)
    if isinstance(v, dict):
        for k in ("avg", "mean"):
            if k in v and isinstance(v[k], (int, float)):
                return v[k]
        return None
    return v


def discover_concurrency(sweep_dir, dataset):
    found = []
    prefix = f"{dataset}_c"
    for name in os.listdir(sweep_dir):
        if name.startswith(prefix):
            m = re.match(rf"^{re.escape(prefix)}(\d+)$", name)
            if m:
                found.append(int(m.group(1)))
    return sorted(found)


def collect(sweeps):
    """sweeps: list of (label, dir). Returns rows: [[label, ds, conc, profile], ...]"""
    rows = []
    for label, sdir in sweeps:
        for ds in DATASETS:
            for conc in discover_concurrency(sdir, ds):
                p = load(sdir, ds, conc)
                if p is not None:
                    rows.append((label, ds, conc, p))
    return rows


def span(a, b):
    """Return an rgba color interpolating between two rgb tuples."""
    n = [int(a[i] + (b[i] - a[i]) * 0.5) for i in range(3)]
    return f"rgb({n[0]},{n[1]},{n[2]})"


def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


GB10_BLUE = (14, 122, 254)


def svg_overlay(rows, ykey, ylabel, xlabel="Concurrency", dashed=None):
    """One line per (sweep,dataset): concurrency on x, ykey on y."""
    W, H, ML, MT, MB, MR = 700, 340, 72, 34, 58, 26
    series = {}
    all_conc = set()
    for label, ds, conc, p in rows:
        v = avg(p, ykey)
        if v is None:
            continue
        series.setdefault((label, ds), {})[conc] = v
        all_conc.add(conc)
    concs = sorted(all_conc)
    if not concs:
        return ""

    ymax = max((v for m in series.values() for v in m.values()), default=1) * 1.15

    def sx(c):
        if len(concs) == 1:
            return (ML + W - MR) / 2
        return ML + (concs.index(c)) / (len(concs) - 1) * (W - ML - MR)

    def sy(v):
        return H - MB - v / ymax * (H - MB - MT)

    out = [
        f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="#fff" stroke="#ccc"/>',
        f'<text x="{(ML + W - MR) / 2:.0f}" y="20" text-anchor="middle" font-size="15" font-weight="bold">{html.escape(ylabel)}</text>',
    ]
    for g in range(1, 6):
        gy = H - MB - g / 6 * (H - MB - MT)
        out.append(f'<line x1="{ML}" y1="{gy:.1f}" x2="{W - MR}" y2="{gy:.1f}" stroke="#ececec"/>')
    out.append(f'<line x1="{ML}" y1="{H - MB}" x2="{W - MR}" y2="{H - MB}" stroke="#777"/>')
    out.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{H - MB}" stroke="#777"/>')
    for i, c in enumerate(concs):
        out.append(f'<text x="{sx(c):.1f}" y="{H - MB + 18}" text-anchor="middle" font-size="12">c{c}</text>')
    out.append(f'<text x="{(ML + W - MR) / 2:.0f}" y="{H - MB + 32}" text-anchor="middle" font-size="13">{html.escape(xlabel)}</text>')
    out.append(f'<text x="14" y="{(MT + H - MB) / 2:.0f}" text-anchor="middle" font-size="13" transform="rotate(-90 14 {(MT + H - MB) / 2:.0f})">{html.escape(ylabel)}</text>')

    # unique sweeps for dash pattern
    sweep_labels = sorted({label for label, _, _, _ in rows})
    for (label, ds), m in series.items():
        pts_idx = [concs.index(c) for c in m if c in concs]
        if len(pts_idx) < 2:
            continue
        pts = " ".join(f"{sx(concs[i]):.1f},{sy(m[concs[i]]):.1f}" for i in pts_idx)
        dash = ""
        si = sweep_labels.index(label)
        if si > 0:
            dash = f' stroke-dasharray="{["6 3", "10 4", "3 4", "12 3 3 3"][si % 4]}"'
        out.append(f'<polyline points="{pts}" fill="none" stroke="{DS_COLORS[ds]}" stroke-width="2.2"{dash}/>')
        for i in pts_idx:
            out.append(f'<circle cx="{sx(concs[i]):.1f}" cy="{sy(m[concs[i]]):.1f}" r="3.5" fill="{DS_COLORS[ds]}"/>')
    out.append("</svg>")
    return "".join(out)


def fmt(v, nd=0):
    if v is None:
        return "-"
    return f"{v:.{nd}f}"


def build_report(sweeps, out_path):
    rows = []
    for label, sdir in sweeps:
        for ds in DATASETS:
            for conc in discover_concurrency(sdir, ds):
                p = load(sdir, ds, conc)
                if p is not None:
                    rows.append((label, ds, conc, p))

    # summary cards
    cards = []
    for label, sdir in sweeps:
        n = sum(1 for l, d, c, p in rows if l == label)
        peak = max((avg(p, "output_token_throughput") for l, d, c, p in rows if l == label if avg(p, "output_token_throughput")), default=0)
        best_eff = max((avg(p, "nvidia_output_tps_per_watt") for l, d, c, p in rows if l == label if avg(p, "nvidia_output_tps_per_watt")), default=0)
        osl = max((avg(p, "output_sequence_length") for l, d, c, p in rows if l == label if avg(p, "output_sequence_length")), default=0)
        cards.append(
            f"<div class='card'><h3>{html.escape(label)}</h3>"
            f"<p>{n} runs</p><p>Peak decode: <b>{peak:.0f}</b> tok/s</p>"
            f"<p>Best eff: <b>{best_eff:.2f}</b> tok/s/W</p>"
            f"<p>Longest OSL: <b>{osl:.0f}</b> tok</p></div>"
        )

    ov_tp = svg_overlay(rows, "output_token_throughput", "Decode Throughput (tok/s)")
    ov_eff = svg_overlay(rows, "nvidia_output_tps_per_watt", "Efficiency (tok/s/W)")
    ov_itl = svg_overlay(rows, "inter_token_latency", "Inter-Token Latency (ms)")

    trows = []
    for label, ds, conc, p in rows:
        trows.append(
            f"<tr><td>{html.escape(label)}</td><td>{ds}</td><td>c{conc}</td>"
            f"<td>{fmt(avg(p, 'request_throughput'))}</td>"
            f"<td>{fmt(avg(p, 'request_latency'))}</td>"
            f"<td>{fmt(avg(p, 'time_to_first_token'), 1)}</td>"
            f"<td>{fmt(avg(p, 'inter_token_latency'), 2)}</td>"
            f"<td>{fmt(avg(p, 'output_token_throughput'), 1)}</td>"
            f"<td>{fmt(avg(p, 'output_sequence_length'))}</td>"
            f"<td>{fmt(avg(p, 'input_sequence_length'))}</td>"
            f"<td>{fmt(avg(p, 'nvidia_average_gpu_power'))}</td>"
            f"<td>{fmt(avg(p, 'nvidia_output_tps_per_watt'), 2)}</td></tr>"
        )
    th = ("<table class='metrics'><thead><tr><th>Sweep</th><th>Dataset</th><th>Conc</th>"
          "<th>Req/s</th><th>Lat(ms)</th><th>TTFT(ms)</th><th>ITL(ms)</th>"
          "<th>Dec tok/s</th><th>OSL</th><th>ISL</th><th>W</th><th>tok/W</th></tr></thead><tbody>"
          + "".join(trows) + "</tbody></table>")

    body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>AIPerf MOE Multi-Sweep Comparison</title>
<style>
 body {{ font-family:-apple-system, Segoe UI, Roboto, sans-serif; margin:24px; color:#222; background:#fafafa; }}
 h1 {{ font-size:22px; }} h2 {{ font-size:18px; margin-top:28px; }}
 .cards {{ display:flex; flex-wrap:wrap; gap:16px; margin-bottom:20px; }}
 .card {{ background:#fff; border:1px solid #ddd; border-radius:8px; padding:12px 16px; min-width:210px; }}
 .card h3 {{ margin:0 0 6px; }} .card p {{ margin:4px 0; font-size:13px; }}
 .plotbox {{ background:#fff; border:1px solid #ddd; border-radius:8px; padding:14px; margin-bottom:20px; }}
 table.metrics {{ border-collapse:collapse; background:#fff; font-size:12.5px; }}
 table.metrics th,td {{ border:1px solid #ddd; padding:5px 8px; text-align:right; }}
 table.metrics th {{ background:#f0f0f0; }}
 table.metrics td:first-child {{ text-align:left; font-weight:700; }}
 .note {{ font-size:12px; color:#666; margin-top:22px; padding-top:12px; border-top:1px solid #ddd; }}
</style></head><body>
<h1>AIPerf MOE Multi-Sweep Comparison</h1>
<p>Model: Qwen3.6-35B-A3B-NVFP4 &middot; Hardware: NVIDIA GB10 (SM121) &middot; Sweeps:</p>
<div class="cards">{''.join(cards)}</div>
<h2>Decode Throughput vs Concurrency</h2>
<div class="plotbox">{ov_tp}</div>
<h2>Efficiency vs Concurrency</h2>
<div class="plotbox">{ov_eff}</div>
<h2>Inter-Token Latency vs Concurrency</h2>
<div class="plotbox">{ov_itl}</div>
<h2>All Metrics</h2>
{th}
<div class="note">Colors: text=blue, random=orange, reasoning=red. Solid = first sweep, dashed = later sweeps. Generated by compare_sweeps.py.</div>
</body></html>"""
    with open(out_path, "w") as f:
        f.write(body)
    print(f"Wrote {out_path}")


def fmt(v, nd=0):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compare AIPerf MOE sweeps.")
    ap.add_argument("--out", default="benchmark_comparison.html", help="output HTML")
    ap.add_argument("sweeps", nargs="+", metavar="LABEL=DIR",
                    help="sweep label and directory; repeatable")
    args = ap.parse_args()
    sweeps = []
    for item in args.sweeps:
        if "=" in item:
            label, _, d = item.partition("=")
            sweeps.append((label.strip(), os.path.abspath(d.strip())))
        else:
            sweeps.append((os.path.basename(item), os.path.abspath(item)))
    build_report(sweeps, args.out)


if __name__ == "__main__":
    main()