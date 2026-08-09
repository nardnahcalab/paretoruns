#!/usr/bin/env python3
"""
make_pareto.py - Generate Pareto curves from AIPerf profile exports.

Aggregates the key throughput/latency metrics across all run directories
(bench_dir/<dataset>_c<concurrency>) and writes an HTML report with inline
SVG Pareto curves showing how MOE performance differs by content type.

Usage:
  python3 make_pareto.py <bench_dir> [output.html]
"""

import csv
import html
import json
import os
import sys


DATASETS = ["text", "random", "reasoning"]
COLORS = {
    "text": "#4e79a7",
    "random": "#f28e2b",
    "reasoning": "#e15759",
}


def metric_value(profile, metric):
    """Extract a scalar from a metrics dict; returns dict with p50/avg/p99 where present."""
    v = profile.get(metric)
    if v is None:
        return None
    if isinstance(v, dict):
        keys = ("p50", "avg", "mean", "p99", "min", "max")
        out = {}
        for k in keys:
            if k in v and isinstance(v[k], (int, float)):
                out[k] = v[k]
        return out
    return {"avg": v}


def load_run(bench_dir, dataset, conc):
    run_dir = os.path.join(bench_dir, f"{dataset}_c{conc}")
    jpath = os.path.join(run_dir, "profile_export_aiperf.json")
    if not os.path.exists(jpath):
        return None
    with open(jpath) as f:
        return json.load(f)


def extract(profile):
    def pick(m, key):
        v = metric_value(profile, m)
        if v is None:
            return None
        return v.get(key)

    phases = profile.get("input_config", {}).get("phases", [])
    conc = None
    for ph in phases:
        if isinstance(ph, dict) and ph.get("concurrency") is not None:
            conc = ph.get("concurrency")
            break

    return {
        "concurrency": conc,
        "req_throughput": pick("request_throughput", "avg"),
        "req_latency_ms": pick("request_latency", "avg"),
        "ttft_ms": pick("time_to_first_token", "avg"),
        "itl_ms": pick("inter_token_latency", "avg"),
        "decode_tok_s": pick("output_token_throughput", "avg"),
        "decode_tok_s_user": pick("output_token_throughput_per_user", "avg"),
        "osl": pick("output_sequence_length", "avg"),
        "isl": pick("input_sequence_length", "avg"),
        "power_w": pick("nvidia_average_gpu_power", "avg"),
        "tps_per_watt": pick("nvidia_output_tps_per_watt", "avg"),
    }


def discover_concurrency(bench_dir, dataset):
    """Find _c<conc> run dirs for a dataset; return sorted list of ints."""
    import re
    found = []
    prefix = f"{dataset}_c"
    for name in os.listdir(bench_dir):
        if name.startswith(prefix):
            m = re.match(rf"^{re.escape(prefix)}(\d+)$", name)
            if m:
                found.append(int(m.group(1)))
    return sorted(found)


def build_dataset(bench_dir, dataset):
    rows = []
    for conc in discover_concurrency(bench_dir, dataset):
        profile = load_run(bench_dir, dataset, conc)
        if profile is None:
            continue
        rows.append(extract(profile))
    return rows


def svg_curves(datasets_data):
    """Render throughput-vs-latency Pareto curves as SVG."""
    plots = []
    W, H, ML, MT, MB, MR = 620, 380, 62, 28, 52, 20
    for i, (ds, rows) in enumerate(datasets_data.items()):
        x_label = "Request Latency (ms)"
        x_key = "req_latency_ms"
        y_label = "Throughput (req/s)"
        y_key = "req_throughput"

        points = [(r[x_key], r[y_key]) for r in rows if r[x_key] is not None and r[y_key] is not None]
        if len(points) < 2:
            continue

        xmin = min(p[0] for p in points) * 0.9
        xmax = max(p[0] for p in points) * 1.1
        ymin = 0
        ymax = max(p[1] for p in points) * 1.15

        def sx(v):
            return ML + (v - xmin) / (xmax - xmin) * (W - ML - MR)

        def sy(v):
            return H - MB - (v - ymin) / (ymax - ymin) * (H - MB - MT)

        color = COLORS[ds]
        body = []
        body.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff" stroke="#cccccc"/>')
        # grid lines
        for g in range(1, 6):
            gy = H - MB - g / 6 * (H - MB - MT)
            body.append(f'<line x1="{ML}" y1="{gy:.1f}" x2="{W - MR}" y2="{gy:.1f}" stroke="#ececec"/>')
        # axes
        body.append(f'<line x1="{ML}" y1="{H - MB}" x2="{W - MR}" y2="{H - MB}" stroke="#666"/>')
        body.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{H - MB}" stroke="#666"/>')
        # x labels
        body.append(f'<text x="{(ML + W - MR) / 2:.0f}" y="{H - MB + 24}" text-anchor="middle" font-size="13">{html.escape(x_label)}</text>')
        # y label (rotated)
        body.append(f'<text x="{14}" y="{(MT + H - MB) / 2:.0f}" text-anchor="middle" font-size="13" transform="rotate(-90 14 {(MT + H - MB) / 2:.0f})">{html.escape(y_label)}</text>')
        # title
        body.append(f'<text x="{(ML + W - MR) / 2:.0f}" y="{18}" text-anchor="middle" font-size="15" font-weight="bold">{ds} dataset</text>')
        # data line + points
        pts_s = " ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in points)
        body.append(f'<polyline points="{pts_s}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for p in points:
            body.append(f'<circle cx="{sx(p[0]):.1f}" cy="{sy(p[1]):.1f}" r="5" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
            conc = int(rows[points.index(p)].get("concurrency") or 0)
            body.append(f'<text x="{sx(p[0]):.1f}" y="{sy(p[1]) - 9:.1f}" text-anchor="middle" font-size="11">c{conc}</text>')

        plots.append(f"<svg width=\"{W}\" height=\"{H}\" xmlns=\"http://www.w3.org/2000/svg\">{''.join(body)}</svg>")
    return plots


def svg_gpu_chart(datasets_data):
    """Bar chart of decode tok/s per concurrency by dataset."""
    W, H, ML, MT, MB, MR = 620, 380, 62, 28, 52, 20
    rows = []
    for ds, data in datasets_data.items():
        for r in data:
            if r["decode_tok_s"] is not None and r["concurrency"] is not None:
                rows.append((ds, int(r["concurrency"]), r["decode_tok_s"]))
    if not rows:
        return ""

    groups = sorted({r[1] for r in rows})
    maxv = max(r[2] for r in rows) * 1.15
    bw = (W - ML - MR) / (len(groups) * 3.4)
    body = []
    body.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff" stroke="#cccccc"/>')
    for g in range(1, 6):
        gy = H - MB - g / 6 * (H - MB - MT)
        body.append(f'<line x1="{ML}" y1="{gy:.1f}" x2="{W - MR}" y2="{gy:.1f}" stroke="#ececec"/>')
    body.append(f'<line x1="{ML}" y1="{H - MB}" x2="{W - MR}" y2="{H - MB}" stroke="#666"/>')
    body.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{H - MB}" stroke="#666"/>')
    body.append(f'<text x="{(ML + W - MR) / 2:.0f}" y="{H - MB + 24}" text-anchor="middle" font-size="13">Concurrency</text>')
    body.append(f'<text x="{14}" y="{(MT + H - MB) / 2:.0f}" text-anchor="middle" font-size="13" transform="rotate(-90 14 {(MT + H - MB) / 2:.0f})">Decode tok/s</text>')
    body.append(f'<text x="{(ML + W - MR) / 2:.0f}" y="{18}" text-anchor="middle" font-size="15" font-weight="bold">Decode throughput by concurrency</text>')

    step = (W - ML - MR) / len(groups)
    for gi, conc in enumerate(groups):
        cx = ML + step * (gi + 0.5)
        for di, ds in enumerate(DATASETS):
            val = next((r[2] for r in rows if r[0] == ds and r[1] == conc), None)
            x0 = cx + (di - 1) * bw
            x1 = x0 + bw * 0.85
            y0 = H - MB
            if val is not None:
                y1 = H - MB - val / maxv * (H - MB - MT)
            else:
                y1 = H - MB
            body.append(f'<rect x="{x0:.1f}" y="{y1:.1f}" width="{bw:.1f}" height="{y0 - y1:.1f}" fill="{COLORS[ds]}" opacity="0.85"/>')
            if val is not None:
                body.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{y1 - 5:.1f}" text-anchor="middle" font-size="10">{val:.0f}</text>')
        body.append(f'<text x="{cx:.1f}" y="{H - MB + 16}" text-anchor="middle" font-size="12">c{conc}</text>')

    return f"<svg width=\"{W}\" height=\"{H}\" xmlns=\"http://www.w3.org/2000/svg\">{''.join(body)}</svg>"


def svg_table(datasets_data):
    """HTML table of key metrics."""
    out = ["<table class=\"metrics\"><thead><tr><th>Dataset</th><th>Conc</th>"
           "<th>Req/s</th><th>Latency (ms)</th><th>TTFT (ms)</th><th>ITL (ms)</th>"
           "<th>Decode tok/s</th><th>ISL</th><th>OSL</th><th>Power (W)</th><th>tok/s/W</th></tr></thead><tbody>"]
    for ds, rows in datasets_data.items():
        for r in rows:
            out.append(
                f"<tr><td>{ds}</td><td>c{r['concurrency']}</td>"
                f"<td>{r['req_throughput']:.2f}</td><td>{r['req_latency_ms']:.0f}</td>"
                f"<td>{r['ttft_ms']:.1f}</td><td>{r['itl_ms']:.2f}</td>"
                f"<td>{r['decode_tok_s']:.1f}</td><td>{r['isl']:.0f}</td>"
                f"<td>{r['osl']:.0f}</td><td>{r['power_w']:.0f}</td>"
                f"<td>{r['tps_per_watt']:.2f}</td></tr>"
            )
    out.append("</tbody></table>")
    return "".join(out)


def build_report(bench_dir, out_path):
    datasets_data = {}
    for ds in DATASETS:
        rows = build_dataset(bench_dir, ds)
        datasets_data[ds] = rows

    plots = svg_curves(datasets_data)
    gpu = svg_gpu_chart(datasets_data)
    table = svg_table(datasets_data)

    legend = "".join(
        f'<span class="legend"><span class="swatch" style="background:{COLORS[ds]}"></span>{ds}</span>'
        for ds in DATASETS
    )

    summary = []
    for ds in DATASETS:
        rows = datasets_data[ds]
        if not rows:
            continue
        tps = [r["decode_tok_s"] for r in rows if r["decode_tok_s"] is not None]
        tps_w = [r["tps_per_watt"] for r in rows if r["tps_per_watt"] is not None]
        max_c = max(rows, key=lambda r: r["decode_tok_s"] or 0)
        summary.append(
            f"<div class=\"card\"><h3>{ds}</h3>"
            f"<p>Peak decode: <b>{max(tps):.1f}</b> tok/s at c{max_c['concurrency']} "
            f"(latency {max_c['req_latency_ms']:.0f} ms).</p>"
            f"<p>Best efficiency: <b>{max(tps_w):.2f}</b> tok/s/W.</p>"
            f"<p>Lowest TTFT: <b>{min(r['ttft_ms'] for r in rows if r['ttft_ms'] is not None):.1f}</b> ms.</p></div>"
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>MOE Pareto Benchmark - {os.path.basename(bench_dir)}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #222; background: #fafafa; }}
 h1 {{ font-size: 22px; }} h2 {{ font-size: 18px; margin-top: 28px; }}
 .grid {{ display: flex; flex-wrap: wrap; gap: 24px; }}
 .plotbox {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
 .summary {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 20px; }}
 .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; min-width: 240px; }}
 .card h3 {{ margin: 0 0 6px 0; text-transform: capitalize; }}
 .card p {{ margin: 4px 0; font-size: 13px; }}
 .legend {{ margin-right: 18px; font-size: 13px; }}
 .swatch {{ display: inline-block; width: 12px; height: 12px; margin-right: 5px; border-radius: 2px; }}
 table.metrics {{ border-collapse: collapse; background: #fff; font-size: 13px; }}
 table.metrics th, table.metrics td {{ border: 1px solid #ddd; padding: 5px 9px; text-align: right; }}
 table.metrics th {{ background: #f0f0f0; }}
 table.metrics td:first-child, table.metrics td:nth-child(2) {{ text-align: left; font-weight: 600; }}
 .note {{ font-size: 12px; color: #666; margin-top: 24px; }}
</style>
</head>
<body>
<h1>MOE Pareto Benchmark</h1>
<p>Model: Qwen3.6-35B-A3B-NVFP4 &middot; Hardware: NVIDIA GB10 &middot; Run dir: {html.escape(os.path.basename(bench_dir))}</p>
{''.join(summary)}
<h2>Legend</h2>
<div>{legend}</div>
<h2>Pareto Curves (Throughput vs Latency)</h2>
<div class="grid">{''.join(f'<div class="plotbox">{p}</div>' for p in plots)}</div>
<h2>GPU Efficiency</h2>
<div class="plotbox">{gpu}</div>
<h2>Metrics Table</h2>
{table}
<p class="note">Generated by make_pareto.py. Latency/TTFT/ITL are averages; throughput is request-level average.</p>
</body>
</html>"""
    with open(out_path, "w") as f:
        f.write(html_doc)
    print(f"Wrote {out_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    bench_dir = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(bench_dir, "pareto_report.html")
    build_report(bench_dir, out_path)


if __name__ == "__main__":
    main()
