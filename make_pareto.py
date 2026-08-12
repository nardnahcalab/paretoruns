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

    # Calculate actual decode speed (tokens during decode phase only)
    osl = pick("output_sequence_length", "avg")
    decode_dur = pick("decode_duration", "avg")
    decode_speed = (osl / (decode_dur / 1000.0)) if osl and decode_dur and decode_dur > 0 else None

    return {
        "concurrency": conc,
        "req_throughput": pick("request_throughput", "avg"),
        "req_latency_ms": pick("request_latency", "avg"),
        "ttft_ms": pick("time_to_first_token", "avg"),
        "itl_ms": pick("inter_token_latency", "avg"),
        "decode_tok_s": pick("output_token_throughput", "avg"),
        "decode_speed": decode_speed,  # Actual tok/s during decode phase
        "decode_duration_ms": decode_dur,
        "decode_tok_s_user": pick("output_token_throughput_per_user", "avg"),
        "osl": pick("output_sequence_length", "avg"),
        "isl": pick("input_sequence_length", "avg"),
        "power_w": pick("nvidia_average_gpu_power", "avg"),
        "tps_per_watt": pick("nvidia_output_tps_per_watt", "avg"),
        "prefill_tok_s": pick("prefill_throughput_per_user", "avg"),
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
    """Render combined throughput-vs-latency Pareto curves as SVG."""
    plots = []
    W, H, ML, MT, MB, MR = 700, 420, 72, 32, 56, 24

    x_label = "Request Latency (ms) - end-to-end time per request"
    x_key = "req_latency_ms"
    y_label = "Decode Speed (tok/s) - tokens generated per second during decode"
    y_key = "decode_speed"

    # Collect all points across datasets
    all_points = []
    for ds, rows in datasets_data.items():
        for r in rows:
            if r[x_key] is not None and r[y_key] is not None:
                all_points.append((r[x_key], r[y_key], ds, int(r.get("concurrency") or 0),
                                   r.get("isl", 0), r.get("osl", 0), r.get("ttft_ms", 0)))

    if len(all_points) < 2:
        return plots

    xmin = min(p[0] for p in all_points) * 0.85
    xmax = max(p[0] for p in all_points) * 1.15
    ymin = 0
    ymax = max(p[1] for p in all_points) * 1.2

    def sx(v):
        return ML + (v - xmin) / (xmax - xmin) * (W - ML - MR)

    def sy(v):
        return H - MB - (v - ymin) / (ymax - ymin) * (H - MB - MT)

    body = []
    body.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff" stroke="#cccccc"/>')

    # Grid lines
    for g in range(1, 7):
        gy = H - MB - g / 7 * (H - MB - MT)
        body.append(f'<line x1="{ML}" y1="{gy:.1f}" x2="{W - MR}" y2="{gy:.1f}" stroke="#ececec" stroke-dasharray="3,3"/>')

    # X-axis tick marks and labels
    x_range = xmax - xmin
    x_step = x_range / 5
    for i in range(6):
        xv = xmin + i * x_step
        tx = sx(xv)
        body.append(f'<line x1="{tx:.1f}" y1="{H - MB}" x2="{tx:.1f}" y2="{H - MB + 5}" stroke="#666"/>')
        body.append(f'<text x="{tx:.1f}" y="{H - MB + 18}" text-anchor="middle" font-size="11" fill="#444">{xv:.0f}</text>')

    # Y-axis tick marks and labels
    y_range = ymax - ymin
    y_step = y_range / 6
    for i in range(7):
        yv = ymin + i * y_step
        ty = sy(yv)
        body.append(f'<line x1="{ML - 5}" y1="{ty:.1f}" x2="{ML}" y2="{ty:.1f}" stroke="#666"/>')
        body.append(f'<text x="{ML - 8}" y="{ty:.1f}" text-anchor="end" font-size="11" fill="#444" dominant-baseline="middle">{yv:.1f}</text>')

    # Axes
    body.append(f'<line x1="{ML}" y1="{H - MB}" x2="{W - MR}" y2="{H - MB}" stroke="#333" stroke-width="1.5"/>')
    body.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{H - MB}" stroke="#333" stroke-width="1.5"/>')

    # Axis labels (larger, bold)
    body.append(f'<text x="{(ML + W - MR) / 2:.0f}" y="{H - 8}" text-anchor="middle" font-size="14" font-weight="bold" fill="#222">{html.escape(x_label)}</text>')
    body.append(f'<text x="16" y="{(MT + H - MB) / 2:.0f}" text-anchor="middle" font-size="14" font-weight="bold" fill="#222" transform="rotate(-90 16 {(MT + H - MB) / 2:.0f})">{html.escape(y_label)}</text>')

    # Title
    body.append(f'<text x="{(ML + W - MR) / 2:.0f}" y="{20}" text-anchor="middle" font-size="16" font-weight="bold" fill="#111">Pareto Curve: Decode Speed vs Latency</text>')

    # Draw lines and points per dataset
    for ds, rows in datasets_data.items():
        pts = [(r[x_key], r[y_key], int(r.get("concurrency") or 0), r.get("isl", 0), r.get("osl", 0), r.get("ttft_ms", 0))
               for r in rows if r[x_key] is not None and r[y_key] is not None]
        if len(pts) < 2:
            continue
        color = COLORS[ds]
        pts_s = " ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in pts)
        body.append(f'<polyline points="{pts_s}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>')
        for px, py, conc, isl, osl, ttft in pts:
            body.append(f'<circle cx="{sx(px):.1f}" cy="{sy(py):.1f}" r="6" fill="{color}" stroke="#fff" stroke-width="2"/>')
            body.append(f'<text x="{sx(px):.1f}" y="{sy(py) - 11:.1f}" text-anchor="middle" font-size="11" fill="{color}" font-weight="bold">c{conc}</text>')

    # Legend (bottom-right)
    lx = W - MR - 160
    ly = MT + 16
    body.append(f'<rect x="{lx - 8}" y="{ly - 14}" width="168" height="{len(DATASETS) * 22 + 8}" fill="#fff" stroke="#ddd" rx="4"/>')
    for i, ds in enumerate(DATASETS):
        y = ly + i * 22
        body.append(f'<rect x="{lx}" y="{y - 5}" width="14" height="10" fill="{COLORS[ds]}" rx="2"/>')
        body.append(f'<text x="{lx + 20}" y="{y + 3}" font-size="12" fill="#333" font-weight="600">{ds}</text>')

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
    out = ["<table class=\"metrics\"><thead><tr>"
           "<th>Dataset</th><th>Conc</th>"
           "<th>ISL</th><th>OSL</th>"
           "<th>TTFT (ms)</th><th>ITL (ms)</th>"
           "<th>E2E Latency (ms)</th><th>Decode Phase (ms)</th>"
           "<th>Decode Speed (tok/s)</th><th>Req/s</th>"
           "<th>Power (W)</th><th>tok/s/W</th>"
           "</tr></thead><tbody>"]
    for ds, rows in datasets_data.items():
        for r in rows:
            decode_speed = r.get('decode_speed')
            decode_speed_str = f"{decode_speed:.1f}" if decode_speed else "N/A"
            out.append(
                f"<tr><td>{ds}</td><td>c{r['concurrency']}</td>"
                f"<td>{r['isl']:.0f}</td><td>{r['osl']:.0f}</td>"
                f"<td>{r['ttft_ms']:.1f}</td><td>{r['itl_ms']:.2f}</td>"
                f"<td>{r['req_latency_ms']:.0f}</td><td>{r.get('decode_duration_ms', 0):.0f}</td>"
                f"<td>{decode_speed_str}</td><td>{r['req_throughput']:.2f}</td>"
                f"<td>{r['power_w']:.0f}</td>"
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
 .definitions {{ background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 16px; margin: 16px 0; }}
 .definitions h3 {{ margin: 0 0 12px 0; font-size: 15px; }}
 .definitions dl {{ margin: 0; }}
 .definitions dt {{ font-weight: 600; font-size: 13px; margin-top: 8px; }}
 .definitions dd {{ margin: 0 0 0 16px; font-size: 12px; color: #555; }}
 .example {{ background: #e8f4e8; border-left: 3px solid #4caf50; padding: 8px 12px; margin: 8px 0; font-size: 12px; }}
 .note-box {{ background: #fff3cd; border-left: 3px solid #ffc107; padding: 8px 12px; margin: 12px 0; font-size: 12px; }}
</style>
</head>
<body>
<h1>MOE Pareto Benchmark</h1>
<p>Run dir: {html.escape(os.path.basename(bench_dir))} &middot; Hardware: NVIDIA GB10</p>

<div class="definitions">
<h3>Metric Definitions</h3>
<dl>
<dt>ISL (Input Sequence Length)</dt>
<dd>Number of tokens in the input prompt. Varies across requests in multi-turn conversations.</dd>

<dt>OSL (Output Sequence Length)</dt>
<dd>Number of tokens generated by the model. Target was 128 tokens (+/- 32).</dd>

<dt>TTFT (Time to First Token)</dt>
<dd>Time from request start to first token received. Includes prefill phase and scheduling overhead.</dd>

<dt>ITL (Inter-Token Latency)</dt>
<dd>Average time between consecutive output tokens during decode phase. Lower is better.</dd>

<dt>E2E Latency (End-to-End)</dt>
<dd>Total request time from start to last token received. = TTFT + (OSL - 1) * ITL.</dd>

<dt>Decode Phase Duration</dt>
<dd>Time spent generating tokens after first token. = E2E Latency - TTFT.</dd>

<dt>Decode Speed (tok/s)</dt>
<dd>Actual token generation rate during decode phase. = OSL / Decode Phase Duration. This is the true decode performance.</dd>

<dt>Req/s (Request Throughput)</dt>
<dd>Total requests completed per second. Affected by concurrency level.</dd>

<dt>tok/s/W (Tokens per Watt)</dt>
<dd>Energy efficiency: output tokens generated per Joule of GPU energy consumed.</dd>
</dl>

<div class="example">
<strong>Example (Muse-Glimmer-30B, text, c1):</strong><br>
ISL=301 tokens, OSL=127 tokens, TTFT=992 ms, ITL=125 ms<br>
E2E Latency = 992 + (127-1) * 125 = <b>16,742 ms</b><br>
Decode Phase = 16,742 - 992 = <b>15,750 ms</b><br>
Decode Speed = 127 / 15.75 = <b>8.1 tok/s</b>
</div>

<div class="note-box">
<strong>Note: Decode Speed vs Throughput tokens/s</strong><br>
<b>Decode Speed (tok/s)</b> = OSL / Decode Phase Duration. This measures how fast the model generates tokens <i>during active generation</i>. It reflects the model's raw decode performance.<br><br>
<b>Throughput tokens/s</b> (in GPU Efficiency chart) = Total Output Tokens / Benchmark Duration. This includes idle time between requests and is affected by concurrency. At c1, it equals Decode Speed. At higher concurrency, it can exceed Decode Speed because multiple requests generate tokens in parallel.<br><br>
<b>When to use which:</b> Use Decode Speed to compare model inference performance. Use Throughput to measure system-level capacity.
</div>
</div>

{''.join(summary)}
<h2>Legend</h2>
<div>{legend}</div>
<h2>Pareto Curve: Decode Speed vs Latency</h2>
<p style="font-size: 13px; color: #666;">X-axis: End-to-end request latency. Y-axis: Token generation speed during decode. Higher Y is better (faster decode). Lower X is better (lower latency). Ideal: top-left corner.</p>
<div class="grid">{''.join(f'<div class="plotbox">{p}</div>' for p in plots)}</div>
<h2>GPU Efficiency</h2>
<div class="plotbox">{gpu}</div>
<h2>Metrics Table</h2>
{table}
<p class="note">Generated by make_pareto.py. Decode Speed = OSL / Decode Phase Duration. Latency includes prefill + decode + scheduling overhead.</p>
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
