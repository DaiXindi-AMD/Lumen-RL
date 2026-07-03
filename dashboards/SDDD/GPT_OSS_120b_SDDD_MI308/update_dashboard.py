#!/usr/bin/env python3
"""Parse Run4 training log and generate dashboard HTML + JSON data."""
import json
import re
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

LOG_FILE = "/dev/shm/run4_launch.log"
OUT_DIR = Path(__file__).parent
HTML_FILE = OUT_DIR / "phase1.html"
JSON_FILE = OUT_DIR / "phase1_data.json"

TOTAL_STEPS = 15871

# Run metadata
RUN_META = {
    "run": "Run4",
    "capture_mode": "varnorm",
    "loss_type": "forward_kl",
    "batch_size": 32,
    "lr": "1e-4",
    "total_steps": TOTAL_STEPS,
}

# Regex for callback lines
CB_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"step=(?P<step>\d+)\s+"
    r"grad_norm=(?P<grad_norm>[\d.e+-]+)\s+"
    r"loss=(?P<loss>[\d.e+-]+)\s+"
    r"lr=(?P<lr>[\d.e+-]+)\s+"
    r"seq/max_len=(?P<seq_len>\d+)\s+"
    r"step_0_acc=(?P<s0_acc>[\d.e+-]+)\s+"
    r"step_0_loss=(?P<s0_loss>[\d.e+-]+)\s+"
    r"step_1_acc=(?P<s1_acc>[\d.e+-]+)\s+"
    r"step_1_loss=(?P<s1_loss>[\d.e+-]+)\s+"
    r"step_2_acc=(?P<s2_acc>[\d.e+-]+)\s+"
    r"step_2_loss=(?P<s2_loss>[\d.e+-]+)\s+"
    r"timing/step_s=(?P<step_s>[\d.e+-]+)\s+"
    r"timing/teacher_s=(?P<teacher_s>[\d.e+-]+)\s+"
    r"timing/train_s=(?P<train_s>[\d.e+-]+)"
)


def parse_log():
    data = {
        "steps": [], "grad_norms": [], "losses": [], "lrs": [],
        "step_0_acc": [], "step_1_acc": [], "step_2_acc": [],
        "step_0_loss": [], "step_1_loss": [], "step_2_loss": [],
        "step_times": [], "teacher_times": [], "train_times": [],
        "eval_steps": [], "eval_loss": [], "eval_acc_len": [],
        "eval_step_0_acc": [], "eval_step_1_acc": [], "eval_step_2_acc": [],
    }
    start_ts = None
    last_ts = None

    with open(LOG_FILE) as f:
        for line in f:
            m = CB_RE.search(line)
            if not m:
                continue
            step = int(m.group("step"))
            ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
            if start_ts is None:
                start_ts = ts
            last_ts = ts

            data["steps"].append(step)
            data["grad_norms"].append(float(m.group("grad_norm")))
            data["losses"].append(float(m.group("loss")))
            data["lrs"].append(float(m.group("lr")))
            data["step_0_acc"].append(float(m.group("s0_acc")))
            data["step_1_acc"].append(float(m.group("s1_acc")))
            data["step_2_acc"].append(float(m.group("s2_acc")))
            data["step_0_loss"].append(float(m.group("s0_loss")))
            data["step_1_loss"].append(float(m.group("s1_loss")))
            data["step_2_loss"].append(float(m.group("s2_loss")))
            data["step_times"].append(float(m.group("step_s")))
            data["teacher_times"].append(float(m.group("teacher_s")))
            data["train_times"].append(float(m.group("train_s")))

    elapsed = (last_ts - start_ts).total_seconds() if start_ts and last_ts else 0
    return data, elapsed, start_ts, last_ts


def make_html(data, elapsed_s, start_ts, last_ts):
    n = len(data["steps"])
    if n == 0:
        return "<html><body><h1>No data yet</h1></body></html>"

    cur_step = data["steps"][-1]
    pct = 100.0 * cur_step / TOTAL_STEPS

    # Last 100-step averages
    tail = min(100, n)
    avg_loss = sum(data["losses"][-tail:]) / tail
    avg_s0 = sum(data["step_0_acc"][-tail:]) / tail
    avg_s1 = sum(data["step_1_acc"][-tail:]) / tail
    avg_s2 = sum(data["step_2_acc"][-tail:]) / tail
    avg_gn = sum(data["grad_norms"][-tail:]) / tail

    # Last 1000-step averages for comparison
    tail_1k = min(1000, n)
    avg_s0_1k = sum(data["step_0_acc"][-tail_1k:]) / tail_1k
    avg_s1_1k = sum(data["step_1_acc"][-tail_1k:]) / tail_1k
    avg_s2_1k = sum(data["step_2_acc"][-tail_1k:]) / tail_1k

    elapsed_h = int(elapsed_s // 3600)
    elapsed_m = int((elapsed_s % 3600) // 60)
    eta_s = elapsed_s / max(cur_step, 1) * (TOTAL_STEPS - cur_step) if cur_step > 0 else 0
    eta_h = int(eta_s // 3600)
    eta_m = int((eta_s % 3600) // 60)

    is_done = cur_step >= TOTAL_STEPS - 1
    status_class = "st-completed" if is_done else "st-training"
    status_text = "Completed" if is_done else "Training"

    # Run3 reference line (postnorm, B=32, forward_kl → 41.9%)
    run3_ref = 0.419

    # Downsample for plotting (keep every Nth point, max ~2000 points)
    stride = max(1, n // 2000)
    ds = lambda arr: arr[::stride]

    steps_ds = ds(data["steps"])
    losses_ds = ds(data["losses"])
    gn_ds = ds(data["grad_norms"])
    lrs_ds = ds(data["lrs"])
    s0_ds = ds(data["step_0_acc"])
    s1_ds = ds(data["step_1_acc"])
    s2_ds = ds(data["step_2_acc"])
    s0l_ds = ds(data["step_0_loss"])
    s1l_ds = ds(data["step_1_loss"])
    s2l_ds = ds(data["step_2_loss"])
    st_ds = ds(data["step_times"])
    tt_ds = ds(data["teacher_times"])
    tr_ds = ds(data["train_times"])

    # Smoothed accuracy (rolling 50)
    def smooth(arr, w=50):
        out = []
        for i in range(len(arr)):
            lo = max(0, i - w + 1)
            out.append(sum(arr[lo:i+1]) / (i - lo + 1))
        return out

    s0_sm = smooth(ds(data["step_0_acc"]))
    s1_sm = smooth(ds(data["step_1_acc"]))
    s2_sm = smooth(ds(data["step_2_acc"]))

    update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GPT-OSS-120B Eagle3 SDDD — Run4 varnorm</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;background:#0d1117;color:#c9d1d9;margin:0;padding:16px}}
.header{{text-align:center;padding:16px 0}}
h1{{color:#58a6ff;margin:0 0 4px 0;font-size:22px;font-weight:600}}
.sub{{color:#8b949e;font-size:13px;margin:0}}
.st{{font-size:14px;margin:6px 0;font-weight:600}}
.st-training{{color:#3fb950}}.st-completed{{color:#58a6ff}}.st-stopped{{color:#f85149}}
.stats{{display:flex;justify-content:center;gap:16px;margin:14px 0;flex-wrap:wrap}}
.s{{background:#161b22;border:1px solid #21262d;padding:10px 18px;border-radius:6px;text-align:center}}
.sv{{font-size:20px;font-weight:600;color:#58a6ff}}
.sl{{font-size:11px;color:#8b949e;margin-top:3px}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:12px;max-width:1600px;margin:0 auto}}
.ch{{background:#161b22;border:1px solid #21262d;border-radius:6px;padding:8px;height:380px}}
@media(max-width:1100px){{.charts{{grid-template-columns:1fr}}}}
.up{{text-align:center;color:#8b949e;font-size:11px;margin-top:12px}}
.ref{{color:#8b949e;font-size:12px;text-align:center;margin:8px 0}}
</style>
</head><body>
<div class="header">
<h1>GPT-OSS-120B Eagle3 SDDD — Run4</h1>
<p class="sub">UltraChat+Magpie 508K | lr=1e-4 | bs=32 | <b>varnorm</b> + forward_kl | spec_length=3 | 8x MI355 (Lumen FSDP2 + ATOM MXFP4)</p>
<p class="st {status_class}">{status_text}</p>
<div class="stats">
<div class="s"><div class="sv">{cur_step:,} / {TOTAL_STEPS:,}</div><div class="sl">Step ({pct:.1f}%)</div></div>
<div class="s"><div class="sv">{elapsed_h}h {elapsed_m}m</div><div class="sl">Elapsed (ETA: {eta_h}h {eta_m}m)</div></div>
<div class="s"><div class="sv">{avg_loss:.4f}</div><div class="sl">Avg Loss (last 100)</div></div>
<div class="s"><div class="sv">{100*avg_s0:.1f}% / {100*avg_s1:.1f}% / {100*avg_s2:.1f}%</div><div class="sl">Acc pos 0/1/2 (last 100)</div></div>
<div class="s"><div class="sv">{avg_gn:.2f}</div><div class="sl">Grad Norm (last 100)</div></div>
</div>
<p class="ref">Run3 reference (postnorm): 41.9% | NVIDIA target: ~70%</p>
</div>

<div class="charts">
<div class="ch" id="c_acc"></div>
<div class="ch" id="c_loss"></div>
<div class="ch" id="c_gn"></div>
<div class="ch" id="c_lr"></div>
<div class="ch" id="c_ploss"></div>
<div class="ch" id="c_time"></div>
</div>

<p class="up">Updated: {update_time} | Auto-refresh every 20 min</p>

<script>
const layout_base = {{
    paper_bgcolor:'#161b22',plot_bgcolor:'#161b22',
    font:{{color:'#c9d1d9',size:11}},
    margin:{{l:55,r:20,t:35,b:40}},
    xaxis:{{gridcolor:'#21262d',title:'Step'}},
    yaxis:{{gridcolor:'#21262d'}},
    legend:{{bgcolor:'rgba(0,0,0,0)',font:{{size:10}}}},
    hovermode:'x unified',
}};

// Accuracy
Plotly.newPlot('c_acc', [
  {{x:{json.dumps(steps_ds)},y:{json.dumps([round(v*100,2) for v in s0_sm])},name:'pos0 (smooth)',mode:'lines',line:{{color:'#58a6ff',width:2}}}},
  {{x:{json.dumps(steps_ds)},y:{json.dumps([round(v*100,2) for v in s1_sm])},name:'pos1 (smooth)',mode:'lines',line:{{color:'#3fb950',width:2}}}},
  {{x:{json.dumps(steps_ds)},y:{json.dumps([round(v*100,2) for v in s2_sm])},name:'pos2 (smooth)',mode:'lines',line:{{color:'#d29922',width:2}}}},
  {{x:[0,{TOTAL_STEPS}],y:[{run3_ref*100},{run3_ref*100}],name:'Run3 ref (41.9%)',mode:'lines',line:{{color:'#f85149',width:1,dash:'dash'}}}},
], {{...layout_base,title:'Accuracy (%)',yaxis:{{...layout_base.yaxis,title:'%',range:[0,70]}}}});

// Total loss
Plotly.newPlot('c_loss', [
  {{x:{json.dumps(steps_ds)},y:{json.dumps([round(v,4) for v in losses_ds])},name:'total loss',mode:'lines',line:{{color:'#58a6ff',width:1.5}}}},
], {{...layout_base,title:'Total Loss'}});

// Grad norm
Plotly.newPlot('c_gn', [
  {{x:{json.dumps(steps_ds)},y:{json.dumps([round(v,3) for v in gn_ds])},name:'grad_norm',mode:'lines',line:{{color:'#bc8cff',width:1.5}}}},
], {{...layout_base,title:'Gradient Norm',yaxis:{{...layout_base.yaxis,type:'log',title:'grad_norm (log)'}}}});

// Learning rate
Plotly.newPlot('c_lr', [
  {{x:{json.dumps(steps_ds)},y:{json.dumps([round(v, 10) for v in lrs_ds])},name:'lr',mode:'lines',line:{{color:'#3fb950',width:1.5}}}},
], {{...layout_base,title:'Learning Rate'}});

// Per-position loss
Plotly.newPlot('c_ploss', [
  {{x:{json.dumps(steps_ds)},y:{json.dumps([round(v,4) for v in s0l_ds])},name:'pos0',mode:'lines',line:{{color:'#58a6ff',width:1.5}}}},
  {{x:{json.dumps(steps_ds)},y:{json.dumps([round(v,4) for v in s1l_ds])},name:'pos1',mode:'lines',line:{{color:'#3fb950',width:1.5}}}},
  {{x:{json.dumps(steps_ds)},y:{json.dumps([round(v,4) for v in s2l_ds])},name:'pos2',mode:'lines',line:{{color:'#d29922',width:1.5}}}},
], {{...layout_base,title:'Per-Position Loss'}});

// Timing
Plotly.newPlot('c_time', [
  {{x:{json.dumps(steps_ds)},y:{json.dumps([round(v,3) for v in st_ds])},name:'step',mode:'lines',line:{{color:'#58a6ff',width:1}}}},
  {{x:{json.dumps(steps_ds)},y:{json.dumps([round(v,3) for v in tt_ds])},name:'teacher',mode:'lines',line:{{color:'#3fb950',width:1}}}},
  {{x:{json.dumps(steps_ds)},y:{json.dumps([round(v,3) for v in tr_ds])},name:'train',mode:'lines',line:{{color:'#d29922',width:1}}}},
], {{...layout_base,title:'Timing (seconds)',yaxis:{{...layout_base.yaxis,title:'s'}}}});
</script>
</body></html>"""
    return html


def main():
    data, elapsed, start_ts, last_ts = parse_log()
    n = len(data["steps"])
    if n == 0:
        print("No training data found in log")
        return

    cur_step = data["steps"][-1]
    tail = min(100, n)
    avg_acc = sum(data["step_0_acc"][-tail:]) / tail

    # Write JSON
    with open(JSON_FILE, "w") as f:
        json.dump(data, f)

    # Write HTML
    html = make_html(data, elapsed, start_ts, last_ts)
    with open(HTML_FILE, "w") as f:
        f.write(html)

    print(f"Dashboard updated: step={cur_step}/{TOTAL_STEPS} "
          f"({100*cur_step/TOTAL_STEPS:.1f}%), "
          f"acc={100*avg_acc:.1f}% (last {tail}), "
          f"n_points={n}")


if __name__ == "__main__":
    main()
