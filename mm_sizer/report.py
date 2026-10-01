"""Ranked table, assumptions file, deployment configs, replay script."""
import contextlib
import json
import os
import shutil
from importlib import metadata

import yaml

from . import sizer


def fmt_gpus(g):
    return " + ".join(f"{n}x {s}" for s, n in sorted(g.items(), key=lambda kv: -kv[1]))


def overlap(a, b):
    return a["cost_p90"] >= b["cost_p10"]


def table(out, cfg):
    mode = cfg["mode"]
    lines = [f"# mm-sizer: {cfg['aic_model']}, {mode} mode", ""]
    if mode == "batch":
        lines.append(f"{cfg['batch']['files']:,} files in {cfg['batch']['deadline_hours']} h. "
                     "Per-request latency relaxed (batch.slo) so AIConfigurator batches for throughput.")
    else:
        lines.append(f"{cfg['online']['files_per_s']} files/s, p95 TTFT <= {cfg['slo']['ttft_ms']} ms, "
                     f"tpot {cfg['slo']['tpot_ms']} ms.")
    lines += ["", "| # | topology | main | encoder | GPUs (P50 sizing) | GPU range P10-P90 | CPU cores | "
              "$/1k files P50 | P10-P90 | binding | shapes |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(out["results"], 1):
        rng = "; ".join(f"{s} {r['gpus_p10'][s]}-{r['gpus_p90'][s]}" for s in r["gpus_p10"])
        g = r.get("online", {}).get("gpus") or r["base"]["gpus"]
        cores = r.get("online", {}).get("cores") or r["base"]["cores"]
        lines.append(f"| {i} | {r['topology']} | {r['main']} | {r['encoder'] or '-'} | {fmt_gpus(g)} | {rng} | {cores} | "
                     f"{r['cost_p50']:.3f} | {r['cost_p10']:.3f}-{r['cost_p90']:.3f} | {r['binding']} | "
                     f"{len(r['shapes'])} |")
    lines.append("")
    rs = out["results"]
    if len(rs) >= 2 and overlap(rs[0], rs[1]):
        lines.append(f"Top two overlap (#1 P90 {rs[0]['cost_p90']:.3f} >= #2 P10 {rs[1]['cost_p10']:.3f}). "
                     "Configs for both are emitted; A/B them with replay.py before committing.")
    for i, r in enumerate(rs[:3], 1):
        if len(r["shapes"]) > 1:
            lines.append(f"#{i}: buckets chose different replica shapes ({' | '.join(r['shapes'])}). Cost assumes "
                         "routing by bucket; a single shared shape costs more.")
        if r.get("warning"):
            lines.append(f"#{i}: {r['warning']}")
        if r.get("online") and not r["online"]["meets_slo"]:
            why = (" Requests in that bucket exceed the SLO even with no queueing; more capacity will not help, "
                   "split those files into smaller requests or relax the SLO." if r["online"]["binding"].startswith("service") else "")
            lines.append(f"#{i}: simulation did not reach the TTFT SLO (p95 {r['online']['p95_ttft_ms']} ms, "
                         f"{r['online']['binding']}).{why}")
    lines += ["", "## Buckets", "", "| bucket | share | images | image px | text tok | out tok | visual tok |",
              "|---|---|---|---|---|---|---|"]
    for b in out["buckets"]:
        s = b["shape"]
        lines.append(f"| {b['id']} | {b['weight']:.0%} | {s['num_images']} | {s['image_height']}x{s['image_width']} | "
                     f"{s['isl']} | {s['osl']} | {s['visual_tokens']} |")
    if rs:
        lines += ["", "## Per-bucket configs, #1", "", "| bucket | shape | seq/s | GPUs | ttft ms | tpot ms | encoder ms |",
                  "|---|---|---|---|---|---|---|"]
        for bid, row in rs[0]["per_bucket"].items():
            lines.append(f"| {bid} | {row['shape']} | {row['seq_s']:.2f} | {fmt_gpus(row['gpus'])} | "
                         f"{row['ttft_ms']:.0f} | {row['tpot_ms']:.1f} | {row['encoder_ms']:.0f} |")
    if out["rejected"]:
        lines += ["", "## Rejected", ""]
        for r in out["rejected"]:
            lines.append(f"- {r['topology']} main={r['main']} enc={r['encoder']}: {r['reason'][:200]}")
    lines += ["", "## Not modeled", "",
              "- Encoder sharing a GPU with its own queue (MPS colocation): needs measured slowdown factors.",
              "- Video and audio requests (skipped by the profiler): " + json.dumps(out.get("skipped", {})),
              "- Rates are AIConfigurator estimates, not measurements, until a calibration run replaces them "
              f"(model_sigma {cfg.get('model_sigma', 0.1)})."]
    return "\n".join(lines) + "\n"


def versions():
    v = {}
    for pkg in ("aiconfigurator", "transformers", "torch"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


def assumptions(cfg, plan, out, profile_path):
    return {
        "request_plan": {k: v for k, v in plan["request_plan"].items() if k != "prompt_text"},
        "prompt_text": plan["request_plan"].get("prompt_text"),
        "model": plan["model"],
        "profile": profile_path,
        "versions": versions(),
        "serving_backend": cfg.get("backend", "vllm"),
        "stack": cfg["stack"],
        "prices": cfg["prices"],
        "cpu_core_price": cfg.get("cpu_core_price"),
        "cpu_speed_factor": cfg.get("cpu_speed_factor", 1.0),
        "link_gbps": cfg["link_gbps"],
        "output_tokens": cfg["output_tokens"],
        "slo": cfg["slo"],
        "mode": cfg["mode"],
        "batch": cfg.get("batch"),
        "online": cfg.get("online"),
        "model_sigma": cfg.get("model_sigma", 0.1),
        "calibration": "aiconfigurator SILICON database (uncalibrated against your deployment)",
        "target_utilization": 0.8,
        "buckets": [{"id": b["id"], "weight": b["weight"], **b["shape"]} for b in out["buckets"]],
    }


def emit_configs(out, cfg, plan, outdir, top=2):
    """deployment.yaml (+ EPD launch scripts) for each top candidate; AIConfigurator's own artifacts
    (k8s_deploy.yaml, run scripts) for the dominant bucket where it can generate them (non-EPD rows)."""
    from aiconfigurator.cli.api import cli_default

    from . import epd_config

    made = []
    bk = {b["id"]: b for b in out["buckets"]}
    for i, r in enumerate(out["results"][:top], 1):
        d = os.path.join(outdir, f"config_{i}_{r['topology']}_{r['main']}" + (f"_{r['encoder']}" if r["encoder"] else ""))
        shutil.rmtree(d, ignore_errors=True)
        epd_config.write(r, out["buckets"], plan, cfg, cfg["_bytes_per_token"], d)
        made.append(d)
        if r["topology"] in ("e_agg", "e_p_d"):
            continue
        # dominant bucket = largest share of this candidate's GPU-hours
        dom = max(r["per_bucket"], key=lambda bid: bk[bid]["weight"] * sum(r["per_bucket"][bid]["gpus"].values())
                  / r["per_bucket"][bid]["seq_s"])
        s = bk[dom]["shape"]
        try:
            log = open(os.path.join(d, "aic_generator.log"), "w")
            with log, contextlib.redirect_stdout(log):
                    cli_default(cfg["aic_model"], cfg.get("max_gpus_per_replica", 8), r["main"], backend=cfg.get("backend", "vllm"),
                            isl=s["isl"], osl=s["osl"], image_height=s["image_height"], image_width=s["image_width"],
                            num_images=s["num_images"], ttft=sizer.slo(cfg)[0], tpot=sizer.slo(cfg)[1], top_n=1,
                            save_dir=os.path.join(d, "aic"))
        except (Exception, SystemExit) as e:
            made.append(f"{d}/aic: generator failed: {type(e).__name__}: {str(e)[:200]}")
    return made


def write_all(out, cfg, plan, profile_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    for name in os.listdir(outdir):            # configs from a previous run would look current
        if name.startswith("config_"):
            shutil.rmtree(os.path.join(outdir, name), ignore_errors=True)
    md = table(out, cfg)
    open(os.path.join(outdir, "report.md"), "w").write(md)
    slim = {k: v for k, v in out.items() if k != "buckets"}
    slim["buckets"] = [{k: v for k, v in b.items() if k != "idx"} for b in out["buckets"]]
    json.dump(slim, open(os.path.join(outdir, "results.json"), "w"), indent=1, default=str)
    yaml.safe_dump(assumptions(cfg, plan, out, profile_path), open(os.path.join(outdir, "assumptions.yaml"), "w"),
                   sort_keys=False)
    shutil.copy(os.path.join(os.path.dirname(__file__), "replay.py"), os.path.join(outdir, "replay.py"))
    return md
