"""AIConfigurator wrapper: one search per (bucket shape, topology, GPU types), cached on disk.

Topologies map onto AIConfigurator's own search:
  agg          encoder inside the aggregated worker           enable_epd=False, mode agg
  pd           encoder inside the prefill worker               enable_epd=False, mode disagg
  e_agg        separate encode workers + aggregated workers    enable_epd=True,  mode agg
  e_p_d        separate encode, prefill, decode workers        enable_epd=True,  mode disagg
The encode pool may run on another GPU type (encoder_system). MPS-style colocation
(encoder sharing a GPU with its own queue) is not modeled by AIConfigurator; it needs
measured slowdown factors and is reported as uncovered.
"""
import contextlib
import hashlib
import json
import logging
import os

TOPOLOGIES = {"agg": (False, "agg"), "pd": (False, "disagg"), "e_agg": (True, "agg"), "e_p_d": (True, "disagg")}


def _key(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:20]


def _gpus(row, enc_sys, main_sys):
    e = 0
    if "(e)workers" in row and row["(e)workers"] == row["(e)workers"]:   # not NaN
        e = int(row["(e)workers"]) * int(row.get("(e)tp", 1)) * int(row.get("(e)pp", 1))
    total = int(row["num_total_gpus"])
    gpus = {main_sys: total - e}
    if e:
        es = enc_sys or main_sys
        gpus[es] = gpus.get(es, 0) + e
    return gpus


def _shape_str(row, mode):
    def part(p):
        w = row.get(f"{p}workers")
        par = row.get(f"{p}parallel") or (row.get("parallel") if p == "(a)" else "")   # EPD agg rows keep tp in 'parallel'
        return f"{int(w)}x{par}" if w is not None and w == w and w > 0 else None
    if mode == "agg":
        parts = [("A", part("(a)") or f"1x{row.get('parallel', '')}")]
    else:
        parts = [("P", part("(p)")), ("D", part("(d)"))]
    e = part("(e)")
    if e:
        parts.append(("E", e))
    return " ".join(f"{k}:{v}" for k, v in parts if v)


def search(model, shape, topology, main_sys, enc_sys, ttft, tpot, backend="vllm", total_gpus=8, top_n=5,
           cache_dir="out/aic_cache"):
    """Rows meeting the SLO for one bucket, best first. Each row: seq_s per replica, gpus per system, latencies."""
    epd, mode = TOPOLOGIES[topology]
    # one cli_default call returns both agg and disagg; cache them together
    args = dict(model=model, shape=shape, epd=epd, main=main_sys, enc=enc_sys if epd else None,
                ttft=ttft, tpot=tpot, backend=backend, total_gpus=total_gpus, top_n=top_n, v=4)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, _key(args) + ".json")
    if os.path.exists(path):
        full = json.load(open(path))
    else:
        full = _run(model, shape, epd, main_sys, args, ttft, tpot, backend, total_gpus, top_n)
        json.dump(full, open(path, "w"))
    return {"args": {**args, "mode": mode}, "rows": full["rows"].get(mode, []), "error": full["error"]}


def _run(model, shape, epd, main_sys, args, ttft, tpot, backend, total_gpus, top_n):
    from aiconfigurator.cli.api import cli_default

    logging.disable(logging.WARNING)
    rows, err = {}, None
    try:
        with open(os.devnull, "w") as null, contextlib.redirect_stdout(null):   # AIC prints banners
            r = cli_default(model, total_gpus, main_sys, backend=backend, isl=shape["isl"], osl=shape["osl"],
                            image_height=shape["image_height"], image_width=shape["image_width"],
                            num_images=shape["num_images"], enable_epd=epd, encoder_system=args["enc"],
                            ttft=ttft, tpot=tpot, top_n=top_n)
        for mode, df in r.best_configs.items():
            if df is None:
                continue
            rows[mode] = []
            for _, row in df.iterrows():
                row = row.to_dict()
                rows[mode].append({
                    "seq_s": float(row["seq/s"]),
                    "gpus": _gpus(row, args["enc"], main_sys),
                    "ttft_ms": float(row["ttft"]),
                    "tpot_ms": float(row["tpot"]),
                    "encoder_ms": float(row.get("encoder_latency") or 0.0),
                    "request_ms": float(row["request_latency"]),
                    "concurrency": int(row["concurrency"]),
                    "shape": _shape_str(row, mode),
                    "backend_version": str(row.get("version", "")),
                })
    except (Exception, SystemExit) as e:   # AIC raises SystemExit on missing perf data
        err = f"{type(e).__name__}: {str(e)[:300]}"
    return {"rows": rows, "error": err}
