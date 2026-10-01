"""Search, bounds, simulation, uncertainty. Profile in, ranked deployments out.

Cost model: each workload bucket is served by a pool shaped by AIConfigurator's best
config for that bucket (equivalent to routing requests by bucket). That is a lower bound
when one shared shape would be deployed; the report flags candidates whose buckets
picked different shapes.
"""
import heapq
import math
import random
import statistics as st
from collections import defaultdict

from . import aic, workload

UTIL = 0.8   # target utilization for every resource


def slo(cfg):
    """Batch mode only needs per-request latency loose enough to batch hard; online uses the real SLO."""
    if cfg["mode"] == "batch":
        b = cfg["batch"].get("slo", {})
        return b.get("ttft_ms", 120000), b.get("tpot_ms", 200)
    return cfg["slo"]["ttft_ms"], cfg["slo"]["tpot_ms"]


# ---------------------------------------------------------------- candidates

def media_types(reqs):
    return sorted({r["type"] for r in reqs})


def allowed_topologies(matrix, stack, types):
    """Topologies every media type in the mix allows, per the pinned support matrix."""
    entry = matrix[stack]
    allowed = None
    for t in types:
        tops = set(entry.get("image" if t == "pdf" else t, []))   # rendered pages are images to the server
        allowed = tops if allowed is None else allowed & tops
    return [t for t in aic.TOPOLOGIES if t in (allowed or set())]


def candidates(cfg, tops):
    out = []
    for main in cfg["main_systems"]:
        for top in tops:
            encs = [None]
            if aic.TOPOLOGIES[top][0]:
                encs = [None] + [e for e in cfg.get("encoder_systems", []) if e != main]
            for enc in encs:
                out.append({"topology": top, "main": main, "encoder": enc or (main if aic.TOPOLOGIES[top][0] else None)})
    return out


# ---------------------------------------------------------------- per-bucket rates

def price_per_hour(gpus, prices):
    return sum(n * prices[s] for s, n in gpus.items())


def evaluate(cand, bks, cfg):
    """Pick, per bucket, the AIC row with the lowest $ per request that meets the SLO."""
    ttft, tpot = slo(cfg)
    per = {}
    for b in bks:
        res = aic.search(cfg["aic_model"], b["shape"], cand["topology"], cand["main"],
                         cand["encoder"] if cand["encoder"] != cand["main"] else None,
                         ttft, tpot, backend=cfg.get("backend", "vllm"), total_gpus=cfg.get("max_gpus_per_replica", 8))
        rows = [r for r in res["rows"] if r["ttft_ms"] <= ttft and r["tpot_ms"] <= tpot]
        if not rows:
            reason = res["error"] or f"no config meets SLO for bucket {b['id']}"
            return None, reason
        if cfg["mode"] == "online":
            lam = cfg["online"]["files_per_s"] * cfg["_req_per_file"] * b["weight"]
            cost = lambda r: math.ceil(lam / (r["seq_s"] * UTIL)) * price_per_hour(r["gpus"], cfg["prices"])
        else:
            cost = lambda r: price_per_hour(r["gpus"], cfg["prices"]) / r["seq_s"]
        per[b["id"]] = min(rows, key=cost)
    return per, None


# ---------------------------------------------------------------- bounds

def files_of(reqs):
    by = defaultdict(list)
    for i, r in enumerate(reqs):
        by[r["file"]].append(i)
    return by


def demand(reqs, idx, bmap, per, cfg, noise=None):
    """Per-file resource demand averaged over the given requests (idx may repeat, from bootstrap)."""
    nfiles = cfg["_nfiles_in"]
    gh = defaultdict(float)
    cpu_ms = emb = kv = nreq = 0.0
    bytes_tok = cfg["_bytes_per_token"]
    for i in idx:
        r = reqs[i]
        row = per[bmap[i]]
        rate = row["seq_s"] * (noise[bmap[i]] if noise else 1.0)
        for s, n in row["gpus"].items():
            gh[s] += n / (rate * 3600)
        cpu_ms += r["cpu_ms"] * cfg.get("cpu_speed_factor", 1.0)
        emb += r["mm_tokens"] * bytes_tok["embedding"]
        kv += r["prompt_tokens"] * bytes_tok["kv"]
        nreq += 1
    f = 1.0 / nfiles
    return {"gpu_h": {s: v * f for s, v in gh.items()}, "cpu_ms": cpu_ms * f, "emb_B": emb * f, "kv_B": kv * f,
            "requests": nreq * f}


def bound(d, cand, cfg):
    """Steady-state sizing for one demand vector. Returns gpus/cores, $ per 1k files, utilizations."""
    prices, mode = cfg["prices"], cfg["mode"]
    if mode == "batch":
        files_s = cfg["batch"]["files"] / (cfg["batch"]["deadline_hours"] * 3600)
    else:
        files_s = cfg["online"]["files_per_s"]
    gpus = {s: max(1, math.ceil(h * 3600 * files_s / UTIL)) for s, h in d["gpu_h"].items()}
    cores = max(1, math.ceil(d["cpu_ms"] / 1000 * files_s / UTIL))
    link_Bps = cfg["link_gbps"] * 1e9 / 8
    util = {f"gpu:{s}": h * 3600 * files_s / gpus[s] for s, h in d["gpu_h"].items()}
    util["cpu"] = d["cpu_ms"] / 1000 * files_s / cores
    if cand["topology"] in ("e_agg", "e_p_d"):
        util["link:encode->prefill"] = d["emb_B"] * files_s / link_Bps
    if cand["topology"] in ("pd", "e_p_d"):
        util["link:prefill->decode"] = d["kv_B"] * files_s / link_Bps
    if mode == "batch":
        # a batch job pays for GPU-hours used; rounding only sets how many GPUs to rent at once
        cost_1k = 1000 * (sum(h * prices[s] for s, h in d["gpu_h"].items())
                          + d["cpu_ms"] / 3.6e6 * cfg.get("cpu_core_price", 0.0))
    else:
        hourly = sum(gpus[s] * prices[s] for s in gpus) + cores * cfg.get("cpu_core_price", 0.0)
        cost_1k = hourly / (files_s * 3600) * 1000
    return {"gpus": gpus, "cores": cores, "cost_per_1k": cost_1k, "util": util,
            "binding": max(util, key=util.get), "links_ok": all(u <= UTIL for k, u in util.items() if k.startswith("link"))}


# ---------------------------------------------------------------- online simulation

def simulate(reqs, bmap, per, cfg, pools, cores, n=4000, seed=0):
    """Discrete-event replay: Poisson request arrivals -> CPU (cores) -> bucket pool (replicas x concurrency slots).
    Returns p95 TTFT and which stage's waiting dominates the slow tail."""
    rnd = random.Random(seed)
    files_s = cfg["online"]["files_per_s"]
    req_s = files_s * len(reqs) / cfg["_nfiles_in"]
    speed = cfg.get("cpu_speed_factor", 1.0)
    cpu_free = [0.0] * cores
    slots = {b: [0.0] * (pools[b] * per[b]["concurrency"]) for b in pools}
    t = 0.0
    rec = []
    for _ in range(n):
        t += rnd.expovariate(req_s)
        i = rnd.randrange(len(reqs))
        r, b = reqs[i], bmap[i]
        row = per[b]
        c0 = heapq.heappop(cpu_free)
        start = max(t, c0)
        done_cpu = start + r["cpu_ms"] * speed / 1000
        heapq.heappush(cpu_free, done_cpu)
        s0 = heapq.heappop(slots[b])
        g_start = max(done_cpu, s0)
        scale = r["prompt_tokens"] / max(1, row.get("_prompt_tokens", r["prompt_tokens"]))
        ttft = row["ttft_ms"] / 1000 * scale
        heapq.heappush(slots[b], g_start + ttft + row["tpot_ms"] / 1000 * r["out_tokens"])
        rec.append((g_start + ttft - t, start - t, g_start - done_cpu, b))
    rec = rec[n // 10:]                       # drop warmup
    p95 = sorted(x[0] for x in rec)[int(0.95 * len(rec))]
    tail = [x for x in rec if x[0] >= p95]
    cpu_wait = st.mean(x[1] for x in tail)
    pool_wait = defaultdict(float)
    for x in tail:
        pool_wait[x[3]] += x[2] / len(tail)
    worst = max(pool_wait, key=pool_wait.get) if pool_wait else None
    if max(cpu_wait, pool_wait.get(worst, 0)) < 0.05 * p95:
        slow = max(set(x[3] for x in tail), key=lambda b: sum(1 for x in tail if x[3] == b))
        return p95 * 1000, f"service:{slow}"
    binding = "cpu" if cpu_wait >= pool_wait.get(worst, 0) else f"pool:{worst}"
    return p95 * 1000, binding


def tighten(reqs, bmap, per, cfg, bks, cand):
    """Size pools per bucket at UTIL, then add capacity to the binding stage until p95 TTFT meets the SLO."""
    files_s = cfg["online"]["files_per_s"]
    req_s = files_s * len(reqs) / cfg["_nfiles_in"]
    pools = {}
    for b in bks:
        lam = req_s * b["weight"]
        pools[b["id"]] = max(1, math.ceil(lam / (per[b["id"]]["seq_s"] * UTIL)))
        per[b["id"]]["_prompt_tokens"] = b["shape"]["visual_tokens"] + b["shape"]["isl"]
    d = demand(reqs, range(len(reqs)), bmap, per, cfg)
    cores = max(1, math.ceil(d["cpu_ms"] / 1000 * files_s / UTIL))
    slo = cfg["slo"]["ttft_ms"]
    for step in range(60):
        p95, binding = simulate(reqs, bmap, per, cfg, pools, cores)
        if p95 <= slo or binding.startswith("service:"):
            break
        if binding == "cpu":
            cores += 1
        else:
            pools[binding.split(":", 1)[1]] += 1
    gpus = defaultdict(int)
    for b, k in pools.items():
        for s, n in per[b]["gpus"].items():
            gpus[s] += k * n
    hourly = sum(gpus[s] * cfg["prices"][s] for s in gpus) + cores * cfg.get("cpu_core_price", 0.0)
    return {"p95_ttft_ms": round(p95, 1), "meets_slo": p95 <= slo, "binding": binding, "pools": pools,
            "gpus": dict(gpus), "cores": cores, "cost_per_1k": hourly / (files_s * 3600) * 1000}


# ---------------------------------------------------------------- driver

def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q / 100 * len(xs)))]


def size(reqs, cfg, unit, matrix, log=print):
    bks = workload.buckets(reqs, unit, k=cfg.get("buckets", 4))
    bmap = workload.assign(reqs, bks)
    for b in bks:
        b["max_prompt_tokens"] = max(reqs[i]["prompt_tokens"] for i in b["idx"])
        b["max_out_tokens"] = max(reqs[i]["out_tokens"] for i in b["idx"])
    types = media_types(reqs)
    tops = allowed_topologies(matrix, cfg["stack"], types)
    cands = candidates(cfg, tops)
    log(f"{len(reqs)} requests, {len(bks)} buckets, media {types}, topologies {tops}, {len(cands)} candidates")
    fmap = files_of(reqs)
    files = list(fmap)
    sigma = cfg.get("model_sigma", 0.10)
    results, rejected = [], []
    for c in cands:
        label = f"{c['topology']} main={c['main']} enc={c['encoder']}"
        per, why = evaluate(c, bks, cfg)
        if per is None:
            rejected.append({**c, "reason": why})
            log(f"  reject {label}: {why[:120]}")
            continue
        base = bound(demand(reqs, range(len(reqs)), bmap, per, cfg), c, cfg)
        # uncertainty: resample files with replacement, perturb each bucket's rate by model error
        rnd = random.Random(1)
        draws = []
        for _ in range(cfg.get("bootstrap", 200)):
            idx = [i for f in (rnd.choice(files) for _ in files) for i in fmap[f]]
            noise = {b["id"]: math.exp(rnd.gauss(0, sigma)) for b in bks}
            draws.append(bound(demand(reqs, idx, bmap, per, cfg, noise), c, cfg))
        res = {**c, "per_bucket": per, "base": base,
               "cost_p10": pct([x["cost_per_1k"] for x in draws], 10),
               "cost_p50": pct([x["cost_per_1k"] for x in draws], 50),
               "cost_p90": pct([x["cost_per_1k"] for x in draws], 90),
               "gpus_p10": {s: pct([x["gpus"].get(s, 0) for x in draws], 10) for s in base["gpus"]},
               "gpus_p90": {s: pct([x["gpus"].get(s, 0) for x in draws], 90) for s in base["gpus"]},
               "shapes": sorted({r["shape"] for r in per.values()}),
               "binding": base["binding"]}
        if not base["links_ok"]:
            res["warning"] = "link over 80% utilization: " + ", ".join(
                f"{k} {v:.0%}" for k, v in base["util"].items() if k.startswith("link") and v > UTIL)
        if cfg["mode"] == "batch":
            # replicas per pool to finish inside the deadline; pools can also time-share one fleet
            files_s = cfg["batch"]["files"] / (cfg["batch"]["deadline_hours"] * 3600)
            req_s = files_s * len(reqs) / cfg["_nfiles_in"]
            res["batch"] = {"pools": {b["id"]: max(1, math.ceil(req_s * b["weight"] / (per[b["id"]]["seq_s"] * UTIL)))
                                      for b in bks}}
        if cfg["mode"] == "online":
            sim = tighten(reqs, bmap, per, cfg, bks, c)
            res["online"] = sim
            # binding: the stage that broke the SLO, else the most utilized resource
            res["binding"] = base["binding"] if sim["meets_slo"] else sim["binding"]
            # simulation sizing replaces the steady-state bound when it needs more capacity
            scale = sim["cost_per_1k"] / base["cost_per_1k"]
            if scale > 1:
                for k in ("cost_p10", "cost_p50", "cost_p90"):
                    res[k] *= scale
            for q in ("gpus_p10", "gpus_p90"):
                res[q] = {g: max(1, round(n * sim["gpus"].get(g, n) / max(1, base["gpus"].get(g, 1))))
                          for g, n in res[q].items()}
        results.append(res)
        log(f"  {label}: ${res['cost_p50']:.3f}/1k files (P10 {res['cost_p10']:.3f}, P90 {res['cost_p90']:.3f}), "
            f"binding {res['binding']}")
    results.sort(key=lambda r: (not r.get("online", {}).get("meets_slo", True), r["cost_p50"]))
    return {"buckets": bks, "results": results, "rejected": rejected, "topologies": tops, "media": types}
