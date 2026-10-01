"""Validation checks 2 and 3: stage accuracy against measurements, decision regret."""
import json


def token_accounting(replay_path):
    rows = [json.loads(l) for l in open(replay_path)]
    ok = [r for r in rows if r["server_prompt_tokens"] is not None]
    miss = [r for r in ok if r["server_prompt_tokens"] != r["predicted_prompt_tokens"]]
    return {"requests": len(ok), "exact": len(ok) - len(miss), "pass": bool(ok) and not miss,
            "worst": max(miss, key=lambda r: abs(r["server_prompt_tokens"] - r["predicted_prompt_tokens"]), default=None)}


def stage_error(predicted, measured):
    """predicted/measured: {stage: value}. Pass bars: encoder 10%, prefill/decode 15%."""
    bars = {"encoder": 0.10, "prefill": 0.15, "decode": 0.15, "ttft": 0.15, "tpot": 0.15}
    out = {}
    for k, m in measured.items():
        if k in predicted and m:
            e = predicted[k] / m - 1
            out[k] = {"error": round(e, 3), "pass": abs(e) <= bars.get(k, 0.15)}
    return out


def regret(chosen_cost, measured):
    """measured: [{name, cost, meets_slo}]. Regret = chosen / best feasible measured - 1. Pass < 10%."""
    feas = [m["cost"] for m in measured if m["meets_slo"]]
    if not feas:
        return {"regret": None, "pass": False, "note": "no measured config met the SLO"}
    r = chosen_cost / min(feas) - 1
    return {"regret": round(r, 4), "pass": r < 0.10}


def p95_check(predicted_ms, replay_path):
    rows = [json.loads(l) for l in open(replay_path)]
    ttfts = sorted(r["ttft_ms"] for r in rows if r["ttft_ms"] is not None)
    p95 = ttfts[int(0.95 * len(ttfts))] if ttfts else None
    e = predicted_ms / p95 - 1 if p95 else None
    return {"measured_p95_ttft_ms": p95, "predicted": predicted_ms, "error": e, "pass": e is not None and abs(e) <= 0.20}
