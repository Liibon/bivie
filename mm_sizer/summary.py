"""Summarize a profile JSONL: work distributions, duplicate rate, link bytes, context overflow."""
import json
from collections import Counter

DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "fp8": 1}


def link_bytes_per_token(text_cfg, vision_cfg=None, dtype="bfloat16"):
    b = DTYPE_BYTES[dtype]
    head_dim = getattr(text_cfg, "head_dim", None) or text_cfg.hidden_size // text_cfg.num_attention_heads
    # encoder output per visual token: the merger's output plus one vector per deepstack layer (Qwen3-VL)
    width = getattr(vision_cfg, "out_hidden_size", None) or text_cfg.hidden_size
    stacks = 1 + len(getattr(vision_cfg, "deepstack_visual_indexes", None) or ())
    return {
        "embedding": width * stacks * b,                                                # encode -> prefill
        "kv": 2 * text_cfg.num_hidden_layers * text_cfg.num_key_value_heads * head_dim * b,  # prefill -> decode
    }


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q / 100 * len(xs)))]


def dist(xs):
    return {"n": len(xs), "p50": pct(xs, 50), "p90": pct(xs, 90), "p95": pct(xs, 95), "max": max(xs) if xs else None,
            "mean": round(sum(xs) / len(xs), 1) if xs else None}


def summarize(path, text_cfg, vision_cfg=None, dtype="bfloat16"):
    recs = [json.loads(l) for l in open(path)]
    done = [r for r in recs if "skipped" not in r]
    skipped = Counter(r["type"] for r in recs if "skipped" in r)
    per_tok = link_bytes_per_token(text_cfg, vision_cfg, dtype)
    max_pos = getattr(text_cfg, "max_position_embeddings", None)

    hashes = [h for r in done for h in r["media_hashes"]]
    dup_rate = 1 - len(set(hashes)) / len(hashes) if hashes else 0.0

    by_type = {}
    for r in done:
        by_type.setdefault(r["type"], []).append(r)

    out = {
        "requests": len(done),
        "files": len({r["file"] for r in done}),
        "skipped": dict(skipped),
        "bytes_per_token": per_tok,
        "media_items": len(hashes),
        "exact_duplicate_rate": round(dup_rate, 4),   # embedding-cache hit ceiling for the encoder
        "over_context": sum(1 for r in done if max_pos and r["prompt_tokens"] > max_pos),
        "max_position_embeddings": max_pos,
        "by_type": {},
    }
    for t, rs in by_type.items():
        mm = [r["mm_tokens"] for r in rs]
        out["by_type"][t] = {
            "prompt_tokens": dist([r["prompt_tokens"] for r in rs]),
            "mm_tokens": dist(mm),
            "patches_per_request": dist([sum(r["patches"]) for r in rs]),
            "cpu_ms": dist([r["cpu_ms"] for r in rs]),
            "embedding_MB": dist([round(m * per_tok["embedding"] / 1e6, 1) for m in mm]),
            "kv_MB": dist([round(r["prompt_tokens"] * per_tok["kv"] / 1e6, 1) for r in rs]),
        }
    capped = [r["effective_dpi"] for r in done if "effective_dpi" in r]
    out["pdf_requests_downscaled"] = len(capped)
    out["pdf_effective_dpi_min"] = min(capped) if capped else None
    tot = lambda k: sum(r[k] for r in done)
    out["totals"] = {
        "prompt_tokens": tot("prompt_tokens"),
        "mm_tokens": tot("mm_tokens"),
        "patches": sum(sum(r["patches"]) for r in done),
        "cpu_s": round(tot("cpu_ms") / 1000, 2),
        "embedding_GB": round(tot("mm_tokens") * per_tok["embedding"] / 1e9, 3),
        "kv_GB": round(tot("prompt_tokens") * per_tok["kv"] / 1e9, 3),
    }
    return out


def table(s):
    rows = ["| type | requests | prompt tok p50/p95/max | patches p95 | cpu ms p95 | emb MB p95 | kv MB p95 |",
            "|---|---|---|---|---|---|---|"]
    for t, d in s["by_type"].items():
        p = d["prompt_tokens"]
        rows.append(f"| {t} | {p['n']} | {p['p50']}/{p['p95']}/{p['max']} | {d['patches_per_request']['p95']} | "
                    f"{d['cpu_ms']['p95']} | {d['embedding_MB']['p95']} | {d['kv_MB']['p95']} |")
    return "\n".join(rows)
