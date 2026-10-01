"""Profile JSONL -> weighted workload buckets AIConfigurator can take.

AIConfigurator describes a workload as one (isl, osl, image_h, image_w, num_images).
We split the real request distribution into quantile buckets on visual tokens, keep a
separate tail bucket for the top requests (they drive p95), and give each bucket a
representative shape whose token counts match the bucket's mean.
"""
import json
import math
import random
import statistics as st


def load_requests(path, out_tokens):
    """Requests with an output length attached. out_tokens: int, or {dist: [values]} to sample from."""
    reqs = [json.loads(l) for l in open(path)]
    reqs = [r for r in reqs if "skipped" not in r]
    rnd = random.Random(0)
    for r in reqs:
        if r.get("out_tokens") is None:
            r["out_tokens"] = out_tokens if isinstance(out_tokens, int) else rnd.choice(out_tokens["dist"])
    return reqs


def _shape(rs, unit):
    """Representative request: mean images, mean tokens per image at the median aspect ratio."""
    n_img = max(1, round(st.mean(len(r["image_tokens"]) for r in rs)))
    tok_per_img = st.mean(t for r in rs for t in r["image_tokens"])
    aspect = st.median(h / w for r in rs for h, w in r["image_hw"])          # h / w
    w_tok = max(1, round(math.sqrt(tok_per_img / aspect)))
    h_tok = max(1, round(tok_per_img / w_tok))
    return {
        "num_images": n_img,
        "image_height": h_tok * unit,
        "image_width": w_tok * unit,
        "isl": max(1, round(st.mean(r["text_tokens"] for r in rs))),   # AIC adds image tokens itself
        "osl": max(1, round(st.mean(r["out_tokens"] for r in rs))),
        "visual_tokens": n_img * h_tok * w_tok,
    }


def buckets(reqs, unit, k=4, tail=0.05):
    """Up to k buckets cut at visual-token quantiles plus a tail bucket above the (1 - tail) quantile.
    Cuts are on values, so identical requests always share a bucket. Returns [{id, shape, idx, weight}]."""
    vals = sorted(r["mm_tokens"] for r in reqs)
    n = len(vals)
    q = lambda f: vals[min(n - 1, int(f * n))]
    tail_cut = q(1 - tail) if n >= 20 else None
    cuts = sorted({q((j + 1) / k) for j in range(k - 1)} | {vals[-1]})
    if tail_cut is not None:
        cuts = sorted({c for c in cuts if c < tail_cut} | {tail_cut, vals[-1]})
    groups, lo = [], None
    for c in cuts:
        g = [i for i, r in enumerate(reqs) if (lo is None or r["mm_tokens"] > lo) and r["mm_tokens"] <= c]
        if g:
            groups.append(g)
        lo = c
    out = []
    for j, g in enumerate(groups):
        rs = [reqs[i] for i in g]
        is_tail = tail_cut is not None and min(r["mm_tokens"] for r in rs) > tail_cut
        b = {"id": f"b{j}" + ("_tail" if is_tail else ""), "idx": g, "weight": len(g) / n}
        b["shape"] = _shape(rs, unit)
        same = next((o for o in out if o["shape"] == b["shape"]), None)
        if same:                                   # identical shape: one bucket, one search
            same["idx"] += b["idx"]
            same["weight"] += b["weight"]
            same["id"] += "+" + b["id"]
        else:
            out.append(b)
    return out


def assign(reqs, bks):
    """request index -> bucket id"""
    m = {}
    for b in bks:
        for i in b["idx"]:
            m[i] = b["id"]
    return m
