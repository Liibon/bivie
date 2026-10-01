#!/usr/bin/env python3
"""Replay profiled real-file requests against a deployed OpenAI-compatible endpoint.

Validation check 1 (token accounting): server usage.prompt_tokens must equal the
profile's prompt_tokens for every request. Also records TTFT, ITL, E2E per request.

Env: BASE_URL (required), MODEL (required), PROFILE (profile.jsonl), PLAN (plan yaml),
     RATE (requests/s, Poisson; 0 = closed loop), CONCURRENCY (16), MAX_TOKENS (512),
     LIMIT (requests, default all), OUT (replay.jsonl)
"""
import base64
import io
import json
import os
import random
import sys
import threading
import time
import urllib.request

import yaml
from PIL import Image

BASE = os.environ["BASE_URL"].rstrip("/")
MODEL = os.environ["MODEL"]
PROFILE = os.getenv("PROFILE", "out/profile.jsonl")
PLAN = os.getenv("PLAN", "plan.example.yaml")
RATE = float(os.getenv("RATE", "0"))
CONC = int(os.getenv("CONCURRENCY", "16"))
MAX_TOK = int(os.getenv("MAX_TOKENS", "512"))
LIMIT = int(os.getenv("LIMIT", "0"))
OUT = os.getenv("OUT", "replay.jsonl")

plan = yaml.safe_load(open(PLAN))
rp = plan["request_plan"]
prompt = open(os.path.join(os.path.dirname(PLAN) or ".", rp["prompt_template"])).read().strip() if rp.get("prompt_template") else ""


RENDER = threading.Lock()   # pdfium is not thread-safe


def images_for(rec):
    if rec["type"] == "pdf":
        import pypdfium2 as pdfium
        with RENDER:
            return _render(pdfium, rec)
    return [Image.open(rec["file"]).convert("RGB")]


def _render(pdfium, rec):
    doc = pdfium.PdfDocument(rec["file"])
    dpi = rp.get("pdf", {}).get("dpi", 150)
    a, b = rec["pages"]
    ims = [doc[i].render(scale=dpi / 72).to_pil().convert("RGB") for i in range(a, b)]
    doc.close()
    return ims


def body(rec):
    content = []
    for im in images_for(rec):
        buf = io.BytesIO()
        im.save(buf, "PNG")
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}})
    content.append({"type": "text", "text": prompt})
    b = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": MAX_TOK,
         "stream": True, "stream_options": {"include_usage": True}, "temperature": 0}
    if rp.get("per_request_overrides"):
        b["mm_processor_kwargs"] = rec["mm_processor_kwargs"]   # exactly what the profile counted with
    return b


def send(rec, lock, fout):
    t0 = time.perf_counter()
    stamps, usage, err = [], None, None
    try:
        req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body(rec)).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.getenv("API_KEY", "none")})
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                if ev.get("usage"):
                    usage = ev["usage"]
                if ev.get("choices") and ev["choices"][0].get("delta", {}).get("content"):
                    stamps.append(time.perf_counter())
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    row = {"file": rec["file"], "pages": rec.get("pages"), "bucket_hint": rec["mm_tokens"],
           "predicted_prompt_tokens": rec["prompt_tokens"],
           "server_prompt_tokens": usage and usage.get("prompt_tokens"),
           "out_tokens": usage and usage.get("completion_tokens"),
           "ttft_ms": (stamps[0] - t0) * 1000 if stamps else None,
           "itl_ms": ((stamps[-1] - stamps[0]) / (len(stamps) - 1) * 1000) if len(stamps) > 1 else None,
           "e2e_ms": (time.perf_counter() - t0) * 1000, "error": err}
    with lock:
        fout.write(json.dumps(row) + "\n")
        fout.flush()


def main():
    recs = [json.loads(l) for l in open(PROFILE)]
    recs = [r for r in recs if "skipped" not in r]
    if LIMIT:
        recs = recs[:LIMIT]
    lock, sem = threading.Lock(), threading.Semaphore(CONC)
    threads = []
    with open(OUT, "w") as fout:
        for rec in recs:
            if RATE > 0:
                time.sleep(random.expovariate(RATE))
            sem.acquire()
            t = threading.Thread(target=lambda r=rec: (send(r, lock, fout), sem.release()))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    rows = [json.loads(l) for l in open(OUT)]
    ok = [r for r in rows if r["server_prompt_tokens"] is not None]
    exact = sum(r["server_prompt_tokens"] == r["predicted_prompt_tokens"] for r in ok)
    print(f"{len(rows)} sent, {len(rows) - len(ok)} errors, token accounting exact {exact}/{len(ok)}")
    sys.exit(0 if ok and exact == len(ok) else 1)


if __name__ == "__main__":
    main()
