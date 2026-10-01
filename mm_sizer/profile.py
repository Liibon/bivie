"""Workload profiler: turn a folder of real files into requests with exact work counts.

Runs the model's own processor on every request. Processor settings are applied per
call, per media type, mirroring vLLM's per-request mm_processor_kwargs. When the plan
says the server does not take per-request overrides, the image settings apply to
rendered PDF pages too and the profiler reports the DPI pages are actually served at.
One JSONL line per request.
"""
import hashlib
import json
import os
import random
import time
from pathlib import Path

import yaml
from PIL import Image

PDF_EXT = {".pdf"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def kind(path):
    ext = path.suffix.lower()
    for name, exts in (("pdf", PDF_EXT), ("image", IMAGE_EXT), ("video", VIDEO_EXT), ("audio", AUDIO_EXT)):
        if ext in exts:
            return name
    return None


def load_plan(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = Path(path).parent
    rp = cfg["request_plan"]
    tmpl = rp.get("prompt_template")
    rp["prompt_text"] = (base / tmpl).read_text().strip() if tmpl else ""
    return cfg


def sample_files(root, n, seed=0):
    """Stratify by type and size, always keep the largest files of each type."""
    files = [p for p in sorted(Path(root).rglob("*")) if p.is_file() and kind(p)]
    if n is None or n >= len(files):
        return files
    rnd = random.Random(seed)
    by_type = {}
    for p in files:
        by_type.setdefault(kind(p), []).append(p)
    picked = []
    for group in by_type.values():
        quota = max(1, round(n * len(group) / len(files)))
        group = sorted(group, key=lambda p: p.stat().st_size)
        tail = group[-max(1, quota // 10):]          # largest ~10%, drives p95
        rest = group[: len(group) - len(tail)]
        k = min(len(rest), quota - len(tail))
        if k > 0:                                     # spread across size quantiles
            step = len(rest) / k
            picked += [rest[min(len(rest) - 1, int(i * step + rnd.random() * step))] for i in range(k)]
        picked += tail
    return picked


def size_kwargs(opts, default):
    """Plan pixel caps -> per-call size dict, the form current transformers expects."""
    size = {"longest_edge": default["longest_edge"], "shortest_edge": default["shortest_edge"]}
    if "max_pixels" in opts:
        size["longest_edge"] = opts["max_pixels"]
    if "min_pixels" in opts:
        size["shortest_edge"] = opts["min_pixels"]
    return size


class Profiler:
    def __init__(self, model, request_plan):
        from transformers import AutoConfig, AutoProcessor

        self.plan = request_plan
        self.model = model
        self.proc = AutoProcessor.from_pretrained(model)
        ip = self.proc.image_processor
        self.patch = ip.patch_size
        self.unit = ip.patch_size * ip.merge_size          # pixels per LLM token edge
        self.merge = ip.merge_size ** 2
        self.pad_id = self.proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        default = dict(ip.size)
        img = request_plan.get("image", {})
        pdf = request_plan.get("pdf", {})
        self.per_request = request_plan.get("per_request_overrides", True)
        self.sizes = {"image": size_kwargs(img, default)}
        # without per-request overrides the server applies the image cap to every image
        self.sizes["pdf"] = size_kwargs(pdf if self.per_request else img, default)
        self.cfg = AutoConfig.from_pretrained(model)
        self.model_cfg = getattr(self.cfg, "text_config", None) or self.cfg
        self.vision_cfg = getattr(self.cfg, "vision_config", None)

    def _text(self, n_images):
        content = [{"type": "image"}] * n_images + [{"type": "text", "text": self.plan["prompt_text"]}]
        return self.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)

    def count(self, images, typ="image"):
        t = time.perf_counter()
        out = self.proc(text=[self._text(len(images))], images=images,
                        images_kwargs={"size": self.sizes[typ]}, return_tensors="np")
        prep_ms = (time.perf_counter() - t) * 1000
        ids = out["input_ids"]
        mm = int((ids == self.pad_id).sum())
        thw = out["image_grid_thw"]
        return {
            "patches": [int(x) for x in thw.prod(-1)],                      # per image, pre-merge (encoder load)
            "image_tokens": [int(x) // self.merge for x in thw.prod(-1)],
            "image_hw": [[int(h) * self.patch, int(w) * self.patch] for _, h, w in thw],  # resized, as served
            "mm_tokens": mm,                                                 # LLM visual tokens (prefill load)
            "text_tokens": int(ids.shape[1]) - mm,
            "prompt_tokens": int(ids.shape[1]),
            "prep_ms": round(prep_ms, 1),
        }

    def requests_for(self, path):
        k = kind(path)
        if k == "pdf":
            yield from self._pdf(path)
        elif k == "image":
            t = time.perf_counter()
            im = Image.open(path)
            im.load()
            im = im.convert("RGB")
            yield self._record(path, "image", [im], (time.perf_counter() - t) * 1000, pages=None)
        else:
            # video/audio not supported by this version; recorded so the gap shows in the summary
            yield {"file": str(path), "type": k, "skipped": f"{k} not supported yet"}

    def _pdf(self, path):
        import pypdfium2 as pdfium

        opts = self.plan.get("pdf", {})
        dpi = opts.get("dpi", 150)
        per = opts.get("pages_per_request", 1)
        doc = pdfium.PdfDocument(str(path))
        n = len(doc)
        for start in range(0, n, per):
            t = time.perf_counter()
            imgs = [doc[i].render(scale=dpi / 72).to_pil().convert("RGB") for i in range(start, min(n, start + per))]
            rec = self._record(path, "pdf", imgs, (time.perf_counter() - t) * 1000, pages=[start, start + len(imgs)])
            # compare against the uncapped grid; the processor's rounding alone can push a page over the cap
            u = self.unit
            ratio = min(tok / (round(im.height / u) * round(im.width / u))
                        for tok, im in zip(rec["image_tokens"], imgs))
            if ratio < 0.999:
                rec["effective_dpi"] = round(dpi * ratio ** 0.5, 1)
            yield rec
        doc.close()

    def _record(self, path, typ, imgs, render_ms, pages):
        rec = {"file": str(path), "type": typ}
        if pages:
            rec["pages"] = pages
        rec.update(self.count(imgs, typ))
        # exact processor settings this count used; the client must send them when overrides are per request
        rec["mm_processor_kwargs"] = {"size": self.sizes[typ]}
        rec["render_ms"] = round(render_ms, 1)
        rec["cpu_ms"] = round(render_ms + rec["prep_ms"], 1)
        rec["media_hashes"] = [hashlib.sha256(im.tobytes()).hexdigest()[:16] for im in imgs]
        rec["out_tokens"] = None  # filled by a pilot run or the plan's output_tokens
        return rec


def run(plan_path, root, out_path, sample=None, seed=0):
    cfg = load_plan(plan_path)
    prof = Profiler(cfg["model"], cfg["request_plan"])
    files = sample_files(root, sample, seed)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = 0
    with open(out_path, "w") as f:
        for p in files:
            for rec in prof.requests_for(p):
                f.write(json.dumps(rec) + "\n")
                n += 1
    return prof, n
