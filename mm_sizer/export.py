"""Render each profiled request once into a ready-to-send OpenAI chat body for bivie-replay (Go).

One JSONL line per request: {file, pages, predicted_prompt_tokens, body}. Images are base64 PNG,
rendered with the plan's settings. With per_request_overrides the body carries the exact
mm_processor_kwargs the profile counted with, so server prompt_tokens must match exactly.
"""
import base64
import io
import json
import os

from PIL import Image


def images_for(rec, plan):
    if rec["type"] == "pdf":
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(rec["file"])
        dpi = plan.get("pdf", {}).get("dpi", 150)
        a, b = rec["pages"]
        ims = [doc[i].render(scale=dpi / 72).to_pil().convert("RGB") for i in range(a, b)]
        doc.close()
        return ims
    return [Image.open(rec["file"]).convert("RGB")]


def body(rec, plan, model, max_tokens):
    content = []
    for im in images_for(rec, plan):
        buf = io.BytesIO()
        im.save(buf, "PNG")
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}})
    content.append({"type": "text", "text": plan["prompt_text"]})
    b = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
         "stream": True, "stream_options": {"include_usage": True}, "temperature": 0}
    if plan.get("per_request_overrides"):
        b["mm_processor_kwargs"] = rec["mm_processor_kwargs"]   # exactly what the profile counted with
    return b


def run(profile_path, plan, model, out_path, max_tokens=512, limit=0):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = 0
    with open(profile_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            if "skipped" in rec:
                continue
            out = {"file": rec["file"], "pages": rec.get("pages"), "predicted_prompt_tokens": rec["prompt_tokens"],
                   "body": body(rec, plan, model, max_tokens)}
            fout.write(json.dumps(out) + "\n")
            n += 1
            if limit and n >= limit:
                break
    return n
