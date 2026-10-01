"""mm-sizer: size multimodal serving from a folder of real files.

  profile PLAN FOLDER [-o out/profile.jsonl] [--sample N]   count work per request
  size    PLAN [-p out/profile.jsonl] [-o out/run]          rank deployments, emit configs
  validate --replay replay.jsonl                            token accounting check
"""
import argparse
import json
import os

from . import profile, summary


def cmd_profile(a):
    prof, n = profile.run(a.plan, a.folder, a.out, a.sample, a.seed)
    s = summary.summarize(a.out, prof.model_cfg, prof.vision_cfg, a.dtype)
    s["model"] = prof.model
    with open(a.out.rsplit(".", 1)[0] + ".summary.json", "w") as f:
        json.dump(s, f, indent=1)
    print(summary.table(s))
    t = s["totals"]
    print(f"\n{s['requests']} requests from {s['files']} files, skipped {s['skipped'] or 'none'}")
    print(f"dup media {s['exact_duplicate_rate']:.1%}, over context {s['over_context']}, "
          f"cpu {t['cpu_s']} s, embeddings {t['embedding_GB']} GB, kv {t['kv_GB']} GB")
    if s["pdf_requests_downscaled"]:
        print(f"WARNING: {s['pdf_requests_downscaled']} pdf requests exceed the pixel cap that applies to pages; "
              f"pages are served at ~{s['pdf_effective_dpi_min']} DPI, not the plan's render DPI. Raise that cap, "
              f"lower pdf.dpi, or set per_request_overrides if the server accepts mm_processor_kwargs per request.")


def cmd_size(a):
    import yaml
    from transformers import AutoConfig

    from . import report, sizer, workload

    plan = profile.load_plan(a.plan)
    cfg = plan["sizing"]
    if a.mode:
        cfg["mode"] = a.mode
    matrix = yaml.safe_load(open(a.matrix))
    reqs = workload.load_requests(a.profile, cfg["output_tokens"])
    skipped = {}
    for l in open(a.profile):
        r = json.loads(l)
        if "skipped" in r:
            skipped[r["type"]] = skipped.get(r["type"], 0) + 1
    mcfg = AutoConfig.from_pretrained(plan["model"])
    text = getattr(mcfg, "text_config", None) or mcfg
    vis = getattr(mcfg, "vision_config", None)
    cfg["_bytes_per_token"] = summary.link_bytes_per_token(text, vis)
    cfg["_nfiles_in"] = len({r["file"] for r in reqs})
    cfg["_req_per_file"] = len(reqs) / cfg["_nfiles_in"]
    unit = vis.patch_size * vis.spatial_merge_size
    out = sizer.size(reqs, cfg, unit, matrix)
    out["skipped"] = skipped
    md = report.write_all(out, cfg, plan, a.profile, a.out)
    if out["results"] and not a.no_configs:
        made = report.emit_configs(out, cfg, plan, a.out)
        md += "\n## Generated configs\n\n" + "\n".join(f"- {m}" for m in made) + "\n"
        open(os.path.join(a.out, "report.md"), "w").write(md)
    print(md)
    print(f"wrote {a.out}/report.md, results.json, assumptions.yaml, replay.py")


def cmd_validate(a):
    from . import validate

    print(json.dumps(validate.token_accounting(a.replay), indent=1))


def main():
    ap = argparse.ArgumentParser(prog="mm-sizer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile", help="count work per request for a folder of real files")
    p.add_argument("plan")
    p.add_argument("folder")
    p.add_argument("-o", "--out", default="out/profile.jsonl")
    p.add_argument("--sample", type=int, help="stratified sample size (files); default all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    p.set_defaults(fn=cmd_profile)
    s = sub.add_parser("size", help="rank deployments for a profiled workload")
    s.add_argument("plan")
    s.add_argument("-p", "--profile", default="out/profile.jsonl")
    s.add_argument("-o", "--out", default="out/run")
    s.add_argument("--matrix", default="support_matrix.yaml")
    s.add_argument("--mode", choices=["batch", "online"])
    s.add_argument("--no-configs", action="store_true")
    s.set_defaults(fn=cmd_size)
    v = sub.add_parser("validate", help="check a replay run's token accounting")
    v.add_argument("--replay", required=True)
    v.set_defaults(fn=cmd_validate)
    a = ap.parse_args()
    a.fn(a)


main()
