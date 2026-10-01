"""Sizer logic with AIConfigurator stubbed out: fast, deterministic."""
import unittest
from unittest import mock

from mm_sizer import sizer, workload


def reqs(n_small=40, n_big=10):
    out = []
    for i in range(n_small):
        out.append({"file": f"img{i}", "type": "image", "image_tokens": [1200], "image_hw": [[1024, 1216]],
                    "mm_tokens": 1200, "text_tokens": 20, "prompt_tokens": 1220, "cpu_ms": 50, "out_tokens": 256})
    for i in range(n_big):
        for k in range(3):
            out.append({"file": f"pdf{i}", "type": "pdf", "image_tokens": [2080] * 4, "image_hw": [[1664, 1280]] * 4,
                        "mm_tokens": 8320, "text_tokens": 25, "prompt_tokens": 8345, "cpu_ms": 200, "out_tokens": 256})
    return out


def fake_search(model, shape, topology, main, enc, ttft, tpot, **kw):
    heavy = shape["visual_tokens"] > 5000
    epd = topology in ("e_agg", "e_p_d")
    gpus = {main: 2}
    if epd:
        gpus[enc or main] = gpus.get(enc or main, 0) + 1
    seq = (2.0 if heavy else 10.0) * (1.4 if epd else 1.0)
    row = {"seq_s": seq, "gpus": gpus, "ttft_ms": 800 if epd else 1500, "tpot_ms": 20, "encoder_ms": 200,
           "request_ms": 6000, "concurrency": 16, "shape": "A:1xtp2" + (" E:1xtp1" if epd else "")}
    return {"rows": [row], "error": None}


CFG = {"aic_model": "m", "stack": "s", "main_systems": ["h100_sxm"], "encoder_systems": ["l40s"],
       "prices": {"h100_sxm": 2.5, "l40s": 1.0}, "cpu_core_price": 0.04, "link_gbps": 100,
       "slo": {"ttft_ms": 3000, "tpot_ms": 40}, "batch": {"files": 100000, "deadline_hours": 24},
       "online": {"files_per_s": 1.0}, "bootstrap": 50, "buckets": 3,
       "_bytes_per_token": {"embedding": 32768, "kv": 147456}}
MATRIX = {"s": {"image": ["agg", "pd", "e_agg", "e_p_d"], "video": ["agg", "pd"]}}


class Buckets(unittest.TestCase):
    def test_weights_sum_and_tail(self):
        r = reqs()
        bks = workload.buckets(r, 32, k=3)
        self.assertAlmostEqual(sum(b["weight"] for b in bks), 1.0)
        self.assertEqual(sorted(i for b in bks for i in b["idx"]), list(range(len(r))))
        self.assertEqual(max(b["shape"]["visual_tokens"] for b in bks), 8320)

    def test_identical_shapes_merge(self):
        bks = workload.buckets(reqs(n_small=0, n_big=10), 32, k=4)
        self.assertEqual(len(bks), 1)


class Size(unittest.TestCase):
    def run_mode(self, mode):
        r = reqs()
        cfg = dict(CFG, mode=mode, _nfiles_in=len({x["file"] for x in r}))
        cfg["_req_per_file"] = len(r) / cfg["_nfiles_in"]
        with mock.patch.object(sizer.aic, "search", fake_search):
            return sizer.size(r, cfg, 32, MATRIX, log=lambda *a: None)

    def test_batch_ranks_by_cost_and_ranges_bracket(self):
        out = self.run_mode("batch")
        costs = [x["cost_p50"] for x in out["results"]]
        self.assertEqual(costs, sorted(costs))
        for x in out["results"]:
            self.assertLessEqual(x["cost_p10"], x["cost_p50"])
            self.assertLessEqual(x["cost_p50"], x["cost_p90"])

    def test_video_filters_topologies(self):
        self.assertEqual(sizer.allowed_topologies(MATRIX, "s", ["image", "video"]), ["agg", "pd"])
        self.assertEqual(sizer.allowed_topologies(MATRIX, "s", ["pdf"]), ["agg", "pd", "e_agg", "e_p_d"])

    def test_online_meets_slo_or_names_cause(self):
        out = self.run_mode("online")
        for x in out["results"]:
            o = x["online"]
            self.assertTrue(o["meets_slo"] or o["binding"].startswith("service"), o)

    def test_hand_computed_batch_cost(self):
        # agg on h100: per file GPU-hours = sum over requests of 2 GPUs / (seq_s * 3600)
        out = self.run_mode("batch")
        agg = next(x for x in out["results"] if x["topology"] == "agg")
        r = reqs()
        gh = sum(2 / ((2.0 if q["mm_tokens"] > 5000 else 10.0) * 3600) for q in r) / 50
        cpu_h = sum(q["cpu_ms"] for q in r) / 50 / 3.6e6
        self.assertAlmostEqual(agg["base"]["cost_per_1k"], 1000 * (gh * 2.5 + cpu_h * 0.04), places=6)


if __name__ == "__main__":
    unittest.main()
