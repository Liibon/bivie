"""Pins worked numbers to the real processors. Run: .venv/bin/python -m unittest discover tests"""
import unittest

from PIL import Image

from mm_sizer.profile import Profiler
from mm_sizer.summary import link_bytes_per_token


def letter(dpi):
    return Image.new("RGB", (round(8.5 * dpi), 11 * dpi), "white")


class Qwen25(unittest.TestCase):
    """The design doc's table (Qwen2.5-VL-7B, 28 px per token)."""
    M = "Qwen/Qwen2.5-VL-7B-Instruct"

    @classmethod
    def setUpClass(cls):
        cls.p = Profiler(cls.M, {"prompt_text": "Extract."})

    def test_letter_page_tokens(self):
        for dpi, tok, patches in ((100, 1170, 4680), (150, 2714, 10856), (300, 10738, 42952)):
            c = self.p.count([letter(dpi)])
            self.assertEqual((c["mm_tokens"], c["patches"][0]), (tok, patches), dpi)

    def test_bytes_per_token(self):
        self.assertEqual(link_bytes_per_token(self.p.model_cfg, self.p.vision_cfg), {"embedding": 7168, "kv": 57344})

    def test_image_cap_spares_pdf_with_per_request_overrides(self):
        plan = {"prompt_text": "x", "image": {"max_pixels": 1003520}, "per_request_overrides": True}
        p = Profiler(self.M, plan)
        self.assertLessEqual(p.count([letter(150)], "image")["mm_tokens"], 1280)
        self.assertEqual(p.count([letter(150)], "pdf")["mm_tokens"], 2714)

    def test_shared_cap_hits_pdf_pages(self):
        plan = {"prompt_text": "x", "image": {"max_pixels": 1003520}, "per_request_overrides": False}
        self.assertLessEqual(Profiler(self.M, plan).count([letter(150)], "pdf")["mm_tokens"], 1280)


class Qwen3(unittest.TestCase):
    """Qwen3-VL-8B: 32 px per token, deepstack makes embeddings 4x hidden."""
    M = "Qwen/Qwen3-VL-8B-Instruct"

    @classmethod
    def setUpClass(cls):
        cls.p = Profiler(cls.M, {"prompt_text": "Extract."})

    def test_letter_page_tokens(self):
        for dpi, tok in ((100, 918), (150, 2080), (300, 8240)):
            self.assertEqual(self.p.count([letter(dpi)])["mm_tokens"], tok, dpi)

    def test_bytes_per_token(self):
        self.assertEqual(link_bytes_per_token(self.p.model_cfg, self.p.vision_cfg), {"embedding": 32768, "kv": 147456})


if __name__ == "__main__":
    unittest.main()
