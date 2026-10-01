"""Synthetic file mix: Letter PDFs of several lengths, mixed-size images, one exact duplicate."""
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def page(i, w=850, h=1100):
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    for y in range(60, h - 60, 24):
        d.text((60, y), f"page {i} line {y} " + "x" * random.randint(10, 60), fill="black")
    return im


def main(root):
    random.seed(0)
    root = Path(root)
    shutil.rmtree(root, ignore_errors=True)
    (root / "pdf").mkdir(parents=True)
    (root / "img").mkdir()
    for n in (1, 3, 9, 40):
        pages = [page(i) for i in range(n)]
        # 850x1100 px at resolution 100 -> 8.5x11 in page
        pages[0].save(root / "pdf" / f"doc{n}.pdf", save_all=True, append_images=pages[1:], resolution=100)
    for i, (w, h) in enumerate([(640, 480), (1920, 1080), (4032, 3024), (300, 300)]):
        Image.new("RGB", (w, h), (i * 40, 100, 200)).save(root / "img" / f"photo{i}.png")
    shutil.copy(root / "img" / "photo1.png", root / "img" / "photo1_copy.png")
    (root / "clip.mp4").write_bytes(b"")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures")
