"""Generate the 1200x630 social-share card for the causal-questions article.

Run from the repo root:
  python assets/images/articles/causal-questions-ai-usage-card.py

Output: assets/images/articles/causal-questions-ai-usage-card.png
"""
from __future__ import annotations
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1200, 630
ARTICLE_MD = Path(__file__).resolve().parents[3] / "_pages" / "causal-questions-ai-usage.md"
NAVY = (17, 54, 90)
NAVY_DEEP = (12, 39, 64)
GOLD = (233, 200, 119)
WHITE = (255, 255, 255)
INK_SOFT = (215, 228, 241)


def vertical_gradient(size, top, bottom):
    img = Image.new("RGB", size, top)
    draw = ImageDraw.Draw(img)
    for y in range(size[1]):
        t = y / max(1, size[1] - 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        draw.line([(0, y), (size[0], y)], fill=(r, g, b))
    return img


def add_radial_glow(base, center, radius, color, alpha=180):
    glow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx, cy = center
    for r in range(radius, 0, -8):
        a = int(alpha * (r / radius) * 0.18)
        gd.ellipse((cx - r, cy - r, cx + r, cy + r),
                   fill=(color[0], color[1], color[2], a))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=40))
    base.alpha_composite(glow)


def front_matter_value(md_path, key):
    text = md_path.read_text(encoding="utf-8")
    fm = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", text, re.S)
    if not fm:
        raise RuntimeError(f"No YAML front matter found in {md_path}")
    value = re.search(rf'^{re.escape(key)}:\s*"([^"]+)"\s*$', fm.group(1), re.M)
    if not value:
        raise RuntimeError(f"{key} missing from front matter of {md_path}")
    return value.group(1)


def composite_translucent_motif(base):
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for r in (80, 140, 200, 260, 320):
        od.ellipse((960 - r, 120 - r, 960 + r, 120 + r),
                   outline=(255, 255, 255, 30), width=1)
    for offset in (0, 40, 80):
        od.line([(440 + offset, 0), (1200, 760 - offset)],
                fill=(GOLD[0], GOLD[1], GOLD[2], 70), width=1)
    base.alpha_composite(overlay)


def load_font(candidates, size):
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap(draw, text, font, max_width):
    words = text.split()
    lines, line = [], []
    for w in words:
        trial = " ".join(line + [w])
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width or not line:
            line.append(w)
        else:
            lines.append(" ".join(line))
            line = [w]
    if line:
        lines.append(" ".join(line))
    return lines


def main():
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "causal-questions-ai-usage-card.png"

    bg = vertical_gradient((W, H), NAVY, NAVY_DEEP).convert("RGBA")
    add_radial_glow(bg, center=(960, 120), radius=420, color=GOLD, alpha=180)
    composite_translucent_motif(bg)

    draw = ImageDraw.Draw(bg)
    draw.rectangle((0, H - 6, W, H), fill=GOLD)

    pad_x, pad_top, text_max = 72, 96, 560

    eyebrow_font = load_font(
        ["seguisb.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"], 22
    )
    draw.text((pad_x, pad_top),
              "COPILOT ANALYTICS LAB  \u00b7  PANDAS TEAM",
              font=eyebrow_font, fill=GOLD)

    title_font = load_font(
        ["georgiab.ttf", "Georgia Bold.ttf", "times.ttf",
         "DejaVuSerif-Bold.ttf"],
        72,
    )
    y = pad_top + 56
    title_step = title_font.getbbox("Ag")[3] - title_font.getbbox("Ag")[1] + 8
    for line in wrap(draw, front_matter_value(ARTICLE_MD, "title"), title_font,
                     text_max):
        draw.text((pad_x, y), line, font=title_font, fill=WHITE)
        y += title_step

    dek_font = load_font(
        ["georgiai.ttf", "Georgia Italic.ttf", "DejaVuSerif.ttf"], 30
    )
    dek = "A decision guide for staggered, function-led AI rollouts."
    y += 18
    for line in wrap(draw, dek, dek_font, text_max):
        draw.text((pad_x, y), line, font=dek_font, fill=INK_SOFT)
        bbox = draw.textbbox((0, 0), line, font=dek_font)
        y += (bbox[3] - bbox[1]) + 6

    foot_font = load_font(
        ["seguisb.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"], 22
    )
    read_time = front_matter_value(ARTICLE_MD, "read_time")
    draw.text((pad_x, H - 72),
              f"A Copilot Analytics Lab methods guide  \u00b7  {read_time}",
              font=foot_font, fill=(255, 255, 255, 200))

    bg.convert("RGB").save(out_path, "PNG", optimize=True)
    print(f"Wrote {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
