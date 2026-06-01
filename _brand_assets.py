"""Generate Viva Insights brand raster assets (favicon PNGs + OG share image).
Original artwork aligned to the Viva Insights purple/magenta identity.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(__file__), "assets", "images")
os.makedirs(OUT, exist_ok=True)

PURPLE = (92, 45, 145)     # #5C2D91
MAGENTA = (180, 0, 158)    # #B4009E
SEGUI = r"C:\Windows\Fonts\segoeui.ttf"
SEGUIB = r"C:\Windows\Fonts\segoeuib.ttf"
SEGUISB = r"C:\Windows\Fonts\seguisb.ttf"


def diagonal_gradient(w, h, c0, c1):
    """Return an (h,w,3) uint8 diagonal gradient from c0 (top-left) to c1 (bottom-right)."""
    yy, xx = np.mgrid[0:h, 0:w]
    t = (xx / max(w - 1, 1) + yy / max(h - 1, 1)) / 2.0
    grad = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        grad[..., i] = (c0[i] + (c1[i] - c0[i]) * t).astype(np.uint8)
    return grad


def rounded_mask(w, h, radius, scale=4):
    """Anti-aliased rounded-rectangle alpha mask."""
    m = Image.new("L", (w * scale, h * scale), 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle([0, 0, w * scale - 1, h * scale - 1], radius=radius * scale, fill=255)
    return m.resize((w, h), Image.LANCZOS)


def make_icon(size, pad_ratio=0.0):
    """Square brand mark: gradient rounded square + ascending bars + pulse line."""
    s = size
    grad = Image.fromarray(diagonal_gradient(s, s, PURPLE, MAGENTA), "RGB").convert("RGBA")
    grad.putalpha(rounded_mask(s, s, radius=int(s * 0.22)))

    draw = ImageDraw.Draw(grad)
    u = s / 64.0  # design units (artwork drawn on a 64-grid, matches favicon.svg)
    white = (255, 255, 255, 235)

    bars = [(16, 34, 7, 14), (28.5, 26, 7, 22), (41, 20, 7, 28)]
    for x, y, bw, bh in bars:
        draw.rounded_rectangle([x * u, y * u, (x + bw) * u, (y + bh) * u],
                               radius=2 * u, fill=white)

    pulse = [(14, 24), (24, 24), (29, 15), (36, 31), (41, 22), (50, 22)]
    draw.line([(px * u, py * u) for px, py in pulse],
              fill=(255, 255, 255, 245), width=max(int(2.6 * u), 1), joint="curve")
    return grad


# --- favicons ---
make_icon(32).save(os.path.join(OUT, "favicon-32.png"))
make_icon(180).save(os.path.join(OUT, "apple-touch-icon.png"))
print("favicon-32.png, apple-touch-icon.png written")

# --- OG share image (1200x630) ---
W, H = 1200, 630
og = Image.fromarray(diagonal_gradient(W, H, PURPLE, MAGENTA), "RGB").convert("RGBA")

# subtle darkening at bottom for text legibility
overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
od = ImageDraw.Draw(overlay)
for i in range(H):
    a = int(70 * (i / H) ** 2)
    od.line([(0, i), (W, i)], fill=(20, 0, 30, a))
og = Image.alpha_composite(og, overlay)

d = ImageDraw.Draw(og)
mark = make_icon(132)
og.alpha_composite(mark, (90, 86))

f_eyebrow = ImageFont.truetype(SEGUISB, 30)
f_title = ImageFont.truetype(SEGUIB, 76)
f_sub = ImageFont.truetype(SEGUI, 36)

d.text((238, 110), "MICROSOFT VIVA INSIGHTS", font=f_eyebrow, fill=(255, 255, 255, 220))
d.text((238, 150), "Sample Code Library", font=f_title, fill=(255, 255, 255, 255))

sub = "Tutorials, advanced analytics & AI prompt libraries\nfor Viva Insights — in R and Python."
d.multiline_text((92, 330), sub, font=f_sub, fill=(245, 238, 252, 245), spacing=12)

# footer URL
d.text((92, 540), "microsoft.github.io/viva-insights-sample-code",
       font=ImageFont.truetype(SEGUISB, 30), fill=(255, 255, 255, 230))

og.convert("RGB").save(os.path.join(OUT, "og-image.png"), quality=92)
print("og-image.png written")
