"""Generate the Open Graph social card for the site (1200x630 PNG).

Run from repo root:  python scripts/make_og_image.py
Output:               assets/images/og-image.png
"""
from PIL import Image, ImageDraw, ImageFont
import os

W, H = 1200, 630
img = Image.new("RGB", (W, H), (255, 255, 255))


def mix(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


# Soft horizontal lavender -> white -> pale-blue wash
ribbon_draw = ImageDraw.Draw(img)
for x in range(W):
    t = x / W
    if t < 0.4:
        a = t / 0.4
        c = mix((240, 231, 255), (255, 255, 255), a)
    else:
        a = (t - 0.4) / 0.6
        c = mix((255, 255, 255), (228, 243, 255), a)
    ribbon_draw.line([(x, 0), (x, H)], fill=c)

# Cool-gradient accent bar near the bottom
d = ImageDraw.Draw(img, "RGBA")
C1, C2, C3 = (118, 79, 245), (63, 108, 233), (32, 187, 198)
bar_y, bar_h = H - 90, 6
for x in range(W):
    t = x / W
    c = mix(C1, C2, t / 0.3) if t < 0.3 else mix(C2, C3, (t - 0.3) / 0.7)
    d.line([(x, bar_y), (x, bar_y + bar_h)], fill=c + (255,))

# Brand tile (purple -> magenta) with bars + pulse line
logo_size = 88
lx, ly = 80, 80
logo = Image.new("RGBA", (logo_size, logo_size), (0, 0, 0, 0))
ld = ImageDraw.Draw(logo)
for y in range(logo_size):
    t = y / logo_size
    c = mix((92, 45, 145), (180, 0, 158), t)
    ld.line([(0, y), (logo_size, y)], fill=c + (255,))
mask = Image.new("L", (logo_size, logo_size), 0)
ImageDraw.Draw(mask).rounded_rectangle((0, 0, logo_size, logo_size), 20, fill=255)
logo.putalpha(mask)
ld = ImageDraw.Draw(logo)
ld.rounded_rectangle((22, 48, 32, 68), 3, fill=(255, 255, 255, 235))
ld.rounded_rectangle((39, 36, 49, 68), 3, fill=(255, 255, 255, 235))
ld.rounded_rectangle((56, 28, 66, 68), 3, fill=(255, 255, 255, 235))
ld.line(
    [(20, 34), (33, 34), (40, 21), (49, 43), (56, 30), (68, 30)],
    fill=(255, 255, 255, 245),
    width=3,
    joint="curve",
)
img.paste(logo, (lx, ly), logo)


def font(size, weight="regular"):
    fonts_dir = r"C:\Windows\Fonts"
    by_weight = {
        "bold": ["segoeuib.ttf", "seguisb.ttf", "segoeui.ttf"],
        "semi": ["seguisb.ttf", "segoeuib.ttf", "segoeui.ttf"],
        "regular": ["segoeui.ttf"],
    }
    for c in by_weight.get(weight, ["segoeui.ttf"]):
        p = os.path.join(fonts_dir, c)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# Eyebrow + wordmark next to logo
d.text((lx + logo_size + 24, ly + 10), "VIVA INSIGHTS", font=font(18, "semi"), fill=(51, 92, 204))
d.text((lx + logo_size + 24, ly + 36), "Sample Code Library", font=font(22, "semi"), fill=(14, 23, 38))

# Display title (2 lines, "guides." accented in primary blue)
title_font = font(82, "bold")
title_y = 215
d.text((80, title_y), "Sample code, prompts,", font=title_font, fill=(14, 23, 38))
# Render "and " + "guides." with the latter in primary blue
prefix = "and "
prefix_w = d.textlength(prefix, font=title_font)
d.text((80, title_y + 100), prefix, font=title_font, fill=(14, 23, 38))
d.text((80 + prefix_w, title_y + 100), "guides.", font=title_font, fill=(51, 92, 204))

# Tagline + URL
d.text((80, H - 165), "R \u00b7 Python \u00b7 Copilot \u00b7 Network \u00b7 Causal Inference",
       font=font(26, "regular"), fill=(97, 97, 97))
d.text((80, H - 55), "microsoft.github.io/viva-insights-sample-code",
       font=font(20, "semi"), fill=(51, 92, 204))

out = "assets/images/og-image.png"
img.save(out, "PNG", optimize=True)
print("Saved", out, os.path.getsize(out), "bytes")
