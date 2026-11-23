#!/usr/bin/env python3
"""Generate the test images with PIL: python img_make.py OUT_DIR"""
import sys, math, random
from PIL import Image, ImageDraw, ImageFont
out = sys.argv[1]
def font(n):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try: return ImageFont.truetype(p, n)
        except OSError: pass
    return ImageFont.load_default()
# 1 text, small
im = Image.new("RGB", (320, 120), "white"); d = ImageDraw.Draw(im)
d.text((12, 20), "TensorFold 4821", fill="black", font=font(40)); d.text((12, 75), "Metal on Apple M4 Pro", fill="navy", font=font(22))
im.save(f"{out}/text-small.png")
# 2 bar chart
im = Image.new("RGB", (640, 420), "white"); d = ImageDraw.Draw(im)
vals = [("Mon", 3), ("Tue", 7), ("Wed", 5), ("Thu", 9), ("Fri", 4)]
d.text((200, 10), "Sales per weekday", fill="black", font=font(24))
d.line((60, 60, 60, 360), fill="black", width=2); d.line((60, 360, 600, 360), fill="black", width=2)
for i, (n, v) in enumerate(vals):
    x = 90 + i * 100; d.rectangle((x, 360 - v * 30, x + 60, 360), fill=("red", "blue", "green", "orange", "purple")[i])
    d.text((x + 10, 368), n, fill="black", font=font(18)); d.text((x + 22, 340 - v * 30), str(v), fill="black", font=font(18))
im.save(f"{out}/chart.png"); im.save(f"{out}/chart.jpg", quality=92)
# 3 gradient scene: sky, sun, hills, sea
W, H = 1024, 768; im = Image.new("RGB", (W, H)); px = im.load()
for y in range(H):
    for x in range(W):
        if y < 420: t = y / 420; px[x, y] = (int(255 - 120 * t), int(140 + 60 * t), int(60 + 180 * t))
        else: t = (y - 420) / 348; px[x, y] = (int(20 + 20 * t), int(90 - 40 * t), int(160 - 60 * t))
d = ImageDraw.Draw(im); d.ellipse((700, 250, 820, 370), fill=(255, 235, 120))
d.polygon([(0, 420), (200, 300), (420, 420)], fill=(40, 90, 50)); d.polygon([(250, 420), (520, 270), (800, 420)], fill=(30, 70, 40))
im.save(f"{out}/scene.jpg", quality=90); im.save(f"{out}/scene.png")
# 4 large document-like image
W, H = 1792, 1344; im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
d.text((60, 40), "INVOICE No. 20931", fill="black", font=font(72))
for i in range(8):
    d.text((60, 200 + i * 110), f"Item {i+1}: widget type {chr(65+i)}   qty {i+2}   price {10*(i+1)}.50", fill="black", font=font(44))
d.rectangle((40, 1180, 1750, 1300), outline="red", width=6); d.text((70, 1200), "TOTAL DUE: 1034.00 EUR", fill="red", font=font(64))
im.save(f"{out}/invoice-large.png"); im.save(f"{out}/invoice-large.jpg", quality=88)
# 5 shapes
im = Image.new("RGB", (512, 512), "white"); d = ImageDraw.Draw(im)
d.ellipse((40, 40, 200, 200), fill="red"); d.rectangle((300, 60, 460, 220), fill="blue"); d.polygon([(250, 300), (150, 470), (350, 470)], fill="green")
im.save(f"{out}/shapes.png")
