import numpy as np
from PIL import Image

src_path = r"C:\Users\sc558\.gemini\antigravity-ide\brain\175fe484-1f6c-4fbb-8b2f-f71f8523da63\isolated_hologram_human_1792644177181.jpg"
dst_path = r"website\assets\pure_human_hologram.png"

# Load image using PIL
img = Image.open(src_path).convert("RGBA")
arr = np.array(img)
h, w, c = arr.shape
print(f"Loaded image: {w}x{h}")

# The platform ring starts at around y = 0.885 of height.
cut_y = int(h * 0.885)
fade_start = int(h * 0.875)

# Calculate luminance for transparency (dark black background becomes 0 alpha)
# lum = 0.299*R + 0.587*G + 0.114*B
r = arr[:, :, 0].astype(np.float32)
g = arr[:, :, 1].astype(np.float32)
b = arr[:, :, 2].astype(np.float32)
lum = 0.299 * r + 0.587 * g + 0.114 * b

# Zero out anything below cut_y (the bottom platform circle)
arr[cut_y:, :, :] = 0

# Smooth fade out between fade_start and cut_y
for y in range(fade_start, cut_y):
    factor = (cut_y - y) / (cut_y - fade_start)
    arr[y, :, :3] = (arr[y, :, :3].astype(np.float32) * factor).astype(np.uint8)

# Recalculate alpha channel: black background is transparent
# boost midtones so glowing hologram remains crisp
alpha = np.clip(lum * 3.0, 0, 255).astype(np.uint8)
alpha[lum < 8] = 0
alpha[cut_y:, :] = 0

arr[:, :, 3] = alpha

out_img = Image.fromarray(arr, "RGBA")
out_img.save(dst_path, "PNG")
print(f"Successfully saved clean transparent hologram without circle to: {dst_path}")
