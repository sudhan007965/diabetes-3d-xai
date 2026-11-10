import numpy as np
from PIL import Image

src_path = r"website\assets\pure_human_hologram.png"
img = Image.open(src_path).convert("RGBA")
arr = np.array(img)
h, w, _ = arr.shape

# Remove any bright cyan platform remnant near the feet/ankles
ankle_y = int(h * 0.82)
for y in range(ankle_y, h):
    for x in range(w):
        r, g, b, a = arr[y, x]
        if a > 0:
            # Check if this pixel is part of the floor ring (high cyan/blue intensity outside the anatomical bone/muscle)
            if y > int(h * 0.865):
                arr[y, x] = 0
            # Platform behind the ankles
            elif (x < 265 or x > 505) and y > int(h * 0.835):
                arr[y, x] = 0
            # Between the ankles
            elif (350 < x < 420) and y > int(h * 0.835):
                arr[y, x] = 0

out = Image.fromarray(arr, "RGBA")
out.save(src_path, "PNG")
print("Cleaned feet and ankles completely.")
