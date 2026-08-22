import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(ROOT, "tests", "fixtures", "sample.mp4")
os.makedirs(os.path.dirname(path), exist_ok=True)

height, width = 240, 320
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(path, fourcc, 10.0, (width, height))

for index in range(30):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        f"frame {index}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    writer.write(frame)

writer.release()
print(f"created {path} ({os.path.getsize(path)} bytes)")
