"""Run real bundled OCR once. No AI service or screenshot permission needed."""
import os
from PIL import Image, ImageDraw, ImageFont
from rapidocr_onnxruntime import RapidOCR
import numpy as np

font_path = ("C:/Windows/Fonts/arial.ttf" if os.name == "nt" else
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
image = Image.new("RGB", (800, 200), "white")
ImageDraw.Draw(image).text((35, 60), "Hello world", fill="black", font=ImageFont.truetype(font_path, 42))
result, _ = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)(np.asarray(image)[:, :, ::-1].copy())
assert result and "hello" in " ".join(str(item[1]).lower() for item in result), result
print("Real OCR model smoke passed")
