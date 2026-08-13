import os
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import pymupdf
from pipeline.gemini_vision import generate_clean_html
from pipeline.html_render import render_html
from pipeline.page_geometry import PageGeometry

# Create a fake prescription image similar to the user's screenshot
img = Image.new("RGB", (800, 1200), (255, 255, 255))
draw = ImageDraw.Draw(img)
# Blue header
draw.rectangle([0, 0, 800, 300], fill=(0, 0, 139))
draw.text((50, 50), "Dr. Nasim Ahmad", fill=(255, 255, 255), font_size=40)
draw.text((50, 120), "MBBS, MD (DMCH)", fill=(255, 255, 255), font_size=20)
# Body
draw.text((50, 350), "Name.................. Age....", fill=(0, 0, 0), font_size=20)
# Footer
draw.rectangle([0, 1100, 800, 1200], fill=(0, 0, 139))

buf = io.BytesIO()
img.save(buf, format="PNG")
img_bytes = buf.getvalue()

try:
    print("Sending to Gemini...")
    page = PageGeometry.from_pixels(*img.size)
    html = generate_clean_html(img_bytes, page=page)
    with open("test_pipeline.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Saved HTML to test_pipeline.html")
    
    print("Rendering HTML...")
    pil_img, pdf_bytes = render_html(html, 300, page=page)
    pil_img.save("test_pipeline_output.png")
    print("Saved PNG to test_pipeline_output.png")
    
    import numpy as np
    arr = np.array(pil_img)
    if (arr == 255).all():
        print("PIPELINE RESULT IS COMPLETELY WHITE!")
    else:
        print("PIPELINE RESULT HAS CONTENT.")
except Exception as e:
    print("Pipeline failed:", e)
