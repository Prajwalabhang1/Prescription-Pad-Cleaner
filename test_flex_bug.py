import pymupdf
from PIL import Image
from weasyprint import HTML

html = """
<!DOCTYPE html>
<html>
<head>
<style>
@page { size: A4; margin: 0; }
body { display: flex; flex-direction: column; height: 100vh; margin: 0; background: white; }
.header { flex: 0 0 100px; background: blue; color: white; }
.content { flex: 1; overflow: hidden; }
</style>
</head>
<body>
    <div class="header">Dr. Nasim Ahmad</div>
    <div class="content">Name........</div>
</body>
</html>
"""

doc = HTML(string=html)
pdf_bytes = doc.write_pdf()

pdf_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
mat = pymupdf.Matrix(300 / 72, 300 / 72)
pix = pdf_doc[0].get_pixmap(matrix=mat, alpha=False)
img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
img.save("flex_bug_test.png")
import numpy as np
arr = np.array(img)
if (arr == 255).all():
    print("FLEX BUG RESULT IS COMPLETELY BLANK!")
else:
    print("FLEX BUG RESULT HAS CONTENT.")
