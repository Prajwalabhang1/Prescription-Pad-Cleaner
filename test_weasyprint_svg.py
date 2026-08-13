import pymupdf
from PIL import Image
from weasyprint import HTML

html = """
<!DOCTYPE html>
<html>
<head>
<style>
@page { size: A4; margin: 0; }
body { margin: 0; padding: 0; background: white; }
</style>
</head>
<body>
    <svg width="210mm" height="297mm" viewBox="0 0 210 297">
        <text x="105" y="50" font-family="Arial" font-size="20" fill="red" text-anchor="middle">RAJESHWARA CLINIC</text>
    </svg>
</body>
</html>
"""

doc = HTML(string=html)
pdf_bytes = doc.write_pdf()

pdf_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
mat = pymupdf.Matrix(300 / 72, 300 / 72)
pix = pdf_doc[0].get_pixmap(matrix=mat, alpha=False)
img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
img.save("test_svg_no_xmlns.png")

import numpy as np
arr = np.array(img)
if (arr == 255).all():
    print("NO XMLNS: BLANK")
else:
    print("NO XMLNS: HAS CONTENT")

html2 = """
<!DOCTYPE html>
<html>
<head>
<style>
@page { size: A4; margin: 0; }
body { margin: 0; padding: 0; background: white; }
</style>
</head>
<body>
    <svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">
        <text x="105" y="50" font-family="Arial" font-size="20" fill="red" text-anchor="middle">RAJESHWARA CLINIC</text>
    </svg>
</body>
</html>
"""
doc2 = HTML(string=html2)
pdf_bytes2 = doc2.write_pdf()
pdf_doc2 = pymupdf.open(stream=pdf_bytes2, filetype="pdf")
pix2 = pdf_doc2[0].get_pixmap(matrix=mat, alpha=False)
img2 = Image.frombytes("RGB", (pix2.width, pix2.height), pix2.samples)
arr2 = np.array(img2)
if (arr2 == 255).all():
    print("WITH XMLNS: BLANK")
else:
    print("WITH XMLNS: HAS CONTENT")
