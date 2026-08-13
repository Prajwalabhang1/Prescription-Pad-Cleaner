from PIL import Image, ImageDraw
img = Image.new('RGB', (800, 1200), color = (255, 255, 255))
d = ImageDraw.Draw(img)
d.rectangle([0, 0, 800, 300], fill=(0, 0, 139))
d.text((50, 50), "Dr. Nasim Ahmad", fill=(255,255,255), font_size=40)
img.save('dummy.png')
